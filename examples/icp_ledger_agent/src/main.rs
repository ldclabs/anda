use anda_core::BoxError;
use anda_engine::{
    context::Web3SDK,
    engine::{AgentInfo, EngineBuilder},
    management::{BaseManagement, Visibility},
    model::{Model, deepseek, openai, xai},
    store::{InMemory, Store},
};
use anda_engine_server::{ServerBuilder, shutdown_signal};
use anda_icp::ledger::BalanceOfTool;
use anda_web3_client::client::{Client as Web3Client, load_identity};
use clap::Parser;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use structured_logger::{Builder, async_json::new_writer, get_env_level};
use tokio_util::sync::CancellationToken;

mod agent;

use agent::ICPLedgerAgent;

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Port to listen on
    #[clap(long, default_value = "8042")]
    port: u16,

    /// ICP API host
    #[clap(long, default_value = "https://icp-api.io")]
    ic_host: String,

    /// Path to ICP identity pem file or 32 bytes identity secret in hex.
    #[arg(short, long, env = "ID_SECRET")]
    id_secret: String,

    /// 48 bytes root secret in hex to derive keys
    #[arg(long, env = "ROOT_SECRET")]
    root_secret: String,

    /// Deepseek API key for AI model
    #[arg(long, env = "DEEPSEEK_API_KEY", default_value = "")]
    deepseek_api_key: String,

    /// OpenAI API key for AI model
    #[arg(long, env = "OPENAI_API_KEY", default_value = "")]
    openai_api_key: String,

    /// XAI API key for AI model
    #[arg(long, env = "XAI_API_KEY", default_value = "")]
    xai_api_key: String,

    /// AI model endpoint, empty for default to auto-detect
    #[arg(long, env = "MODEL_ENDPOINT", default_value = "")]
    model_endpoint: String,

    /// AI model name, empty for default to auto-detect
    #[arg(long, env = "MODEL_NAME", default_value = "")]
    model_name: String,
}

/// Main entry point for the ICP Ledger Agent service.
///
/// This service provides an AI-powered agent that interacts with the Internet Computer (ICP)
/// ledger and other token ledgers. It exposes a web interface for interacting with the agent and
/// managing blockchain operations.
///
/// # Configuration
/// The service can be configured via command line arguments or environment variables:
/// - Port: The port to listen on (default: 8042)
/// - ICP API host: The ICP network endpoint (default: https://icp-api.io)
/// - ID Secret: 32-byte hex-encoded secret for identity management
/// - Root Secret: 48-byte hex-encoded root secret for cryptographic operations
/// - AI Model: Supports both Deepseek and OpenAI models (Deepseek is default)
///
/// # Features
/// - Real-time interaction with ICP ledger
/// - Support for multiple token ledgers
/// - REST API interface for external integration
///
/// # Example Usage
/// ```bash
/// cargo run -p icp_ledger_agent -- \
///     --id-secret <32-byte-hex> \
///     --root-secret <48-byte-hex> \
///     --deepseek-api-key <key>
/// ```
///
/// or with environment variables in a `.env` file:
/// ```bash
/// cargo run -p icp_ledger_agent
/// ```
#[tokio::main]
async fn main() -> Result<(), BoxError> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    // Initialize structured logging with JSON format
    Builder::with_level(&get_env_level().to_string())
        .with_target_writer("*", new_writer(tokio::io::stdout()))
        .init();

    // Create global cancellation token for graceful shutdown
    let global_cancel_token = CancellationToken::new();

    // Parse and validate cryptographic secrets
    let identity = load_identity(&cli.id_secret)?;
    let root_secret = hex::decode(&cli.root_secret)?;
    let root_secret: [u8; 48] = root_secret
        .try_into()
        .map_err(|_| format!("invalid root_secret: {:?}", cli.root_secret))?;

    // Initialize Web3 client for ICP network interaction
    let web3 = Web3Client::builder()
        .with_ic_host(&cli.ic_host)
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build()
        .await?;

    let my_principal = web3.get_principal();
    log::info!(
        "start local service, principal: {:?}",
        my_principal.to_text()
    );

    // Configure AI model
    let model = Model::with_completer(if !cli.deepseek_api_key.is_empty() {
        Arc::new(
            deepseek::Client::new(&cli.deepseek_api_key, Some(cli.model_endpoint))
                .completion_model(&cli.model_name),
        )
    } else if !cli.openai_api_key.is_empty() {
        Arc::new(
            openai::Client::new(&cli.openai_api_key, Some(cli.model_endpoint))
                .completion_model(&cli.model_name),
        )
    } else if !cli.xai_api_key.is_empty() {
        Arc::new(
            xai::Client::new(&cli.xai_api_key, Some(cli.model_endpoint))
                .completion_model(&cli.model_name),
        )
    } else {
        return Err("missing AI model API key".into());
    });

    // Initialize in-memory object store.
    // For production use, consider using a local file system store or ic_obejct_store_canister:
    // let object_store = Arc::new(LocalFileSystem::new_with_prefix(store_path)?);
    let object_store = Arc::new(InMemory::new());

    // Configure supported token ledgers (ICP and PANDA)
    let token_ledgers: Vec<&str> =
        vec!["ryjl3-tyaaa-aaaaa-aaaba-cai", "druyg-tyaaa-aaaaq-aactq-cai"];
    let agent = ICPLedgerAgent::load(&web3, &token_ledgers).await?;

    // Build agent engine with all configured components
    let engine = EngineBuilder::new()
        .with_info(AgentInfo {
            handle: "icp_ledger_agent".to_string(),
            name: "ICP Agent".to_string(),
            description: "Test ICP Agent".to_string(),
            endpoint: "https://localhost:8443/default".to_string(),
            ..Default::default()
        })
        .with_cancellation_token(global_cancel_token.clone())
        .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3.clone()))))
        .with_model(model)
        .with_store(Store::new(object_store))
        .with_management(Arc::new(BaseManagement {
            controller: my_principal,
            managers: BTreeSet::new(),
            visibility: Visibility::Public,
        }))
        .register_tools(agent.tools()?)?
        .register_agent(agent, None)?
        .export_tools(vec![BalanceOfTool::NAME.to_string()]);

    // Initialize and start the server
    let engine = engine.build(ICPLedgerAgent::NAME.to_string()).await?;
    let mut engines = BTreeMap::new();
    engines.insert(engine.id(), engine);

    ServerBuilder::new()
        .with_app_name(APP_NAME.to_string())
        .with_app_version(APP_VERSION.to_string())
        .with_addr(format!("127.0.0.1:{}", cli.port))
        .with_engines(engines, None)
        .serve(shutdown_signal(global_cancel_token))
        .await?;

    Ok(())
}
