use anda_core::{Agent, BoxError, Path as DBPath, derivation_path_with};
use anda_db::{
    database::{AndaDB, DBConfig},
    storage::StorageConfig,
};
use anda_engine::{
    context::{Web3ClientFeatures, Web3SDK},
    engine::{AgentInfo, EchoEngineInfo, EngineBuilder},
    management::{BaseManagement, SYSTEM_PATH, Visibility},
    store::{InMemory, LocalFileSystem, ObjectStore, Store},
};
use anda_engine_server::{ServerBuilder, shutdown_signal};
use anda_nexus::{Conf, NexusNode};
use anda_object_store::MetaStoreBuilder;
use anda_web3_client::client::{Client as Web3Client, load_identity};
use clap::Parser;
use ic_auth_types::ByteBufB64;
use ic_auth_verifier::sha3_256;
use object_store::aws::AmazonS3Builder;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use structured_logger::{Builder, async_json::new_writer, get_env_level};
use tokio_util::sync::CancellationToken;

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

    #[clap(long, env = "CONFIG_FILE_PATH", default_value = "./Config.toml")]
    config: String,
}

/// Main entry point for the Anda nexus service.
///
/// # Example Usage
/// ```bash
/// cargo run -p anda_nexus -- \
///     --config ./agents/anda_nexus/Config.toml
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

    let cfg = Conf::from_file(&cli.config)?;
    log::debug!("{:?}", cfg);

    // Parse and validate cryptographic secrets
    let identity = load_identity(&cfg.id_secret)?;
    let root_secret = hex::decode(&cfg.root_secret)?;
    let root_secret: [u8; 48] = root_secret
        .try_into()
        .map_err(|_| format!("invalid root_secret: {:?}", cfg.root_secret))?;

    // Initialize Web3 client for ICP network interaction
    let web3 = Web3Client::builder()
        .with_ic_host(&cli.ic_host)
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build()
        .await?;
    let web3 = Arc::new(web3);

    let my_principal = web3.get_principal();
    log::info!(
        "start local service, principal: {:?}",
        my_principal.to_text()
    );

    let os_secret = web3
        .a256gcm_key(derivation_path_with(
            &DBPath::from(SYSTEM_PATH),
            vec![b"object_store".to_vec(), b"A256GCM".to_vec()],
        ))
        .await?;
    let lock = sha3_256(&os_secret);
    let object_store = build_object_store(cfg.object_store, cfg.object_store_config)?;

    let db_config = DBConfig {
        name: "anda_db".to_string(),
        description: "Anda DB".to_string(),
        storage: StorageConfig {
            cache_max_capacity: 100000,
            compress_level: 3,
            object_chunk_size: 256 * 1024,
            bucket_overload_size: 1024 * 1024,
            max_small_object_size: 1024 * 1024 * 10,
        },
        lock: Some(ByteBufB64(lock.into())),
    };

    let db = AndaDB::connect(object_store.clone(), db_config).await?;

    let nexus = NexusNode::connect(Arc::new(db)).await?;
    let nexus = Arc::new(nexus);
    let tools = NexusNode::tools(nexus)?;
    let tools_name = tools.names();
    let info = AgentInfo {
        handle: "icp_ledger_agent".to_string(),
        name: "ICP Agent".to_string(),
        description: "Test ICP Agent".to_string(),
        endpoint: "https://localhost:8443/default".to_string(),
        ..Default::default()
    };
    let agent = EchoEngineInfo::new(info.clone());
    let agent_name = agent.name();
    let engine = EngineBuilder::new()
        .with_info(info)
        .with_cancellation_token(global_cancel_token.clone())
        .with_web3_client(Arc::new(Web3SDK::from_web3(web3.clone())))
        .with_store(Store::new(object_store))
        .with_management(Arc::new(BaseManagement {
            controller: my_principal,
            managers: BTreeSet::new(),
            visibility: Visibility::Public,
        }))
        .register_tools(tools)?
        .register_agent(agent)?
        .export_tools(tools_name);

    // Initialize and start the server
    let engine = engine.build(agent_name).await?;
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

fn build_object_store(
    ty: String,
    cfg: Option<BTreeMap<String, String>>,
) -> Result<Arc<dyn ObjectStore>, BoxError> {
    match ty.as_str() {
        "" | "memory" | "in_memory" => Ok(Arc::new(InMemory::new())),
        "s3" => {
            let mut builder: AmazonS3Builder = Default::default();
            for (k, v) in cfg.unwrap_or_default().iter() {
                if let Ok(config_key) = k.to_ascii_lowercase().parse() {
                    builder = builder.with_config(config_key, v);
                }
            }

            let os = builder.build()?;
            Ok(Arc::new(os))
        }
        _ => {
            let os = LocalFileSystem::new_with_prefix(ty)?;
            let os = MetaStoreBuilder::new(os, 100000).build();
            Ok(Arc::new(os))
        }
    }
}
