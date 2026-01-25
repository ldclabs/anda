use anda_assistant::Assistant;
use anda_core::{AgentInput, ToolInput};
use anda_db::{
    database::{AndaDB, DBConfig},
    storage::StorageConfig,
};
use anda_engine::{
    context::Web3SDK,
    engine::Engine,
    management::{BaseManagement, Visibility},
    memory::{Conversation, ConversationStatus, MemoryTool},
    model::{Model, Proxy, gemini, request_client_builder},
    store::{InMemory, Store},
};
use anda_kip::Response;
use anda_web3_client::client::Client as Web3Client;
use candid::Principal;
use clap::Parser;
use ic_agent::identity::BasicIdentity;
use std::{collections::BTreeSet, sync::Arc};
use structured_logger::{Builder, async_json::new_writer, get_env_level};
use tokio::time::sleep;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to ICP identity pem file or 32 bytes identity secret in hex.
    #[arg(short, long, env = "ID_SECRET")]
    id_secret: String,

    /// 48 bytes root secret in hex to derive keys
    #[arg(long, env = "ROOT_SECRET")]
    root_secret: String,

    /// Gemini API key for AI model
    #[arg(long, env = "GEMINI_API_KEY", default_value = "")]
    gemini_api_key: String,

    #[arg(
        long,
        env = "GEMINI_API_BASE",
        default_value = "https://generativelanguage.googleapis.com/v1beta/models"
    )]
    gemini_api_base: String,

    #[arg(long, env = "GEMINI_MODEL", default_value = "gemini-3-pro-preview")]
    gemini_model: String,

    #[arg(long, env = "HTTPS_PROXY")]
    https_proxy: Option<String>,
}

/// cargo run --example hello_assistant
#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    // Initialize structured logging with JSON format
    Builder::with_level(&get_env_level().to_string())
        .with_target_writer("*", new_writer(tokio::io::stdout()))
        .init();

    let id_secret = hex::decode(&cli.id_secret).unwrap();
    let id_secret: [u8; 32] = id_secret
        .try_into()
        .map_err(|_| format!("invalid id_secret: {:?}", cli.id_secret))
        .unwrap();
    let identity = BasicIdentity::from_raw_key(&id_secret);

    let root_secret = hex::decode(&cli.root_secret).unwrap();
    let root_secret: [u8; 48] = root_secret
        .try_into()
        .map_err(|_| format!("invalid root_secret: {:?}", cli.root_secret))
        .unwrap();

    let mut http_client = request_client_builder();
    if let Some(proxy) = &cli.https_proxy {
        http_client = http_client.proxy(Proxy::all(proxy).unwrap());
    }
    let http_client = http_client.build().unwrap();

    // Initialize Web3 client for ICP network interaction
    let web3 = Web3Client::builder()
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .with_http_client(http_client.clone())
        .build()
        .await
        .unwrap();
    let web3 = Arc::new(web3);
    let my_principal = web3.get_principal();

    let management = Arc::new(BaseManagement {
        controller: my_principal,
        managers: BTreeSet::new(),
        visibility: Visibility::Public,
    });

    // Configure AI model
    let model = Model::with_completer(Arc::new(
        gemini::Client::new(&cli.gemini_api_key, Some(cli.gemini_api_base))
            .with_client(http_client.clone())
            .completion_model(&cli.gemini_model),
    ));

    let storage = InMemory::new();

    let object_store = Arc::new(storage);
    let db_config = DBConfig {
        name: "anda_db".to_string(),
        description: "Anda DB".to_string(),
        storage: StorageConfig {
            cache_max_capacity: 10000,
            compress_level: 3,
            object_chunk_size: 256 * 1024,
            bucket_overload_size: 1024 * 1024,
            max_small_object_size: 1024 * 1024 * 10,
        },
        lock: None,
    };

    let web3 = Arc::new(Web3SDK::from_web3(web3));
    let db = AndaDB::connect(object_store.clone(), db_config)
        .await
        .unwrap();
    let assistant = Assistant::connect(Arc::new(db), None).await.unwrap();
    let memory_tool = MemoryTool::new(assistant.memory());
    let engine = Engine::builder()
        .with_web3_client(web3)
        .with_store(Store::new(object_store))
        .with_management(management)
        .register_tools(assistant.tools().unwrap())
        .unwrap()
        .register_tool(memory_tool)
        .unwrap()
        .register_agent(assistant, None)
        .unwrap()
        .export_tools(vec![MemoryTool::NAME.to_string()])
        .with_model(model);

    let caller = Principal::management_canister();
    let engine = engine.build(Assistant::NAME.to_string()).await.unwrap();
    let rt = engine
        .agent_run(
            caller,
            AgentInput {
                name: Assistant::NAME.to_string(),
                prompt: "Hello, I'm Jarvis, nice to meet you. Please tell me about yourself."
                    .to_string(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    println!("Assistant response:\n{:?}", rt);
    let id = rt.conversation.unwrap();
    assert_eq!(id, 1);

    loop {
        let rt = engine
            .tool_call(
                caller,
                ToolInput {
                    name: MemoryTool::NAME.to_string(),
                    args: serde_json::json!({
                        "type": "GetConversation",
                        "_id": id,
                    }),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        // println!("Assistant tool call response: {:?}", rt);

        let rt: Response = serde_json::from_value(rt.output).unwrap();
        let rt = rt.into_result().unwrap();
        let conversation: Conversation = serde_json::from_value(rt).unwrap();

        println!(
            "Conversation history:\n{}\n",
            serde_json::to_string_pretty(&conversation).unwrap()
        );
        match conversation.status {
            ConversationStatus::Completed
            | ConversationStatus::Failed
            | ConversationStatus::Cancelled => {
                println!("Conversation completed.");
                break;
            }
            _ => {
                sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    }
}
