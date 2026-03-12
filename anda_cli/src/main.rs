use anda_core::{AgentInput, AgentOutput, BoxError, HttpFeatures, ToolInput, ToolOutput};
use anda_web3_client::client::{Client as Web3Client, load_identity};
use base64::{Engine, prelude::BASE64_URL_SAFE};
use ciborium::value::Value;
use clap::{Parser, Subcommand};
use ic_cose_types::cose::ed25519::{SigningKey, VerifyingKey};
use rand::RngCore;
use std::sync::Arc;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[clap(long, default_value = "https://icp-api.io")]
    host: String,

    /// Path to ICP identity pem file or 32 bytes identity secret in hex.
    #[arg(long, env = "ID_SECRET", default_value = "Anonymous")]
    id: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate random bytes with the given length and format
    RandBytes {
        /// Length of the random bytes, default is 32
        #[arg(short, long, default_value = "32")]
        len: usize,
        /// Output format: hex or base64, default is hex
        #[arg(short, long, default_value = "hex")]
        format: String,

        /// Whether to generate an ed25519 key pair, if true, the len will be ignored.
        #[arg(long)]
        ed25519: bool,
    },

    /// make an signed RPC call to the endpoint with the given ICP identity, method and args.
    /// The RPC response from the endpoint should be string.
    /// Example: `anda_engine_cli rpc -i ./identity.pem -e 'https://andaicp.anda.bot/proposal'  -m start_x_bot`
    Rpc {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        /// RPC method name
        #[arg(short, long)]
        method: String,

        /// RPC arguments in JSON string, default is [], means no arguments.
        #[arg(short, long, default_value = "[]")]
        data: String,
    },

    /// Run an AI agent with the given prompt and name on the endpoint.
    AgentRun {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        #[arg(short, long)]
        prompt: String,

        #[arg(short, long)]
        name: Option<String>,
    },

    /// Call a tool with the given name and args on the endpoint.
    ToolCall {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        #[arg(short, long)]
        name: String,

        #[arg(short, long)]
        args: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();
    let identity = load_identity(&cli.id)?;
    println!("principal: {}", identity.sender()?);

    match &cli.command {
        Some(Commands::RandBytes {
            len,
            format,
            ed25519,
        }) => {
            let mut rng = rand::rng();

            if *ed25519 {
                let mut bytes = [0u8; 32];
                rng.fill_bytes(&mut bytes);
                let signing_key = SigningKey::from_bytes(&bytes);
                let verifying_key = VerifyingKey::from(&signing_key);
                match format.as_str() {
                    "hex" => {
                        println!("Secret Key: {}", hex::encode(&bytes));
                        println!("Public Key: {}", hex::encode(verifying_key.to_bytes()));
                    }
                    _ => {
                        println!("Secret Key: {}", BASE64_URL_SAFE.encode(&bytes));
                        println!(
                            "Public Key: {}",
                            BASE64_URL_SAFE.encode(verifying_key.to_bytes())
                        );
                    }
                }
            } else {
                let mut bytes = vec![0u8; (*len).min(1024)];
                rng.fill_bytes(&mut bytes);
                match format.as_str() {
                    "hex" => {
                        println!("{}", hex::encode(&bytes));
                    }
                    "base64" => {
                        println!("{}", BASE64_URL_SAFE.encode(&bytes));
                    }
                    _ => {
                        println!("{:?}", bytes);
                    }
                }
            }
        }

        Some(Commands::Rpc {
            endpoint,
            method,
            data,
        }) => {
            let web3 = Web3Client::builder()
                .with_ic_host(&cli.host)
                .with_identity(Arc::new(identity))
                .with_allow_http(true)
                .build()
                .await?;

            println!("principal: {}", web3.get_principal());
            let args: serde_json::Value = serde_json::from_str(data)?;
            let args = if args.is_array() {
                args
            } else {
                serde_json::json!(vec![args])
            };

            let res: Value = web3.https_signed_rpc(endpoint, method, &args).await?;
            println!("{:?}", res);
        }

        Some(Commands::AgentRun {
            endpoint,
            name,
            prompt,
        }) => {
            let web3 = Web3Client::builder()
                .with_ic_host(&cli.host)
                .with_identity(Arc::new(identity))
                .with_allow_http(true)
                .build()
                .await?;

            println!("principal: {}", web3.get_principal());

            let res: AgentOutput = web3
                .https_signed_rpc(
                    endpoint,
                    "agent_run",
                    &(&AgentInput {
                        name: name.clone().unwrap_or_else(|| "".to_string()),
                        prompt: prompt.clone(),
                        ..Default::default()
                    },),
                )
                .await?;
            println!("{:?}", res);
        }

        Some(Commands::ToolCall {
            endpoint,
            name,
            args,
        }) => {
            let web3 = Web3Client::builder()
                .with_ic_host(&cli.host)
                .with_identity(Arc::new(identity))
                .with_allow_http(true)
                .build()
                .await?;

            println!("principal: {}", web3.get_principal());
            let args: serde_json::Value = serde_json::from_str(args)?;

            let res: ToolOutput<serde_json::Value> = web3
                .https_signed_rpc(
                    endpoint,
                    "tool_call",
                    &(&ToolInput {
                        name: name.clone(),
                        args,
                        ..Default::default()
                    },),
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }

        None => {
            println!("no command");
        }
    }

    Ok(())
}
