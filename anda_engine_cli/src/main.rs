use anda_core::{AgentOutput, BoxError, HttpFeatures};
use anda_web3_client::client::Client as Web3Client;
use base64::{prelude::BASE64_URL_SAFE_NO_PAD, Engine};
use clap::{Parser, Subcommand};
use rand::{thread_rng, RngCore};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, default_value = "https://icp-api.io")]
    ic_host: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    RandBytes {
        #[arg(short, long, default_value = "32")]
        len: usize,

        #[arg(short, long, default_value = "hex")]
        format: String,
    },
    AgentRun {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        #[arg(short, long, env = "ID_SECRET")]
        id_secret: String,

        #[arg(short, long)]
        prompt: String,

        #[arg(short, long)]
        name: Option<String>,
    },
    ToolCall {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        #[arg(short, long, env = "ID_SECRET")]
        id_secret: String,

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

    match &cli.command {
        Some(Commands::RandBytes { len, format }) => {
            let mut rng = thread_rng();
            let mut bytes = vec![0u8; (*len).min(1024)];
            rng.fill_bytes(&mut bytes);
            match format.as_str() {
                "hex" => {
                    println!("{}", const_hex::encode(&bytes));
                }
                "base64" => {
                    println!("{}", BASE64_URL_SAFE_NO_PAD.encode(&bytes));
                }
                _ => {
                    println!("{:?}", bytes);
                }
            }
        }

        Some(Commands::AgentRun {
            endpoint,
            id_secret,
            prompt,
            name,
        }) => {
            let id_secret = const_hex::decode(id_secret)?;
            let id_secret: [u8; 32] = id_secret.try_into().map_err(|_| "invalid id_secret")?;
            let web3 =
                Web3Client::new(&cli.ic_host, id_secret, [0u8; 48], None, Some(true)).await?;
            println!("principal: {}", web3.get_principal());

            let res: AgentOutput = web3
                .https_signed_rpc(endpoint, "agent_run", &(&name, &prompt, None::<Vec<u8>>))
                .await?;
            println!("{:?}", res);
        }

        Some(Commands::ToolCall {
            endpoint,
            id_secret,
            name,
            args,
        }) => {
            let id_secret = const_hex::decode(id_secret)?;
            let id_secret: [u8; 32] = id_secret.try_into().map_err(|_| "invalid id_secret")?;
            let web3 =
                Web3Client::new(&cli.ic_host, id_secret, [0u8; 48], None, Some(true)).await?;
            println!("principal: {}", web3.get_principal());

            let res: (String, bool) = web3
                .https_signed_rpc(endpoint, "tool_call", &(&name, &args))
                .await?;
            println!("{:?}", res);
        }

        None => {
            println!("no command");
        }
    }

    Ok(())
}
