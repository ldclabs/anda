//! Command-line client for interacting with Anda engine servers.
//!
//! The binary can generate random material, make signed RPC calls, run agents,
//! and call tools against an `anda_engine_server` endpoint.

use anda_core::{AgentInput, AgentOutput, BoxError, HttpFeatures, ToolInput, ToolOutput};
use anda_web3_client::client::{Client as Web3Client, load_identity};
use base64::{Engine, prelude::BASE64_URL_SAFE};
use ciborium::value::Value;
use clap::{Parser, Subcommand};
use ic_cose_types::cose::ed25519::{SigningKey, VerifyingKey};
use rand::Rng;
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

/// CLI subcommands supported by `anda`.
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
        /// Signed RPC endpoint URL.
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
        /// Engine endpoint URL.
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        /// Prompt to send to the agent.
        #[arg(short, long)]
        prompt: String,

        /// Optional agent name. Empty means the server default.
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Call a tool with the given name and args on the endpoint.
    ToolCall {
        /// Engine endpoint URL.
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        /// Tool name to call.
        #[arg(short, long)]
        name: String,

        /// Tool arguments as a JSON string.
        #[arg(short, long)]
        args: String,
    },
}

fn normalize_rpc_data(data: &str) -> Result<serde_json::Value, serde_json::Error> {
    let args: serde_json::Value = serde_json::from_str(data)?;
    Ok(if args.is_array() {
        args
    } else {
        serde_json::json!(vec![args])
    })
}

fn agent_input(name: &Option<String>, prompt: &str) -> AgentInput {
    AgentInput {
        name: name.clone().unwrap_or_default(),
        prompt: prompt.to_string(),
        ..Default::default()
    }
}

fn tool_input(name: &str, args: &str) -> Result<ToolInput<serde_json::Value>, serde_json::Error> {
    Ok(ToolInput {
        name: name.to_string(),
        args: serde_json::from_str(args)?,
        ..Default::default()
    })
}

fn bounded_rand_len(len: usize) -> usize {
    len.min(1024)
}

fn format_bytes(bytes: &[u8], format: &str) -> String {
    match format {
        "hex" => hex::encode(bytes),
        "base64" => BASE64_URL_SAFE.encode(bytes),
        _ => format!("{bytes:?}"),
    }
}

fn format_ed25519_key_pair(bytes: [u8; 32], format: &str) -> (String, String) {
    let signing_key = SigningKey::from_bytes(&bytes);
    let verifying_key = VerifyingKey::from(&signing_key);
    match format {
        "hex" => (hex::encode(bytes), hex::encode(verifying_key.to_bytes())),
        _ => (
            BASE64_URL_SAFE.encode(bytes),
            BASE64_URL_SAFE.encode(verifying_key.to_bytes()),
        ),
    }
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
                let (secret_key, public_key) = format_ed25519_key_pair(bytes, format);
                println!("Secret Key: {secret_key}");
                println!("Public Key: {public_key}");
            } else {
                let mut bytes = vec![0u8; bounded_rand_len(*len)];
                rng.fill_bytes(&mut bytes);
                println!("{}", format_bytes(&bytes, format));
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
            let args = normalize_rpc_data(data)?;

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
                .https_signed_rpc(endpoint, "agent_run", &(&agent_input(name, prompt),))
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
            let input = tool_input(name, args)?;

            let res: ToolOutput<serde_json::Value> = web3
                .https_signed_rpc(endpoint, "tool_call", &(&input,))
                .await?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }

        None => {
            println!("no command");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_parses_defaults_and_all_subcommands() {
        Cli::command().debug_assert();

        let cli = Cli::parse_from(["anda"]);
        assert_eq!(cli.host, "https://icp-api.io");
        assert_eq!(cli.id, "Anonymous");
        assert!(cli.command.is_none());

        let cli = Cli::parse_from(["anda", "rand-bytes", "--len", "8", "--format", "base64"]);
        match cli.command.unwrap() {
            Commands::RandBytes {
                len,
                format,
                ed25519,
            } => {
                assert_eq!(len, 8);
                assert_eq!(format, "base64");
                assert!(!ed25519);
            }
            _ => panic!("expected rand-bytes"),
        }

        let cli = Cli::parse_from([
            "anda",
            "--host",
            "http://localhost",
            "--id",
            "Anonymous",
            "rpc",
            "--endpoint",
            "http://127.0.0.1:8042/default",
            "--method",
            "status",
            "--data",
            "{\"ok\":true}",
        ]);
        assert_eq!(cli.host, "http://localhost");
        match cli.command.unwrap() {
            Commands::Rpc {
                endpoint,
                method,
                data,
            } => {
                assert!(endpoint.ends_with("/default"));
                assert_eq!(method, "status");
                assert_eq!(data, "{\"ok\":true}");
            }
            _ => panic!("expected rpc"),
        }

        let cli = Cli::parse_from(["anda", "agent-run", "-p", "hello", "-n", "writer"]);
        match cli.command.unwrap() {
            Commands::AgentRun {
                endpoint,
                prompt,
                name,
            } => {
                assert!(endpoint.contains("127.0.0.1"));
                assert_eq!(prompt, "hello");
                assert_eq!(name.as_deref(), Some("writer"));
            }
            _ => panic!("expected agent-run"),
        }

        let cli = Cli::parse_from([
            "anda",
            "tool-call",
            "-n",
            "lookup",
            "-a",
            "{\"q\":\"anda\"}",
        ]);
        match cli.command.unwrap() {
            Commands::ToolCall {
                endpoint,
                name,
                args,
            } => {
                assert!(endpoint.contains("127.0.0.1"));
                assert_eq!(name, "lookup");
                assert_eq!(args, "{\"q\":\"anda\"}");
            }
            _ => panic!("expected tool-call"),
        }
    }

    #[test]
    fn pure_command_helpers_prepare_outputs_and_inputs() {
        assert_eq!(bounded_rand_len(8), 8);
        assert_eq!(bounded_rand_len(2048), 1024);
        assert_eq!(format_bytes(&[0, 15, 255], "hex"), "000fff");
        assert_eq!(format_bytes(&[1, 2, 3], "base64"), "AQID");
        assert_eq!(format_bytes(&[1, 2], "debug"), "[1, 2]");

        let (secret_hex, public_hex) = format_ed25519_key_pair([7_u8; 32], "hex");
        assert_eq!(secret_hex.len(), 64);
        assert_eq!(public_hex.len(), 64);
        let (secret_b64, public_b64) = format_ed25519_key_pair([7_u8; 32], "base64");
        assert!(!secret_b64.is_empty());
        assert!(!public_b64.is_empty());
        assert_ne!(secret_hex, secret_b64);

        assert_eq!(
            normalize_rpc_data("[1,2]").unwrap(),
            serde_json::json!([1, 2])
        );
        assert_eq!(
            normalize_rpc_data("{\"ok\":true}").unwrap(),
            serde_json::json!([{"ok": true}])
        );
        assert!(normalize_rpc_data("not json").is_err());

        let input = agent_input(&Some("writer".to_string()), "draft");
        assert_eq!(input.name, "writer");
        assert_eq!(input.prompt, "draft");
        let input = agent_input(&None, "draft");
        assert_eq!(input.name, "");

        let input = tool_input("lookup", "{\"q\":\"anda\"}").unwrap();
        assert_eq!(input.name, "lookup");
        assert_eq!(input.args["q"], "anda");
        assert!(tool_input("lookup", "bad json").is_err());
    }
}
