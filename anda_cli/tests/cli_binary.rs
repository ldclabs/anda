use std::process::{Command, Output};

fn anda_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anda_cli"))
        .args(args)
        .output()
        .expect("run anda_cli binary")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn binary_runs_no_command_and_rand_bytes_modes() {
    let output = anda_cli(&["--id", "Anonymous"]);
    assert!(output.status.success());
    let text = stdout(&output);
    assert!(text.contains("principal:"));
    assert!(text.contains("no command"));

    let output = anda_cli(&[
        "rand-bytes",
        "--id",
        "no-identity-needed",
        "--len",
        "4",
        "--format",
        "hex",
    ]);
    assert!(output.status.success());
    let lines = stdout(&output)
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].len(), 8);

    let output = anda_cli(&[
        "--id",
        "Anonymous",
        "rand-bytes",
        "--ed25519",
        "--format",
        "base64",
    ]);
    assert!(output.status.success());
    let text = stdout(&output);
    assert!(text.contains("Secret Key:"));
    assert!(text.contains("Public Key:"));
}

#[test]
fn help_hides_identity_environment_values() {
    let output = Command::new(env!("CARGO_BIN_EXE_anda_cli"))
        .env("ID_SECRET", "test-secret-marker")
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = stdout(&output);
    assert!(text.contains("ID_SECRET"));
    assert!(!text.contains("test-secret-marker"));
}

#[tokio::test]
async fn default_identity_can_call_a_public_rpc_method() {
    use anda_core::{RPCRequest, RPCResponse};
    use axum::{Router, body::Bytes, http::HeaderMap, routing::post};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/default", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/default",
        post(|headers: HeaderMap, body: Bytes| async move {
            assert!(!headers.contains_key("authorization"));
            let req: RPCRequest = cbor2::from_slice(&body).unwrap();
            assert_eq!(req.method, "ping");
            let value = cbor2::to_canonical_vec(&"pong").unwrap();
            cbor2::to_canonical_vec(&RPCResponse::Ok(value.into())).unwrap()
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_anda_cli"))
        .current_dir(std::env::temp_dir())
        .env_remove("ID_SECRET")
        .args(["rpc", "--endpoint", &endpoint, "--method", "ping"])
        .output()
        .await
        .unwrap();
    server.abort();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout(&output).contains("pong"));
    assert_eq!(stdout(&output).matches("principal:").count(), 1);
}

#[test]
fn binary_reaches_rpc_agent_and_tool_error_paths_without_remote_services() {
    let rpc_bad_json = anda_cli(&[
        "--id",
        "Anonymous",
        "rpc",
        "--method",
        "status",
        "--data",
        "not-json",
    ]);
    assert!(!rpc_bad_json.status.success());
    assert!(stdout(&rpc_bad_json).contains("principal:"));

    let rpc_bad_endpoint = anda_cli(&[
        "--id",
        "Anonymous",
        "rpc",
        "--endpoint",
        "not-a-url",
        "--method",
        "status",
        "--data",
        "[]",
    ]);
    assert!(!rpc_bad_endpoint.status.success());
    assert!(stdout(&rpc_bad_endpoint).contains("principal:"));

    let agent_bad_endpoint = anda_cli(&[
        "--id",
        "Anonymous",
        "agent-run",
        "--endpoint",
        "not-a-url",
        "--prompt",
        "hello",
        "--name",
        "writer",
    ]);
    assert!(!agent_bad_endpoint.status.success());
    assert!(stdout(&agent_bad_endpoint).contains("principal:"));

    let tool_bad_args = anda_cli(&[
        "--id",
        "Anonymous",
        "tool-call",
        "--name",
        "lookup",
        "--args",
        "not-json",
    ]);
    assert!(!tool_bad_args.status.success());
    assert!(stdout(&tool_bad_args).contains("principal:"));

    let tool_bad_endpoint = anda_cli(&[
        "--id",
        "Anonymous",
        "tool-call",
        "--endpoint",
        "not-a-url",
        "--name",
        "lookup",
        "--args",
        "{\"q\":\"anda\"}",
    ]);
    assert!(!tool_bad_endpoint.status.success());
    assert!(stdout(&tool_bad_endpoint).contains("principal:"));
}
