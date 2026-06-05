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
        "--id",
        "Anonymous",
        "rand-bytes",
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
    assert_eq!(lines.last().unwrap().len(), 8);

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
