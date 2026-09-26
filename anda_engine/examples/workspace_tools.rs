//! Run a simple command with the coding tool bundle, including session polling.
use anda_core::{AgentContext, BoxError, ToolInput};
use anda_engine::{
    engine::EngineBuilder,
    extension::{
        shell::{NativeRuntime, ShellSessionScope},
        workspace::coding_tools,
    },
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let runtime = Arc::new(NativeRuntime::new(std::env::current_dir()?));
    let ctx = EngineBuilder::new()
        .register_tools(coding_tools(runtime, vec![])?)?
        .mock_ctx();
    // Production hosts create one scope per conversation and reuse it across turns.
    ctx.base.set_state(ShellSessionScope::new());
    let (mut result, _) = ctx
        .tool_call(ToolInput {
            name: "shell".into(),
            args: json!({"command": "echo workspace tools ready"}),
            ..Default::default()
        })
        .await?;
    println!("{}", result.output);
    while result.output.get("state").and_then(|state| state.as_str()) == Some("running") {
        let task_id = result.output["task_id"].clone();
        let (reply, _) = ctx
            .tool_call(ToolInput {
                name: "shell_session".into(),
                args: json!({"action": "poll", "task_id": task_id}),
                ..Default::default()
            })
            .await?;
        println!("{}", reply.output);
        result.output = reply.output["command"].clone();
    }
    Ok(())
}
