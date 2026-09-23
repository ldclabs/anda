use anda_core::AgentInput;
use anda_engine::{
    ANONYMOUS,
    engine::{AgentInfo, EchoEngineInfo, Engine},
    management::{BaseManagement, Visibility},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let echo_info = AgentInfo {
        handle: "echo".to_string(),
        name: "Echo Agent".to_string(),
        description: "Returns engine metadata as JSON.".to_string(),
        ..Default::default()
    };

    // This demo explicitly permits anonymous callers; engines default to private.
    let engine = Engine::builder()
        .with_management(Arc::new(BaseManagement {
            controller: ANONYMOUS,
            managers: Default::default(),
            visibility: Visibility::Public,
        }))
        .register_agent(Arc::new(EchoEngineInfo::new(echo_info)), None)?
        .build("echo".to_string())
        .await?;

    let output = engine
        .agent_run(
            ANONYMOUS,
            AgentInput::new("echo".to_string(), "hello".to_string()),
        )
        .await?;

    println!("{}", output.content);
    Ok(())
}
