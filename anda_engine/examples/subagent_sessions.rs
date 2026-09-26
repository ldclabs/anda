//! Run with `cargo run -p anda_engine --example subagent_sessions`.
//! Uses a deterministic local completer; no provider account or network access is needed.
use anda_core::{
    Agent, AgentOutput, BoxError, BoxPinFut, CompletionRequest, ContentPart, Path, Usage,
};
use anda_engine::{
    engine::EngineBuilder,
    model::{CompletionFeaturesDyn, Model},
    subagent::{
        MessageDelivery, SubAgent, SubAgentArgs, SubAgentCheckpoints, SubAgentHandoff,
        SubAgentLimits, SubAgentMessage, SubAgentScope, WaitMode,
    },
};
use std::{sync::Arc, time::Duration};

#[derive(Debug)]
struct Echo;
impl CompletionFeaturesDyn for Echo {
    fn model_name(&self) -> String {
        "local-echo".into()
    }
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(async move {
            let text = std::iter::once(req.prompt)
                .chain(req.content.into_iter().filter_map(|part| {
                    if let ContentPart::Text { text } = part {
                        Some(text)
                    } else {
                        None
                    }
                }))
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            Ok(AgentOutput {
                content: format!("Completed: {text}"),
                usage: Usage {
                    requests: 1,
                    output_tokens: 1,
                    ..Default::default()
                },
                ..Default::default()
            })
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Production hosts obtain the entry context through Engine::ctx_with and install the same
    // scope clone for each turn of one root task. Separate tasks must use separate scopes.
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(Echo)))
        .mock_ctx();
    let scope = SubAgentScope::new(SubAgentLimits {
        max_sessions: 4,
        max_parallel_requests: 2,
        max_requests: Some(20),
        ..Default::default()
    });
    ctx.base.set_state(scope.clone());
    ctx.base.set_state(SubAgentCheckpoints::object_store(
        Arc::new(object_store::memory::InMemory::new()),
        Path::from("worker-checkpoints"),
    ));
    ctx.base.set_state(SubAgentHandoff::summary(
        "Host-selected task background.".into(),
        4096,
    )?);
    let worker = SubAgent {
        name: "research".into(),
        instructions: "Return a concise result.".into(),
        ..Default::default()
    };
    worker
        .run(
            ctx.clone(),
            serde_json::to_string(&SubAgentArgs {
                prompt: "Inspect the selected resources".into(),
                session: "review".into(),
                ..Default::default()
            })?,
            vec![],
        )
        .await?;
    let session = worker
        .subsessions
        .get_session_in_scope(&candid::Principal::anonymous(), &scope, "review")
        .ok_or("session missing")?;
    let targets = vec![session.execution().id.clone()];
    let first = scope
        .wait(
            0,
            &targets,
            WaitMode::All,
            Duration::from_secs(5),
            anda_core::CancellationToken::new(),
        )
        .await?;
    assert!(!first.timed_out);
    println!("{}", serde_json::to_string_pretty(&first)?);
    session.send(
        SubAgentMessage {
            sender: scope.id().into(),
            id: "note-1".into(),
            content: "Additional background".into(),
            resources: vec![],
        },
        MessageDelivery::QueueOnly,
    )?;
    session.send(
        SubAgentMessage {
            sender: scope.id().into(),
            id: "task-2".into(),
            content: "Use the background to finish".into(),
            resources: vec![],
        },
        MessageDelivery::TriggerTurn,
    )?;
    let second = scope
        .wait(
            first.cursor,
            &targets,
            WaitMode::All,
            Duration::from_secs(5),
            anda_core::CancellationToken::new(),
        )
        .await?;
    assert!(!second.timed_out);
    println!("{}", serde_json::to_string_pretty(&second)?);
    session.close();
    Ok(())
}
