//! Note/todo lifecycle regressions through the public engine/runner APIs.
use anda_core::{AgentOutput, CompletionFeatures, CompletionRequest, Tool, ToolCall};
use anda_engine::{
    engine::EngineBuilder,
    extension::{
        note::{NoteArgs, NoteContextConfig, NoteItemInput, NoteTool},
        todo::{TodoItemInput, TodoTool, todo_session},
    },
    model::{Model, testing::ScriptedCompleter},
};
use serde_json::json;
use std::sync::Arc;

fn task(id: &str, content: &str, status: &str) -> TodoItemInput {
    TodoItemInput {
        id: id.into(),
        content: Some(content.into()),
        status: Some(status.into()),
    }
}
fn request_text(request: &CompletionRequest) -> String {
    serde_json::to_string(&request.chat_history).unwrap()
}

#[tokio::test]
async fn handoff_recovers_active_tasks_once_per_window() {
    let model = ScriptedCompleter::new("tasks")
        .push_output(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "todo".into(),
                args: json!({"op":"set", "items":[
                    {"id":"a","content":"continue implementation","status":"in_progress"},
                    {"id":"b","content":"run validation","status":"pending"},
                    {"id":"c","content":"already done","status":"completed"}
                ]}),
                call_id: Some("tasks-1".into()),
                ..Default::default()
            }],
            ..Default::default()
        })
        .push_output(AgentOutput {
            content: "work started".into(),
            ..Default::default()
        })
        .push_output(AgentOutput {
            content: "A deliberately incomplete summary".into(),
            ..Default::default()
        })
        .push_output(AgentOutput {
            content: "continued".into(),
            ..Default::default()
        })
        .push_output(AgentOutput {
            content: "Another incomplete summary".into(),
            ..Default::default()
        })
        .into_arc();
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(model.clone()))
        .register_tool(Arc::new(TodoTool::new()))
        .unwrap()
        .mock_ctx();
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                prompt: "work".into(),
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    runner.next().await.unwrap();
    runner.next().await.unwrap();
    let (mut runner, _) = runner.handoff(None).await.unwrap();
    runner.follow_up("continue".to_string());
    runner.next().await.unwrap();
    let sent = request_text(&model.requests()[3]);
    assert!(sent.contains("continue implementation"));
    assert!(sent.contains("run validation"));
    assert!(!sent.contains("already done"));
    assert_eq!(sent.matches("active task list was preserved").count(), 1);
    let (mut runner, _) = runner.handoff(None).await.unwrap();
    runner.follow_up("continue again".to_string());
    runner.next().await.unwrap();
    assert_eq!(
        request_text(&model.requests()[5])
            .matches("active task list was preserved")
            .count(),
        1
    );
}

#[tokio::test]
async fn failed_handoff_preserves_tasks_and_does_not_inject_a_snapshot() {
    let model = ScriptedCompleter::new("tasks")
        .push_output(AgentOutput {
            content: "started".into(),
            ..Default::default()
        })
        .push_error("summary failed")
        .into_arc();
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(model.clone()))
        .mock_ctx();
    let session = todo_session(&ctx.base);
    let initial = session
        .set(vec![task("a", "unfinished", "pending")])
        .unwrap();
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                prompt: "start".into(),
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    runner.next().await.unwrap();
    let before = runner.chat_history().to_vec();
    assert!(runner.handoff(None).await.is_err());
    assert_eq!(session.snapshot(), initial);
    assert_eq!(runner.chat_history(), &before);
    assert!(!runner.is_done());
    runner.follow_up("retry work".to_string());
    runner.next().await.unwrap();
    assert!(!request_text(&model.requests()[2]).contains("active task list was preserved"));
}

#[tokio::test]
async fn malformed_todo_call_is_returned_to_model_without_clearing_tasks() {
    let model = ScriptedCompleter::new("tasks")
        .push_output(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "todo".into(),
                args: json!({"op":"set"}),
                call_id: Some("bad".into()),
                ..Default::default()
            }],
            ..Default::default()
        })
        .into_arc();
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(model.clone()))
        .register_tool(Arc::new(TodoTool::new()))
        .unwrap()
        .mock_ctx();
    let session = todo_session(&ctx.base);
    let initial = session.set(vec![task("a", "keep", "completed")]).unwrap();
    let mut runner = ctx.completion_iter(
        CompletionRequest {
            prompt: "update".into(),
            ..Default::default()
        },
        vec![],
    );
    runner.next().await.unwrap();
    let result = runner.next().await.unwrap().unwrap();
    assert_eq!(
        result.tool_calls[0].result.as_ref().unwrap().is_error,
        Some(true)
    );
    assert_eq!(session.snapshot(), initial);
    assert!(
        serde_json::to_string(&model.requests()[1].content)
            .unwrap()
            .contains("items are required")
    );
}

#[tokio::test]
async fn note_index_requires_opt_in_and_tool_permission() {
    for (registered, enabled, offered, allowed) in [
        (true, false, true, true),
        (true, true, true, false),
        (true, true, false, true),
        (true, true, true, true),
        (false, true, true, true),
    ] {
        let model = ScriptedCompleter::new("notes").into_arc();
        let tool = Arc::new(NoteTool::new());
        let builder = EngineBuilder::new().with_model(Model::with_completer(model.clone()));
        let builder = if registered {
            builder.register_tool(tool.clone()).unwrap()
        } else {
            builder
        };
        let ctx = builder.mock_ctx().child("writer", "writer").unwrap();
        tool.call(
            ctx.child_base("note").unwrap(),
            NoteArgs {
                op: Some("set".into()),
                items: Some(vec![NoteItemInput {
                    id: "preference".into(),
                    content: Some("run the narrow tests first".into()),
                }]),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap();
        if enabled {
            ctx.base.set_state(NoteContextConfig::default());
        }
        let mut runner = ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "work".into(),
                    tools: if offered {
                        vec![tool.definition()]
                    } else {
                        Vec::new()
                    },
                    ..Default::default()
                },
                vec![],
            )
            .unbound();
        if !allowed {
            runner = runner.with_allowed_callables(Some(Default::default()));
        }
        let output = runner.next().await.unwrap().unwrap();
        let first = request_text(&model.requests()[0]);
        assert_eq!(
            first.contains("run the narrow tests first"),
            registered && enabled && offered && allowed
        );
        // Request-only context: output history that hosts persist never carries the index.
        assert!(
            !serde_json::to_string(&output.chat_history)
                .unwrap()
                .contains("Saved note index")
        );
        runner.follow_up("next step".to_string());
        runner.next().await.unwrap();
        assert!(!request_text(&model.requests()[1]).contains("Saved note index"));
    }
}

#[tokio::test]
async fn note_index_refreshes_after_handoff_and_remains_agent_scoped() {
    let model = ScriptedCompleter::new("notes")
        .push_output(AgentOutput {
            content: "started".into(),
            ..Default::default()
        })
        .push_output(AgentOutput {
            content: "summary without notes".into(),
            ..Default::default()
        })
        .into_arc();
    let tool = Arc::new(NoteTool::new());
    let parent = EngineBuilder::new()
        .with_model(Model::with_completer(model.clone()))
        .register_tool(tool.clone())
        .unwrap()
        .mock_ctx();
    let ctx = parent.child("writer", "writer").unwrap();
    ctx.base.set_state(NoteContextConfig::default());
    for (agent, content) in [
        (&ctx, "old knowledge"),
        (
            &parent.child("other", "other").unwrap(),
            "other agent private note",
        ),
    ] {
        tool.call(
            agent.child_base("note").unwrap(),
            NoteArgs {
                op: Some("set".into()),
                items: Some(vec![NoteItemInput {
                    id: "fact".into(),
                    content: Some(content.into()),
                }]),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap();
    }
    let mut runner = ctx
        .clone()
        .completion_iter(
            CompletionRequest {
                prompt: "start".into(),
                tools: vec![tool.definition()],
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    runner.next().await.unwrap();
    tool.call(
        ctx.child_base("note").unwrap(),
        NoteArgs {
            op: Some("upsert".into()),
            items: Some(vec![NoteItemInput {
                id: "fact".into(),
                content: Some("updated knowledge".into()),
            }]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    let (mut runner, _) = runner.handoff(None).await.unwrap();
    runner.follow_up("continue".to_string());
    runner.next().await.unwrap();
    let sent = request_text(&model.requests()[2]);
    assert_eq!(sent.matches("Saved note index").count(), 1);
    assert!(sent.contains("updated knowledge"));
    assert!(!sent.contains("old knowledge"));
    assert!(!sent.contains("other agent private note"));
}

#[tokio::test]
async fn resumed_conversations_carry_one_fresh_note_index() {
    let model = ScriptedCompleter::new("notes").into_arc();
    let tool = Arc::new(NoteTool::new());
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(model.clone()))
        .register_tool(tool.clone())
        .unwrap()
        .mock_ctx();
    ctx.base.set_state(NoteContextConfig::default());
    tool.call(
        ctx.child_base("note").unwrap(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![NoteItemInput {
                id: "fact".into(),
                content: Some("saved context".into()),
            }]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    // A host persisting each run's output history and resuming from it.
    let mut history = Vec::new();
    for prompt in ["first", "second", "third"] {
        let output = ctx
            .completion(
                CompletionRequest {
                    prompt: prompt.into(),
                    tools: vec![tool.definition()],
                    chat_history: history.clone(),
                    ..Default::default()
                },
                vec![],
            )
            .await
            .unwrap();
        history.extend(output.chat_history);
    }
    for request in model.requests() {
        assert_eq!(
            request_text(&request).matches("Saved note index").count(),
            1
        );
    }
}

#[tokio::test]
async fn note_index_does_not_interrupt_a_tool_response_boundary() {
    use anda_core::ContentPart;
    let model = ScriptedCompleter::new("notes").into_arc();
    let tool = Arc::new(NoteTool::new());
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(model.clone()))
        .register_tool(tool.clone())
        .unwrap()
        .mock_ctx();
    ctx.base.set_state(NoteContextConfig::default());
    tool.call(
        ctx.child_base("note").unwrap(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![NoteItemInput {
                id: "fact".into(),
                content: Some("saved context".into()),
            }]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                role: Some("tool".into()),
                content: vec![ContentPart::ToolOutput {
                    name: "echo".into(),
                    output: json!("done"),
                    is_error: None,
                    call_id: Some("call-1".into()),
                    remote_id: None,
                }],
                tools: vec![tool.definition()],
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    runner.next().await.unwrap();
    assert!(!request_text(&model.requests()[0]).contains("Saved note index"));
    runner.follow_up("continue".to_string());
    runner.next().await.unwrap();
    assert!(request_text(&model.requests()[1]).contains("Saved note index"));
}
