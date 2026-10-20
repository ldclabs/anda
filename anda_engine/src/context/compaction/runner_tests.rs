use anda_core::{
    AgentOutput, BoxError, CompletionRequest, ContentPart, FunctionDefinition, Json, Message,
    Resource, Tool, ToolCall, ToolOutput, Usage,
};
use serde_json::json;
use std::sync::Arc;

use super::{COMPACT_CONTEXT_TOOL, COMPACTION_CONTINUE_PROMPT, NOTE_PREFIX, SelfCompaction};
use crate::{
    context::{AgentCtx, BaseCtx, COMPACTION_PROMPT, test_fixtures::EchoTool},
    engine::EngineBuilder,
    model::{Model, testing::ScriptedCompleter},
};

fn handoff_text() -> String {
    "Objective: finish the parser refactor in /work/parser.rs. ".repeat(5)
}

fn compact_call(handoff: &str) -> ToolCall {
    ToolCall {
        name: COMPACT_CONTEXT_TOOL.to_string(),
        args: json!({"reason": "earlier exploration is done", "handoff": handoff}),
        call_id: Some("compact_call".into()),
        result: None,
        remote_id: None,
    }
}

fn echo_call() -> ToolCall {
    ToolCall {
        name: "echo_tool".to_string(),
        args: json!({"input": "lookup"}),
        call_id: Some("echo_call".into()),
        result: None,
        remote_id: None,
    }
}

/// Mirrors the provider driver: the request history is replayed but not echoed, while the
/// request content message and the reply are; raw history holds everything sent plus the reply.
fn driver_reply(
    req: &CompletionRequest,
    text: &str,
    tool_calls: Vec<ToolCall>,
    input_tokens: u64,
) -> AgentOutput {
    let mut chat_history = Vec::new();
    let mut content = req.content.clone();
    if !req.prompt.is_empty() {
        content.insert(0, req.prompt.clone().into());
    }
    if !content.is_empty() {
        chat_history.push(Message {
            role: req.role.clone().unwrap_or_else(|| "user".into()),
            content,
            ..Default::default()
        });
    }
    let mut assistant = Message {
        role: "assistant".into(),
        ..Default::default()
    };
    if !text.is_empty() {
        assistant.content.push(text.to_string().into());
    }
    for call in &tool_calls {
        assistant.content.push(ContentPart::ToolCall {
            name: call.name.clone(),
            args: call.args.clone(),
            call_id: call.call_id.clone(),
        });
    }
    chat_history.push(assistant);
    let raw_history = req
        .chat_history
        .iter()
        .chain(&chat_history)
        .map(|message| serde_json::to_value(message).unwrap())
        .collect();

    AgentOutput {
        content: text.to_string(),
        tool_calls,
        chat_history,
        raw_history,
        usage: Usage {
            input_tokens,
            output_tokens: 100,
            cached_tokens: 0,
            requests: 1,
        },
        ..Default::default()
    }
}

fn ctx_with(completer: Arc<ScriptedCompleter>, context_window: usize) -> AgentCtx {
    let mut model = Model::with_completer(completer);
    model.context_window = context_window;
    EngineBuilder::new()
        .with_model(model)
        .register_tool(Arc::new(EchoTool))
        .unwrap()
        .mock_ctx()
}

fn prompt_req(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        prompt: prompt.to_string(),
        ..Default::default()
    }
}

fn text_of(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Tool outputs a request carries, wherever the runner placed them.
fn tool_outputs(req: &CompletionRequest) -> Vec<(String, Json, Option<bool>)> {
    req.chat_history
        .iter()
        .flat_map(|message| &message.content)
        .chain(&req.content)
        .filter_map(|part| match part {
            ContentPart::ToolOutput {
                name,
                output,
                is_error,
                ..
            } => Some((name.clone(), output.clone(), *is_error)),
            _ => None,
        })
        .collect()
}

fn mentions_note(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|message| text_of(&message.content).contains(NOTE_PREFIX))
}

#[tokio::test(flavor = "current_thread")]
async fn admitted_request_uses_the_model_handoff_and_redelivers_the_input() {
    let handoff = handoff_text();
    let completer = ScriptedCompleter::new("compactor")
        .push_with({
            let handoff = handoff.clone();
            move |req| Ok(driver_reply(req, "", vec![compact_call(&handoff)], 50_000))
        })
        .push_with(|req| Ok(driver_reply(req, "answered", vec![], 2_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("an unrelated new task"), Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.tool_calls[0].name, COMPACT_CONTEXT_TOOL);
    assert!(runner.needs_compaction_with(|| 0));

    let (mut runner, output) = runner.handoff(None).await.unwrap();
    // The model wrote the handoff: no summary turn was sent.
    assert_eq!(completer.requests().len(), 1);
    assert_eq!(output.content, handoff.trim());
    let call = output
        .tool_calls
        .iter()
        .find(|call| call.name == COMPACT_CONTEXT_TOOL)
        .unwrap();
    assert_eq!(
        call.result.as_ref().unwrap().output,
        json!({"status": "accepted"})
    );
    assert_eq!(
        output.chat_history.last().unwrap().content[0],
        ContentPart::ToolOutput {
            name: COMPACT_CONTEXT_TOOL.to_string(),
            output: json!({"status": "accepted"}),
            is_error: None,
            call_id: Some("compact_call".into()),
            remote_id: None,
        }
    );
    assert!(runner.self_compaction().is_some());
    assert_eq!(
        runner.chat_history()[0].text().as_deref(),
        Some(handoff.trim())
    );
    // The model compacted before acting on its input, so the input comes back verbatim.
    assert!(!runner.is_idle());

    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "answered");
    let requests = completer.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].tools.last().map(|tool| tool.name.as_str()),
        Some(COMPACT_CONTEXT_TOOL)
    );
    assert_eq!(
        requests[1].chat_history[0].text().as_deref(),
        Some(handoff.trim())
    );
    assert_eq!(text_of(&requests[1].content), "an unrelated new task");
    assert_eq!(
        requests[1].tools.last().map(|tool| tool.name.as_str()),
        Some(COMPACT_CONTEXT_TOOL)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn admitted_request_after_tool_rounds_resumes_with_the_continuation_prompt() {
    let handoff = handoff_text();
    let completer = ScriptedCompleter::new("compactor")
        .push_with(|req| Ok(driver_reply(req, "", vec![echo_call()], 10_000)))
        .push_with({
            let handoff = handoff.clone();
            move |req| Ok(driver_reply(req, "", vec![compact_call(&handoff)], 45_000))
        })
        .push_with(|req| Ok(driver_reply(req, "resumed", vec![], 2_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("refactor the parser"), Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.tool_calls[0].name, COMPACT_CONTEXT_TOOL);
    assert!(runner.needs_compaction_with(|| 0));

    let (mut runner, _) = runner.handoff(None).await.unwrap();
    assert_eq!(completer.requests().len(), 2);
    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "resumed");
    let requests = completer.requests();
    assert_eq!(text_of(&requests[2].content), COMPACTION_CONTINUE_PROMPT);
    assert_eq!(
        requests[2].chat_history[0].text().as_deref(),
        Some(handoff.trim())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn rejected_requests_answer_the_model_and_the_task_continues() {
    let handoff = handoff_text();
    let completer = ScriptedCompleter::new("compactor")
        .push_with({
            let handoff = handoff.clone();
            move |req| Ok(driver_reply(req, "", vec![compact_call(&handoff)], 10_000))
        })
        .push_with(|req| Ok(driver_reply(req, "carried on", vec![], 11_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("small task"), Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    // Too early to be worth the loss: the host is not asked to compact.
    assert!(!runner.needs_compaction_with(|| 0));
    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "carried on");
    let requests = completer.requests();
    let outputs = tool_outputs(&requests[1]);
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].0, COMPACT_CONTEXT_TOOL);
    assert_eq!(outputs[0].2, Some(true));
    assert!(
        outputs[0].1["error"]
            .as_str()
            .unwrap()
            .contains("only 10% used"),
        "{}",
        outputs[0].1
    );
}

#[tokio::test(flavor = "current_thread")]
async fn mixed_requests_run_the_other_calls_and_reject_the_compaction() {
    let handoff = handoff_text();
    let completer = ScriptedCompleter::new("compactor")
        .push_with({
            let handoff = handoff.clone();
            move |req| {
                Ok(driver_reply(
                    req,
                    "",
                    vec![echo_call(), compact_call(&handoff)],
                    40_000,
                ))
            }
        })
        .push_with(|req| Ok(driver_reply(req, "done", vec![], 41_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("mixed"), Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    assert!(!runner.needs_compaction_with(|| 0));
    runner.next().await.unwrap().unwrap();
    let outputs = tool_outputs(&completer.requests()[1]);
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].0, "echo_tool");
    assert_eq!(outputs[0].1, json!("echoed:lookup"));
    assert_eq!(outputs[1].0, COMPACT_CONTEXT_TOOL);
    assert!(
        outputs[1].1["error"]
            .as_str()
            .unwrap()
            .contains("only call")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn requests_the_host_does_not_run_or_cannot_see_are_unavailable() {
    let handoff = handoff_text();
    let script = |handoff: String| {
        ScriptedCompleter::new("compactor")
            .push_with(move |req| Ok(driver_reply(req, "", vec![compact_call(&handoff)], 50_000)))
            .push_with(|req| Ok(driver_reply(req, "done", vec![], 51_000)))
            .into_arc()
    };

    // An admitted request reaching `next` means the host skipped `needs_compaction_with`.
    let completer = script(handoff.clone());
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("task"), Vec::new())
        .with_self_compaction(Some(SelfCompaction {
            notice_percent: None,
            ..Default::default()
        }))
        .unwrap();
    runner.next().await.unwrap().unwrap();
    runner.next().await.unwrap().unwrap();
    let outputs = tool_outputs(&completer.requests()[1]);
    assert!(
        outputs[0].1["error"]
            .as_str()
            .unwrap()
            .contains("unavailable")
    );

    // Disabled: the definition is not offered and the name is dispatched like any other.
    let completer = script(handoff);
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx.completion_iter(prompt_req("task"), Vec::new());
    runner.next().await.unwrap().unwrap();
    assert!(!runner.needs_compaction_with(|| 0));
    runner.next().await.unwrap().unwrap();
    let requests = completer.requests();
    assert!(requests[0].tools.is_empty());
    let outputs = tool_outputs(&requests[1]);
    assert!(
        outputs[0].1["error"]
            .as_str()
            .unwrap()
            .contains("not found")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn host_compaction_mid_task_resumes_the_task() {
    let completer = ScriptedCompleter::new("summarizer")
        .push_with(|req| Ok(driver_reply(req, "", vec![echo_call()], 1_000)))
        .push_with(|req| Ok(driver_reply(req, "summary of the work", vec![], 1_000)))
        .push_with(|req| Ok(driver_reply(req, "resumed", vec![], 1_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx.completion_iter(prompt_req("long task"), Vec::new());

    runner.next().await.unwrap().unwrap();
    let (mut runner, output) = runner.handoff(None).await.unwrap();
    assert_eq!(output.content, "summary of the work");
    let requests = completer.requests();
    assert_eq!(
        text_of(&requests[1].content).trim(),
        COMPACTION_PROMPT.trim()
    );
    // The echo result was folded into the summary before the model saw it.
    assert!(!runner.is_idle());

    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "resumed");
    let requests = completer.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(text_of(&requests[2].content), COMPACTION_CONTINUE_PROMPT);

    // Compacting at an idle boundary leaves nothing to resume.
    let completer = ScriptedCompleter::new("summarizer")
        .push_with(|req| Ok(driver_reply(req, "finished", vec![], 1_000)))
        .push_with(|req| Ok(driver_reply(req, "summary", vec![], 1_000)))
        .into_arc();
    let ctx = ctx_with(completer, 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("short task"), Vec::new())
        .unbound();
    runner.next().await.unwrap().unwrap();
    let (runner, _) = runner.handoff(None).await.unwrap();
    assert!(runner.is_idle());
}

#[tokio::test(flavor = "current_thread")]
async fn long_tasks_compact_on_usage_not_on_turn_count() {
    let mut completer = ScriptedCompleter::new("worker");
    for _ in 0..100 {
        completer = completer.push_with(|req| Ok(driver_reply(req, "", vec![echo_call()], 1_000)));
    }
    let completer = completer
        .push_with(|req| Ok(driver_reply(req, "", vec![echo_call()], 80_000)))
        .into_arc();
    let ctx = ctx_with(completer, 100_000);
    let mut runner = ctx.completion_iter(prompt_req("long task"), Vec::new());

    // However many turns a task takes, little context in use asks for no compaction.
    for _ in 0..100 {
        runner.next().await.unwrap().unwrap();
    }
    assert_eq!(runner.turns(), 100);
    assert!(!runner.needs_compaction_with(|| 0));

    // Usage at 80% of the window does.
    runner.next().await.unwrap().unwrap();
    assert!(runner.needs_compaction_with(|| 0));
}

#[tokio::test(flavor = "current_thread")]
async fn context_notes_follow_tool_outputs_and_stay_out_of_the_history() {
    let completer = ScriptedCompleter::new("noted")
        .push_with(|req| Ok(driver_reply(req, "", vec![echo_call()], 55_000)))
        .push_with(|req| Ok(driver_reply(req, "done", vec![], 56_000)))
        .push_with(|req| Ok(driver_reply(req, "ok", vec![], 57_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("task"), Vec::new())
        .unbound()
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "done");

    let requests = completer.requests();
    assert!(!mentions_note(&requests[0].chat_history));
    // Tool round: the outputs, then the note, with no content message after them.
    let tool_round = &requests[1];
    assert!(tool_round.content.is_empty());
    assert_eq!(tool_round.role, None);
    let tail = &tool_round.chat_history[tool_round.chat_history.len() - 2..];
    assert_eq!(tail[0].role, "tool");
    assert!(matches!(
        &tail[0].content[0],
        ContentPart::ToolOutput { name, .. } if name == "echo_tool"
    ));
    assert_eq!(tail[1].role, "user");
    let note = tail[1].text().unwrap();
    assert!(note.starts_with(NOTE_PREFIX), "{note}");
    assert!(note.contains("(55%)"), "{note}");
    assert!(note.contains("including these tool results"), "{note}");

    // The outputs are committed in order; the note only reaches the raw history.
    let history = runner.chat_history();
    let tool_index = history
        .iter()
        .position(|message| message.role == "tool")
        .unwrap();
    assert_eq!(history[tool_index + 1].text().as_deref(), Some("done"));
    assert!(!mentions_note(history));
    let raw = serde_json::to_string(&runner.req().raw_history).unwrap();
    assert!(raw.contains(NOTE_PREFIX));

    // User turns are reported ahead of the new input.
    runner.follow_up("next question".to_string());
    runner.next().await.unwrap().unwrap();
    let requests = completer.requests();
    let user_turn = &requests[2];
    assert_eq!(text_of(&user_turn.content), "next question");
    let note = user_turn.chat_history.last().unwrap().text().unwrap();
    assert!(note.contains("from this message"), "{note}");
    assert!(note.contains("call compact_context first"), "{note}");
    assert!(!mentions_note(runner.chat_history()));
}

struct ReservedTool;

impl Tool<BaseCtx> for ReservedTool {
    type Args = Json;
    type Output = Json;

    fn name(&self) -> String {
        COMPACT_CONTEXT_TOOL.to_string()
    }

    fn description(&self) -> String {
        "collides with the runner".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({"type": "object", "properties": {}}),
            strict: None,
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Json,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        Ok(ToolOutput::new(args))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn enabling_rejects_the_reserved_name_and_bad_percentages() {
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(
            ScriptedCompleter::new("plain").into_arc(),
        ))
        .register_tool(Arc::new(ReservedTool))
        .unwrap()
        .mock_ctx();
    let mut runner = ctx.completion_iter(CompletionRequest::default(), Vec::new());
    let err = runner
        .set_self_compaction(Some(SelfCompaction::default()))
        .unwrap_err();
    assert!(err.to_string().contains("reserved"), "{err}");
    assert!(runner.self_compaction().is_none());

    let ctx = ctx_with(ScriptedCompleter::new("plain").into_arc(), 0);
    let mut runner = ctx.completion_iter(CompletionRequest::default(), Vec::new());
    assert!(
        runner
            .set_self_compaction(Some(SelfCompaction {
                min_percent: 101,
                ..Default::default()
            }))
            .is_err()
    );
    runner
        .set_self_compaction(Some(SelfCompaction::default()))
        .unwrap();
    assert_eq!(runner.self_compaction(), Some(&SelfCompaction::default()));
    runner.set_self_compaction(None).unwrap();
    assert!(runner.self_compaction().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn steering_during_a_model_request_follows_the_redelivered_input() {
    let handoff = handoff_text();
    let completer = ScriptedCompleter::new("compactor")
        .push_with({
            let handoff = handoff.clone();
            move |req| Ok(driver_reply(req, "", vec![compact_call(&handoff)], 50_000))
        })
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("new task"), Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    runner.steer("but keep it short".to_string());
    assert!(runner.needs_compaction_with(|| 0));

    let (mut runner, output) = runner.handoff(None).await.unwrap();
    // The request is still the model's own: answered as accepted, not interrupted.
    let call = output
        .tool_calls
        .iter()
        .find(|call| call.name == COMPACT_CONTEXT_TOOL)
        .unwrap();
    assert_eq!(
        call.result.as_ref().unwrap().output,
        json!({"status": "accepted"})
    );
    assert_eq!(output.content, handoff.trim());

    // The input the model compacted before acting on comes first, the steering after it.
    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "new task\n\nbut keep it short");
    let requests = completer.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        text_of(&requests[1].content),
        "new task\n\nbut keep it short"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_request_right_after_a_compaction_is_refused() {
    let handoff = handoff_text();
    let compact = |handoff: String| {
        move |req: &CompletionRequest| {
            Ok(driver_reply(req, "", vec![compact_call(&handoff)], 50_000))
        }
    };
    let completer = ScriptedCompleter::new("compactor")
        .push_with(compact(handoff.clone()))
        .push_with(compact(handoff.clone()))
        .push_with(|req| Ok(driver_reply(req, "worked", vec![], 51_000)))
        .into_arc();
    let ctx = ctx_with(completer.clone(), 100_000);
    let mut runner = ctx
        .completion_iter(prompt_req("task"), Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    let (mut runner, _) = runner.handoff(None).await.unwrap();

    // Usage would admit it, but nothing was done since the handoff.
    runner.next().await.unwrap().unwrap();
    assert!(!runner.needs_compaction_with(|| 0));
    let step = runner.next().await.unwrap().unwrap();
    assert_eq!(step.content, "worked");
    let outputs = tool_outputs(&completer.requests()[2]);
    assert_eq!(outputs[0].0, COMPACT_CONTEXT_TOOL);
    assert!(
        outputs[0].1["error"]
            .as_str()
            .unwrap()
            .contains("compacted just before"),
        "{}",
        outputs[0].1
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_seeded_conversation_is_estimated_before_its_first_turn() {
    let completer = ScriptedCompleter::new("estimated").into_arc();
    let ctx = ctx_with(completer.clone(), 20_000);
    let req = CompletionRequest {
        chat_history: vec![Message {
            role: "user".into(),
            content: vec!["x".repeat(48_000).into()],
            ..Default::default()
        }],
        prompt: "task".to_string(),
        ..Default::default()
    };
    let mut runner = ctx
        .completion_iter(req, Vec::new())
        .with_self_compaction(Some(SelfCompaction::default()))
        .unwrap();

    runner.next().await.unwrap().unwrap();
    let requests = completer.requests();
    let note = requests[0].chat_history.last().unwrap().text().unwrap();
    assert!(note.starts_with(NOTE_PREFIX), "{note}");
    assert!(
        note.contains("About 12k of 20k context tokens are in use (60%)"),
        "{note}"
    );
    assert!(!mentions_note(runner.chat_history()));
}
