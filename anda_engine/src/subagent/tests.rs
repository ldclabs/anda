use super::*;
use crate::context::COMPACTION_PROMPT;
use crate::engine::EngineBuilder;
use crate::management::{BaseManagement, Visibility};
use crate::memory::Conversations;
use crate::model::{CompletionFeaturesDyn, Model, Models};
use anda_core::{AgentInput, BoxPinFut, Message, Tool};
use anda_db::database::{AndaDB, DBConfig};
use async_trait::async_trait;
use candid::Principal;
use object_store::memory::InMemory;
use parking_lot::Mutex;
use serde_json::json;
use std::collections::BTreeSet;

async fn test_conversations(name: &str) -> Conversations {
    let db = Arc::new(
        AndaDB::connect(Arc::new(InMemory::new()), DBConfig::default())
            .await
            .unwrap(),
    );
    Conversations::connect(db, name.to_string()).await.unwrap()
}

fn public_management() -> Arc<BaseManagement> {
    Arc::new(BaseManagement {
        controller: Principal::management_canister(),
        managers: BTreeSet::new(),
        visibility: Visibility::Public,
    })
}

async fn wait_for_conversation_status(
    conversations: &Conversations,
    id: u64,
    status: ConversationStatus,
) -> Conversation {
    for _ in 0..40 {
        let conversation = conversations.get_conversation(id).await.unwrap();
        if conversation.status == status {
            return conversation;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    panic!("conversation {id} did not reach status {status}");
}

#[derive(Clone, Debug)]
struct EchoCompleter;

impl CompletionFeaturesDyn for EchoCompleter {
    fn model_name(&self) -> String {
        "echo".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let content = request_text(&req);
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct HistoryCompleter;

impl CompletionFeaturesDyn for HistoryCompleter {
    fn model_name(&self) -> String {
        "history".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let content = request_text(&req);
        let timestamp = unix_ms();
        let mut chat_history = req.chat_history.clone();
        if !content.is_empty() {
            chat_history.push(Message {
                role: req.role.unwrap_or_else(|| "user".to_string()),
                content: vec![content.clone().into()],
                timestamp: Some(timestamp),
                ..Default::default()
            });
        }
        chat_history.push(Message {
            role: "assistant".to_string(),
            content: vec![format!("done: {content}").into()],
            name: Some("history".to_string()),
            timestamp: Some(timestamp),
            ..Default::default()
        });

        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: format!("done: {content}"),
            chat_history,
            usage: Usage {
                input_tokens: 2,
                output_tokens: 3,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct RecordingCompactionCompleter {
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for RecordingCompactionCompleter {
    fn model_name(&self) -> String {
        "recording-compaction".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().push(req.clone());

        let prompt = request_text(&req);
        let history = req
            .chat_history
            .iter()
            .filter_map(Message::text)
            .collect::<Vec<_>>()
            .join(" | ");

        let content = if prompt.trim() == COMPACTION_PROMPT.trim() {
            "compacted handoff".to_string()
        } else if history.is_empty() {
            prompt.clone()
        } else {
            format!("history={history}; input={prompt}")
        };

        let input_tokens = if prompt.trim() == COMPACTION_PROMPT.trim() {
            1
        } else {
            100_000
        };

        Box::pin(futures::future::ready(Ok(AgentOutput {
            content,
            usage: Usage {
                input_tokens,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct RecordingLowUsageCompactionCompleter {
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for RecordingLowUsageCompactionCompleter {
    fn model_name(&self) -> String {
        "recording-low-usage-compaction".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().push(req.clone());

        let prompt = request_text(&req);
        let history = req
            .chat_history
            .iter()
            .filter_map(Message::text)
            .collect::<Vec<_>>()
            .join(" | ");

        let content = if prompt.trim() == COMPACTION_PROMPT.trim() {
            "compacted handoff".to_string()
        } else if history.is_empty() {
            prompt.clone()
        } else {
            format!("history={history}; input={prompt}")
        };

        Box::pin(futures::future::ready(Ok(AgentOutput {
            content,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct PendingToolCompactionCompleter {
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for PendingToolCompactionCompleter {
    fn model_name(&self) -> String {
        "pending-tool-compaction".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().push(req.clone());

        let prompt = request_text(&req);
        let is_compaction = prompt.trim() == COMPACTION_PROMPT.trim();
        let output = if is_compaction {
            AgentOutput {
                content: "compacted handoff".to_string(),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            }
        } else {
            AgentOutput {
                content: "looking up".to_string(),
                tool_calls: vec![anda_core::ToolCall {
                    name: "compact_echo".to_string(),
                    args: json!({"input": "lookup"}),
                    result: None,
                    call_id: Some("call-compact".to_string()),
                    remote_id: None,
                }],
                chat_history: vec![Message {
                    role: "assistant".to_string(),
                    content: vec![ContentPart::ToolCall {
                        name: "compact_echo".to_string(),
                        args: json!({"input": "lookup"}),
                        call_id: Some("call-compact".to_string()),
                    }],
                    ..Default::default()
                }],
                usage: Usage {
                    input_tokens: 100_000,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            }
        };

        Box::pin(futures::future::ready(Ok(output)))
    }
}

#[derive(Clone, Debug)]
struct EmptyCompactionCompleter;

impl CompletionFeaturesDyn for EmptyCompactionCompleter {
    fn model_name(&self) -> String {
        "empty-compaction".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let prompt = request_text(&req);
        let is_compaction = prompt.trim() == COMPACTION_PROMPT.trim();
        Box::pin(futures::future::ready(Ok(AgentOutput {
            // Normal turns echo their input and push usage over the compaction threshold; the
            // compaction turn returns no usable summary at all.
            content: if is_compaction { String::new() } else { prompt },
            usage: Usage {
                input_tokens: if is_compaction { 1 } else { 100_000 },
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct RecordingRequestCompleter {
    name: &'static str,
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for RecordingRequestCompleter {
    fn model_name(&self) -> String {
        self.name.to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let content = request_text(&req);
        self.requests.lock().push(req);
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct ErrorCompleter;

impl CompletionFeaturesDyn for ErrorCompleter {
    fn model_name(&self) -> String {
        "error".to_string()
    }

    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Err("model failed".into())))
    }
}

struct CompactEchoTool;

#[derive(Debug, Deserialize)]
struct CompactEchoArgs {
    input: String,
}

impl Tool<BaseCtx> for CompactEchoTool {
    type Args = CompactEchoArgs;
    type Output = String;

    fn name(&self) -> String {
        "compact_echo".to_string()
    }

    fn description(&self) -> String {
        "Echoes input during compaction tests".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "compact_echo".to_string(),
            description: "Echoes input during compaction tests".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<String>, BoxError> {
        Ok(ToolOutput {
            output: format!("echoed:{}", args.input),
            usage: Usage {
                input_tokens: 2,
                output_tokens: 3,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })
    }
}

#[derive(Clone, Debug)]
struct ToolCallProgressCompleter;

impl CompletionFeaturesDyn for ToolCallProgressCompleter {
    fn model_name(&self) -> String {
        "tool-call-progress".to_string()
    }

    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![anda_core::ToolCall {
                name: "long_running_tool".to_string(),
                args: json!({}),
                result: None,
                call_id: Some("call-1".to_string()),
                remote_id: None,
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

// Emits visible narration alongside a tool call, so the step carries a progress signal while
// tool calls keep the runner busy (non-idle).
#[derive(Clone, Debug)]
struct NarratingToolCallCompleter;

impl CompletionFeaturesDyn for NarratingToolCallCompleter {
    fn model_name(&self) -> String {
        "narrating-tool-call".to_string()
    }

    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: "searching now".to_string(),
            tool_calls: vec![anda_core::ToolCall {
                name: "long_running_tool".to_string(),
                args: json!({}),
                result: None,
                call_id: Some("call-1".to_string()),
                remote_id: None,
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
struct StopResumeCompleter {
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for StopResumeCompleter {
    fn model_name(&self) -> String {
        "stop-resume".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let content = request_text(&req);
        self.requests.lock().push(req);

        let output = if content == "seed task" {
            AgentOutput {
                content: "working".to_string(),
                tool_calls: vec![anda_core::ToolCall {
                    name: "long_running_tool".to_string(),
                    args: json!({}),
                    result: None,
                    call_id: Some("call-1".to_string()),
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            }
        } else {
            AgentOutput {
                content,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            }
        };

        Box::pin(futures::future::ready(Ok(output)))
    }
}

#[derive(Clone, Default)]
struct RecordingAgentHook {
    progress: Arc<Mutex<Vec<(String, AgentOutput)>>>,
}

impl RecordingAgentHook {
    fn progress_events(&self) -> Vec<(String, AgentOutput)> {
        self.progress.lock().clone()
    }
}

#[derive(Clone, Default)]
struct FailingAfterAgentHook {
    starts: Arc<Mutex<Vec<String>>>,
}

#[derive(Default)]
struct EmptySubAgentSet;

impl SubAgentSet for EmptySubAgentSet {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn contains_lowercase(&self, _lowercase_name: &str) -> bool {
        false
    }

    fn get_lowercase(&self, _lowercase_name: &str) -> Option<SubAgent> {
        None
    }

    fn definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
        Vec::new()
    }

    fn select_resources(&self, _name: &str, _resources: &mut Vec<Resource>) -> Vec<Resource> {
        Vec::new()
    }
}

#[async_trait]
impl AgentHook for FailingAfterAgentHook {
    async fn after_agent_run(
        &self,
        _ctx: &AgentCtx,
        _output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        Err("after hook rejected output".into())
    }

    async fn on_background_start(
        &self,
        _ctx: &AgentCtx,
        handle: BackgroundHandle,
        _req: &CompletionRequest,
    ) {
        self.starts.lock().push(handle.task_id().to_string());
    }
}

#[async_trait]
impl AgentHook for RecordingAgentHook {
    async fn on_background_progress(
        &self,
        _ctx: &AgentCtx,
        session_id: String,
        output: AgentOutput,
    ) {
        self.progress.lock().push((session_id, output));
    }
}

fn request_text(req: &CompletionRequest) -> String {
    if !req.prompt.is_empty() {
        return req.prompt.clone();
    }

    Message {
        content: req.content.clone(),
        ..Default::default()
    }
    .text()
    .unwrap_or_default()
}

async fn recv_subagent_prompt(rx: &mut tokio::sync::mpsc::Receiver<SubAgentInput>) -> String {
    match rx.recv().await.unwrap().command {
        PromptCommand::Plain { prompt } => prompt,
        other => panic!("unexpected command: {other:?}"),
    }
}

fn resource(id: u64, tags: &[&str]) -> Resource {
    Resource {
        _id: id,
        name: format!("resource-{id}"),
        tags: tags.iter().map(|tag| tag.to_string()).collect(),
        ..Default::default()
    }
}

#[test]
fn subagent_metadata_args_and_content_helpers_cover_edge_inputs() {
    assert_eq!(
        summarize_items(&["a".to_string(), "b".to_string(), "c".to_string()], 1),
        "a, and 2 more"
    );
    assert_eq!(selected_model_label("  Pro "), Some("pro".to_string()));
    assert_eq!(selected_model_label("  "), None);

    let agent = SubAgent {
        name: "meta_worker".to_string(),
        description: "  ".to_string(),
        tools: (0..10).map(|idx| format!("tool_{idx}")).collect(),
        tags: (0..10).map(|idx| format!("tag_{idx}")).collect(),
        output_schema: Some(json!({"type": "object"})),
        model: " Pro ".to_string(),
        effort: Some(ModelEffort::Max),
        idle_timeout: 120,
        ..Default::default()
    };
    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    agent.subsessions.insert_session(Arc::new(SubSession::new(
        "Job-A".to_string(),
        "meta_worker".to_string(),
        sender,
        0,
    )));
    let definition = agent.definition();
    assert!(
        definition
            .description
            .starts_with("Delegated subagent worker.")
    );
    assert!(definition.description.contains("Tags: tag_0"));
    assert!(definition.description.contains("and 2 more"));
    assert!(
        definition
            .description
            .contains("Returns structured output.")
    );
    assert!(definition.description.contains("Default model label: pro."));
    assert!(definition.description.contains("Default effort: max."));
    assert!(
        definition
            .description
            .contains("Session idle timeout: 120s.")
    );
    assert!(definition.description.contains("Active sessions: Job-A."));
    assert_eq!(Agent::<AgentCtx>::description(&agent), "  ");
    assert_eq!(
        Agent::<AgentCtx>::tool_dependencies(&agent),
        (0..10).map(|idx| format!("tool_{idx}")).collect::<Vec<_>>()
    );
    assert_eq!(
        Agent::<AgentCtx>::supported_resource_tags(&agent),
        (0..10).map(|idx| format!("tag_{idx}")).collect::<Vec<_>>()
    );

    let args = SubAgentArgs::from_prompt(
        json!({
            "prompt": "structured",
            "session": "CASE",
            "model": "Flash",
            "effort": null
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(args.prompt, "structured");
    assert_eq!(args.effort, None);
    assert_eq!(
        SubAgentArgs::from_prompt("plain".to_string())
            .unwrap()
            .prompt,
        "plain"
    );

    // Invalid structured arguments must surface as errors instead of silently degrading
    // into a blocking run with the raw JSON as the prompt.
    assert!(
        SubAgentArgs::from_prompt(json!({"prompt": "task", "effort": "ultra"}).to_string())
            .is_err()
    );
    // JSON payloads that are not subagent arguments are task data, not arguments.
    let data_prompt = json!({"city": "Reykjavik", "population": 139000}).to_string();
    let args = SubAgentArgs::from_prompt(data_prompt.clone()).unwrap();
    assert_eq!(args.prompt, data_prompt);
    assert!(args.session.is_empty());
    // Text that merely starts with '{' is also a plain prompt.
    let text_prompt = "{\"a\":1} and {\"b\":2} differ, explain why".to_string();
    let args = SubAgentArgs::from_prompt(text_prompt.clone()).unwrap();
    assert_eq!(args.prompt, text_prompt);

    let manager_args = SubAgentManagerArgs::from_prompt(
        json!({
            "output_schema": {"type": "object"},
            "effort": null,
            "idle_timeout": 90,
            "task": "  run task  ",
            "session": "  Thread  ",
            "model": "  Pro  ",
            "persist": true
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(manager_args.output_schema, Some(json!({"type": "object"})));
    assert_eq!(manager_args.effort, None);
    assert_eq!(manager_args.idle_timeout, 90);
    let (agent, task, session, persist) = manager_args.into_subagent();
    assert_eq!(task.as_deref(), Some("run task"));
    assert_eq!(session, "thread");
    assert_eq!(agent.model, "Pro");
    assert_eq!(agent.idle_timeout, 90);
    assert!(persist);
    assert!(SubAgentManagerArgs::from_prompt("not json".to_string()).is_err());

    let manager_args = SubAgentManagerArgs::from_prompt(
        json!({
            "output_schema": "{\"type\":\"array\"}",
            "effort": " HIGH "
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(manager_args.output_schema, Some(json!({"type": "array"})));
    assert_eq!(manager_args.effort, Some(ModelEffort::High));

    let manager_args = SubAgentManagerArgs::from_prompt(
        json!({
            "output_schema": "",
            "effort": ""
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(manager_args.output_schema, None);
    assert_eq!(manager_args.effort, None);

    let manager_args = SubAgentManagerArgs::from_prompt(
        json!({
            "output_schema": null,
            "effort": null
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(manager_args.output_schema, None);
    assert_eq!(manager_args.effort, None);

    assert!(SubAgentManagerArgs::from_prompt(json!({"output_schema": "{"}).to_string()).is_err());
    assert!(
        SubAgentManagerArgs::from_prompt(json!({"effort": {"bad": true}}).to_string()).is_err()
    );

    let mut resources = vec![
        Resource {
            blob: Some(b"text resource".to_vec().into()),
            ..Default::default()
        },
        Resource {
            uri: Some("file://image.png".to_string()),
            mime_type: Some("image/png".to_string()),
            ..Default::default()
        },
        Resource::default(),
    ];
    let content = prompt_and_resources_into_content("prompt".to_string(), &mut resources);
    assert_eq!(content.len(), 3);
    assert!(resources.is_empty());
    assert!(resources_into_content(vec![Resource::default()]).is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_managers_cover_status_errors_persistence_and_set_routing() {
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(EchoCompleter)))
        .mock_ctx();
    let manager = SubAgentManager::new().with_models(vec!["flash".to_string(), "pro".to_string()]);
    assert!(manager.description().contains("flash, pro"));
    assert_eq!(manager.supported_resource_tags(), vec!["*".to_string()]);
    assert_eq!(manager.name(), SubAgentManager::NAME);

    let unsupported = Agent::<AgentCtx>::run(
        &manager,
        ctx.clone(),
        serde_json::to_string(&SubAgentManagerArgs {
            operation: "delete".to_string(),
            ..Default::default()
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap_err();
    assert!(unsupported.to_string().contains("unsupported"));

    let created = Agent::<AgentCtx>::run(
        &manager,
        ctx.clone(),
        serde_json::to_string(&SubAgentManagerArgs {
            operation: "create".to_string(),
            name: "router".to_string(),
            description: "Routes tagged resources.".to_string(),
            instructions: "Echo tasks.".to_string(),
            tags: vec!["text".to_string()],
            persist: true,
            ..Default::default()
        })
        .unwrap(),
        vec![resource(9, &["text"])],
    )
    .await
    .unwrap();
    let created_json: Json = serde_json::from_str(&created.content).unwrap();
    assert_eq!(created_json["result"], json!("persisted"));
    assert_eq!(created_json["callable"], json!("SA_router"));

    let status = Agent::<AgentCtx>::run(
        &manager,
        ctx.clone(),
        serde_json::to_string(&SubAgentManagerArgs {
            operation: "status".to_string(),
            ..Default::default()
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();
    assert_eq!(
        serde_json::from_str::<Json>(&status.content).unwrap()["count"],
        json!(1)
    );

    let filtered = manager.definitions(Some(&["ROUTER".to_string(), "missing".to_string()]));
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].name, "router");

    let mut resources = vec![resource(1, &["image"]), resource(2, &["text"])];
    let selected = SubAgentSet::select_resources(&manager, "ROUTER", &mut resources);
    assert_eq!(
        selected
            .iter()
            .map(|resource| resource._id)
            .collect::<Vec<_>>(),
        vec![2]
    );
    assert_eq!(
        resources
            .iter()
            .map(|resource| resource._id)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert!(SubAgentSet::select_resources(&manager, "missing", &mut resources).is_empty());
    resources.clear();
    assert!(SubAgentSet::select_resources(&manager, "router", &mut resources).is_empty());

    let set_manager = Arc::new(SubAgentSetManager::default());
    assert!(set_manager.definitions(None).is_empty());
    assert!(!set_manager.contains_lowercase("router"));
    assert!(set_manager.get_lowercase("router").is_none());
    let downcast = set_manager.clone().into_any();
    assert!(downcast.downcast_ref::<SubAgentSetManager>().is_some());
    assert!(set_manager.insert(Arc::new(manager)).is_none());
    assert!(set_manager.contains_lowercase("router"));
    assert!(set_manager.get::<SubAgentManager>().is_some());
    assert!(set_manager.get_lowercase("router").is_some());
    assert_eq!(
        set_manager.definitions(Some(&["router".to_string()])).len(),
        1
    );
    let mut resources = vec![resource(3, &["text"]), resource(4, &["audio"])];
    let selected = SubAgentSet::select_resources(set_manager.as_ref(), "router", &mut resources);
    assert_eq!(
        selected
            .iter()
            .map(|resource| resource._id)
            .collect::<Vec<_>>(),
        vec![3]
    );

    let empty_set_manager = Arc::new(SubAgentSetManager::default());
    assert!(
        empty_set_manager
            .insert(Arc::new(EmptySubAgentSet))
            .is_none()
    );
    assert!(!empty_set_manager.contains_lowercase("router"));
    assert!(empty_set_manager.get_lowercase("router").is_none());
    assert!(empty_set_manager.definitions(None).is_empty());
    let mut resources = vec![resource(5, &["text"])];
    assert!(
        SubAgentSet::select_resources(empty_set_manager.as_ref(), "router", &mut resources)
            .is_empty()
    );
    assert_eq!(
        resources
            .iter()
            .map(|resource| resource._id)
            .collect::<Vec<_>>(),
        vec![5]
    );
}

#[test]
fn resolve_idle_timeout_ms_applies_default_and_clamps() {
    // 0 keeps the engine default.
    assert_eq!(resolve_idle_timeout_ms(0), CONVERSATION_IDLE_MS);
    // A positive value converts seconds to milliseconds.
    assert_eq!(resolve_idle_timeout_ms(30), 30_000);
    // Anything above the background-task ceiling is clamped down to it.
    assert_eq!(
        resolve_idle_timeout_ms(u64::MAX),
        CONVERSATION_WAIT_BACKGROUND_TASK_MS
    );
    assert_eq!(
        resolve_idle_timeout_ms(CONVERSATION_WAIT_BACKGROUND_TASK_MS / 1000 + 60),
        CONVERSATION_WAIT_BACKGROUND_TASK_MS
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_blocking_persists_conversation() {
    let conversations = test_conversations("subagent_blocking_persists_conversation").await;
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(HistoryCompleter)))
        .with_subagent_conversations(conversations.clone())
        .mock_ctx();
    let agent = SubAgent {
        name: "auditor".to_string(),
        description: "Audits work.".to_string(),
        instructions: "Return the task result.".to_string(),
        model: "history".to_string(),
        ..Default::default()
    };

    let output =
        Agent::<AgentCtx>::run(&agent, ctx, "inspect blocking task".to_string(), Vec::new())
            .await
            .unwrap();

    let conversation_id = output.conversation.expect("conversation id");
    assert_eq!(output.content, "done: inspect blocking task");
    let conversation = conversations
        .get_conversation(conversation_id)
        .await
        .unwrap();
    assert_eq!(conversation.status, ConversationStatus::Completed);
    assert_eq!(conversation.label.as_deref(), Some("subagent:auditor"));
    assert_eq!(conversation.usage.requests, 1);
    assert_eq!(
        conversation.extra.as_ref().unwrap()["mode"],
        json!("blocking")
    );
    assert_eq!(
        conversation.extra.as_ref().unwrap()["subagent"],
        json!("auditor")
    );
    let messages = serde_json::to_string(&conversation.messages).unwrap();
    assert!(messages.contains("inspect blocking task"));
    assert!(messages.contains("done: inspect blocking task"));
}

#[tokio::test(flavor = "current_thread")]
async fn engine_agent_run_preserves_subagent_conversation_recorder_state() {
    let conversations = test_conversations("engine_agent_run_subagent_conversation").await;
    let agent = Arc::new(SubAgent {
        name: "auditor".to_string(),
        description: "Audits work.".to_string(),
        instructions: "Return the task result.".to_string(),
        model: "history".to_string(),
        ..Default::default()
    });
    let engine = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(HistoryCompleter)))
        .with_subagent_conversations(conversations.clone())
        .with_management(public_management())
        .register_agent(agent, Some("history".to_string()))
        .unwrap()
        .export_agents(vec!["auditor".to_string()])
        .empty()
        .await
        .unwrap();

    let output = engine
        .agent_run(
            Principal::anonymous(),
            AgentInput::new("auditor".to_string(), "inspect engine task".to_string()),
        )
        .await
        .unwrap();

    let conversation_id = output.conversation.expect("conversation id");
    assert_eq!(output.content, "done: inspect engine task");
    let conversation = conversations
        .get_conversation(conversation_id)
        .await
        .unwrap();
    assert_eq!(conversation.status, ConversationStatus::Completed);
    assert_eq!(
        conversation.extra.as_ref().unwrap()["mode"],
        json!("blocking")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_session_persists_conversation_and_reports_status_id() {
    let conversations = test_conversations("subagent_session_persists_conversation").await;
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(HistoryCompleter)))
        .with_subagent_conversations(conversations.clone())
        .mock_ctx();
    let agent = SubAgent {
        name: "worker".to_string(),
        description: "Runs background work.".to_string(),
        instructions: "Return the task result.".to_string(),
        model: "history".to_string(),
        ..Default::default()
    };

    let start = Agent::<AgentCtx>::run(
        &agent,
        ctx.clone(),
        serde_json::to_string(&SubAgentArgs {
            prompt: "session seed".to_string(),
            session: "AuditJob".to_string(),
            model: String::new(),
            effort: None,
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();

    let conversation_id = start.conversation.expect("conversation id");
    assert_eq!(start.session.as_deref(), Some("auditjob"));
    let status = Agent::<AgentCtx>::run(
        &agent,
        ctx.clone(),
        serde_json::to_string(&SubAgentArgs {
            prompt: "/status".to_string(),
            session: "AuditJob".to_string(),
            model: String::new(),
            effort: None,
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();
    let status_json: Json = serde_json::from_str(&status.content).unwrap();
    assert_eq!(status_json["conversation"], json!(conversation_id));
    assert_eq!(status.conversation, Some(conversation_id));

    Agent::<AgentCtx>::run(
        &agent,
        ctx,
        serde_json::to_string(&SubAgentArgs {
            prompt: "/cancel audit complete".to_string(),
            session: "AuditJob".to_string(),
            model: String::new(),
            effort: None,
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();

    let conversation = wait_for_conversation_status(
        &conversations,
        conversation_id,
        ConversationStatus::Cancelled,
    )
    .await;
    assert_eq!(
        conversation.failed_reason.as_deref(),
        Some("audit complete")
    );
    assert_eq!(
        conversation.extra.as_ref().unwrap()["mode"],
        json!("session")
    );
    assert_eq!(
        conversation.extra.as_ref().unwrap()["session"],
        json!("auditjob")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_honors_configured_idle_timeout() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    // Builds a runner whose session uses the given idle window, seeds it with one step, then
    // rewinds `active_at` by a second to simulate the window elapsing.
    async fn aged_runner(
        ctx: &AgentCtx,
        idle_timeout_ms: u64,
    ) -> (Arc<SubSession>, SubSessionRunner) {
        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession::new(
            "session-1".to_string(),
            "worker".to_string(),
            sender,
            idle_timeout_ms,
        ));
        let mut runner = SubSessionRunner {
            session: session.clone(),
            agent_hook: None,
            runner: ctx
                .clone()
                .completion_iter(
                    CompletionRequest {
                        prompt: "seed task".to_string(),
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .unbound(),
            conversation: None,
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };
        // First step consumes the seed and keeps the session active.
        assert!(runner.run(Vec::new()).await.unwrap());
        session
            .active_at
            .store(unix_ms().saturating_sub(1000), Ordering::SeqCst);
        (session, runner)
    }

    // A tiny window: ~1s of inactivity exceeds it, so the idle session is reclaimed.
    let (_session, mut tight) = aged_runner(&ctx, 5).await;
    assert!(!tight.run(Vec::new()).await.unwrap());

    // A generous window: the same ~1s of inactivity stays well within it, so the session
    // survives. This proves the boundary tracks the configured value, not merely "active_at is
    // old".
    let (_session, mut roomy) = aged_runner(&ctx, 60_000).await;
    assert!(roomy.run(Vec::new()).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_compacts_context_and_continues_from_handoff() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Model::with_completer(Arc::new(RecordingCompactionCompleter {
        requests: requests.clone(),
    }));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let hook = Arc::new(RecordingAgentHook::default());

    let req = CompletionRequest {
        instructions: "Keep working".to_string(),
        prompt: "seed task".to_string(),
        role: Some("user".to_string()),
        tools: vec![FunctionDefinition {
            name: "lookup".to_string(),
            description: "Lookup data.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            strict: Some(true),
        }],
        output_schema: Some(json!({
            "type": "object",
            "additionalProperties": false
        })),
        model: Some("flash".to_string()),
        effort: Some(ModelEffort::High),
        ..Default::default()
    };

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "compactor".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: Some(DynAgentHook::new(hook.clone())),
        runner: ctx.clone().completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert_eq!(
        runner.last_output.as_ref().map(|v| v.content.as_str()),
        Some("seed task")
    );

    let role_before_compaction = runner.runner.req().role.clone();
    let output_schema_before_compaction = runner.runner.req().output_schema.clone();
    let model_before_compaction = runner.runner.req().model.clone();
    let effort_before_compaction = runner.runner.req().effort;

    assert!(runner.run(Vec::new()).await.unwrap());

    // Usage accumulated before compaction is carried into the replacement runner.
    assert_eq!(runner.runner.total_usage().input_tokens, 100_001);
    assert_eq!(runner.runner.total_usage().requests, 2);

    let progress = hook.progress_events();
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].0, "session-1");
    assert_eq!(progress[0].1.content, "seed task");

    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 2);
    assert_eq!(request_text(&recorded[0]), "seed task");
    assert_eq!(request_text(&recorded[1]).trim(), COMPACTION_PROMPT.trim());
    assert!(recorded[1].tools.is_empty());

    assert_eq!(runner.runner.chat_history().len(), 1);
    assert_eq!(runner.runner.chat_history()[0].role, "assistant");
    assert_eq!(
        runner.runner.chat_history()[0].text().as_deref(),
        Some("compacted handoff")
    );
    assert_eq!(runner.runner.req().instructions, "Keep working");
    assert_eq!(runner.runner.req().role, role_before_compaction);
    assert_eq!(
        runner.runner.req().output_schema,
        output_schema_before_compaction
    );
    assert_eq!(runner.runner.req().model, model_before_compaction);
    assert_eq!(runner.runner.req().effort, effort_before_compaction);
    assert_eq!(runner.runner.req().tools.len(), 1);
    assert_eq!(runner.runner.req().tools[0].name, "lookup");
    assert!(runner.runner.req().prompt.is_empty());
    assert!(runner.runner.req().content.is_empty());
    assert_eq!(
        runner
            .last_output
            .as_ref()
            .map(|output| output.content.as_str()),
        Some("seed task")
    );

    assert!(
        runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Plain {
                    prompt: "continue after compaction".to_string(),
                },
                resources: Vec::new(),
                usage: Usage::default(),
                model: None,
                effort: None,
            }])
            .await
            .unwrap()
    );

    let resumed = runner.last_output.as_ref().unwrap();
    assert_eq!(resumed.session.as_deref(), Some("session-1"));
    assert_eq!(
        resumed.content,
        "history=compacted handoff; input=continue after compaction"
    );

    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 3);
    assert_eq!(
        recorded[2]
            .chat_history
            .iter()
            .filter_map(Message::text)
            .collect::<Vec<_>>(),
        vec!["compacted handoff".to_string()]
    );
    assert_eq!(request_text(&recorded[2]), "continue after compaction");
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_compacts_oversized_input_batch_before_queueing() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut model = Model::with_completer(Arc::new(RecordingLowUsageCompactionCompleter {
        requests: requests.clone(),
    }));
    model.context_window = 1_000; // compaction threshold = 800 tokens
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-batch".to_string(),
        "compactor".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx
            .completion_iter(CompletionRequest::default(), Vec::new())
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    // Each input carries no usage, so only the batch content estimate can trigger compaction.
    // Each chunk is ~400 tokens (under the 800 threshold); the three batched together are
    // ~1200 (over it). The per-input check missed this: queueing the first follow-up made the
    // runner report not-idle, so the rest bypassed the size check.
    let chunk = "y".repeat(1_600);
    let plain = |prompt: String| SubAgentInput {
        command: PromptCommand::Plain { prompt },
        resources: Vec::new(),
        usage: Usage::default(),
        model: None,
        effort: None,
    };

    assert!(
        runner
            .run(vec![
                plain(chunk.clone()),
                plain(chunk.clone()),
                plain(chunk.clone()),
            ])
            .await
            .unwrap()
    );

    let recorded = requests.lock().clone();
    // Compaction runs once up front, then the whole batch is queued on top of the compacted
    // handoff in a single follow-up request.
    assert_eq!(recorded.len(), 2);
    assert_eq!(request_text(&recorded[0]).trim(), COMPACTION_PROMPT.trim());
    assert_eq!(
        recorded[1]
            .chat_history
            .iter()
            .filter_map(Message::text)
            .collect::<Vec<_>>(),
        vec!["compacted handoff".to_string()]
    );
    assert_eq!(
        request_text(&recorded[1]).matches(chunk.as_str()).count(),
        3
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_compaction_executes_pending_tool_calls_first() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Model::with_completer(Arc::new(PendingToolCompactionCompleter {
        requests: requests.clone(),
    }));
    let ctx = EngineBuilder::new()
        .with_model(model)
        .register_tool(Arc::new(CompactEchoTool))
        .unwrap()
        .mock_ctx();

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-tool-compact".to_string(),
        "compactor".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "seed task".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert!(!runner.runner.no_pending_tool_calls());

    assert!(runner.run(Vec::new()).await.unwrap());

    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 2);
    assert_eq!(request_text(&recorded[0]), "seed task");
    assert_eq!(request_text(&recorded[1]).trim(), COMPACTION_PROMPT.trim());
    assert!(recorded[1].tools.is_empty());
    assert!(recorded[1].chat_history.iter().any(|message| {
        message.content.iter().any(|part| {
            matches!(
                part,
                ContentPart::ToolOutput {
                    name,
                    output,
                    call_id: Some(call_id),
                    ..
                } if name == "compact_echo"
                    && call_id == "call-compact"
                    && output.as_str() == Some("echoed:lookup")
            )
        })
    }));

    assert!(runner.runner.no_pending_tool_calls());
    assert_eq!(
        runner.runner.chat_history()[0].text().as_deref(),
        Some("compacted handoff")
    );
    assert_eq!(
        runner
            .runner
            .tools_usage()
            .get("compact_echo")
            .map(|usage| usage.requests),
        Some(1)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_fails_when_compaction_summary_is_empty() {
    let model = Model::with_completer(Arc::new(EmptyCompactionCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let req = CompletionRequest {
        instructions: "Keep working".to_string(),
        prompt: "seed task".to_string(),
        ..Default::default()
    };

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "compactor".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx.completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None, // A previously rescued artifact must survive even when compaction fails.
        carried_artifacts: vec![resource(5, &["artifact"])],
        closing: false,
    };

    // First step seeds the conversation and pushes usage over the compaction threshold.
    assert!(runner.run(Vec::new()).await.unwrap());

    // The next idle step triggers compaction. An empty summary must fail loudly instead of
    // replacing the entire conversation with an empty handoff message.
    let err = runner.run(Vec::new()).await.unwrap_err();
    assert!(err.to_string().contains("empty summary"), "{err}");

    let mut output = runner.latest_output();
    runner.merge_carried_artifacts(&mut output);
    assert_eq!(
        output.failed_reason.as_deref(),
        Some("context compaction produced an empty summary")
    );
    assert_eq!(output.session.as_deref(), Some("session-1"));
    assert!(output.content.is_empty());
    assert_eq!(
        output
            .artifacts
            .iter()
            .map(|artifact| artifact._id)
            .collect::<Vec<_>>(),
        vec![5]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_emits_progress_for_signal_steps_without_waiting_for_idle() {
    // A step that carries visible narration emits progress immediately, even while tool calls
    // keep the runner busy (i.e. without waiting for an idle boundary).
    let model = Model::with_completer(Arc::new(NarratingToolCallCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let hook = Arc::new(RecordingAgentHook::default());

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: Some(DynAgentHook::new(hook.clone())),
        runner: ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "start long work".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert!(!runner.runner.is_idle());

    let progress = hook.progress_events();
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].0, "session-1");
    assert_eq!(progress[0].1.session.as_deref(), Some("session-1"));
    assert_eq!(progress[0].1.content, "searching now");
    assert_eq!(progress[0].1.tool_calls.len(), 1);
    assert_eq!(
        runner
            .last_output
            .as_ref()
            .map(|output| output.tool_calls.len()),
        Some(1)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_filters_signalless_tool_call_steps_from_progress() {
    // A pure tool-call step carries no visible signal; it must be filtered out so the parent
    // is not flooded with mechanical "still calling tools" noise. The step is still tracked in
    // last_output for the eventual final report.
    let model = Model::with_completer(Arc::new(ToolCallProgressCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let hook = Arc::new(RecordingAgentHook::default());

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: Some(DynAgentHook::new(hook.clone())),
        runner: ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "start long work".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert!(!runner.runner.is_idle());

    assert!(hook.progress_events().is_empty());
    assert_eq!(
        runner
            .last_output
            .as_ref()
            .map(|output| output.tool_calls.len()),
        Some(1)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_accepts_resource_only_follow_up() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Model::with_completer(Arc::new(RecordingRequestCompleter {
        name: "recorder",
        requests: requests.clone(),
    }));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let req = CompletionRequest {
        prompt: "seed task".to_string(),
        ..Default::default()
    };

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx.completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert_eq!(runner.last_output.as_ref().unwrap().content, "seed task");

    assert!(
        runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Ping,
                resources: vec![Resource {
                    blob: Some(b"resource follow-up".to_vec().into()),
                    ..Default::default()
                }],
                usage: Usage::default(),
                model: None,
                effort: None,
            }])
            .await
            .unwrap()
    );

    assert_eq!(
        runner.last_output.as_ref().unwrap().content,
        "resource follow-up"
    );
    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 2);
    assert!(recorded[1].prompt.is_empty());
    assert_eq!(request_text(&recorded[1]), "resource follow-up");
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_handles_control_inputs_model_effort_and_finalize_errors() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Model::with_completer(Arc::new(RecordingRequestCompleter {
        name: "recorder",
        requests: requests.clone(),
    }));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "seed task".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert!(
        runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Command {
                    command: "steer".to_string(),
                    prompt: "correct course".to_string(),
                },
                resources: Vec::new(),
                usage: Usage {
                    input_tokens: 3,
                    output_tokens: 4,
                    cached_tokens: 1,
                    requests: 1,
                },
                model: Some("analysis".to_string()),
                effort: Some(ModelEffort::Low),
            }])
            .await
            .unwrap()
    );
    assert_eq!(
        runner.last_output.as_ref().unwrap().session.as_deref(),
        Some("session-1")
    );

    assert!(
        runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Command {
                    command: "note".to_string(),
                    prompt: "/note custom follow-up".to_string(),
                },
                resources: Vec::new(),
                usage: Usage::default(),
                model: None,
                effort: None,
            }])
            .await
            .unwrap()
    );

    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 3);
    assert_eq!(request_text(&recorded[0]), "seed task");
    assert_eq!(request_text(&recorded[1]), "correct course");
    assert_eq!(recorded[1].model.as_deref(), Some("analysis"));
    assert_eq!(recorded[1].effort, Some(ModelEffort::Low));
    assert_eq!(request_text(&recorded[2]), "/note custom follow-up");

    let error_ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(ErrorCompleter)))
        .mock_ctx();
    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-err".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));
    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: error_ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "will fail".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };
    let output = runner.finalize_output().await;
    assert_eq!(output.failed_reason.as_deref(), Some("model failed"));
    assert_eq!(output.session.as_deref(), Some("session-err"));
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_finalizes_with_latest_visible_output_after_compaction() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Model::with_completer(Arc::new(RecordingCompactionCompleter {
        requests: requests.clone(),
    }));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let req = CompletionRequest {
        instructions: "Keep working".to_string(),
        prompt: "seed task".to_string(),
        ..Default::default()
    };

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "compactor".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx.completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert!(runner.run(Vec::new()).await.unwrap());
    assert_eq!(
        runner.runner.chat_history()[0].text().as_deref(),
        Some("compacted handoff")
    );

    runner.carried_artifacts.push(resource(11, &["artifact"]));
    let output = runner.finalize_output().await;
    assert_eq!(output.content, "seed task");
    assert_eq!(output.session.as_deref(), Some("session-1"));
    // Artifacts rescued from the pre-compaction runner survive into the final output.
    assert_eq!(
        output
            .artifacts
            .iter()
            .map(|artifact| artifact._id)
            .collect::<Vec<_>>(),
        vec![11]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_keeps_latest_output_for_final_end_after_progress() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let req = CompletionRequest {
        prompt: "seed task".to_string(),
        ..Default::default()
    };

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx.completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert_eq!(
        runner
            .last_output
            .as_ref()
            .map(|output| output.content.as_str()),
        Some("seed task")
    );

    let progress_output = runner.last_output.take().unwrap();
    assert_eq!(progress_output.content, "seed task");

    let final_output = runner.finalize_output().await;
    assert_eq!(final_output.content, "seed task");
    assert_eq!(final_output.session.as_deref(), Some("session-1"));
    assert!(final_output.failed_reason.is_none());
    assert!(runner.runner.is_done());
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_output_classification_and_failure_defaults_are_covered() {
    assert!(!SubSessionRunner::has_observable_output(
        &AgentOutput::default()
    ));

    for output in [
        AgentOutput {
            thoughts: Some("thinking".to_string()),
            ..Default::default()
        },
        AgentOutput {
            failed_reason: Some("failed".to_string()),
            ..Default::default()
        },
        AgentOutput {
            tool_calls: vec![anda_core::ToolCall {
                name: "lookup".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        },
        AgentOutput {
            chat_history: vec![Message {
                role: "assistant".to_string(),
                content: vec!["history".to_string().into()],
                ..Default::default()
            }],
            ..Default::default()
        },
        AgentOutput {
            artifacts: vec![Resource {
                _id: 42,
                ..Default::default()
            }],
            ..Default::default()
        },
        AgentOutput {
            conversation: Some(7),
            ..Default::default()
        },
        AgentOutput {
            model: Some("flash".to_string()),
            ..Default::default()
        },
        AgentOutput {
            usage: Usage {
                requests: 1,
                ..Default::default()
            },
            ..Default::default()
        },
        AgentOutput {
            tools_usage: HashMap::from([(
                "lookup".to_string(),
                Usage {
                    requests: 1,
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
    ] {
        assert!(SubSessionRunner::has_observable_output(&output));
    }

    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "seed task".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert_eq!(
        runner.record_failed_output(""),
        "subagent session cancelled"
    );
    let output = runner.latest_output();
    assert_eq!(
        output.failed_reason.as_deref(),
        Some("subagent session cancelled")
    );
    assert_eq!(output.session.as_deref(), Some("session-1"));
    assert!(output.content.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_reports_cancel_failure_as_latest_output() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let req = CompletionRequest {
        prompt: "seed task".to_string(),
        ..Default::default()
    };

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: None,
        runner: ctx.completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());

    let err = runner
        .run(vec![SubAgentInput {
            command: PromptCommand::Command {
                command: "cancel".to_string(),
                prompt: "/cancel stop because requested".to_string(),
            },
            resources: Vec::new(),
            usage: Usage::default(),
            model: None,
            effort: None,
        }])
        .await
        .unwrap_err();

    assert_eq!(err.to_string(), "stop because requested");
    let final_output = runner.latest_output();
    assert_eq!(
        final_output.failed_reason.as_deref(),
        Some("stop because requested")
    );
    assert_eq!(final_output.session.as_deref(), Some("session-1"));
    assert!(final_output.content.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_stop_idles_current_task_without_ending_session() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Model::with_completer(Arc::new(StopResumeCompleter {
        requests: requests.clone(),
    }));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let hook = Arc::new(RecordingAgentHook::default());

    let req = CompletionRequest {
        prompt: "seed task".to_string(),
        ..Default::default()
    };

    let (sender, mut rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session,
        agent_hook: Some(DynAgentHook::new(hook.clone())),
        runner: ctx.clone().completion_iter(req, Vec::new()).unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    assert!(runner.run(Vec::new()).await.unwrap());
    assert!(!runner.runner.is_idle());
    assert_eq!(requests.lock().len(), 1);

    runner.session.background_tasks.write().insert(
        "child-session".to_string(),
        BackgroundTaskInfo {
            agent_name: "child".to_string(),
            ..Default::default()
        },
    );

    assert!(
        runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Command {
                    command: "stop".to_string(),
                    prompt: "/stop wrong branch".to_string(),
                },
                resources: Vec::new(),
                usage: Usage::default(),
                model: None,
                effort: None,
            }])
            .await
            .unwrap()
    );

    assert!(runner.runner.is_idle());
    assert!(!runner.runner.is_done());
    assert_eq!(requests.lock().len(), 1);
    assert!(
        runner
            .session
            .background_tasks
            .read()
            .get("child-session")
            .is_some_and(|info| info.stopped)
    );

    let progress = hook.progress_events();
    assert_eq!(progress.len(), 2);
    assert_eq!(progress[0].1.content, "working");
    assert_eq!(
        progress[1].1.content,
        "Subagent session stopped: wrong branch"
    );
    assert!(progress[1].1.failed_reason.is_none());
    assert_eq!(progress[1].1.session.as_deref(), Some("session-1"));

    AgentHook::on_background_end(
        runner.session.as_ref(),
        &ctx,
        "child-session".to_string(),
        AgentOutput {
            content: "late child result".to_string(),
            usage: Usage {
                requests: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    assert!(
        !runner
            .session
            .background_tasks
            .read()
            .contains_key("child-session")
    );
    assert!(rx.try_recv().is_err());

    assert!(
        runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Plain {
                    prompt: "next task".to_string(),
                },
                resources: Vec::new(),
                usage: Usage::default(),
                model: None,
                effort: None,
            }])
            .await
            .unwrap()
    );

    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 2);
    assert_eq!(request_text(&recorded[1]), "next task");
    assert_eq!(
        runner
            .last_output
            .as_ref()
            .map(|output| output.content.as_str()),
        Some("next task")
    );
}

#[test]
fn subagent_definition_guides_self_contained_prompts() {
    let agent = SubAgent {
        name: "research_assistant".to_string(),
        description: "Handles recurring research tasks with concise synthesis.".to_string(),
        instructions: "Research carefully and synthesize findings.".to_string(),
        tools: vec!["google_web_search".to_string()],
        model: "pro".to_string(),
        effort: Some(ModelEffort::High),
        ..Default::default()
    };

    let definition = agent.definition();

    assert_eq!(definition.name, "research_assistant");
    assert!(
        definition
            .description
            .starts_with("Handles recurring research tasks with concise synthesis.")
    );
    assert!(
        definition
            .description
            .contains("Allowed tools: google_web_search.")
    );
    assert!(definition.description.contains("Default model label: pro."));
    assert!(definition.description.contains("Default effort: high."));
    assert_eq!(
        definition.parameters["description"],
        json!(
            "Run this subagent as a focused worker process. The caller acts as the scheduler: use blocking mode for short one-shot work, or session mode for long-running, parallel, asynchronous, or follow-up work. Keep each prompt as a self-contained handoff."
        )
    );
    assert!(
        definition.parameters["properties"]["prompt"]["description"]
            .as_str()
            .unwrap()
            .contains("Self-contained task handoff")
    );
    assert_eq!(
        definition.parameters["properties"]["session"]["default"],
        json!("")
    );
    assert_eq!(
        definition.parameters["properties"]["model"]["default"],
        json!("")
    );
    assert_eq!(
        definition.parameters["properties"]["effort"]["enum"],
        json!(["minimal", "low", "medium", "high", "max", null])
    );
    assert_eq!(definition.parameters["additionalProperties"], json!(false));
}

#[test]
fn subagents_manager_definition_uses_strict_safe_output_schema() {
    let definition = SubAgentManager::new()
        .definition()
        .normalize_strict_parameters();

    assert_eq!(
        definition.parameters["properties"]["output_schema"]["type"],
        json!(["string", "null"])
    );
    assert_eq!(
        definition.parameters["properties"]["operation"]["default"],
        json!("upsert")
    );
    assert_eq!(
        definition.parameters["properties"]["model"]["default"],
        json!("")
    );
    assert_eq!(
        definition.parameters["properties"]["effort"]["type"],
        json!(["string", "null"])
    );
    assert!(
        definition.parameters["required"]
            .as_array()
            .unwrap()
            .contains(&json!("operation"))
    );
    assert_eq!(definition.parameters["additionalProperties"], json!(false));
}

#[test]
fn subagents_manager_args_accept_json_encoded_output_schema() {
    let schema = json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" }
        },
        "required": ["summary"],
        "additionalProperties": false
    });

    let args = SubAgentManagerArgs::from_prompt(
        json!({
            "name": "structured_helper",
            "description": "Creates structured output.",
            "instructions": "Return structured output.",
            "tools": [],
            "tags": [],
            "output_schema": serde_json::to_string(&schema).unwrap(),
            "model": "Pro",
            "effort": "HIGH",
            "task": "",
            "session": "",
            "persist": false
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(args.operation, "upsert");
    assert_eq!(args.output_schema, Some(schema));
    assert_eq!(args.model, "Pro");
    assert_eq!(args.effort, Some(ModelEffort::High));
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_run_allows_model_and_effort_selection() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let models = Arc::new(Models::default());
    models.set_model(Model::with_completer(Arc::new(EchoCompleter)));
    models.set(
        "analysis".to_string(),
        Model::with_completer(Arc::new(RecordingRequestCompleter {
            name: "analysis-model",
            requests: requests.clone(),
        })),
    );
    let ctx = EngineBuilder::new().with_models(models).mock_ctx();
    let agent = SubAgent {
        name: "analysis_helper".to_string(),
        description: "Runs analysis work.".to_string(),
        instructions: "Analyze carefully.".to_string(),
        ..Default::default()
    };

    let output = agent
        .run(
            ctx,
            serde_json::to_string(&SubAgentArgs {
                prompt: "inspect this".to_string(),
                session: String::new(),
                model: "analysis".to_string(),
                effort: Some(ModelEffort::High),
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(output.content, "inspect this");
    assert_eq!(output.model.as_deref(), Some("analysis-model"));
    let recorded = requests.lock().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].model.as_deref(), Some("analysis"));
    assert_eq!(recorded[0].effort, Some(ModelEffort::High));
}

#[test]
fn subagents_manager_upsert_preserves_active_sessions() {
    let manager = SubAgentManager::new();
    let agent = SubAgent {
        name: "worker".to_string(),
        description: "Handles delegated work.".to_string(),
        instructions: "Work carefully.".to_string(),
        ..Default::default()
    };
    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    agent.subsessions.insert_session(Arc::new(SubSession::new(
        "job-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    )));

    manager.upsert_temporary(agent).unwrap();
    manager
        .upsert_temporary(SubAgent {
            name: "worker".to_string(),
            description: "Updated routing description.".to_string(),
            instructions: "Use the updated workflow.".to_string(),
            ..Default::default()
        })
        .unwrap();

    let loaded = manager.get_lowercase("worker").unwrap();
    assert_eq!(loaded.description, "Updated routing description.");
    assert_eq!(loaded.subsessions.active_session_ids(), vec!["job-1"]);
    assert!(loaded.subsessions.get_session("job-1").is_some());
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_run_accepts_plain_prompts_and_structured_session_args() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let agent = SubAgent {
        name: "echo_helper".to_string(),
        description: "Echoes input.".to_string(),
        instructions: "Echo the prompt.".to_string(),
        ..Default::default()
    };

    let output = agent
        .run(ctx.clone(), "plain task".to_string(), Vec::new())
        .await
        .unwrap();
    assert_eq!(output.content, "plain task");

    let output = agent
        .run(
            ctx,
            serde_json::to_string(&SubAgentArgs {
                prompt: "session task".to_string(),
                session: "ThreadA".to_string(),
                model: String::new(),
                effort: None,
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(output.session.as_deref(), Some("threada"));
    assert!(output.content.contains("session mode"));
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_session_does_not_emit_start_hook_when_ack_hook_fails() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let hook = Arc::new(FailingAfterAgentHook::default());
    ctx.base.set_state(DynAgentHook::new(hook.clone()));

    let agent = SubAgent {
        name: "echo_helper".to_string(),
        description: "Echoes input.".to_string(),
        instructions: "Echo the prompt.".to_string(),
        ..Default::default()
    };

    let err = agent
        .run(
            ctx,
            serde_json::to_string(&SubAgentArgs {
                prompt: "session task".to_string(),
                session: "ThreadA".to_string(),
                model: String::new(),
                effort: None,
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap_err();

    assert_eq!(err.to_string(), "after hook rejected output");
    assert!(hook.starts.lock().is_empty());
    assert!(agent.subsessions.active_session_ids().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_background_hooks_forward_outputs_and_manage_registry() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let session = Arc::new(SubSession::new("session-1".into(), "worker".into(), tx, 0));
    let ctx = EngineBuilder::new().mock_ctx();

    let sessions = SubSessions::default();
    sessions.insert_session(session.clone());
    assert_eq!(sessions.active_session_ids(), vec!["session-1".to_string()]);
    assert!(sessions.get_session("session-1").is_some());

    AgentHook::on_background_start(
        session.as_ref(),
        &ctx,
        BackgroundHandle::new("child-session", anda_core::CancellationToken::new()),
        &CompletionRequest::default(),
    )
    .await;
    assert_eq!(
        session
            .background_tasks
            .read()
            .get("child-session")
            .unwrap()
            .agent_name,
        "Mocker"
    );

    AgentHook::on_background_progress(
        session.as_ref(),
        &ctx,
        "child-session".into(),
        AgentOutput {
            content: "partial".into(),
            usage: Usage {
                requests: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    assert!(
        recv_subagent_prompt(&mut rx)
            .await
            .contains("intermediate output")
    );

    AgentHook::on_background_progress(
        session.as_ref(),
        &ctx,
        "child-session".into(),
        AgentOutput {
            failed_reason: Some("bad".into()),
            ..Default::default()
        },
    )
    .await;
    assert!(
        recv_subagent_prompt(&mut rx)
            .await
            .contains("failed with reason")
    );

    AgentHook::on_background_progress(
        session.as_ref(),
        &ctx,
        "child-session".into(),
        AgentOutput::default(),
    )
    .await;
    assert!(recv_subagent_prompt(&mut rx).await.contains("completed"));

    AgentHook::on_background_end(
        session.as_ref(),
        &ctx,
        "child-session".into(),
        AgentOutput {
            content: "final".into(),
            ..Default::default()
        },
    )
    .await;
    assert!(
        !session
            .background_tasks
            .read()
            .contains_key("child-session")
    );
    assert!(recv_subagent_prompt(&mut rx).await.contains("final output"));

    ToolBackgroundHook::on_background_start(
        session.as_ref(),
        &ctx.base,
        BackgroundHandle::new("fetch:task-1", anda_core::CancellationToken::new()),
        json!({"url": "https://example.test"}),
    )
    .await;
    assert_eq!(
        session
            .background_tasks
            .read()
            .get("fetch:task-1")
            .unwrap()
            .tool_name
            .as_deref(),
        Some("fetch")
    );

    ToolBackgroundHook::on_background_progress(
        session.as_ref(),
        &ctx.base,
        "fetch:task-1".into(),
        ToolOutput::new(json!({"status": "half"})),
    )
    .await;
    assert_eq!(
        session
            .background_tasks
            .read()
            .get("fetch:task-1")
            .unwrap()
            .progress_message
            .as_deref(),
        Some(r#"{"status":"half"}"#)
    );

    ToolBackgroundHook::on_background_start(
        session.as_ref(),
        &ctx.base,
        BackgroundHandle::new("unprefixed", anda_core::CancellationToken::new()),
        Json::Null,
    )
    .await;
    assert!(
        session
            .background_tasks
            .read()
            .get("unprefixed")
            .unwrap()
            .tool_name
            .is_none()
    );

    ToolBackgroundHook::on_background_end(
        session.as_ref(),
        &ctx.base,
        "fetch:task-1".into(),
        ToolOutput {
            output: json!({"done": true}),
            artifacts: vec![resource(7, &["artifact"])],
            usage: Usage {
                requests: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    assert!(!session.background_tasks.read().contains_key("fetch:task-1"));
    let forwarded = rx.recv().await.unwrap();
    assert!(matches!(forwarded.command, PromptCommand::Plain { .. }));
    assert_eq!(forwarded.resources.len(), 1);
    assert_eq!(forwarded.usage.requests, 1);

    sessions.remove_session("session-1");
    assert!(sessions.get_session("session-1").is_none());

    let (closed_tx, closed_rx) = tokio::sync::mpsc::channel(1);
    let closed = Arc::new(SubSession::new(
        "closed".into(),
        "worker".into(),
        closed_tx,
        0,
    ));
    drop(closed_rx);
    sessions.insert_session(closed);
    assert!(sessions.active_session_ids().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn subagents_manager_agent_entrypoint_uses_structured_args() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let manager: Arc<SubAgentManager> = ctx.subagents.get().unwrap();

    let output = Agent::<AgentCtx>::run(
        manager.as_ref(),
        ctx,
        serde_json::to_string(&SubAgentManagerArgs {
            name: "agent_helper".to_string(),
            description: "Created through the agent entrypoint.".to_string(),
            instructions: "Echo agent entrypoint tasks.".to_string(),
            task: "agent task".to_string(),
            ..Default::default()
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();
    let content: Json = serde_json::from_str(&output.content).unwrap();

    assert_eq!(content["subagent"]["result"], json!("created"));
    assert_eq!(content["output"], json!("agent task"));
    assert!(manager.get_lowercase("agent_helper").is_some());
}

#[tokio::test(flavor = "current_thread")]
async fn subagents_manager_lists_registry_and_active_sessions() {
    let ctx = EngineBuilder::new().mock_ctx();
    let manager: Arc<SubAgentManager> = ctx.subagents.get().unwrap();
    let agent = SubAgent {
        name: "planner".to_string(),
        description: "Plans delegated work.".to_string(),
        instructions: "Break work into concrete steps.".to_string(),
        tools: vec!["tools_select".to_string()],
        tags: vec!["planning".to_string()],
        model: "flash".to_string(),
        effort: Some(ModelEffort::Low),
        ..Default::default()
    };
    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    agent.subsessions.insert_session(Arc::new(SubSession::new(
        "plan-1".to_string(),
        "planner".to_string(),
        sender,
        0,
    )));
    manager.upsert_temporary(agent).unwrap();

    let output = Agent::<AgentCtx>::run(
        manager.as_ref(),
        ctx,
        serde_json::to_string(&SubAgentManagerArgs {
            operation: "list".to_string(),
            ..Default::default()
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();
    let content: Json = serde_json::from_str(&output.content).unwrap();

    assert_eq!(content["result"], json!("listed"));
    assert_eq!(content["count"], json!(1));
    assert_eq!(content["subagents"][0]["name"], json!("planner"));
    assert_eq!(content["subagents"][0]["callable"], json!("SA_planner"));
    assert_eq!(content["subagents"][0]["model"], json!("flash"));
    assert_eq!(content["subagents"][0]["effort"], json!("low"));
    assert_eq!(
        content["subagents"][0]["active_sessions"],
        json!(["plan-1"])
    );
}

#[tokio::test(flavor = "current_thread")]
async fn load_restores_all_persisted_subagents() {
    let ctx = EngineBuilder::new().mock_ctx();
    let tool: Arc<SubAgentManager> = ctx.subagents.get().unwrap();

    let agents = vec![
        SubAgent {
            name: "Research_Assistant".to_string(),
            description: "Handles recurring research tasks with concise synthesis.".to_string(),
            instructions: "Research carefully and synthesize findings.".to_string(),
            tools: vec!["google_web_search".to_string()],
            tags: vec!["research".to_string()],
            ..Default::default()
        },
        SubAgent {
            name: "code_reviewer".to_string(),
            description: "Reviews code for correctness and risks.".to_string(),
            instructions: "Review code changes and summarize findings.".to_string(),
            tools: vec!["read_file".to_string(), "grep_search".to_string()],
            tags: vec!["code".to_string(), "review".to_string()],
            model: "pro".to_string(),
            effort: Some(ModelEffort::Medium),
            idle_timeout: 300,
            ..Default::default()
        },
        SubAgent {
            name: "writer_helper".to_string(),
            description: "Drafts concise written content.".to_string(),
            instructions: "Write clearly and keep the response concise.".to_string(),
            tags: vec!["writing".to_string()],
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string" }
                },
                "required": ["summary"],
                "additionalProperties": false
            })),
            ..Default::default()
        },
    ];

    for agent in agents.clone() {
        tool.upsert(ctx.clone(), agent).await.unwrap();
    }

    let legacy_agent = SubAgent {
        name: "legacy_agent".to_string(),
        description: "Stored outside the subagents directory.".to_string(),
        instructions: "This should not be loaded by SubAgentManager::load.".to_string(),
        ..Default::default()
    };
    ctx.store_put(
        &Path::from("legacy_agent"),
        PutMode::Overwrite,
        to_canonical_vec(&legacy_agent).unwrap().into(),
    )
    .await
    .unwrap();

    let stored = ctx
        .store_list(Some(&SubAgentManager::store_prefix()), &Path::from(""))
        .await
        .unwrap();
    assert_eq!(stored.len(), agents.len());
    assert!(stored.iter().all(|meta| {
        meta.location
            .prefix_match(&SubAgentManager::store_prefix())
            .is_some()
    }));

    for meta in &stored {
        let (data, _) = ctx.store_get(&meta.location).await.unwrap();
        let loaded = from_slice::<SubAgent>(&data[..]).unwrap();
        assert!(agents.iter().any(|agent| agent.name == loaded.name));
    }

    let reloaded = SubAgentManager::new();
    reloaded.load(ctx).await.unwrap();

    assert_eq!(reloaded.definitions(None).len(), agents.len());
    assert!(reloaded.get_lowercase("legacy_agent").is_none());

    for expected in agents {
        let loaded = reloaded
            .get_lowercase(&expected.name.to_ascii_lowercase())
            .unwrap();

        assert_eq!(loaded.name, expected.name);
        assert_eq!(loaded.description, expected.description);
        assert_eq!(loaded.instructions, expected.instructions);
        assert_eq!(loaded.tools, expected.tools);
        assert_eq!(loaded.tags, expected.tags);
        assert_eq!(loaded.output_schema, expected.output_schema);
        assert_eq!(loaded.model, expected.model);
        assert_eq!(loaded.effort, expected.effort);
        assert_eq!(loaded.idle_timeout, expected.idle_timeout);
    }
}

fn test_session(id: &str) -> (Arc<SubSession>, tokio::sync::mpsc::Receiver<SubAgentInput>) {
    let (sender, rx) = tokio::sync::mpsc::channel(16);
    (
        Arc::new(SubSession::new(
            id.to_string(),
            "worker".to_string(),
            sender,
            0,
        )),
        rx,
    )
}

#[tokio::test(flavor = "current_thread")]
async fn stop_background_task_marks_stopped_and_cancels_handle() {
    let (session, _rx) = test_session("parent");
    let ctx = EngineBuilder::new().mock_ctx();

    let token = anda_core::CancellationToken::new();
    ToolBackgroundHook::on_background_start(
        session.as_ref(),
        &ctx.base,
        BackgroundHandle::new("fetch:task-1", token.clone()),
        Json::Null,
    )
    .await;
    assert!(!token.is_cancelled());

    // Stopping a live task reports it was found, marks it stopped, and cancels its token so the
    // producer actually terminates the work (not just suppresses forwarding).
    assert!(session.stop_background_task("fetch:task-1"));
    assert!(token.is_cancelled());
    assert!(
        session
            .background_tasks
            .read()
            .get("fetch:task-1")
            .unwrap()
            .stopped
    );

    // The handle is consumed once; a second stop finds no live task.
    assert!(!session.stop_background_task("fetch:task-1"));

    // stop_background_tasks() cancels every remaining registered handle.
    let token2 = anda_core::CancellationToken::new();
    ToolBackgroundHook::on_background_start(
        session.as_ref(),
        &ctx.base,
        BackgroundHandle::new("fetch:task-2", token2.clone()),
        Json::Null,
    )
    .await;
    session.stop_background_tasks();
    assert!(token2.is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn stop_task_command_stops_single_background_task_by_id() {
    let ctx = EngineBuilder::new().mock_ctx();
    let subsessions = Arc::new(SubSessions::default());
    // Keep `_rx` alive so the session's sender stays open and `get_session` treats it as active.
    let (session, _rx) = test_session("job");
    subsessions.insert_session(session.clone());

    // Register two background tasks in the session with known tokens.
    let token1 = anda_core::CancellationToken::new();
    let token2 = anda_core::CancellationToken::new();
    ToolBackgroundHook::on_background_start(
        session.as_ref(),
        &ctx.base,
        BackgroundHandle::new("fetch:task-1", token1.clone()),
        Json::Null,
    )
    .await;
    ToolBackgroundHook::on_background_start(
        session.as_ref(),
        &ctx.base,
        BackgroundHandle::new("fetch:task-2", token2.clone()),
        Json::Null,
    )
    .await;

    let agent = SubAgent {
        name: "worker".to_string(),
        subsessions: subsessions.clone(),
        ..Default::default()
    };

    let stop_task = |prompt: &str| {
        serde_json::to_string(&SubAgentArgs {
            prompt: prompt.to_string(),
            session: "job".to_string(),
            model: String::new(),
            effort: None,
        })
        .unwrap()
    };

    // Stopping one task by id cancels only its token and leaves the sibling running.
    let rt = Agent::<AgentCtx>::run(
        &agent,
        ctx.clone(),
        stop_task("/stop_task fetch:task-1"),
        vec![],
    )
    .await
    .unwrap();
    assert!(rt.content.contains("Stopped background task fetch:task-1"));
    assert!(token1.is_cancelled());
    assert!(!token2.is_cancelled());
    assert!(
        session
            .background_tasks
            .read()
            .get("fetch:task-1")
            .unwrap()
            .stopped
    );

    // Unknown task id reports not-found without disturbing the session.
    let rt = Agent::<AgentCtx>::run(
        &agent,
        ctx.clone(),
        stop_task("/stop_task fetch:missing"),
        vec![],
    )
    .await
    .unwrap();
    assert!(
        rt.content
            .contains("No active background task fetch:missing")
    );

    // A missing argument is rejected with guidance.
    let rt = Agent::<AgentCtx>::run(&agent, ctx.clone(), stop_task("/stop_task"), vec![])
        .await
        .unwrap();
    assert!(rt.content.contains("requires a task_id argument"));
    // The sibling task is still live.
    assert!(!token2.is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn stop_task_command_reports_inactive_session() {
    let ctx = EngineBuilder::new().mock_ctx();
    let agent = SubAgent {
        name: "worker".to_string(),
        ..Default::default()
    };

    let rt = Agent::<AgentCtx>::run(
        &agent,
        ctx,
        serde_json::to_string(&SubAgentArgs {
            prompt: "/stop_task fetch:task-1".to_string(),
            session: "ghost".to_string(),
            model: String::new(),
            effort: None,
        })
        .unwrap(),
        vec![],
    )
    .await
    .unwrap();
    assert!(rt.content.contains("is not active"));
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_agent_hook_forwards_usage_deltas() {
    let (session, mut rx) = test_session("parent");
    let ctx = EngineBuilder::new().mock_ctx();

    AgentHook::on_background_start(
        session.as_ref(),
        &ctx,
        BackgroundHandle::new("child", anda_core::CancellationToken::new()),
        &CompletionRequest::default(),
    )
    .await;

    // Background agent outputs carry cumulative usage; the session must receive deltas.
    AgentHook::on_background_progress(
        session.as_ref(),
        &ctx,
        "child".into(),
        AgentOutput {
            content: "step-1".into(),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 10,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        },
    )
    .await;
    let input = rx.recv().await.unwrap();
    assert_eq!(input.usage.input_tokens, 100);
    assert_eq!(input.usage.output_tokens, 10);
    assert_eq!(input.usage.requests, 1);

    AgentHook::on_background_progress(
        session.as_ref(),
        &ctx,
        "child".into(),
        AgentOutput {
            content: "step-2".into(),
            usage: Usage {
                input_tokens: 250,
                output_tokens: 25,
                cached_tokens: 0,
                requests: 3,
            },
            ..Default::default()
        },
    )
    .await;
    let input = rx.recv().await.unwrap();
    assert_eq!(input.usage.input_tokens, 150);
    assert_eq!(input.usage.output_tokens, 15);
    assert_eq!(input.usage.requests, 2);

    // A failure output with empty usage must not reset the watermark.
    AgentHook::on_background_progress(
        session.as_ref(),
        &ctx,
        "child".into(),
        AgentOutput {
            failed_reason: Some("transient".into()),
            ..Default::default()
        },
    )
    .await;
    let input = rx.recv().await.unwrap();
    assert_eq!(input.usage.input_tokens, 0);
    assert_eq!(input.usage.requests, 0);

    AgentHook::on_background_end(
        session.as_ref(),
        &ctx,
        "child".into(),
        AgentOutput {
            content: "done".into(),
            usage: Usage {
                input_tokens: 300,
                output_tokens: 30,
                cached_tokens: 0,
                requests: 4,
            },
            ..Default::default()
        },
    )
    .await;
    let input = rx.recv().await.unwrap();
    assert_eq!(input.usage.input_tokens, 50);
    assert_eq!(input.usage.output_tokens, 5);
    assert_eq!(input.usage.requests, 1);
    assert!(session.background_tasks.read().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn subsessions_claim_and_conditional_remove_handle_races() {
    let sessions = SubSessions::default();
    let (first, _rx1) = test_session("job");
    assert!(sessions.try_insert_session(first.clone()).is_none());

    // A concurrent claim joins the active session instead of replacing it.
    let (second, _rx2) = test_session("job");
    let existing = sessions.try_insert_session(second.clone()).unwrap();
    assert!(Arc::ptr_eq(&existing, &first));

    // A stale runner must not remove a session it no longer owns.
    sessions.remove_session_if(&second);
    assert!(sessions.get_session("job").is_some());
    sessions.remove_session_if(&first);
    assert!(sessions.get_session("job").is_none());

    // A closed session is replaced by a fresh claim.
    let (closed, closed_rx) = test_session("job");
    drop(closed_rx);
    sessions.insert_session(closed);
    let (fresh, _rx3) = test_session("job");
    assert!(sessions.try_insert_session(fresh.clone()).is_none());
    assert!(Arc::ptr_eq(&sessions.get_session("job").unwrap(), &fresh));
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_control_commands_report_inactive_sessions() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let agent = SubAgent {
        name: "echo_helper".to_string(),
        description: "Echoes input.".to_string(),
        instructions: "Echo the prompt.".to_string(),
        ..Default::default()
    };

    for (prompt, expected_op) in [
        ("/stop finish now", "nothing to stop"),
        ("/cancel", "nothing to cancel"),
        ("", "nothing to ping"),
        ("/ping", "nothing to ping"),
    ] {
        let output = agent
            .run(
                ctx.clone(),
                serde_json::to_string(&SubAgentArgs {
                    prompt: prompt.to_string(),
                    session: "Ghost".to_string(),
                    model: String::new(),
                    effort: None,
                })
                .unwrap(),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.session.as_deref(), Some("ghost"));
        assert!(output.content.contains("not active"), "{}", output.content);
        assert!(output.content.contains(expected_op), "{}", output.content);
        assert!(output.failed_reason.is_none());
        assert!(agent.subsessions.active_session_ids().is_empty());
    }

    // A blocking run still rejects an empty prompt explicitly.
    let err = agent
        .run(ctx, "".to_string(), Vec::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("prompt cannot be empty"));
}

#[tokio::test(flavor = "current_thread")]
async fn subsession_runner_sync_status_snapshots_progress_and_usage() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();

    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "session-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));

    let mut runner = SubSessionRunner {
        session: session.clone(),
        agent_hook: None,
        runner: ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "seed task".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound(),
        conversation: None,
        last_output: None,
        carried_artifacts: Vec::new(),
        closing: false,
    };

    // Before any step the snapshot is empty.
    let detail = session.detail();
    assert_eq!(detail["turns"], json!(0));
    assert_eq!(detail["usage"]["requests"], json!(0));
    assert_eq!(detail["last_progress"], Json::Null);

    assert!(runner.run(Vec::new()).await.unwrap());
    runner.sync_status();

    let detail = session.detail();
    assert_eq!(detail["session"], json!("session-1"));
    assert_eq!(detail["agent"], json!("worker"));
    assert_eq!(detail["active"], json!(true));
    assert_eq!(detail["turns"], json!(1));
    assert_eq!(detail["usage"]["requests"], json!(1));
    assert_eq!(detail["last_progress"], json!("seed task"));
    // EchoCompleter returns no tool calls, so the completed turn leaves the runner idle.
    assert_eq!(detail["busy"], json!(false));
    assert!(detail["running_ms"].is_u64());
    assert!(detail["idle_ms"].is_u64());
}

#[test]
fn progress_text_prefers_content_then_failure_and_truncates() {
    assert_eq!(progress_text(&AgentOutput::default()), None);
    assert_eq!(
        progress_text(&AgentOutput {
            content: "  done  ".to_string(),
            ..Default::default()
        }),
        Some("done".to_string())
    );
    assert_eq!(
        progress_text(&AgentOutput {
            failed_reason: Some("boom".to_string()),
            ..Default::default()
        }),
        Some("failed: boom".to_string())
    );
    assert_eq!(
        progress_text(&AgentOutput {
            thoughts: Some("thinking".to_string()),
            ..Default::default()
        }),
        Some("thinking".to_string())
    );

    // ASCII over the byte budget: truncated to exactly the budget, plus the ellipsis marker.
    let long = "a".repeat(STATUS_PROGRESS_MAX_BYTES + 50);
    let truncated = progress_text(&AgentOutput {
        content: long,
        ..Default::default()
    })
    .unwrap();
    assert!(truncated.ends_with('…'));
    let body = truncated.strip_suffix('…').unwrap();
    assert_eq!(body.len(), STATUS_PROGRESS_MAX_BYTES);

    // A multi-code-point grapheme cluster (family emoji joined by ZWJ) must never be split
    // mid-character: the kept body stays within the byte budget and holds only whole clusters.
    let family = "👨‍👩‍👧‍👦"; // 25 bytes, a single grapheme cluster
    let text = family.repeat(200); // 5000 bytes, well over the budget
    let truncated = progress_text(&AgentOutput {
        content: text,
        ..Default::default()
    })
    .unwrap();
    let body = truncated.strip_suffix('…').unwrap();
    assert!(body.len() <= STATUS_PROGRESS_MAX_BYTES);
    assert_eq!(
        body,
        family.repeat(STATUS_PROGRESS_MAX_BYTES / family.len())
    );
    // A single oversized cluster is shorter than the limit, so it survives intact.
    assert_eq!(
        progress_text(&AgentOutput {
            content: family.to_string(),
            ..Default::default()
        }),
        Some(family.to_string())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn subagent_status_command_reports_live_session_snapshot() {
    let model = Model::with_completer(Arc::new(EchoCompleter));
    let ctx = EngineBuilder::new().with_model(model).mock_ctx();
    let agent = SubAgent {
        name: "status_worker".to_string(),
        description: "Reports status.".to_string(),
        instructions: "Work.".to_string(),
        ..Default::default()
    };

    // An inactive session reports active=false without starting a runner.
    let output = agent
        .run(
            ctx.clone(),
            serde_json::to_string(&SubAgentArgs {
                prompt: "/status".to_string(),
                session: "Job-1".to_string(),
                model: String::new(),
                effort: None,
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(output.session.as_deref(), Some("job-1"));
    let detail: Json = serde_json::from_str(&output.content).unwrap();
    assert_eq!(detail["active"], json!(false));
    assert_eq!(detail["session"], json!("job-1"));
    assert!(agent.subsessions.active_session_ids().is_empty());

    // An active session returns its live snapshot synchronously, without enqueuing into the
    // runner.
    let (sender, mut rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "job-1".to_string(),
        "status_worker".to_string(),
        sender,
        0,
    ));
    session.background_tasks.write().insert(
        "fetch:task-1".to_string(),
        BackgroundTaskInfo {
            agent_name: "status_worker".to_string(),
            tool_name: Some("fetch".to_string()),
            progress_message: Some("halfway".to_string()),
            ..Default::default()
        },
    );
    // A stopped task must be excluded from the report.
    session.background_tasks.write().insert(
        "fetch:task-2".to_string(),
        BackgroundTaskInfo {
            agent_name: "status_worker".to_string(),
            stopped: true,
            ..Default::default()
        },
    );
    session.record_status(SubSessionStatus {
        usage: Usage {
            input_tokens: 120,
            output_tokens: 30,
            cached_tokens: 0,
            requests: 2,
        },
        turns: 2,
        model: Some("flash".to_string()),
        busy: true,
        last_progress: Some("still working".to_string()),
        ..Default::default()
    });
    agent.subsessions.insert_session(session);

    let output = agent
        .run(
            ctx,
            serde_json::to_string(&SubAgentArgs {
                // Case-insensitive session ID and trailing argument are tolerated.
                prompt: "/status please".to_string(),
                session: "JOB-1".to_string(),
                model: String::new(),
                effort: None,
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(output.session.as_deref(), Some("job-1"));
    let detail: Json = serde_json::from_str(&output.content).unwrap();
    assert_eq!(detail["active"], json!(true));
    assert_eq!(detail["busy"], json!(true));
    assert_eq!(detail["turns"], json!(2));
    assert_eq!(detail["model"], json!("flash"));
    assert_eq!(detail["usage"]["input_tokens"], json!(120));
    assert_eq!(detail["usage"]["requests"], json!(2));
    assert_eq!(detail["last_progress"], json!("still working"));
    let tasks = detail["background_tasks"].as_array().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0]["task_id"], json!("fetch:task-1"));
    assert_eq!(tasks[0]["tool"], json!("fetch"));
    assert_eq!(tasks[0]["progress"], json!("halfway"));
    assert!(detail["running_ms"].is_u64());
    assert!(detail["idle_ms"].is_u64());

    // The poll is read-only: nothing was enqueued into the session runner.
    assert!(rx.try_recv().is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn subagents_manager_status_reports_session_details_and_filters_by_name() {
    let ctx = EngineBuilder::new().mock_ctx();
    let manager: Arc<SubAgentManager> = ctx.subagents.get().unwrap();

    let worker = SubAgent {
        name: "worker".to_string(),
        description: "Works.".to_string(),
        instructions: "Work.".to_string(),
        ..Default::default()
    };
    let (sender, _rx) = tokio::sync::mpsc::channel(4);
    let session = Arc::new(SubSession::new(
        "job-1".to_string(),
        "worker".to_string(),
        sender,
        0,
    ));
    session.record_status(SubSessionStatus {
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
            cached_tokens: 0,
            requests: 1,
        },
        turns: 1,
        busy: true,
        last_progress: Some("in progress".to_string()),
        ..Default::default()
    });
    worker.subsessions.insert_session(session);
    manager.upsert_temporary(worker).unwrap();

    manager
        .upsert_temporary(SubAgent {
            name: "idle_worker".to_string(),
            description: "Idle.".to_string(),
            instructions: "Idle.".to_string(),
            ..Default::default()
        })
        .unwrap();

    // No filter: both subagents are reported.
    let output = Agent::<AgentCtx>::run(
        manager.as_ref(),
        ctx.clone(),
        serde_json::to_string(&SubAgentManagerArgs {
            operation: "status".to_string(),
            ..Default::default()
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();
    let content: Json = serde_json::from_str(&output.content).unwrap();
    assert_eq!(content["count"], json!(2));

    // A name filter narrows the report to one worker and includes its live session detail.
    let output = Agent::<AgentCtx>::run(
        manager.as_ref(),
        ctx,
        serde_json::to_string(&SubAgentManagerArgs {
            operation: "status".to_string(),
            name: "WORKER".to_string(),
            ..Default::default()
        })
        .unwrap(),
        Vec::new(),
    )
    .await
    .unwrap();
    let content: Json = serde_json::from_str(&output.content).unwrap();
    assert_eq!(content["count"], json!(1));
    assert_eq!(content["subagents"][0]["name"], json!("worker"));
    assert_eq!(content["subagents"][0]["active_sessions"], json!(["job-1"]));
    let session = &content["subagents"][0]["sessions"][0];
    assert_eq!(session["session"], json!("job-1"));
    assert_eq!(session["busy"], json!(true));
    assert_eq!(session["turns"], json!(1));
    assert_eq!(session["usage"]["input_tokens"], json!(10));
    assert_eq!(session["last_progress"], json!("in progress"));
}
