use anda_core::{
    Agent, AgentContext, AgentInput, AgentOutput, BoxError, BoxPinFut, CacheStoreFeatures,
    CompletionRequest, Function, FunctionDefinition, Json, Path, PutMode, StateFeatures, ToolInput,
};
use anda_engine::{
    context::{
        AgentCtx, AgentInfo, DYNAMIC_REMOTE_ENGINES, DiscoveredTools, EngineCard, RemoteEngines,
    },
    engine::Engine,
    hook::{Hook, Hooks, SingleThreadHook},
    model::{CompletionFeaturesDyn, Model, testing::ScriptedCompleter},
    store::{InMemory, Store},
    subagent::SubAgent,
};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn definition(name: &str) -> FunctionDefinition {
    FunctionDefinition {
        name: name.into(),
        parameters: json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}),
        ..Default::default()
    }
}

#[tokio::test]
async fn store_list_stays_in_namespace() {
    let store = Store::new(Arc::new(InMemory::new()));
    for ns in ["a", "z"] {
        store
            .store_put(
                &Path::from(ns),
                &Path::from("file"),
                PutMode::Overwrite,
                "x".into(),
            )
            .await
            .unwrap();
    }
    let locations: Vec<_> = store
        .store_list(&Path::from("a"), None, &Path::default())
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.location.to_string())
        .collect();
    assert_eq!(locations, ["a/file"]);
}

#[test]
fn discovery_upgrades_directory_entries_to_full_schemas() {
    let mut discovered = DiscoveredTools::default();
    discovered.set_merge_policy(Some(true));
    let mut thin = definition("echo");
    thin.parameters = json!({});
    discovered.observe_output("tools_search", &json!({"tools":[thin]}));
    discovered.observe_output("tools_select", &json!({"tools":[definition("echo")]}));
    let mut req = CompletionRequest::default();
    discovered.merge_into_request(&mut req);
    assert_eq!(req.tools[0].parameters, definition("echo").parameters);
}

#[test]
fn discovery_retains_schemas_beyond_the_request_merge_budget() {
    let mut discovered = DiscoveredTools::default();
    discovered.set_merge_policy(Some(true));
    let defs: Vec<_> = (0..17).map(|n| definition(&format!("tool_{n}"))).collect();
    let mut output = json!({"tools":defs});
    discovered.observe_output("tools_select", &output);
    discovered.compact_output_for_context("tools_select", &mut output);
    let mut req = CompletionRequest::default();
    discovered.merge_into_request(&mut req);
    assert_eq!(req.tools.len(), 16);
    assert_eq!(output["tools"].as_array().unwrap().len(), 17);
    assert!(output["tools"][16].get("parameters").is_some());
    assert!(discovered.contains("tool_16"));
}

#[tokio::test]
async fn dynamic_remote_routes_reach_the_transport() {
    let ctx = Engine::builder().mock_ctx();
    let remotes = RemoteEngines {
        engines: BTreeMap::from([(
            "remote".into(),
            EngineCard {
                id: candid::Principal::self_authenticating([3]),
                info: AgentInfo {
                    endpoint: "https://example.invalid/engine".into(),
                    ..Default::default()
                },
                agents: vec![Function {
                    definition: definition("work"),
                    ..Default::default()
                }],
                tools: vec![Function {
                    definition: definition("echo"),
                    ..Default::default()
                }],
            },
        )]),
    };
    ctx.base
        .cache_store_set(DYNAMIC_REMOTE_ENGINES, remotes, None)
        .await
        .unwrap();
    assert_eq!(
        ctx.remote_tool_definitions(None, None).await.unwrap()[0].name,
        "RT_remote_echo"
    );
    let err = ctx
        .tool_call(ToolInput {
            name: "RT_remote_echo".into(),
            args: json!({}),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not implemented"));
    let err = ctx
        .agent_run(AgentInput {
            name: "RA_remote_work".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not implemented"));
}

#[tokio::test]
async fn compaction_failure_keeps_the_runner_usable() {
    let completer = ScriptedCompleter::new("test")
        .push_output(AgentOutput {
            failed_reason: Some("max_tokens".into()),
            ..Default::default()
        })
        .into_arc();
    let ctx = Engine::builder()
        .with_model(Model::with_completer(completer))
        .mock_ctx();
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                prompt: "original task".into(),
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    assert!(runner.handoff(None).await.is_err());
    assert!(!runner.is_done());
    assert_eq!(runner.req().prompt, "original task");
    runner.follow_up("retry".to_string());
    assert_eq!(runner.follow_up_message_iter().count(), 1);
    assert!(runner.next().await.unwrap().is_some());
}

struct GatedCompleter {
    started: Arc<tokio::sync::Notify>,
    ready: Arc<tokio::sync::Notify>,
}
impl CompletionFeaturesDyn for GatedCompleter {
    fn model_name(&self) -> String {
        "gated".into()
    }
    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let ready = self.ready.clone();
        let started = self.started.clone();
        Box::pin(async move {
            started.notify_one();
            ready.notified().await;
            Ok(AgentOutput {
                content: "done".into(),
                ..Default::default()
            })
        })
    }
}
#[tokio::test]
async fn stream_delivers_followup_queued_during_final_step() {
    use futures::StreamExt;
    let ready = Arc::new(tokio::sync::Notify::new());
    let ctx = Engine::builder()
        .with_model(Model::with_completer(Arc::new(GatedCompleter {
            started: Arc::new(tokio::sync::Notify::new()),
            ready: ready.clone(),
        })))
        .mock_ctx();
    let mut stream = ctx.completion_stream(
        CompletionRequest {
            prompt: "first".into(),
            ..Default::default()
        },
        vec![],
    );
    assert!(futures::poll!(stream.next()).is_pending());
    stream.follow_up("second".to_string());
    ready.notify_one();
    assert_eq!(stream.next().await.unwrap().unwrap().content, "done");
    assert!(!stream.runner.is_done());
    ready.notify_one();
    assert_eq!(stream.next().await.unwrap().unwrap().content, "done");
    assert_eq!(stream.runner.turns(), 2);
    assert!(stream.next().await.is_none());
    assert_eq!(stream.runner.follow_up_message_iter().count(), 0);
}

#[test]
fn anthropic_server_tools_are_not_dispatched_locally() {
    let res: anda_engine::model::anthropic::types::CreateMessageResponse = serde_json::from_value(json!({"id":"msg","type":"message","role":"assistant","model":"test","stop_reason":"end_turn","content":[{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"hello"}},{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]},{"type":"text","text":"finished"}],"usage":{"input_tokens":1,"output_tokens":1}})).unwrap();
    let out = res.try_into(vec![], vec![]).unwrap();
    assert!(out.failed_reason.is_none());
    assert!(out.tool_calls.is_empty());
    assert_eq!(
        out.chat_history[0].content[0],
        anda_core::ContentPart::Any(out.raw_history[0]["content"][0].clone())
    );
}

#[tokio::test]
async fn subagent_session_status_is_caller_scoped() {
    let ctx = Engine::builder()
        .with_model(Model::with_completer(
            ScriptedCompleter::new("echo").into_arc(),
        ))
        .mock_ctx();
    let agent = SubAgent {
        name: "worker".into(),
        ..Default::default()
    };
    agent
        .run(
            ctx.with_caller(candid::Principal::self_authenticating([1])),
            json!({"prompt":"caller A private text", "session":"shared"}).to_string(),
            vec![],
        )
        .await
        .unwrap();
    let mut seen = Json::Null;
    for _ in 0..20 {
        tokio::task::yield_now().await;
        let out = agent
            .run(
                ctx.with_caller(candid::Principal::self_authenticating([2])),
                json!({"prompt":"/status", "session":"shared"}).to_string(),
                vec![],
            )
            .await
            .unwrap();
        seen = serde_json::from_str(&out.content).unwrap();
        if seen["last_progress"].is_string() {
            break;
        }
    }
    assert_eq!(seen["active"], false);
    assert!(seen["last_progress"].is_null());
    ctx.cancellation_token().cancel();
}

struct RejectEnd;
#[async_trait::async_trait]
impl Hook for RejectEnd {
    async fn on_agent_end(
        &self,
        _ctx: &AgentCtx,
        _agent: &str,
        _out: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        Err("end failed".into())
    }
}
#[tokio::test]
async fn end_hook_failure_still_releases_later_lease() {
    let ctx = Engine::builder()
        .mock_ctx()
        .with_caller(candid::Principal::self_authenticating([1]));
    let mut hooks = Hooks::new();
    hooks.add(Box::new(RejectEnd));
    hooks.add(Box::new(SingleThreadHook::new(Duration::from_secs(60))));
    hooks.on_agent_start(&ctx, "worker").await.unwrap();
    assert!(
        hooks
            .on_agent_end(&ctx, "worker", AgentOutput::default())
            .await
            .is_err()
    );
    hooks.on_agent_start(&ctx, "worker").await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn foreground_shell_observes_cancellation() {
    use anda_engine::extension::shell::{ExecArgs, Executor, NativeRuntime};
    let ctx = Engine::builder().mock_ctx();
    let token = ctx.cancellation_token();
    let runtime = NativeRuntime::new(std::env::temp_dir());
    let fut = runtime.execute(
        ctx.base,
        ExecArgs {
            command: "sleep 0.1; printf finished".into(),
            ..Default::default()
        },
        Default::default(),
    );
    tokio::pin!(fut);
    assert!(futures::poll!(&mut fut).is_pending());
    token.cancel();
    assert!(fut.await.unwrap_err().to_string().contains("cancelled"));
}

#[tokio::test]
async fn prefixed_subagent_allowlist_resolves_definitions() {
    let builder = Engine::builder();
    let ctx = builder.mock_ctx();
    // The built-in registry can be updated through its normal public manager agent.
    ctx.clone()
        .agent_run(AgentInput {
            name: "subagents_manager".into(),
            prompt:
                json!({"name":"worker","description":"worker","instructions":"work","tools":[]})
                    .to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        ctx.definitions(Some(&["SA_worker".to_string()])).await[0].name,
        "SA_worker"
    );
    assert_eq!(
        ctx.definitions(Some(&["worker".to_string()])).await[0].name,
        "SA_worker"
    );
}

#[tokio::test]
async fn selector_honors_limit_and_preserves_usage() {
    use anda_engine::context::{ToolsOutput, ToolsSelect};
    let reply = AgentOutput {
        content: json!({"tools":["tools_search","tools_groups"]}).to_string(),
        usage: anda_core::Usage {
            input_tokens: 77,
            output_tokens: 5,
            requests: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let ctx = Engine::builder()
        .with_model(Model::with_completer(
            ScriptedCompleter::new("selector")
                .push_output(reply)
                .into_arc(),
        ))
        .mock_ctx();
    let out = ToolsSelect::new()
        .run(
            ctx,
            json!({"query":"search for tools","limit":1}).to_string(),
            vec![],
        )
        .await
        .unwrap();
    let selected: ToolsOutput = serde_json::from_str(&out.content).unwrap();
    assert_eq!(selected.tools.len(), 1);
    assert_eq!(out.usage.input_tokens, 77);
    assert_eq!(out.usage.requests, 1);
}

#[cfg(unix)]
#[tokio::test]
async fn mcp_task_deadline_bounds_an_unanswered_get() {
    use anda_core::ToolProvider;
    use anda_engine::extension::mcp::{
        McpLifecycle, McpServerConfig, McpStdioTransport, McpTasksConfig, McpToolProvider,
        McpTransportConfig,
    };
    let script = r#"import sys,json
for line in sys.stdin:
 r=json.loads(line); m=r.get('method'); v=None
 if m=='initialize': v={'protocolVersion':'2025-11-25','capabilities':{'tools':{}},'serverInfo':{'name':'fake','version':'1'}}
 elif m=='tools/list': v={'tools':[{'name':'echo','inputSchema':{'type':'object'}}]}
 elif m=='tools/call': v={'resultType':'task','taskId':'t1','status':'working','createdAt':'2026-07-28T00:00:00Z','lastUpdatedAt':'2026-07-28T00:00:00Z','ttlMs':None,'pollIntervalMs':10}
 elif m=='tasks/cancel': v={}
 if v is not None: print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':v}),flush=True)
"#;
    let mut config = McpServerConfig::stdio("slow", "python3");
    config.lifecycle = McpLifecycle::Initialize;
    config.tasks = Some(McpTasksConfig { max_wait_secs: 1 });
    config.transport = McpTransportConfig::Stdio(McpStdioTransport {
        command: "python3".into(),
        args: vec!["-u".into(), "-c".into(), script.into()],
        ..Default::default()
    });
    let provider = McpToolProvider::new(vec![config]).unwrap();
    provider.refresh_server("slow").await.unwrap();
    let ctx = Engine::builder().mock_ctx();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        provider.call(
            ctx.base,
            ToolInput {
                name: "mcp_slow_echo".into(),
                args: json!({}),
                ..Default::default()
            },
        ),
    )
    .await;
    assert!(
        result
            .expect("MCP deadline must finish the call")
            .unwrap_err()
            .to_string()
            .contains("did not finish within 1s")
    );
    provider.remove_server("slow");
}

#[tokio::test]
async fn responses_stream_error_is_reported() {
    use axum::{Router, routing::post};
    let events = format!(
        "data: {}\n\ndata: {}\n\n",
        json!({"type":"response.created","response":{"id":"resp_1","created_at":1,"model":"test","status":"in_progress","output":[]}}),
        json!({"type":"error","code":"server_error","message":"upstream failed","sequence_number":1})
    );
    let app = Router::new().route(
        "/responses",
        post(move || {
            let body = events.clone();
            async move { ([("content-type", "text/event-stream")], body) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let model = anda_engine::model::openai::Client::new("test", Some(format!("http://{address}")))
        .with_client(reqwest::Client::builder().no_proxy().build().unwrap())
        .completion_model_v2("test");
    let err = model
        .completion(CompletionRequest {
            prompt: "hello".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("upstream failed"));
    assert!(anda_engine::model::is_retryable_box_error(&err));
    server.abort();
}

#[derive(Default)]
struct ProgressIds(parking_lot::Mutex<Vec<String>>);
#[async_trait::async_trait]
impl anda_engine::hook::AgentHook for ProgressIds {
    async fn on_background_start(
        &self,
        _ctx: &AgentCtx,
        handle: anda_engine::hook::BackgroundHandle,
        _req: &CompletionRequest,
    ) {
        self.0.lock().push(format!("start:{}", handle.task_id()));
    }
    async fn on_background_progress(&self, _ctx: &AgentCtx, id: String, _out: AgentOutput) {
        self.0.lock().push(format!("progress:{id}"));
    }
}
#[tokio::test]
async fn subagent_progress_uses_the_registered_task_id() {
    let ctx = Engine::builder()
        .with_model(Model::with_completer(
            ScriptedCompleter::new("echo").into_arc(),
        ))
        .mock_ctx();
    let events = Arc::new(ProgressIds::default());
    ctx.base
        .set_state(anda_engine::hook::DynAgentHook::new(events.clone()));
    let agent = SubAgent {
        name: "worker".into(),
        ..Default::default()
    };
    agent
        .run(
            ctx.clone(),
            json!({"session":"s1","prompt":"work"}).to_string(),
            vec![],
        )
        .await
        .unwrap();
    for _ in 0..20 {
        tokio::task::yield_now().await;
        if events.0.lock().len() > 1 {
            break;
        }
    }
    assert_eq!(*events.0.lock(), ["start:worker:s1", "progress:worker:s1"]);
    ctx.cancellation_token().cancel();
}

#[test]
fn gemini_tools_use_json_schema_fields() {
    use anda_core::Tool;
    let def = anda_engine::extension::todo::TodoTool::new().definition();
    let wire: anda_engine::model::gemini::types::Tool = vec![def].into();
    let value = serde_json::to_value(wire).unwrap();
    let params = &value["functionDeclarations"][0]["parametersJsonSchema"];
    assert!(params["properties"]["op"]["type"].is_array());
    assert!(value["functionDeclarations"][0].get("parameters").is_none());
}

#[tokio::test]
async fn namespace_listing_respects_prefix_offset_and_root() {
    let store = Store::new(Arc::new(InMemory::new()));
    for (ns, path) in [
        ("a", "dir/one"),
        ("a", "dir/two"),
        ("a", "other"),
        ("ab", "dir/three"),
    ] {
        store
            .store_put(
                &Path::from(ns),
                &Path::from(path),
                PutMode::Overwrite,
                "x".into(),
            )
            .await
            .unwrap();
    }
    for prefix in [None, Some(Path::default())] {
        assert_eq!(
            store
                .store_list(&Path::from("a"), prefix.as_ref(), &Path::default())
                .await
                .unwrap()
                .len(),
            3
        );
    }
    let listed = store
        .store_list(
            &Path::from("a"),
            Some(&Path::from("dir")),
            &Path::from("dir/one"),
        )
        .await
        .unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].location.as_ref(), "a/dir/two");
    assert_eq!(
        store
            .store_list(&Path::default(), None, &Path::default())
            .await
            .unwrap()
            .len(),
        4
    );
}

#[tokio::test]
async fn compaction_empty_summary_preserves_history_and_queued_input() {
    use anda_core::{ContentPart, Message, Usage};
    let model = ScriptedCompleter::new("summary")
        .push_output(AgentOutput {
            content: "answer".into(),
            chat_history: vec![Message {
                role: "assistant".into(),
                content: vec![ContentPart::from("answer".to_string())],
                ..Default::default()
            }],
            usage: Usage {
                input_tokens: 3,
                requests: 1,
                ..Default::default()
            },
            ..Default::default()
        })
        .push_output(AgentOutput {
            usage: Usage {
                input_tokens: 5,
                requests: 1,
                ..Default::default()
            },
            ..Default::default()
        })
        .into_arc();
    let ctx = Engine::builder()
        .with_model(Model::with_completer(model))
        .mock_ctx();
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                prompt: "task".into(),
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    runner.next().await.unwrap();
    runner.follow_up("continue".to_string());
    assert!(runner.handoff(None).await.is_err());
    assert!(!runner.is_done());
    assert_eq!(runner.chat_history().len(), 1);
    assert_eq!(runner.total_usage().input_tokens, 8);
    assert_eq!(runner.follow_up_message_iter().count(), 1);
    assert_eq!(runner.next().await.unwrap().unwrap().content, "continue");
}

#[tokio::test]
async fn concurrent_note_updates_preserve_both_items() {
    use anda_core::Tool;
    use anda_engine::extension::note::{NoteArgs, NoteItemInput, NoteTool};
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    // Delay writes so an unprotected pair of read-modify-writes reads the same old value.
    let store = ThrottledStore::new(
        InMemory::new(),
        ThrottleConfig {
            wait_put_per_call: Duration::from_millis(20),
            ..Default::default()
        },
    );
    let ctx = Engine::builder()
        .with_store(Store::new(Arc::new(store)))
        .mock_ctx()
        .child_base("note")
        .unwrap();
    let tool = NoteTool::new();
    let clone = tool.clone();
    let args = |id: &str| NoteArgs {
        op: Some("upsert".into()),
        items: Some(vec![NoteItemInput {
            id: id.into(),
            content: Some(id.into()),
        }]),
    };
    let (first, second) = tokio::join!(
        tool.call(ctx.clone(), args("first"), vec![]),
        clone.call(ctx.clone(), args("second"), vec![])
    );
    first.unwrap();
    second.unwrap();
    let output = tool.call(ctx, NoteArgs::default(), vec![]).await.unwrap();
    let ids: std::collections::BTreeSet<_> = output
        .output
        .items
        .iter()
        .map(|item| item.id.as_str())
        .collect();
    assert_eq!(ids, std::collections::BTreeSet::from(["first", "second"]));
}

struct LazyProvider(std::sync::atomic::AtomicBool);
impl anda_core::ToolProvider<anda_engine::context::BaseCtx> for LazyProvider {
    fn name(&self) -> String {
        "lazy".into()
    }
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        if self.0.load(std::sync::atomic::Ordering::SeqCst)
            && names.is_none_or(|names| names.iter().any(|name| name == "dynamic_echo"))
        {
            vec![definition("dynamic_echo")]
        } else {
            vec![]
        }
    }
    fn init(
        &self,
        _ctx: anda_engine::context::BaseCtx,
    ) -> anda_core::BoxFut<'_, Result<(), BoxError>> {
        Box::pin(async move {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        })
    }
    fn call(
        &self,
        _ctx: anda_engine::context::BaseCtx,
        _input: ToolInput<Json>,
    ) -> anda_core::BoxFut<'_, Result<anda_core::ToolOutput<Json>, BoxError>> {
        Box::pin(async { Ok(anda_core::ToolOutput::new(json!("echo"))) })
    }
}
struct DependentAgent(&'static str);
impl Agent<AgentCtx> for DependentAgent {
    fn name(&self) -> String {
        "dependent".into()
    }
    fn description(&self) -> String {
        "Needs a provider tool".into()
    }
    fn tool_dependencies(&self) -> Vec<String> {
        vec![self.0.into()]
    }
    async fn run(
        &self,
        _ctx: AgentCtx,
        _prompt: String,
        _resources: Vec<anda_core::Resource>,
    ) -> Result<AgentOutput, BoxError> {
        Ok(AgentOutput::default())
    }
}
#[tokio::test]
async fn agent_dependencies_are_checked_after_provider_initialization() {
    for dependency in ["dynamic_echo", "missing"] {
        let provider = Arc::new(LazyProvider(std::sync::atomic::AtomicBool::new(false)));
        let builder = Engine::builder()
            .register_tool_provider(provider)
            .unwrap()
            .register_agent(Arc::new(DependentAgent(dependency)), None)
            .unwrap();
        let result = builder.build("dependent".into()).await;
        if dependency == "dynamic_echo" {
            assert!(result.is_ok());
        } else {
            assert!(
                result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("dependent tool missing not found")
            );
        }
    }
}

#[test]
fn mcp_debug_never_prints_custom_header_values() {
    use anda_engine::extension::mcp::McpStreamableHttpTransport;
    let config = McpStreamableHttpTransport {
        url: "https://example.invalid/mcp".into(),
        headers: BTreeMap::from([
            ("Authorization".into(), "Bearer secret-review-marker".into()),
            ("X-API-Key".into(), "secret-review-marker".into()),
        ]),
        ..Default::default()
    };
    let text = format!("{config:?}");
    assert!(text.contains("Authorization"));
    assert!(!text.contains("secret-review-marker"));
}

#[tokio::test]
async fn callers_with_the_same_session_id_remain_independent() {
    let ctx = Engine::builder()
        .with_model(Model::with_completer(
            ScriptedCompleter::new("echo").into_arc(),
        ))
        .mock_ctx();
    let agent = SubAgent {
        name: "worker".into(),
        ..Default::default()
    };
    let a = candid::Principal::self_authenticating([1]);
    let b = candid::Principal::self_authenticating([2]);
    for (caller, prompt) in [(a, "private-a"), (b, "private-b")] {
        agent
            .run(
                ctx.with_caller(caller),
                json!({"session":"same","prompt":prompt}).to_string(),
                vec![],
            )
            .await
            .unwrap();
    }
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let mut ready = true;
            for (caller, expected) in [(a, "private-a"), (b, "private-b")] {
                let out = agent
                    .run(
                        ctx.with_caller(caller),
                        json!({"session":"same","prompt":"/status"}).to_string(),
                        vec![],
                    )
                    .await
                    .unwrap();
                let value: Json = serde_json::from_str(&out.content).unwrap();
                ready &= value["last_progress"] == expected;
            }
            if ready {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        agent.subsessions.get_session("same").is_none(),
        "host lookup must reject ambiguous IDs"
    );
    let manager = anda_engine::subagent::SubAgentManager::new();
    manager.upsert_temporary(agent.clone()).unwrap();
    let out = manager
        .run(
            ctx.with_caller(b),
            json!({"operation":"list"}).to_string(),
            vec![],
        )
        .await
        .unwrap();
    assert!(out.content.contains("private-b"));
    assert!(!out.content.contains("private-a"));
    agent
        .run(
            ctx.with_caller(a),
            json!({"session":"same","prompt":"/cancel finished-a"}).to_string(),
            vec![],
        )
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while agent.subsessions.get_session_for(&a, "same").is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(agent.subsessions.get_session_for(&b, "same").is_some());
    ctx.cancellation_token().cancel();
}

#[tokio::test]
async fn stop_and_cancel_interrupt_an_in_flight_subagent_model() {
    let started = Arc::new(tokio::sync::Notify::new());
    let ctx = Engine::builder()
        .with_model(Model::with_completer(Arc::new(GatedCompleter {
            started: started.clone(),
            ready: Arc::new(tokio::sync::Notify::new()),
        })))
        .mock_ctx();
    let agent = SubAgent {
        name: "worker".into(),
        ..Default::default()
    };
    let call = |prompt: &str| json!({"session":"slow","prompt":prompt}).to_string();
    agent.run(ctx.clone(), call("work"), vec![]).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    agent
        .run(ctx.clone(), call("/stop pause"), vec![])
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let out = agent
                .run(ctx.clone(), call("/status"), vec![])
                .await
                .unwrap();
            let status: Json = serde_json::from_str(&out.content).unwrap();
            if status["busy"] == false
                && status["last_progress"]
                    .as_str()
                    .is_some_and(|text| text.contains("pause"))
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    agent
        .run(ctx.clone(), call("resume"), vec![])
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    agent
        .run(ctx.clone(), call("/cancel done"), vec![])
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while agent.subsessions.get_session("slow").is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    ctx.cancellation_token().cancel();
}

#[cfg(unix)]
#[tokio::test]
async fn dropping_foreground_shell_stops_its_descendants() {
    use anda_engine::extension::shell::{ExecArgs, Executor, NativeRuntime};
    let dir = std::env::temp_dir().join(format!("anda-cancel-tree-{:016x}", rand::random::<u64>()));
    tokio::fs::create_dir(&dir).await.unwrap();
    let runtime = NativeRuntime::new(dir.clone());
    let ctx = Engine::builder().mock_ctx();
    let task = tokio::spawn(async move {
        runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: "(sleep 0.3; printf leaked > marker) & printf ready > started; wait"
                        .into(),
                    ..Default::default()
                },
                Default::default(),
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while !dir.join("started").exists() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(!dir.join("marker").exists());
    tokio::fs::remove_dir_all(dir).await.unwrap();
}

struct CapturedLogs(std::sync::Mutex<Vec<String>>);
static LOGS: CapturedLogs = CapturedLogs(std::sync::Mutex::new(Vec::new()));
impl log::Log for CapturedLogs {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        struct Fields<'a>(&'a mut String);
        impl<'kvs> log::kv::VisitSource<'kvs> for Fields<'_> {
            fn visit_pair(
                &mut self,
                key: log::kv::Key<'kvs>,
                value: log::kv::Value<'kvs>,
            ) -> Result<(), log::kv::Error> {
                use std::fmt::Write;
                let _ = write!(self.0, " {key}={value}");
                Ok(())
            }
        }
        let mut message = record.args().to_string();
        let _ = record.key_values().visit(&mut Fields(&mut message));
        self.0.lock().unwrap().push(message);
    }
    fn flush(&self) {}
}
#[tokio::test]
async fn completion_logs_keep_metadata_without_conversation_content() {
    use axum::{Router, routing::post};
    log::set_logger(&LOGS).unwrap();
    log::set_max_level(log::LevelFilter::Debug);
    let app = Router::new().route("/messages", post(|| async {
        axum::Json(json!({"id":"message","type":"message","role":"assistant","model":"log-test","stop_reason":"end_turn","content":[{"type":"text","text":"private-response-review-marker"}],"usage":{"input_tokens":2,"output_tokens":3}}))
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let model = anda_engine::model::anthropic::Client::new(
        "private-key-review-marker",
        Some(format!("http://{address}")),
    )
    .with_client(reqwest::Client::builder().no_proxy().build().unwrap())
    .completion_model("log-test");
    model
        .completion(CompletionRequest {
            instructions: "private-system-review-marker".into(),
            prompt: "private-prompt-review-marker".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let logs = LOGS.0.lock().unwrap().join("\n");
    assert!(logs.contains("Completion response"));
    assert!(logs.contains("log-test"));
    for marker in [
        "private-key-review-marker",
        "private-system-review-marker",
        "private-prompt-review-marker",
        "private-response-review-marker",
    ] {
        assert!(
            !logs.contains(marker),
            "sensitive marker appeared in logs: {marker}"
        );
    }
    server.abort();
}
