use super::*;
use anda_core::{ByteBufB64, ToolPresentation};
use axum::response::IntoResponse;
use axum::{Json as HttpJson, Router, extract::State, routing::post};
use serde_json::json;
use std::{
    str::FromStr,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct ServerState {
    revision: AtomicUsize,
    requests: AtomicUsize,
    auth_fail: AtomicBool,
    lists: AtomicUsize,
    active: AtomicUsize,
    max_active: AtomicUsize,
    calls: SyncMutex<Vec<Json>>,
    list_started: tokio::sync::Notify,
    unblock_list: tokio::sync::Notify,
    block_list: AtomicBool,
    elicitation: AtomicBool,
}
struct TestServer {
    url: String,
    state: Arc<ServerState>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl TestServer {
    async fn start() -> Self {
        async fn handle(
            State(state): State<Arc<ServerState>>,
            HttpJson(request): HttpJson<Json>,
        ) -> axum::response::Response {
            state.requests.fetch_add(1, Ordering::SeqCst);
            if state.auth_fail.load(Ordering::SeqCst) {
                return (
                    axum::http::StatusCode::UNAUTHORIZED,
                    [("www-authenticate", "Bearer error=\"invalid_token\"")],
                    "unauthorized",
                )
                    .into_response();
            }
            let result = match request["method"].as_str().unwrap_or("") {
                "server/discover" => {
                    json!({"resultType":"complete","supportedVersions":["2026-07-28"], "capabilities":{"tools":{},"resources":{}}, "ttlMs":0,"cacheScope":"private","_meta":{"io.modelcontextprotocol/serverInfo":{"name":"test","version":"1"}}})
                }
                "tools/list" => {
                    state.lists.fetch_add(1, Ordering::SeqCst);
                    let revision = state.revision.load(Ordering::SeqCst);
                    if state.block_list.load(Ordering::SeqCst) {
                        state.list_started.notify_one();
                        state.unblock_list.notified().await;
                    }
                    json!({"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[{"name":"echo","description":format!("revision {revision}"),"inputSchema":{"type":"object"}, "annotations":{"readOnlyHint":true}}]})
                }
                "tools/call" => {
                    state.calls.lock().push(request["params"].clone());
                    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
                    state.max_active.fetch_max(active, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    state.active.fetch_sub(1, Ordering::SeqCst);
                    if state.elicitation.load(Ordering::SeqCst)
                        && request["params"]["inputResponses"].is_null()
                    {
                        json!({"resultType":"input_required","requestState":"continuation","inputRequests":{"form":{"method":"elicitation/create","params":{"mode":"form","message":"Choose a value","requestedSchema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}}}})
                    } else {
                        json!({"resultType":"complete","content":[{"type":"text","text":"done"}],"isError":false})
                    }
                }
                "resources/list" => {
                    json!({"resultType":"complete","resources":[{"uri":"test://one","name":"one"}]})
                }
                "resources/templates/list" => {
                    json!({"resultType":"complete","resourceTemplates":[{"uriTemplate":"test://{id}","name":"by_id"}]})
                }
                "resources/read" => {
                    json!({"resultType":"complete","contents":[{"uri":"test://one","text":"resource text"}]})
                }
                _ => json!({}),
            };
            HttpJson(json!({"jsonrpc":"2.0","id":request["id"],"result":result})).into_response()
        }
        let state = Arc::new(ServerState::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/mcp", post(handle))
            .with_state(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, state, task }
    }
    fn config(&self) -> McpServerConfig {
        let mut config = McpServerConfig::streamable_http("test", &self.url);
        config.lifecycle = McpLifecycle::Discover;
        config
    }
}
fn remote_tool(name: &str) -> McpTool {
    serde_json::from_value(json!({"name":name,"inputSchema":{"type":"object"}})).unwrap()
}

#[tokio::test]
async fn stale_routes_and_replaced_registrations_cannot_execute() {
    let server = TestServer::start().await;
    let provider = McpToolProvider::new(vec![server.config()]).unwrap();
    provider.refresh_server("test").await.unwrap();
    let old = provider.routes().remove(0);
    server.state.revision.store(1, Ordering::SeqCst);
    provider.refresh_server("test").await.unwrap();
    let err = provider
        .call_route(old.clone(), ToolInput::new(old.name.clone(), json!({})))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("catalog changed"));
    assert!(server.state.calls.lock().is_empty());
    provider.remove_server("test");
    provider.add_server(server.config()).await.unwrap();
    assert!(
        provider
            .call_route(old.clone(), ToolInput::new(old.name, json!({})))
            .await
            .unwrap_err()
            .to_string()
            .contains("registration changed")
    );
    assert!(server.state.calls.lock().is_empty());
}

#[tokio::test]
async fn cancelled_refresh_rearms_dirty_and_removed_refresh_cannot_publish() {
    let server = TestServer::start().await;
    let provider = McpToolProvider::new(vec![server.config()]).unwrap();
    provider.refresh_server("test").await.unwrap();
    let session = provider.live_session("test").await.unwrap();
    server.state.block_list.store(true, Ordering::SeqCst);
    let task = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.refresh_server("test").await })
    };
    server.state.list_started.notified().await;
    assert!(!session.dirty.load(Ordering::SeqCst));
    task.abort();
    let _ = task.await;
    assert!(session.dirty.load(Ordering::SeqCst));
    server.state.unblock_list.notify_one();
    let task = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.refresh_server("test").await })
    };
    server.state.list_started.notified().await;
    provider.remove_server("test");
    server.state.unblock_list.notify_one();
    assert!(task.await.unwrap().is_err());
    assert!(provider.routes().is_empty());
    assert!(provider.inner.index.read().sessions.is_empty());
}

#[test]
fn colliding_names_are_order_independent_and_retained_across_catalog_changes() {
    let provider = McpToolProvider::new(vec![McpServerConfig::stdio("test", "unused")]).unwrap();
    let first = provider
        .routes_for_tools(
            "test",
            vec![remote_tool("read.file"), remote_tool("read_file")],
        )
        .unwrap();
    let reversed = provider
        .routes_for_tools(
            "test",
            vec![remote_tool("read_file"), remote_tool("read.file")],
        )
        .unwrap();
    let names = |routes: &[McpToolRoute]| {
        routes
            .iter()
            .map(|r| (r.remote_name.clone(), r.name.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(names(&first), names(&reversed));
    let mut index = McpToolIndex::default();
    index.replace_server_routes("test", first.clone());
    index.replace_server_routes(
        "test",
        provider
            .routes_for_tools("test", vec![remote_tool("read.file")])
            .unwrap(),
    );
    assert_eq!(index.routes.values().next().unwrap().name, first[0].name);
    index.replace_server_routes("test", reversed);
    assert_eq!(
        names(&index.routes.values().cloned().collect::<Vec<_>>()),
        names(&first)
    );
}

#[tokio::test]
async fn pagination_rejects_cycles_and_cumulative_limits() {
    let limits = McpLimits::default();
    let mut page = 0;
    let err = collect_pages::<u8, _, _>(&limits, |_| {
        page += 1;
        async move { Ok((vec![1], Some(if page % 2 == 0 { "b" } else { "a" }.into()))) }
    })
    .await
    .unwrap_err();
    assert!(err.to_string().contains("repeated"));
    assert_eq!(page, 3);
    let limits = McpLimits {
        catalog_items: 1,
        ..Default::default()
    };
    assert!(
        collect_pages(&limits, |_| async { Ok((vec![1, 2], None)) })
            .await
            .unwrap_err()
            .to_string()
            .contains("item limit")
    );
    let limits = McpLimits {
        catalog_pages: 1,
        ..Default::default()
    };
    assert!(
        collect_pages(&limits, |_| async { Ok((vec![1], Some("".into()))) })
            .await
            .unwrap_err()
            .to_string()
            .contains("page limit")
    );
    let limits = McpLimits {
        cursor_bytes: 2,
        ..Default::default()
    };
    assert!(
        collect_pages(&limits, |_| async { Ok((vec![1], Some("long".into()))) })
            .await
            .unwrap_err()
            .to_string()
            .contains("cursor limit")
    );
}

#[test]
fn presentation_preserves_media_and_raw_data_but_excludes_private_metadata() {
    let limits = McpLimits {
        output_text_bytes: 256,
        ..Default::default()
    };
    let raw = json!({"structured_content":{"value":42},"_meta":{"secret":"private"},"content":[{"type":"image","mimeType":"image/png","data":"AQID"},{"type":"text","text":"中".repeat(500),"_meta":{"secret":"private"}}]});
    let view = presentation::present_result(&raw, &limits);
    assert!(view.text.len() <= limits.output_text_bytes);
    assert!(view.text.contains("truncated"));
    assert!(!view.text.contains("private"));
    assert_eq!(view.media.len(), 1);
    assert_eq!(view.media[0].data, ByteBufB64::from_str("AQID").unwrap());
    assert_eq!(
        ToolPresentation::from_output(&view.clone().into_output()),
        Some(view)
    );
    assert_eq!(raw["_meta"]["secret"], "private");
}

#[tokio::test]
async fn concurrency_policy_serializes_writes_and_allows_opted_in_reads() {
    for (policy, expected) in [
        (McpConcurrency::Serial, 1),
        (McpConcurrency::ReadOnlyParallel, 2),
        (McpConcurrency::Parallel, 2),
    ] {
        let server = TestServer::start().await;
        let mut config = server.config();
        config.concurrency = policy;
        let provider = McpToolProvider::new(vec![config]).unwrap();
        provider.refresh_server("test").await.unwrap();
        let route = provider.routes().remove(0);
        let input = ToolInput::new(route.name.clone(), json!({}));
        let (a, b) = tokio::join!(
            provider.call_route(route.clone(), input.clone()),
            provider.call_route(route, input)
        );
        a.unwrap();
        b.unwrap();
        assert_eq!(server.state.max_active.load(Ordering::SeqCst), expected);
    }
}

struct FormHandler;
#[async_trait::async_trait]
impl McpElicitationHandler for FormHandler {
    async fn elicit(
        &self,
        server_id: &str,
        _request: rmcp::model::ElicitRequestParams,
        _cancellation: CancellationToken,
    ) -> Result<rmcp::model::ElicitResult, BoxError> {
        assert_eq!(server_id, "test");
        Ok(serde_json::from_value(
            json!({"action":"accept","content":{"answer":"yes"}}),
        )?)
    }
}
#[tokio::test]
async fn modern_elicitation_continues_and_resources_remain_explicit() {
    let server = TestServer::start().await;
    server.state.elicitation.store(true, Ordering::SeqCst);
    let mut config = server.config();
    config.elicitation = true;
    config.resources = true;
    let provider = McpToolProvider::builder()
        .server(config)
        .elicitation_handler(Arc::new(FormHandler))
        .build()
        .unwrap();
    provider.refresh_server("test").await.unwrap();
    let route = provider.routes().remove(0);
    let result = provider
        .call_route(route.clone(), ToolInput::new(route.name, json!({})))
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(false));
    let calls = server.state.calls.lock().clone();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1]["requestState"], "continuation");
    assert_eq!(
        calls[1]["inputResponses"]["form"]["content"]["answer"],
        "yes"
    );
    assert_eq!(
        provider
            .list_resources("test", CancellationToken::new())
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        provider
            .list_resource_templates("test", CancellationToken::new())
            .await
            .unwrap()
            .len(),
        1
    );
    let result = provider
        .read_resource("test", "test://one".into(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(result).unwrap()["contents"][0]["text"],
        "resource text"
    );
}

#[tokio::test]
async fn refresh_guards_coordinate_independent_credential_scopes() {
    let store = Arc::new(InMemoryMcpCredentialStore::new());
    let first = store.acquire_refresh_guard("one").await.unwrap();
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            store.acquire_refresh_guard("one")
        )
        .await
        .is_err()
    );
    let other = tokio::time::timeout(
        Duration::from_millis(20),
        store.acquire_refresh_guard("two"),
    )
    .await
    .unwrap()
    .unwrap();
    drop(first);
    drop(other);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            store.acquire_refresh_guard("one")
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn stdio_reader_bounds_split_lines_without_reimplementing_protocol() {
    use tokio::io::AsyncReadExt;
    let mut reader = bounded::LineLimitedReader::new(&b"123\n1234\n"[..], 3);
    let mut bytes = Vec::new();
    assert!(reader.read_to_end(&mut bytes).await.is_err());
    let mut reader = bounded::LineLimitedReader::new(&b"123\n123\n"[..], 3);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await.unwrap();
    assert_eq!(bytes, b"123\n123\n");
}

#[test]
fn sse_budget_handles_crlf_boundaries_and_unterminated_events() {
    let mut budget = http_client::EventBudget::default();
    budget.observe(b"data: x\r", 10).unwrap();
    budget.observe(b"\n\r\ndata: y\n\n", 10).unwrap();
    assert!(budget.observe(b"data: too long", 10).is_err());
}

#[test]
fn app_only_tools_stay_out_of_model_catalogs_and_metadata_is_retained() {
    let provider = McpToolProvider::new(vec![McpServerConfig::stdio("test", "unused")]).unwrap();
    let tools = vec![serde_json::from_value(json!({"name":"ui_only","inputSchema":{},"_meta":{"ui":{"visibility":["app"]}}})).unwrap(), serde_json::from_value(json!({"name":"visible","inputSchema":{"type":"object"},"outputSchema":{"type":"object"},"annotations":{"readOnlyHint":true}})).unwrap()];
    let routes = provider.routes_for_tools("test", tools).unwrap();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].remote_name, "visible");
    assert!(routes[0].tool.output_schema.is_some());
    assert_eq!(routes[0].definition.parameters["properties"], json!({}));
}

#[tokio::test]
async fn http_bodies_are_bounded_and_redirects_are_not_followed() {
    use axum::{body::Body, response::Response};
    async fn large() -> Response {
        Response::builder()
            .header("content-type", "application/json")
            .body(Body::from_stream(futures::stream::iter([
                Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b' '; 100])),
                Ok(bytes::Bytes::from(vec![b' '; 100])),
            ])))
            .unwrap()
    }
    let captured = Arc::new(AtomicBool::new(false));
    let target = captured.clone();
    let app = Router::new()
        .route("/large", post(large))
        .route(
            "/redirect",
            post(|| async {
                Response::builder()
                    .status(307)
                    .header("location", "/capture")
                    .body(Body::empty())
                    .unwrap()
            }),
        )
        .route(
            "/capture",
            post(move || {
                let target = target.clone();
                async move {
                    target.store(true, Ordering::SeqCst);
                    "unexpected"
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = http_client::McpHttpClient::new(150).unwrap();
    use rmcp::transport::streamable_http_client::StreamableHttpClient;
    let request =
        || serde_json::from_value(json!({"jsonrpc":"2.0","id":1,"method":"tools/list"})).unwrap();
    let error = client
        .post_message(
            format!("http://{address}/large").into(),
            request(),
            None,
            None,
            HashMap::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("byte budget"));
    let error = client
        .post_message(
            format!("http://{address}/redirect").into(),
            request(),
            None,
            Some("secret".into()),
            HashMap::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("307"));
    assert!(!captured.load(Ordering::SeqCst));
    task.abort();
}

#[tokio::test]
async fn required_startup_fails_and_background_startup_does_not_wait() {
    let mut config = McpServerConfig::stdio("missing", "anda_nonexistent_mcp_command_xyz");
    config.required = true;
    let provider = McpToolProvider::new(vec![config]).unwrap();
    let ctx = crate::engine::Engine::builder().mock_ctx();
    assert!(provider.init(ctx.base.clone()).await.is_err());
    assert_eq!(
        provider.server_statuses().await["missing"],
        McpServerStatus::Failed
    );
    let server = TestServer::start().await;
    server.state.block_list.store(true, Ordering::SeqCst);
    let mut config = server.config();
    config.startup = McpStartup::Background;
    let provider = McpToolProvider::new(vec![config]).unwrap();
    tokio::time::timeout(Duration::from_millis(200), provider.init(ctx.base.clone()))
        .await
        .unwrap()
        .unwrap();
    server.state.list_started.notified().await;
    assert!(provider.routes().is_empty());
    server.state.unblock_list.notify_one();
    tokio::time::timeout(Duration::from_secs(2), async {
        while provider.routes().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        provider.server_statuses().await["test"],
        McpServerStatus::Ready
    );
}

#[tokio::test]
async fn simultaneous_dirty_calls_share_one_refresh() {
    let server = TestServer::start().await;
    let provider = McpToolProvider::new(vec![server.config()]).unwrap();
    provider.refresh_server("test").await.unwrap();
    let session = provider.live_session("test").await.unwrap();
    session.dirty.store(true, Ordering::SeqCst);
    let (a, b) = tokio::join!(
        provider.refresh_if_dirty("test"),
        provider.refresh_if_dirty("test")
    );
    a.unwrap();
    b.unwrap();
    assert_eq!(server.state.lists.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn tool_deadline_includes_waiting_for_server_concurrency() {
    let server = TestServer::start().await;
    let mut config = server.config();
    config.timeouts.call_secs = 1;
    let provider = McpToolProvider::new(vec![config]).unwrap();
    provider.refresh_server("test").await.unwrap();
    let registration = provider.server_config("test").unwrap();
    let _occupied = registration.calls.write().await;
    let route = provider.routes().remove(0);
    let err = provider
        .call_route(route.clone(), ToolInput::new(route.name, json!({})))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("logical tool call timed out"));
    assert!(server.state.calls.lock().is_empty());
}

#[test]
fn inherited_environment_is_explicit_and_secrets_are_not_in_definitions() {
    let config = McpStdioTransport {
        command: "server".into(),
        env: BTreeMap::from([("ONLY_THIS_SECRET".into(), "private".into())]),
        ..Default::default()
    };
    let command = config.command();
    let std = command.as_std();
    let vars = std
        .get_envs()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.map(|s| s.to_string_lossy().into_owned()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(vars["ONLY_THIS_SECRET"].as_deref(), Some("private"));
    assert!(!format!("{config:?}").contains("private"));
    assert!(!vars.contains_key("CODEX_HOME"));
}

#[tokio::test]
async fn auth_failures_never_trigger_lifecycle_fallback_or_replay_a_tool() {
    let server = TestServer::start().await;
    let mut config = server.config();
    config.lifecycle = McpLifecycle::Auto;
    let provider = McpToolProvider::new(vec![config]).unwrap();
    server.state.auth_fail.store(true, Ordering::SeqCst);
    assert!(provider.refresh_server("test").await.is_err());
    assert_eq!(server.state.requests.load(Ordering::SeqCst), 1);
    assert_eq!(
        provider.server_statuses().await["test"],
        McpServerStatus::AuthorizationRequired
    );
    server.state.auth_fail.store(false, Ordering::SeqCst);
    provider.refresh_server("test").await.unwrap();
    let route = provider.routes().remove(0);
    let before = server.state.requests.load(Ordering::SeqCst);
    server.state.auth_fail.store(true, Ordering::SeqCst);
    let result = provider
        .call_route(route.clone(), ToolInput::new(route.name, json!({})))
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(true));
    assert_eq!(result.output["error"]["code"], "authorization_required");
    assert_eq!(server.state.requests.load(Ordering::SeqCst), before + 1);
}

#[cfg(unix)]
#[tokio::test]
async fn dropping_stdio_session_closes_descendant_pipes() {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
    let mut command = tokio::process::Command::new("/bin/sh");
    command.args(["-c", "sleep 30 & printf 'ready\\n'; wait"]);
    let (process, (read, _write)) = bounded::spawn(command, 1024).unwrap();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(2), reader.read_line(&mut line))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(line.trim(), "ready");
    drop(process);
    let mut remaining = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), reader.read_to_end(&mut remaining))
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn failed_add_never_rolls_back_a_replacement_registration() {
    let server = TestServer::start().await;
    server.state.block_list.store(true, Ordering::SeqCst);
    let provider = McpToolProvider::new(Vec::new()).unwrap();
    let config = server.config();
    let task = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.add_server(config).await })
    };
    server.state.list_started.notified().await;
    provider.remove_server("test");
    provider.register_server(server.config()).unwrap();
    let generation = provider.server_config("test").unwrap().generation;
    server.state.unblock_list.notify_one();
    assert!(task.await.unwrap().is_err());
    assert_eq!(
        provider.server_config("test").unwrap().generation,
        generation
    );
    assert!(provider.routes().is_empty());
}

#[tokio::test]
async fn sign_out_releases_credentials_before_waiting_for_connection_locks() {
    #[derive(Default)]
    struct Store {
        inner: InMemoryMcpCredentialStore,
        cleared: tokio::sync::Notify,
    }
    #[async_trait::async_trait]
    impl McpCredentialStore for Store {
        async fn load(&self, id: &str) -> Result<Option<StoredCredentials>, BoxError> {
            self.inner.load(id).await
        }
        async fn save(&self, id: &str, credentials: StoredCredentials) -> Result<(), BoxError> {
            self.inner.save(id, credentials).await
        }
        async fn clear(&self, id: &str) -> Result<(), BoxError> {
            self.inner.clear(id).await?;
            self.cleared.notify_one();
            Ok(())
        }
        async fn acquire_refresh_guard(
            &self,
            id: &str,
        ) -> Result<Option<rmcp::transport::auth::CredentialRefreshGuard>, BoxError> {
            self.inner.acquire_refresh_guard(id).await
        }
    }
    let store = Arc::new(Store::default());
    let provider = McpToolProvider::builder()
        .server(McpServerConfig::stdio("test", "unused"))
        .credential_store(store.clone())
        .build()
        .unwrap();
    let registration = provider.server_config("test").unwrap();
    let refresh = registration.refresh.lock().await;
    let task = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.clear_credentials("test").await })
    };
    store.cleared.notified().await;
    let credentials = tokio::time::timeout(
        Duration::from_millis(200),
        store.acquire_refresh_guard("test"),
    )
    .await
    .unwrap()
    .unwrap();
    drop(credentials);
    drop(refresh);
    task.await.unwrap().unwrap();
}
