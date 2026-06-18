//! MCP tool provider extension.
//!
//! This module makes Anda an MCP host/client for tool execution. It discovers
//! tools from configured MCP servers, maps them to legal Anda function names,
//! and dispatches calls back to the original MCP tool name. It intentionally
//! does not expose deprecated MCP client utility capabilities such as Roots,
//! Sampling, or Logging control.
//!
//! # Example
//!
//! Register an MCP provider with the engine, then dynamically add an MCP server
//! without rebuilding the engine. The provider refreshes the server's
//! `tools/list` response and exposes each remote MCP tool as an Anda-compatible
//! function name.
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use anda_engine::{
//!     engine::Engine,
//!     extension::mcp::{McpServerConfig, McpToolProvider, McpTransportConfig},
//! };
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let mcp_provider = Arc::new(McpToolProvider::new(Vec::new())?);
//!
//! let builder = Engine::builder()
//!     .register_tool_provider(mcp_provider.clone())?;
//!
//! // Continue configuring the engine:
//! // let engine = builder
//! //     .register_agent(Arc::new(my_agent), None)?
//! //     .build("my_agent".to_string())
//! //     .await?;
//!
//! let mut filesystem = McpServerConfig::stdio("filesystem", "npx");
//! if let McpTransportConfig::Stdio(stdio) = &mut filesystem.transport {
//!     stdio.args = vec![
//!         "-y".to_string(),
//!         "@modelcontextprotocol/server-filesystem".to_string(),
//!         "/path/to/workspace".to_string(),
//!     ];
//! }
//! mcp_provider.add_server(filesystem).await?;
//! # let _ = builder;
//! # Ok(())
//! # }
//! ```

use anda_core::{
    BoxError, BoxFut, FunctionDefinition, Json, ToolInput, ToolOutput, ToolProvider, Usage,
    validate_function_name,
};
use http::{HeaderName, HeaderValue};
use parking_lot::RwLock;
use rmcp::{
    ClientHandler, ErrorData as McpError, RoleClient,
    model::{
        CallToolRequestParams, CallToolResult, ClientInfo, Implementation, ListRootsRequestMethod,
        ListRootsResult, Tool as McpTool,
    },
    serve_client,
    service::{RequestContext, RunningService},
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, hash_map::DefaultHasher},
    future::Future,
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::{process::Command, sync::Mutex};

use crate::context::BaseCtx;

/// Default model-facing prefix for MCP-backed tools.
pub const DEFAULT_MCP_TOOL_PREFIX: &str = "mcp";

/// Dynamic tool provider backed by one or more MCP servers.
#[derive(Clone)]
pub struct McpToolProvider {
    inner: Arc<McpToolProviderInner>,
}

impl std::fmt::Debug for McpToolProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolProvider")
            .field("name", &self.inner.name)
            .field("tool_prefix", &self.inner.tool_prefix)
            .field("servers", &self.server_ids())
            .finish()
    }
}

impl McpToolProvider {
    /// Creates a provider named `mcp` with the default `mcp` tool prefix.
    pub fn new(servers: Vec<McpServerConfig>) -> Result<Self, BoxError> {
        Self::builder().servers(servers).build()
    }

    /// Creates a configurable provider builder.
    pub fn builder() -> McpToolProviderBuilder {
        McpToolProviderBuilder::default()
    }

    /// Adds an MCP server at runtime and refreshes its tool snapshot.
    ///
    /// The same provider instance can already be registered in an [`Engine`].
    /// Once this method succeeds, newly discovered tools are visible through the
    /// provider snapshot used by `tools_select`, `Engine::tools`, and tool calls
    /// without rebuilding the engine.
    ///
    /// If the initial refresh fails, the server registration is rolled back so
    /// callers do not observe a partially added server.
    ///
    /// [`Engine`]: crate::engine::Engine
    pub async fn add_server(&self, server: McpServerConfig) -> Result<(), BoxError> {
        let server_id = server.id.clone();
        self.insert_server(server)?;

        if let Err(err) = self.refresh_server(&server_id).await {
            self.remove_server_state(&server_id);
            return Err(err);
        }

        Ok(())
    }

    /// Returns whether a server id is currently registered.
    pub fn contains_server(&self, server_id: &str) -> bool {
        self.inner.servers.read().contains_key(server_id)
    }

    /// Returns registered MCP server ids.
    pub fn server_ids(&self) -> Vec<String> {
        self.inner.servers.read().keys().cloned().collect()
    }

    /// Refreshes a single configured server and updates the provider snapshot.
    pub async fn refresh_server(&self, server_id: &str) -> Result<(), BoxError> {
        let config = self.server_config(server_id)?;
        let session = self.ensure_session(&config).await?;
        let peer = {
            let service = session.service.lock().await;
            service.peer().clone()
        };
        let tools = peer.list_all_tools().await?;

        let routes = self.routes_for_tools(&config.id, tools)?;
        self.inner
            .index
            .write()
            .replace_server_routes(&config.id, routes);
        session.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Returns all currently known MCP routes keyed by Anda-facing tool name.
    pub fn routes(&self) -> Vec<McpToolRoute> {
        self.inner.index.read().routes.values().cloned().collect()
    }

    /// Refreshes every configured server.
    ///
    /// When `tolerant` is set, per-server failures are logged and skipped so a
    /// single unreachable server cannot abort the whole operation (used at
    /// startup). Otherwise the failing servers are reported as an aggregated
    /// error.
    async fn refresh_servers(&self, tolerant: bool) -> Result<(), BoxError> {
        let mut errors = Vec::new();
        for server_id in self.server_ids() {
            if let Err(err) = self.refresh_server(&server_id).await {
                if tolerant {
                    log::warn!(
                        "MCP provider {}: failed to refresh server {server_id}: {err}",
                        self.inner.name
                    );
                } else {
                    errors.push(format!("{server_id}: {err}"));
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("failed to refresh MCP servers: {}", errors.join("; ")).into())
        }
    }

    fn server_config(&self, server_id: &str) -> Result<Arc<McpServerConfig>, BoxError> {
        self.inner
            .servers
            .read()
            .get(server_id)
            .cloned()
            .ok_or_else(|| format!("MCP server {} not configured", server_id).into())
    }

    /// Returns the cached session for a server if it exists and is still open.
    async fn live_session(&self, server_id: &str) -> Option<Arc<McpSession>> {
        let session = self.inner.index.read().sessions.get(server_id).cloned()?;
        if session.is_closed().await {
            None
        } else {
            Some(session)
        }
    }

    async fn ensure_session(&self, config: &McpServerConfig) -> Result<Arc<McpSession>, BoxError> {
        if let Some(session) = self.live_session(&config.id).await {
            return Ok(session);
        }

        // Serialize connection establishment per server so concurrent callers
        // racing to (re)connect don't spawn duplicate child processes/sessions.
        let connect_lock = self
            .inner
            .connect_locks
            .read()
            .get(&config.id)
            .cloned()
            .ok_or_else(|| format!("MCP server {} not configured", config.id))?;
        let _guard = connect_lock.lock().await;

        // Re-check: another caller may have connected while we waited.
        if let Some(session) = self.live_session(&config.id).await {
            return Ok(session);
        }

        let dirty = Arc::new(AtomicBool::new(false));
        let handler = AndaMcpClient::new(dirty.clone());
        let service = match &config.transport {
            McpTransportConfig::Stdio(stdio) => {
                let transport = TokioChildProcess::new(stdio.command())?;
                serve_client(handler, transport).await?
            }
            McpTransportConfig::StreamableHttp(http) => {
                let transport =
                    StreamableHttpClientTransport::from_config(http.transport_config()?);
                serve_client(handler, transport).await?
            }
        };

        let session = Arc::new(McpSession {
            service: Mutex::new(service),
            dirty,
        });
        self.inner
            .index
            .write()
            .sessions
            .insert(config.id.clone(), session.clone());
        Ok(session)
    }

    async fn refresh_if_dirty(&self, server_id: &str) -> Result<(), BoxError> {
        // Atomically claim the refresh so concurrent callers triggered by the
        // same `tools/list_changed` notification don't all refresh at once.
        let claimed = self
            .inner
            .index
            .read()
            .sessions
            .get(server_id)
            .map(|session| session.dirty.swap(false, Ordering::SeqCst))
            .unwrap_or(false);
        if claimed && let Err(err) = self.refresh_server(server_id).await {
            // Restore the dirty flag so a later call retries the refresh.
            if let Some(session) = self.inner.index.read().sessions.get(server_id) {
                session.dirty.store(true, Ordering::SeqCst);
            }
            return Err(err);
        }
        Ok(())
    }

    fn routes_for_tools(
        &self,
        server_id: &str,
        tools: Vec<McpTool>,
    ) -> Result<Vec<McpToolRoute>, BoxError> {
        let mut routes = Vec::new();
        let mut used = BTreeSet::new();
        for tool in tools {
            let remote_name = tool.name.to_string();
            if !self.includes_tool(server_id, &remote_name) {
                continue;
            }

            let mut local_name = self.local_tool_name(server_id, &remote_name, None)?;
            if used.contains(&local_name) {
                local_name = self.local_tool_name(server_id, &remote_name, Some(&remote_name))?;
            }
            used.insert(local_name.clone());

            let definition = self.function_definition(server_id, &local_name, &tool);
            routes.push(McpToolRoute {
                name: local_name,
                server_id: server_id.to_string(),
                remote_name,
                definition,
            });
        }
        Ok(routes)
    }

    fn includes_tool(&self, server_id: &str, remote_name: &str) -> bool {
        let Some(config) = self.inner.servers.read().get(server_id).cloned() else {
            return false;
        };
        if config.exclude.contains(remote_name) {
            return false;
        }
        config.include.is_empty() || config.include.contains(remote_name)
    }

    fn function_definition(
        &self,
        server_id: &str,
        local_name: &str,
        tool: &McpTool,
    ) -> FunctionDefinition {
        let mut description = format!("MCP server `{server_id}` tool `{}`.", tool.name);
        if let Some(title) = tool.title.as_ref().filter(|title| !title.trim().is_empty()) {
            description.push_str(" Title: ");
            description.push_str(title.trim());
            description.push('.');
        }
        if let Some(remote_description) = tool
            .description
            .as_ref()
            .map(|description| description.trim())
            .filter(|description| !description.is_empty())
        {
            description.push(' ');
            description.push_str(remote_description);
        }

        FunctionDefinition {
            name: local_name.to_string(),
            description,
            parameters: Json::Object((*tool.input_schema).clone()),
            strict: Some(false),
        }
    }

    fn local_tool_name(
        &self,
        server_id: &str,
        remote_name: &str,
        collision_key: Option<&str>,
    ) -> Result<String, BoxError> {
        let server = sanitize_name_part(server_id);
        let tool = sanitize_name_part(remote_name);
        let base = format!("{}_{}_{}", self.inner.tool_prefix, server, tool);
        let name = match collision_key {
            Some(key) => shorten_with_hash(&base, &format!("{server_id}:{key}")),
            None if base.len() > 64 => {
                shorten_with_hash(&base, &format!("{server_id}:{remote_name}"))
            }
            None => base,
        };
        validate_function_name(&name)?;
        Ok(name)
    }

    async fn call_route(
        &self,
        route: McpToolRoute,
        input: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        self.refresh_if_dirty(&route.server_id).await?;
        // Reconnect on demand: a server may have crashed without sending a
        // `tools/list_changed` notification, leaving a closed session.
        let config = self.server_config(&route.server_id)?;
        let session = self.ensure_session(&config).await?;

        let arguments = match input.args {
            Json::Object(map) => map,
            Json::Null => Map::new(),
            other => {
                return Err(format!(
                    "MCP tool {} expects JSON object arguments, got {}",
                    route.name, other
                )
                .into());
            }
        };

        let params =
            CallToolRequestParams::new(route.remote_name.clone()).with_arguments(arguments);
        // Clone the peer so the session lock is not held across the round-trip.
        // rmcp multiplexes concurrent requests, so this lets parallel calls to
        // the same server run concurrently instead of being serialized.
        let peer = {
            let service = session.service.lock().await;
            service.peer().clone()
        };
        let result = peer.call_tool(params).await?;
        Ok(mcp_result_to_tool_output(&route, result))
    }

    fn insert_server(&self, server: McpServerConfig) -> Result<(), BoxError> {
        server.validate()?;
        let server_id = server.id.clone();
        let sanitized = sanitize_name_part(&server_id);

        let mut servers = self.inner.servers.write();
        if servers.contains_key(&server_id) {
            return Err(format!("MCP server {} already exists", server_id).into());
        }
        if servers
            .keys()
            .map(|existing| sanitize_name_part(existing))
            .any(|existing| existing == sanitized)
        {
            return Err(format!(
                "MCP server id {} collides with another server after normalization to {}",
                server_id, sanitized
            )
            .into());
        }

        servers.insert(server_id.clone(), Arc::new(server));
        self.inner
            .connect_locks
            .write()
            .insert(server_id, Arc::new(Mutex::new(())));
        Ok(())
    }

    fn remove_server_state(&self, server_id: &str) {
        self.inner.servers.write().remove(server_id);
        self.inner.connect_locks.write().remove(server_id);
        self.inner.index.write().remove_server(server_id);
    }
}

impl ToolProvider<BaseCtx> for McpToolProvider {
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        let index = self.inner.index.read();
        match names {
            Some([]) => Vec::new(),
            Some(names) => names
                .iter()
                .filter_map(|name| {
                    index
                        .routes
                        .get(&name.to_ascii_lowercase())
                        .map(|route| route.definition.clone())
                })
                .collect(),
            None => index
                .routes
                .values()
                .map(|route| route.definition.clone())
                .collect(),
        }
    }

    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.inner.index.read().routes.contains_key(lowercase_name)
    }

    fn init(&self, _ctx: BaseCtx) -> BoxFut<'_, Result<(), BoxError>> {
        // Startup must not fail because a single MCP server is unreachable;
        // failed servers are logged and can be refreshed later, on demand or
        // via an explicit `refresh()`.
        Box::pin(async move { self.refresh_servers(true).await })
    }

    fn refresh(&self) -> BoxFut<'_, Result<(), BoxError>> {
        // Explicit refresh reports per-server failures to the caller.
        Box::pin(async move { self.refresh_servers(false).await })
    }

    fn call(
        &self,
        _ctx: BaseCtx,
        mut input: ToolInput<Json>,
    ) -> BoxFut<'_, Result<ToolOutput<Json>, BoxError>> {
        Box::pin(async move {
            input.name.make_ascii_lowercase();
            let route = self
                .inner
                .index
                .read()
                .routes
                .get(&input.name)
                .cloned()
                .ok_or_else(|| format!("MCP tool {} not found", input.name))?;
            self.call_route(route, input).await
        })
    }
}

/// Builder for [`McpToolProvider`].
#[derive(Debug, Default)]
pub struct McpToolProviderBuilder {
    name: Option<String>,
    tool_prefix: Option<String>,
    servers: Vec<McpServerConfig>,
}

impl McpToolProviderBuilder {
    /// Sets the provider registry name. Defaults to `mcp`.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the model-facing tool prefix. Defaults to `mcp`.
    pub fn tool_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    /// Adds one MCP server.
    pub fn server(mut self, server: McpServerConfig) -> Self {
        self.servers.push(server);
        self
    }

    /// Replaces the MCP server list.
    pub fn servers(mut self, servers: Vec<McpServerConfig>) -> Self {
        self.servers = servers;
        self
    }

    /// Builds the provider.
    pub fn build(self) -> Result<McpToolProvider, BoxError> {
        let name = self
            .name
            .unwrap_or_else(|| DEFAULT_MCP_TOOL_PREFIX.to_string());
        let name = name.to_ascii_lowercase();
        validate_function_name(&name)?;

        let tool_prefix = self
            .tool_prefix
            .unwrap_or_else(|| DEFAULT_MCP_TOOL_PREFIX.to_string());
        let tool_prefix = sanitize_name_part(&tool_prefix);
        validate_function_name(&tool_prefix)?;

        let mut servers = BTreeMap::new();
        let mut sanitized_ids = BTreeSet::new();
        for server in self.servers {
            server.validate()?;
            if servers.contains_key(&server.id) {
                return Err(format!("duplicate MCP server id {}", server.id).into());
            }
            // Distinct ids that normalize to the same part would produce
            // colliding local tool names across servers.
            let sanitized = sanitize_name_part(&server.id);
            if !sanitized_ids.insert(sanitized.clone()) {
                return Err(format!(
                    "MCP server id {} collides with another server after normalization to {}",
                    server.id, sanitized
                )
                .into());
            }
            servers.insert(server.id.clone(), Arc::new(server));
        }

        let connect_locks = servers
            .keys()
            .map(|id| (id.clone(), Arc::new(Mutex::new(()))))
            .collect();

        Ok(McpToolProvider {
            inner: Arc::new(McpToolProviderInner {
                name,
                tool_prefix,
                servers: RwLock::new(servers),
                connect_locks: RwLock::new(connect_locks),
                index: RwLock::new(McpToolIndex::default()),
            }),
        })
    }
}

struct McpToolProviderInner {
    name: String,
    tool_prefix: String,
    servers: RwLock<BTreeMap<String, Arc<McpServerConfig>>>,
    /// Per-server lock serializing connection establishment, so concurrent
    /// callers racing to (re)connect a server don't spawn duplicate sessions.
    connect_locks: RwLock<BTreeMap<String, Arc<Mutex<()>>>>,
    index: RwLock<McpToolIndex>,
}

#[derive(Default)]
struct McpToolIndex {
    routes: BTreeMap<String, McpToolRoute>,
    sessions: BTreeMap<String, Arc<McpSession>>,
}

impl McpToolIndex {
    fn replace_server_routes(&mut self, server_id: &str, routes: Vec<McpToolRoute>) {
        self.routes.retain(|_, route| route.server_id != server_id);
        for route in routes {
            self.routes.insert(route.name.clone(), route);
        }
    }

    fn remove_server(&mut self, server_id: &str) {
        self.routes.retain(|_, route| route.server_id != server_id);
        self.sessions.remove(server_id);
    }
}

/// One Anda-facing route to an MCP tool.
#[derive(Debug, Clone)]
pub struct McpToolRoute {
    /// Anda-facing tool name.
    pub name: String,
    /// Configured MCP server id.
    pub server_id: String,
    /// Original MCP tool name.
    pub remote_name: String,
    /// Model-facing function definition.
    pub definition: FunctionDefinition,
}

struct McpSession {
    service: Mutex<RunningService<RoleClient, AndaMcpClient>>,
    dirty: Arc<AtomicBool>,
}

impl McpSession {
    async fn is_closed(&self) -> bool {
        self.service.lock().await.is_closed()
    }
}

#[derive(Debug, Clone)]
struct AndaMcpClient {
    info: ClientInfo,
    dirty: Arc<AtomicBool>,
}

impl AndaMcpClient {
    fn new(dirty: Arc<AtomicBool>) -> Self {
        let mut info = ClientInfo::default();
        info.client_info = Implementation::new("anda_engine", env!("CARGO_PKG_VERSION"))
            .with_title("Anda Engine MCP Host");
        Self { info, dirty }
    }
}

impl ClientHandler for AndaMcpClient {
    fn get_info(&self) -> ClientInfo {
        self.info.clone()
    }

    fn list_roots(
        &self,
        _context: RequestContext<RoleClient>,
    ) -> impl Future<Output = Result<ListRootsResult, McpError>> + Send + '_ {
        std::future::ready(Err(McpError::method_not_found::<ListRootsRequestMethod>()))
    }

    fn on_tool_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.dirty.store(true, Ordering::SeqCst);
        std::future::ready(())
    }
}

/// MCP server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    /// Stable server id used in local tool names and audit output.
    pub id: String,
    /// Server transport.
    pub transport: McpTransportConfig,
    /// Optional remote tool allowlist. Empty means all tools except excluded.
    #[serde(default)]
    pub include: BTreeSet<String>,
    /// Optional remote tool denylist.
    #[serde(default)]
    pub exclude: BTreeSet<String>,
}

impl McpServerConfig {
    /// Creates a stdio server configuration.
    pub fn stdio(id: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            transport: McpTransportConfig::Stdio(McpStdioTransport {
                command: command.into(),
                ..Default::default()
            }),
            include: BTreeSet::new(),
            exclude: BTreeSet::new(),
        }
    }

    /// Creates a Streamable HTTP server configuration.
    pub fn streamable_http(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            transport: McpTransportConfig::StreamableHttp(McpStreamableHttpTransport {
                url: url.into(),
                ..Default::default()
            }),
            include: BTreeSet::new(),
            exclude: BTreeSet::new(),
        }
    }

    fn validate(&self) -> Result<(), BoxError> {
        validate_function_name(&sanitize_name_part(&self.id))?;
        if self.id.trim().is_empty() {
            return Err("MCP server id must not be empty".into());
        }
        self.transport.validate()
    }
}

/// MCP transport configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpTransportConfig {
    /// stdio child process transport.
    Stdio(McpStdioTransport),
    /// Streamable HTTP transport.
    StreamableHttp(McpStreamableHttpTransport),
}

impl McpTransportConfig {
    fn validate(&self) -> Result<(), BoxError> {
        match self {
            Self::Stdio(config) => config.validate(),
            Self::StreamableHttp(config) => config.validate(),
        }
    }
}

/// stdio child process transport configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct McpStdioTransport {
    /// Executable to spawn.
    pub command: String,
    /// Command arguments. These are passed without shell interpolation.
    #[serde(default)]
    pub args: Vec<String>,
    /// Additional environment variables.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Optional working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
}

impl McpStdioTransport {
    fn validate(&self) -> Result<(), BoxError> {
        if self.command.trim().is_empty() {
            return Err("MCP stdio command must not be empty".into());
        }
        Ok(())
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.command);
        command.args(&self.args);
        command.envs(&self.env);
        if let Some(cwd) = &self.cwd {
            command.current_dir(cwd);
        }
        command
    }
}

/// Streamable HTTP transport configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct McpStreamableHttpTransport {
    /// MCP endpoint URL.
    pub url: String,
    /// Bearer token value, without the `Bearer ` prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bearer_token: Option<String>,
    /// Custom HTTP headers sent with every request.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

impl McpStreamableHttpTransport {
    fn validate(&self) -> Result<(), BoxError> {
        if self.url.trim().is_empty() {
            return Err("MCP HTTP URL must not be empty".into());
        }
        Ok(())
    }

    fn transport_config(&self) -> Result<StreamableHttpClientTransportConfig, BoxError> {
        let mut headers = HashMap::new();
        for (name, value) in &self.headers {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes())?,
                HeaderValue::from_str(value)?,
            );
        }
        let mut config =
            StreamableHttpClientTransportConfig::with_uri(self.url.clone()).custom_headers(headers);
        if let Some(token) = self
            .bearer_token
            .as_ref()
            .map(|token| token.trim())
            .filter(|token| !token.is_empty())
        {
            config = config.auth_header(token.to_string());
        }
        Ok(config)
    }
}

fn mcp_result_to_tool_output(route: &McpToolRoute, result: CallToolResult) -> ToolOutput<Json> {
    let mut output = ToolOutput::new(json!({
        "server_id": route.server_id,
        "tool": route.remote_name,
        "structured_content": result.structured_content,
        "content": result.content,
        "_meta": result.meta,
    }));
    output.is_error = result.is_error;
    output.usage = Usage {
        requests: 1,
        ..Usage::default()
    };
    output
}

fn sanitize_name_part(input: &str) -> String {
    let mut out = String::new();
    let mut previous_underscore = false;
    for c in input.chars() {
        let c = c.to_ascii_lowercase();
        let valid = matches!(c, 'a'..='z' | '0'..='9');
        if valid {
            out.push(c);
            previous_underscore = false;
        } else if !previous_underscore {
            out.push('_');
            previous_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    let mut normalized = if trimmed.is_empty() {
        "x".to_string()
    } else {
        trimmed
    };
    if !normalized
        .chars()
        .next()
        .is_some_and(|c: char| c.is_ascii_lowercase())
    {
        normalized.insert(0, 'x');
    }
    normalized
}

fn shorten_with_hash(base: &str, key: &str) -> String {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let suffix = format!("{:08x}", hasher.finish() as u32);
    let max_prefix = 64usize.saturating_sub(suffix.len() + 1);
    let mut prefix = base.chars().take(max_prefix).collect::<String>();
    prefix = prefix.trim_end_matches('_').to_string();
    format!("{}_{}", prefix, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn tool(name: &'static str, description: &'static str) -> McpTool {
        McpTool::new(
            Cow::Borrowed(name),
            Cow::Borrowed(description),
            Arc::new(Map::from_iter([
                ("type".to_string(), json!("object")),
                ("properties".to_string(), json!({})),
            ])),
        )
    }

    #[test]
    fn sanitizes_and_bounds_tool_names() {
        let provider = McpToolProvider::builder()
            .server(McpServerConfig::stdio("GitHub-Prod", "server"))
            .build()
            .unwrap();

        let short = provider
            .local_tool_name("GitHub-Prod", "issues.get-by-id", None)
            .unwrap();
        assert_eq!(short, "mcp_github_prod_issues_get_by_id");

        let long = provider
            .local_tool_name("server", &"x".repeat(120), None)
            .unwrap();
        assert!(long.len() <= 64);
        validate_function_name(&long).unwrap();
    }

    #[test]
    fn include_exclude_filter_remote_tool_names() {
        let mut server = McpServerConfig::stdio("repo", "server");
        server.include.insert("allowed".to_string());
        server.exclude.insert("blocked".to_string());
        let provider = McpToolProvider::new(vec![server]).unwrap();

        assert!(provider.includes_tool("repo", "allowed"));
        assert!(!provider.includes_tool("repo", "other"));
        assert!(!provider.includes_tool("repo", "blocked"));
    }

    #[tokio::test]
    async fn refresh_tolerates_unreachable_servers_only_in_tolerant_mode() {
        let provider = McpToolProvider::new(vec![McpServerConfig::stdio(
            "down",
            "anda_nonexistent_mcp_command_xyz",
        )])
        .unwrap();

        // Tolerant mode (used by init/startup) succeeds and discovers nothing.
        provider.refresh_servers(true).await.unwrap();
        assert!(provider.routes().is_empty());

        // Strict mode (explicit refresh) reports the failing server.
        let err = provider.refresh_servers(false).await.unwrap_err();
        assert!(err.to_string().contains("down"));
    }

    #[tokio::test]
    async fn add_server_rolls_back_when_initial_refresh_fails() {
        let provider = McpToolProvider::new(Vec::new()).unwrap();
        let err = provider
            .add_server(McpServerConfig::stdio(
                "down",
                "anda_nonexistent_mcp_command_xyz",
            ))
            .await
            .unwrap_err();

        assert!(!err.to_string().is_empty());
        assert!(!provider.contains_server("down"));
        assert!(provider.server_ids().is_empty());
        assert!(provider.routes().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn add_server_discovers_and_calls_tools_at_runtime() {
        use std::os::unix::fs::PermissionsExt;

        let script_path = std::env::temp_dir().join(format!(
            "anda_fake_mcp_server_{}_{}",
            std::process::id(),
            "runtime_add"
        ));
        let script = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"initialize"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"fake","version":"1.0.0"}}}\n' "$id"
      ;;
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echoes input.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}}\n' "$id"
      ;;
    *"tools/call"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}}\n' "$id"
      ;;
  esac
done
"#;
        std::fs::write(&script_path, script).unwrap();
        let mut permissions = std::fs::metadata(&script_path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script_path, permissions).unwrap();

        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "runtime",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(provider.server_ids(), vec!["runtime".to_string()]);
        let routes = provider.routes();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].name, "mcp_runtime_echo");
        assert_eq!(routes[0].remote_name, "echo");

        let output = provider
            .call_route(
                routes[0].clone(),
                ToolInput::new("mcp_runtime_echo".to_string(), json!({"text": "hi"})),
            )
            .await
            .unwrap();
        assert_eq!(output.output["server_id"], "runtime");
        assert_eq!(output.output["tool"], "echo");
        assert_eq!(output.is_error, Some(false));

        let _ = std::fs::remove_file(script_path);
    }

    #[test]
    fn add_server_rejects_duplicate_and_normalized_colliding_ids() {
        let provider =
            McpToolProvider::new(vec![McpServerConfig::stdio("GitHub-Prod", "server")]).unwrap();

        let err = provider
            .insert_server(McpServerConfig::stdio("GitHub-Prod", "server"))
            .unwrap_err();
        assert!(err.to_string().contains("already exists"));

        let err = provider
            .insert_server(McpServerConfig::stdio("github_prod", "server"))
            .unwrap_err();
        assert!(err.to_string().contains("collides"));
    }

    #[test]
    fn rejects_server_ids_that_collide_after_normalization() {
        let err = McpToolProvider::new(vec![
            McpServerConfig::stdio("GitHub-Prod", "server"),
            McpServerConfig::stdio("github_prod", "server"),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("collides"));
    }

    #[test]
    fn converts_mcp_tools_to_function_definitions() {
        let provider =
            McpToolProvider::new(vec![McpServerConfig::stdio("repo", "server")]).unwrap();
        let routes = provider
            .routes_for_tools("repo", vec![tool("issues/get", "Fetch an issue")])
            .unwrap();

        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].name, "mcp_repo_issues_get");
        assert_eq!(routes[0].remote_name, "issues/get");
        assert!(routes[0].definition.description.contains("Fetch an issue"));
        assert_eq!(routes[0].definition.strict, Some(false));
    }

    #[test]
    fn converts_mcp_call_result_to_audited_output() {
        let route = McpToolRoute {
            name: "mcp_repo_echo".to_string(),
            server_id: "repo".to_string(),
            remote_name: "echo".to_string(),
            definition: FunctionDefinition::default(),
        };
        let result = CallToolResult::structured(json!({"ok": true}));

        let output = mcp_result_to_tool_output(&route, result);
        assert_eq!(output.is_error, Some(false));
        assert_eq!(output.usage.requests, 1);
        assert_eq!(output.output["server_id"], "repo");
        assert_eq!(output.output["tool"], "echo");
        assert_eq!(output.output["structured_content"], json!({"ok": true}));
    }
}
