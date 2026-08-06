//! MCP tool provider extension.
//!
//! This module makes Anda an MCP host/client for tool execution. It discovers
//! tools from configured MCP servers, maps them to legal Anda function names,
//! and dispatches calls back to the original MCP tool name. Each server is also
//! exposed as a [`ToolGroup`] (carrying its title and `instructions` from the
//! handshake) so the discovery layer can present a server's tools as a coherent
//! capability bundle. It intentionally does not expose deprecated MCP client
//! utility capabilities such as Roots, Sampling, or Logging control.
//!
//! # Protocol revisions
//!
//! The host speaks MCP `2026-07-28` and the older `initialize`-based revisions.
//! `2026-07-28` made the protocol stateless: there is no `initialize` handshake
//! and no session header, servers advertise themselves through `server/discover`,
//! and `notifications/tools/list_changed` only reaches a client that opened a
//! `subscriptions/listen` stream for it. [`McpLifecycle`] selects how a server is
//! approached; the default probes the modern lifecycle and falls back to the
//! legacy handshake, so both server generations work unchanged.
//!
//! Two `2026-07-28` response shapes replace what used to be a plain result:
//!
//! - MRTR (SEP-2322) `input_required` rounds. This host advertises neither
//!   Sampling, Elicitation, nor Roots, so a round that genuinely asks for input
//!   comes back as a tool-level error; a round that only carries `requestState`
//!   is echoed back and the call continues.
//! - Tasks (SEP-2663). Opt in per server with [`McpTasksConfig`]; the provider
//!   then polls `tasks/get` until the task finishes, so a long-running tool no
//!   longer has to hold its response open. Without the opt-in the extension is
//!   not declared and servers must answer inline.
//!
//! # Authentication
//!
//! Streamable HTTP servers can authenticate with a static bearer token
//! ([`McpStreamableHttpTransport::bearer_token`]) or via OAuth 2.1
//! ([`McpOAuthConfig`]). Two OAuth flows are supported side by side:
//!
//! - [`McpOAuthConfig::ClientCredentials`] — headless server-to-server auth,
//!   obtained automatically when a session is established.
//! - [`McpOAuthConfig::AuthorizationCode`] — interactive, browser-based auth.
//!   As a library, this module only drives the protocol: it returns the
//!   authorization URL from [`McpToolProvider::begin_authorization`] and
//!   consumes the redirect via [`McpToolProvider::complete_authorization`]. The
//!   consuming application owns the browser, the redirect callback, and — via
//!   [`McpCredentialStore`] — where tokens are persisted.
//!
//! The browser does not have to run on the machine hosting the engine. On a
//! headless server (a user attached over SSH, say), present the authorization
//! URL as text; the user opens it in their local browser and either tunnels the
//! loopback redirect back with `ssh -L`, or simply pastes the final redirect URL
//! — code and state are in its query string — into the conversation for
//! [`complete_authorization`]. No listener is required for the paste variant.
//!
//! Re-authorization needs no special mode: [`begin_authorization`] can run at
//! any time, valid token or not, and completing the flow persists the new grant
//! and drops the live session so the next call uses it. To force a from-scratch
//! consent instead of a silent refresh, call
//! [`McpToolProvider::clear_credentials`] first.
//!
//! [`begin_authorization`]: McpToolProvider::begin_authorization
//! [`complete_authorization`]: McpToolProvider::complete_authorization
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
    BoxError, BoxFut, FunctionDefinition, Json, ToolGroup, ToolInput, ToolOutput, ToolProvider,
    validate_function_name,
};
use parking_lot::{Mutex as SyncMutex, RwLock};
use reqwest::Client as ReqwestClient;
use rmcp::{
    Peer, RoleClient,
    model::{CallToolRequestParams, ServerPeerInfo, Tool as McpTool},
    serve_client_with_lifecycle,
    service::ClientLifecycleMode,
    transport::{
        AuthClient, AuthorizationManager, StreamableHttpClientTransport, TokioChildProcess,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Map;

/// Re-exported from `rmcp`: the OAuth credentials an [`McpCredentialStore`]
/// persists on behalf of a server. Carries the (possibly dynamically
/// registered) `client_id` and the token response including the refresh token.
pub use rmcp::transport::StoredCredentials;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::sync::Mutex;

use crate::context::BaseCtx;

mod auth;
mod router;
mod session;

pub use auth::{
    InMemoryMcpCredentialStore, McpAuthorizationRequired, McpCredentialStore, McpOAuthConfig,
    McpOAuthMetadata, OAuthAuthorizationCodeConfig, OAuthClientCredentialsConfig,
};
pub use router::McpToolRoute;
pub use session::{McpStdioTransport, McpStreamableHttpTransport, McpTransportConfig};

use auth::{ScopedCredentialStore, authorization_required_hint, is_authorization_error};
use router::{
    DEFAULT_TASK_MAX_WAIT_SECS, MAX_LOCAL_NAME_ATTEMPTS, MAX_TASK_MAX_WAIT_SECS, call_tool_rounds,
    mcp_result_to_tool_output, sanitize_name_part, shorten_with_hash,
};
use session::{
    AndaMcpClient, DISCOVERY_PROBE_TIMEOUT, McpSession, SUBSCRIPTION_ACK_TIMEOUT,
    legacy_protocol_version, needs_tool_subscription, preferred_protocol_versions,
    pump_tool_subscription, serve_bounded, tool_subscription_filter,
};

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

    /// Registers a server configuration without connecting to it.
    ///
    /// This is the entry point for the interactive OAuth Authorization Code
    /// flow, where a session cannot be established until authorization has
    /// completed. The usual sequence is: `register_server` →
    /// [`begin_authorization`] → (user authorizes) →
    /// [`complete_authorization`] → [`refresh_server`]. For servers that need no
    /// interactive authorization (stdio, static bearer, or client credentials),
    /// prefer [`add_server`], which registers *and* connects in one step.
    ///
    /// [`add_server`]: Self::add_server
    /// [`begin_authorization`]: Self::begin_authorization
    /// [`complete_authorization`]: Self::complete_authorization
    /// [`refresh_server`]: Self::refresh_server
    pub fn register_server(&self, server: McpServerConfig) -> Result<(), BoxError> {
        self.insert_server(server)
    }

    /// Removes a server along with its session, routes, and any pending
    /// authorization state. Returns whether the server had been registered.
    ///
    /// Persisted OAuth credentials in the [`McpCredentialStore`] are left intact;
    /// clear them separately (`store.clear(server_id)`) if removal should also
    /// drop stored access and refresh tokens.
    pub fn remove_server(&self, server_id: &str) -> bool {
        let existed = self.contains_server(server_id);
        self.remove_server_state(server_id);
        existed
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
        self.refresh_server_inner(server_id, true).await
    }

    /// Refreshes one server after optionally clearing a pre-existing dirty notification.
    ///
    /// A caller that already claimed `dirty` with `swap(false)` must pass `false`: clearing it
    /// again would erase a second notification delivered between the claim and `tools/list`.
    async fn refresh_server_inner(
        &self,
        server_id: &str,
        clear_dirty: bool,
    ) -> Result<(), BoxError> {
        let config = self.server_config(server_id)?;
        let session = self.ensure_session(&config).await?;
        let peer = {
            let service = session.service.lock().await;
            service.peer().clone()
        };
        // Clear `dirty` *before* listing, not after. `on_tool_list_changed` runs on the rmcp
        // service task and can fire while `list_all_tools` is in flight; clearing afterwards
        // would swallow that notification, and since the server will not re-announce an
        // already-sent change, the route table would stay stale indefinitely. Clearing first
        // means a concurrent change re-arms the flag and is picked up by the next refresh.
        if clear_dirty {
            session.dirty.store(false, Ordering::SeqCst);
            // SEP-2549 lets a server declare `tools/list` fresh for `ttlMs`, which rmcp
            // honors with a client-side response cache. An explicit refresh promises a
            // live listing, so drop the cache first. The notification-driven path keeps
            // it, since rmcp already invalidates the tool cache on `tools/list_changed`.
            peer.clear_response_cache().await;
        }

        // Capture the server's self-description (title, instructions) from the
        // initialize handshake so the discovery layer can present each server as
        // a coherent capability bundle, not just a flat list of tools.
        let meta = McpServerMeta::from_peer_info(&config.id, peer.peer_info().as_deref());
        let tools = match peer.list_all_tools().await {
            Ok(tools) => tools,
            Err(err) => {
                // The listing failed, so the snapshot was not applied; re-arm the flag so
                // the next call retries instead of trusting a stale route table.
                session.dirty.store(true, Ordering::SeqCst);
                return Err(err.into());
            }
        };

        let routes = self.routes_for_tools(&config.id, tools)?;
        {
            let mut index = self.inner.index.write();
            index.replace_server_routes(&config.id, routes);
            index.metas.insert(config.id.clone(), meta);
        }
        Ok(())
    }

    /// Returns one [`ToolGroup`] per configured server that currently exposes
    /// tools, bundling the server's tools together with its title and
    /// `instructions` from the MCP `initialize` handshake.
    pub fn tool_groups(&self) -> Vec<ToolGroup> {
        let index = self.inner.index.read();
        let mut members: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for route in index.routes.values() {
            members
                .entry(route.server_id.clone())
                .or_default()
                .push(route.name.clone());
        }

        members
            .into_iter()
            .map(|(server_id, mut names)| {
                names.sort();
                let meta = index.metas.get(&server_id).cloned().unwrap_or_default();
                ToolGroup {
                    id: format!("{}:{}", self.inner.name, server_id),
                    title: meta.resolved_title(&server_id),
                    description: meta.resolved_description(&server_id),
                    instructions: meta.instructions,
                    members: names,
                }
            })
            .collect()
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

        let attempt = match self.connect(config, config.lifecycle.into_mode()).await {
            Ok(session) => Ok(session),
            // A pre-2026-07-28 server can answer the `server/discover` opener with
            // something other than a JSON-RPC "method not found" — an HTTP error, or,
            // over stdio, by rejecting the message and exiting — which rmcp's in-band
            // fallback cannot recover from because the transport itself is gone. Retry
            // once on a fresh transport with the legacy handshake. A failed
            // authorization is not a lifecycle problem, so it is reported as-is.
            Err(err)
                if config.lifecycle == McpLifecycle::Auto
                    && !is_authorization_error(err.as_ref()) =>
            {
                log::info!(
                    "MCP server {}: discovery lifecycle failed ({err}); retrying with the legacy initialize handshake",
                    config.id
                );
                self.connect(config, ClientLifecycleMode::Initialize).await
            }
            Err(err) => Err(err),
        };
        let session = attempt.map_err(|err| authorization_required_hint(config, err))?;

        self.inner
            .index
            .write()
            .sessions
            .insert(config.id.clone(), session.clone());
        Ok(session)
    }

    /// Establishes one session over a freshly built transport.
    ///
    /// Each attempt needs its own transport: an stdio child that refused the
    /// opener has already exited, and a streamable-HTTP worker binds its lifecycle
    /// mode when it sends the first message.
    async fn connect(
        &self,
        config: &McpServerConfig,
        lifecycle: ClientLifecycleMode,
    ) -> Result<Arc<McpSession>, BoxError> {
        let dirty = Arc::new(AtomicBool::new(false));
        let handler = AndaMcpClient::new(dirty.clone(), config.tasks.is_some());
        // Only a discovery opener can go unanswered by a server that does not know
        // it; the legacy handshake is answered or refused by every MCP server.
        let probe_timeout = (!matches!(lifecycle, ClientLifecycleMode::Initialize))
            .then_some(DISCOVERY_PROBE_TIMEOUT);
        let mut expires_at = None;
        let service = match &config.transport {
            McpTransportConfig::Stdio(stdio) => {
                let transport = TokioChildProcess::new(stdio.command())?;
                serve_bounded(
                    serve_client_with_lifecycle(handler, transport, lifecycle),
                    probe_timeout,
                )
                .await?
            }
            McpTransportConfig::StreamableHttp(http) => match &http.auth {
                None => {
                    let transport =
                        StreamableHttpClientTransport::from_config(http.transport_config()?);
                    serve_bounded(
                        serve_client_with_lifecycle(handler, transport, lifecycle),
                        probe_timeout,
                    )
                    .await?
                }
                Some(McpOAuthConfig::ClientCredentials(cc)) => {
                    // Headless: obtain a token at connection time, no human loop. The grant
                    // issues no refresh token, so the session carries the token's deadline
                    // and reconnects to mint a new one.
                    let (manager, deadline) =
                        auth::authorize_client_credentials(http.url.as_str(), cc).await?;
                    expires_at = deadline;
                    let transport = StreamableHttpClientTransport::with_client(
                        AuthClient::new(ReqwestClient::new(), manager),
                        http.base_transport_config()?,
                    );
                    serve_bounded(
                        serve_client_with_lifecycle(handler, transport, lifecycle),
                        probe_timeout,
                    )
                    .await?
                }
                Some(McpOAuthConfig::AuthorizationCode(_)) => {
                    // Interactive: reuse credentials persisted by a prior
                    // begin/complete_authorization; refresh happens on demand.
                    let manager = auth::authorize_from_store(
                        &config.id,
                        http.url.as_str(),
                        self.scoped_store(&config.id),
                    )
                    .await?;
                    let transport = StreamableHttpClientTransport::with_client(
                        AuthClient::new(ReqwestClient::new(), manager),
                        http.base_transport_config()?,
                    );
                    serve_bounded(
                        serve_client_with_lifecycle(handler, transport, lifecycle),
                        probe_timeout,
                    )
                    .await?
                }
            },
        };

        let subscription = self
            .subscribe_tool_changes(&config.id, service.peer(), dirty.clone())
            .await;
        Ok(Arc::new(McpSession {
            service: Mutex::new(service),
            dirty,
            expires_at,
            subscription,
        }))
    }

    /// Opens the `tools/list_changed` stream on a peer that requires one.
    ///
    /// SEP-2575 removed the unsolicited server-push channel: on `2026-07-28` a
    /// list-changed notification is only delivered on a `subscriptions/listen`
    /// stream the client asked for, so without this the route table would silently
    /// go stale. Older peers keep pushing the notification into the handler
    /// callback and need no stream. A failure here costs live updates, not the
    /// session, so it is logged rather than propagated.
    async fn subscribe_tool_changes(
        &self,
        server_id: &str,
        peer: &Peer<RoleClient>,
        dirty: Arc<AtomicBool>,
    ) -> Option<tokio::task::JoinHandle<()>> {
        if !needs_tool_subscription(peer.peer_info().as_deref()) {
            return None;
        }

        // A server that accepts the request but never acknowledges it must not wedge
        // session establishment, so the wait is bounded like the discovery probe.
        match tokio::time::timeout(
            SUBSCRIPTION_ACK_TIMEOUT,
            peer.listen(tool_subscription_filter()),
        )
        .await
        {
            Ok(Ok(subscription)) => {
                let peer = peer.clone();
                let server_id = server_id.to_string();
                Some(tokio::spawn(pump_tool_subscription(
                    server_id,
                    peer,
                    subscription,
                    dirty,
                )))
            }
            Ok(Err(err)) => {
                log::warn!(
                    "MCP server {server_id}: could not subscribe to tools/list_changed: {err}"
                );
                None
            }
            Err(_) => {
                log::warn!(
                    "MCP server {server_id}: tools/list_changed subscription was not acknowledged \
                     within {}s",
                    SUBSCRIPTION_ACK_TIMEOUT.as_secs()
                );
                None
            }
        }
    }

    /// Probes an HTTP MCP endpoint to determine whether it requires OAuth.
    ///
    /// Performs RFC 9728 protected-resource / RFC 8414 authorization-server
    /// discovery against `url`. Returns `None` when the endpoint advertises no
    /// OAuth support (it uses a static bearer token or no auth), or `Some` with
    /// the discovered metadata otherwise. A consuming application can use this to
    /// decide, from a bare URL, whether to connect directly or run the
    /// authorization flow.
    pub async fn discover_http_oauth(url: &str) -> Result<Option<McpOAuthMetadata>, BoxError> {
        auth::discover_http_oauth(url).await
    }

    /// Starts the interactive OAuth Authorization Code flow for `server_id` and
    /// returns the authorization URL to open in a browser.
    ///
    /// `anda_engine` is a library: it does not open the browser or receive the
    /// redirect. The consuming application presents the URL however it likes
    /// (loopback server, a route it hosts, or manual paste), then hands the
    /// resulting redirect URL to [`Self::complete_authorization`]. The
    /// intermediate PKCE/CSRF state is kept in memory on this provider instance,
    /// so both calls must run against the same instance in the same process.
    ///
    /// The server must be registered and configured with
    /// [`McpOAuthConfig::AuthorizationCode`]; otherwise this returns an error.
    pub async fn begin_authorization(&self, server_id: &str) -> Result<String, BoxError> {
        let config = self.server_config(server_id)?;
        let McpTransportConfig::StreamableHttp(http) = &config.transport else {
            return Err(format!("MCP server {server_id} does not use the HTTP transport").into());
        };
        let Some(McpOAuthConfig::AuthorizationCode(ac)) = &http.auth else {
            return Err(format!(
                "MCP server {server_id} is not configured for the OAuth authorization_code flow"
            )
            .into());
        };

        let (manager, auth_url) =
            auth::begin_authorization_manager(http.url.as_str(), ac, self.scoped_store(server_id))
                .await?;

        self.inner
            .pending_auth
            .lock()
            .insert(server_id.to_string(), manager);
        Ok(auth_url)
    }

    /// Completes the interactive OAuth Authorization Code flow started by
    /// [`Self::begin_authorization`], using the full redirect URL the
    /// authorization server sent back (carrying `code`, `state`, and optionally
    /// RFC 9207 `iss`).
    ///
    /// On success the resulting credentials — including the refresh token — are
    /// persisted through the configured [`McpCredentialStore`], so subsequent
    /// sessions establish without further interaction. The pending in-memory
    /// state is consumed whether or not the exchange succeeds; on failure, call
    /// [`Self::begin_authorization`] again.
    ///
    /// Any live session for the server is dropped afterwards: a session pins the
    /// token it connected with, so without the drop, freshly granted credentials
    /// (a re-authorization for new scopes, for example) would not take effect
    /// until the old session happened to die. The next refresh or tool call
    /// reconnects with the new credentials.
    pub async fn complete_authorization(
        &self,
        server_id: &str,
        redirect_url: &str,
    ) -> Result<(), BoxError> {
        let manager = self
            .inner
            .pending_auth
            .lock()
            .remove(server_id)
            .ok_or_else(|| format!("no pending OAuth authorization for MCP server {server_id}"))?;

        auth::complete_authorization_exchange(manager, redirect_url).await?;
        self.disconnect_server(server_id).await;
        Ok(())
    }

    /// Discards any pending interactive-authorization state for `server_id`.
    ///
    /// Returns whether a pending flow was actually cancelled.
    pub fn cancel_authorization(&self, server_id: &str) -> bool {
        self.inner.pending_auth.lock().remove(server_id).is_some()
    }

    /// Drops the cached session for `server_id`, keeping the server registered
    /// and its discovered routes intact. Returns whether a session existed.
    ///
    /// The next refresh or tool call re-establishes the session from scratch,
    /// re-running credential acquisition against the [`McpCredentialStore`] (or
    /// the Client Credentials exchange). In-flight tool calls on the old session
    /// finish undisturbed; the connection is torn down once they complete.
    pub async fn disconnect_server(&self, server_id: &str) -> bool {
        // Hold the per-server connect lock while dropping the session. `ensure_session`
        // keeps that lock across the handshake *and* the insert, so without it a
        // reconnect already in flight would install its session — built with the
        // credentials this call means to retire — right after the removal.
        let connect_lock = self.inner.connect_locks.read().get(server_id).cloned();
        let _guard = match &connect_lock {
            Some(lock) => Some(lock.lock().await),
            None => None,
        };
        self.inner
            .index
            .write()
            .sessions
            .remove(server_id)
            .is_some()
    }

    /// Removes the persisted OAuth credentials for `server_id` and drops its
    /// session, forcing the next connection to start from a clean slate.
    ///
    /// For an Authorization Code server this is the "sign out / re-consent"
    /// primitive: the next session attempt fails with
    /// [`McpAuthorizationRequired`] until [`begin_authorization`] /
    /// [`complete_authorization`] run again — regardless of whether the
    /// discarded access token was still valid.
    ///
    /// [`begin_authorization`]: Self::begin_authorization
    /// [`complete_authorization`]: Self::complete_authorization
    pub async fn clear_credentials(&self, server_id: &str) -> Result<(), BoxError> {
        self.inner.credential_store.clear(server_id).await?;
        self.disconnect_server(server_id).await;
        Ok(())
    }

    fn scoped_store(&self, server_id: &str) -> ScopedCredentialStore {
        ScopedCredentialStore {
            server_id: server_id.to_string(),
            inner: self.inner.credential_store.clone(),
        }
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
        if claimed && let Err(err) = self.refresh_server_inner(server_id, false).await {
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

            // Two remote names can sanitize or hash-truncate onto the same local name, and
            // the hash suffix uses a fixed-key hasher truncated to 32 bits, so a server
            // operator can compute a collision offline. Keep disambiguating until the name
            // is unique: silently reusing it would overwrite the earlier route below, so
            // the local name the model already learned would dispatch to a different remote
            // tool. If no unique name can be found, drop the tool rather than hijack.
            let mut local_name = self.local_tool_name(server_id, &remote_name, None)?;
            let mut attempt = 0usize;
            while used.contains(&local_name) {
                if attempt >= MAX_LOCAL_NAME_ATTEMPTS {
                    log::warn!(
                        "skipping MCP tool {remote_name:?} on server {server_id:?}: could not derive a unique local name"
                    );
                    break;
                }
                let key = format!("{remote_name}#{attempt}");
                local_name = self.local_tool_name(server_id, &remote_name, Some(&key))?;
                attempt += 1;
            }
            if used.contains(&local_name) {
                continue;
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
        let result = call_tool_rounds(&route, &peer, params, config.tasks.as_ref()).await?;
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
        self.inner.pending_auth.lock().remove(server_id);
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

    fn groups(&self) -> Vec<ToolGroup> {
        self.tool_groups()
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
#[derive(Default)]
pub struct McpToolProviderBuilder {
    name: Option<String>,
    tool_prefix: Option<String>,
    servers: Vec<McpServerConfig>,
    credential_store: Option<Arc<dyn McpCredentialStore>>,
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

    /// Sets the persistence backend for OAuth credentials. Defaults to an
    /// in-memory store ([`InMemoryMcpCredentialStore`]) that does not survive a
    /// process restart.
    pub fn credential_store(mut self, store: Arc<dyn McpCredentialStore>) -> Self {
        self.credential_store = Some(store);
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

        let credential_store = self
            .credential_store
            .unwrap_or_else(|| Arc::new(InMemoryMcpCredentialStore::new()));

        Ok(McpToolProvider {
            inner: Arc::new(McpToolProviderInner {
                name,
                tool_prefix,
                servers: RwLock::new(servers),
                connect_locks: RwLock::new(connect_locks),
                index: RwLock::new(McpToolIndex::default()),
                credential_store,
                pending_auth: SyncMutex::new(HashMap::new()),
            }),
        })
    }
}

impl std::fmt::Debug for McpToolProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolProviderBuilder")
            .field("name", &self.name)
            .field("tool_prefix", &self.tool_prefix)
            .field("servers", &self.servers)
            .field("credential_store", &self.credential_store.is_some())
            .finish()
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
    /// Application-supplied persistence for OAuth credentials, keyed by server.
    credential_store: Arc<dyn McpCredentialStore>,
    /// In-memory Authorization Code flow state (PKCE verifier + CSRF) held
    /// between `begin_authorization` and `complete_authorization`. This lives in
    /// the process only, so both calls must target the same provider instance.
    pending_auth: SyncMutex<HashMap<String, AuthorizationManager>>,
}

#[derive(Default)]
struct McpToolIndex {
    routes: BTreeMap<String, McpToolRoute>,
    sessions: BTreeMap<String, Arc<McpSession>>,
    metas: BTreeMap<String, McpServerMeta>,
}

impl McpToolIndex {
    fn replace_server_routes(&mut self, server_id: &str, routes: Vec<McpToolRoute>) {
        self.routes.retain(|_, route| route.server_id != server_id);
        for mut route in routes {
            // Cross-server local-name collision: another server already owns
            // this local name. Intra-server dedup in `routes_for_tools` cannot
            // see other servers, so disambiguate the newcomer here with a stable
            // hash suffix. Without this, a compromised server could publish a
            // tool whose local name shadows another server's, hijacking calls.
            if self
                .routes
                .get(&route.name)
                .is_some_and(|existing| existing.server_id != route.server_id)
            {
                let disambiguated = shorten_with_hash(
                    &route.name,
                    &format!("{}:{}", route.server_id, route.remote_name),
                );
                log::warn!(
                    "MCP local tool name collision on {:?}; remapping server {:?} tool {:?} to {:?}",
                    route.name,
                    route.server_id,
                    route.remote_name,
                    disambiguated
                );
                route.name = disambiguated.clone();
                route.definition.name = disambiguated;
            }

            // If the disambiguated name still collides with a different server,
            // drop the tool rather than silently hijack an existing route.
            if self
                .routes
                .get(&route.name)
                .is_some_and(|existing| existing.server_id != route.server_id)
            {
                log::error!(
                    "MCP tool {:?} from server {:?} dropped: local name {:?} still collides",
                    route.remote_name,
                    route.server_id,
                    route.name
                );
                continue;
            }

            self.routes.insert(route.name.clone(), route);
        }
    }

    fn remove_server(&mut self, server_id: &str) {
        self.routes.retain(|_, route| route.server_id != server_id);
        self.sessions.remove(server_id);
        self.metas.remove(server_id);
    }
}

/// Cached self-description of an MCP server captured during the `initialize`
/// handshake. Powers the per-server [`ToolGroup`] surfaced to the discovery
/// layer. All fields are untrusted remote metadata.
#[derive(Debug, Clone, Default)]
struct McpServerMeta {
    title: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
}

impl McpServerMeta {
    fn from_peer_info(server_id: &str, info: Option<&ServerPeerInfo>) -> Self {
        let Some(info) = info else {
            return Self::default();
        };
        let implementation = info.server_info.as_ref();
        let title = implementation
            .and_then(|implementation| {
                non_empty(implementation.title.as_deref())
                    .or_else(|| non_empty(Some(implementation.name.as_str())))
            })
            .filter(|title| title != server_id);
        Self {
            title,
            description: implementation
                .and_then(|implementation| non_empty(implementation.description.as_deref())),
            instructions: non_empty(info.instructions.as_deref()),
        }
    }

    fn resolved_title(&self, server_id: &str) -> String {
        self.title
            .clone()
            .unwrap_or_else(|| format!("MCP server `{server_id}`"))
    }

    fn resolved_description(&self, server_id: &str) -> String {
        self.description
            .clone()
            .unwrap_or_else(|| format!("Tools provided by MCP server `{server_id}`."))
    }
}

/// Returns the trimmed string when it carries non-whitespace content.
fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
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
    /// How the session negotiates the MCP protocol revision.
    #[serde(default)]
    pub lifecycle: McpLifecycle,
    /// SEP-2663 tasks extension. Omitted (the default) leaves the extension
    /// undeclared, so the server must answer `tools/call` inline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tasks: Option<McpTasksConfig>,
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
            lifecycle: McpLifecycle::default(),
            tasks: None,
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
            lifecycle: McpLifecycle::default(),
            tasks: None,
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

/// How a session negotiates the MCP protocol revision.
///
/// `2026-07-28` replaced the `initialize` handshake with a `server/discover`
/// probe, so which opener a client sends decides which revisions are reachable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum McpLifecycle {
    /// Probe `server/discover` first and fall back to the legacy handshake when
    /// the server does not implement it. Works with both server generations.
    #[default]
    Auto,
    /// Require the stateless `server/discover` lifecycle. Connecting fails on a
    /// server that predates `2026-07-28`.
    Discover,
    /// Use the legacy `initialize` handshake only, negotiating at most
    /// `2025-11-25`. Use it to pin a server that mishandles unknown methods.
    Initialize,
}

impl McpLifecycle {
    fn into_mode(self) -> ClientLifecycleMode {
        match self {
            Self::Auto => ClientLifecycleMode::Auto {
                preferred_versions: preferred_protocol_versions(),
                legacy_version: Some(legacy_protocol_version()),
            },
            Self::Discover => ClientLifecycleMode::Discover {
                preferred_versions: preferred_protocol_versions(),
            },
            Self::Initialize => ClientLifecycleMode::Initialize,
        }
    }
}

/// SEP-2663 tasks extension settings for one MCP server.
///
/// Declaring the extension tells the server it may answer a `tools/call` with a
/// task handle instead of a result. The provider then polls `tasks/get` until the
/// task reaches a terminal state, so a long-running tool does not have to hold its
/// response open for the whole run. The tool call still blocks until the task
/// finishes or [`max_wait_secs`] elapses.
///
/// [`max_wait_secs`]: Self::max_wait_secs
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpTasksConfig {
    /// Longest one tool call waits for a task to finish, in seconds. Defaults to
    /// 300, and is clamped to at least 1 second and at most a day. On timeout the
    /// task is cancelled best-effort and the call fails.
    #[serde(default = "default_task_max_wait_secs")]
    pub max_wait_secs: u64,
}

fn default_task_max_wait_secs() -> u64 {
    DEFAULT_TASK_MAX_WAIT_SECS
}

impl Default for McpTasksConfig {
    fn default() -> Self {
        Self {
            max_wait_secs: DEFAULT_TASK_MAX_WAIT_SECS,
        }
    }
}

impl McpTasksConfig {
    /// The configured wait, clamped so a deserialized value cannot overflow the
    /// poll deadline (`Instant + Duration` panics on overflow).
    fn max_wait(&self) -> Duration {
        Duration::from_secs(self.max_wait_secs.clamp(1, MAX_TASK_MAX_WAIT_SECS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{CallToolResult, InputRequiredResult, ProtocolVersion};
    use rmcp::transport::AuthError;
    use serde_json::json;
    use std::borrow::Cow;
    use std::path::PathBuf;

    use super::auth::client_credentials_deadline;
    use super::router::{
        TASK_POLL_INTERVAL, TASK_POLL_INTERVAL_MAX, TASK_POLL_INTERVAL_MIN, input_required_error,
        task_poll_interval,
    };
    use std::time::Instant;

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
        // A well-behaved pre-2026 server: it refuses the discovery probe with
        // "method not found", which rmcp answers by falling back to `initialize`
        // on the same transport.
        let script = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"Method not found"}}\n' "$id"
      ;;
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

    #[cfg(unix)]
    #[tokio::test]
    async fn captures_server_metadata_into_a_tool_group() {
        use std::os::unix::fs::PermissionsExt;

        let script_path = std::env::temp_dir().join(format!(
            "anda_fake_mcp_server_{}_{}",
            std::process::id(),
            "group_meta"
        ));
        // The initialize result advertises a server title and instructions; the
        // tools/list response exposes two tools that should bundle into one
        // group keyed by the configured server id.
        let script = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"Method not found"}}\n' "$id"
      ;;
    *"initialize"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"fs","title":"Filesystem","version":"1.0.0"},"instructions":"Call list_dir before read_file."}}\n' "$id"
      ;;
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"read_file","description":"Read a file.","inputSchema":{"type":"object","properties":{}}},{"name":"list_dir","description":"List a directory.","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
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
                "files",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();

        let groups = provider.tool_groups();
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.id, "mcp:files");
        assert_eq!(group.title, "Filesystem");
        assert_eq!(
            group.instructions.as_deref(),
            Some("Call list_dir before read_file.")
        );
        // Members list every tool the server exposes so the model can pull in
        // siblings after discovering one of them.
        assert_eq!(
            group.members,
            vec![
                "mcp_files_list_dir".to_string(),
                "mcp_files_read_file".to_string()
            ]
        );

        let _ = std::fs::remove_file(script_path);
    }

    #[test]
    fn server_meta_falls_back_to_server_id_without_peer_info() {
        let meta = McpServerMeta::from_peer_info("files", None);
        assert!(meta.title.is_none());
        assert!(meta.instructions.is_none());
        assert_eq!(meta.resolved_title("files"), "MCP server `files`");
        assert_eq!(
            meta.resolved_description("files"),
            "Tools provided by MCP server `files`."
        );
    }

    #[test]
    fn tool_groups_is_empty_without_discovered_routes() {
        let provider =
            McpToolProvider::new(vec![McpServerConfig::stdio("files", "server")]).unwrap();
        assert!(provider.tool_groups().is_empty());
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
    fn cross_server_local_name_collision_is_disambiguated() {
        let provider = McpToolProvider::new(vec![
            McpServerConfig::stdio("github", "server"),
            McpServerConfig::stdio("github_prod", "server"),
        ])
        .unwrap();

        // Both resolve to the same local name `mcp_github_prod_list_issues`.
        let a = provider
            .routes_for_tools("github", vec![tool("prod_list_issues", "A")])
            .unwrap();
        let b = provider
            .routes_for_tools("github_prod", vec![tool("list_issues", "B")])
            .unwrap();
        assert_eq!(a[0].name, "mcp_github_prod_list_issues");
        assert_eq!(a[0].name, b[0].name, "the two servers' local names collide");

        {
            let mut index = provider.inner.index.write();
            index.replace_server_routes("github", a);
            index.replace_server_routes("github_prod", b);
        }

        let routes = provider.routes();
        let github = routes
            .iter()
            .find(|route| route.server_id == "github")
            .expect("github route retained");
        let github_prod = routes
            .iter()
            .find(|route| route.server_id == "github_prod")
            .expect("github_prod route retained");

        // The first server keeps the un-suffixed name; the colliding newcomer is
        // remapped so its calls are never routed to the other server's tool.
        assert_eq!(github.name, "mcp_github_prod_list_issues");
        assert_ne!(github_prod.name, github.name);
        assert_eq!(github_prod.remote_name, "list_issues");
        assert_eq!(github_prod.definition.name, github_prod.name);
    }

    fn http_auth_code_server(id: &str, client_id: Option<&str>) -> McpServerConfig {
        let mut server = McpServerConfig::streamable_http(id, "https://example.com/mcp");
        if let McpTransportConfig::StreamableHttp(http) = &mut server.transport {
            http.auth = Some(McpOAuthConfig::AuthorizationCode(
                OAuthAuthorizationCodeConfig {
                    redirect_uri: "http://127.0.0.1:8080/callback".to_string(),
                    scopes: vec!["mcp:tools".to_string()],
                    client_name: Some("test".to_string()),
                    client_id: client_id.map(str::to_string),
                },
            ));
        }
        server
    }

    #[test]
    fn oauth_config_round_trips_through_serde() {
        let server = http_auth_code_server("gh", None);
        let json = serde_json::to_value(&server).unwrap();
        assert_eq!(json["transport"]["auth"]["flow"], "authorization_code");

        let parsed: McpServerConfig = serde_json::from_value(json).unwrap();
        let McpTransportConfig::StreamableHttp(http) = &parsed.transport else {
            panic!("expected streamable http transport");
        };
        match &http.auth {
            Some(McpOAuthConfig::AuthorizationCode(ac)) => {
                assert_eq!(ac.redirect_uri, "http://127.0.0.1:8080/callback");
                assert_eq!(ac.scopes, vec!["mcp:tools".to_string()]);
                assert!(ac.client_id.is_none());
            }
            other => panic!("unexpected auth config: {other:?}"),
        }
    }

    #[test]
    fn client_credentials_config_round_trips_through_serde() {
        let mut server = McpServerConfig::streamable_http("svc", "https://example.com/mcp");
        if let McpTransportConfig::StreamableHttp(http) = &mut server.transport {
            http.auth = Some(McpOAuthConfig::ClientCredentials(
                OAuthClientCredentialsConfig {
                    client_id: "cid".to_string(),
                    client_secret: "secret".to_string(),
                    scopes: vec!["a".to_string()],
                    resource: Some("https://api.example.com".to_string()),
                },
            ));
        }
        let json = serde_json::to_value(&server).unwrap();
        assert_eq!(json["transport"]["auth"]["flow"], "client_credentials");
        let parsed: McpServerConfig = serde_json::from_value(json).unwrap();
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn validate_rejects_bearer_token_combined_with_oauth() {
        let mut server = http_auth_code_server("gh", None);
        if let McpTransportConfig::StreamableHttp(http) = &mut server.transport {
            http.bearer_token = Some("tok".to_string());
        }
        let err = server.validate().unwrap_err().to_string();
        assert!(err.contains("cannot set both"), "{err}");
    }

    #[test]
    fn debug_output_redacts_secrets() {
        // A client secret must never appear in `Debug` output.
        let cfg = OAuthClientCredentialsConfig {
            client_id: "cid".to_string(),
            client_secret: "super-secret-value".to_string(),
            scopes: vec![],
            resource: None,
        };
        let rendered = format!("{cfg:?}");
        assert!(!rendered.contains("super-secret-value"), "{rendered}");
        assert!(rendered.contains("cid"), "{rendered}");
        assert!(rendered.contains("[REDACTED]"), "{rendered}");

        // Neither a static bearer token nor an embedded client secret must leak
        // through the full `McpServerConfig` -> transport `Debug` chain.
        let mut server = McpServerConfig::streamable_http("svc", "https://example.com/mcp");
        if let McpTransportConfig::StreamableHttp(http) = &mut server.transport {
            http.bearer_token = Some("super-secret-token".to_string());
        }
        let rendered = format!("{server:?}");
        assert!(!rendered.contains("super-secret-token"), "{rendered}");
        assert!(rendered.contains("[REDACTED]"), "{rendered}");

        let mut server = McpServerConfig::streamable_http("svc", "https://example.com/mcp");
        if let McpTransportConfig::StreamableHttp(http) = &mut server.transport {
            http.auth = Some(McpOAuthConfig::ClientCredentials(cfg));
        }
        let rendered = format!("{server:?}");
        assert!(!rendered.contains("super-secret-value"), "{rendered}");

        // The stdio `env` map is where a host application expands per-server secrets, so its
        // values must be redacted too. Keys stay visible for diagnostics.
        let mut server = McpServerConfig::stdio("files", "mcp-files");
        if let McpTransportConfig::Stdio(stdio) = &mut server.transport {
            stdio
                .env
                .insert("GITHUB_TOKEN".to_string(), "ghp_live_value".to_string());
        }
        let rendered = format!("{server:?}");
        assert!(!rendered.contains("ghp_live_value"), "{rendered}");
        assert!(rendered.contains("GITHUB_TOKEN"), "{rendered}");
        assert!(rendered.contains("[REDACTED]"), "{rendered}");
    }

    #[test]
    fn client_credentials_deadlines_preserve_short_token_lifetime_and_do_not_overflow() {
        let now = Instant::now();
        let after = |secs| {
            client_credentials_deadline(now, Duration::from_secs(secs))
                .unwrap()
                .duration_since(now)
        };

        assert_eq!(after(3_600), Duration::from_secs(3_480));
        assert_eq!(after(240), Duration::from_secs(120));
        assert_eq!(after(120), Duration::from_secs(60));
        // Stay just ahead of rmcp's 30-second refresh threshold without throwing away 90% of
        // a one-minute token's lifetime.
        assert_eq!(after(60), Duration::from_secs(29));
        assert_eq!(after(30), Duration::ZERO);

        assert!(
            client_credentials_deadline(now, Duration::MAX).is_none(),
            "an untrusted, unrepresentable expires_in must not panic"
        );
    }

    #[test]
    fn validate_rejects_incomplete_oauth_configs() {
        let mut server = http_auth_code_server("gh", None);
        if let McpTransportConfig::StreamableHttp(http) = &mut server.transport
            && let Some(McpOAuthConfig::AuthorizationCode(ac)) = &mut http.auth
        {
            ac.redirect_uri = "  ".to_string();
        }
        assert!(server.validate().is_err());

        let mut creds = McpServerConfig::streamable_http("svc", "https://example.com/mcp");
        if let McpTransportConfig::StreamableHttp(http) = &mut creds.transport {
            http.auth = Some(McpOAuthConfig::ClientCredentials(
                OAuthClientCredentialsConfig {
                    client_id: "cid".to_string(),
                    client_secret: String::new(),
                    scopes: vec![],
                    resource: None,
                },
            ));
        }
        assert!(creds.validate().is_err());
    }

    #[tokio::test]
    async fn begin_authorization_rejects_non_oauth_servers() {
        let provider = McpToolProvider::new(vec![
            McpServerConfig::stdio("cli", "server"),
            McpServerConfig::streamable_http("plain", "https://example.com/mcp"),
        ])
        .unwrap();

        let err = provider.begin_authorization("cli").await.unwrap_err();
        assert!(err.to_string().contains("does not use the HTTP transport"));

        let err = provider.begin_authorization("plain").await.unwrap_err();
        assert!(
            err.to_string()
                .contains("not configured for the OAuth authorization_code flow")
        );
    }

    #[tokio::test]
    async fn authorization_code_without_credentials_reports_auth_required() {
        // No network: an empty credential store short-circuits before discovery.
        let provider = McpToolProvider::new(vec![http_auth_code_server("gh", None)]).unwrap();
        let err = provider.refresh_server("gh").await.unwrap_err();
        let required = err
            .downcast_ref::<McpAuthorizationRequired>()
            .expect("expected McpAuthorizationRequired");
        assert_eq!(required.server_id, "gh");
    }

    #[tokio::test]
    async fn register_server_registers_without_connecting() {
        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .register_server(http_auth_code_server("gh", None))
            .unwrap();
        assert!(provider.contains_server("gh"));
        // No connection attempted, so no routes are discovered yet.
        assert!(provider.routes().is_empty());

        // Duplicate registration is rejected.
        assert!(
            provider
                .register_server(http_auth_code_server("gh", None))
                .is_err()
        );

        assert!(provider.remove_server("gh"));
        assert!(!provider.contains_server("gh"));
        assert!(!provider.remove_server("gh"));
    }

    #[tokio::test]
    async fn complete_and_cancel_authorization_without_pending_state() {
        let provider = McpToolProvider::new(vec![http_auth_code_server("gh", None)]).unwrap();
        assert!(!provider.cancel_authorization("gh"));
        let err = provider
            .complete_authorization("gh", "http://127.0.0.1:8080/callback?code=x&state=y")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no pending OAuth authorization"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn disconnect_drops_the_session_but_keeps_the_server_and_routes() {
        let script_path = write_fake_server("disconnect", DISCOVER_SERVER);
        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "stateless",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();
        assert!(
            provider
                .inner
                .index
                .read()
                .sessions
                .contains_key("stateless")
        );

        // Nothing to disconnect for an unknown id.
        assert!(!provider.disconnect_server("missing").await);

        assert!(provider.disconnect_server("stateless").await);
        assert!(!provider.disconnect_server("stateless").await);
        assert!(provider.inner.index.read().sessions.is_empty());
        // The server stays registered and its discovered tools stay routable…
        assert!(provider.contains_server("stateless"));
        let route = provider.routes().remove(0);

        // …and the next call transparently reconnects.
        let output = provider
            .call_route(
                route,
                ToolInput::new("mcp_stateless_echo".to_string(), json!({})),
            )
            .await
            .unwrap();
        assert_eq!(output.is_error, Some(false));
        assert!(
            provider
                .inner
                .index
                .read()
                .sessions
                .contains_key("stateless")
        );

        let _ = std::fs::remove_file(script_path);
    }

    #[tokio::test]
    async fn clear_credentials_forces_reauthorization() {
        let store = Arc::new(InMemoryMcpCredentialStore::new());
        let provider = McpToolProvider::builder()
            .server(http_auth_code_server("gh", None))
            .credential_store(store.clone())
            .build()
            .unwrap();

        let creds = StoredCredentials::new("client".to_string(), None, vec!["a".to_string()], None);
        store.save("gh", creds).await.unwrap();

        provider.clear_credentials("gh").await.unwrap();
        assert!(store.load("gh").await.unwrap().is_none());

        // With the stored grant gone, the next connection reports that the
        // interactive flow must run again.
        let err = provider.refresh_server("gh").await.unwrap_err();
        assert!(err.downcast_ref::<McpAuthorizationRequired>().is_some());
    }

    #[tokio::test]
    async fn in_memory_credential_store_round_trip() {
        let store = InMemoryMcpCredentialStore::new();
        assert!(store.load("gh").await.unwrap().is_none());

        let creds = StoredCredentials::new("client".to_string(), None, vec!["a".to_string()], None);
        store.save("gh", creds).await.unwrap();
        let loaded = store.load("gh").await.unwrap().expect("stored");
        assert_eq!(loaded.client_id, "client");

        store.clear("gh").await.unwrap();
        assert!(store.load("gh").await.unwrap().is_none());
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

    #[test]
    fn lifecycles_map_to_the_matching_rmcp_modes() {
        // `2026-07-28` is offered only through discovery, never proposed to the
        // legacy handshake, which a peer could echo back without implementing it.
        match McpLifecycle::Auto.into_mode() {
            ClientLifecycleMode::Auto {
                preferred_versions,
                legacy_version,
            } => {
                assert_eq!(preferred_versions[0], ProtocolVersion::V_2026_07_28);
                assert_eq!(legacy_version, Some(ProtocolVersion::V_2025_11_25));
            }
            other => panic!("unexpected lifecycle mode: {other:?}"),
        }
        match McpLifecycle::Discover.into_mode() {
            ClientLifecycleMode::Discover { preferred_versions } => {
                assert!(preferred_versions.contains(&ProtocolVersion::V_2026_07_28));
            }
            other => panic!("unexpected lifecycle mode: {other:?}"),
        }
        assert_eq!(
            McpLifecycle::Initialize.into_mode(),
            ClientLifecycleMode::Initialize
        );
        assert_eq!(
            AndaMcpClient::new(Arc::new(AtomicBool::new(false)), false)
                .info
                .protocol_version,
            ProtocolVersion::V_2025_11_25
        );
    }

    #[test]
    fn declares_the_tasks_extension_only_when_configured() {
        let dirty = Arc::new(AtomicBool::new(false));
        assert!(
            AndaMcpClient::new(dirty.clone(), false)
                .info
                .capabilities
                .extensions
                .is_none()
        );
        let capabilities = AndaMcpClient::new(dirty, true).info.capabilities;
        assert!(capabilities.supports_tasks());
    }

    #[test]
    fn tool_subscriptions_are_required_only_from_2026_07_28() {
        let peer_info = |version: &str, list_changed: bool| -> ServerPeerInfo {
            serde_json::from_value(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": list_changed}},
            }))
            .unwrap()
        };

        assert!(needs_tool_subscription(Some(&peer_info(
            "2026-07-28",
            true
        ))));
        // Older peers still push the notification without an opt-in stream.
        assert!(!needs_tool_subscription(Some(&peer_info(
            "2025-11-25",
            true
        ))));
        // Nothing to subscribe to when the server never announces changes.
        assert!(!needs_tool_subscription(Some(&peer_info(
            "2026-07-28",
            false
        ))));
        assert!(!needs_tool_subscription(None));
    }

    #[test]
    fn task_poll_intervals_clamp_untrusted_server_hints() {
        assert_eq!(task_poll_interval(None), TASK_POLL_INTERVAL);
        assert_eq!(task_poll_interval(Some(2_000)), Duration::from_secs(2));
        assert_eq!(task_poll_interval(Some(0)), TASK_POLL_INTERVAL_MIN);
        assert_eq!(task_poll_interval(Some(u64::MAX)), TASK_POLL_INTERVAL_MAX);
    }

    #[test]
    fn task_max_wait_is_clamped_so_the_deadline_cannot_overflow() {
        assert_eq!(
            McpTasksConfig::default().max_wait(),
            Duration::from_secs(DEFAULT_TASK_MAX_WAIT_SECS)
        );
        assert_eq!(
            McpTasksConfig { max_wait_secs: 0 }.max_wait(),
            Duration::from_secs(1)
        );

        // A configured value comes from deserialized config; `Instant + Duration`
        // panics on overflow, so the ceiling must hold before it is added.
        let huge = McpTasksConfig {
            max_wait_secs: u64::MAX,
        };
        assert_eq!(huge.max_wait(), Duration::from_secs(MAX_TASK_MAX_WAIT_SECS));
        let _ = Instant::now() + huge.max_wait();
    }

    #[tokio::test]
    async fn revoked_authorization_is_reported_as_authorization_required() {
        // `authorize_from_store` only raises `McpAuthorizationRequired` when nothing
        // is stored; a grant the authorization server later rejects surfaces as a
        // transport-level auth failure, which must map to the same signal so the
        // application knows to re-run the interactive flow.
        let config = http_auth_code_server("gh", None);
        let err: BoxError = AuthError::AuthorizationRequired.into();
        assert!(is_authorization_error(err.as_ref()));

        let mapped = authorization_required_hint(&config, err);
        let required = mapped
            .downcast_ref::<McpAuthorizationRequired>()
            .expect("expected McpAuthorizationRequired");
        assert_eq!(required.server_id, "gh");

        // A plain lifecycle failure is left alone so the fallback path still sees it.
        let other: BoxError = "transport closed".into();
        assert!(
            authorization_required_hint(&config, other)
                .downcast_ref::<McpAuthorizationRequired>()
                .is_none()
        );

        // Servers without the interactive flow keep their original error.
        let stdio = McpServerConfig::stdio("cli", "server");
        let err: BoxError = McpAuthorizationRequired {
            server_id: "other".to_string(),
        }
        .into();
        assert_eq!(
            authorization_required_hint(&stdio, err)
                .downcast_ref::<McpAuthorizationRequired>()
                .map(|required| required.server_id.clone()),
            Some("other".to_string())
        );
    }

    #[test]
    fn authorization_failures_are_not_treated_as_lifecycle_failures() {
        let err: BoxError = McpAuthorizationRequired {
            server_id: "gh".to_string(),
        }
        .into();
        assert!(is_authorization_error(err.as_ref()));

        let err: BoxError = "transport closed".into();
        assert!(!is_authorization_error(err.as_ref()));
    }

    #[test]
    fn unsupported_input_rounds_become_tool_level_errors() {
        let route = McpToolRoute {
            name: "mcp_repo_echo".to_string(),
            server_id: "repo".to_string(),
            remote_name: "echo".to_string(),
            definition: FunctionDefinition::default(),
        };
        let result: InputRequiredResult = serde_json::from_value(json!({
            "resultType": "input_required",
            "inputRequests": {"pick_root": {"method": "roots/list"}},
        }))
        .unwrap();

        let output = mcp_result_to_tool_output(&route, input_required_error(&route, &result));
        assert_eq!(output.is_error, Some(true));
        let rendered = output.output.to_string();
        assert!(rendered.contains("pick_root"), "{rendered}");
        assert!(rendered.contains("does not provide"), "{rendered}");
    }

    #[test]
    fn server_config_defaults_to_the_auto_lifecycle_without_tasks() {
        let parsed: McpServerConfig = serde_json::from_value(json!({
            "id": "files",
            "transport": {"type": "stdio", "command": "server"},
        }))
        .unwrap();
        assert_eq!(parsed.lifecycle, McpLifecycle::Auto);
        assert!(parsed.tasks.is_none());

        let mut server = McpServerConfig::stdio("files", "server");
        server.lifecycle = McpLifecycle::Discover;
        server.tasks = Some(McpTasksConfig::default());
        let json = serde_json::to_value(&server).unwrap();
        assert_eq!(json["lifecycle"], "discover");
        assert_eq!(json["tasks"]["max_wait_secs"], 300);

        let parsed: McpServerConfig = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.lifecycle, McpLifecycle::Discover);
        assert_eq!(
            parsed.tasks.map(|tasks| tasks.max_wait()),
            Some(Duration::from_secs(300))
        );
    }

    #[cfg(unix)]
    fn write_fake_server(name: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "anda_fake_mcp_server_{}_{name}",
            std::process::id()
        ));
        std::fs::write(&path, script).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    /// A stateless `2026-07-28` server: no `initialize`, self-description through
    /// `server/discover`, and results carrying the SEP-2322 `resultType`.
    #[cfg(unix)]
    const DISCOVER_SERVER: &str = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{"listChanged":false}},"instructions":"Call list_dir first.","ttlMs":0,"cacheScope":"private","_meta":{"io.modelcontextprotocol/serverInfo":{"name":"fs","title":"Stateless Files","version":"1.0.0"}}}}\n' "$id"
      ;;
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[{"name":"echo","description":"Echoes input.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}}\n' "$id"
      ;;
    *"tools/call"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","content":[{"type":"text","text":"ok"}],"isError":false}}\n' "$id"
      ;;
  esac
done
"#;

    #[cfg(unix)]
    #[tokio::test]
    async fn negotiates_the_2026_lifecycle_and_calls_tools() {
        let script_path = write_fake_server("discover", DISCOVER_SERVER);
        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "stateless",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();

        let routes = provider.routes();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].name, "mcp_stateless_echo");

        // Discovery metadata replaces the handshake as the source of group data.
        let groups = provider.tool_groups();
        assert_eq!(groups[0].title, "Stateless Files");
        assert_eq!(
            groups[0].instructions.as_deref(),
            Some("Call list_dir first.")
        );

        let output = provider
            .call_route(
                routes[0].clone(),
                ToolInput::new("mcp_stateless_echo".to_string(), json!({"text": "hi"})),
            )
            .await
            .unwrap();
        assert_eq!(output.is_error, Some(false));

        let _ = std::fs::remove_file(script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn falls_back_to_the_legacy_handshake_when_discovery_is_refused() {
        // Pre-2026 servers do not answer `server/discover` with a JSON-RPC "method
        // not found"; an stdio child simply rejects the message and exits, taking
        // the transport with it. Only a retry on a fresh transport recovers.
        let script_path = write_fake_server(
            "legacy_only",
            r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      exit 1
      ;;
    *"initialize"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"legacy","version":"1.0.0"}}}\n' "$id"
      ;;
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echoes input.","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      ;;
  esac
done
"#,
        );

        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "legacy",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(provider.routes().len(), 1);

        // Pinning the lifecycle skips the probe entirely.
        let mut pinned =
            McpServerConfig::stdio("pinned", script_path.to_string_lossy().to_string());
        pinned.lifecycle = McpLifecycle::Initialize;
        provider.add_server(pinned).await.unwrap();
        assert_eq!(provider.routes().len(), 2);

        let _ = std::fs::remove_file(script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn input_required_results_surface_as_failed_tool_calls() {
        let script_path = write_fake_server(
            "mrtr",
            r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{"listChanged":false}},"ttlMs":0,"cacheScope":"private"}}\n' "$id"
      ;;
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[{"name":"ask","description":"Asks first.","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      ;;
    *"tools/call"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"input_required","inputRequests":{"pick_root":{"method":"roots/list"}},"requestState":"opaque"}}\n' "$id"
      ;;
  esac
done
"#,
        );

        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "asker",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();

        let route = provider.routes().remove(0);
        let output = provider
            .call_route(
                route,
                ToolInput::new("mcp_asker_ask".to_string(), json!({})),
            )
            .await
            .unwrap();

        // The turn survives: the model sees a failed tool call, not a hard error.
        assert_eq!(output.is_error, Some(true));
        assert!(output.output.to_string().contains("pick_root"));

        let _ = std::fs::remove_file(script_path);
    }

    /// A `2026-07-28` server that answers `tools/call` with a SEP-2663 task
    /// handle, reports one `working` poll, then completes.
    #[cfg(unix)]
    const TASK_SERVER: &str = r#"#!/bin/sh
polls=0
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{"listChanged":false},"extensions":{"io.modelcontextprotocol/tasks":{}}},"ttlMs":0,"cacheScope":"private"}}\n' "$id"
      ;;
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[{"name":"slow","description":"Takes a while.","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      ;;
    *"tools/call"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"task","taskId":"t1","status":"working","createdAt":"2026-07-28T00:00:00Z","lastUpdatedAt":"2026-07-28T00:00:00Z","ttlMs":null,"pollIntervalMs":10}}\n' "$id"
      ;;
    *"tasks/get"*)
      polls=$((polls+1))
      if [ "$polls" -le 1 ]; then
        printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","taskId":"t1","status":"working","createdAt":"2026-07-28T00:00:00Z","lastUpdatedAt":"2026-07-28T00:00:01Z","ttlMs":null,"pollIntervalMs":10}}\n' "$id"
      else
        printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","taskId":"t1","status":"completed","createdAt":"2026-07-28T00:00:00Z","lastUpdatedAt":"2026-07-28T00:00:02Z","ttlMs":null,"result":{"resultType":"complete","content":[{"type":"text","text":"done"}],"structuredContent":{"ok":true},"isError":false}}}\n' "$id"
      fi
      ;;
    *"tasks/cancel"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete"}}\n' "$id"
      ;;
  esac
done
"#;

    #[cfg(unix)]
    #[tokio::test]
    async fn polls_tasks_to_completion_when_the_extension_is_enabled() {
        let script_path = write_fake_server("tasks", TASK_SERVER);
        let mut server =
            McpServerConfig::stdio("worker", script_path.to_string_lossy().to_string());
        server.tasks = Some(McpTasksConfig::default());

        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider.add_server(server).await.unwrap();

        let route = provider.routes().remove(0);
        let output = provider
            .call_route(
                route,
                ToolInput::new("mcp_worker_slow".to_string(), json!({})),
            )
            .await
            .unwrap();

        // The task result stands in for the `tools/call` result, so the caller sees
        // the usual shape.
        assert_eq!(output.is_error, Some(false));
        assert_eq!(output.output["structured_content"], json!({"ok": true}));

        let _ = std::fs::remove_file(script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_task_handles_when_the_extension_is_not_enabled() {
        let script_path = write_fake_server("tasks_undeclared", TASK_SERVER);
        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "worker",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();

        let route = provider.routes().remove(0);
        let err = provider
            .call_route(
                route,
                ToolInput::new("mcp_worker_slow".to_string(), json!({})),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("tasks extension is not enabled"), "{err}");

        let _ = std::fs::remove_file(script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn task_waiting_is_bounded_by_max_wait() {
        let script_path = write_fake_server(
            "tasks_slow",
            &TASK_SERVER.replace("\"pollIntervalMs\":10", "\"pollIntervalMs\":9000"),
        );
        let mut server =
            McpServerConfig::stdio("worker", script_path.to_string_lossy().to_string());
        // The next poll would land past the deadline, so the call gives up instead
        // of blocking the turn on a task that outlives its budget.
        server.tasks = Some(McpTasksConfig { max_wait_secs: 1 });

        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider.add_server(server).await.unwrap();

        let route = provider.routes().remove(0);
        let err = provider
            .call_route(
                route,
                ToolInput::new("mcp_worker_slow".to_string(), json!({})),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("did not finish within 1s"), "{err}");

        let _ = std::fs::remove_file(script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn tool_list_changes_arrive_on_the_subscription_stream() {
        // On 2026-07-28 the notification only reaches a client that opened a
        // `subscriptions/listen` stream, so this covers the whole opt-in path:
        // subscribe, receive, re-list.
        let script_path = write_fake_server(
            "subscriptions",
            r#"#!/bin/sh
count=0
sub=0
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *"server/discover"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{"listChanged":true}},"ttlMs":0,"cacheScope":"private"}}\n' "$id"
      ;;
    *"subscriptions/listen"*)
      sub=$id
      printf '{"jsonrpc":"2.0","method":"notifications/subscriptions/acknowledged","params":{"_meta":{"io.modelcontextprotocol/subscriptionId":%s},"notifications":{"toolsListChanged":true}}}\n' "$sub"
      ;;
    *"tools/list"*)
      count=$((count+1))
      if [ "$count" -le 1 ]; then
        printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[{"name":"echo","description":"Echoes.","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      else
        printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[{"name":"echo","description":"Echoes.","inputSchema":{"type":"object","properties":{}}},{"name":"ping","description":"Pings.","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      fi
      ;;
    *"tools/call"*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","content":[{"type":"text","text":"ok"}],"isError":false}}\n' "$id"
      printf '{"jsonrpc":"2.0","method":"notifications/tools/list_changed","params":{"_meta":{"io.modelcontextprotocol/subscriptionId":%s}}}\n' "$sub"
      ;;
  esac
done
"#,
        );

        let provider = McpToolProvider::new(Vec::new()).unwrap();
        provider
            .add_server(McpServerConfig::stdio(
                "live",
                script_path.to_string_lossy().to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(provider.routes().len(), 1);

        // Every call answers and then announces a change on the stream; the next
        // call picks the notification up and re-lists.
        let route = provider.routes().remove(0);
        for _ in 0..20 {
            provider
                .call_route(route.clone(), ToolInput::new(route.name.clone(), json!({})))
                .await
                .unwrap();
            if provider.routes().len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let names: Vec<String> = provider
            .routes()
            .into_iter()
            .map(|route| route.name)
            .collect();
        assert_eq!(names, vec!["mcp_live_echo", "mcp_live_ping"]);

        let _ = std::fs::remove_file(script_path);
    }
}
