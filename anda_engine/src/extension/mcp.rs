//! MCP tool provider extension.
//!
//! This module makes Anda an MCP host/client for tool execution. It discovers
//! tools from configured MCP servers, maps them to legal Anda function names,
//! and dispatches calls back to the original MCP tool name. Each server is also
//! exposed as a [`ToolGroup`] (carrying its title and `instructions` from the
//! initialize handshake) so the discovery layer can present a server's tools as
//! a coherent capability bundle. It intentionally does not expose deprecated MCP
//! client utility capabilities such as Roots, Sampling, or Logging control.
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
    Usage, validate_function_name,
};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use parking_lot::{Mutex as SyncMutex, RwLock};
use reqwest::Client as ReqwestClient;
use rmcp::{
    ClientHandler, RoleClient,
    model::{
        CallToolRequestParams, CallToolResult, ClientInfo, Implementation, ServerPeerInfo,
        Tool as McpTool,
    },
    serve_client,
    service::RunningService,
    transport::{
        AuthClient, AuthError, AuthorizationManager, ClientCredentialsConfig, CredentialStore,
        StreamableHttpClientTransport, TokioChildProcess,
        auth::{AuthorizationCallback, AuthorizationMetadataSource, OAuthClientConfig, OAuthState},
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};

/// Re-exported from `rmcp`: the OAuth credentials an [`McpCredentialStore`]
/// persists on behalf of a server. Carries the (possibly dynamically
/// registered) `client_id` and the token response including the refresh token.
pub use rmcp::transport::StoredCredentials;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, hash_map::DefaultHasher},
    future::Future,
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{process::Command, sync::Mutex};

use crate::context::BaseCtx;

/// Default model-facing prefix for MCP-backed tools.
pub const DEFAULT_MCP_TOOL_PREFIX: &str = "mcp";

/// How many times to re-derive a local tool name before giving up on a collision.
const MAX_LOCAL_NAME_ATTEMPTS: usize = 8;

/// How far ahead of a client-credentials token's expiry to re-establish the session.
///
/// Comfortably wider than rmcp's own 30s refresh buffer, so the reconnect happens before any
/// request can fail with `AuthorizationRequired`.
const CLIENT_CREDENTIALS_RENEW_BUFFER: Duration = Duration::from_secs(120);

/// Minimum buffer that stays just ahead of rmcp's 30-second proactive refresh threshold.
const CLIENT_CREDENTIALS_MIN_RENEW_BUFFER: Duration = Duration::from_secs(31);

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

        let dirty = Arc::new(AtomicBool::new(false));
        let handler = AndaMcpClient::new(dirty.clone());
        let mut expires_at = None;
        let service = match &config.transport {
            McpTransportConfig::Stdio(stdio) => {
                let transport = TokioChildProcess::new(stdio.command())?;
                serve_client(handler, transport).await?
            }
            McpTransportConfig::StreamableHttp(http) => match &http.auth {
                None => {
                    let transport =
                        StreamableHttpClientTransport::from_config(http.transport_config()?);
                    serve_client(handler, transport).await?
                }
                Some(McpOAuthConfig::ClientCredentials(cc)) => {
                    // Headless: obtain a token at connection time, no human loop. The grant
                    // issues no refresh token, so the session carries the token's deadline
                    // and reconnects to mint a new one.
                    let (manager, deadline) = self.authorize_client_credentials(http, cc).await?;
                    expires_at = deadline;
                    let transport = StreamableHttpClientTransport::with_client(
                        AuthClient::new(ReqwestClient::new(), manager),
                        http.base_transport_config()?,
                    );
                    serve_client(handler, transport).await?
                }
                Some(McpOAuthConfig::AuthorizationCode(_)) => {
                    // Interactive: reuse credentials persisted by a prior
                    // begin/complete_authorization; refresh happens on demand.
                    let manager = self.authorize_from_store(&config.id, http).await?;
                    let transport = StreamableHttpClientTransport::with_client(
                        AuthClient::new(ReqwestClient::new(), manager),
                        http.base_transport_config()?,
                    );
                    serve_client(handler, transport).await?
                }
            },
        };

        let session = Arc::new(McpSession {
            service: Mutex::new(service),
            dirty,
            expires_at,
        });
        self.inner
            .index
            .write()
            .sessions
            .insert(config.id.clone(), session.clone());
        Ok(session)
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
        let manager = AuthorizationManager::new(url).await?;
        let resolution = manager.resolve_metadata().await?;
        if resolution.source == AuthorizationMetadataSource::LegacyEndpointFallback {
            return Ok(None);
        }
        let metadata = resolution.metadata;
        Ok(Some(McpOAuthMetadata {
            scopes_supported: metadata.scopes_supported.unwrap_or_default(),
            registration_supported: metadata.registration_endpoint.is_some(),
        }))
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

        let mut manager = AuthorizationManager::new(http.url.as_str()).await?;
        manager.set_credential_store(self.scoped_store(server_id));
        let metadata = manager.resolve_metadata().await?.metadata;
        manager.set_metadata(metadata);

        let scope_refs: Vec<&str> = ac.scopes.iter().map(String::as_str).collect();
        let client_config = match &ac.client_id {
            // Pre-registered public client.
            Some(client_id) => {
                let mut cfg = OAuthClientConfig::new(client_id.clone(), ac.redirect_uri.clone());
                if !ac.scopes.is_empty() {
                    cfg = cfg.with_scopes(ac.scopes.clone());
                }
                cfg
            }
            // Dynamic client registration (RFC 7591).
            None => {
                manager
                    .register_client(
                        ac.client_name.as_deref().unwrap_or("Anda Engine MCP Host"),
                        &ac.redirect_uri,
                        &scope_refs,
                    )
                    .await?
            }
        };
        manager.configure_client(client_config)?;
        let auth_url = manager.get_authorization_url(&scope_refs).await?;

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

        let callback = AuthorizationCallback::from_redirect_url(redirect_url)?;
        manager
            .exchange_code_for_token_with_issuer(
                &callback.code,
                &callback.csrf_token,
                callback.issuer.as_deref(),
            )
            .await?;
        Ok(())
    }

    /// Discards any pending interactive-authorization state for `server_id`.
    ///
    /// Returns whether a pending flow was actually cancelled.
    pub fn cancel_authorization(&self, server_id: &str) -> bool {
        self.inner.pending_auth.lock().remove(server_id).is_some()
    }

    fn scoped_store(&self, server_id: &str) -> ScopedCredentialStore {
        ScopedCredentialStore {
            server_id: server_id.to_string(),
            inner: self.inner.credential_store.clone(),
        }
    }

    /// Rebuilds an authorized manager from persisted Authorization Code
    /// credentials, refreshing on demand. Errors with [`McpAuthorizationRequired`]
    /// when no usable credentials exist yet, so the caller can trigger the
    /// interactive flow.
    async fn authorize_from_store(
        &self,
        server_id: &str,
        http: &McpStreamableHttpTransport,
    ) -> Result<AuthorizationManager, BoxError> {
        let mut manager = AuthorizationManager::new(http.url.as_str()).await?;
        manager.set_credential_store(self.scoped_store(server_id));
        if !manager.initialize_from_store().await? {
            return Err(McpAuthorizationRequired {
                server_id: server_id.to_string(),
            }
            .into());
        }
        Ok(manager)
    }

    /// Obtains an authorized manager via the headless Client Credentials flow.
    /// Runs the headless Client Credentials exchange and reports when the token expires.
    ///
    /// RFC 6749 §4.4.3 says a client-credentials grant SHOULD NOT issue a refresh token, and
    /// rmcp's only renewal path is `refresh_token`. Once the access token expires, every
    /// request fails with `AuthorizationRequired` and nothing re-runs this exchange. The
    /// returned deadline lets the session expire itself slightly early so `ensure_session`
    /// reconnects and mints a fresh token instead of failing permanently.
    async fn authorize_client_credentials(
        &self,
        http: &McpStreamableHttpTransport,
        config: &OAuthClientCredentialsConfig,
    ) -> Result<(AuthorizationManager, Option<Instant>), BoxError> {
        let mut state = OAuthState::new(http.url.as_str(), Some(ReqwestClient::new())).await?;
        state
            .authenticate_client_credentials(ClientCredentialsConfig::ClientSecret {
                client_id: config.client_id.clone(),
                client_secret: config.client_secret.clone(),
                scopes: config.scopes.clone(),
                resource: config.resource.clone(),
            })
            .await?;

        // Renew before rmcp's own 30s refresh buffer would kick in and fail. `expires_in` is
        // read through the token response's `Serialize` impl so this does not depend on
        // `oauth2` directly, which would risk a version skew with the one rmcp uses.
        let expires_at = match state.get_credentials().await {
            Ok((_, Some(token))) => serde_json::to_value(&token)
                .ok()
                .and_then(|token| token.get("expires_in").and_then(Json::as_u64))
                .and_then(|secs| {
                    client_credentials_deadline(Instant::now(), Duration::from_secs(secs))
                }),
            _ => None,
        };

        let manager = state
            .into_authorization_manager()
            .ok_or("MCP client_credentials authorization did not complete")?;
        Ok((manager, expires_at))
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

/// Computes the reconnect deadline for a client-credentials token.
///
/// Long-lived tokens renew 120 seconds early. Shorter tokens keep at least half their useful
/// lifetime while still staying ahead of rmcp's 30-second proactive refresh threshold. A remote
/// authorization server controls `expires_in`, so `checked_add` turns an unrepresentable duration
/// into "no local deadline" instead of panicking the process.
fn client_credentials_deadline(now: Instant, ttl: Duration) -> Option<Instant> {
    let buffer = if ttl >= CLIENT_CREDENTIALS_RENEW_BUFFER.saturating_mul(2) {
        CLIENT_CREDENTIALS_RENEW_BUFFER
    } else {
        (ttl / 2).max(CLIENT_CREDENTIALS_MIN_RENEW_BUFFER).min(ttl)
    };
    now.checked_add(ttl.saturating_sub(buffer))
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
    /// Deadline after which the session's credentials are stale and it must be re-established.
    ///
    /// Only set for the Client Credentials flow, whose grant issues no refresh token.
    expires_at: Option<Instant>,
}

impl McpSession {
    /// Reports whether the session can still carry requests.
    ///
    /// `RunningService::is_closed` only reflects a *locally* initiated shutdown: it is
    /// `handle.is_none() || cancellation_token.is_cancelled()`, and rmcp's serve loop exits
    /// with `QuitReason::Closed` on a peer-initiated close without cancelling that token. A
    /// crashed or exited MCP server would therefore look alive forever and every later call
    /// would fail with `TransportClosed` instead of triggering a reconnect. The peer's
    /// transport state is what actually flips, so check both — plus the credential deadline,
    /// since an expired token makes the session unusable while the transport is still open.
    async fn is_closed(&self) -> bool {
        if self
            .expires_at
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return true;
        }

        let service = self.service.lock().await;
        service.is_closed() || service.peer().is_transport_closed()
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
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct McpStdioTransport {
    /// Executable to spawn.
    pub command: String,
    /// Command arguments. These are passed without shell interpolation.
    #[serde(default)]
    pub args: Vec<String>,
    /// Additional environment variables.
    ///
    /// This is where a host application expands per-server secrets (API keys and the like),
    /// so the values are redacted from [`Debug`] output.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Optional working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
}

// Custom `Debug` to keep expanded environment secrets out of logs and error output. Keys are
// kept because they are useful for diagnosing a misconfigured server; only values are hidden.
impl std::fmt::Debug for McpStdioTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpStdioTransport")
            .field("command", &self.command)
            .field("args", &self.args)
            .field(
                "env",
                &self
                    .env
                    .keys()
                    .map(|key| (key, "[REDACTED]"))
                    .collect::<BTreeMap<_, _>>(),
            )
            .field("cwd", &self.cwd)
            .finish()
    }
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
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct McpStreamableHttpTransport {
    /// MCP endpoint URL.
    pub url: String,
    /// Bearer token value, without the `Bearer ` prefix. Mutually exclusive with
    /// [`auth`]: setting both is a validation error.
    ///
    /// [`auth`]: Self::auth
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bearer_token: Option<String>,
    /// Custom HTTP headers sent with every request, including OAuth-authorized
    /// requests.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Optional OAuth 2.1 authorization. Mutually exclusive with [`bearer_token`]
    /// (setting both is a validation error); access tokens are obtained and
    /// refreshed through the configured flow.
    ///
    /// [`bearer_token`]: Self::bearer_token
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<McpOAuthConfig>,
}

// Custom `Debug` to keep the static bearer token out of logs and error output.
// The `auth` field redacts its own secrets (see `OAuthClientCredentialsConfig`).
impl std::fmt::Debug for McpStreamableHttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpStreamableHttpTransport")
            .field("url", &self.url)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("headers", &self.headers)
            .field("auth", &self.auth)
            .finish()
    }
}

impl McpStreamableHttpTransport {
    fn validate(&self) -> Result<(), BoxError> {
        if self.url.trim().is_empty() {
            return Err("MCP HTTP URL must not be empty".into());
        }
        if let Some(auth) = &self.auth {
            if self.bearer_token.is_some() {
                return Err("MCP HTTP transport cannot set both `bearer_token` and `auth`".into());
            }
            auth.validate()?;
        }
        Ok(())
    }

    /// Custom headers applied to every request regardless of auth mode.
    fn custom_headers(&self) -> Result<HashMap<HeaderName, HeaderValue>, BoxError> {
        let mut headers = HashMap::new();
        for (name, value) in &self.headers {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes())?,
                HeaderValue::from_str(value)?,
            );
        }
        Ok(headers)
    }

    /// Base transport config (URI + custom headers) without a static bearer
    /// header, so an [`AuthClient`] can inject the OAuth access token instead.
    fn base_transport_config(&self) -> Result<StreamableHttpClientTransportConfig, BoxError> {
        Ok(
            StreamableHttpClientTransportConfig::with_uri(self.url.clone())
                .custom_headers(self.custom_headers()?),
        )
    }

    /// Transport config for the static (non-OAuth) path, attaching the optional
    /// bearer token as the `Authorization` header.
    fn transport_config(&self) -> Result<StreamableHttpClientTransportConfig, BoxError> {
        let mut config = self.base_transport_config()?;
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

/// OAuth 2.1 authorization for a Streamable HTTP MCP server.
///
/// `anda_engine` is a library: it drives the OAuth *protocol* and exposes the
/// seams, but never opens a browser, runs a callback server, or decides where
/// tokens live. The consuming application owns those concerns (see
/// [`McpToolProvider::begin_authorization`] and [`McpCredentialStore`]).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "flow", rename_all = "snake_case")]
pub enum McpOAuthConfig {
    /// Interactive Authorization Code flow with PKCE. Requires a one-time,
    /// out-of-band browser authorization that persists credentials; afterwards
    /// sessions are established from the stored refresh token with no human in
    /// the loop.
    AuthorizationCode(OAuthAuthorizationCodeConfig),
    /// Server-to-server Client Credentials flow (SEP-1046). Fully headless:
    /// tokens are obtained at connection time with no human interaction.
    ClientCredentials(OAuthClientCredentialsConfig),
}

impl McpOAuthConfig {
    fn validate(&self) -> Result<(), BoxError> {
        match self {
            Self::AuthorizationCode(config) => config.validate(),
            Self::ClientCredentials(config) => config.validate(),
        }
    }
}

/// Configuration for the interactive OAuth Authorization Code flow.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OAuthAuthorizationCodeConfig {
    /// Redirect URI registered/used for the authorization request. The consuming
    /// application decides how the redirect is received (loopback, a server
    /// route, or manual paste) and passes the resulting URL back through
    /// [`McpToolProvider::complete_authorization`].
    pub redirect_uri: String,
    /// Requested OAuth scopes.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Client name advertised during dynamic client registration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    /// Pre-registered public `client_id`. When omitted, the client is registered
    /// dynamically (RFC 7591 DCR) at authorization time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

impl OAuthAuthorizationCodeConfig {
    fn validate(&self) -> Result<(), BoxError> {
        if self.redirect_uri.trim().is_empty() {
            return Err("MCP OAuth authorization_code redirect_uri must not be empty".into());
        }
        Ok(())
    }
}

/// Configuration for the headless OAuth Client Credentials flow.
#[derive(Clone, Deserialize, Serialize)]
pub struct OAuthClientCredentialsConfig {
    /// Confidential client id.
    pub client_id: String,
    /// Confidential client secret.
    pub client_secret: String,
    /// Requested OAuth scopes.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Optional explicit resource indicator (RFC 8707).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
}

// Custom `Debug` to keep the client secret out of logs and error output, matching
// how `rmcp` redacts its own credential types.
impl std::fmt::Debug for OAuthClientCredentialsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthClientCredentialsConfig")
            .field("client_id", &self.client_id)
            .field("client_secret", &"[REDACTED]")
            .field("scopes", &self.scopes)
            .field("resource", &self.resource)
            .finish()
    }
}

impl OAuthClientCredentialsConfig {
    fn validate(&self) -> Result<(), BoxError> {
        if self.client_id.trim().is_empty() {
            return Err("MCP OAuth client_credentials client_id must not be empty".into());
        }
        if self.client_secret.trim().is_empty() {
            return Err("MCP OAuth client_credentials client_secret must not be empty".into());
        }
        Ok(())
    }
}

/// Outcome of [`McpToolProvider::discover_http_oauth`]: the OAuth capabilities
/// an HTTP MCP endpoint advertises.
#[derive(Debug, Clone)]
pub struct McpOAuthMetadata {
    /// Scopes the authorization server advertises (may be empty).
    pub scopes_supported: Vec<String>,
    /// Whether the server supports dynamic client registration (RFC 7591).
    pub registration_supported: bool,
}

/// Pluggable persistence for MCP OAuth credentials, keyed by server id.
///
/// The library never decides where tokens live; the consuming application
/// supplies an implementation (e.g. backed by an encrypted store) through
/// [`McpToolProviderBuilder::credential_store`]. Refresh tokens are secrets and
/// must be persisted securely.
#[async_trait]
pub trait McpCredentialStore: Send + Sync {
    /// Loads the stored credentials for `server_id`, if any.
    async fn load(&self, server_id: &str) -> Result<Option<StoredCredentials>, BoxError>;
    /// Persists credentials for `server_id`, replacing any previous value.
    async fn save(&self, server_id: &str, credentials: StoredCredentials) -> Result<(), BoxError>;
    /// Removes any stored credentials for `server_id`.
    async fn clear(&self, server_id: &str) -> Result<(), BoxError>;
}

/// Default in-memory [`McpCredentialStore`]. Credentials do not survive a
/// process restart; supply a persistent implementation in production.
#[derive(Debug, Default)]
pub struct InMemoryMcpCredentialStore {
    credentials: RwLock<HashMap<String, StoredCredentials>>,
}

impl InMemoryMcpCredentialStore {
    /// Creates an empty in-memory credential store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl McpCredentialStore for InMemoryMcpCredentialStore {
    async fn load(&self, server_id: &str) -> Result<Option<StoredCredentials>, BoxError> {
        Ok(self.credentials.read().get(server_id).cloned())
    }

    async fn save(&self, server_id: &str, credentials: StoredCredentials) -> Result<(), BoxError> {
        self.credentials
            .write()
            .insert(server_id.to_string(), credentials);
        Ok(())
    }

    async fn clear(&self, server_id: &str) -> Result<(), BoxError> {
        self.credentials.write().remove(server_id);
        Ok(())
    }
}

/// Adapts a keyed [`McpCredentialStore`] to rmcp's per-manager (keyless)
/// `CredentialStore`, bound to a single server id.
struct ScopedCredentialStore {
    server_id: String,
    inner: Arc<dyn McpCredentialStore>,
}

#[async_trait]
impl CredentialStore for ScopedCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        self.inner
            .load(&self.server_id)
            .await
            .map_err(|err| AuthError::InternalError(err.to_string()))
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        self.inner
            .save(&self.server_id, credentials)
            .await
            .map_err(|err| AuthError::InternalError(err.to_string()))
    }

    async fn clear(&self) -> Result<(), AuthError> {
        self.inner
            .clear(&self.server_id)
            .await
            .map_err(|err| AuthError::InternalError(err.to_string()))
    }
}

/// Error returned when establishing a session for a server configured with the
/// OAuth Authorization Code flow, but no usable stored credentials exist yet.
///
/// The consuming application should catch this (via [`BoxError`] downcast) and
/// run [`McpToolProvider::begin_authorization`] /
/// [`McpToolProvider::complete_authorization`] before retrying.
#[derive(Debug, Clone)]
pub struct McpAuthorizationRequired {
    /// The MCP server id that needs interactive authorization.
    pub server_id: String,
}

impl std::fmt::Display for McpAuthorizationRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MCP server {} requires interactive OAuth authorization; \
             call begin_authorization/complete_authorization first",
            self.server_id
        )
    }
}

impl std::error::Error for McpAuthorizationRequired {}

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
}
