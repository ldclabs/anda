//! MCP transport construction and session lifecycle.
//!
//! Transport configuration (stdio child process, streamable HTTP), the live
//! [`McpSession`] handle, protocol-revision negotiation for the `2026-07-28`
//! discovery lifecycle, and the `tools/list_changed` subscription pump that
//! keeps the route table fresh.

use anda_core::BoxError;
use http::{HeaderName, HeaderValue};
use rmcp::{
    ClientHandler, Peer, RoleClient,
    model::{
        ClientInfo, ExtensionCapabilities, Implementation, ProtocolVersion, ServerNotification,
        ServerPeerInfo, SubscriptionFilter, TASKS_EXTENSION_ID,
    },
    service::{ClientInitializeError, RunningService, Subscription},
    transport::streamable_http_client::StreamableHttpClientTransportConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::Map;
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{process::Command, sync::Mutex};

use super::auth::McpOAuthConfig;

/// How long a `server/discover` opener may go unanswered before the attempt is
/// abandoned.
///
/// A server that predates `2026-07-28` is not obliged to reject an unknown
/// method: plenty of them read the line, match nothing, and wait for the next
/// one, which would leave the handshake pending forever. Bounding the probe turns
/// that into a fallback ([`McpLifecycle::Auto`]) or an error
/// ([`McpLifecycle::Discover`]). Pin `initialize` to skip the wait entirely.
pub(crate) const DISCOVERY_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long to wait for a `subscriptions/listen` acknowledgment before giving up
/// on live `tools/list_changed` delivery for that session.
pub(crate) const SUBSCRIPTION_ACK_TIMEOUT: Duration = Duration::from_secs(10);

/// Delay before reopening a `subscriptions/listen` stream that ended while the
/// session is still usable. The streams are not resumable, so reopening is the
/// only way back to live `tools/list_changed` delivery.
const SUBSCRIPTION_REOPEN_DELAY: Duration = Duration::from_secs(2);

/// How long a subscription stream must last to count as working. Above this, a
/// server is just recycling idle streams and reopening is expected; below it,
/// the peer is ending streams as fast as they are opened.
const SUBSCRIPTION_HEALTHY_LIFETIME: Duration = Duration::from_secs(30);

/// Consecutive short-lived streams tolerated before the pump stops reopening.
const MAX_SHORT_LIVED_SUBSCRIPTIONS: usize = 5;

/// Protocol revisions this host offers `server/discover`, newest first.
///
/// `2025-11-25` stays in the list so a peer that implements the discovery RPC but
/// not the stateless revision still negotiates a usable version.
pub(crate) fn preferred_protocol_versions() -> Vec<ProtocolVersion> {
    vec![ProtocolVersion::V_2026_07_28, ProtocolVersion::V_2025_11_25]
}

/// Revision proposed by the legacy `initialize` handshake.
///
/// `2026-07-28` is deliberately not proposed here: it is only negotiated through
/// `server/discover`, which proves the peer implements the stateless lifecycle
/// rather than relying on it echoing back a version it does not serve.
pub(crate) fn legacy_protocol_version() -> ProtocolVersion {
    ProtocolVersion::V_2025_11_25
}


pub(crate) struct McpSession {
    pub(crate) service: Mutex<RunningService<RoleClient, AndaMcpClient>>,
    pub(crate) dirty: Arc<AtomicBool>,
    /// Deadline after which the session's credentials are stale and it must be re-established.
    ///
    /// Only set for the Client Credentials flow, whose grant issues no refresh token.
    pub(crate) expires_at: Option<Instant>,
    /// Task draining the `subscriptions/listen` stream, for peers that require one
    /// to deliver `tools/list_changed` (2026-07-28 and newer).
    pub(crate) subscription: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for McpSession {
    fn drop(&mut self) {
        // The pump holds a peer clone and would otherwise outlive the session it
        // feeds; dropping its `Subscription` also cancels the server-side stream.
        if let Some(subscription) = &self.subscription {
            subscription.abort();
        }
    }
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
    pub(crate) async fn is_closed(&self) -> bool {
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
pub(crate) struct AndaMcpClient {
    pub(crate) info: ClientInfo,
    dirty: Arc<AtomicBool>,
}

impl AndaMcpClient {
    pub(crate) fn new(dirty: Arc<AtomicBool>, tasks: bool) -> Self {
        let mut info = ClientInfo::default();
        info.client_info = Implementation::new("anda_engine", env!("CARGO_PKG_VERSION"))
            .with_title("Anda Engine MCP Host");
        // Only the legacy handshake reads this; the discovery lifecycle proposes
        // `preferred_protocol_versions` instead. Capabilities are sent either way —
        // in the `initialize` params, or in each request's `_meta` when the peer
        // negotiated the stateless revision.
        info.protocol_version = legacy_protocol_version();
        if tasks {
            info.capabilities
                .extensions
                .get_or_insert_with(ExtensionCapabilities::new)
                .insert(TASKS_EXTENSION_ID.to_string(), Map::new());
        }
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
    pub(crate) fn validate(&self) -> Result<(), BoxError> {
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
    pub(crate) fn validate(&self) -> Result<(), BoxError> {
        if self.command.trim().is_empty() {
            return Err("MCP stdio command must not be empty".into());
        }
        Ok(())
    }

    pub(crate) fn command(&self) -> Command {
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
    pub(crate) fn validate(&self) -> Result<(), BoxError> {
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
    pub(crate) fn base_transport_config(&self) -> Result<StreamableHttpClientTransportConfig, BoxError> {
        Ok(
            StreamableHttpClientTransportConfig::with_uri(self.url.clone())
                .custom_headers(self.custom_headers()?),
        )
    }

    /// Transport config for the static (non-OAuth) path, attaching the optional
    /// bearer token as the `Authorization` header.
    pub(crate) fn transport_config(&self) -> Result<StreamableHttpClientTransportConfig, BoxError> {
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


/// Whether a peer only delivers `tools/list_changed` on a subscription stream.
///
/// `2026-07-28` removed unsolicited server pushes, so from that revision on the
/// notification requires an explicit `subscriptions/listen` opt-in — and only if
/// the server advertises `tools.listChanged` at all.
pub(crate) fn needs_tool_subscription(info: Option<&ServerPeerInfo>) -> bool {
    let Some(info) = info else {
        return false;
    };
    info.protocol_version.as_str() >= ProtocolVersion::V_2026_07_28.as_str()
        && info
            .capabilities
            .tools
            .as_ref()
            .is_some_and(|tools| tools.list_changed == Some(true))
}

pub(crate) fn tool_subscription_filter() -> SubscriptionFilter {
    SubscriptionFilter::builder().tools_list_changed().build()
}

/// Drains one server's `tools/list_changed` subscription for the life of a session.
///
/// Subscription streams are not resumable, so an ended stream is replaced while
/// the transport is still up, and the session is marked dirty across the gap: a
/// change announced while nothing was listening must not leave the routes stale.
///
/// Reopening is bounded. A stream that survives [`SUBSCRIPTION_HEALTHY_LIFETIME`]
/// is treated as working — a server closing idle streams periodically keeps its
/// subscription forever — but a peer that acknowledges and immediately ends the
/// stream would otherwise spin here for the life of the process, re-listing on
/// every tool call, so it gets a limited number of consecutive attempts.
pub(crate) async fn pump_tool_subscription(
    server_id: String,
    peer: Peer<RoleClient>,
    mut subscription: Subscription,
    dirty: Arc<AtomicBool>,
) {
    let mut short_lived = 0usize;
    loop {
        let opened_at = Instant::now();
        loop {
            match subscription.next().await {
                Ok(Some(ServerNotification::ToolListChangedNotification(_))) => {
                    dirty.store(true, Ordering::SeqCst);
                }
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(err) => {
                    log::debug!("MCP server {server_id}: tools subscription failed: {err}");
                    break;
                }
            }
        }

        dirty.store(true, Ordering::SeqCst);
        if peer.is_transport_closed() {
            return;
        }

        short_lived = if opened_at.elapsed() >= SUBSCRIPTION_HEALTHY_LIFETIME {
            0
        } else {
            short_lived + 1
        };
        if short_lived > MAX_SHORT_LIVED_SUBSCRIPTIONS {
            log::warn!(
                "MCP server {server_id}: tools/list_changed stream ended immediately \
                 {short_lived} times; giving up on live tool updates"
            );
            return;
        }

        tokio::time::sleep(SUBSCRIPTION_REOPEN_DELAY).await;
        if peer.is_transport_closed() {
            return;
        }

        // Bounded like the initial subscription: a peer that accepts the request
        // and never acknowledges it must not park this task forever.
        subscription = match tokio::time::timeout(
            SUBSCRIPTION_ACK_TIMEOUT,
            peer.listen(tool_subscription_filter()),
        )
        .await
        {
            Ok(Ok(subscription)) => subscription,
            Ok(Err(err)) => {
                log::warn!(
                    "MCP server {server_id}: could not reopen the tools/list_changed \
                     subscription: {err}"
                );
                return;
            }
            Err(_) => {
                log::warn!(
                    "MCP server {server_id}: reopened tools/list_changed subscription was not \
                     acknowledged within {}s",
                    SUBSCRIPTION_ACK_TIMEOUT.as_secs()
                );
                return;
            }
        };
    }
}

/// Awaits a client handshake, optionally bounding how long it may stay pending.
pub(crate) async fn serve_bounded<F, S>(handshake: F, timeout: Option<Duration>) -> Result<S, BoxError>
where
    F: Future<Output = Result<S, ClientInitializeError>>,
{
    match timeout {
        None => Ok(handshake.await?),
        Some(limit) => match tokio::time::timeout(limit, handshake).await {
            Ok(result) => Ok(result?),
            Err(_) => Err(format!(
                "the MCP server did not answer the server/discover probe within {}s",
                limit.as_secs()
            )
            .into()),
        },
    }
}

