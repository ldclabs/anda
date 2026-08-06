//! OAuth authorization for MCP servers.
//!
//! Configuration types for the Authorization Code (interactive) and Client
//! Credentials (headless) flows, the [`McpCredentialStore`] persistence seam
//! the consuming application implements, and the error classification that
//! tells the session layer an authorization problem from a lifecycle problem.
//! `anda_engine` is a library: it drives the OAuth protocol but never opens a
//! browser, runs a callback server, or decides where tokens live.

use anda_core::{BoxError, Json};
use async_trait::async_trait;
use parking_lot::RwLock;
use reqwest::Client as ReqwestClient;
use rmcp::{
    service::ClientInitializeError,
    transport::{
        AuthError, AuthorizationManager, ClientCredentialsConfig, CredentialStore,
        StoredCredentials,
        auth::{AuthorizationCallback, AuthorizationMetadataSource, OAuthClientConfig, OAuthState},
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use super::McpServerConfig;
use super::session::McpTransportConfig;

/// How far ahead of a client-credentials token's expiry to re-establish the session.
///
/// Comfortably wider than rmcp's own 30s refresh buffer, so the reconnect happens before any
/// request can fail with `AuthorizationRequired`.
pub(crate) const CLIENT_CREDENTIALS_RENEW_BUFFER: Duration = Duration::from_secs(120);

/// Minimum buffer that stays just ahead of rmcp's 30-second proactive refresh threshold.
pub(crate) const CLIENT_CREDENTIALS_MIN_RENEW_BUFFER: Duration = Duration::from_secs(31);

/// Computes the reconnect deadline for a client-credentials token.
///
/// Long-lived tokens renew 120 seconds early. Shorter tokens keep at least half their useful
/// lifetime while still staying ahead of rmcp's 30-second proactive refresh threshold. A remote
/// authorization server controls `expires_in`, so `checked_add` turns an unrepresentable duration
/// into "no local deadline" instead of panicking the process.
pub(crate) fn client_credentials_deadline(now: Instant, ttl: Duration) -> Option<Instant> {
    let buffer = if ttl >= CLIENT_CREDENTIALS_RENEW_BUFFER.saturating_mul(2) {
        CLIENT_CREDENTIALS_RENEW_BUFFER
    } else {
        (ttl / 2).max(CLIENT_CREDENTIALS_MIN_RENEW_BUFFER).min(ttl)
    };
    now.checked_add(ttl.saturating_sub(buffer))
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
    pub(crate) fn validate(&self) -> Result<(), BoxError> {
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
    pub(crate) fn validate(&self) -> Result<(), BoxError> {
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
    pub(crate) fn validate(&self) -> Result<(), BoxError> {
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
pub(crate) struct ScopedCredentialStore {
    pub(crate) server_id: String,
    pub(crate) inner: Arc<dyn McpCredentialStore>,
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

/// Whether a failed connection attempt was about credentials rather than the
/// lifecycle, in which case retrying with a different opener cannot help.
pub(crate) fn is_authorization_error(err: &(dyn std::error::Error + 'static)) -> bool {
    if err.is::<McpAuthorizationRequired>() {
        return true;
    }
    // Credential acquisition reports a revoked or unusable grant directly, before
    // any transport exists; the handshake reports it wrapped in a transport error.
    if matches!(
        err.downcast_ref::<AuthError>(),
        Some(AuthError::AuthorizationRequired | AuthError::TokenRefreshRejected(_))
    ) {
        return true;
    }
    err.downcast_ref::<ClientInitializeError>()
        .is_some_and(ClientInitializeError::is_authorization_required)
}

/// Re-labels a credential failure on an interactive server as
/// [`McpAuthorizationRequired`], the signal applications are told to act on.
///
/// [`authorize_from_store`] raises that error when nothing is stored, but a grant
/// the authorization server has since revoked only shows up later, as a rejected
/// refresh or a `401` during the handshake. Both mean the same thing to the
/// caller — run the interactive flow again — so both are reported the same way.
/// The underlying cause is logged rather than dropped silently.
///
/// [`authorize_from_store`]: McpToolProvider::authorize_from_store
pub(crate) fn authorization_required_hint(config: &McpServerConfig, err: BoxError) -> BoxError {
    if err.is::<McpAuthorizationRequired>() || !is_authorization_error(err.as_ref()) {
        return err;
    }
    let McpTransportConfig::StreamableHttp(http) = &config.transport else {
        return err;
    };
    if !matches!(&http.auth, Some(McpOAuthConfig::AuthorizationCode(_))) {
        return err;
    }

    log::warn!(
        "MCP server {}: stored authorization is no longer usable ({err}); interactive \
         authorization must run again",
        config.id
    );
    McpAuthorizationRequired {
        server_id: config.id.clone(),
    }
    .into()
}

/// Probes an HTTP MCP endpoint to determine whether it requires OAuth.
///
/// Performs RFC 9728 protected-resource / RFC 8414 authorization-server
/// discovery against `url`. Returns `None` when the endpoint advertises no
/// OAuth support (it uses a static bearer token or no auth), or `Some` with
/// the discovered metadata otherwise.
pub(crate) async fn discover_http_oauth(url: &str) -> Result<Option<McpOAuthMetadata>, BoxError> {
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

/// Builds the manager for one interactive Authorization Code flow and returns
/// it with the authorization URL to open in a browser.
///
/// Resolves the server's OAuth metadata, then either configures the
/// pre-registered public client or performs dynamic client registration
/// (RFC 7591). The returned manager holds the PKCE/CSRF state consumed by
/// [`complete_authorization_exchange`].
pub(crate) async fn begin_authorization_manager(
    url: &str,
    ac: &OAuthAuthorizationCodeConfig,
    store: ScopedCredentialStore,
) -> Result<(AuthorizationManager, String), BoxError> {
    let mut manager = AuthorizationManager::new(url).await?;
    manager.set_credential_store(store);
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
    Ok((manager, auth_url))
}

/// Completes the Authorization Code exchange using the full redirect URL the
/// authorization server sent back (carrying `code`, `state`, and optionally
/// RFC 9207 `iss`). On success the credentials are persisted through the
/// manager's credential store.
pub(crate) async fn complete_authorization_exchange(
    manager: AuthorizationManager,
    redirect_url: &str,
) -> Result<(), BoxError> {
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

/// Rebuilds an authorized manager from persisted Authorization Code
/// credentials, refreshing on demand. Errors with [`McpAuthorizationRequired`]
/// when no usable credentials exist yet, so the caller can trigger the
/// interactive flow.
pub(crate) async fn authorize_from_store(
    server_id: &str,
    url: &str,
    store: ScopedCredentialStore,
) -> Result<AuthorizationManager, BoxError> {
    let mut manager = AuthorizationManager::new(url).await?;
    manager.set_credential_store(store);
    if !manager.initialize_from_store().await? {
        return Err(McpAuthorizationRequired {
            server_id: server_id.to_string(),
        }
        .into());
    }
    Ok(manager)
}

/// Obtains an authorized manager via the headless Client Credentials flow and
/// reports when the token expires.
///
/// RFC 6749 §4.4.3 says a client-credentials grant SHOULD NOT issue a refresh token, and
/// rmcp's only renewal path is `refresh_token`. Once the access token expires, every
/// request fails with `AuthorizationRequired` and nothing re-runs this exchange. The
/// returned deadline lets the session expire itself slightly early so the caller
/// reconnects and mints a fresh token instead of failing permanently.
pub(crate) async fn authorize_client_credentials(
    url: &str,
    config: &OAuthClientCredentialsConfig,
) -> Result<(AuthorizationManager, Option<Instant>), BoxError> {
    let mut state = OAuthState::new(url, Some(ReqwestClient::new())).await?;
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
