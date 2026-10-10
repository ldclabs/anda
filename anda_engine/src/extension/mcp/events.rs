//! MCP Events client for the experimental `events/*` extension.
//!
//! A server lists its event types with `events/list` and delivers occurrences
//! in up to three ways:
//!
//! - **poll**: `events/poll` returns the events after a cursor;
//! - **push**: a long-lived `events/stream` request, whose notifications carry
//!   the request id as `_meta["io.modelcontextprotocol/subscriptionId"]`;
//! - **webhook**: `events/subscribe` registers a callback URL that the
//!   application receives on itself (this module only subscribes, refreshes and
//!   unsubscribes).
//!
//! The provider keeps no event state. The application persists the cursor and
//! the event ids it has seen and passes the cursor back when it subscribes again.
//! A [`McpEventSink`] receives [`McpEventSignal`]s in delivery order, and a cursor
//! is only used for the next request after the sink accepted the signal that
//! carried it: delivery is at least once, deduplicated by `eventId`.
//!
//! rmcp 3.5 drops `capabilities.events` and hands each notification to its own
//! task, which can reorder them. Support is therefore detected by calling
//! `events/list`, and [`EventTap`] takes event notifications off the transport in
//! arrival order before rmcp sees them.

use super::{McpToolProvider, catalog::Registration, session::McpSession};
use super::{auth::is_authorization_error, catalog::collect_pages};
use anda_core::{BoxError, BoxFut, CancellationToken, Json};
use parking_lot::Mutex as SyncMutex;
use rmcp::{
    RoleClient,
    model::{
        ClientRequest, CustomRequest, JsonRpcMessage, JsonRpcNotification, RequestId,
        ServerNotification, ServerResult,
    },
    service::{PeerRequestOptions, RxJsonRpcMessage, ServiceError},
    transport::Transport,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, json};
use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch};

/// Floor for the delay between two polls, whatever `nextPollMs` asks for.
const MIN_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Ceiling for the delay between two polls.
const MAX_POLL_INTERVAL: Duration = Duration::from_secs(300);
/// Delay between polls when the server suggests none.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(30);
/// `maxEvents` sent with every poll.
const MAX_EVENTS_PER_POLL: u64 = 100;
/// Servers send a heartbeat at least every 30 seconds; a stream silent for twice
/// that is treated as dead and reopened.
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(60);
/// How often a stream checks its session and heartbeat deadline.
const STREAM_CHECK_INTERVAL: Duration = Duration::from_secs(5);
/// Notifications buffered per stream before it is reopened from its cursor.
const STREAM_BUFFER: usize = 256;
/// Largest event accepted, serialized; the webhook delivery limit of the draft.
pub const MAX_EVENT_BYTES: usize = 256 * 1024;
/// Notifications kept for a stream whose request id is not registered yet.
const EARLY_NOTICES: usize = 64;
const EARLY_NOTICE_TTL: Duration = Duration::from_secs(30);
/// First and last delay of the retry backoff after a recoverable failure.
const RETRY_MIN: Duration = Duration::from_secs(2);
const RETRY_MAX: Duration = Duration::from_secs(300);
/// Delay before reopening a stream the server closed normally.
const REOPEN_DELAY: Duration = Duration::from_secs(1);
/// Longest wait for `events/list` to answer before the server is assumed not to
/// know the method; some servers ignore unknown methods instead of rejecting them.
const LIST_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// Longest event id and event name accepted.
const MAX_ID_BYTES: usize = 1024;

const NOTIFICATION_PREFIX: &str = "notifications/events/";

/// How a server can deliver an event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpEventDeliveryMode {
    /// `events/poll`.
    Poll,
    /// `events/stream`.
    Push,
    /// `events/subscribe` with a callback URL.
    Webhook,
}

impl McpEventDeliveryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Poll => "poll",
            Self::Push => "push",
            Self::Webhook => "webhook",
        }
    }
}

/// One event type from `events/list`. Every field is untrusted server data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpEventDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Supported delivery modes; unknown modes are dropped.
    #[serde(default, deserialize_with = "known_modes")]
    pub delivery: Vec<McpEventDeliveryMode>,
    /// JSON Schema of the subscription arguments.
    #[serde(default = "object_schema")]
    pub input_schema: Json,
    /// JSON Schema of an event's `data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_schema: Option<Json>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Json>,
}

impl McpEventDefinition {
    /// The mode a local host prefers: push, then poll. `None` when the server
    /// only delivers by webhook.
    pub fn local_mode(&self) -> Option<McpEventDeliveryMode> {
        [McpEventDeliveryMode::Push, McpEventDeliveryMode::Poll]
            .into_iter()
            .find(|mode| self.delivery.contains(mode))
    }
}

fn object_schema() -> Json {
    json!({"type": "object"})
}

fn known_modes<'de, D>(deserializer: D) -> Result<Vec<McpEventDeliveryMode>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = Option::<Vec<Json>>::deserialize(deserializer)?.unwrap_or_default();
    let mut modes = Vec::new();
    for value in values {
        if let Ok(mode) = serde_json::from_value::<McpEventDeliveryMode>(value)
            && !modes.contains(&mode)
        {
            modes.push(mode);
        }
    }
    Ok(modes)
}

/// One event occurrence. `data` is untrusted server data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpEvent {
    /// Deduplication key; stays the same when the server delivers again.
    pub event_id: String,
    pub name: String,
    /// ISO 8601 time the event happened.
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub data: Json,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Json>,
}

/// What a subscription reports to its [`McpEventSink`], in delivery order.
#[derive(Debug, Clone, PartialEq)]
pub enum McpEventSignal {
    /// The subscription started, or restarted after a failure. `truncated` means
    /// the server could not replay everything since the cursor: events were lost.
    Active {
        mode: McpEventDeliveryMode,
        truncated: bool,
    },
    /// Events, possibly none, and the cursor to resume after them. `None` keeps
    /// the previous cursor. Store both together: the next request starts at
    /// this cursor once the sink accepts the signal.
    Events {
        events: Vec<McpEvent>,
        cursor: Option<String>,
    },
    /// A recoverable problem; the subscription keeps trying.
    Error { message: String },
    /// The server's event types changed; read `events/list` again.
    ListChanged,
    /// The subscription ended and will not resume.
    Terminated(McpEventError),
}

/// Receives the signals of one subscription.
pub trait McpEventSink: Send + Sync {
    /// `Ok` means the signal is durably accepted. An error on `Active` or
    /// `Events` makes the subscription start again from the last accepted
    /// cursor after a backoff; errors on other signals are ignored.
    fn deliver(&self, signal: McpEventSignal) -> BoxFut<'_, Result<(), BoxError>>;
}

/// Starts a poll or push subscription.
#[derive(Debug, Clone)]
pub struct McpEventSubscribeRequest {
    pub name: String,
    pub arguments: Map<String, Json>,
    /// [`McpEventDeliveryMode::Poll`] or [`McpEventDeliveryMode::Push`].
    pub mode: McpEventDeliveryMode,
    /// Where to resume; `None` starts from now.
    pub cursor: Option<String>,
    /// Replay floor: nothing older than this is replayed.
    pub max_age_ms: Option<u64>,
}

/// A running poll or push subscription. Dropping it stops the subscription.
#[derive(Debug)]
pub struct McpEventSubscription {
    mode: McpEventDeliveryMode,
    cancel: CancellationToken,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl McpEventSubscription {
    pub fn mode(&self) -> McpEventDeliveryMode {
        self.mode
    }

    /// Whether the subscription has ended, by cancellation or termination.
    pub fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(|task| task.is_finished())
    }

    /// Stops the subscription and waits until it has: a push stream is
    /// cancelled on the server before this returns.
    pub async fn cancel(mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for McpEventSubscription {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Registers or refreshes a webhook subscription. Subscribing again with the
/// same name, arguments and URL before `refresh_before` refreshes it.
#[derive(Clone)]
pub struct McpWebhookSubscribeRequest {
    pub name: String,
    pub arguments: Map<String, Json>,
    /// HTTPS callback URL.
    pub url: String,
    /// `whsec_` followed by base64 of 24 to 64 random bytes.
    pub secret: String,
    pub cursor: Option<String>,
    pub max_age_ms: Option<u64>,
    /// `None` leaves the lifetime to the server; `Some(None)` asks for no expiry.
    pub ttl_ms: Option<Option<u64>>,
}

impl std::fmt::Debug for McpWebhookSubscribeRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpWebhookSubscribeRequest")
            .field("name", &self.name)
            .field("arguments", &self.arguments)
            .field("url", &self.url)
            .field("secret", &"[REDACTED]")
            .field("cursor", &self.cursor)
            .field("max_age_ms", &self.max_age_ms)
            .field("ttl_ms", &self.ttl_ms)
            .finish()
    }
}

/// The server's answer to `events/subscribe`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpWebhookSubscription {
    /// Subscription id derived by the server from the subscription identity.
    #[serde(deserialize_with = "string_or_number")]
    pub id: String,
    /// ISO 8601 time to subscribe again by; `None` means no expiry.
    #[serde(default)]
    pub refresh_before: Option<String>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_status: Option<Json>,
}

fn string_or_number<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    match Json::deserialize(deserializer)? {
        Json::String(value) => Ok(value),
        Json::Number(value) => Ok(value.to_string()),
        _ => Err(serde::de::Error::custom("expected a string or number id")),
    }
}

/// Why an events request or subscription failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpEventErrorKind {
    /// The event type or subscription does not exist.
    NotFound,
    /// The server refused the subscription.
    Forbidden,
    /// A server limit was reached; retrying later may work.
    ResourceExhausted,
    /// The server does not support the method, mode or arguments.
    Unsupported,
    /// The webhook callback could not be verified or reached.
    CallbackEndpoint,
    /// The arguments do not match the event's input schema.
    InvalidParams,
    /// The server needs (re)authorization.
    AuthorizationRequired,
    /// The server was removed or re-registered on the provider.
    ServerRemoved,
    /// A connection, protocol or application problem; retrying may work.
    Other,
}

/// Error of an events request, or the reason a subscription ended.
#[derive(Debug, Clone, PartialEq)]
pub struct McpEventError {
    pub kind: McpEventErrorKind,
    /// JSON-RPC error code, when the server sent one.
    pub code: Option<i64>,
    pub message: String,
    /// JSON-RPC error data (`reason`, `kind`, ...), untrusted.
    pub data: Option<Json>,
}

impl std::fmt::Display for McpEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Some(code) => write!(f, "{} ({code})", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for McpEventError {}

impl McpEventError {
    fn new(kind: McpEventErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: None,
            message: message.into(),
            data: None,
        }
    }

    fn other(message: impl Into<String>) -> Self {
        Self::new(McpEventErrorKind::Other, message)
    }

    /// Classifies a JSON-RPC error. The draft renumbered its codes, so both the
    /// current (-32023..-32027) and the earlier (-32011..-32015, still used by
    /// OpenAI) numbers are recognized.
    fn from_rpc(code: i64, message: &str, data: Option<Json>) -> Self {
        let kind = match code {
            -32601 => McpEventErrorKind::Unsupported,
            -32602 => McpEventErrorKind::InvalidParams,
            -32023 | -32011 => McpEventErrorKind::NotFound,
            -32024 | -32012 => McpEventErrorKind::Forbidden,
            -32025 | -32013 => McpEventErrorKind::ResourceExhausted,
            -32026 | -32014 => McpEventErrorKind::Unsupported,
            -32027 | -32015 => McpEventErrorKind::CallbackEndpoint,
            _ => McpEventErrorKind::Other,
        };
        let mut message = truncate(message, 512);
        if let Some(reason) = data
            .as_ref()
            .and_then(|data| data.get("reason").or_else(|| data.get("kind")))
            .and_then(Json::as_str)
        {
            message = format!("{message}: {}", truncate(reason, 256));
        }
        Self {
            kind,
            code: Some(code),
            message,
            data,
        }
    }

    fn from_service(err: ServiceError) -> Self {
        match err {
            ServiceError::McpError(error) => {
                Self::from_rpc(error.code.0 as i64, &error.message, error.data)
            }
            err if is_authorization_error(&err) => {
                Self::new(McpEventErrorKind::AuthorizationRequired, err.to_string())
            }
            err => Self::other(err.to_string()),
        }
    }

    fn from_box(err: BoxError) -> Self {
        match err.downcast::<ServiceError>() {
            Ok(err) => Self::from_service(*err),
            Err(err) => match err.downcast::<McpEventError>() {
                Ok(err) => *err,
                Err(err) if is_authorization_error(err.as_ref()) => {
                    Self::new(McpEventErrorKind::AuthorizationRequired, err.to_string())
                }
                Err(err) => Self::other(err.to_string()),
            },
        }
    }

    /// Error object of an `error` or `terminated` notification.
    fn from_notice(params: &Json) -> Self {
        let error = params.get("error").unwrap_or(params);
        Self::from_rpc(
            error.get("code").and_then(Json::as_i64).unwrap_or(-32603),
            error
                .get("message")
                .and_then(Json::as_str)
                .unwrap_or("subscription ended"),
            error.get("data").cloned(),
        )
    }

    /// Whether retrying the same subscription cannot succeed.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.kind,
            McpEventErrorKind::NotFound
                | McpEventErrorKind::Forbidden
                | McpEventErrorKind::Unsupported
                | McpEventErrorKind::InvalidParams
                | McpEventErrorKind::AuthorizationRequired
                | McpEventErrorKind::ServerRemoved
        )
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

impl McpToolProvider {
    /// Lists a server's event types, or `None` when it does not implement MCP
    /// Events. Definitions over the server's description or schema limits are
    /// skipped.
    pub async fn list_events(
        &self,
        server_id: &str,
        cancellation: CancellationToken,
    ) -> Result<Option<Vec<McpEventDefinition>>, BoxError> {
        let config = self.server_config(server_id)?;
        let limit = Duration::from_secs(config.timeouts.setup_secs)
            + Duration::from_secs(config.timeouts.list_secs);
        let listing = async {
            let session = self.ensure_session(&config).await?;
            let peer = session.service.lock().await.peer().clone();
            let mut first = true;
            let result = collect_pages(&config.limits, |params| {
                let peer = peer.clone();
                let probe = std::mem::take(&mut first);
                async move {
                    let params = serde_json::to_value(params.unwrap_or_default())?;
                    let request = peer.send_request(custom("events/list", params));
                    let result = if probe {
                        tokio::time::timeout(LIST_PROBE_TIMEOUT, request)
                            .await
                            .map_err(|_| {
                                format!(
                                    "MCP server did not answer events/list within {}s; it \
                                     probably does not support MCP Events",
                                    LIST_PROBE_TIMEOUT.as_secs()
                                )
                            })?
                    } else {
                        request.await
                    };
                    let page: ListPage = serde_json::from_value(custom_result(result?)?)?;
                    Ok((page.events, page.next_cursor))
                }
            })
            .await;
            match result {
                Ok(events) => Ok(Some(events)),
                Err(err) => match err.downcast::<ServiceError>() {
                    Ok(err) => match *err {
                        ServiceError::McpError(error) if error.code.0 == -32601 => Ok(None),
                        err => Err(Box::new(err) as BoxError),
                    },
                    Err(err) => Err(err),
                },
            }
        };
        let events = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err("MCP events listing cancelled".into()),
            _ = config.cancelled.cancelled() => return Err("MCP server removed".into()),
            result = tokio::time::timeout(limit, listing) => result.map_err(|_| "MCP events listing timed out")??,
        };
        Ok(events.map(|events| {
            events
                .into_iter()
                .filter_map(|event| definition(&config, event))
                .collect()
        }))
    }

    /// Starts a poll or push subscription on a background task. The server must
    /// be registered; its session is established, and re-established after
    /// failures, as needed. The subscription ends on [`McpEventSubscription::cancel`],
    /// when it is dropped, or with a [`McpEventSignal::Terminated`] signal, which
    /// includes the server being removed or re-registered.
    pub fn subscribe_events(
        &self,
        server_id: &str,
        request: McpEventSubscribeRequest,
        sink: Arc<dyn McpEventSink>,
    ) -> Result<McpEventSubscription, BoxError> {
        if request.mode == McpEventDeliveryMode::Webhook {
            return Err("webhook subscriptions use subscribe_webhook".into());
        }
        if request.name.is_empty() || request.name.len() > MAX_ID_BYTES {
            return Err("MCP event name must be 1 to 1024 bytes".into());
        }
        let config = self.server_config(server_id)?;
        let cancel = CancellationToken::new();
        let mode = request.mode;
        let worker = Subscriber {
            provider: self.clone(),
            config,
            request,
            sink,
            cancel: cancel.clone(),
        };
        let task = tokio::spawn(worker.run());
        Ok(McpEventSubscription {
            mode,
            cancel,
            task: Some(task),
        })
    }

    /// Registers or refreshes a webhook subscription with `events/subscribe`.
    /// The application receives and verifies the deliveries itself. Errors are
    /// [`McpEventError`]s.
    pub async fn subscribe_webhook(
        &self,
        server_id: &str,
        request: McpWebhookSubscribeRequest,
        cancellation: CancellationToken,
    ) -> Result<McpWebhookSubscription, BoxError> {
        let mut params = json!({
            "name": request.name,
            "arguments": request.arguments,
            "delivery": {"mode": "webhook", "url": request.url, "secret": request.secret},
            "cursor": request.cursor,
        });
        if let Some(max_age_ms) = request.max_age_ms {
            params["maxAgeMs"] = max_age_ms.into();
        }
        if let Some(ttl_ms) = request.ttl_ms {
            params["ttlMs"] = ttl_ms.map_or(Json::Null, Json::from);
        }
        let result = self
            .events_request(server_id, "events/subscribe", params, cancellation)
            .await?;
        let subscription: McpWebhookSubscription =
            serde_json::from_value(result).map_err(|err| {
                McpEventError::other(format!("invalid events/subscribe result: {err}"))
            })?;
        let limits = &self.server_config(server_id)?.limits;
        if subscription.id.len() > MAX_ID_BYTES
            || subscription
                .cursor
                .as_ref()
                .is_some_and(|cursor| cursor.len() > limits.cursor_bytes)
        {
            return Err(McpEventError::other("events/subscribe result exceeds limits").into());
        }
        Ok(subscription)
    }

    /// Removes a webhook subscription with `events/unsubscribe`. Idempotent on
    /// the server.
    pub async fn unsubscribe_webhook(
        &self,
        server_id: &str,
        name: &str,
        arguments: Map<String, Json>,
        url: &str,
        cancellation: CancellationToken,
    ) -> Result<(), BoxError> {
        let params = json!({
            "name": name,
            "arguments": arguments,
            "delivery": {"mode": "webhook", "url": url},
        });
        self.events_request(server_id, "events/unsubscribe", params, cancellation)
            .await?;
        Ok(())
    }

    async fn events_request(
        &self,
        server_id: &str,
        method: &'static str,
        params: Json,
        cancellation: CancellationToken,
    ) -> Result<Json, McpEventError> {
        let config = self
            .server_config(server_id)
            .map_err(McpEventError::from_box)?;
        let limit = Duration::from_secs(config.timeouts.setup_secs)
            + Duration::from_secs(config.timeouts.request_secs);
        let request = async {
            let session = self
                .ensure_session(&config)
                .await
                .map_err(McpEventError::from_box)?;
            let peer = session.service.lock().await.peer().clone();
            let result = peer
                .send_request(custom(method, params))
                .await
                .map_err(McpEventError::from_service)?;
            custom_result(result).map_err(McpEventError::from_box)
        };
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(McpEventError::other(format!("MCP {method} cancelled"))),
            _ = config.cancelled.cancelled() => Err(McpEventError::new(McpEventErrorKind::ServerRemoved, "MCP server removed")),
            result = tokio::time::timeout(limit, request) => result
                .map_err(|_| McpEventError::other(format!("MCP {method} timed out")))?,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListPage {
    #[serde(default)]
    events: Vec<Json>,
    #[serde(default)]
    next_cursor: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PollPage {
    #[serde(default)]
    events: Vec<Json>,
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    has_more: bool,
    #[serde(default)]
    next_poll_ms: Option<u64>,
}

#[derive(Default, Deserialize)]
struct Position {
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default)]
    truncated: bool,
}

fn custom(method: &str, params: Json) -> ClientRequest {
    ClientRequest::CustomRequest(CustomRequest::new(method, Some(params)))
}

/// The JSON of a result to a custom request. rmcp parses results into an
/// untagged union, so a result can land in a typed variant that only carries a
/// subset of it; serializing the variant back keeps what it read.
fn custom_result(result: ServerResult) -> Result<Json, BoxError> {
    match result {
        ServerResult::CustomResult(result) => Ok(result.0),
        result => Ok(serde_json::to_value(result)?),
    }
}

fn definition(config: &Registration, value: Json) -> Option<McpEventDefinition> {
    let definition: McpEventDefinition = match serde_json::from_value(value) {
        Ok(definition) => definition,
        Err(err) => {
            log::warn!(
                "MCP server {}: skipping invalid event type: {err}",
                config.id
            );
            return None;
        }
    };
    let schema_bytes = |schema: &Json| serde_json::to_vec(schema).map_or(usize::MAX, |v| v.len());
    if definition.name.is_empty()
        || definition.name.len() > MAX_ID_BYTES
        || definition
            .description
            .as_ref()
            .is_some_and(|description| description.len() > config.limits.description_bytes)
        || schema_bytes(&definition.input_schema) > config.limits.schema_bytes
        || definition
            .payload_schema
            .as_ref()
            .is_some_and(|schema| schema_bytes(schema) > config.limits.schema_bytes)
        || definition
            .meta
            .as_ref()
            .is_some_and(|meta| schema_bytes(meta) > config.limits.schema_bytes)
    {
        log::warn!(
            "MCP server {}: skipping event type {:?} over the description or schema limits",
            config.id,
            truncate(&definition.name, 64)
        );
        return None;
    }
    Some(definition)
}

/// Parses and bounds one event. Returns a message for the sink when it is
/// rejected.
fn event(config: &Registration, value: Json) -> Result<McpEvent, String> {
    let bytes = serde_json::to_vec(&value).map_or(usize::MAX, |v| v.len());
    let event: McpEvent =
        serde_json::from_value(value).map_err(|err| format!("skipped an invalid event: {err}"))?;
    if bytes > MAX_EVENT_BYTES {
        return Err(format!(
            "skipped event {} of {bytes} bytes, over the {MAX_EVENT_BYTES}-byte limit",
            truncate(&event.event_id, 64)
        ));
    }
    if event.event_id.is_empty()
        || event.event_id.len() > MAX_ID_BYTES
        || event.name.len() > MAX_ID_BYTES
        || event
            .cursor
            .as_ref()
            .is_some_and(|cursor| cursor.len() > config.limits.cursor_bytes)
    {
        return Err("skipped an event with an invalid id, name or cursor".to_string());
    }
    Ok(event)
}

/// Exponential retry delay.
struct Backoff(Duration);

impl Backoff {
    fn new() -> Self {
        Self(RETRY_MIN)
    }
    fn reset(&mut self) {
        self.0 = RETRY_MIN;
    }
    fn next(&mut self) -> Duration {
        let delay = self.0;
        self.0 = (self.0 * 2).min(RETRY_MAX);
        delay
    }
}

/// How one round of a subscription ended.
enum Round {
    Cancelled,
    /// The server closed the push stream normally; reopen it.
    Closed,
    Failed(McpEventError),
}

struct Subscriber {
    provider: McpToolProvider,
    config: Arc<Registration>,
    request: McpEventSubscribeRequest,
    sink: Arc<dyn McpEventSink>,
    cancel: CancellationToken,
}

impl Subscriber {
    async fn run(mut self) {
        let mut backoff = Backoff::new();
        loop {
            let round = match self.request.mode {
                McpEventDeliveryMode::Push => self.stream(&mut backoff).await,
                _ => self.poll(&mut backoff).await,
            };
            let delay = match round {
                Round::Cancelled => return,
                Round::Closed => REOPEN_DELAY,
                Round::Failed(err) if err.is_terminal() => {
                    let _ = self.sink.deliver(McpEventSignal::Terminated(err)).await;
                    return;
                }
                Round::Failed(err) => {
                    log::debug!(
                        "MCP server {}: event subscription {} failed: {err}",
                        self.config.id,
                        self.request.name
                    );
                    let _ = self
                        .sink
                        .deliver(McpEventSignal::Error {
                            message: err.to_string(),
                        })
                        .await;
                    backoff.next()
                }
            };
            if let Some(round) = self.sleep(delay).await {
                if let Round::Failed(err) = round {
                    let _ = self.sink.deliver(McpEventSignal::Terminated(err)).await;
                }
                return;
            }
        }
    }

    /// Waits, unless the subscription is cancelled or the server removed first.
    async fn sleep(&self, delay: Duration) -> Option<Round> {
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Some(Round::Cancelled),
            _ = self.config.cancelled.cancelled() => Some(removed()),
            _ = tokio::time::sleep(delay) => None,
        }
    }

    async fn session(&self) -> Result<Arc<McpSession>, Round> {
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Err(Round::Cancelled),
            _ = self.config.cancelled.cancelled() => Err(removed()),
            result = tokio::time::timeout(
                Duration::from_secs(self.config.timeouts.setup_secs),
                self.provider.ensure_session(&self.config),
            ) => match result {
                Ok(Ok(session)) => Ok(session),
                Ok(Err(err)) => Err(Round::Failed(McpEventError::from_box(err))),
                Err(_) => Err(Round::Failed(McpEventError::other("MCP session setup timed out"))),
            },
        }
    }

    /// Delivers a signal that carries progress; a failure restarts the round.
    async fn deliver(&self, signal: McpEventSignal) -> Result<(), Round> {
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Err(Round::Cancelled),
            result = self.sink.deliver(signal) => result.map_err(|err| {
                Round::Failed(McpEventError::other(format!("event sink failed: {err}")))
            }),
        }
    }

    /// Delivers events and records the new cursor once the sink accepted them.
    async fn deliver_events(
        &mut self,
        events: Vec<McpEvent>,
        cursor: Option<String>,
    ) -> Result<(), Round> {
        if events.is_empty() && (cursor.is_none() || cursor == self.request.cursor) {
            return Ok(());
        }
        self.deliver(McpEventSignal::Events {
            events,
            cursor: cursor.clone(),
        })
        .await?;
        if cursor.is_some() {
            self.request.cursor = cursor;
        }
        Ok(())
    }

    /// Parses events, reporting the rejected ones as errors.
    async fn accept(&self, values: Vec<Json>) -> Vec<McpEvent> {
        let mut events = Vec::with_capacity(values.len());
        for value in values {
            match event(&self.config, value) {
                Ok(event) => events.push(event),
                Err(message) => {
                    log::warn!("MCP server {}: {message}", self.config.id);
                    let _ = self.sink.deliver(McpEventSignal::Error { message }).await;
                }
            }
        }
        events
    }

    async fn poll(&mut self, backoff: &mut Backoff) -> Round {
        let mut started = false;
        loop {
            let session = match self.session().await {
                Ok(session) => session,
                Err(round) => return round,
            };
            let mut list_changed = session.events.list_changed();
            let mut params = json!({
                "name": self.request.name,
                "arguments": self.request.arguments,
                "cursor": self.request.cursor,
                "maxEvents": MAX_EVENTS_PER_POLL,
            });
            if let Some(max_age_ms) = self.request.max_age_ms {
                params["maxAgeMs"] = max_age_ms.into();
            }
            let peer = session.service.lock().await.peer().clone();
            let request = tokio::time::timeout(
                Duration::from_secs(self.config.timeouts.request_secs),
                peer.send_request(custom("events/poll", params)),
            );
            let result = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => return Round::Cancelled,
                _ = self.config.cancelled.cancelled() => return removed(),
                result = request => result,
            };
            let page = match result {
                Err(_) => return Round::Failed(McpEventError::other("MCP events/poll timed out")),
                Ok(Err(err)) => return Round::Failed(McpEventError::from_service(err)),
                Ok(Ok(result)) => match custom_result(result)
                    .and_then(|value| Ok(serde_json::from_value::<PollPage>(value)?))
                {
                    Ok(page) => page,
                    Err(err) => {
                        return Round::Failed(McpEventError::other(format!(
                            "invalid events/poll result: {err}"
                        )));
                    }
                },
            };
            if page
                .cursor
                .as_ref()
                .is_some_and(|cursor| cursor.len() > self.config.limits.cursor_bytes)
            {
                return Round::Failed(McpEventError::other("MCP event cursor limit exceeded"));
            }
            if !started || page.truncated {
                started = true;
                if let Err(round) = self
                    .deliver(McpEventSignal::Active {
                        mode: McpEventDeliveryMode::Poll,
                        truncated: page.truncated,
                    })
                    .await
                {
                    return round;
                }
            }
            backoff.reset();
            let events = self.accept(page.events).await;
            if let Err(round) = self.deliver_events(events, page.cursor).await {
                return round;
            }
            if list_changed.has_changed().unwrap_or(false) {
                list_changed.mark_unchanged();
                let _ = self.sink.deliver(McpEventSignal::ListChanged).await;
            }
            let delay = if page.has_more {
                Duration::ZERO
            } else {
                page.next_poll_ms
                    .map_or(DEFAULT_POLL_INTERVAL, Duration::from_millis)
                    .clamp(MIN_POLL_INTERVAL, MAX_POLL_INTERVAL)
            };
            if let Some(round) = self.sleep(delay).await {
                return round;
            }
        }
    }

    async fn stream(&mut self, backoff: &mut Backoff) -> Round {
        let session = match self.session().await {
            Ok(session) => session,
            Err(round) => return round,
        };
        let router = session.events.clone();
        let mut list_changed = router.list_changed();
        let mut params = json!({
            "name": self.request.name,
            "arguments": self.request.arguments,
            "cursor": self.request.cursor,
        });
        if let Some(max_age_ms) = self.request.max_age_ms {
            params["maxAgeMs"] = max_age_ms.into();
        }
        let peer = session.service.lock().await.peer().clone();
        // No request timeout: the stream lives until it is cancelled, and the
        // heartbeat deadline below detects a dead one.
        let mut handle = match peer
            .send_request_with_option(
                custom("events/stream", params),
                PeerRequestOptions::no_options(),
            )
            .await
        {
            Ok(handle) => handle,
            Err(err) => return Round::Failed(McpEventError::from_service(err)),
        };
        let id = handle.id.clone();
        let mut notices = router.register(id.clone());
        let mut checks = tokio::time::interval(STREAM_CHECK_INTERVAL);
        checks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_traffic = Instant::now();
        let round = loop {
            tokio::select! {
                biased;
                _ = self.cancel.cancelled() => break Round::Cancelled,
                _ = self.config.cancelled.cancelled() => break removed(),
                notice = notices.recv() => {
                    let Some(notice) = notice else {
                        break Round::Failed(McpEventError::other(
                            "MCP event stream fell behind; reopening it from the last cursor",
                        ));
                    };
                    last_traffic = Instant::now();
                    match self.notice(notice, backoff).await {
                        Ok(()) => {}
                        Err(round) => break round,
                    }
                }
                response = &mut handle.rx => {
                    router.unregister(&id);
                    return match response {
                        Ok(Ok(_)) => Round::Closed,
                        Ok(Err(err)) => Round::Failed(McpEventError::from_service(err)),
                        Err(_) => Round::Failed(McpEventError::other("MCP event stream closed")),
                    };
                }
                Ok(()) = list_changed.changed() => {
                    let _ = self.sink.deliver(McpEventSignal::ListChanged).await;
                }
                _ = checks.tick() => {
                    if session.is_closed().await {
                        break Round::Failed(McpEventError::other("MCP session closed"));
                    }
                    if last_traffic.elapsed() > HEARTBEAT_TIMEOUT {
                        break Round::Failed(McpEventError::other(format!(
                            "no MCP event stream traffic for {}s; reopening it",
                            HEARTBEAT_TIMEOUT.as_secs()
                        )));
                    }
                }
            }
        };
        router.unregister(&id);
        // Tell the server to end the stream: on stdio a cancellation
        // notification, on HTTP rmcp also aborts the request's response stream.
        let _ = tokio::time::timeout(
            Duration::from_secs(5),
            handle.cancel(Some("subscription ended".to_string())),
        )
        .await;
        round
    }

    async fn notice(&mut self, notice: StreamNotice, backoff: &mut Backoff) -> Result<(), Round> {
        match notice.kind.as_str() {
            "active" => {
                let position: Position = serde_json::from_value(notice.params).unwrap_or_default();
                self.deliver(McpEventSignal::Active {
                    mode: McpEventDeliveryMode::Push,
                    truncated: position.truncated,
                })
                .await?;
                backoff.reset();
                self.position(position.cursor).await
            }
            "event" => {
                let mut params = notice.params;
                if let Some(params) = params.as_object_mut() {
                    params.remove("_meta");
                }
                let events = self.accept(vec![params]).await;
                let cursor = events.last().and_then(|event| event.cursor.clone());
                let cursor = self.checked(cursor)?;
                self.deliver_events(events, cursor).await
            }
            "heartbeat" => {
                let position: Position = serde_json::from_value(notice.params).unwrap_or_default();
                self.position(position.cursor).await
            }
            "error" => {
                let error = McpEventError::from_notice(&notice.params);
                let _ = self
                    .sink
                    .deliver(McpEventSignal::Error {
                        message: error.to_string(),
                    })
                    .await;
                Ok(())
            }
            "terminated" => Err(Round::Failed(McpEventError::from_notice(&notice.params))),
            _ => Ok(()),
        }
    }

    async fn position(&mut self, cursor: Option<String>) -> Result<(), Round> {
        let cursor = self.checked(cursor)?;
        self.deliver_events(Vec::new(), cursor).await
    }

    fn checked(&self, cursor: Option<String>) -> Result<Option<String>, Round> {
        if cursor
            .as_ref()
            .is_some_and(|cursor| cursor.len() > self.config.limits.cursor_bytes)
        {
            return Err(Round::Failed(McpEventError::other(
                "MCP event cursor limit exceeded",
            )));
        }
        Ok(cursor)
    }
}

fn removed() -> Round {
    Round::Failed(McpEventError::new(
        McpEventErrorKind::ServerRemoved,
        "MCP server removed",
    ))
}

/// An events notification for one stream: the part of the method after
/// `notifications/events/`, and its params.
#[derive(Debug)]
pub(super) struct StreamNotice {
    kind: String,
    params: Json,
}

/// Routes the event notifications of one session to their streams in arrival
/// order, and counts `notifications/events/list_changed`.
pub(super) struct EventRouter {
    state: SyncMutex<RouterState>,
    list_changed: watch::Sender<u64>,
}

#[derive(Default)]
struct RouterState {
    streams: HashMap<RequestId, mpsc::Sender<StreamNotice>>,
    /// Notifications that arrived before their stream registered: the server
    /// can answer an `events/stream` request before its sender sees the id.
    early: VecDeque<(Instant, RequestId, StreamNotice)>,
}

impl EventRouter {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: SyncMutex::new(RouterState::default()),
            list_changed: watch::Sender::new(0),
        })
    }

    fn list_changed(&self) -> watch::Receiver<u64> {
        self.list_changed.subscribe()
    }

    /// Takes event notifications out of the message flow; returns every other
    /// message unchanged.
    fn tap(&self, message: RxJsonRpcMessage<RoleClient>) -> Option<RxJsonRpcMessage<RoleClient>> {
        match message {
            JsonRpcMessage::Notification(JsonRpcNotification {
                notification: ServerNotification::CustomNotification(notification),
                ..
            }) if notification.method.starts_with(NOTIFICATION_PREFIX) => {
                let kind = &notification.method[NOTIFICATION_PREFIX.len()..];
                if kind == "list_changed" {
                    self.list_changed.send_modify(|count| *count += 1);
                    return None;
                }
                use rmcp::model::GetMeta;
                let Some(id) = notification.get_meta().subscription_id() else {
                    log::debug!(
                        "dropping MCP {} without a subscription id",
                        notification.method
                    );
                    return None;
                };
                let notice = StreamNotice {
                    kind: kind.to_string(),
                    params: notification.params.unwrap_or(Json::Null),
                };
                self.route(id, notice);
                None
            }
            message => Some(message),
        }
    }

    fn route(&self, id: RequestId, notice: StreamNotice) {
        let mut state = self.state.lock();
        if let Some(stream) = state.streams.get(&id) {
            // A full buffer drops the stream's sender, which ends its receiver;
            // the stream then reopens from its last accepted cursor.
            if stream.try_send(notice).is_err() {
                state.streams.remove(&id);
            }
            return;
        }
        let now = Instant::now();
        state
            .early
            .retain(|(at, _, _)| now.duration_since(*at) < EARLY_NOTICE_TTL);
        if state.early.len() >= EARLY_NOTICES {
            state.early.pop_front();
        }
        state.early.push_back((now, id, notice));
    }

    fn register(&self, id: RequestId) -> mpsc::Receiver<StreamNotice> {
        let (sender, receiver) = mpsc::channel(STREAM_BUFFER);
        let mut state = self.state.lock();
        let early = std::mem::take(&mut state.early);
        for (at, notice_id, notice) in early {
            if notice_id == id {
                let _ = sender.try_send(notice);
            } else {
                state.early.push_back((at, notice_id, notice));
            }
        }
        state.streams.insert(id, sender);
        receiver
    }

    fn unregister(&self, id: &RequestId) {
        let mut state = self.state.lock();
        state.streams.remove(id);
        state.early.retain(|(_, notice_id, _)| notice_id != id);
    }
}

/// Transport wrapper that hands event notifications to an [`EventRouter`] as
/// they are read, before rmcp dispatches them on separate tasks.
pub(super) struct EventTap<T> {
    inner: T,
    router: Arc<EventRouter>,
}

impl<T> EventTap<T> {
    pub(super) fn new(inner: T, router: Arc<EventRouter>) -> Self {
        Self { inner, router }
    }
}

impl<T> Transport<RoleClient> for EventTap<T>
where
    T: Transport<RoleClient>,
{
    type Error = T::Error;

    fn name() -> std::borrow::Cow<'static, str> {
        T::name()
    }

    fn send(
        &mut self,
        item: rmcp::service::TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.inner.send(item)
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        loop {
            let message = self.inner.receive().await?;
            if let Some(message) = self.router.tap(message) {
                return Some(message);
            }
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_both_error_code_generations() {
        for (code, kind) in [
            (-32023, McpEventErrorKind::NotFound),
            (-32011, McpEventErrorKind::NotFound),
            (-32024, McpEventErrorKind::Forbidden),
            (-32012, McpEventErrorKind::Forbidden),
            (-32025, McpEventErrorKind::ResourceExhausted),
            (-32013, McpEventErrorKind::ResourceExhausted),
            (-32026, McpEventErrorKind::Unsupported),
            (-32014, McpEventErrorKind::Unsupported),
            (-32027, McpEventErrorKind::CallbackEndpoint),
            (-32015, McpEventErrorKind::CallbackEndpoint),
            (-32602, McpEventErrorKind::InvalidParams),
            (-32603, McpEventErrorKind::Other),
        ] {
            assert_eq!(
                McpEventError::from_rpc(code, "x", None).kind,
                kind,
                "{code}"
            );
        }
        let error = McpEventError::from_rpc(
            -32027,
            "CallbackEndpointError",
            Some(json!({"reason": "challenge_failed"})),
        );
        assert_eq!(error.message, "CallbackEndpointError: challenge_failed");
        assert!(!error.is_terminal());
        assert!(McpEventError::from_rpc(-32024, "Forbidden", None).is_terminal());
        assert!(!McpEventError::from_rpc(-32025, "ResourceExhausted", None).is_terminal());
    }

    #[test]
    fn definitions_keep_known_modes_and_prefer_push() {
        let definition: McpEventDefinition = serde_json::from_value(json!({
            "name": "issue.opened",
            "delivery": ["webhook", "carrier-pigeon", "poll", "push", "poll"],
        }))
        .unwrap();
        assert_eq!(
            definition.delivery,
            vec![
                McpEventDeliveryMode::Webhook,
                McpEventDeliveryMode::Poll,
                McpEventDeliveryMode::Push
            ]
        );
        assert_eq!(definition.input_schema, json!({"type": "object"}));
        assert_eq!(definition.local_mode(), Some(McpEventDeliveryMode::Push));
        let webhook_only: McpEventDefinition =
            serde_json::from_value(json!({"name": "x", "delivery": ["webhook"]})).unwrap();
        assert_eq!(webhook_only.local_mode(), None);
    }

    fn notification(method: &str, id: Option<i64>, params: Json) -> RxJsonRpcMessage<RoleClient> {
        let mut params = params;
        if let Some(id) = id {
            params["_meta"] = json!({"io.modelcontextprotocol/subscriptionId": id});
        }
        serde_json::from_value(json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .unwrap()
    }

    #[test]
    fn router_keeps_order_and_holds_notices_for_late_streams() {
        let router = EventRouter::new();
        let list = router.list_changed();
        // Arrives before its stream registers.
        assert!(
            router
                .tap(notification(
                    "notifications/events/active",
                    Some(7),
                    json!({"cursor": "c0"})
                ))
                .is_none()
        );
        let mut stream = router.register(RequestId::Number(7));
        for n in 1..=3 {
            assert!(
                router
                    .tap(notification(
                        "notifications/events/event",
                        Some(7),
                        json!({"eventId": format!("e{n}"), "name": "x"}),
                    ))
                    .is_none()
            );
        }
        assert!(
            router
                .tap(notification(
                    "notifications/events/list_changed",
                    None,
                    json!({})
                ))
                .is_none()
        );
        assert!(list.has_changed().unwrap());
        // Other notifications pass through untouched.
        assert!(
            router
                .tap(notification(
                    "notifications/tools/list_changed",
                    None,
                    json!({})
                ))
                .is_some()
        );

        let kinds: Vec<_> = std::iter::from_fn(|| stream.try_recv().ok())
            .map(|notice| {
                (
                    notice.kind,
                    notice.params["eventId"].as_str().unwrap_or("").to_string(),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                ("active".to_string(), String::new()),
                ("event".to_string(), "e1".to_string()),
                ("event".to_string(), "e2".to_string()),
                ("event".to_string(), "e3".to_string()),
            ]
        );
        router.unregister(&RequestId::Number(7));
        assert!(stream.try_recv().is_err());
    }

    #[test]
    fn a_full_stream_buffer_ends_the_stream() {
        let router = EventRouter::new();
        let mut stream = router.register(RequestId::Number(1));
        for _ in 0..=STREAM_BUFFER {
            router.tap(notification(
                "notifications/events/heartbeat",
                Some(1),
                json!({}),
            ));
        }
        let mut received = 0;
        while let Ok(_notice) = stream.try_recv() {
            received += 1;
        }
        assert_eq!(received, STREAM_BUFFER);
        // The sender is gone: the stream sees the end and reopens.
        assert!(matches!(
            stream.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    /// Sink that forwards signals to the test, failing the first `fail_events`
    /// non-empty event deliveries.
    struct TestSink {
        signals: mpsc::UnboundedSender<McpEventSignal>,
        fail_events: std::sync::atomic::AtomicUsize,
    }

    impl McpEventSink for TestSink {
        fn deliver(&self, signal: McpEventSignal) -> BoxFut<'_, Result<(), BoxError>> {
            use std::sync::atomic::Ordering;
            Box::pin(async move {
                let failing = matches!(&signal, McpEventSignal::Events { events, .. } if !events.is_empty())
                    && self
                        .fail_events
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                        .is_ok();
                let _ = self.signals.send(signal);
                if failing {
                    return Err("store unavailable".into());
                }
                Ok(())
            })
        }
    }

    fn sink(fail_events: usize) -> (Arc<TestSink>, mpsc::UnboundedReceiver<McpEventSignal>) {
        let (signals, receiver) = mpsc::unbounded_channel();
        (
            Arc::new(TestSink {
                signals,
                fail_events: fail_events.into(),
            }),
            receiver,
        )
    }

    async fn next(receiver: &mut mpsc::UnboundedReceiver<McpEventSignal>) -> McpEventSignal {
        tokio::time::timeout(Duration::from_secs(10), receiver.recv())
            .await
            .expect("signal in time")
            .expect("sink open")
    }

    fn ids(signal: &McpEventSignal) -> (Vec<String>, Option<String>) {
        match signal {
            McpEventSignal::Events { events, cursor } => (
                events.iter().map(|event| event.event_id.clone()).collect(),
                cursor.clone(),
            ),
            other => panic!("expected events, got {other:?}"),
        }
    }

    fn request(name: &str, mode: McpEventDeliveryMode) -> McpEventSubscribeRequest {
        McpEventSubscribeRequest {
            name: name.to_string(),
            arguments: Map::from_iter([("repo".to_string(), json!("ldclabs/anda"))]),
            mode,
            cursor: None,
            max_age_ms: None,
        }
    }

    /// A stateless `2026-07-28` server with MCP Events: poll pages chained by
    /// cursor, a push stream that sends a burst and stays open, a stream that is
    /// terminated at once, and a marker file touched when a stream is cancelled.
    #[cfg(unix)]
    const EVENTS_SERVER: &str = r#"#!/bin/sh
marker="$1"
event() {
  printf '{"jsonrpc":"2.0","method":"notifications/events/%s","params":%s}\n' "$1" "$2"
}
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  meta="\"_meta\":{\"io.modelcontextprotocol/subscriptionId\":$id}"
  case "$line" in
    *'"method":"server/discover"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}},"ttlMs":0,"cacheScope":"private"}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","ttlMs":0,"cacheScope":"private","tools":[]}}\n' "$id"
      ;;
    *'"method":"events/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"events":[{"name":"issue.opened","description":"A new issue.","delivery":["push","poll","webhook"],"inputSchema":{"type":"object","properties":{"repo":{"type":"string"}}},"payloadSchema":{"type":"object"}},{"name":"huge","description":"%9000s"}]}}\n' "$id" ""
      ;;
    *'"method":"events/poll"'*)
      case "$line" in
        *'"cursor":null'*)
          printf '{"jsonrpc":"2.0","id":%s,"result":{"events":[{"eventId":"e1","name":"issue.opened","timestamp":"2026-10-10T00:00:00Z","data":{"n":1},"cursor":"c1"}],"cursor":"c1","truncated":false,"hasMore":true}}\n' "$id"
          ;;
        *'"cursor":"c1"'*)
          printf '{"jsonrpc":"2.0","id":%s,"result":{"events":[{"eventId":"e2","name":"issue.opened","timestamp":"2026-10-10T00:00:01Z","data":{"n":2},"cursor":"c2"}],"cursor":"c2","truncated":false,"hasMore":false,"nextPollMs":60000}}\n' "$id"
          ;;
        *)
          printf '{"jsonrpc":"2.0","id":%s,"result":{"events":[],"cursor":"c2","truncated":false,"hasMore":false,"nextPollMs":60000}}\n' "$id"
          ;;
      esac
      ;;
    *'"method":"events/stream"'*)
      case "$line" in
        *'"name":"doomed"'*)
          event terminated "{\"error\":{\"code\":-32024,\"message\":\"Forbidden\",\"data\":{\"reason\":\"Access revoked\"}},$meta}"
          ;;
        *)
          event active "{\"cursor\":\"s0\",\"truncated\":false,$meta}"
          for n in 1 2 3; do
            event event "{\"eventId\":\"p$n\",\"name\":\"issue.opened\",\"timestamp\":\"2026-10-10T00:00:0${n}Z\",\"data\":{\"n\":$n},\"cursor\":\"s$n\",$meta}"
          done
          event heartbeat "{\"cursor\":\"s3\",$meta}"
          printf '{"jsonrpc":"2.0","method":"notifications/events/list_changed","params":{}}\n'
          ;;
      esac
      ;;
    *'"method":"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","content":[{"type":"text","text":"acked"}],"structuredContent":{"ok":true},"isError":false}}\n' "$id"
      ;;
    *'"method":"notifications/cancelled"'*)
      touch "$marker"
      ;;
    *)
      if [ -n "$id" ]; then
        printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"Method not found"}}\n' "$id"
      fi
      ;;
  esac
done
"#;

    #[cfg(unix)]
    fn events_server(name: &str) -> (McpToolProvider, std::path::PathBuf, std::path::PathBuf) {
        use super::super::{McpLifecycle, McpServerConfig, McpTransportConfig};
        let script = super::super::tests::write_fake_server(name, EVENTS_SERVER);
        let marker = std::env::temp_dir().join(format!(
            "anda_fake_mcp_events_{}_{name}.cancelled",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&marker);
        let mut config = McpServerConfig::stdio("events", script.to_string_lossy().to_string());
        config.lifecycle = McpLifecycle::Discover;
        if let McpTransportConfig::Stdio(stdio) = &mut config.transport {
            stdio.args = vec![marker.to_string_lossy().to_string()];
        }
        let provider = McpToolProvider::new(vec![config]).unwrap();
        (provider, script, marker)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn lists_event_types_within_limits() {
        let (provider, script, _) = events_server("list");
        let events = provider
            .list_events("events", CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        // The type whose description exceeds the server's limit is skipped.
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "issue.opened");
        assert_eq!(events[0].local_mode(), Some(McpEventDeliveryMode::Push));
        assert_eq!(events[0].payload_schema, Some(json!({"type": "object"})));
        let _ = std::fs::remove_file(script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_application_calls_a_tool_the_model_does_not_see() {
        let (provider, script, _) = events_server("tool");
        let result = provider
            .call_server_tool(
                "events",
                "dmsg_events_ack",
                Map::from_iter([("cursor".to_string(), json!("r1"))]),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.structured_content, Some(json!({"ok": true})));
        // The server lists no tools: the call does not need a route.
        assert!(provider.routes().is_empty());
        let _ = std::fs::remove_file(script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_without_events_lists_none() {
        use super::super::{McpLifecycle, McpServerConfig};
        let script = super::super::tests::write_fake_server(
            "no_events",
            r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([^,}]*\).*/\1/p')
  case "$line" in
    *'"method":"server/discover"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}},"ttlMs":0,"cacheScope":"private"}}\n' "$id"
      ;;
    *)
      if [ -n "$id" ]; then
        printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"Method not found"}}\n' "$id"
      fi
      ;;
  esac
done
"#,
        );
        let mut config = McpServerConfig::stdio("plain", script.to_string_lossy().to_string());
        config.lifecycle = McpLifecycle::Discover;
        let provider = McpToolProvider::new(vec![config]).unwrap();
        assert_eq!(
            provider
                .list_events("plain", CancellationToken::new())
                .await
                .unwrap(),
            None
        );
        let _ = std::fs::remove_file(script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn polls_pages_in_order_and_redelivers_after_a_sink_failure() {
        let (provider, script, _) = events_server("poll");
        let (sink, mut signals) = sink(1);
        let subscription = provider
            .subscribe_events(
                "events",
                request("issue.opened", McpEventDeliveryMode::Poll),
                sink,
            )
            .unwrap();
        assert_eq!(
            next(&mut signals).await,
            McpEventSignal::Active {
                mode: McpEventDeliveryMode::Poll,
                truncated: false
            }
        );
        // The sink fails e1: the cursor stays where it was, so after the backoff
        // the subscription starts over and e1 comes again.
        assert_eq!(
            ids(&next(&mut signals).await),
            (vec!["e1".into()], Some("c1".into()))
        );
        assert!(
            matches!(next(&mut signals).await, McpEventSignal::Error { message } if message.contains("store unavailable"))
        );
        assert!(matches!(
            next(&mut signals).await,
            McpEventSignal::Active { .. }
        ));
        assert_eq!(
            ids(&next(&mut signals).await),
            (vec!["e1".into()], Some("c1".into()))
        );
        assert_eq!(
            ids(&next(&mut signals).await),
            (vec!["e2".into()], Some("c2".into()))
        );
        subscription.cancel().await;
        let _ = std::fs::remove_file(script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streams_a_burst_in_order_and_cancels_the_stream_on_the_server() {
        let (provider, script, marker) = events_server("push");
        let (sink, mut signals) = sink(0);
        let subscription = provider
            .subscribe_events(
                "events",
                request("issue.opened", McpEventDeliveryMode::Push),
                sink,
            )
            .unwrap();
        let mut received = Vec::new();
        let mut list_changed = false;
        while received.len() < 5 || !list_changed {
            match next(&mut signals).await {
                McpEventSignal::ListChanged => list_changed = true,
                signal => received.push(signal),
            }
        }
        assert_eq!(
            received[0],
            McpEventSignal::Active {
                mode: McpEventDeliveryMode::Push,
                truncated: false
            }
        );
        // The active cursor, then each event with its own; the heartbeat at s3
        // repeats the last cursor and adds nothing.
        let positions: Vec<_> = received[1..].iter().map(ids).collect();
        assert_eq!(
            positions,
            vec![
                (vec![], Some("s0".into())),
                (vec!["p1".into()], Some("s1".into())),
                (vec!["p2".into()], Some("s2".into())),
                (vec!["p3".into()], Some("s3".into())),
            ]
        );
        assert!(signals.try_recv().is_err());

        subscription.cancel().await;
        for _ in 0..50 {
            if marker.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(marker.exists(), "the server saw notifications/cancelled");
        let _ = std::fs::remove_file(marker);
        let _ = std::fs::remove_file(script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_terminated_stream_ends_the_subscription() {
        let (provider, script, _) = events_server("terminated");
        let (sink, mut signals) = sink(0);
        let subscription = provider
            .subscribe_events(
                "events",
                request("doomed", McpEventDeliveryMode::Push),
                sink,
            )
            .unwrap();
        let McpEventSignal::Terminated(error) = next(&mut signals).await else {
            panic!("expected termination");
        };
        assert_eq!(error.kind, McpEventErrorKind::Forbidden);
        assert_eq!(error.code, Some(-32024));
        assert!(
            error.message.contains("Access revoked"),
            "{}",
            error.message
        );
        for _ in 0..50 {
            if subscription.is_finished() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(subscription.is_finished());
        let _ = std::fs::remove_file(script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn removing_the_server_terminates_its_subscriptions() {
        let (provider, script, _) = events_server("removed");
        let (sink, mut signals) = sink(0);
        let _subscription = provider
            .subscribe_events(
                "events",
                request("issue.opened", McpEventDeliveryMode::Poll),
                sink,
            )
            .unwrap();
        assert!(matches!(
            next(&mut signals).await,
            McpEventSignal::Active { .. }
        ));
        provider.remove_server("events");
        loop {
            if let McpEventSignal::Terminated(error) = next(&mut signals).await {
                assert_eq!(error.kind, McpEventErrorKind::ServerRemoved);
                break;
            }
        }
        let _ = std::fs::remove_file(script);
    }

    /// Streamable HTTP server for webhook subscriptions and an SSE event stream.
    async fn http_server() -> (
        String,
        Arc<SyncMutex<Vec<Json>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use axum::{
            Json as HttpJson, Router, body::Body, extract::State, http::StatusCode,
            response::IntoResponse, routing::post,
        };
        use futures::StreamExt;

        async fn handle(
            State(seen): State<Arc<SyncMutex<Vec<Json>>>>,
            HttpJson(request): HttpJson<Json>,
        ) -> axum::response::Response {
            seen.lock().push(request.clone());
            let id = request["id"].clone();
            if id.is_null() {
                return StatusCode::ACCEPTED.into_response();
            }
            let params = &request["params"];
            let result = match request["method"].as_str().unwrap_or("") {
                "server/discover" => {
                    json!({"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}},"ttlMs":0,"cacheScope":"private"})
                }
                "events/subscribe" if params["delivery"]["url"] == "https://hooks.example/bad" => {
                    return HttpJson(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32015,"message":"CallbackEndpointError","data":{"reason":"challenge_failed"}}})).into_response();
                }
                "events/subscribe" => {
                    json!({"id":"sub_1","refreshBefore":"2026-10-11T00:00:00Z","cursor":"w5","truncated":false})
                }
                "events/unsubscribe" => json!({}),
                "events/stream" => {
                    let meta = json!({"io.modelcontextprotocol/subscriptionId": id});
                    let frames = [
                        json!({"jsonrpc":"2.0","method":"notifications/events/active","params":{"cursor":"h0","truncated":true,"_meta":meta}}),
                        json!({"jsonrpc":"2.0","method":"notifications/events/event","params":{"eventId":"h1","name":"push.http","timestamp":"2026-10-10T00:00:00Z","data":{"ok":true},"cursor":"h1","_meta":meta}}),
                    ];
                    let body = futures::stream::iter(
                        frames.map(|frame| Ok::<_, std::io::Error>(format!("data: {frame}\n\n"))),
                    )
                    .chain(futures::stream::pending());
                    return axum::response::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from_stream(body))
                        .unwrap();
                }
                _ => {
                    return HttpJson(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"Method not found"}})).into_response();
                }
            };
            HttpJson(json!({"jsonrpc":"2.0","id":id,"result":result})).into_response()
        }

        let seen = Arc::new(SyncMutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/mcp", post(handle))
            .with_state(seen.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, seen, task)
    }

    fn http_provider(url: &str) -> McpToolProvider {
        let mut config = super::super::McpServerConfig::streamable_http("remote", url);
        config.lifecycle = super::super::McpLifecycle::Discover;
        McpToolProvider::new(vec![config]).unwrap()
    }

    #[tokio::test]
    async fn subscribes_refreshes_and_unsubscribes_webhooks() {
        let (url, seen, task) = http_server().await;
        let provider = http_provider(&url);
        let mut webhook = McpWebhookSubscribeRequest {
            name: "issue.opened".to_string(),
            arguments: Map::from_iter([("repo".to_string(), json!("ldclabs/anda"))]),
            url: "https://hooks.example/e/1".to_string(),
            secret: "whsec_c2VjcmV0LXNlY3JldC1zZWNyZXQtc2VjcmV0".to_string(),
            cursor: Some("w4".to_string()),
            max_age_ms: None,
            ttl_ms: Some(None),
        };
        assert!(!format!("{webhook:?}").contains("whsec_"));
        let subscription = provider
            .subscribe_webhook("remote", webhook.clone(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(subscription.id, "sub_1");
        assert_eq!(
            subscription.refresh_before.as_deref(),
            Some("2026-10-11T00:00:00Z")
        );
        assert_eq!(subscription.cursor.as_deref(), Some("w5"));
        let sent = seen
            .lock()
            .iter()
            .find(|request| request["method"] == "events/subscribe")
            .cloned()
            .unwrap();
        assert_eq!(sent["params"]["delivery"]["mode"], "webhook");
        assert_eq!(sent["params"]["delivery"]["secret"], webhook.secret);
        assert_eq!(sent["params"]["cursor"], "w4");
        assert!(sent["params"]["ttlMs"].is_null() && sent["params"].get("ttlMs").is_some());

        webhook.url = "https://hooks.example/bad".to_string();
        let err = provider
            .subscribe_webhook("remote", webhook, CancellationToken::new())
            .await
            .unwrap_err();
        let err = err.downcast_ref::<McpEventError>().unwrap();
        assert_eq!(err.kind, McpEventErrorKind::CallbackEndpoint);
        assert!(err.message.contains("challenge_failed"));

        provider
            .unsubscribe_webhook(
                "remote",
                "issue.opened",
                Map::new(),
                "https://hooks.example/e/1",
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let sent = seen
            .lock()
            .iter()
            .rfind(|request| request["method"] == "events/unsubscribe")
            .cloned()
            .unwrap();
        assert_eq!(
            sent["params"]["delivery"],
            json!({"mode": "webhook", "url": "https://hooks.example/e/1"})
        );
        task.abort();
    }

    #[tokio::test]
    async fn streams_events_over_http_sse() {
        let (url, _seen, task) = http_server().await;
        let provider = http_provider(&url);
        let (sink, mut signals) = sink(0);
        let subscription = provider
            .subscribe_events(
                "remote",
                request("push.http", McpEventDeliveryMode::Push),
                sink,
            )
            .unwrap();
        assert_eq!(
            next(&mut signals).await,
            McpEventSignal::Active {
                mode: McpEventDeliveryMode::Push,
                truncated: true
            }
        );
        assert_eq!(ids(&next(&mut signals).await), (vec![], Some("h0".into())));
        assert_eq!(
            ids(&next(&mut signals).await),
            (vec!["h1".into()], Some("h1".into()))
        );
        subscription.cancel().await;
        task.abort();
    }
}
