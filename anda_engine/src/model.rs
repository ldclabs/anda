//! Model provider integration and label-based routing.
//!
//! This module adapts provider-specific completion APIs to the common
//! [`CompletionRequest`] and [`AgentOutput`] contract used by Anda agents.
//! Built-in providers currently include OpenAI-compatible APIs, Anthropic, and
//! Google Gemini.
//!
//! The [`Models`] registry maps model labels such as `primary`, `pro`,
//! `flash`, or `lite` to concrete [`Model`] instances. Labels let agents
//! request capability tiers without hard-coding provider model names.
//!
//! Custom providers can implement [`CompletionFeaturesDyn`] and be wrapped with
//! [`Model::with_completer`].

use anda_core::{AgentOutput, BoxError, BoxPinFut, CONTENT_TYPE_JSON, CompletionRequest, ToolCall};
use arc_swap::ArcSwap;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Duration;
use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    error::Error,
    fmt,
    sync::Arc,
};

pub mod anthropic;
pub mod gemini;
pub mod openai;

pub use reqwest;
pub use reqwest::Proxy;

use crate::APP_USER_AGENT;

pub use anda_core::ModelEffort;

const MODEL_REQUEST_MAX_RETRIES: usize = 1;
const MODEL_RETRY_BACKOFF: Duration = Duration::from_millis(300);
const MODEL_RETRY_MAX_BACKOFF: Duration = Duration::from_secs(1);

/// Serializable configuration for constructing a model adapter.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Provider family, such as `gemini`, `anthropic`, or `openai`.
    pub family: String,

    /// Provider-specific model name.
    pub model: String,

    /// Base URL for the provider API.
    pub api_base: String,

    /// API key used by the provider adapter.
    pub api_key: String,

    /// Optional labels for selecting this model in the engine.
    ///
    /// If omitted, the provider model name is used as the only label. Common
    /// labels include `primary`, `pro`, `flash`, `lite`, `audio`, `video`, `image`, `memory`.
    #[serde(default)]
    pub labels: Vec<String>,

    #[serde(default)]
    pub context_window: usize,

    #[serde(default)]
    pub max_output: usize,

    /// Optional reasoning/thinking effort for providers and models that support it.
    ///
    /// Supported config values are `minimal`, `low`, `medium`, `high`, and `max`.
    /// The effective set depends on the selected provider and model.
    #[serde(default)]
    pub effort: Option<ModelEffort>,

    /// Skips this model when loading a list of configs.
    #[serde(default)]
    pub disabled: bool,

    /// Sends Anthropic credentials with bearer authentication instead of the
    /// provider-specific API-key header.
    #[serde(default)]
    pub bearer_auth: bool,

    #[serde(default)]
    pub stream: bool,
}

impl ModelConfig {
    /// Builds a [`Model`] from this configuration.
    pub fn model(&self, http_client: reqwest::Client) -> Result<Model, BoxError> {
        if self.disabled {
            return Err("model is disabled".into());
        }
        if self.model.is_empty() {
            return Err(format!("{}: model name is required", self.model).into());
        }
        if self.family.is_empty() {
            return Err(format!("{}: model family is required", self.model).into());
        }
        if self.api_base.is_empty() {
            return Err(format!("{}: api_base is required", self.model).into());
        }
        if self.api_key.is_empty() {
            return Err(format!("{}: api_key is required", self.model).into());
        }

        let mut model = match self.family.as_str() {
            "gemini" => Model::with_completer(Arc::new(
                gemini::Client::new(&self.api_key, Some(self.api_base.clone()))
                    .with_client(http_client)
                    .completion_model(&self.model)
                    .with_stream(self.stream)
                    .with_effort(self.effort),
            )),
            "anthropic" => {
                let mut cli = anthropic::Client::new(&self.api_key, Some(self.api_base.clone()))
                    .with_client(http_client);
                if self.bearer_auth {
                    cli = cli.with_bearer_auth(true);
                }
                Model::with_completer(Arc::new(
                    cli.completion_model(&self.model)
                        .with_stream(self.stream)
                        .with_effort(self.effort),
                ))
            }
            "openai" => {
                if self.model.starts_with("gpt") {
                    Model::with_completer(Arc::new(
                        openai::Client::new(&self.api_key, Some(self.api_base.clone()))
                            .with_client(http_client)
                            .completion_model_v2(&self.model)
                            .with_stream(self.stream)
                            .with_effort(self.effort),
                    ))
                } else {
                    Model::with_completer(Arc::new(
                        openai::Client::new(&self.api_key, Some(self.api_base.clone()))
                            .with_client(http_client)
                            .completion_model(&self.model)
                            .with_stream(self.stream)
                            .with_effort(self.effort),
                    ))
                }
            }
            _ => return Err(format!("unsupported model family: {}", self.family).into()),
        };

        let labels = if self.labels.is_empty() {
            vec![self.model.to_ascii_lowercase()]
        } else {
            self.labels.clone()
        };
        model.context_window = self.context_window;
        model.max_output = self.max_output;
        Ok(model.with_labels(labels))
    }
}

/// Thread-safe model registry used by the engine.
///
/// It maintains two layers:
/// - `model`: the primary default model for general requests
/// - `models`: a label-based map for selecting specific models
///
/// The dedicated primary slot can be set explicitly via [`Models::set_model`]
/// or derived from the special label `primary` in the label map. This keeps
/// direct lookup (`get`) separate from default-routing (`get_model`).
pub struct Models {
    model: ArcSwap<Option<Model>>,
    models: ArcSwap<HashMap<String, Vec<Model>>>,
}

impl Default for Models {
    fn default() -> Self {
        Self {
            model: ArcSwap::new(Arc::new(None)),
            models: ArcSwap::new(Arc::new(HashMap::new())),
        }
    }
}

impl Models {
    /// Creates a new Models instance by cloning the internal state of another Models instance.
    pub fn from_clone(other: &Models) -> Self {
        let models: HashMap<String, Vec<Model>> = HashMap::from_iter(
            other
                .models
                .load()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );
        Self {
            model: ArcSwap::new(other.model.load_full()),
            models: ArcSwap::new(Arc::new(models)),
        }
    }

    /// Builds a registry from model configs by registering every resolved label.
    pub fn from_configs(configs: &[ModelConfig], http_client: reqwest::Client) -> Self {
        let models = Self::default();
        for config in configs {
            if let Ok(model) = config.model(http_client.clone()) {
                models.inner_set(model.labels.clone(), model);
            }
        }
        models
    }

    /// Returns whether a label exists in the direct lookup table.
    pub fn contains(&self, label: &str) -> bool {
        self.models.load().contains_key(&label.to_ascii_lowercase())
    }

    /// Returns the set of all registered model names across all labels.
    pub fn model_names(&self) -> BTreeSet<String> {
        self.models
            .load()
            .values()
            .flatten()
            .map(|m| m.model_name())
            .collect()
    }

    /// Sets the primary default model without mutating the label map.
    pub fn set_model(&self, model: Model) {
        self.inner_set(model.labels.clone(), model.clone());
        self.model.store(Arc::new(Some(model)));
    }

    /// Inserts or updates a single labeled model.
    ///
    /// The special label `primary` also updates the dedicated routing slot.
    /// If no primary exists yet, any inserted model is promoted
    /// to become the primary default.
    pub fn set(&self, label: String, model: Model) {
        self.inner_set(vec![label], model);
    }

    fn inner_set(&self, mut labels: Vec<String>, model: Model) {
        if self.model.load().is_none() {
            self.model.store(Arc::new(Some(model.clone())));
        }

        let model_name = model.model_name();
        labels.push(model_name.to_ascii_lowercase());
        for label in labels.iter_mut() {
            label.make_ascii_lowercase();
            if label == "primary" {
                self.model.store(Arc::new(Some(model.clone())));
            }
        }

        // rcu keeps concurrent inserts from losing each other's labels.
        self.models.rcu(|models| {
            let mut models = models.as_ref().clone();
            for label in &labels {
                match models.entry(label.clone()) {
                    Entry::Vacant(e) => {
                        e.insert(vec![model.clone()]);
                    }
                    Entry::Occupied(mut e) => {
                        e.get_mut().retain(|m| m.model_name() != model_name);
                        e.get_mut().push(model.clone());
                    }
                }
            }
            models
        });
    }

    /// Returns a model by lowercase label if it exists.
    ///
    /// This is a direct lookup only and never falls back to default routing.
    pub fn get(&self, label: &str) -> Option<Model> {
        self.models
            .load()
            .get(&label.to_ascii_lowercase())
            .and_then(|v| v.last().cloned())
    }

    /// Returns the primary model if available; otherwise returns any remaining
    /// labeled model.
    pub fn get_model(&self) -> Option<Model> {
        if let Some(m) = self.model.load().as_ref() {
            return Some(m.clone());
        }
        self.models
            .load()
            .values()
            .next()
            .and_then(|v| v.last().cloned())
    }

    /// Resolves a model for lowercase-label-aware routing.
    ///
    /// Resolution order is:
    /// - the exact label match when `label` is non-empty
    /// - the default routing result from [`Models::get_model`]
    pub fn resolve(&self, label: &str) -> Option<Model> {
        if label.is_empty() {
            return self.get_model();
        }
        self.get(label).or_else(|| self.get_model())
    }
}

/// Object-safe completion provider interface.
pub trait CompletionFeaturesDyn: Send + Sync + 'static {
    /// Performs a completion request and returns the agent-facing output.
    ///
    /// Built-in adapters wrap exhausted transient provider failures in
    /// [`ModelError`]. Use [`is_retryable_box_error`] to decide whether an upper
    /// layer should schedule a delayed retry.
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>>;

    /// Returns the provider model name used for diagnostics and usage reports.
    fn model_name(&self) -> String;
}

/// Placeholder implementation that returns errors for completion requests.
#[derive(Clone, Debug)]
pub struct NotImplemented;

impl CompletionFeaturesDyn for NotImplemented {
    fn model_name(&self) -> String {
        "not_implemented".to_string()
    }

    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

/// Mock implementation for tests and examples.
#[derive(Clone, Debug)]
pub struct MockImplemented;

impl CompletionFeaturesDyn for MockImplemented {
    fn model_name(&self) -> String {
        "not_implemented".to_string()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: req.prompt.clone(),
            tool_calls: req
                .tools
                .iter()
                .filter_map(|tool| {
                    if req.prompt.is_empty() {
                        return None;
                    }
                    Some(ToolCall {
                        name: tool.name.clone(),
                        args: serde_json::from_str(&req.prompt).unwrap_or_default(),
                        call_id: None,
                        result: None,
                        remote_id: None,
                    })
                })
                .collect(),
            ..Default::default()
        })))
    }
}

/// Concrete model entry registered with the engine.
#[derive(Clone)]
pub struct Model {
    /// Completion provider implementation.
    pub completer: Arc<dyn CompletionFeaturesDyn>,

    /// Labels that can route requests to this model.
    pub labels: Vec<String>,

    pub context_window: usize,

    pub max_output: usize,
}

impl Model {
    /// Creates a model from a completion provider.
    pub fn new(completer: Arc<dyn CompletionFeaturesDyn>) -> Self {
        Self {
            completer,
            labels: Vec::new(),
            context_window: 0,
            max_output: 0,
        }
    }

    /// Creates a model from a completion provider.
    pub fn with_completer(completer: Arc<dyn CompletionFeaturesDyn>) -> Self {
        Self {
            completer,
            labels: Vec::new(),
            context_window: 0,
            max_output: 0,
        }
    }

    /// Assigns labels used by [`Models`] for routing.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Creates a model whose completion calls return `not implemented` errors.
    pub fn not_implemented() -> Self {
        Self {
            completer: Arc::new(NotImplemented),
            labels: Vec::new(),
            context_window: 0,
            max_output: 0,
        }
    }

    /// Creates a model with deterministic mock completion behavior for tests.
    pub fn mock_implemented() -> Self {
        Self {
            completer: Arc::new(MockImplemented),
            labels: Vec::new(),
            context_window: 0,
            max_output: 0,
        }
    }

    /// Returns the provider model name for this model.
    pub fn model_name(&self) -> String {
        self.completer.model_name()
    }

    /// Executes a completion request with the underlying provider.
    pub async fn completion(&self, req: CompletionRequest) -> Result<AgentOutput, BoxError> {
        self.completer.completion(req).await
    }
}

/// Error returned by built-in model adapters when the caller can inspect retry
/// semantics after the SDK-level retry has already been attempted.
#[derive(Debug)]
pub struct ModelError {
    message: String,
    retryable: bool,
    status: Option<http::StatusCode>,
    retry_after: Option<Duration>,
    source: Option<BoxError>,
}

impl ModelError {
    /// Creates a non-retryable model error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            status: None,
            retry_after: None,
            source: None,
        }
    }

    /// Marks whether this error should be considered retryable by the caller.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    /// Attaches the upstream HTTP status, if the provider returned one.
    pub fn with_status(mut self, status: http::StatusCode) -> Self {
        self.status = Some(status);
        self
    }

    /// Attaches a suggested retry delay from the upstream response.
    pub fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after = retry_after;
        self
    }

    /// Attaches the lower-level transport/read error.
    pub fn with_source(mut self, source: BoxError) -> Self {
        self.source = Some(source);
        self
    }

    /// Returns true when the upper layer may choose a delayed retry.
    pub fn is_retryable(&self) -> bool {
        self.retryable
    }

    /// Returns the upstream HTTP status, when present.
    pub fn status(&self) -> Option<http::StatusCode> {
        self.status
    }

    /// Returns the upstream retry delay, when present.
    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

/// Returns true if the error chain carries a retryable model error signal.
pub fn is_retryable_model_error(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<ModelError>()
            && error.is_retryable()
        {
            return true;
        }
        if let Some(error) = error.downcast_ref::<reqwest::Error>()
            && is_retryable_reqwest_error(error)
        {
            return true;
        }
        current = error.source();
    }
    false
}

/// Convenience wrapper for callers that keep completion errors as [`BoxError`].
pub fn is_retryable_box_error(error: &BoxError) -> bool {
    is_retryable_model_error(error.as_ref() as &(dyn Error + 'static))
}

/// Returns the first [`ModelError`] HTTP status found in the error chain.
pub fn model_error_status(error: &(dyn Error + 'static)) -> Option<http::StatusCode> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<ModelError>()
            && error.status().is_some()
        {
            return error.status();
        }
        current = error.source();
    }
    None
}

/// Returns the first upstream retry delay found in the error chain.
pub fn model_error_retry_after(error: &(dyn Error + 'static)) -> Option<Duration> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<ModelError>()
            && error.retry_after().is_some()
        {
            return error.retry_after();
        }
        current = error.source();
    }
    None
}

/// Statuses that are transient enough for one immediate SDK retry and for an
/// upper-layer delayed retry after the SDK retry has been exhausted.
pub fn is_retryable_status(status: http::StatusCode) -> bool {
    matches!(
        status,
        http::StatusCode::REQUEST_TIMEOUT
            | http::StatusCode::TOO_MANY_REQUESTS
            | http::StatusCode::INTERNAL_SERVER_ERROR
            | http::StatusCode::BAD_GATEWAY
            | http::StatusCode::SERVICE_UNAVAILABLE
            | http::StatusCode::GATEWAY_TIMEOUT
    ) || status.as_u16() == 529
}

pub(crate) fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    err.is_timeout()
        || err.is_connect()
        || err.is_body()
        || err.is_decode()
        || err.status().is_some_and(is_retryable_status)
}

pub(crate) fn completion_transport_error(
    model: &str,
    action: &str,
    err: reqwest::Error,
) -> BoxError {
    let retryable = is_retryable_reqwest_error(&err);
    let message = format!("{action}, model: {model}, error: {err}");
    Box::new(
        ModelError::new(message)
            .with_retryable(retryable)
            .with_source(Box::new(err)),
    )
}

pub(crate) async fn read_completion_response_bytes(
    response: reqwest::Response,
    model: &str,
) -> Result<bytes::Bytes, BoxError> {
    response
        .bytes()
        .await
        .map_err(|err| completion_transport_error(model, "Failed to read completion response", err))
}

pub(crate) async fn execute_completion_request_with_retry<T, BuildRequest, HandleResponse, Fut>(
    model: &str,
    build_request: BuildRequest,
    handle_response: HandleResponse,
) -> Result<T, BoxError>
where
    BuildRequest: Fn() -> reqwest::RequestBuilder,
    HandleResponse: Fn(reqwest::Response) -> Fut,
    Fut: Future<Output = Result<T, BoxError>>,
{
    for attempt in 0..=MODEL_REQUEST_MAX_RETRIES {
        let response = match build_request().send().await {
            Ok(response) => response,
            Err(err) => {
                let retryable = is_retryable_reqwest_error(&err);
                let message = format!(
                    "Failed to send completion request, model: {}, error: {}",
                    model, err
                );
                if retryable && attempt < MODEL_REQUEST_MAX_RETRIES {
                    log_completion_retry(model, attempt + 1, &message);
                    backoff_before_retry(None).await;
                    continue;
                }

                return Err(Box::new(
                    ModelError::new(message)
                        .with_retryable(retryable)
                        .with_source(Box::new(err)),
                ));
            }
        };

        let status = response.status();
        if status.is_success() {
            match handle_response(response).await {
                Ok(output) => return Ok(output),
                Err(err) if is_retryable_box_error(&err) && attempt < MODEL_REQUEST_MAX_RETRIES => {
                    log_completion_retry(model, attempt + 1, &err.to_string());
                    backoff_before_retry(None).await;
                    continue;
                }
                Err(err) => return Err(err),
            }
        }

        let retryable = is_retryable_status(status);
        let retry_after = retry_after_duration(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(err) => {
                let retryable = retryable || is_retryable_reqwest_error(&err);
                let message = format!(
                    "Completion failed, model: {}, status: {}; failed to read error body: {}",
                    model, status, err
                );
                if retryable && attempt < MODEL_REQUEST_MAX_RETRIES {
                    log_completion_retry(model, attempt + 1, &message);
                    backoff_before_retry(retry_after).await;
                    continue;
                }

                return Err(Box::new(
                    ModelError::new(message)
                        .with_retryable(retryable)
                        .with_status(status)
                        .with_retry_after(retry_after)
                        .with_source(Box::new(err)),
                ));
            }
        };
        let message = format!(
            "Completion failed, model: {}, status: {}, body: {}",
            model, status, body
        );

        if retryable && attempt < MODEL_REQUEST_MAX_RETRIES {
            log_completion_retry(model, attempt + 1, &message);
            backoff_before_retry(retry_after).await;
            continue;
        }

        return Err(Box::new(
            ModelError::new(message)
                .with_retryable(retryable)
                .with_status(status)
                .with_retry_after(retry_after),
        ));
    }

    unreachable!("completion retry loop always returns before exhausting attempts")
}

/// Sleeps briefly before the single in-SDK retry so transient overload
/// (429/5xx/connection flaps) has a chance to clear. The upstream `Retry-After`
/// hint is honored up to a small cap; longer waits are the responsibility of
/// upper layers, which receive the hint via [`ModelError::retry_after`].
async fn backoff_before_retry(retry_after: Option<Duration>) {
    let delay = retry_after
        .unwrap_or(MODEL_RETRY_BACKOFF)
        .min(MODEL_RETRY_MAX_BACKOFF);
    tokio::time::sleep(delay).await;
}

fn retry_after_duration(headers: &http::HeaderMap) -> Option<Duration> {
    let value = headers
        .get(http::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    // HTTP-date form, e.g. "Wed, 21 Oct 2026 07:28:00 GMT", common from
    // gateways and CDNs in front of model providers.
    let when = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    (when.with_timezone(&chrono::Utc) - chrono::Utc::now())
        .to_std()
        .ok()
}

fn log_completion_retry(model: &str, retry: usize, reason: &str) {
    log::warn!(
        "Retrying completion request, model: {}, retry: {}/{}, error: {}",
        model,
        retry,
        MODEL_REQUEST_MAX_RETRIES,
        reason
    );
}

#[derive(Clone, Copy, Debug)]
pub struct AnyHost;

impl PartialEq<&str> for AnyHost {
    fn eq(&self, _other: &&str) -> bool {
        true
    }
}

/// Creates a reqwest client builder with Anda Engine defaults.
pub fn request_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .use_rustls_tls()
        .https_only(true)
        .retry(
            reqwest::retry::for_host(AnyHost)
                .max_retries_per_request(1)
                .classify_fn(|req_rep| {
                    let is_idempotent = matches!(
                        req_rep.method(),
                        &http::Method::GET
                            | &http::Method::HEAD
                            | &http::Method::OPTIONS
                            | &http::Method::TRACE
                            | &http::Method::PUT
                            | &http::Method::DELETE
                    );

                    if !is_idempotent {
                        return req_rep.success();
                    }

                    if req_rep.error().is_some() {
                        return req_rep.retryable();
                    }

                    match req_rep.status() {
                        Some(status) if is_retryable_status(status) => req_rep.retryable(),
                        _ => req_rep.success(),
                    }
                }),
        )
        .http2_keep_alive_interval(Some(Duration::from_secs(25)))
        .http2_keep_alive_timeout(Duration::from_secs(15))
        .http2_keep_alive_while_idle(true)
        .connect_timeout(Duration::from_secs(10))
        // Total request timeout, including the streamed body. Heavy reasoning
        // completions can run for many minutes; provider SDKs default to 10
        // minutes. Stalled connections are detected earlier by h2 keep-alive.
        .timeout(Duration::from_secs(600))
        .user_agent(APP_USER_AGENT)
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            let ct: http::HeaderValue = http::HeaderValue::from_static(CONTENT_TYPE_JSON);
            headers.insert(http::header::CONTENT_TYPE, ct.clone());
            headers.insert(http::header::ACCEPT, ct);
            headers
        })
}

const SSE_DONE_MARKER: &[u8] = b"data: [DONE]";

pub(crate) async fn read_sse_json_events<T>(
    response: reqwest::Response,
    model: &str,
) -> Result<Vec<T>, BoxError>
where
    T: DeserializeOwned,
{
    let mut body = Vec::new();
    let mut scanned: usize = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| {
            completion_transport_error(model, "Failed to read streaming completion response", err)
        })?;
        body.extend_from_slice(&chunk);
        // Only scan the unscanned tail (with marker-sized overlap), so long
        // streams are not rescanned from the start on every chunk.
        let start = scanned.saturating_sub(SSE_DONE_MARKER.len());
        if body_contains_sse_done(&body, start) {
            return parse_streaming_json_events(&body, model);
        }
        scanned = body.len();
    }

    parse_streaming_json_events(&body, model)
}

/// Returns true when a `data: [DONE]` line exists at or after `from`.
///
/// The marker must be anchored at the start of an SSE line (buffer start or a
/// preceding `\n`). Generated content inside a JSON string can legitimately
/// contain the marker text, and must not terminate the stream early.
fn body_contains_sse_done(body: &[u8], from: usize) -> bool {
    if from == 0 && body.starts_with(SSE_DONE_MARKER) {
        return true;
    }
    body[from..]
        .windows(SSE_DONE_MARKER.len() + 1)
        .any(|window| window[0] == b'\n' && &window[1..] == SSE_DONE_MARKER)
}

fn parse_streaming_json_events<T>(body: &[u8], model: &str) -> Result<Vec<T>, BoxError>
where
    T: DeserializeOwned,
{
    let body = std::str::from_utf8(body).map_err(|err| {
        format!(
            "Invalid UTF-8 in streaming completion response, model: {}, error: {}",
            model, err
        )
    })?;
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);

    if !looks_like_sse(body) {
        return parse_json_event_payload(body, model);
    }

    let mut data = String::new();
    let mut events = Vec::new();

    for line in body.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        handle_sse_text_line(line, &mut data, &mut events, model)?;
    }
    flush_sse_data(&mut data, &mut events, model)?;

    Ok(events)
}

fn looks_like_sse(body: &str) -> bool {
    body.lines().any(|line| {
        let line = line.strip_prefix('\u{feff}').unwrap_or(line);
        line.starts_with("data:")
            || line.starts_with("event:")
            || line.starts_with("id:")
            || line.starts_with("retry:")
            || line.starts_with(':')
    })
}

fn handle_sse_text_line<T>(
    line: &str,
    data: &mut String,
    events: &mut Vec<T>,
    model: &str,
) -> Result<(), BoxError>
where
    T: DeserializeOwned,
{
    if line.is_empty() {
        return flush_sse_data(data, events, model);
    }
    if line.starts_with(':') {
        return Ok(());
    }

    let Some(value) = line.strip_prefix("data:") else {
        return Ok(());
    };
    let value = value.strip_prefix(' ').unwrap_or(value);
    if !data.is_empty() {
        data.push('\n');
    }
    data.push_str(value);
    Ok(())
}

fn flush_sse_data<T>(data: &mut String, events: &mut Vec<T>, model: &str) -> Result<(), BoxError>
where
    T: DeserializeOwned,
{
    let value = data.trim_end();
    if value.is_empty() || value == "[DONE]" {
        data.clear();
        return Ok(());
    }

    let event = serde_json::from_str::<T>(value).map_err(|err| {
        format!(
            "Invalid streaming completion event, model: {}, error: {}, body: {}",
            model, err, value
        )
    })?;
    events.push(event);
    data.clear();
    Ok(())
}

fn parse_json_event_payload<T>(body: &str, model: &str) -> Result<Vec<T>, BoxError>
where
    T: DeserializeOwned,
{
    let value = body.trim().strip_prefix('\u{feff}').unwrap_or(body.trim());
    if value.is_empty() || value == "[DONE]" {
        return Ok(Vec::new());
    }

    if value.starts_with('[')
        && let Ok(events) = serde_json::from_str::<Vec<T>>(value)
    {
        return Ok(events);
    }

    match serde_json::from_str::<T>(value) {
        Ok(event) => Ok(vec![event]),
        Err(single_err) => match serde_json::from_str::<Vec<T>>(value) {
            Ok(events) => Ok(events),
            Err(array_err) => {
                let mut events = Vec::new();
                let mut saw_line = false;
                for line in value.lines() {
                    let line = line.trim();
                    if line.is_empty() || line == "[DONE]" {
                        continue;
                    }
                    saw_line = true;
                    let event = serde_json::from_str::<T>(line).map_err(|line_err| {
                        format!(
                            "Invalid streaming completion event, model: {}, error: {}, body: {}",
                            model, line_err, line
                        )
                    })?;
                    events.push(event);
                }

                if saw_line {
                    return Ok(events);
                }

                Err(format!(
                    "Invalid streaming completion event, model: {}, error: {}; array error: {}, body: {}",
                    model, single_err, array_err, value
                )
                .into())
            }
        },
    }
}

pub(crate) fn streaming_completion_request(
    request: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    request
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .header(reqwest::header::ACCEPT_ENCODING, "identity")
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::FunctionDefinition;
    use axum::{Router, body::Bytes, extract::State, response::IntoResponse, routing::any};
    use http::{HeaderMap, HeaderValue, Method, StatusCode};
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Clone)]
    struct TestCompleter {
        name: &'static str,
    }

    impl CompletionFeaturesDyn for TestCompleter {
        fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Ok(AgentOutput::default())))
        }

        fn model_name(&self) -> String {
            self.name.to_string()
        }
    }

    fn test_model(name: &'static str) -> Model {
        Model::new(Arc::new(TestCompleter { name }))
    }

    fn http_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    #[derive(Clone)]
    struct MockHttpResponse {
        status: StatusCode,
        headers: HeaderMap,
        body: Vec<u8>,
    }

    type RetryState = Arc<Mutex<(VecDeque<MockHttpResponse>, usize)>>;

    async fn retry_mock_handler(
        State(state): State<RetryState>,
        _method: Method,
        _body: Bytes,
    ) -> impl IntoResponse {
        let mut state = state.lock().unwrap();
        state.1 += 1;
        let mock = state.0.pop_front().expect("mock response should exist");
        let mut response = (mock.status, mock.body).into_response();
        for (name, value) in mock.headers.iter() {
            response.headers_mut().insert(name, value.clone());
        }
        response
    }

    async fn spawn_retry_mock_server(responses: Vec<MockHttpResponse>) -> (String, RetryState) {
        let state = Arc::new(Mutex::new((responses.into(), 0)));
        let app = Router::new()
            .fallback(any(retry_mock_handler))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), state)
    }

    async fn spawn_truncated_sse_after_done_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Content-Length: 4096\r\n\
                      Connection: close\r\n\
                      \r\n\
                      data: {\"a\":1}\n\n\
                      data: [DONE]\n\n",
                )
                .await
                .unwrap();
            let _ = socket.shutdown().await;
        });
        format!("http://{addr}")
    }

    /// Sends an event whose JSON content embeds the literal `data: [DONE]`
    /// text in an early chunk, then a second event and the real terminator
    /// in a later chunk.
    async fn spawn_sse_with_done_marker_in_content_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n\
                      data: {\"text\":\"sse ends with data: [DONE]\"}\n\n",
                )
                .await
                .unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            socket
                .write_all(b"data: {\"b\":2}\n\ndata: [DONE]\n\n")
                .await
                .unwrap();
            let _ = socket.shutdown().await;
        });
        format!("http://{addr}")
    }

    fn retry_count(state: &RetryState) -> usize {
        state.lock().unwrap().1
    }

    fn model_config(family: &str, model: &str) -> ModelConfig {
        ModelConfig {
            family: family.to_string(),
            model: model.to_string(),
            api_base: "https://example.com".to_string(),
            api_key: "test-key".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn model_effort_serializes_config_values() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "family": "openai",
            "model": "gpt-5",
            "api_base": "http://localhost",
            "api_key": "test-key",
            "effort": "max"
        }))
        .unwrap();

        assert_eq!(config.effort, Some(ModelEffort::Max));
        assert_eq!(
            serde_json::to_value(ModelEffort::Minimal).unwrap(),
            "minimal"
        );
    }

    #[test]
    fn models_default_is_empty() {
        let models = Models::default();

        assert!(models.get_model().is_none());
        assert!(models.get("missing").is_none());
        assert!(models.resolve("missing").is_none());
    }

    #[test]
    fn set_model_sets_primary_without_registering_a_label() {
        let models = Models::default();
        models.set_model(test_model("primary"));

        assert_eq!(
            models
                .get_model()
                .expect("primary model should exist")
                .model_name(),
            "primary"
        );
        assert!(models.get("primary").is_some());
    }

    #[test]
    fn set_promotes_first_inserted_model_to_primary() {
        let models = Models::default();
        models.set("x".to_string(), test_model("X"));

        assert_eq!(
            models.get("x").expect("label x should exist").model_name(),
            "X"
        );
        assert_eq!(
            models
                .get_model()
                .expect("primary model should be initialized")
                .model_name(),
            "X"
        );
    }

    #[test]
    fn fallback_label_has_no_default_routing_semantics() {
        let models = Models::default();
        models.set_model(test_model("primary"));
        models.set("fallback".to_string(), test_model("fallback"));

        assert_eq!(
            models
                .get("fallback")
                .expect("fallback is still a normal label")
                .model_name(),
            "fallback"
        );
        assert_eq!(
            models
                .get_model()
                .expect("primary model should stay the default")
                .model_name(),
            "primary"
        );
        assert_eq!(
            models
                .resolve("unknown")
                .expect("missing label should use default routing")
                .model_name(),
            "primary"
        );
    }

    #[test]
    fn resolve_prefers_exact_label_then_default() {
        let models = Models::default();
        models.set_model(test_model("primary"));
        models.set("flash".to_string(), test_model("flash"));

        assert_eq!(
            models
                .resolve("flash")
                .expect("exact label should win")
                .model_name(),
            "flash"
        );
        assert_eq!(
            models
                .resolve("missing")
                .expect("missing label should use default routing")
                .model_name(),
            "primary"
        );
        assert_eq!(
            models
                .resolve("")
                .expect("empty label should use default routing")
                .model_name(),
            "primary"
        );
    }

    #[test]
    fn model_config_validates_required_fields_and_builds_supported_families() {
        let client = http_client();

        let mut config = model_config("openai", "gpt-5");
        config.disabled = true;
        let Err(err) = config.model(client.clone()) else {
            panic!("disabled model should fail");
        };
        assert!(err.to_string().contains("disabled"));

        for (field, config) in [
            (
                "model name",
                ModelConfig {
                    model: String::new(),
                    ..model_config("openai", "gpt-5")
                },
            ),
            (
                "model family",
                ModelConfig {
                    family: String::new(),
                    ..model_config("openai", "gpt-5")
                },
            ),
            (
                "api_base",
                ModelConfig {
                    api_base: String::new(),
                    ..model_config("openai", "gpt-5")
                },
            ),
            (
                "api_key",
                ModelConfig {
                    api_key: String::new(),
                    ..model_config("openai", "gpt-5")
                },
            ),
        ] {
            let Err(err) = config.model(client.clone()) else {
                panic!("{field} should fail");
            };
            let err = err.to_string();
            assert!(err.contains(field), "{field}: {err}");
        }

        let Err(err) = model_config("unknown", "m").model(client.clone()) else {
            panic!("unsupported family should fail");
        };
        assert!(err.to_string().contains("unsupported model family"));

        let mut gemini = model_config("gemini", "gemini-2.5-pro");
        gemini.context_window = 123;
        gemini.max_output = 45;
        let model = gemini.model(client.clone()).unwrap();
        assert_eq!(model.model_name(), "gemini-2.5-pro");
        assert_eq!(model.labels, vec!["gemini-2.5-pro"]);
        assert_eq!(model.context_window, 123);
        assert_eq!(model.max_output, 45);

        let mut anthropic = model_config("anthropic", "claude-sonnet-4-5");
        anthropic.labels = vec!["pro".to_string(), "primary".to_string()];
        anthropic.bearer_auth = true;
        anthropic.stream = true;
        anthropic.effort = Some(ModelEffort::High);
        let model = anthropic.model(client.clone()).unwrap();
        assert_eq!(model.model_name(), "claude-sonnet-4-5");
        assert_eq!(model.labels, vec!["pro", "primary"]);

        let model = model_config("openai", "gpt-5")
            .model(client.clone())
            .unwrap();
        assert_eq!(model.model_name(), "gpt-5");
        let model = model_config("openai", "deepseek-chat")
            .model(client)
            .unwrap();
        assert_eq!(model.model_name(), "deepseek-chat");
    }

    #[test]
    fn models_registry_clones_names_replaces_labels_and_loads_configs() {
        let models = Models::default();
        models.set_model(test_model("flash-v1").with_labels(vec!["FAST".into()]));
        assert!(models.contains("fast"));
        assert_eq!(
            models.model_names(),
            BTreeSet::from(["flash-v1".to_string()])
        );

        models.set("flash".to_string(), test_model("flash-v2"));
        assert!(models.contains("flash"));
        assert_eq!(models.get("FLASH").unwrap().model_name(), "flash-v2");
        assert_eq!(
            models.model_names(),
            BTreeSet::from(["flash-v1".to_string(), "flash-v2".to_string()])
        );

        models.set("primary".to_string(), test_model("primary-v2"));
        assert_eq!(models.get_model().unwrap().model_name(), "primary-v2");

        let cloned = Models::from_clone(&models);
        assert_eq!(cloned.get("primary").unwrap().model_name(), "primary-v2");
        assert_eq!(
            cloned.resolve("missing").unwrap().model_name(),
            "primary-v2"
        );

        let configs = vec![
            ModelConfig {
                labels: vec!["primary".to_string()],
                ..model_config("openai", "gpt-5")
            },
            ModelConfig {
                disabled: true,
                ..model_config("openai", "disabled")
            },
        ];
        let loaded = Models::from_configs(&configs, http_client());
        assert!(loaded.contains("primary"));
        assert!(!loaded.contains("disabled"));
        assert_eq!(loaded.get_model().unwrap().model_name(), "gpt-5");
    }

    #[tokio::test]
    async fn model_completion_placeholders_and_mock_tool_calls_are_stable() {
        let not_implemented = Model::not_implemented();
        assert_eq!(not_implemented.model_name(), "not_implemented");
        let err = not_implemented
            .completion(CompletionRequest::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not implemented"));

        let mock = Model::mock_implemented().with_labels(vec!["mock".into()]);
        assert_eq!(mock.model_name(), "not_implemented");
        let output = mock
            .completion(CompletionRequest {
                prompt: "{\"q\":\"anda\"}".to_string(),
                tools: vec![FunctionDefinition {
                    name: "lookup".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(output.content, "{\"q\":\"anda\"}");
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "lookup");
        assert_eq!(output.tool_calls[0].args["q"], "anda");

        let output = mock
            .completion(CompletionRequest {
                prompt: String::new(),
                tools: vec![FunctionDefinition {
                    name: "lookup".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(output.tool_calls.is_empty());
    }

    #[test]
    fn streaming_json_event_parser_accepts_bom_sse_ndjson_and_arrays() {
        let events = parse_streaming_json_events::<serde_json::Value>(
            b"\xef\xbb\xbfdata: {\"a\":1}\n\ndata: [DONE]\n\n",
            "test-model",
        )
        .unwrap();
        assert_eq!(events, vec![serde_json::json!({"a": 1})]);

        let events = parse_streaming_json_events::<serde_json::Value>(
            b"{\"a\":1}\n{\"b\":2}\n[DONE]\n",
            "test-model",
        )
        .unwrap();
        assert_eq!(
            events,
            vec![serde_json::json!({"a": 1}), serde_json::json!({"b": 2})]
        );

        let events =
            parse_streaming_json_events::<serde_json::Value>(br#"[{"a":1},{"b":2}]"#, "test-model")
                .unwrap();
        assert_eq!(
            events,
            vec![serde_json::json!({"a": 1}), serde_json::json!({"b": 2})]
        );
    }

    #[tokio::test]
    async fn streaming_reader_ignores_mislabelled_content_encoding() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        let (endpoint, _) = spawn_retry_mock_server(vec![MockHttpResponse {
            status: StatusCode::OK,
            headers,
            body: b"data: {\"a\":1}\n\ndata: [DONE]\n\n".to_vec(),
        }])
        .await;
        let client = request_client_builder()
            .https_only(false)
            .no_proxy()
            .build()
            .unwrap();
        let response = client.get(endpoint).send().await.unwrap();

        let events = read_sse_json_events::<serde_json::Value>(response, "test-model")
            .await
            .unwrap();

        assert_eq!(events, vec![serde_json::json!({"a": 1})]);
    }

    #[tokio::test]
    async fn streaming_reader_returns_after_done_before_late_body_error() {
        let endpoint = spawn_truncated_sse_after_done_server().await;
        let client = request_client_builder()
            .https_only(false)
            .no_proxy()
            .build()
            .unwrap();
        let response = client.get(endpoint).send().await.unwrap();

        let events = read_sse_json_events::<serde_json::Value>(response, "test-model")
            .await
            .unwrap();

        assert_eq!(events, vec![serde_json::json!({"a": 1})]);
    }

    #[test]
    fn sse_done_detection_is_line_anchored() {
        assert!(body_contains_sse_done(b"data: [DONE]\n\n", 0));
        assert!(body_contains_sse_done(
            b"data: {\"a\":1}\n\ndata: [DONE]\n\n",
            0
        ));
        // The marker text inside generated JSON content must not terminate
        // the stream.
        assert!(!body_contains_sse_done(
            b"data: {\"text\":\"sse ends with data: [DONE]\"}\n\n",
            0
        ));
    }

    #[tokio::test]
    async fn streaming_reader_is_not_truncated_by_done_marker_in_content() {
        let endpoint = spawn_sse_with_done_marker_in_content_server().await;
        let client = request_client_builder()
            .https_only(false)
            .no_proxy()
            .build()
            .unwrap();
        let response = client.get(endpoint).send().await.unwrap();

        let events = read_sse_json_events::<serde_json::Value>(response, "test-model")
            .await
            .unwrap();

        assert_eq!(
            events,
            vec![
                serde_json::json!({"text": "sse ends with data: [DONE]"}),
                serde_json::json!({"b": 2})
            ]
        );
    }

    #[test]
    fn retry_after_parses_seconds_and_http_date() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, HeaderValue::from_static("42"));
        assert_eq!(
            retry_after_duration(&headers),
            Some(Duration::from_secs(42))
        );

        let when = chrono::Utc::now() + chrono::Duration::seconds(90);
        headers.insert(
            http::header::RETRY_AFTER,
            HeaderValue::from_str(&when.to_rfc2822()).unwrap(),
        );
        let parsed = retry_after_duration(&headers).expect("http-date should parse");
        assert!(parsed <= Duration::from_secs(90));
        assert!(parsed >= Duration::from_secs(80));

        // A date in the past yields no delay hint.
        let when = chrono::Utc::now() - chrono::Duration::seconds(90);
        headers.insert(
            http::header::RETRY_AFTER,
            HeaderValue::from_str(&when.to_rfc2822()).unwrap(),
        );
        assert_eq!(retry_after_duration(&headers), None);

        headers.insert(
            http::header::RETRY_AFTER,
            HeaderValue::from_static("not-a-date"),
        );
        assert_eq!(retry_after_duration(&headers), None);
    }

    #[tokio::test]
    async fn custom_client_streaming_decode_errors_are_retryable() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert(
            http::header::CONTENT_ENCODING,
            HeaderValue::from_static("gzip"),
        );
        let (endpoint, _) = spawn_retry_mock_server(vec![MockHttpResponse {
            status: StatusCode::OK,
            headers,
            body: b"data: {\"a\":1}\n\ndata: [DONE]\n\n".to_vec(),
        }])
        .await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let response = streaming_completion_request(client.get(endpoint))
            .send()
            .await
            .unwrap();

        let err = read_sse_json_events::<serde_json::Value>(response, "test-model")
            .await
            .unwrap_err();

        assert!(err.to_string().contains("error decoding response body"));
        assert!(is_retryable_box_error(&err));
    }

    #[tokio::test]
    async fn completion_request_retry_once_and_exposes_retry_signal() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, HeaderValue::from_static("60"));
        let (endpoint, state) = spawn_retry_mock_server(vec![
            MockHttpResponse {
                status: StatusCode::TOO_MANY_REQUESTS,
                headers,
                body: b"rate limited".to_vec(),
            },
            MockHttpResponse {
                status: StatusCode::OK,
                headers: HeaderMap::new(),
                body: b"ok".to_vec(),
            },
        ])
        .await;
        let client = http_client();

        let body = execute_completion_request_with_retry(
            "retry-test",
            || client.post(&endpoint),
            |response| async { read_completion_response_bytes(response, "retry-test").await },
        )
        .await
        .unwrap();

        assert_eq!(&body[..], b"ok");
        assert_eq!(retry_count(&state), 2);

        let mut headers = HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, HeaderValue::from_static("45"));
        let (endpoint, state) = spawn_retry_mock_server(vec![
            MockHttpResponse {
                status: StatusCode::TOO_MANY_REQUESTS,
                headers: headers.clone(),
                body: b"first limit".to_vec(),
            },
            MockHttpResponse {
                status: StatusCode::TOO_MANY_REQUESTS,
                headers,
                body: b"still limited".to_vec(),
            },
        ])
        .await;
        let err = execute_completion_request_with_retry(
            "retry-test",
            || client.post(&endpoint),
            |response| async { read_completion_response_bytes(response, "retry-test").await },
        )
        .await
        .unwrap_err();
        let err_ref = err.as_ref() as &(dyn Error + 'static);

        assert_eq!(retry_count(&state), 2);
        assert!(is_retryable_box_error(&err));
        assert_eq!(
            model_error_status(err_ref),
            Some(StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(
            model_error_retry_after(err_ref),
            Some(Duration::from_secs(45))
        );

        let (endpoint, state) = spawn_retry_mock_server(vec![MockHttpResponse {
            status: StatusCode::BAD_REQUEST,
            headers: HeaderMap::new(),
            body: b"bad request".to_vec(),
        }])
        .await;
        let err = execute_completion_request_with_retry(
            "retry-test",
            || client.post(&endpoint),
            |response| async { read_completion_response_bytes(response, "retry-test").await },
        )
        .await
        .unwrap_err();

        assert_eq!(retry_count(&state), 1);
        assert!(!is_retryable_box_error(&err));
    }
}
