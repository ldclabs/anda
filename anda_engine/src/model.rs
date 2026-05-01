//! Model provider integration and label-based routing.
//!
//! This module adapts provider-specific completion APIs to the common
//! [`CompletionRequest`] and [`AgentOutput`] contract used by Anda agents.
//! Built-in providers currently include OpenAI-compatible APIs, Anthropic, and
//! Google Gemini.
//!
//! The [`Models`] registry maps model labels such as `primary`, `fallback`,
//! `pro`, `flash`, or `lite` to concrete [`Model`] instances. Labels let agents
//! request capability tiers without hard-coding provider model names.
//!
//! Custom providers can implement [`CompletionFeaturesDyn`] and be wrapped with
//! [`Model::with_completer`].

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CONTENT_TYPE_JSON, CompletionRequest, Json, Message, ToolCall,
};
use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    sync::Arc,
};

pub mod anthropic;
pub mod gemini;
pub mod openai;

pub use reqwest;
pub use reqwest::Proxy;

use crate::APP_USER_AGENT;

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
    /// labels include `primary`, `fallback`, `pro`, `flash`, and `lite`.
    #[serde(default)]
    pub labels: Vec<String>,

    #[serde(default)]
    pub context_window: usize,

    #[serde(default)]
    pub max_output: usize,

    /// Skips this model when loading a list of configs.
    #[serde(default)]
    pub disabled: bool,

    /// Sends Anthropic credentials with bearer authentication instead of the
    /// provider-specific API-key header.
    #[serde(default)]
    pub bearer_auth: bool,
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
                    .completion_model(&self.model),
            )),
            "anthropic" => {
                let mut cli = anthropic::Client::new(&self.api_key, Some(self.api_base.clone()))
                    .with_client(http_client);
                if self.bearer_auth {
                    cli = cli.with_bearer_auth(true);
                }
                Model::with_completer(Arc::new(cli.completion_model(&self.model)))
            }
            "openai" => Model::with_completer(Arc::new(
                openai::Client::new(&self.api_key, Some(self.api_base.clone()))
                    .with_client(http_client)
                    .completion_model_v2(&self.model),
            )),
            _ => return Err(format!("unsupported model family: {}", self.family).into()),
        };

        let labels = if self.labels.is_empty() {
            vec![self.model.clone()]
        } else {
            self.labels.clone()
        };
        model.context_window = self.context_window;
        model.max_output = self.max_output;
        Ok(model.with_labels(labels))
    }

    #[deprecated(
        since = "0.11.22",
        note = "use the `model` method which returns a Result and allows error handling instead of silently returning a not_implemented model"
    )]
    pub fn build_model(&self, http_client: reqwest::Client) -> Model {
        self.model(http_client).unwrap_or_else(|err| {
            eprintln!("warning: failed to build model from config: {err}");
            Model::not_implemented()
        })
    }
}

/// Thread-safe model registry used by the engine.
///
/// It maintains three layers:
/// - `model`: the primary default model for general requests
/// - `models`: a label-based map for selecting specific models
/// - `fallback_model`: a safety fallback when primary lookup is missing
///
/// The dedicated primary and fallback slots can be set explicitly via
/// [`Models::set_model`] and [`Models::set_fallback_model`], or derived from the
/// special labels `primary` and `fallback` in the label map. This keeps direct
/// lookup (`get`) separate from default-routing (`get_model`).
pub struct Models {
    model: ArcSwap<Option<Model>>,
    models: ArcSwap<HashMap<String, Vec<Model>>>,
    fallback_model: ArcSwap<Option<Model>>,
}

impl Default for Models {
    fn default() -> Self {
        Self {
            model: ArcSwap::new(Arc::new(None)),
            models: ArcSwap::new(Arc::new(HashMap::new())),
            fallback_model: ArcSwap::new(Arc::new(None)),
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
            fallback_model: ArcSwap::new(other.fallback_model.load_full()),
        }
    }

    /// Builds a registry from model configs by registering every resolved label.
    pub fn from_configs(configs: &[ModelConfig], http_client: reqwest::Client) -> Self {
        let models = Self::default();
        for config in configs {
            if let Ok(model) = config.model(http_client.clone()) {
                for label in &model.labels {
                    models.set(label.clone(), model.clone());
                }
            }
        }
        models
    }

    /// Returns whether a label exists in the direct lookup table.
    pub fn contains(&self, label: &str) -> bool {
        self.models.load().contains_key(label)
    }

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

    /// Sets the fallback model used when primary lookup fails without mutating
    /// the label map.
    pub fn set_fallback_model(&self, model: Model) {
        self.inner_set(model.labels.clone(), model.clone());
        self.fallback_model.store(Arc::new(Some(model)));
    }

    /// Inserts or updates a single labeled model.
    ///
    /// The special labels `primary` and `fallback` also update the dedicated
    /// routing slots. If no primary exists yet, any inserted model is promoted
    /// to become the primary default.
    pub fn set(&self, label: String, model: Model) {
        self.inner_set(vec![label], model);
    }

    fn inner_set(&self, labels: Vec<String>, model: Model) {
        if self.model.load().is_none() {
            self.model.store(Arc::new(Some(model.clone())));
        }

        let model_name = model.model_name();
        let mut models = self.models.load().as_ref().clone();
        for label in labels {
            if label == "primary" {
                self.model.store(Arc::new(Some(model.clone())));
            } else if label == "fallback" {
                self.fallback_model.store(Arc::new(Some(model.clone())));
            }

            match models.entry(label) {
                Entry::Vacant(e) => {
                    e.insert(vec![model.clone()]);
                }
                Entry::Occupied(mut e) => {
                    e.get_mut().retain(|m| m.model_name() != model_name);
                    e.get_mut().push(model.clone());
                }
            }
        }

        self.models.store(Arc::new(models));
    }

    /// Returns a model by label if it exists.
    ///
    /// This is a direct lookup only and never falls back to the primary or
    /// fallback slots.
    pub fn get(&self, label: &str) -> Option<Model> {
        self.models
            .load()
            .get(label)
            .and_then(|v| v.last().cloned())
    }

    /// Returns the primary model if available; otherwise returns the fallback
    /// model, and finally any remaining labeled model.
    pub fn get_model(&self) -> Option<Model> {
        if let Some(m) = self.model.load().as_ref() {
            return Some(m.clone());
        }
        if let Some(m) = self.fallback_model.load().as_ref() {
            return Some(m.clone());
        }
        self.models
            .load()
            .values()
            .next()
            .and_then(|v| v.last().cloned())
    }

    /// Returns the configured fallback model if one exists.
    pub fn fallback_model(&self) -> Option<Model> {
        self.fallback_model.load().as_ref().clone()
    }

    /// Resolves a model for label-aware routing.
    ///
    /// Resolution order is:
    /// - the exact label match when `label` is non-empty
    /// - the configured fallback model
    /// - the default routing result from [`Models::get_model`]
    pub fn resolve(&self, label: &str) -> Option<Model> {
        if label.is_empty() {
            return self.get_model();
        }
        self.get(label)
            .or_else(|| self.fallback_model())
            .or_else(|| {
                self.models
                    .load()
                    .values()
                    .next()
                    .and_then(|v| v.last().cloned())
            })
    }
}

/// Object-safe completion provider interface.
pub trait CompletionFeaturesDyn: Send + Sync + 'static {
    /// Performs a completion request and returns the agent-facing output.
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>>;

    /// Returns the provider model name used for diagnostics and usage reports.
    fn model_name(&self) -> String;

    /// Prunes a raw-history message in-place to reduce token usage.
    ///
    /// `raw_history` stores provider-native JSON (Anthropic blocks, Gemini parts,
    /// OpenAI items, DeepSeek chat messages, …). Each provider must know how to
    /// shrink non-text payloads (tool calls, tool outputs, images, files) without
    /// breaking the request contract (e.g. preserving tool_call ↔ tool_output
    /// pairing and required fields).
    ///
    /// The default implementation falls back to the unified [`Message`] shape
    /// for backward compatibility with non-provider producers (mocks, tests,
    /// custom integrations).
    ///
    /// Returns the number of items pruned (0 if the message was already minimal
    /// or the shape was not recognized).
    fn prune_raw_message(&self, value: &mut Json) -> usize {
        let Ok(mut msg) = serde_json::from_value::<Message>(value.clone()) else {
            return 0;
        };
        let pruned = msg.prune_content();
        if pruned > 0
            && let Ok(raw) = serde_json::to_value(&msg)
        {
            *value = raw;
        }
        pruned
    }
}

/// Standard placeholder text for content pruned from raw history.
///
/// Kept identical in wording to [`anda_core::Message::prune_content`] so the
/// model observes a consistent marker regardless of provider.
pub(crate) fn pruned_placeholder(count: usize) -> String {
    format!(
        "[{} items (tool calls or files) pruned due to limits]",
        count
    )
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

    /// Prunes a raw-history message in-place using the underlying provider's
    /// knowledge of its own JSON shape.
    pub fn prune_raw_message(&self, value: &mut Json) -> usize {
        self.completer.prune_raw_message(value)
    }
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
                        Some(
                            http::StatusCode::REQUEST_TIMEOUT
                            | http::StatusCode::TOO_MANY_REQUESTS
                            | http::StatusCode::BAD_GATEWAY
                            | http::StatusCode::SERVICE_UNAVAILABLE
                            | http::StatusCode::GATEWAY_TIMEOUT,
                        ) => req_rep.retryable(),
                        _ => req_rep.success(),
                    }
                }),
        )
        .http2_keep_alive_interval(Some(Duration::from_secs(25)))
        .http2_keep_alive_timeout(Duration::from_secs(15))
        .http2_keep_alive_while_idle(true)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(300))
        .gzip(true)
        .user_agent(APP_USER_AGENT)
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            let ct: http::HeaderValue = http::HeaderValue::from_static(CONTENT_TYPE_JSON);
            headers.insert(http::header::CONTENT_TYPE, ct.clone());
            headers.insert(http::header::ACCEPT, ct);
            headers
        })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn models_default_is_empty() {
        let models = Models::default();

        assert!(models.get_model().is_none());
        assert!(models.get("missing").is_none());
        assert!(models.resolve("missing").is_none());
        assert!(models.fallback_model().is_none());
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
        assert!(models.get("primary").is_none());
        assert!(models.fallback_model().is_none());
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
    fn fallback_is_used_when_primary_or_label_missing() {
        let models = Models::default();
        models.set_fallback_model(test_model("fallback"));

        assert_eq!(
            models
                .get_model()
                .expect("fallback should be returned when primary is missing")
                .model_name(),
            "fallback"
        );
        assert_eq!(
            models
                .resolve("unknown")
                .expect("fallback should be returned when label is missing")
                .model_name(),
            "fallback"
        );
        assert_eq!(
            models
                .fallback_model()
                .expect("fallback slot should be set")
                .model_name(),
            "fallback"
        );
    }

    #[test]
    fn resolve_prefers_exact_label_then_fallback_then_default() {
        let models = Models::default();
        models.set_model(test_model("primary"));
        models.set_fallback_model(test_model("fallback"));
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
                .expect("missing label should use fallback")
                .model_name(),
            "fallback"
        );
        assert_eq!(
            models
                .resolve("")
                .expect("empty label should use default routing")
                .model_name(),
            "primary"
        );
    }
}
