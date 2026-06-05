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

pub use anda_core::ModelEffort;

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
        let mut models = self.models.load().as_ref().clone();
        for mut label in labels {
            label.make_ascii_lowercase();
            if label == "primary" {
                self.model.store(Arc::new(Some(model.clone())));
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
        self.get(&label.to_ascii_lowercase())
            .or_else(|| self.get_model())
    }
}

/// Object-safe completion provider interface.
pub trait CompletionFeaturesDyn: Send + Sync + 'static {
    /// Performs a completion request and returns the agent-facing output.
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

pub(crate) async fn read_sse_json_events<T>(
    response: reqwest::Response,
    model: &str,
) -> Result<Vec<T>, BoxError>
where
    T: DeserializeOwned,
{
    let mut stream = response.bytes_stream();
    let mut pending = Vec::new();
    let mut data = String::new();
    let mut events = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| {
            format!(
                "Failed to read streaming completion response, model: {}, error: {}",
                model, err
            )
        })?;
        pending.extend_from_slice(&chunk);

        while let Some(pos) = pending.iter().position(|byte| *byte == b'\n') {
            let mut line = pending.drain(..=pos).collect::<Vec<_>>();
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            handle_sse_line(&line, &mut data, &mut events, model)?;
        }
    }

    if !pending.is_empty() {
        let line = std::mem::take(&mut pending);
        handle_sse_line(&line, &mut data, &mut events, model)?;
    }
    flush_sse_data(&mut data, &mut events, model)?;

    Ok(events)
}

fn handle_sse_line<T>(
    line: &[u8],
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
    if line.starts_with(b":") {
        return Ok(());
    }

    let Some(value) = line.strip_prefix(b"data:") else {
        return Ok(());
    };
    let value = value.strip_prefix(b" ").unwrap_or(value);
    let value = std::str::from_utf8(value).map_err(|err| {
        format!(
            "Invalid UTF-8 in streaming completion response, model: {}, error: {}",
            model, err
        )
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::FunctionDefinition;

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
    fn request_client_builder_and_sse_line_parser_cover_success_and_error_paths() {
        let _client = request_client_builder().no_proxy().build().unwrap();
        assert!(AnyHost == "anything");

        let mut data = String::new();
        let mut events = Vec::<serde_json::Value>::new();

        handle_sse_line(b": keep-alive", &mut data, &mut events, "test-model").unwrap();
        handle_sse_line(b"event: ignored", &mut data, &mut events, "test-model").unwrap();
        handle_sse_line(b"data: {\"a\":", &mut data, &mut events, "test-model").unwrap();
        handle_sse_line(b"data: 1}", &mut data, &mut events, "test-model").unwrap();
        handle_sse_line(b"", &mut data, &mut events, "test-model").unwrap();
        assert_eq!(events, vec![serde_json::json!({"a": 1})]);

        handle_sse_line(b"data: [DONE]", &mut data, &mut events, "test-model").unwrap();
        handle_sse_line(b"", &mut data, &mut events, "test-model").unwrap();
        assert!(data.is_empty());
        assert_eq!(events.len(), 1);

        let err = handle_sse_line(b"data: \xff", &mut data, &mut events, "test-model").unwrap_err();
        assert!(err.to_string().contains("Invalid UTF-8"));

        data.clear();
        data.push_str("{bad json");
        let err =
            flush_sse_data::<serde_json::Value>(&mut data, &mut events, "test-model").unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid streaming completion event")
        );
        assert!(err.to_string().contains("{bad json"));
    }
}
