//! Model integration module for Anda Engine
//!
//! This module provides implementations for various AI model providers, including:
//! - OpenAI (completion and embedding models)
//! - DeepSeek (completion models)
//! - Cohere (embedding models)
//!
//! Each provider implementation includes:
//! - Client configuration and management
//! - API request/response handling
//! - Conversion to Anda's internal data structures
//!
//! The module is designed to be extensible, allowing easy addition of new model providers
//! while maintaining a consistent interface through the `CompletionFeaturesDyn` and
//! `EmbeddingFeaturesDyn` traits.

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CONTENT_TYPE_JSON, CompletionRequest, Embedding, ToolCall,
    Usage,
};
use arc_swap::ArcSwap;
use std::time::Duration;
use std::{collections::HashMap, sync::Arc};

pub mod cohere;
pub mod deepseek;
pub mod doubao;
pub mod gemini;
pub mod kimi;
pub mod mimo;
pub mod openai;
pub mod xai;

pub use reqwest;
pub use reqwest::Proxy;

use crate::APP_USER_AGENT;

/// Thread-safe model registry used by the engine.
///
/// It maintains three layers:
/// - `model`: the primary default model for general requests
/// - `models`: a label-based map for selecting specific models
/// - `fallback_model`: a safety fallback when primary lookup is missing
pub struct Models {
    model: ArcSwap<Option<Model>>,
    models: ArcSwap<HashMap<String, Model>>,
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
    /// Sets the primary default model.
    pub fn set_model(&self, model: Model) {
        self.model.store(Arc::new(Some(model)));
    }

    /// Replaces all labeled models.
    ///
    /// If no primary model is configured yet, the first model in the provided
    /// map is promoted to the primary default model.
    pub fn set_models(&self, models: HashMap<String, Model>) {
        if self.model.load().is_none()
            && let Some((_, m)) = models.iter().next() {
                self.model.store(Arc::new(Some(m.clone())));
            }
        self.models.store(Arc::new(models));
    }

    /// Inserts or updates a model by label.
    ///
    /// If no primary model is configured yet, this inserted model becomes the
    /// primary default model.
    pub fn set_model_by(&self, label: String, model: Model) {
        let mut models = self.models.load().as_ref().clone();
        models.insert(label, model.clone());
        self.models.store(Arc::new(models));
        if self.model.load().is_none() {
            self.model.store(Arc::new(Some(model)));
        }
    }

    /// Sets the fallback model used when primary lookup fails.
    pub fn set_fallback_model(&self, model: Model) {
        self.fallback_model.store(Arc::new(Some(model)));
    }

    /// Returns the primary model if available; otherwise returns the fallback model.
    pub fn get_model(&self) -> Option<Model> {
        if let Some(m) = self.model.load().as_ref() {
            return Some(m.clone());
        }
        if let Some(m) = self.fallback_model.load().as_ref() {
            return Some(m.clone());
        }
        None
    }

    /// Returns a model by label.
    ///
    /// If the label is not found, returns the fallback model when configured.
    pub fn get_model_by(&self, label: &str) -> Option<Model> {
        if label.is_empty() {
            return self.get_model();
        }
        if let Some(m) = self.models.load().get(label) {
            return Some(m.clone());
        }
        if let Some(m) = self.fallback_model.load().as_ref() {
            return Some(m.clone());
        }
        None
    }

    /// Returns the fallback model if configured.
    pub fn fallback_model(&self) -> Option<Model> {
        self.fallback_model.load().as_ref().clone()
    }
}

/// Trait for dynamic completion features that can be used across threads
pub trait CompletionFeaturesDyn: Send + Sync + 'static {
    /// Performs a completion request and returns a future with the agent's output
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>>;

    fn model_name(&self) -> String;
}

/// Trait for dynamic embedding features that can be used across threads
pub trait EmbeddingFeaturesDyn: Send + Sync + 'static {
    /// Returns the number of dimensions for the embedding model
    fn ndims(&self) -> usize;

    /// Embeds multiple texts and returns a future with the resulting embeddings
    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<(Vec<Embedding>, Usage), BoxError>>;

    /// Embeds a single query text and returns a future with the resulting embedding
    fn embed_query(&self, text: String) -> BoxPinFut<Result<(Embedding, Usage), BoxError>>;
}

/// A placeholder implementation for unimplemented features
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

impl EmbeddingFeaturesDyn for NotImplemented {
    fn ndims(&self) -> usize {
        0
    }

    fn embed(&self, _texts: Vec<String>) -> BoxPinFut<Result<(Vec<Embedding>, Usage), BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn embed_query(&self, _text: String) -> BoxPinFut<Result<(Embedding, Usage), BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

/// A mock implementation for testing purposes
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

impl EmbeddingFeaturesDyn for MockImplemented {
    fn ndims(&self) -> usize {
        384 // EMBED_MULTILINGUAL_LIGHT_V3
    }

    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<(Vec<Embedding>, Usage), BoxError>> {
        Box::pin(futures::future::ready(Ok((
            texts
                .into_iter()
                .map(|text| Embedding {
                    text,
                    vec: vec![0.0; 384],
                })
                .collect(),
            Usage::default(),
        ))))
    }

    fn embed_query(&self, _text: String) -> BoxPinFut<Result<(Embedding, Usage), BoxError>> {
        Box::pin(futures::future::ready(Ok((
            Embedding {
                text: "test".to_string(),
                vec: vec![0.0; 384],
            },
            Usage::default(),
        ))))
    }
}

/// Main model struct that combines embedding and completion capabilities
#[derive(Clone)]
pub struct Model {
    /// Embedding feature implementation
    pub embedder: Arc<dyn EmbeddingFeaturesDyn>,
    /// Completion feature implementation
    pub completer: Arc<dyn CompletionFeaturesDyn>,
}

impl Model {
    /// Creates a new Model with specified embedder and completer
    pub fn new(
        completer: Arc<dyn CompletionFeaturesDyn>,
        embedder: Arc<dyn EmbeddingFeaturesDyn>,
    ) -> Self {
        Self {
            embedder,
            completer,
        }
    }

    /// Creates a Model with only completion features
    pub fn with_completer(completer: Arc<dyn CompletionFeaturesDyn>) -> Self {
        Self {
            completer,
            embedder: Arc::new(NotImplemented),
        }
    }

    /// Creates a Model with unimplemented features (returns errors for all operations)
    pub fn not_implemented() -> Self {
        Self {
            completer: Arc::new(NotImplemented),
            embedder: Arc::new(NotImplemented),
        }
    }

    /// Creates a Model with mock implementations for testing
    pub fn mock_implemented() -> Self {
        Self {
            completer: Arc::new(MockImplemented),
            embedder: Arc::new(MockImplemented),
        }
    }

    pub fn model_name(&self) -> String {
        self.completer.model_name()
    }

    pub async fn completion(&self, req: CompletionRequest) -> Result<AgentOutput, BoxError> {
        self.completer.completion(req).await
    }

    pub fn ndims(&self) -> usize {
        self.embedder.ndims()
    }

    pub async fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<(Vec<Embedding>, Usage), BoxError> {
        self.embedder.embed(texts.into_iter().collect()).await
    }

    pub async fn embed_query(&self, text: &str) -> Result<(Embedding, Usage), BoxError> {
        self.embedder.embed_query(text.to_string()).await
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnyHost;

impl PartialEq<&str> for AnyHost {
    fn eq(&self, _other: &&str) -> bool {
        true
    }
}

/// Creates a new reqwest client builder with default settings
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

    #[derive(Clone)]
    struct TestEmbedder;

    impl EmbeddingFeaturesDyn for TestEmbedder {
        fn ndims(&self) -> usize {
            3
        }

        fn embed(
            &self,
            _texts: Vec<String>,
        ) -> BoxPinFut<Result<(Vec<Embedding>, Usage), BoxError>> {
            Box::pin(futures::future::ready(Ok((vec![], Usage::default()))))
        }

        fn embed_query(&self, _text: String) -> BoxPinFut<Result<(Embedding, Usage), BoxError>> {
            Box::pin(futures::future::ready(Ok((
                Embedding {
                    text: String::new(),
                    vec: vec![0.0; 3],
                },
                Usage::default(),
            ))))
        }
    }

    fn test_model(name: &'static str) -> Model {
        Model::new(Arc::new(TestCompleter { name }), Arc::new(TestEmbedder))
    }

    #[test]
    fn models_default_is_empty() {
        let models = Models::default();

        assert!(models.get_model().is_none());
        assert!(models.get_model_by("missing").is_none());
    }

    #[test]
    fn set_model_sets_primary_and_fallback() {
        let models = Models::default();
        models.set_model(test_model("primary"));

        assert_eq!(
            models
                .get_model()
                .expect("primary model should exist")
                .model_name(),
            "primary"
        );
        assert!(models.get_model_by("missing").is_none());
    }

    #[test]
    fn set_models_updates_map_and_promotes_one_when_primary_absent() {
        let models = Models::default();
        let mut map = HashMap::new();
        map.insert("a".to_string(), test_model("A"));
        map.insert("b".to_string(), test_model("B"));

        models.set_models(map);

        let primary = models
            .get_model()
            .expect("primary model should be promoted from set_models")
            .model_name();
        assert!(primary == "A" || primary == "B");

        assert_eq!(
            models
                .get_model_by("a")
                .expect("label a should exist")
                .model_name(),
            "A"
        );
        assert_eq!(
            models
                .get_model_by("b")
                .expect("label b should exist")
                .model_name(),
            "B"
        );
    }

    #[test]
    fn set_model_by_inserts_and_sets_primary_when_empty() {
        let models = Models::default();
        models.set_model_by("x".to_string(), test_model("X"));

        assert_eq!(
            models
                .get_model_by("x")
                .expect("label x should exist")
                .model_name(),
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
                .get_model_by("unknown")
                .expect("fallback should be returned when label is missing")
                .model_name(),
            "fallback"
        );
    }
}
