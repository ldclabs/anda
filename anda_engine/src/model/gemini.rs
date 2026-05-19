//! Gemini Moonshot API client implementation for Anda Engine
//!
//! This module provides integration with Gemini's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, Message, Resource,
};
use log::{Level::Debug, log_enabled};
use serde_json::json;

use super::{CompletionFeaturesDyn, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

pub mod types;

// ================================================================
// Main Gemini Client
// ================================================================
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub static DEFAULT_COMPLETION_MODEL: &str = "gemini-flash-latest";

/// Gemini API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new Gemini client instance with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - Gemini API key for authentication
    ///
    /// # Returns
    /// Configured Gemini client instance
    pub fn new(api_key: &str, endpoint: Option<String>) -> Self {
        let endpoint = endpoint.unwrap_or_else(|| API_BASE_URL.to_string());
        let endpoint = if endpoint.is_empty() {
            API_BASE_URL.to_string()
        } else {
            endpoint
        };
        Self {
            endpoint,
            api_key: api_key.to_string(),
            http: request_client_builder()
                .build()
                .expect("Gemini reqwest client should build"),
        }
    }

    /// Sets a custom HTTP client for the client
    pub fn with_client(self, http: reqwest::Client) -> Self {
        Self {
            endpoint: self.endpoint,
            api_key: self.api_key,
            http,
        }
    }

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url).header("x-goog-api-key", &self.api_key)
    }

    /// Creates a new completion model instance using the default Gemini model
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(
            self.clone(),
            if model.is_empty() {
                DEFAULT_COMPLETION_MODEL
            } else {
                model
            },
        )
    }
}

/// Completion model wrapper for Gemini API
#[derive(Clone)]
pub struct CompletionModel {
    /// Gemini client instance
    client: Client,
    /// Default request template
    default_request: types::GenerateContentRequest,
    /// Model identifier
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - Gemini client instance
    /// * `model` - Model identifier string
    pub fn new(client: Client, model: &str) -> Self {
        let mut default_request = types::GenerateContentRequest::default();
        default_request.generation_config.top_p = Some(0.95);
        Self {
            client,
            default_request,
            model: model.to_string(),
        }
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, greq: types::GenerateContentRequest) -> Self {
        self.default_request = greq;
        self
    }
}

impl CompletionFeatures for CompletionModel {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    async fn completion(
        &self,
        req: CompletionRequest,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(self, req).await
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let mut r = self.default_request.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                r.system_instruction = Some(types::Content {
                    role: Some(types::Role::Model),
                    parts: vec![types::Part {
                        data: types::PartKind::Text(req.instructions),
                        ..Default::default()
                    }],
                });
            };

            r.contents.append(&mut req.raw_history);
            let skip_raw = r.contents.len();
            for msg in req.chat_history {
                let val = types::Content::from(msg);
                let val = serde_json::to_value(val)?;
                r.contents.push(val);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let val = types::Content::from(msg);
                let val = serde_json::to_value(val)?;
                r.contents.push(val);
            }

            let mut content = req.content;
            if !req.prompt.is_empty() {
                content.insert(0, req.prompt.into());
            }
            if !content.is_empty() {
                let msg = Message {
                    role: req.role.unwrap_or_else(|| "user".to_string()),
                    content,
                    timestamp: Some(timestamp),
                    ..Default::default()
                };

                chat_history.push(msg.clone());
                let val = types::Content::from(msg);
                let val = serde_json::to_value(val)?;
                r.contents.push(val);
            }

            if let Some(temperature) = req.temperature {
                r.generation_config.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                r.generation_config.max_output_tokens = Some(max_tokens as i32);
            }

            if let Some(output_schema) = req.output_schema {
                r.generation_config.response_mime_type = Some("application/json".to_string());
                r.generation_config.response_schema = Some(output_schema);
            }

            if let Some(stop) = req.stop {
                r.generation_config.stop_sequences = Some(stop);
            }

            if !req.tools.is_empty() {
                r.tools = vec![req.tools.into()];
                r.tool_config = Some(types::ToolConfig::default());
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&r)
            {
                log::debug!(request = val; "Completion request");
            }

            let response = client
                .post(&format!("/{}:generateContent", model))
                .json(&r)
                .send()
                .await
                .map_err(|err| {
                    format!(
                        "Failed to send completion request, model: {}, error: {}",
                        model, err
                    )
                })?;

            r.system_instruction = None; // avoid logging tedious instructions
            if response.status().is_success() {
                let text = response.text().await.map_err(|err| {
                    format!(
                        "Failed to read completion response, model: {}, error: {}",
                        model, err
                    )
                })?;

                match serde_json::from_str::<types::GenerateContentResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            log::debug!(
                                model = model,
                                request:serde = r,
                                response:serde = res;
                                "Completion response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                model = model,
                                request:serde = r,
                                response:serde = res;
                                "Completion maybe failed");
                        }
                        if skip_raw > 0 {
                            r.contents.drain(0..skip_raw);
                        }

                        res.try_into(
                            r.contents.into_iter().map(|v| json!(v)).collect(),
                            chat_history,
                        )
                    }
                    Err(err) => Err(format!(
                        "Invalid completion response, model: {}, error: {}, body: {}",
                        model, err, text
                    )
                    .into()),
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = model,
                    request:serde = r;
                    "Completion request failed: {status}, body: {msg}",
                );
                Err(format!(
                    "Completion failed, model: {}, error: {}, body: {}",
                    model, status, msg
                )
                .into())
            }
        })
    }
}
