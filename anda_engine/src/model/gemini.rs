//! Gemini Moonshot API client implementation for Anda Engine
//!
//! This module provides integration with Gemini's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, Json, Message,
    Resource,
};
use log::{Level::Debug, log_enabled};

use super::{CompletionFeaturesDyn, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

pub mod types;

// ================================================================
// Main Gemini Client
// ================================================================
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub static GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
pub static GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";

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
                GEMINI_2_5_PRO
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
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl CompletionFeatures for CompletionModel {
    async fn completion(
        &self,
        req: CompletionRequest,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(self, req).await
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut raw_history: Vec<Json> = Vec::new();
            let mut chat_history: Vec<Message> = Vec::new();
            let mut greq = types::GenerateContentRequest::default();

            if !req.instructions.is_empty() {
                greq.system_instruction = Some(types::Content {
                    role: Some(types::Role::Model),
                    parts: vec![types::Part {
                        data: types::PartKind::Text(req.instructions),
                        ..Default::default()
                    }],
                });
            };

            for msg in req.raw_history {
                greq.contents.push(serde_json::from_value(msg)?);
            }

            for msg in req.chat_history {
                let val = types::Content::from(msg);
                raw_history.push(serde_json::to_value(&val)?);
                greq.contents.push(val);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let msg = types::Content::from(msg);
                raw_history.push(serde_json::to_value(&msg)?);
                greq.contents.push(msg);
            }

            let mut content = req.content;
            if !req.prompt.is_empty() {
                content.push(req.prompt.into());
            }
            if !content.is_empty() {
                let msg = Message {
                    role: req.role.unwrap_or_else(|| "user".to_string()),
                    content,
                    timestamp: Some(timestamp),
                    ..Default::default()
                };

                chat_history.push(msg.clone());
                let msg = types::Content::from(msg);
                raw_history.push(serde_json::to_value(&msg)?);
                greq.contents.push(msg);
            }

            greq.generation_config.top_p = Some(0.95);
            if let Some(temperature) = req.temperature {
                greq.generation_config.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                greq.generation_config.max_output_tokens = Some(max_tokens as i32);
            }

            if let Some(output_schema) = req.output_schema {
                greq.generation_config.response_mime_type = Some("application/json".to_string());
                greq.generation_config.response_schema = Some(output_schema);
            }

            if let Some(stop) = req.stop {
                greq.generation_config.stop_sequences = Some(stop);
            }

            if !req.tools.is_empty() {
                greq.tools = vec![req.tools.into()];
                greq.tool_config = Some(types::ToolConfig::default());
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&greq)
            {
                log::debug!(request = val; "Gemini completions request");
            }

            let response = client
                .post(&format!("/{}:generateContent", model))
                .json(&greq)
                .send()
                .await?;

            greq.system_instruction = None; // avoid logging tedious instructions
            if response.status().is_success() {
                let text = response.text().await?;

                match serde_json::from_str::<types::GenerateContentResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            log::debug!(
                                model = model,
                                request:serde = greq,
                                messages:serde = raw_history,
                                response:serde = res;
                                "Gemini completions response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                model = model,
                                request:serde = greq,
                                messages:serde = raw_history,
                                response:serde = res;
                                "Gemini completions maybe failed");
                        }

                        res.try_into(raw_history, chat_history)
                    }
                    Err(err) => Err(format!(
                        "Gemini {} completions error: {}, body: {}",
                        model, err, text
                    )
                    .into()),
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = model,
                    request:serde = greq,
                    messages:serde = raw_history;
                    "Gemini completions request failed: {status}, body: {msg}",
                );
                Err(format!("Gemini {} completions error: {}", model, msg).into())
            }
        })
    }
}
