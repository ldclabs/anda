//! Anthropic Claude API client implementation for Anda Engine
//!
//! This module provides integration with Anthropic's Claude API, including:
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
// Main Anthropic Client
// ================================================================
const API_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";

pub static CLAUDE_SONNET_4: &str = "claude-sonnet-4-6";
pub static CLAUDE_OPUS_4: &str = "claude-opus-4-6";

/// Anthropic Claude API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    api_version: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new Anthropic client instance with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - Anthropic API key for authentication
    /// * `endpoint` - Optional custom API endpoint
    ///
    /// # Returns
    /// Configured Anthropic client instance
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
            api_version: API_VERSION.to_string(),
            http: request_client_builder()
                .build()
                .expect("Anthropic reqwest client should build"),
        }
    }

    /// Sets a custom HTTP client for the client
    pub fn with_client(self, http: reqwest::Client) -> Self {
        Self {
            endpoint: self.endpoint,
            api_key: self.api_key,
            api_version: self.api_version,
            http,
        }
    }

    pub fn with_api_version(mut self, api_version: String) -> Self {
        self.api_version = api_version;
        self
    }

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
    }

    /// Creates a new completion model instance
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(
            self.clone(),
            if model.is_empty() {
                CLAUDE_SONNET_4
            } else {
                model
            },
        )
    }
}

/// Completion model wrapper for Anthropic Claude API
#[derive(Clone)]
pub struct CompletionModel {
    /// Anthropic client instance
    client: Client,
    /// Default request template
    default_request: types::CreateMessageParams,
    /// Model identifier
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - Anthropic client instance
    /// * `model` - Model identifier string
    pub fn new(client: Client, model: &str) -> Self {
        let default_request = types::CreateMessageParams {
            max_tokens: 65535,
            ..Default::default()
        };
        Self {
            client,
            default_request,
            model: model.to_string(),
        }
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, req: types::CreateMessageParams) -> Self {
        self.default_request = req;
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

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let mut creq = self.default_request.clone();
        creq.model = model.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut raw_history: Vec<Json> = Vec::new();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                creq.system = Some(req.instructions);
            }

            for msg in req.raw_history {
                creq.messages
                    .push(serde_json::from_value::<types::Message>(msg)?);
            }

            for msg in req.chat_history {
                let val = types::Message::from(msg);
                raw_history.push(serde_json::to_value(&val)?);
                creq.messages.push(val);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let msg = types::Message::from(msg);
                raw_history.push(serde_json::to_value(&msg)?);
                creq.messages.push(msg);
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
                let msg = types::Message::from(msg);
                raw_history.push(serde_json::to_value(&msg)?);
                creq.messages.push(msg);
            }

            if let Some(temperature) = req.temperature {
                creq.temperature = Some(temperature as f32);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                creq.max_tokens = max_tokens as u32;
            }

            if let Some(stop) = req.stop {
                creq.stop_sequences = Some(stop);
            }

            if !req.tools.is_empty() {
                creq.tools = Some(req.tools.into_iter().map(|v| v.into()).collect());
                if req.tool_choice_required {
                    creq.tool_choice = Some(types::ToolChoice::Any);
                } else {
                    creq.tool_choice = Some(types::ToolChoice::Auto);
                }
            }

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&creq)
            {
                log::debug!(request = val; "Anthropic completions request");
            }

            let response = client.post("/messages").json(&creq).send().await?;

            creq.system = None; // avoid logging tedious instructions
            if response.status().is_success() {
                let text = response.text().await?;

                match serde_json::from_str::<types::CreateMessageResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            log::debug!(
                                model = model,
                                request:serde = creq,
                                messages:serde = raw_history,
                                response:serde = res;
                                "Anthropic completions response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                model = model,
                                request:serde = creq,
                                messages:serde = raw_history,
                                response:serde = res;
                                "Anthropic completions maybe failed");
                        }

                        res.try_into(raw_history, chat_history)
                    }
                    Err(err) => Err(format!(
                        "Anthropic {} completions error: {}, body: {}",
                        model, err, text
                    )
                    .into()),
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = model,
                    request:serde = creq,
                    messages:serde = raw_history;
                    "Anthropic completions request failed: {status}, body: {msg}",
                );
                Err(format!("Anthropic {} completions error: {}", model, msg).into())
            }
        })
    }
}
