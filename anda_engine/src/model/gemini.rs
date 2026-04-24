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
use serde_json::json;

use super::{CompletionFeaturesDyn, pruned_placeholder, request_client_builder};
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

    /// Prune a Gemini-native content JSON in-place.
    ///
    /// Gemini messages have shape `{role?, parts: [Part, ...]}` where each
    /// part is one of `text`, `functionCall`, `functionResponse`, `fileData`,
    /// `inlineData`, … distinguished by which field is present.
    ///
    /// `functionCall` and `functionResponse` parts must stay paired across
    /// turns (each `functionResponse.name` refers to a prior `functionCall`),
    /// so pruning must not drop either side — otherwise the API rejects the
    /// request. We therefore preserve those parts' envelope (name / id) and
    /// only shrink their heavy payload (`args` / `response`). Other
    /// non-text parts (files, inline data, …) are dropped and summarized by
    /// a single placeholder text part.
    fn prune_raw_message(&self, value: &mut Json) -> usize {
        let Some(obj) = value.as_object_mut() else {
            return 0;
        };
        let Some(parts) = obj.get_mut("parts").and_then(|v| v.as_array_mut()) else {
            return 0;
        };
        let mut pruned = 0usize;
        let mut dropped = 0usize;
        parts.retain_mut(|part| {
            let Some(map) = part.as_object_mut() else {
                dropped += 1;
                return false;
            };
            if map.get("text").map(|t| t.is_string()).unwrap_or(false) {
                return true;
            }
            if let Some(fc) = map.get_mut("functionCall")
                && let Some(fc_obj) = fc.as_object_mut()
            {
                if let Some(args) = fc_obj.get_mut("args") {
                    let already = matches!(args, Json::Object(m) if m.is_empty());
                    if !already {
                        *args = json!({});
                        pruned += 1;
                    }
                }
                return true;
            }
            if let Some(fr) = map.get_mut("functionResponse")
                && let Some(fr_obj) = fr.as_object_mut()
            {
                if let Some(resp) = fr_obj.get_mut("response") {
                    let placeholder = pruned_placeholder(1);
                    let already = matches!(resp, Json::Object(m)
                        if m.get("output").and_then(|v| v.as_str()) == Some(placeholder.as_str())
                            && m.len() == 1);
                    if !already {
                        *resp = json!({ "output": placeholder });
                        pruned += 1;
                    }
                }
                return true;
            }
            dropped += 1;
            false
        });
        if dropped > 0 {
            parts.push(json!({ "text": pruned_placeholder(dropped) }));
            pruned += dropped;
        }
        pruned
    }

    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let mut greq = self.default_request.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut raw_history: Vec<Json> = Vec::new();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                greq.system_instruction = Some(types::Content {
                    role: Some(types::Role::Model),
                    parts: vec![types::Part {
                        data: types::PartKind::Text(req.instructions),
                        ..Default::default()
                    }],
                });
            };

            greq.contents.append(&mut req.raw_history);
            for msg in req.chat_history {
                let val = types::Content::from(msg);
                let val = serde_json::to_value(val)?;
                raw_history.push(val.clone());
                greq.contents.push(val);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let val = types::Content::from(msg);
                let val = serde_json::to_value(val)?;
                raw_history.push(val.clone());
                greq.contents.push(val);
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
                let val = types::Content::from(msg);
                let val = serde_json::to_value(val)?;
                raw_history.push(val.clone());
                greq.contents.push(val);
            }

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
                log::debug!(request = val; "Completion request");
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
                                "Completion response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                model = model,
                                request:serde = greq,
                                messages:serde = raw_history,
                                response:serde = res;
                                "Completion maybe failed");
                        }

                        res.try_into(raw_history, chat_history)
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
                    request:serde = greq,
                    messages:serde = raw_history;
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
