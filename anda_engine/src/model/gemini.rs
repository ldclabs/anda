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
use reqwest::header::ACCEPT;
use serde_json::json;
use std::collections::BTreeMap;

use super::{CompletionFeaturesDyn, ModelEffort, read_sse_json_events, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

pub mod types;

impl From<ModelEffort> for types::ThinkingLevel {
    fn from(value: ModelEffort) -> Self {
        match value {
            ModelEffort::Minimal => Self::Minimal,
            ModelEffort::Low => Self::Low,
            ModelEffort::Medium => Self::Medium,
            ModelEffort::High => Self::High,
            ModelEffort::XHigh => Self::High,
        }
    }
}

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
        let default_request = types::GenerateContentRequest::default();
        Self {
            client,
            default_request,
            model: model.to_string(),
        }
    }

    /// Sets whether the completion request should run in streaming mode
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.default_request.stream = stream;
        self
    }

    /// Sets the default thinking effort for compatible models
    pub fn with_effort(mut self, effort: Option<ModelEffort>) -> Self {
        if let Some(effort) = effort {
            let thinking_config = self
                .default_request
                .generation_config
                .thinking_config
                .get_or_insert_with(types::ThinkingConfig::default);
            thinking_config.thinking_level = Some(effort.into());
        }
        self
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, greq: types::GenerateContentRequest) -> Self {
        self.default_request = greq;
        self
    }
}

fn append_gemini_parts(target: &mut Vec<types::Part>, parts: Vec<types::Part>) {
    for part in parts {
        match (target.last_mut(), part) {
            (
                Some(types::Part {
                    thought: last_thought,
                    thought_signature: last_signature,
                    data: types::PartKind::Text(last_text),
                }),
                types::Part {
                    thought,
                    thought_signature,
                    data: types::PartKind::Text(text),
                },
            ) if *last_thought == thought => {
                last_text.push_str(&text);
                if let Some(signature) = thought_signature {
                    *last_signature = Some(signature);
                }
            }
            (_, part) => target.push(part),
        }
    }
}

fn response_from_stream_chunks(
    chunks: Vec<types::GenerateContentResponse>,
) -> Result<types::GenerateContentResponse, BoxError> {
    let mut candidates = BTreeMap::<u32, types::Candidate>::new();
    let mut prompt_feedback = None;
    let mut usage_metadata = types::UsageMetadata::default();
    let mut model_version = None;
    let mut response_id = None;
    let mut model_status = None;

    for chunk in chunks {
        if chunk.prompt_feedback.is_some() {
            prompt_feedback = chunk.prompt_feedback;
        }
        if chunk.usage_metadata != types::UsageMetadata::default() {
            usage_metadata = chunk.usage_metadata;
        }
        if chunk.model_version.is_some() {
            model_version = chunk.model_version;
        }
        if chunk.response_id.is_some() {
            response_id = chunk.response_id;
        }
        if chunk.model_status.is_some() {
            model_status = chunk.model_status;
        }

        for candidate in chunk.candidates {
            let index = candidate.index.unwrap_or(0);
            match candidates.entry(index) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(candidate);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.content.role.is_none() {
                        existing.content.role = candidate.content.role;
                    }
                    append_gemini_parts(&mut existing.content.parts, candidate.content.parts);
                    if candidate.finish_reason.is_some() {
                        existing.finish_reason = candidate.finish_reason;
                    }
                    if candidate.safety_ratings.is_some() {
                        existing.safety_ratings = candidate.safety_ratings;
                    }
                    if candidate.citation_metadata.is_some() {
                        existing.citation_metadata = candidate.citation_metadata;
                    }
                    if candidate.token_count.is_some() {
                        existing.token_count = candidate.token_count;
                    }
                    if !candidate.grounding_attributions.is_empty() {
                        existing.grounding_attributions = candidate.grounding_attributions;
                    }
                    if candidate.grounding_metadata.is_some() {
                        existing.grounding_metadata = candidate.grounding_metadata;
                    }
                    if candidate.avg_logprobs.is_some() {
                        existing.avg_logprobs = candidate.avg_logprobs;
                    }
                    if candidate.logprobs_result.is_some() {
                        existing.logprobs_result = candidate.logprobs_result;
                    }
                    if candidate.url_context_metadata.is_some() {
                        existing.url_context_metadata = candidate.url_context_metadata;
                    }
                    if candidate.finish_message.is_some() {
                        existing.finish_message = candidate.finish_message;
                    }
                }
            }
        }
    }

    if candidates.is_empty() && prompt_feedback.is_none() {
        return Err("No streamed Gemini response".into());
    }

    Ok(types::GenerateContentResponse {
        candidates: candidates.into_values().collect(),
        prompt_feedback,
        usage_metadata,
        model_version,
        response_id,
        model_status,
    })
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

            let path = if r.stream {
                format!("/{}:streamGenerateContent?alt=sse", model)
            } else {
                format!("/{}:generateContent", model)
            };
            let mut request = client.post(&path).json(&r);
            if r.stream {
                request = request.header(ACCEPT, "text/event-stream");
            }
            let response = request.send().await.map_err(|err| {
                format!(
                    "Failed to send completion request, model: {}, error: {}",
                    model, err
                )
            })?;

            r.system_instruction = None; // avoid logging tedious instructions
            if response.status().is_success() {
                let res = if r.stream {
                    let chunks = read_sse_json_events(response, &model).await?;
                    response_from_stream_chunks(chunks)?
                } else {
                    let text = response.text().await.map_err(|err| {
                        format!(
                            "Failed to read completion response, model: {}, error: {}",
                            model, err
                        )
                    })?;

                    match serde_json::from_str::<types::GenerateContentResponse>(&text) {
                        Ok(res) => res,
                        Err(err) => {
                            return Err(format!(
                                "Invalid completion response, model: {}, error: {}, body: {}",
                                model, err, text
                            )
                            .into());
                        }
                    }
                };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_model_applies_default_effort() {
        let model = Client::new("test-key", Some("http://localhost".into()))
            .completion_model("gemini-3-pro")
            .with_effort(Some(ModelEffort::High));

        let thinking_config = model
            .default_request
            .generation_config
            .thinking_config
            .expect("thinking config should be configured");
        assert_eq!(
            thinking_config.thinking_level,
            Some(types::ThinkingLevel::High)
        );
    }

    #[test]
    fn aggregates_gemini_stream_chunks() {
        let chunks = vec![
            serde_json::from_value::<types::GenerateContentResponse>(json!({
                "candidates": [{
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hel"}]
                    }
                }],
                "usageMetadata": {"promptTokenCount": 2},
                "modelVersion": "gemini-flash-latest"
            }))
            .unwrap(),
            serde_json::from_value::<types::GenerateContentResponse>(json!({
                "candidates": [{
                    "index": 0,
                    "content": {"parts": [{"text": "lo"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1
                },
                "responseId": "resp_1"
            }))
            .unwrap(),
        ];

        let response = response_from_stream_chunks(chunks).unwrap();
        assert!(!response.maybe_failed());
        assert_eq!(response.response_id.as_deref(), Some("resp_1"));

        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Hello");
        assert_eq!(output.usage.input_tokens, 2);
        assert_eq!(output.usage.output_tokens, 1);
        assert_eq!(output.chat_history.len(), 1);
    }
}
