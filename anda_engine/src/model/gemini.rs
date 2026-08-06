//! Gemini API client implementation for Anda Engine
//!
//! This module provides integration with Google's Gemini API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, FunctionDefinition, Json, Message,
};
use std::collections::BTreeMap;

use super::driver::{SamplingOptions, WireFormat, drive_completion};
use super::{CompletionFeaturesDyn, ModelEffort, request_client_builder};

pub mod types;

impl From<ModelEffort> for types::ThinkingLevel {
    fn from(value: ModelEffort) -> Self {
        match value {
            ModelEffort::Minimal => Self::Minimal,
            ModelEffort::Low => Self::Low,
            ModelEffort::Medium => Self::Medium,
            ModelEffort::High => Self::High,
            ModelEffort::Max => Self::High,
        }
    }
}

// ================================================================
// Main Gemini Client
// ================================================================
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Default Gemini completion model used when no model is configured.
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
        Self {
            endpoint: super::resolve_endpoint(endpoint, API_BASE_URL),
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

impl CompletionFeaturesDyn for CompletionModel {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let r = self.default_request.clone();

        Box::pin(async move {
            drive_completion::<CompletionModel>(model, move |path| client.post(path), r, req).await
        })
    }
}

impl WireFormat for CompletionModel {
    type Request = types::GenerateContentRequest;
    type Response = types::GenerateContentResponse;
    type StreamItem = types::GenerateContentResponse;

    fn set_instructions(r: &mut Self::Request, instructions: String) {
        r.system_instruction = Some(types::Content {
            role: Some(types::Role::Model),
            parts: vec![types::Part {
                data: types::PartKind::Text(instructions),
                ..Default::default()
            }],
        });
    }

    fn append_raw_history(r: &mut Self::Request, mut raw_history: Vec<Json>) -> usize {
        r.contents.append(&mut raw_history);
        r.contents.len()
    }

    fn push_message(r: &mut Self::Request, msg: Message) -> Result<(), BoxError> {
        let val = types::Content::from(msg);
        r.contents.push(serde_json::to_value(val)?);
        Ok(())
    }

    fn apply_sampling(r: &mut Self::Request, options: SamplingOptions) -> Result<(), BoxError> {
        if let Some(temperature) = options.temperature {
            r.generation_config.temperature = Some(temperature);
        }
        if let Some(max_tokens) = options.max_output_tokens {
            r.generation_config.max_output_tokens = Some(max_tokens as i32);
        }
        if let Some(effort) = options.effort {
            let thinking_config = r
                .generation_config
                .thinking_config
                .get_or_insert_with(types::ThinkingConfig::default);
            thinking_config.thinking_level = Some(effort.into());
        }
        if let Some(output_schema) = options.output_schema {
            r.generation_config.response_mime_type = Some("application/json".to_string());
            r.generation_config.response_schema = Some(output_schema);
        }
        if let Some(stop) = options.stop {
            r.generation_config.stop_sequences = Some(stop);
        }
        Ok(())
    }

    fn apply_tools(r: &mut Self::Request, tools: Vec<FunctionDefinition>, required: bool) {
        r.tools = vec![tools.into()];
        // Honor forced tool choice: `Any` constrains the model to emit a
        // function call, matching the OpenAI/Anthropic adapters.
        let mode = if required {
            types::FunctionCallingMode::Any
        } else {
            types::FunctionCallingMode::Auto
        };
        r.tool_config = Some(types::ToolConfig {
            function_calling_config: types::FunctionCallingConfig {
                mode,
                allowed_function_names: None,
            },
        });
    }

    fn is_stream(r: &Self::Request) -> bool {
        r.stream
    }

    fn endpoint(r: &Self::Request, model: &str) -> String {
        if r.stream {
            format!("/{}:streamGenerateContent?alt=sse", model)
        } else {
            format!("/{}:generateContent", model)
        }
    }

    fn aggregate_stream(items: Vec<Self::StreamItem>) -> Result<Self::Response, BoxError> {
        response_from_stream_chunks(items)
    }

    fn parse_response(
        model: &str,
        data: &[u8],
    ) -> Result<(Self::Response, Option<Json>), BoxError> {
        match serde_json::from_slice::<types::GenerateContentResponse>(data) {
            Ok(res) => Ok((res, None)),
            Err(err) => Err(format!(
                "Invalid completion response, model: {}, error: {}, body: {}",
                model,
                err,
                String::from_utf8_lossy(data)
            )
            .into()),
        }
    }

    fn maybe_failed(res: &Self::Response) -> bool {
        res.maybe_failed()
    }

    fn redacted_for_log(r: &Self::Request) -> Self::Request {
        let mut logged = r.clone();
        logged.system_instruction = None;
        logged
    }

    fn sent_messages(mut r: Self::Request, skip_raw: usize) -> Vec<Json> {
        if skip_raw > 0 {
            r.contents.drain(0..skip_raw);
        }
        r.contents
    }

    fn into_output(
        res: Self::Response,
        sent_messages: Vec<Json>,
        chat_history: Vec<Message>,
        _assistant_raw_message: Option<Json>,
    ) -> Result<AgentOutput, BoxError> {
        res.try_into(sent_messages, chat_history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_support::{no_proxy_client, recorded, spawn_mock_server, sse_headers};
    use anda_core::FunctionDefinition;
    use http::{HeaderMap, Method, StatusCode};
    use reqwest::header::ACCEPT;
    use serde_json::{Value, json};

    async fn complete(
        model: &CompletionModel,
        req: CompletionRequest,
    ) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(model, req).await
    }

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
    fn client_defaults_effort_mapping_and_default_request_are_covered() {
        assert_eq!(
            types::ThinkingLevel::from(ModelEffort::Minimal),
            types::ThinkingLevel::Minimal
        );
        assert_eq!(
            types::ThinkingLevel::from(ModelEffort::Low),
            types::ThinkingLevel::Low
        );
        assert_eq!(
            types::ThinkingLevel::from(ModelEffort::Medium),
            types::ThinkingLevel::Medium
        );
        assert_eq!(
            types::ThinkingLevel::from(ModelEffort::High),
            types::ThinkingLevel::High
        );
        assert_eq!(
            types::ThinkingLevel::from(ModelEffort::Max),
            types::ThinkingLevel::High
        );

        let default_client = Client::new("test-key", None);
        assert_eq!(default_client.endpoint, API_BASE_URL);
        let empty_endpoint_client = Client::new("test-key", Some(String::new()));
        assert_eq!(empty_endpoint_client.endpoint, API_BASE_URL);

        let default_model = empty_endpoint_client.completion_model("");
        assert_eq!(default_model.model, DEFAULT_COMPLETION_MODEL);
        assert_eq!(
            CompletionFeaturesDyn::model_name(&default_model),
            DEFAULT_COMPLETION_MODEL
        );

        let no_effort = default_model.clone().with_effort(None);
        assert!(
            no_effort
                .default_request
                .generation_config
                .thinking_config
                .is_none()
        );

        let custom_request = types::GenerateContentRequest {
            stream: true,
            generation_config: types::GenerationConfig {
                temperature: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        };
        let custom_model = default_model
            .with_stream(false)
            .with_default_request(custom_request);
        assert!(custom_model.default_request.stream);
        assert_eq!(
            custom_model.default_request.generation_config.temperature,
            Some(0.25)
        );
    }

    #[test]
    fn stream_chunk_aggregation_covers_metadata_replacements_and_errors() {
        assert_eq!(
            response_from_stream_chunks(Vec::new())
                .unwrap_err()
                .to_string(),
            "No streamed Gemini response"
        );

        let chunks = vec![
            serde_json::from_value::<types::GenerateContentResponse>(json!({
                "candidates": [{
                    "index": 0,
                    "content": {
                        "parts": [
                            {"text": "plain"},
                            {"thought": true, "text": "think"}
                        ]
                    }
                }]
            }))
            .unwrap(),
            serde_json::from_value::<types::GenerateContentResponse>(json!({
                "promptFeedback": {"blockReason": "OTHER"},
                "modelStatus": {"modelStage": "PREVIEW"},
                "candidates": [{
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "thought": true,
                                "thoughtSignature": "sig-1",
                                "text": " more"
                            },
                            {"functionCall": {"name": "lookup", "args": {"q": "anda"}}}
                        ]
                    },
                    "finishReason": "MAX_TOKENS",
                    "citationMetadata": {
                        "citationSources": [{"uri": "https://example.com"}]
                    },
                    "tokenCount": 9,
                    "groundingAttributions": [{
                        "sourceId": {"groundingPassage": {"passageId": "p1"}}
                    }],
                    "groundingMetadata": {
                        "webSearchQueries": ["anda coverage"]
                    },
                    "avgLogprobs": -0.5,
                    "logprobsResult": {"logProbabilitySum": -1.0},
                    "urlContextMetadata": {
                        "urlMetadata": [{
                            "retrievedUrl": "https://example.com",
                            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                        }]
                    },
                    "finishMessage": "cut short"
                }]
            }))
            .unwrap(),
            serde_json::from_value::<types::GenerateContentResponse>(json!({
                "candidates": [{
                    "index": 1,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "second candidate"}]
                    },
                    "finishReason": "STOP"
                }]
            }))
            .unwrap(),
        ];

        let response = response_from_stream_chunks(chunks).unwrap();
        assert_eq!(
            response.prompt_feedback.unwrap().block_reason,
            Some(types::BlockReason::Other)
        );
        assert_eq!(
            response.model_status.unwrap().model_stage,
            Some(types::ModelStage::Preview)
        );
        assert_eq!(response.candidates.len(), 2);

        let first = &response.candidates[0];
        assert_eq!(first.content.role, Some(types::Role::Model));
        assert!(matches!(
            &first.content.parts[1],
            types::Part {
                thought: Some(true),
                thought_signature: Some(signature),
                data: types::PartKind::Text(text),
            } if signature == "sig-1" && text == "think more"
        ));
        assert!(matches!(
            &first.content.parts[2].data,
            types::PartKind::FunctionCall { name, args, .. }
                if name == "lookup" && args.as_ref() == Some(&json!({"q": "anda"}))
        ));
        assert_eq!(first.finish_reason, Some(types::FinishReason::MaxTokens));
        assert_eq!(first.token_count, Some(9));
        assert_eq!(first.finish_message.as_deref(), Some("cut short"));
        assert_eq!(first.avg_logprobs, Some(-0.5));
        assert!(first.citation_metadata.is_some());
        assert_eq!(first.grounding_attributions.len(), 1);
        assert!(first.grounding_metadata.is_some());
        assert!(first.logprobs_result.is_some());
        assert!(first.url_context_metadata.is_some());
    }

    #[tokio::test]
    async fn completion_model_posts_request_and_parses_non_stream_response() {
        let body = serde_json::to_vec(&json!({
            "candidates": [{
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"text": "hello from gemini"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 7,
                "cachedContentTokenCount": 2,
                "candidatesTokenCount": 3
            },
            "modelVersion": "gemini-test",
            "responseId": "resp_1"
        }))
        .unwrap();
        let (endpoint, state) = spawn_mock_server(StatusCode::OK, HeaderMap::new(), body).await;
        let model = Client::new("test-key", Some(endpoint))
            .with_client(no_proxy_client())
            .completion_model("gemini-test")
            .with_stream(false);

        let output = complete(
            &model,
            CompletionRequest {
                instructions: "system rules".into(),
                prompt: "say hello".into(),
                temperature: Some(0.4),
                max_output_tokens: Some(128),
                output_schema: Some(json!({"type": "object"})),
                stop: Some(vec!["END".into()]),
                effort: Some(ModelEffort::Low),
                tools: vec![FunctionDefinition {
                    name: "lookup".into(),
                    description: "Lookup docs".into(),
                    parameters: json!({"type": "object"}),
                    strict: Some(false),
                }],
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(output.content, "hello from gemini");
        assert_eq!(output.model.as_deref(), Some("gemini-test"));
        assert_eq!(output.usage.input_tokens, 7);
        assert_eq!(output.usage.cached_tokens, 2);
        assert_eq!(output.usage.output_tokens, 3);

        let req = recorded(&state);
        assert_eq!(req.method, Method::POST);
        assert_eq!(req.uri.path(), "/gemini-test:generateContent");
        assert_eq!(req.headers.get("x-goog-api-key").unwrap(), "test-key");
        assert_ne!(
            req.headers.get(ACCEPT).and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let sent: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(sent["systemInstruction"]["role"], "model");
        assert_eq!(sent["contents"][0]["role"], "user");
        assert_eq!(sent["generationConfig"]["temperature"], 0.4);
        assert_eq!(sent["generationConfig"]["maxOutputTokens"], 128);
        assert_eq!(
            sent["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            sent["generationConfig"]["responseSchema"],
            json!({"type": "object"})
        );
        assert_eq!(sent["generationConfig"]["stopSequences"], json!(["END"]));
        assert_eq!(
            sent["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "LOW"
        );
        assert_eq!(
            sent["tools"][0]["functionDeclarations"][0]["name"],
            "lookup"
        );
        assert!(sent.get("toolConfig").is_some());
    }

    #[tokio::test]
    async fn completion_model_reports_http_and_invalid_json_errors() {
        let (endpoint, _) = spawn_mock_server(
            StatusCode::SERVICE_UNAVAILABLE,
            HeaderMap::new(),
            "unavailable",
        )
        .await;
        let model = Client::new("test-key", Some(endpoint))
            .with_client(no_proxy_client())
            .completion_model("gemini-test")
            .with_stream(false);
        let err = complete(
            &model,
            CompletionRequest {
                prompt: "hello".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("Completion failed"));
        assert!(err.to_string().contains("unavailable"));

        let (endpoint, _) = spawn_mock_server(StatusCode::OK, HeaderMap::new(), "not json").await;
        let model = Client::new("test-key", Some(endpoint))
            .with_client(no_proxy_client())
            .completion_model("gemini-test")
            .with_stream(false);
        let err = complete(
            &model,
            CompletionRequest {
                prompt: "hello".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("Invalid completion response"));
        assert!(err.to_string().contains("not json"));
    }

    #[tokio::test]
    async fn completion_model_streams_sse_chunks() {
        let chunks = [
            json!({
                "candidates": [{
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hel"}]
                    }
                }],
                "usageMetadata": {"promptTokenCount": 4},
                "modelVersion": "gemini-stream"
            }),
            json!({
                "candidates": [{
                    "index": 0,
                    "content": {"parts": [{"text": "lo"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 2
                },
                "responseId": "resp_stream"
            }),
        ]
        .into_iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect::<String>();

        let (endpoint, state) =
            spawn_mock_server(StatusCode::OK, sse_headers(), chunks.into_bytes()).await;
        let model = Client::new("test-key", Some(endpoint))
            .with_client(no_proxy_client())
            .completion_model("gemini-stream")
            .with_stream(true);
        let output = complete(
            &model,
            CompletionRequest {
                prompt: "stream".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(output.content, "Hello");
        assert_eq!(output.usage.input_tokens, 4);
        assert_eq!(output.usage.output_tokens, 2);
        let req = recorded(&state);
        assert_eq!(req.uri.path(), "/gemini-stream:streamGenerateContent");
        assert_eq!(req.uri.query(), Some("alt=sse"));
        assert_eq!(
            req.headers.get(ACCEPT).and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(
            req.headers
                .get(http::header::ACCEPT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("identity")
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
                    "safetyRatings": [{
                        "category": "HARM_CATEGORY_VENDOR_SPECIFIC",
                        "probability": "VERY_LOW"
                    }]
                }]
            }))
            .unwrap(),
            serde_json::from_value::<types::GenerateContentResponse>(json!({
                "candidates": [{
                    "index": 0,
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                    "promptTokensDetails": null,
                    "candidatesTokensDetails": [{
                        "modality": "FUTURE_MODALITY",
                        "tokenCount": null
                    }]
                },
                "responseId": "resp_1"
            }))
            .unwrap(),
        ];

        let response = response_from_stream_chunks(chunks).unwrap();
        assert!(!response.maybe_failed());
        assert_eq!(response.response_id.as_deref(), Some("resp_1"));
        assert!(matches!(
            &response.candidates[0]
                .safety_ratings
                .as_ref()
                .unwrap()[0]
                .category,
            types::HarmCategory::Unknown(category)
                if category == "HARM_CATEGORY_VENDOR_SPECIFIC"
        ));
        assert!(matches!(
            &response.usage_metadata.candidates_tokens_details[0].modality,
            Some(types::Modality::Unknown(modality)) if modality == "FUTURE_MODALITY"
        ));

        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Hello");
        assert_eq!(output.usage.input_tokens, 2);
        assert_eq!(output.usage.output_tokens, 1);
        assert_eq!(output.chat_history.len(), 1);
    }
}
