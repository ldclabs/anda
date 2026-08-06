//! Anthropic Claude API client implementation for Anda Engine
//!
//! This module provides integration with Anthropic's Claude API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, FunctionDefinition, Json, Message,
};
use serde_json::Value;
use std::collections::BTreeMap;

use super::driver::{SamplingOptions, WireFormat, drive_completion};
use super::{CompletionFeaturesDyn, ModelEffort, ModelError, request_client_builder};

pub mod types;

impl From<ModelEffort> for types::OutputEffort {
    fn from(value: ModelEffort) -> Self {
        match value {
            ModelEffort::Minimal => Self::Low,
            ModelEffort::Low => Self::Medium,
            ModelEffort::Medium => Self::High,
            ModelEffort::High => Self::XHigh,
            ModelEffort::Max => Self::Max,
        }
    }
}

// ================================================================
// Main Anthropic Client
// ================================================================
const API_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";

/// Default Anthropic completion model used when no model is configured.
pub static DEFAULT_COMPLETION_MODEL: &str = "claude-sonnet-4-6";

/// Anthropic Claude API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    api_version: String,
    bearer_auth: bool,
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
        Self {
            endpoint: super::resolve_endpoint(endpoint, API_BASE_URL),
            bearer_auth: false,
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
            bearer_auth: self.bearer_auth,
            api_key: self.api_key,
            api_version: self.api_version,
            http,
        }
    }

    /// Overrides the Anthropic API version header value.
    pub fn with_api_version(mut self, api_version: String) -> Self {
        self.api_version = api_version;
        self
    }

    /// Selects bearer authentication instead of the `x-api-key` header.
    pub fn with_bearer_auth(mut self, bearer_auth: bool) -> Self {
        self.bearer_auth = bearer_auth;
        self
    }

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        if self.bearer_auth {
            self.http.post(url).bearer_auth(&self.api_key)
        } else {
            self.http
                .post(url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", &self.api_version)
        }
    }

    /// Creates a new completion model instance
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
            max_tokens: 64000,
            ..Default::default()
        };
        Self {
            client,
            default_request,
            model: model.to_string(),
        }
    }

    /// Sets whether the completion request should run in streaming mode
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.default_request.stream = Some(stream);
        self
    }

    /// Sets the default reasoning effort for compatible models
    pub fn with_effort(mut self, effort: Option<ModelEffort>) -> Self {
        if let Some(effort) = effort {
            self.default_request
                .output_config
                .get_or_insert_default()
                .effort = Some(effort.into());
        }
        self
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, req: types::CreateMessageParams) -> Self {
        self.default_request = req;
        self
    }
}

fn merge_usage(usage: &mut types::Usage, delta: types::Usage) {
    let types::Usage {
        input_tokens,
        cache_creation,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        inference_geo,
        output_tokens,
        server_tool_use,
        service_tier,
    } = delta;

    if input_tokens != 0 {
        usage.input_tokens = input_tokens;
    }
    if cache_creation.is_some() {
        usage.cache_creation = cache_creation;
    }
    if cache_creation_input_tokens != 0 {
        usage.cache_creation_input_tokens = cache_creation_input_tokens;
    }
    if cache_read_input_tokens != 0 {
        usage.cache_read_input_tokens = cache_read_input_tokens;
    }
    if inference_geo.is_some() {
        usage.inference_geo = inference_geo;
    }
    if output_tokens != 0 {
        usage.output_tokens = output_tokens;
    }
    if server_tool_use.is_some() {
        usage.server_tool_use = server_tool_use;
    }
    if service_tier.is_some() {
        usage.service_tier = service_tier;
    }
}

/// Upper bound on the content-block index accepted from a stream.
///
/// `index` is provider-supplied and drives a `Vec` resize, so without a bound a single small
/// SSE event could request an arbitrarily large allocation (aborting the process, since an
/// allocation failure is not a catchable per-task error), and `usize::MAX` would overflow
/// `index + 1`. Real responses carry a handful of blocks; this is far above any legitimate
/// count while keeping the worst-case allocation trivial.
const MAX_STREAM_CONTENT_BLOCKS: usize = 4096;

/// Returns the slot for `index`, growing `blocks` as needed.
///
/// Returns `None` when the index exceeds [`MAX_STREAM_CONTENT_BLOCKS`], in which case the
/// event is dropped rather than honored.
fn ensure_content_block(
    blocks: &mut Vec<Option<types::ContentBlock>>,
    index: usize,
) -> Option<&mut Option<types::ContentBlock>> {
    if index >= MAX_STREAM_CONTENT_BLOCKS {
        log::warn!("ignoring content block index {index} beyond the supported range");
        return None;
    }

    if blocks.len() <= index {
        blocks.resize_with(index + 1, || None);
    }
    Some(&mut blocks[index])
}

fn apply_content_delta(
    blocks: &mut Vec<Option<types::ContentBlock>>,
    json_buffers: &mut BTreeMap<usize, String>,
    index: usize,
    delta: types::ContentBlockDelta,
) {
    match delta {
        types::ContentBlockDelta::TextDelta { text: delta_text } => {
            let Some(slot) = ensure_content_block(blocks, index) else {
                return;
            };
            match slot {
                Some(types::ContentBlock::Text { text, .. }) => text.push_str(&delta_text),
                block @ None => {
                    *block = Some(types::ContentBlock::Text {
                        text: delta_text,
                        cache_control: None,
                        citations: None,
                    });
                }
                _ => {}
            }
        }
        types::ContentBlockDelta::InputJsonDelta { partial_json } => {
            json_buffers
                .entry(index)
                .or_default()
                .push_str(&partial_json);
        }
        types::ContentBlockDelta::ThinkingDelta { thinking } => {
            let Some(slot) = ensure_content_block(blocks, index) else {
                return;
            };
            match slot {
                Some(types::ContentBlock::Thinking { thinking: text, .. }) => {
                    text.push_str(&thinking)
                }
                block @ None => {
                    *block = Some(types::ContentBlock::Thinking {
                        thinking,
                        signature: String::new(),
                    });
                }
                _ => {}
            }
        }
        types::ContentBlockDelta::SignatureDelta { signature } => {
            let Some(slot) = ensure_content_block(blocks, index) else {
                return;
            };
            match slot {
                Some(types::ContentBlock::Thinking {
                    signature: text, ..
                }) => text.push_str(&signature),
                block @ None => {
                    *block = Some(types::ContentBlock::Thinking {
                        thinking: String::new(),
                        signature,
                    });
                }
                _ => {}
            }
        }
        types::ContentBlockDelta::CitationsDelta { citation } => {
            if let Some(Some(types::ContentBlock::Text { citations, .. })) =
                ensure_content_block(blocks, index)
            {
                citations.get_or_insert_with(Vec::new).push(citation);
            }
        }
        types::ContentBlockDelta::Any(_) => {}
    }
}

fn finalize_content_block(
    blocks: &mut [Option<types::ContentBlock>],
    json_buffers: &mut BTreeMap<usize, String>,
    index: usize,
) {
    let Some(partial_json) = json_buffers.remove(&index) else {
        return;
    };
    // An empty (or whitespace-only) buffer means a no-argument tool call. Keep
    // the block's existing `input` (initialized to `{}` by `content_block_start`)
    // instead of overwriting it with an empty string, mirroring the official
    // SDK's `JSON.parse(buf || "{}")` guard.
    if partial_json.trim().is_empty() {
        return;
    }
    let input = serde_json::from_str::<Value>(&partial_json).unwrap_or(Value::String(partial_json));
    if let Some(Some(
        types::ContentBlock::ToolUse { input: target, .. }
        | types::ContentBlock::ServerToolUse { input: target, .. },
    )) = blocks.get_mut(index)
    {
        *target = input;
    }
}

fn response_from_stream_events(
    events: Vec<types::StreamEvent>,
) -> Result<types::CreateMessageResponse, BoxError> {
    let mut id = String::new();
    let mut r#type = "message".to_string();
    let mut role = types::Role::Assistant;
    let mut model = String::new();
    let mut stop_reason = None;
    let mut stop_sequence = None;
    let mut usage = types::Usage::default();
    let mut content = Vec::<Option<types::ContentBlock>>::new();
    let mut json_buffers = BTreeMap::<usize, String>::new();
    let mut saw_message = false;

    for event in events {
        match event {
            types::StreamEvent::MessageStart { message } => {
                saw_message = true;
                id = message.id;
                r#type = message.r#type;
                role = message.role;
                model = message.model;
                stop_reason = message.stop_reason;
                stop_sequence = message.stop_sequence;
                usage = message.usage;
                content = message.content.into_iter().map(Some).collect();
            }
            types::StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if let Some(slot) = ensure_content_block(&mut content, index) {
                    *slot = Some(content_block);
                }
            }
            types::StreamEvent::ContentBlockDelta { index, delta } => {
                apply_content_delta(&mut content, &mut json_buffers, index, delta);
            }
            types::StreamEvent::ContentBlockStop { index } => {
                finalize_content_block(&mut content, &mut json_buffers, index);
            }
            types::StreamEvent::MessageDelta {
                delta,
                usage: delta_usage,
            } => {
                if delta.stop_reason.is_some() {
                    stop_reason = delta.stop_reason;
                }
                if delta.stop_sequence.is_some() {
                    stop_sequence = delta.stop_sequence;
                }
                if let Some(delta_usage) = delta_usage {
                    merge_usage(&mut usage, delta_usage);
                }
            }
            types::StreamEvent::Error { error } => {
                let retryable = matches!(
                    error.r#type.as_str(),
                    "overloaded_error" | "rate_limit_error"
                );
                return Err(Box::new(
                    ModelError::new(format!(
                        "Completion stream failed, type: {}, message: {}",
                        error.r#type, error.message
                    ))
                    .with_retryable(retryable),
                ));
            }
            types::StreamEvent::MessageStop
            | types::StreamEvent::Ping
            | types::StreamEvent::Any(_) => {}
        }
    }

    if !saw_message {
        return Err("No streamed Anthropic message".into());
    }

    Ok(types::CreateMessageResponse {
        content: content.into_iter().flatten().collect(),
        id,
        container: None,
        model,
        role,
        stop_reason,
        stop_sequence,
        stop_details: None,
        r#type,
        usage,
    })
}

impl CompletionFeaturesDyn for CompletionModel {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let mut r = self.default_request.clone();
        r.model = model.clone();

        Box::pin(async move {
            drive_completion::<CompletionModel>(model, move |path| client.post(path), r, req).await
        })
    }
}

impl WireFormat for CompletionModel {
    type Request = types::CreateMessageParams;
    type Response = types::CreateMessageResponse;
    type StreamItem = types::StreamEvent;

    fn set_instructions(r: &mut Self::Request, instructions: String) {
        r.system = Some(instructions.into());
    }

    fn append_raw_history(r: &mut Self::Request, mut raw_history: Vec<Json>) -> usize {
        r.messages.append(&mut raw_history);
        r.messages.len()
    }

    fn push_message(r: &mut Self::Request, msg: Message) -> Result<(), BoxError> {
        let val = types::Message::from(msg);
        r.messages.push(serde_json::to_value(val)?);
        Ok(())
    }

    fn apply_sampling(r: &mut Self::Request, options: SamplingOptions) -> Result<(), BoxError> {
        if let Some(temperature) = options.temperature {
            r.temperature = Some(temperature as f32);
        }
        if let Some(max_tokens) = options.max_output_tokens {
            r.max_tokens = max_tokens as u32;
        }
        if let Some(effort) = options.effort {
            r.output_config.get_or_insert_default().effort = Some(effort.into());
        }
        if let Some(output_schema) = options.output_schema {
            r.output_config.get_or_insert_default().format = Some(types::JsonOutputFormat {
                schema: output_schema,
                r#type: types::JsonOutputFormatType::JsonSchema,
            });
        }
        if let Some(stop) = options.stop {
            r.stop_sequences = Some(stop);
        }
        Ok(())
    }

    fn apply_tools(r: &mut Self::Request, tools: Vec<FunctionDefinition>, required: bool) {
        r.tools = Some(tools.into_iter().map(|v| v.into()).collect());
        r.tool_choice = Some(if required {
            types::ToolChoice::any()
        } else {
            types::ToolChoice::auto()
        });
    }

    fn is_stream(r: &Self::Request) -> bool {
        r.stream == Some(true)
    }

    fn endpoint(_r: &Self::Request, _model: &str) -> String {
        "/messages".to_string()
    }

    fn aggregate_stream(items: Vec<Self::StreamItem>) -> Result<Self::Response, BoxError> {
        response_from_stream_events(items)
    }

    fn parse_response(
        model: &str,
        data: &[u8],
    ) -> Result<(Self::Response, Option<Json>), BoxError> {
        let raw_response = match serde_json::from_slice::<Value>(data) {
            Ok(value) => value,
            Err(err) => {
                return Err(format!(
                    "Completion error, model: {}, error: {}, body: {}",
                    model,
                    err,
                    String::from_utf8_lossy(data)
                )
                .into());
            }
        };
        let assistant_raw_message = types::assistant_raw_history_message(&raw_response);

        match serde_json::from_value::<types::CreateMessageResponse>(raw_response) {
            Ok(res) => Ok((res, assistant_raw_message)),
            Err(err) => Err(format!(
                "Completion error, model: {}, error: {}, body: {}",
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
        logged.system = None;
        logged
    }

    fn sent_messages(mut r: Self::Request, skip_raw: usize) -> Vec<Json> {
        if skip_raw > 0 {
            r.messages.drain(0..skip_raw);
        }
        r.messages
    }

    fn into_output(
        res: Self::Response,
        sent_messages: Vec<Json>,
        chat_history: Vec<Message>,
        assistant_raw_message: Option<Json>,
    ) -> Result<AgentOutput, BoxError> {
        res.try_into_with_raw(sent_messages, chat_history, assistant_raw_message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_support::{no_proxy_client, recorded, spawn_mock_server, sse_headers};
    use anda_core::{ContentPart, FunctionDefinition};
    use http::{HeaderMap, Method, StatusCode};
    use reqwest::header::ACCEPT;
    use serde_json::json;

    async fn complete(
        model: &CompletionModel,
        req: CompletionRequest,
    ) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(model, req).await
    }

    #[test]
    fn completion_model_applies_default_effort() {
        let model = Client::new("test-key", Some("http://localhost".into()))
            .completion_model("claude-sonnet-4-6")
            .with_effort(Some(ModelEffort::Max));

        assert_eq!(
            model.default_request.output_config.unwrap().effort,
            Some(types::OutputEffort::Max)
        );
    }

    #[test]
    fn client_defaults_effort_mapping_and_default_request_are_covered() {
        assert_eq!(
            types::OutputEffort::from(ModelEffort::Minimal),
            types::OutputEffort::Low
        );
        assert_eq!(
            types::OutputEffort::from(ModelEffort::Low),
            types::OutputEffort::Medium
        );
        assert_eq!(
            types::OutputEffort::from(ModelEffort::Medium),
            types::OutputEffort::High
        );
        assert_eq!(
            types::OutputEffort::from(ModelEffort::High),
            types::OutputEffort::XHigh
        );
        assert_eq!(
            types::OutputEffort::from(ModelEffort::Max),
            types::OutputEffort::Max
        );

        let default_client = Client::new("test-key", None);
        assert_eq!(default_client.endpoint, API_BASE_URL);
        assert_eq!(default_client.api_version, API_VERSION);
        assert!(!default_client.bearer_auth);

        let empty_endpoint_client = Client::new("test-key", Some(String::new()));
        assert_eq!(empty_endpoint_client.endpoint, API_BASE_URL);

        let default_model = empty_endpoint_client.completion_model("");
        assert_eq!(default_model.model, DEFAULT_COMPLETION_MODEL);
        assert_eq!(
            CompletionFeaturesDyn::model_name(&default_model),
            DEFAULT_COMPLETION_MODEL
        );

        let no_effort = default_model.clone().with_effort(None);
        assert!(no_effort.default_request.output_config.is_none());

        let custom_request = types::CreateMessageParams {
            max_tokens: 17,
            stream: Some(true),
            temperature: Some(0.2),
            ..Default::default()
        };
        let custom_model = default_model
            .with_stream(false)
            .with_default_request(custom_request);
        assert_eq!(custom_model.default_request.max_tokens, 17);
        assert_eq!(custom_model.default_request.stream, Some(true));
        assert_eq!(custom_model.default_request.temperature, Some(0.2));
    }

    #[test]
    fn stream_event_aggregation_covers_usage_merges_and_errors() {
        assert_eq!(
            response_from_stream_events(Vec::new())
                .unwrap_err()
                .to_string(),
            "No streamed Anthropic message"
        );

        let stream_error = response_from_stream_events(vec![
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "try later"}
            }))
            .unwrap(),
        ])
        .unwrap_err();
        assert!(stream_error.to_string().contains("overloaded_error"));
        assert!(stream_error.to_string().contains("try later"));

        let events = vec![
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "message_start",
                "message": {
                    "id": "msg_usage",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-usage",
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 1, "output_tokens": 0}
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan "}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": "answer"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "citations_delta",
                    "citation": {
                        "type": "char_location",
                        "cited_text": "answer",
                        "document_index": 0,
                        "document_title": "Doc",
                        "start_char_index": 0,
                        "end_char_index": 6
                    }
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srv_1",
                    "name": "web_search",
                    "input": {}
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": "{\"query\":\"anda\"}"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_stop",
                "index": 2
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "message_delta",
                "delta": {"stop_reason": "stop_sequence", "stop_sequence": "END"},
                "usage": {
                    "input_tokens": 11,
                    "cache_creation": {
                        "ephemeral_1h_input_tokens": 3,
                        "ephemeral_5m_input_tokens": 4
                    },
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 6,
                    "inference_geo": "us",
                    "output_tokens": 7,
                    "server_tool_use": {
                        "web_fetch_requests": 1,
                        "web_search_requests": 2
                    },
                    "service_tier": "priority"
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({"type": "ping"})).unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({"type": "message_stop"})).unwrap(),
        ];

        let response = response_from_stream_events(events).unwrap();
        assert_eq!(response.stop_reason, Some(types::StopReason::StopSequence));
        assert_eq!(response.stop_sequence.as_deref(), Some("END"));
        assert_eq!(response.usage.input_tokens, 11);
        assert_eq!(
            response
                .usage
                .cache_creation
                .as_ref()
                .unwrap()
                .ephemeral_1h_input_tokens,
            3
        );
        assert_eq!(response.usage.cache_creation_input_tokens, 5);
        assert_eq!(response.usage.cache_read_input_tokens, 6);
        assert_eq!(response.usage.inference_geo.as_deref(), Some("us"));
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(
            response
                .usage
                .server_tool_use
                .as_ref()
                .unwrap()
                .web_search_requests,
            2
        );
        assert_eq!(
            response.usage.service_tier,
            Some(types::UsageServiceTier::Priority)
        );
        assert!(matches!(
            &response.content[0],
            types::ContentBlock::Thinking { thinking, signature }
                if thinking == "plan " && signature == "sig"
        ));
        assert!(matches!(
            &response.content[1],
            types::ContentBlock::Text { citations: Some(citations), .. }
                if citations.len() == 1
        ));
        assert!(matches!(
            &response.content[2],
            types::ContentBlock::ServerToolUse { input, .. }
                if input == &json!({"query": "anda"})
        ));
    }

    #[tokio::test]
    async fn completion_model_posts_request_and_parses_non_stream_response() {
        let body = serde_json::to_vec(&json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "hello from claude"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 6,
                "cache_read_input_tokens": 2,
                "output_tokens": 3
            }
        }))
        .unwrap();
        let (endpoint, state) = spawn_mock_server(StatusCode::OK, HeaderMap::new(), body).await;
        let model = Client::new("test-key", Some(endpoint))
            .with_api_version("2024-01-01".to_string())
            .with_client(no_proxy_client())
            .completion_model("claude-test")
            .with_stream(false);

        let output = complete(
            &model,
            CompletionRequest {
                instructions: "system rules".into(),
                prompt: "say hello".into(),
                temperature: Some(0.3),
                max_output_tokens: Some(256),
                stop: Some(vec!["END".into()]),
                effort: Some(ModelEffort::High),
                output_schema: Some(json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                })),
                tool_choice_required: true,
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

        assert_eq!(output.content, "hello from claude");
        assert_eq!(output.model.as_deref(), Some("claude-test"));
        assert_eq!(output.usage.input_tokens, 8);
        assert_eq!(output.usage.cached_tokens, 2);
        assert_eq!(output.usage.output_tokens, 3);

        let req = recorded(&state);
        assert_eq!(req.method, Method::POST);
        assert_eq!(req.uri.path(), "/messages");
        assert_eq!(req.headers.get("x-api-key").unwrap(), "test-key");
        assert_eq!(req.headers.get("anthropic-version").unwrap(), "2024-01-01");
        assert_ne!(
            req.headers.get(ACCEPT).and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let sent: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(sent["model"], "claude-test");
        assert_eq!(sent["system"], "system rules");
        assert_eq!(sent["messages"][0]["role"], "user");
        assert_eq!(sent["max_tokens"], 256);
        assert_eq!(sent["temperature"], 0.3);
        assert_eq!(sent["stop_sequences"], json!(["END"]));
        assert_eq!(sent["tools"][0]["name"], "lookup");
        assert_eq!(sent["tool_choice"]["type"], "any");
        assert_eq!(sent["output_config"]["effort"], "xhigh");
        assert_eq!(sent["output_config"]["format"]["type"], "json_schema");
        assert_eq!(
            sent["output_config"]["format"]["schema"],
            json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            })
        );
    }

    #[tokio::test]
    async fn completion_model_reports_http_and_invalid_json_errors_with_bearer_auth() {
        let (endpoint, state) =
            spawn_mock_server(StatusCode::BAD_REQUEST, HeaderMap::new(), "bad request").await;
        let model = Client::new("test-key", Some(endpoint))
            .with_bearer_auth(true)
            .with_client(no_proxy_client())
            .completion_model("claude-test")
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
        assert!(err.to_string().contains("bad request"));
        let req = recorded(&state);
        assert_eq!(
            req.headers.get(http::header::AUTHORIZATION).unwrap(),
            "Bearer test-key"
        );
        assert!(req.headers.get("x-api-key").is_none());

        let (endpoint, _) = spawn_mock_server(StatusCode::OK, HeaderMap::new(), "not json").await;
        let model = Client::new("test-key", Some(endpoint))
            .with_client(no_proxy_client())
            .completion_model("claude-test")
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
        assert!(err.to_string().contains("Completion error"));
        assert!(err.to_string().contains("not json"));
    }

    #[tokio::test]
    async fn completion_model_streams_sse_events() {
        let events = [
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_stream_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-stream",
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 3, "output_tokens": 0}
                }
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"}
            }),
            json!({"type": "content_block_stop", "index": 0}),
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"output_tokens": 2}
            }),
            json!({"type": "message_stop"}),
        ]
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();

        let (endpoint, state) =
            spawn_mock_server(StatusCode::OK, sse_headers(), events.into_bytes()).await;
        let model = Client::new("test-key", Some(endpoint))
            .with_client(no_proxy_client())
            .completion_model("claude-stream")
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

        assert_eq!(output.content, "Hi");
        assert_eq!(output.usage.input_tokens, 3);
        assert_eq!(output.usage.output_tokens, 2);
        let req = recorded(&state);
        assert_eq!(req.uri.path(), "/messages");
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
        let sent: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(sent["stream"], true);
    }

    #[test]
    fn aggregates_anthropic_stream_events() {
        let events = vec![
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "message_start",
                "message": {
                    "id": "msg_stream_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 3, "output_tokens": 0}
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi "}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "server_tool_delta", "foo": "bar"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_pause",
                "index": 0,
                "metadata": {"vendor": "compat"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "there"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_stop",
                "index": 0
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {}
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"q\""}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": ":\"anda\"}"}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "content_block_stop",
                "index": 1
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                "usage": {"output_tokens": 5}
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({"type": "message_stop"})).unwrap(),
        ];

        let response = response_from_stream_events(events).unwrap();
        assert!(!response.maybe_failed());

        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Hi there");
        assert_eq!(output.usage.input_tokens, 3);
        assert_eq!(output.usage.output_tokens, 5);
        assert!(matches!(
            &output.chat_history[0].content[1],
            ContentPart::ToolCall { name, args, call_id: Some(call_id) }
                if name == "lookup" && args == &json!({"q": "anda"}) && call_id == "toolu_1"
        ));
    }

    #[test]
    fn stream_content_block_index_is_bounded() {
        // `index` is provider-supplied and drives a `Vec` resize. Without a bound, a huge
        // value requests an allocation that aborts the process, and `usize::MAX` overflows
        // `index + 1` (release builds have no overflow checks), truncating the vec and then
        // panicking on the index. Oversized indices must be dropped instead.
        for oversized in [MAX_STREAM_CONTENT_BLOCKS, 1_000_000_000, usize::MAX] {
            let events = vec![
                serde_json::from_value::<types::StreamEvent>(json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_bounds",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-4-6",
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": 1, "output_tokens": 0}
                    }
                }))
                .unwrap(),
                serde_json::from_value::<types::StreamEvent>(json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }))
                .unwrap(),
                serde_json::from_value::<types::StreamEvent>(json!({
                    "type": "content_block_start",
                    "index": oversized,
                    "content_block": {"type": "text", "text": "ignored"}
                }))
                .unwrap(),
                serde_json::from_value::<types::StreamEvent>(json!({
                    "type": "content_block_delta",
                    "index": oversized,
                    "delta": {"type": "text_delta", "text": "ignored"}
                }))
                .unwrap(),
                serde_json::from_value::<types::StreamEvent>(json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "kept"}
                }))
                .unwrap(),
                serde_json::from_value::<types::StreamEvent>(json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": 1}
                }))
                .unwrap(),
                serde_json::from_value::<types::StreamEvent>(json!({"type": "message_stop"}))
                    .unwrap(),
            ];

            let response = response_from_stream_events(events).unwrap();
            let output = response.try_into(vec![], vec![]).unwrap();
            assert_eq!(
                output.content, "kept",
                "index {oversized} must be dropped without disturbing valid blocks"
            );
        }
    }
}
