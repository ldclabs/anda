//! Anthropic Claude API client implementation for Anda Engine
//!
//! This module provides integration with Anthropic's Claude API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, Message, Resource,
};
use log::{Level::Debug, log_enabled};
use reqwest::header::ACCEPT;
use serde_json::{Value, json};
use std::collections::BTreeMap;

use super::{
    CompletionFeaturesDyn, ModelEffort, ModelError, execute_completion_request_with_retry,
    read_completion_response_bytes, read_sse_json_events, request_client_builder,
    streaming_completion_request,
};
use crate::{rfc3339_datetime, unix_ms};

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
        let endpoint = endpoint.unwrap_or_else(|| API_BASE_URL.to_string());
        let endpoint = if endpoint.is_empty() {
            API_BASE_URL.to_string()
        } else {
            endpoint
        };
        Self {
            endpoint,
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

    pub fn with_api_version(mut self, api_version: String) -> Self {
        self.api_version = api_version;
        self
    }

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
            let output_config =
                self.default_request
                    .output_config
                    .get_or_insert(types::OutputConfig {
                        effort: None,
                        format: None,
                    });
            output_config.effort = Some(effort.into());
        }
        self
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, req: types::CreateMessageParams) -> Self {
        self.default_request = req;
        self
    }
}

fn empty_usage() -> types::Usage {
    types::Usage {
        input_tokens: 0,
        cache_creation: None,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        inference_geo: None,
        output_tokens: 0,
        server_tool_use: None,
        service_tier: None,
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

fn ensure_content_block(
    blocks: &mut Vec<Option<types::ContentBlock>>,
    index: usize,
) -> &mut Option<types::ContentBlock> {
    if blocks.len() <= index {
        blocks.resize_with(index + 1, || None);
    }
    &mut blocks[index]
}

fn apply_content_delta(
    blocks: &mut Vec<Option<types::ContentBlock>>,
    json_buffers: &mut BTreeMap<usize, String>,
    index: usize,
    delta: types::ContentBlockDelta,
) {
    match delta {
        types::ContentBlockDelta::TextDelta { text: delta_text } => {
            match ensure_content_block(blocks, index) {
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
            match ensure_content_block(blocks, index) {
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
            match ensure_content_block(blocks, index) {
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
            if let Some(types::ContentBlock::Text { citations, .. }) =
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
    let mut usage = empty_usage();
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
                *ensure_content_block(&mut content, index) = Some(content_block);
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
        r.model = model.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                r.system = Some(req.instructions.into());
            }

            r.messages.append(&mut req.raw_history);
            let skip_raw = r.messages.len();
            for msg in req.chat_history {
                let val = types::Message::from(msg);
                let val = serde_json::to_value(val)?;
                r.messages.push(val);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let val = types::Message::from(msg);
                let val = serde_json::to_value(val)?;
                r.messages.push(val);
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
                let val = types::Message::from(msg);
                let val = serde_json::to_value(val)?;
                r.messages.push(val);
            }

            if let Some(temperature) = req.temperature {
                r.temperature = Some(temperature as f32);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                r.max_tokens = max_tokens as u32;
            }

            if let Some(effort) = req.effort {
                let output_config = r.output_config.get_or_insert(types::OutputConfig {
                    effort: None,
                    format: None,
                });
                output_config.effort = Some(effort.into());
            }

            if let Some(stop) = req.stop {
                r.stop_sequences = Some(stop);
            }

            if !req.tools.is_empty() {
                r.tools = Some(req.tools.into_iter().map(|v| v.into()).collect());
                if req.tool_choice_required {
                    r.tool_choice = Some(types::ToolChoice::any());
                } else {
                    r.tool_choice = Some(types::ToolChoice::auto());
                }
            }

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&r)
            {
                log::debug!(request = val; "Completion request");
            }

            let (res, assistant_raw_message) = execute_completion_request_with_retry(
                &model,
                || {
                    let mut request = client.post("/messages").json(&r);
                    if r.stream == Some(true) {
                        request = streaming_completion_request(request);
                    }
                    request
                },
                |response| async {
                    let mut assistant_raw_message = None;
                    let res = if r.stream == Some(true) {
                        let events = read_sse_json_events(response, &model).await?;
                        response_from_stream_events(events)?
                    } else {
                        let data = read_completion_response_bytes(response, &model).await?;

                        let raw_response = match serde_json::from_slice::<Value>(&data) {
                            Ok(value) => value,
                            Err(err) => {
                                return Err(format!(
                                    "Completion error, model: {}, error: {}, body: {}",
                                    model,
                                    err,
                                    String::from_utf8_lossy(&data)
                                )
                                .into());
                            }
                        };
                        assistant_raw_message = types::assistant_raw_history_message(&raw_response);

                        match serde_json::from_value::<types::CreateMessageResponse>(
                            raw_response.clone(),
                        ) {
                            Ok(res) => res,
                            Err(err) => {
                                return Err(format!(
                                    "Completion error, model: {}, error: {}, body: {}",
                                    model,
                                    err,
                                    String::from_utf8_lossy(&data)
                                )
                                .into());
                            }
                        }
                    };
                    Ok((res, assistant_raw_message))
                },
            )
            .await?;

            let mut logged_request = r.clone();
            logged_request.system = None;
            if log_enabled!(Debug) {
                log::debug!(
                    model = model,
                    request:serde = logged_request,
                    response:serde = res;
                    "Completion response");
            } else if res.maybe_failed() {
                log::warn!(
                    model = model,
                    request:serde = logged_request,
                    response:serde = res;
                    "Completion maybe failed");
            }
            if skip_raw > 0 {
                r.messages.drain(0..skip_raw);
            }

            res.try_into_with_raw(
                r.messages.into_iter().map(|v| json!(v)).collect(),
                chat_history,
                assistant_raw_message,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::{ContentPart, FunctionDefinition};
    use axum::{Router, body::Bytes, extract::State, response::IntoResponse, routing::any};
    use http::{HeaderMap, HeaderValue, Method, StatusCode, Uri};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct MockResponse {
        status: StatusCode,
        headers: HeaderMap,
        body: Vec<u8>,
    }

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        body: Vec<u8>,
    }

    type MockState = Arc<Mutex<(MockResponse, Option<RecordedRequest>)>>;

    async fn mock_handler(
        State(state): State<MockState>,
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
    ) -> impl IntoResponse {
        let mut state = state.lock().unwrap();
        state.1 = Some(RecordedRequest {
            method,
            uri,
            headers,
            body: body.to_vec(),
        });
        let mut response = (state.0.status, state.0.body.clone()).into_response();
        for (name, value) in state.0.headers.iter() {
            response.headers_mut().insert(name, value.clone());
        }
        response
    }

    async fn spawn_mock_server(
        status: StatusCode,
        headers: HeaderMap,
        body: impl Into<Vec<u8>>,
    ) -> (String, MockState) {
        let state = Arc::new(Mutex::new((
            MockResponse {
                status,
                headers,
                body: body.into(),
            },
            None,
        )));
        let app = Router::new()
            .fallback(any(mock_handler))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), state)
    }

    fn no_proxy_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    fn recorded(state: &MockState) -> RecordedRequest {
        state.lock().unwrap().1.clone().unwrap()
    }

    fn sse_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        headers
    }

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
            CompletionFeatures::model_name(&default_model),
            DEFAULT_COMPLETION_MODEL
        );
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
}
