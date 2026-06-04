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

use super::{CompletionFeaturesDyn, ModelEffort, read_sse_json_events, request_client_builder};
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
                return Err(format!(
                    "Completion stream failed, type: {}, message: {}",
                    error.r#type, error.message
                )
                .into());
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

            let mut request = client.post("/messages").json(&r);
            if r.stream == Some(true) {
                request = request.header(ACCEPT, "text/event-stream");
            }
            let response = request.send().await.map_err(|err| {
                format!(
                    "Failed to send completion request, model: {}, error: {}",
                    model, err
                )
            })?;

            r.system = None; // avoid logging tedious instructions
            if response.status().is_success() {
                let mut assistant_raw_message = None;
                let res = if r.stream == Some(true) {
                    let events = read_sse_json_events(response, &model).await?;
                    response_from_stream_events(events)?
                } else {
                    let data = response.bytes().await.map_err(|err| {
                        format!(
                            "Failed to read completion response, model: {}, error: {}",
                            model, err
                        )
                    })?;

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
                    r.messages.drain(0..skip_raw);
                }

                res.try_into_with_raw(
                    r.messages.into_iter().map(|v| json!(v)).collect(),
                    chat_history,
                    assistant_raw_message,
                )
            } else {
                let status = response.status();
                let msg = response.text().await.map_err(|err| {
                    format!(
                        "Failed to read no-success response, model: {}, error: {}",
                        model, err
                    )
                })?;
                log::error!(
                    model = model,
                    request:serde = r;
                    "Completion request failed: {status}, body: {msg}",
                );
                Err(format!("Completion failed, model: {}, error: {}", model, msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::ContentPart;

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
