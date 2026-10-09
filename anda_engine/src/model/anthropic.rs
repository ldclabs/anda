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

use super::driver::{SamplingOptions, WireFormat, assign_tool_call_ids, drive_completion};
use super::{CompletionFeaturesDyn, ModelEffort, ModelError, request_client_builder};

pub mod types;

impl From<ModelEffort> for types::OutputEffort {
    fn from(value: ModelEffort) -> Self {
        match value {
            ModelEffort::Minimal => Self::Low,
            ModelEffort::Low => Self::Low,
            ModelEffort::Medium => Self::Medium,
            ModelEffort::High => Self::High,
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
        Self::new_with_client(
            api_key,
            endpoint,
            request_client_builder()
                .build()
                .expect("Anthropic reqwest client should build"),
        )
    }

    /// Creates a client that uses the given HTTP client, without building a default one.
    pub fn new_with_client(api_key: &str, endpoint: Option<String>, http: reqwest::Client) -> Self {
        Self {
            endpoint: super::resolve_endpoint(endpoint, API_BASE_URL),
            bearer_auth: false,
            api_key: api_key.to_string(),
            api_version: API_VERSION.to_string(),
            http,
        }
    }

    /// Sets a custom HTTP client for the client
    pub fn with_client(self, http: reqwest::Client) -> Self {
        Self { http, ..self }
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
        let request = self
            .http
            .post(url)
            .header("anthropic-version", &self.api_version);
        if self.bearer_auth {
            request.bearer_auth(&self.api_key)
        } else {
            request.header("x-api-key", &self.api_key)
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

    /// Uses the model's known output limit as the default `max_tokens`.
    ///
    /// Anthropic requires `max_tokens` on every request and rejects a value above the model's
    /// limit, so a model with a lower limit than the built-in default must say so here. `0`
    /// (unknown) keeps the default.
    pub fn with_max_output(mut self, max_output: usize) -> Self {
        if max_output > 0 {
            self.default_request.max_tokens = u32::try_from(max_output).unwrap_or(u32::MAX);
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
    let mut saw_stop = false;

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
            types::StreamEvent::MessageStop => saw_stop = true,
            types::StreamEvent::Ping | types::StreamEvent::Any(_) => {}
        }
    }

    if !saw_message {
        return Err("No streamed Anthropic message".into());
    }
    if !saw_stop || stop_reason.is_none() {
        return Err(Box::new(
            ModelError::new("Anthropic stream ended before message_stop and stop_reason")
                .with_retryable(true),
        ));
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

    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let mut r = self.default_request.clone();
        r.model = model.clone();

        Box::pin(async move {
            assign_tool_call_ids(&mut req);
            drive_completion::<CompletionModel>(model, move |path| client.post(path), r, req).await
        })
    }
}

/// Anthropic caps the strict schemas of one request in total, and a request
/// over any cap fails with a 400.
/// <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
const MAX_STRICT_TOOLS: usize = 20;
const MAX_OPTIONAL_PARAMETERS: usize = 24;
const MAX_UNION_PARAMETERS: usize = 16;

/// What a strict schema spends from the request caps: properties left out of
/// `required`, and subschemas with `anyOf` or a type array. Every nested schema
/// is counted, which can only overestimate the API's count.
#[derive(Clone, Copy, Default)]
struct StrictCost {
    optional: usize,
    unions: usize,
}

impl StrictCost {
    fn plus(self, other: Self) -> Self {
        Self {
            optional: self.optional + other.optional,
            unions: self.unions + other.unions,
        }
    }

    fn within_caps(self) -> bool {
        self.optional <= MAX_OPTIONAL_PARAMETERS && self.unions <= MAX_UNION_PARAMETERS
    }
}

/// Close object schemas without making optional properties required. Reject
/// common unsupported constraints rather than silently weakening the schema.
fn normalize_output_schema(schema: &mut Value) -> Result<StrictCost, BoxError> {
    let mut cost = StrictCost::default();
    normalize_schema_node(schema, &mut cost)?;
    Ok(cost)
}

fn normalize_schema_node(schema: &mut Value, cost: &mut StrictCost) -> Result<(), BoxError> {
    let Some(map) = schema.as_object_mut() else {
        return Ok(());
    };
    for key in [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
    ] {
        if map.contains_key(key) {
            return Err(format!(
                "Anthropic structured outputs do not support schema constraint {key}"
            )
            .into());
        }
    }
    if map
        .get("minItems")
        .is_some_and(|value| !matches!(value.as_u64(), Some(0 | 1)))
    {
        return Err("Anthropic structured outputs only support minItems of 0 or 1".into());
    }
    if map
        .get("$ref")
        .and_then(Value::as_str)
        .is_some_and(|reference| !reference.starts_with('#'))
    {
        return Err(
            "Anthropic structured outputs do not support external schema references".into(),
        );
    }
    // Strict mode checks every enum value against a single declared type, so
    // it rejects the nullable enum that OpenAI strict mode documents
    // (`"type": ["string", "null"], "enum": ["a", null]`). The enum already
    // lists every allowed value, and a bare enum is not a union parameter.
    if map.contains_key("enum") && map.get("type").is_some_and(Value::is_array) {
        map.remove("type");
    }
    if map.contains_key("anyOf") || map.get("type").is_some_and(Value::is_array) {
        cost.unions += 1;
    }
    if let Some(Value::Object(properties)) = map.get("properties") {
        let required = map.get("required").and_then(Value::as_array);
        cost.optional += properties
            .keys()
            .filter(|name| {
                !required
                    .is_some_and(|required| required.iter().any(|value| value == name.as_str()))
            })
            .count();
    }
    let object = map.contains_key("properties")
        || match map.get("type") {
            Some(Value::String(kind)) => kind == "object",
            Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind == "object"),
            _ => false,
        };
    if object {
        let additional = map
            .entry("additionalProperties")
            .or_insert(Value::Bool(false));
        if *additional != Value::Bool(false) {
            return Err("Anthropic structured outputs require additionalProperties: false".into());
        }
    }
    for key in [
        "properties",
        "$defs",
        "$def",
        "definitions",
        "patternProperties",
    ] {
        if let Some(Value::Object(children)) = map.get_mut(key) {
            for child in children.values_mut() {
                normalize_schema_node(child, cost)?;
            }
        }
    }
    for key in ["items", "not", "if", "then", "else"] {
        if let Some(child) = map.get_mut(key) {
            normalize_schema_node(child, cost)?;
        }
    }
    for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
        if let Some(Value::Array(children)) = map.get_mut(key) {
            for child in children {
                normalize_schema_node(child, cost)?;
            }
        }
    }
    Ok(())
}

/// Anthropic rejects `oneOf`, `allOf` and `anyOf` at the top level of every
/// tool's input schema, strict or not, and fails the whole request with a 400.
/// Tools state cross-field rules there, such as KIP's `command` xor
/// `operations`, which their argument parsing enforces again.
fn drop_top_level_combinators(schema: &mut Value) {
    if let Some(map) = schema.as_object_mut() {
        for key in ["oneOf", "allOf", "anyOf"] {
            map.remove(key);
        }
    }
}

/// Anthropic caches a prompt only up to a `cache_control` breakpoint, so a
/// request without one bills its whole prefix again on every round. The
/// top-level marker turns on automatic caching: Anthropic puts the breakpoint on
/// the last cacheable block and moves it forward as the conversation grows, so
/// the messages, which the returned raw history replays, are sent as given. A
/// marker on the system prompt also caches the tools and system on their own,
/// for conversations that share them and for Anthropic-compatible endpoints that
/// honor only block markers.
///
/// A top-level marker set in the default request is kept, and the system marker
/// copies its TTL, since a longer-lived entry may not follow a shorter one. A
/// system prompt already given as blocks keeps the markers its author chose.
fn mark_cache_breakpoints(r: &mut types::CreateMessageParams) {
    let cache_control = r
        .cache_control
        .get_or_insert(types::CacheControlEphemeral {
            r#type: types::CacheControlType::Ephemeral,
            ttl: None,
        })
        .clone();
    if let Some(types::SystemPrompt::Text(text)) = &mut r.system
        && !text.is_empty()
    {
        r.system = Some(types::SystemPrompt::Blocks(vec![
            types::ContentBlock::Text {
                text: std::mem::take(text),
                cache_control: Some(cache_control),
                citations: None,
            },
        ]));
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

    fn push_message(r: &mut Self::Request, msg: Message, _model: &str) -> Result<(), BoxError> {
        let val = types::Message::from(msg);
        r.messages.push(serde_json::to_value(val)?);
        Ok(())
    }

    fn apply_sampling(
        r: &mut Self::Request,
        options: SamplingOptions,
        _model: &str,
    ) -> Result<(), BoxError> {
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

    fn finalize_request(r: &mut Self::Request) -> Result<(), BoxError> {
        let mut spent = StrictCost::default();
        if let Some(format) = r
            .output_config
            .as_mut()
            .and_then(|config| config.format.as_mut())
        {
            spent = normalize_output_schema(&mut format.schema)?;
        }
        for schema in r
            .tools
            .iter_mut()
            .flatten()
            .filter_map(|tool| tool.input_schema.as_mut())
        {
            drop_top_level_combinators(schema);
        }
        let mut strict_tools = 0;
        for tool in r
            .tools
            .iter_mut()
            .flatten()
            .filter(|tool| tool.strict == Some(true))
        {
            let Some(schema) = &mut tool.input_schema else {
                continue;
            };
            // Generated tool schemas routinely carry bounds outside the strict
            // subset (`minimum: 0` on unsigned integers, `minLength`), and a
            // tool set can outgrow the request caps. A tool that does not fit
            // is sent as a regular tool with its schema unchanged.
            let mut strict_schema = schema.clone();
            match normalize_output_schema(&mut strict_schema) {
                Ok(cost) if strict_tools < MAX_STRICT_TOOLS && spent.plus(cost).within_caps() => {
                    *schema = strict_schema;
                    spent = spent.plus(cost);
                    strict_tools += 1;
                }
                _ => tool.strict = None,
            }
        }
        mark_cache_breakpoints(r);
        Ok(())
    }

    fn is_stream(r: &Self::Request) -> bool {
        r.stream == Some(true)
    }

    fn endpoint(_r: &Self::Request, _model: &str) -> String {
        "/messages".to_string()
    }

    fn aggregate_stream(
        items: Vec<Self::StreamItem>,
        _done: bool,
    ) -> Result<Self::Response, BoxError> {
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
                    super::error_body_excerpt(data)
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
                super::error_body_excerpt(data)
            )
            .into()),
        }
    }

    fn maybe_failed(res: &Self::Response) -> bool {
        res.maybe_failed()
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
    use serde_json::{Map, json};

    #[test]
    fn cache_breakpoints_close_the_system_prompt_and_leave_messages_as_given() {
        let marker = json!({"type": "ephemeral"});
        let tool_round = vec![
            json!({"role": "user", "content": "find it"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}
            ]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "found"}
            ]}),
        ];
        let mut r = types::CreateMessageParams {
            system: Some("rules".into()),
            messages: tool_round.clone(),
            ..Default::default()
        };
        mark_cache_breakpoints(&mut r);
        let sent = serde_json::to_value(&r).unwrap();
        assert_eq!(
            sent["system"],
            json!([{"type": "text", "text": "rules", "cache_control": marker}])
        );
        // Automatic caching places the conversation's breakpoint, so the
        // messages that the raw history replays carry none.
        assert_eq!(sent["cache_control"], marker);
        assert_eq!(r.messages, tool_round);

        // A TTL chosen in the default request applies to the system marker
        // too; an author's system blocks are left as given.
        let hour = json!({"type": "ephemeral", "ttl": "1h"});
        let mut r = types::CreateMessageParams {
            system: Some("rules".into()),
            cache_control: Some(serde_json::from_value(hour.clone()).unwrap()),
            ..Default::default()
        };
        mark_cache_breakpoints(&mut r);
        let sent = serde_json::to_value(&r).unwrap();
        assert_eq!(sent["cache_control"], hour);
        assert_eq!(sent["system"][0]["cache_control"], hour);
        let mut r = types::CreateMessageParams {
            system: Some(vec![types::ContentBlock::text("rules")].into()),
            ..Default::default()
        };
        mark_cache_breakpoints(&mut r);
        let sent = serde_json::to_value(&r).unwrap();
        assert_eq!(sent["system"], json!([{"type": "text", "text": "rules"}]));
        assert_eq!(sent["cache_control"], marker);

        // No system prompt: only the conversation is cached.
        let mut r = types::CreateMessageParams::default();
        mark_cache_breakpoints(&mut r);
        assert!(r.system.is_none());
        assert_eq!(serde_json::to_value(&r).unwrap()["cache_control"], marker);
    }

    #[test]
    fn strict_schemas_drop_type_arrays_beside_enums_and_count_their_cost() {
        let mut schema = json!({"type":"object", "required":["behavior","nullable","single","items"],
            "properties":{
                "behavior":{"type":["string","null"], "enum":["auto","smooth",null],
                    "description":"Scroll behavior.", "default":null},
                "nullable":{"type":["string","null"]},
                "single":{"type":"string", "enum":["a"]},
                "items":{"type":"array", "items":{"type":"object", "properties":{
                    "name":{"type":["string","null"]}, "note":{"type":"string"}
                }}},
                "choice":{"anyOf":[{"type":"string"},{"type":"integer"}]}
        }});
        let cost = normalize_output_schema(&mut schema).unwrap();
        let properties = &schema["properties"];
        assert_eq!(
            properties["behavior"],
            json!({"enum":["auto","smooth",null], "description":"Scroll behavior.", "default":null})
        );
        assert_eq!(properties["nullable"], json!({"type":["string","null"]}));
        assert_eq!(properties["single"], json!({"type":"string","enum":["a"]}));
        // `nullable`, the nested `name`, and `choice` are unions; `choice` and
        // the nested `name` and `note` are optional.
        assert_eq!((cost.unions, cost.optional), (3, 3));
    }

    #[test]
    fn strict_tools_beyond_the_request_caps_are_sent_as_regular_tools() {
        let nullable = |count: usize, optional: bool| {
            let names: Vec<String> = (0..count).map(|i| format!("p{i}")).collect();
            let kind = if optional {
                json!("string")
            } else {
                json!(["string", "null"])
            };
            let properties: Map<String, Value> = names
                .iter()
                .map(|name| (name.clone(), json!({"type": kind})))
                .collect();
            let required = if optional { vec![] } else { names };
            json!({"type":"object", "properties":properties, "required":required,
                "additionalProperties":false})
        };
        let mut schemas = vec![
            nullable(10, false),
            nullable(10, false),
            nullable(5, false),
            nullable(25, true),
        ];
        schemas.extend((0..19).map(|_| nullable(0, false)));
        let mut request = types::CreateMessageParams {
            output_config: Some(types::OutputConfig {
                effort: None,
                format: Some(types::JsonOutputFormat {
                    schema: nullable(1, false),
                    r#type: types::JsonOutputFormatType::JsonSchema,
                }),
            }),
            ..Default::default()
        };
        <CompletionModel as WireFormat>::apply_tools(
            &mut request,
            schemas
                .iter()
                .enumerate()
                .map(|(i, schema)| FunctionDefinition {
                    name: format!("tool{i}"),
                    parameters: schema.clone(),
                    strict: Some(true),
                    ..Default::default()
                })
                .collect(),
            false,
        );
        <CompletionModel as WireFormat>::finalize_request(&mut request).unwrap();
        let tools = request.tools.unwrap();
        let strict: Vec<bool> = tools.iter().map(|tool| tool.strict == Some(true)).collect();
        // The output schema spends 1 union, so the second 10-union tool and the
        // 25-optional tool do not fit; the empty tools fill the 20 strict slots.
        let mut expected = vec![true, false, true, false];
        expected.extend((0..19).map(|i| i < 18));
        assert_eq!(strict, expected);
        assert_eq!(tools[1].input_schema.as_ref(), Some(&schemas[1]));
    }

    #[test]
    fn tool_schemas_drop_top_level_combinators() {
        // KIP 0.14.1's `execute_kip` required exactly one of `command` and
        // `operations` through a top-level `oneOf`, which Anthropic rejects.
        let properties = json!({
            "command":{"type":"string"},
            "operations":{"type":"array", "items":{"oneOf":[{"type":"string"},
                {"type":"object", "properties":{"command":{"type":"string"}}}]}}
        });
        let kip = json!({"type":"object", "properties":properties,
            "oneOf":[{"required":["command"]},{"required":["operations"]}],
            "additionalProperties":false});
        let union = json!({"type":"object", "properties":{"id":{"type":"string"}},
            "anyOf":[{"required":["id"]}], "allOf":[{"required":["id"]}]});
        let mut request = types::CreateMessageParams::default();
        <CompletionModel as WireFormat>::apply_tools(
            &mut request,
            vec![
                FunctionDefinition {
                    name: "execute_kip".into(),
                    parameters: kip,
                    ..Default::default()
                },
                FunctionDefinition {
                    name: "union".into(),
                    parameters: union,
                    strict: Some(true),
                    ..Default::default()
                },
            ],
            false,
        );
        <CompletionModel as WireFormat>::finalize_request(&mut request).unwrap();
        let tools = request.tools.unwrap();
        let schema = tools[0].input_schema.as_ref().unwrap();
        for key in ["oneOf", "allOf", "anyOf"] {
            assert!(schema.get(key).is_none(), "{key}");
            assert!(tools[1].input_schema.as_ref().unwrap().get(key).is_none());
        }
        assert_eq!(schema["properties"], properties);
        // Nested combinators are supported and stay.
        assert!(schema["properties"]["operations"]["items"]["oneOf"].is_array());
        assert_eq!(tools[1].strict, Some(true));
    }

    #[test]
    fn strict_schemas_close_objects_and_preserve_optional_fields() {
        let schema = json!({"type":"object", "required":["id"], "properties":{
            "id":{"type":"string"}, "optional":{"type":["object","null"], "properties":{"name":{"type":"string"}}},
            "literal":{"type":"object", "properties":{}, "default":{"minimum":3}}
        }});
        let bounded = json!({"type":"object", "properties":{
            "prompt":{"type":"string", "minLength":1}, "limit":{"type":"integer", "minimum":0}
        }});
        let mut request = types::CreateMessageParams {
            output_config: Some(types::OutputConfig {
                effort: None,
                format: Some(types::JsonOutputFormat {
                    schema: schema.clone(),
                    r#type: types::JsonOutputFormatType::JsonSchema,
                }),
            }),
            ..Default::default()
        };
        <CompletionModel as WireFormat>::apply_tools(
            &mut request,
            vec![
                FunctionDefinition {
                    name: "lookup".into(),
                    parameters: schema.clone(),
                    strict: Some(true),
                    ..Default::default()
                },
                FunctionDefinition {
                    name: "ordinary".into(),
                    parameters: json!({"type":"number","minimum":0}),
                    strict: Some(false),
                    ..Default::default()
                },
                FunctionDefinition {
                    name: "bounded".into(),
                    parameters: bounded.clone(),
                    strict: Some(true),
                    ..Default::default()
                },
            ],
            false,
        );
        <CompletionModel as WireFormat>::finalize_request(&mut request).unwrap();
        let mut unsupported_output = request.clone();
        unsupported_output.output_config.as_mut().unwrap().format = Some(types::JsonOutputFormat {
            schema: bounded.clone(),
            r#type: types::JsonOutputFormatType::JsonSchema,
        });
        assert!(
            <CompletionModel as WireFormat>::finalize_request(&mut unsupported_output).is_err()
        );
        let value = serde_json::to_value(request).unwrap();
        let prepared = &value["output_config"]["format"]["schema"];
        assert_eq!(prepared["required"], json!(["id"]));
        assert_eq!(prepared["additionalProperties"], false);
        assert_eq!(
            prepared["properties"]["optional"]["additionalProperties"],
            false
        );
        assert!(prepared["properties"]["optional"].get("required").is_none());
        assert_eq!(
            prepared["properties"]["literal"]["default"],
            json!({"minimum":3})
        );
        assert_eq!(value["tools"][0]["input_schema"], *prepared);
        assert_eq!(value["tools"][0]["strict"], true);
        assert_eq!(value["tools"][1]["input_schema"]["minimum"], 0);
        // Strict tools outside the supported subset degrade to regular tools.
        assert_eq!(value["tools"][2]["input_schema"], bounded);
        assert!(value["tools"][2].get("strict").is_none());
        for mut invalid in [
            json!({"type":"object","additionalProperties":true}),
            json!({"type":"object","properties":{"n":{"type":"number","minimum":0}}}),
            json!({"type":"array","minItems":2}),
        ] {
            assert!(normalize_output_schema(&mut invalid).is_err());
        }
    }

    #[test]
    fn bearer_auth_keeps_the_configured_api_version() {
        for bearer in [false, true] {
            let request = Client::new_with_client("fake-key", None, no_proxy_client())
                .with_bearer_auth(bearer)
                .with_api_version("2023-06-01".into())
                .post("/messages")
                .build()
                .unwrap();
            assert_eq!(request.headers()["anthropic-version"], "2023-06-01");
            assert_eq!(request.headers().contains_key("authorization"), bearer);
            assert_eq!(request.headers().contains_key("x-api-key"), !bearer);
        }
    }

    #[test]
    fn pdf_encoding_does_not_depend_on_utf8_validity() {
        for bytes in [b"%PDF-1.4\nASCII PDF".to_vec(), vec![0xff, 0xfe]] {
            let data = anda_core::ByteBufB64(bytes);
            let expected = data.to_base64();
            let block: types::ContentBlock = ContentPart::InlineData {
                mime_type: "application/pdf".into(),
                data,
            }
            .into();
            let value = serde_json::to_value(block).unwrap();
            assert_eq!(
                value["source"],
                json!({"type":"base64","media_type":"application/pdf","data":expected})
            );
        }
    }

    #[test]
    fn stream_needs_message_stop_before_exposing_tool_calls() {
        let start = json!({"type":"message_start","message":{
            "id":"msg", "type":"message", "role":"assistant", "model":"claude", "usage":{}, "content":[]
        }});
        let tool = json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}});
        let events = [
            start,
            tool,
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}),
        ];
        let parse = || {
            events
                .iter()
                .map(|event| serde_json::from_value(event.clone()).unwrap())
                .collect::<Vec<_>>()
        };
        let err = response_from_stream_events(parse()).unwrap_err();
        assert!(crate::model::is_retryable_box_error(&err));
        let mut complete = parse();
        complete.push(types::StreamEvent::MessageStop);
        let output = response_from_stream_events(complete)
            .unwrap()
            .try_into(vec![], vec![])
            .unwrap();
        assert_eq!(output.tool_calls.len(), 1);
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
    fn configured_max_output_replaces_the_default_max_tokens() {
        let model = Client::new("test-key", None).completion_model("claude-opus-4-1");
        assert_eq!(model.default_request.max_tokens, 64000);
        assert_eq!(
            model.clone().with_max_output(0).default_request.max_tokens,
            64000
        );
        assert_eq!(
            model.with_max_output(32_000).default_request.max_tokens,
            32_000
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
            types::OutputEffort::Low
        );
        assert_eq!(
            types::OutputEffort::from(ModelEffort::Medium),
            types::OutputEffort::Medium
        );
        assert_eq!(
            types::OutputEffort::from(ModelEffort::High),
            types::OutputEffort::High
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
        assert_eq!(
            sent["system"],
            json!([{
                "type": "text",
                "text": "system rules",
                "cache_control": {"type": "ephemeral"}
            }])
        );
        assert_eq!(sent["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(sent["messages"][0]["role"], "user");
        // The next round replays the exact message this round sent.
        assert_eq!(output.raw_history[0], sent["messages"][0]);
        assert_eq!(sent["max_tokens"], 256);
        assert_eq!(sent["temperature"], 0.3);
        assert_eq!(sent["stop_sequences"], json!(["END"]));
        assert_eq!(sent["tools"][0]["name"], "lookup");
        assert_eq!(sent["tool_choice"]["type"], "any");
        assert_eq!(sent["output_config"]["effort"], "high");
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

    #[test]
    fn prune_inline_media_covers_anthropic_media_blocks() {
        let bytes = b"inline attachment bytes".to_vec();
        let encoded = anda_core::ByteBufB64(bytes.clone()).to_base64();
        let file_encoded = anda_core::ByteBufB64(b"data uri attachment".to_vec()).to_base64();
        let msg = anda_core::Message {
            role: "user".into(),
            content: vec![
                "look".to_string().into(),
                ContentPart::InlineData {
                    mime_type: "image/png".into(),
                    data: anda_core::ByteBufB64(bytes.clone()),
                },
                ContentPart::InlineData {
                    mime_type: "application/pdf".into(),
                    data: anda_core::ByteBufB64(bytes),
                },
                ContentPart::FileData {
                    file_uri: format!("data:application/pdf;base64,{file_encoded}"),
                    mime_type: Some("application/pdf".into()),
                },
            ],
            ..Default::default()
        };
        let mut raw = vec![serde_json::to_value(types::Message::from(msg)).unwrap()];
        crate::model::raw::prune_inline_media(&mut raw);
        let sent = serde_json::to_string(&raw).unwrap();
        assert!(!sent.contains(&encoded), "{sent}");
        assert!(!sent.contains(&file_encoded), "{sent}");
        assert!(sent.contains("[inline image/png data omitted]"), "{sent}");
        assert!(
            sent.contains("[inline application/pdf data omitted]"),
            "{sent}"
        );
        assert!(sent.contains("look"), "{sent}");
    }
}
