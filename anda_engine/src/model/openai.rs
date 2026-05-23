//! OpenAI API client implementation for Anda Engine
//!
//! This module provides integration with OpenAI's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Embedding model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, ContentPart, FunctionDefinition, Json,
    Message, Usage as ModelUsage, inline_data_from_data_url, part_to_data_url,
};
use log::{Level::Debug, log_enabled};
use reqwest::header::ACCEPT;
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};
use std::collections::{BTreeMap, HashMap};

pub mod types;

use super::{CompletionFeaturesDyn, read_sse_json_events, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

// ================================================================
// Main OpenAI Client
// ================================================================
const API_BASE_URL: &str = "https://api.openai.com/v1";

/// Default completion model to use if not specified
pub const DEFAULT_COMPLETION_MODEL: &str = "gpt-5.4-mini";

fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

fn is_json_null(value: &Json) -> bool {
    value.is_null()
}

/// OpenAI API client for handling completions
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new OpenAI client with the given API key
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key for authentication
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
                .expect("OpenAI reqwest client should build"),
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

    /// Creates a POST request builder for the given API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url).bearer_auth(&self.api_key)
    }

    /// Creates a completion model with the given name
    ///
    /// # Arguments
    /// * `model` - Name of the completion model to use
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

    /// Completion model with Responses API
    pub fn completion_model_v2(&self, model: &str) -> CompletionModelV2 {
        CompletionModelV2::new(
            self.clone(),
            if model.is_empty() {
                DEFAULT_COMPLETION_MODEL
            } else {
                model
            },
        )
    }
}

/// Token usage information from OpenAI API
#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
    #[serde(default)]
    pub total_tokens: usize,
    #[serde(default)]
    pub prompt_tokens_details: PromptTokensDetails,
    #[serde(default)]
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: usize,
    #[serde(default)]
    pub audio_tokens: usize,
}

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub accepted_prediction_tokens: usize,
    #[serde(default)]
    pub audio_tokens: usize,
    #[serde(default)]
    pub reasoning_tokens: usize,
    #[serde(default)]
    pub rejected_prediction_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {}, completion tokens: {}",
            self.prompt_tokens, self.completion_tokens
        )
    }
}

/// Response structure for OpenAI completion API
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    #[serde(default, deserialize_with = "null_default")]
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn parse_output(&mut self) {
        for choice in self.choices.iter_mut() {
            if let Ok(msg) = serde_json::from_value::<MessageOutput>(choice.message.clone()) {
                choice.parsed_message = Some(msg);
            }
        }
    }

    fn try_into(
        mut self,
        raw_history: Vec<Json>,
        chat_history: Vec<Message>,
    ) -> Result<AgentOutput, BoxError> {
        let mut output = AgentOutput {
            raw_history,
            chat_history,
            model: Some(self.model.clone()),
            usage: ModelUsage {
                input_tokens: self.usage.prompt_tokens as u64,
                output_tokens: self.usage.completion_tokens as u64,
                cached_tokens: self.usage.prompt_tokens_details.cached_tokens as u64,
                requests: 1,
            },
            ..Default::default()
        };

        let choice = self.choices.pop().ok_or("No completion choice")?;
        if !matches!(choice.finish_reason.as_str(), "stop" | "tool_calls") {
            output.failed_reason = Some(choice.finish_reason);
        } else {
            output.raw_history.push(choice.message);
            let timestamp = unix_ms();
            let msg = choice
                .parsed_message
                .ok_or("Failed to parse message output")?;

            if let Some(refusal) = msg.refusal() {
                output.failed_reason = Some(refusal.to_string());
            }

            let mut msg: Message = msg.into();
            msg.name = Some(self.model);
            msg.timestamp = Some(timestamp);
            output.content = msg.text().unwrap_or_default();
            output.thoughts = msg.thoughts();
            output.tool_calls = msg.tool_calls();
            output.chat_history.push(msg);
        }

        Ok(output)
    }

    fn maybe_failed(&self) -> bool {
        !self.choices.iter().any(|choice| {
            matches!(choice.finish_reason.as_str(), "stop" | "tool_calls")
                && choice
                    .parsed_message
                    .as_ref()
                    .map(|m| !m.has_refusal() && m.has_output())
                    .unwrap_or(false)
        })
    }
}

/// Request body shape for OpenAI's Chat Completions API.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudioParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<FunctionDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<ChatCompletionModality>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<ChatCompletionPredictionContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionModality {
    Text,
    Audio,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Stop {
    String(String),
    Strings(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionAudioParam {
    pub format: String,
    pub voice: ChatCompletionVoice,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ChatCompletionVoice {
    Name(String),
    Id { id: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ChatCompletionPredictionContent {
    pub content: ChatCompletionMessageContent,
    #[serde(rename = "type")]
    pub r#type: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Json },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JsonSchemaResponseFormat {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub schema: Map<String, Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct UserLocation {
    pub r#type: String,
    pub approximate: ApproximateLocation,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ApproximateLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionToolChoice {
    Mode(ChatCompletionToolChoiceMode),
    Object(ChatCompletionToolChoiceObject),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionToolChoiceObject {
    AllowedTools {
        allowed_tools: ChatCompletionAllowedTools,
    },
    Function {
        function: NamedTool,
    },
    Custom {
        custom: NamedTool,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ChatCompletionAllowedTools {
    pub mode: ChatCompletionAllowedToolsMode,
    pub tools: Vec<ChatCompletionAllowedTool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAllowedToolsMode {
    Auto,
    Required,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionAllowedTool {
    Function { function: NamedTool },
    Custom { custom: NamedTool },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct NamedTool {
    pub name: String,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionMessageContent {
    Text(String),
    Parts(Vec<ChatCompletionContentPart>),
}

impl Default for ChatCompletionMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl ChatCompletionMessageContent {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }

    fn refusal(&self) -> Option<&str> {
        match self {
            Self::Text(_) => None,
            Self::Parts(parts) => parts.iter().find_map(|part| match part {
                ChatCompletionContentPart::Refusal { refusal } => Some(refusal.as_str()),
                _ => None,
            }),
        }
    }
}

impl<'de> Deserialize<'de> for ChatCompletionMessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Text(String),
            Parts(Vec<ChatCompletionContentPart>),
        }

        Ok(match Option::<Helper>::deserialize(deserializer)? {
            Some(Helper::Text(text)) => Self::Text(text),
            Some(Helper::Parts(parts)) => Self::Parts(parts),
            None => Self::default(),
        })
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ChatCompletionContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
    #[serde(rename = "file")]
    File { file: FileContent },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(untagged)]
    Any(Json),
}

impl<'de> Deserialize<'de> for ChatCompletionContentPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Json::deserialize(deserializer)?;
        match &value {
            Json::Object(map)
                if matches!(
                    map.get("type").and_then(|t| t.as_str()),
                    Some("text" | "image_url" | "input_audio" | "file" | "refusal")
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type")]
                enum Helper {
                    #[serde(rename = "text")]
                    Text { text: String },
                    #[serde(rename = "image_url")]
                    ImageUrl { image_url: ImageUrl },
                    #[serde(rename = "input_audio")]
                    InputAudio { input_audio: InputAudio },
                    #[serde(rename = "file")]
                    File { file: FileContent },
                    #[serde(rename = "refusal")]
                    Refusal { refusal: String },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(Helper::Text { text }) => Ok(Self::Text { text }),
                    Ok(Helper::ImageUrl { image_url }) => Ok(Self::ImageUrl { image_url }),
                    Ok(Helper::InputAudio { input_audio }) => Ok(Self::InputAudio { input_audio }),
                    Ok(Helper::File { file }) => Ok(Self::File { file }),
                    Ok(Helper::Refusal { refusal }) => Ok(Self::Refusal { refusal }),
                    Err(_) => Ok(Self::Any(value)),
                }
            }
            _ => Ok(Self::Any(value)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct FileContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct MessageInput {
    pub role: String,

    #[serde(default, skip_serializing_if = "is_json_null")]
    pub content: Json,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOutput>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Function>,
}

fn push_message_input(
    messages: &mut Vec<MessageInput>,
    msg: &Message,
    content: &mut Vec<Json>,
    tool_calls: &mut Vec<ToolCallOutput>,
) {
    if content.is_empty() && tool_calls.is_empty() {
        return;
    }

    let mut input = MessageInput {
        role: msg.role.clone(),
        content: Json::Null,
        name: msg.name.clone(),
        ..Default::default()
    };
    if !content.is_empty() {
        input.content = Json::Array(std::mem::take(content));
    }
    if !tool_calls.is_empty() {
        input.tool_calls = Some(std::mem::take(tool_calls));
    }
    messages.push(input);
}

fn tool_output_to_string(output: &Json) -> String {
    serde_json::to_string(output).unwrap_or_default()
}

fn to_message_inputs(msg: &Message) -> Vec<MessageInput> {
    let mut messages: Vec<MessageInput> = Vec::new();
    let mut content: Vec<Json> = Vec::new();
    let mut tool_calls: Vec<ToolCallOutput> = Vec::new();

    for part in msg.content.iter() {
        match part {
            ContentPart::Text { text } => content.push(json!({
                "type": "text",
                "text": text,
            })),
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => tool_calls.push(ToolCallOutput {
                id: call_id.clone().unwrap_or_default(),
                r#type: "function".into(),
                function: Some(Function {
                    name: name.clone(),
                    arguments: serde_json::to_string(args).unwrap_or_default(),
                }),
                custom: None,
            }),
            ContentPart::ToolOutput {
                output, call_id, ..
            } => {
                push_message_input(&mut messages, msg, &mut content, &mut tool_calls);
                messages.push(MessageInput {
                    role: "tool".into(),
                    content: tool_output_to_string(output).into(),
                    tool_call_id: call_id.clone(),
                    ..Default::default()
                });
            }
            ContentPart::FileData {
                file_uri,
                mime_type,
            } => match mime_type.clone().unwrap_or_default().as_str() {
                mt if mt.starts_with("image") => content.push(json!({
                    "type": "image_url",
                    "image_url":  {
                        "url": file_uri,
                    },
                })),
                mt if mt.starts_with("video") => {
                    content.push(json!({
                        "type": "video_url",
                        "video_url": {
                            "url": file_uri,
                        },
                    }));
                }
                mt if mt.starts_with("audio") => {
                    content.push(json!({
                        "type": "input_audio",
                        "input_audio": {
                            "data": file_uri,
                            "format": if mt.contains("wav") { "wav" } else { "mp3" },
                        },
                    }));
                }
                _ => content.push(json!({
                    "type": "file",
                    "file":  {
                        "file_data": file_uri,
                    },
                })),
            },
            ContentPart::InlineData { data, mime_type } => {
                match mime_type.as_str() {
                    mt if mt.starts_with("image") => {
                        content.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": part_to_data_url(data, Some(mime_type)),
                            },
                        }));
                    }
                    mt if mt.starts_with("video") => {
                        content.push(json!({
                            "type": "video_url",
                            "video_url": {
                                "url": part_to_data_url(data, Some(mime_type)),
                            },
                        }));
                    }
                    mt if mt.starts_with("audio") => {
                        content.push(json!({
                            "type": "input_audio",
                            "input_audio": {
                                "data": part_to_data_url(data, Some(mime_type)),
                                "format": if mt.contains("wav") { "wav" } else { "mp3" },
                            },
                        }));
                    }
                    _ => content.push(json!({
                        "type": "file",
                        "file":  {
                            "file_data": data,
                        },
                    })),
                };
            }
            v => content.push(json!(v)),
        }
    }
    push_message_input(&mut messages, msg, &mut content, &mut tool_calls);
    messages
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Json,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
    #[serde(default, deserialize_with = "null_default")]
    pub finish_reason: String,

    #[serde(skip)]
    pub parsed_message: Option<MessageOutput>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamChunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    created: u64,
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<ChatCompletionStreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    service_tier: Option<ServiceTier>,
    #[serde(default)]
    system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamChoice {
    pub index: usize,
    #[serde(default, deserialize_with = "null_default")]
    pub delta: ChatCompletionStreamDelta,
    #[serde(default)]
    pub logprobs: Option<ChoiceLogprobs>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatCompletionStreamDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<ChatCompletionStreamContent>,
    #[serde(default)]
    function_call: Option<FunctionCallDelta>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatCompletionStreamContent {
    Text(String),
    Parts(Vec<ChatCompletionContentPart>),
}

#[derive(Debug, Default, Deserialize)]
struct FunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct CustomToolCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCallDelta>,
    #[serde(default)]
    pub custom: Option<CustomToolCallDelta>,
}

#[derive(Default)]
struct FunctionCallStreamBuilder {
    name: String,
    arguments: String,
}

#[derive(Default)]
struct CustomToolCallStreamBuilder {
    name: String,
    input: String,
}

#[derive(Default)]
struct ToolCallStreamBuilder {
    id: String,
    r#type: String,
    function: Option<FunctionCallStreamBuilder>,
    custom: Option<CustomToolCallStreamBuilder>,
}

#[derive(Default)]
struct ChatCompletionChoiceStreamBuilder {
    role: Option<String>,
    content_text: String,
    content_parts: Vec<ChatCompletionContentPart>,
    reasoning_content: String,
    refusal: String,
    function_call: Option<FunctionCallStreamBuilder>,
    tool_calls: BTreeMap<usize, ToolCallStreamBuilder>,
    finish_reason: String,
    logprobs: Option<ChoiceLogprobs>,
}

impl ChatCompletionChoiceStreamBuilder {
    fn apply_delta(&mut self, delta: ChatCompletionStreamDelta) {
        if let Some(role) = delta.role {
            self.role = Some(role);
        }
        if let Some(content) = delta.content {
            match content {
                ChatCompletionStreamContent::Text(text) => self.content_text.push_str(&text),
                ChatCompletionStreamContent::Parts(parts) => self.content_parts.extend(parts),
            }
        }
        if let Some(reasoning_content) = delta.reasoning_content {
            self.reasoning_content.push_str(&reasoning_content);
        }
        if let Some(refusal) = delta.refusal {
            self.refusal.push_str(&refusal);
        }
        if let Some(function_call) = delta.function_call {
            let builder = self
                .function_call
                .get_or_insert_with(FunctionCallStreamBuilder::default);
            if let Some(name) = function_call.name {
                builder.name.push_str(&name);
            }
            if let Some(arguments) = function_call.arguments {
                builder.arguments.push_str(&arguments);
            }
        }
        if let Some(tool_calls) = delta.tool_calls {
            for tool_call in tool_calls {
                let builder = self.tool_calls.entry(tool_call.index).or_default();
                if let Some(id) = tool_call.id {
                    builder.id = id;
                }
                if let Some(r#type) = tool_call.r#type {
                    builder.r#type = r#type;
                }
                if let Some(function) = tool_call.function {
                    let function_builder = builder
                        .function
                        .get_or_insert_with(FunctionCallStreamBuilder::default);
                    if let Some(name) = function.name {
                        function_builder.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        function_builder.arguments.push_str(&arguments);
                    }
                }
                if let Some(custom) = tool_call.custom {
                    let custom_builder = builder
                        .custom
                        .get_or_insert_with(CustomToolCallStreamBuilder::default);
                    if let Some(name) = custom.name {
                        custom_builder.name.push_str(&name);
                    }
                    if let Some(input) = custom.input {
                        custom_builder.input.push_str(&input);
                    }
                }
            }
        }
    }

    fn into_choice(self, index: usize) -> Choice {
        let mut message = Map::new();
        message.insert(
            "role".to_string(),
            Json::String(self.role.unwrap_or_else(|| "assistant".to_string())),
        );

        if self.content_parts.is_empty() {
            message.insert("content".to_string(), Json::String(self.content_text));
        } else {
            let mut parts = Vec::new();
            if !self.content_text.is_empty() {
                parts.push(json!({"type": "text", "text": self.content_text}));
            }
            parts.extend(
                self.content_parts
                    .into_iter()
                    .map(|part| serde_json::to_value(part).unwrap_or(Json::Null)),
            );
            message.insert("content".to_string(), Json::Array(parts));
        }

        if !self.reasoning_content.is_empty() {
            message.insert(
                "reasoning_content".to_string(),
                Json::String(self.reasoning_content),
            );
        }
        if !self.refusal.is_empty() {
            message.insert("refusal".to_string(), Json::String(self.refusal));
        }
        if let Some(function_call) = self.function_call {
            message.insert(
                "function_call".to_string(),
                json!(Function {
                    name: function_call.name,
                    arguments: function_call.arguments,
                }),
            );
        }

        let tool_calls = self
            .tool_calls
            .into_values()
            .map(|tool_call| ToolCallOutput {
                id: tool_call.id,
                r#type: if tool_call.r#type.is_empty() {
                    "function".to_string()
                } else {
                    tool_call.r#type
                },
                function: tool_call.function.map(|function| Function {
                    name: function.name,
                    arguments: function.arguments,
                }),
                custom: tool_call.custom.map(|custom| CustomToolCall {
                    name: custom.name,
                    input: custom.input,
                }),
            })
            .collect::<Vec<_>>();
        if !tool_calls.is_empty() {
            message.insert("tool_calls".to_string(), json!(tool_calls));
        }

        Choice {
            index,
            message: Json::Object(message),
            logprobs: self.logprobs,
            finish_reason: self.finish_reason,
            parsed_message: None,
        }
    }
}

fn chat_completion_response_from_stream_chunks(
    chunks: Vec<ChatCompletionStreamChunk>,
) -> Result<CompletionResponse, BoxError> {
    let mut id = String::new();
    let mut object = String::new();
    let mut created = 0;
    let mut model = String::new();
    let mut usage = Usage::default();
    let mut service_tier = None;
    let mut system_fingerprint = None;
    let mut choices = BTreeMap::<usize, ChatCompletionChoiceStreamBuilder>::new();

    for chunk in chunks {
        if id.is_empty() {
            id = chunk.id;
        }
        if object.is_empty() {
            object = chunk.object;
        }
        if created == 0 {
            created = chunk.created;
        }
        if model.is_empty() {
            model = chunk.model;
        }
        if let Some(chunk_usage) = chunk.usage {
            usage = chunk_usage;
        }
        if chunk.service_tier.is_some() {
            service_tier = chunk.service_tier;
        }
        if chunk.system_fingerprint.is_some() {
            system_fingerprint = chunk.system_fingerprint;
        }

        for choice in chunk.choices {
            let builder = choices.entry(choice.index).or_default();
            builder.apply_delta(choice.delta);
            if let Some(logprobs) = choice.logprobs {
                builder.logprobs = Some(logprobs);
            }
            if let Some(finish_reason) = choice.finish_reason {
                builder.finish_reason = finish_reason;
            }
        }
    }

    if choices.is_empty() {
        return Err("No streamed completion choice".into());
    }
    if object == "chat.completion.chunk" || object.is_empty() {
        object = "chat.completion".to_string();
    }

    let mut response = CompletionResponse {
        id,
        object,
        created,
        model,
        choices: choices
            .into_iter()
            .map(|(index, choice)| choice.into_choice(index))
            .collect(),
        service_tier,
        system_fingerprint,
        usage,
    };
    response.parse_output();
    Ok(response)
}

fn responses_response_from_stream_events(
    events: Vec<types::StreamEvent>,
) -> Result<types::CompletionResponse, BoxError> {
    let mut last_response = None;
    let mut output_items = BTreeMap::<usize, types::MessageItem>::new();

    for event in events {
        match event {
            types::StreamEvent::ResponseCreated { response }
            | types::StreamEvent::ResponseInProgress { response }
            | types::StreamEvent::ResponseCompleted { response }
            | types::StreamEvent::ResponseFailed { response }
            | types::StreamEvent::ResponseIncomplete { response } => {
                last_response = Some(response);
            }
            types::StreamEvent::ResponseOutputItemDone { output_index, item } => {
                output_items.insert(output_index, item);
            }
            _ => {}
        }
    }

    let mut response = last_response.ok_or("No streamed completion response")?;
    if response.output.is_empty() && !output_items.is_empty() {
        response.output = output_items
            .into_values()
            .map(|item| serde_json::to_value(item).unwrap_or(Json::Null))
            .collect();
    }
    response.parse_output();
    Ok(response)
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct ChoiceLogprobs {
    #[serde(default)]
    pub content: Vec<ChatCompletionTokenLogprob>,
    #[serde(default)]
    pub refusal: Vec<ChatCompletionTokenLogprob>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionTokenLogprob {
    pub token: String,
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    #[serde(default)]
    pub top_logprobs: Vec<ChatCompletionTopLogprob>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionTopLogprob {
    pub token: String,
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessageOutput {
    pub role: String,

    #[serde(default)]
    pub content: ChatCompletionMessageContent,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<MessageAnnotation>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudio>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Function>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

impl MessageOutput {
    fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref().or_else(|| self.content.refusal())
    }

    fn has_refusal(&self) -> bool {
        self.refusal().is_some()
    }

    fn has_output(&self) -> bool {
        !self.content.is_empty()
            || self.reasoning_content.is_some()
            || self.function_call.is_some()
            || self
                .tool_calls
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
    }
}

fn file_data_content_part(file_uri: String, mime_type: Option<String>) -> ContentPart {
    if let Some((data, mime_type)) = inline_data_from_data_url(&file_uri) {
        ContentPart::InlineData { mime_type, data }
    } else {
        ContentPart::FileData {
            file_uri,
            mime_type,
        }
    }
}

fn chat_completion_content_part_into(part: ChatCompletionContentPart) -> Option<ContentPart> {
    match part {
        ChatCompletionContentPart::Text { text } => Some(ContentPart::Text { text }),
        ChatCompletionContentPart::ImageUrl { image_url } => {
            Some(file_data_content_part(image_url.url, None))
        }
        ChatCompletionContentPart::InputAudio { input_audio } => {
            let mime_type = Some(format!("audio/{}", input_audio.format));
            Some(file_data_content_part(input_audio.data, mime_type))
        }
        ChatCompletionContentPart::File { file } => {
            let FileContent {
                file_data,
                file_id,
                filename,
            } = file;
            if let Some(file_data) = file_data {
                Some(file_data_content_part(file_data, None))
            } else {
                let mut fields = Map::new();
                if let Some(file_id) = file_id {
                    fields.insert("file_id".into(), file_id.into());
                }
                if let Some(filename) = filename {
                    fields.insert("filename".into(), filename.into());
                }
                if fields.is_empty() {
                    None
                } else {
                    Some(ContentPart::Any(json!({
                        "type": "file",
                        "file": fields,
                    })))
                }
            }
        }
        ChatCompletionContentPart::Refusal { refusal } => Some(ContentPart::Any(json!({
            "type": "refusal",
            "refusal": refusal,
        }))),
        ChatCompletionContentPart::Any(value) => Some(ContentPart::Any(value)),
    }
}

fn chat_completion_content_into_parts(content: ChatCompletionMessageContent) -> Vec<ContentPart> {
    match content {
        ChatCompletionMessageContent::Text(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![ContentPart::Text { text }]
            }
        }
        ChatCompletionMessageContent::Parts(parts) => parts
            .into_iter()
            .filter_map(chat_completion_content_part_into)
            .collect(),
    }
}

impl From<MessageOutput> for Message {
    fn from(msg: MessageOutput) -> Self {
        let mut content = Vec::new();
        content.extend(chat_completion_content_into_parts(msg.content));
        if let Some(reasoning_content) = msg.reasoning_content {
            content.push(ContentPart::Reasoning {
                text: reasoning_content,
            });
        }
        if let Some(function_call) = msg.function_call {
            content.push(ContentPart::ToolCall {
                name: function_call.name,
                args: serde_json::from_str(&function_call.arguments).unwrap_or_default(),
                call_id: None,
            });
        }
        if let Some(tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                let ToolCallOutput {
                    id,
                    r#type,
                    function,
                    custom,
                } = tc;
                match (function, custom) {
                    (Some(function), _) => content.push(ContentPart::ToolCall {
                        name: function.name,
                        args: serde_json::from_str(&function.arguments).unwrap_or_default(),
                        call_id: Some(id),
                    }),
                    (None, Some(custom)) => {
                        let CustomToolCall { name, input } = custom;
                        let args = match serde_json::from_str(&input) {
                            Ok(args) => args,
                            Err(_) => Json::String(input),
                        };
                        content.push(ContentPart::ToolCall {
                            name,
                            args,
                            call_id: Some(id),
                        });
                    }
                    (None, None) => content.push(ContentPart::Any(json!({
                        "id": id,
                        "type": r#type,
                    }))),
                }
            }
        }
        Self {
            role: msg.role,
            content,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MessageAnnotation {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    pub fields: Map<String, Json>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionAudio {
    pub id: String,
    #[serde(default)]
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(default)]
    pub transcript: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolCallOutput {
    pub id: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Function>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<CustomToolCall>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function { function: FunctionDefinition },
    Custom { custom: CustomToolDefinition },
}

impl From<FunctionDefinition> for ToolDefinition {
    fn from(f: FunctionDefinition) -> Self {
        Self::Function {
            function: f.normalize_strict_parameters(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CustomToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<CustomToolFormat>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CustomToolFormat {
    Text,
    Grammar { grammar: Json },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CustomToolCall {
    pub name: String,
    pub input: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

/// Completion model implementation for OpenAI API
#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    default_request: ChatCompletionRequest,
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - OpenAI client instance
    /// * `model` - Name of the completion model
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            default_request: ChatCompletionRequest {
                store: Some(false),
                ..Default::default()
            },
        }
    }

    /// Sets whether the completion request should run in streaming mode
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.default_request.stream = Some(stream);
        self
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, req: ChatCompletionRequest) -> Self {
        self.default_request = req;
        self
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let client = self.client.clone();
        let mut r = self.default_request.clone();
        r.model = self.model.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                r.messages.push(json!(MessageInput {
                    role: "system".into(),
                    content: req.instructions.clone().into(),
                    ..Default::default()
                }));
            };

            r.messages.append(&mut req.raw_history);
            let skip_raw = r.messages.len();

            for msg in req.chat_history {
                for input in to_message_inputs(&msg) {
                    r.messages.push(serde_json::to_value(input)?);
                }
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                for input in to_message_inputs(&msg) {
                    r.messages.push(serde_json::to_value(input)?);
                }
                chat_history.push(msg);
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

                for input in to_message_inputs(&msg) {
                    r.messages.push(serde_json::to_value(input)?);
                }
                chat_history.push(msg);
            }

            if let Some(temperature) = req.temperature {
                r.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                r.max_tokens = Some(max_tokens as u64);
            }

            if let Some(json_schema) = req.output_schema {
                r.response_format = Some(ResponseFormat::JsonSchema { json_schema });
            }

            if let Some(stop) = req.stop {
                r.stop = Some(Stop::Strings(stop));
            }

            if !req.tools.is_empty() {
                r.tools = req
                    .tools
                    .into_iter()
                    .map(ToolDefinition::from)
                    .collect::<Vec<_>>();
                if !r.model.starts_with("deepseek") {
                    r.tool_choice = Some(if req.tool_choice_required {
                        ChatCompletionToolChoice::Mode(ChatCompletionToolChoiceMode::Required)
                    } else {
                        ChatCompletionToolChoice::Mode(ChatCompletionToolChoiceMode::Auto)
                    });
                }
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&r)
            {
                log::debug!(request = val; "OpenAI completions request");
            }

            let mut request = client.post("/chat/completions").json(&r);
            if r.stream == Some(true) {
                request = request.header(ACCEPT, "text/event-stream");
            }
            let response = request.send().await.map_err(|err| {
                format!(
                    "Failed to send completion request, model: {}, error: {}",
                    r.model, err
                )
            })?;
            if response.status().is_success() {
                let res = if r.stream == Some(true) {
                    let chunks = read_sse_json_events(response, &r.model).await?;
                    chat_completion_response_from_stream_chunks(chunks)?
                } else {
                    let text = response.text().await.map_err(|err| {
                        format!(
                            "Failed to read completion response, model: {}, error: {}",
                            r.model, err
                        )
                    })?;
                    match serde_json::from_str::<CompletionResponse>(&text) {
                        Ok(mut res) => {
                            res.parse_output();
                            res
                        }
                        Err(err) => {
                            return Err(format!(
                                "Invalid completion response, model: {}, error: {}, body: {}",
                                r.model, err, text
                            )
                            .into());
                        }
                    }
                };

                if log_enabled!(Debug) {
                    log::debug!(
                        model = r.model,
                        request:serde = r,
                        response:serde = res;
                        "Completion response");
                } else if res.maybe_failed() {
                    log::warn!(
                        model = r.model,
                        request:serde = r,
                        response:serde = res;
                        "Completion maybe failed");
                }
                if skip_raw > 0 {
                    r.messages.drain(0..skip_raw);
                }
                res.try_into(r.messages, chat_history)
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = r.model,
                    request:serde = r;
                    "Completion request failed: {status}, body: {msg}",
                );
                Err(format!("Completion failed, model: {}, error: {}", r.model, msg).into())
            }
        })
    }
}

/// Completion model implementation for OpenAI API
#[derive(Clone)]
pub struct CompletionModelV2 {
    client: Client,
    /// Default request template
    default_request: types::CompletionRequest,
    pub model: String,
}

impl CompletionModelV2 {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - OpenAI client instance
    /// * `model` - Name of the completion model
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            default_request: types::CompletionRequest {
                model: model.to_string(),
                stream: Some(true),
                additional_parameters: types::AdditionalParameters {
                    store: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            },
            model: model.to_string(),
        }
    }

    /// Sets whether the completion request should run in streaming mode
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.default_request.stream = Some(stream);
        self
    }

    /// Sets a default request template for the model
    pub fn with_default_request(mut self, req: types::CompletionRequest) -> Self {
        self.default_request = req;
        self
    }
}

impl CompletionFeaturesDyn for CompletionModelV2 {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let client = self.client.clone();
        let mut r = self.default_request.clone();
        r.model = self.model.clone();
        r.stream = Some(true);
        r.additional_parameters.store = Some(false);

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                r.instructions = Some(req.instructions);
            };

            for raw in req.raw_history {
                r.input.extend(types::raw_history_into(raw));
            }
            let skip_raw = r.input.len();

            for msg in req.chat_history {
                let vals = types::message_into(msg);
                r.input.extend(vals);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let vals = types::message_into(msg);
                r.input.extend(vals);
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
                let vals = types::message_into(msg);
                r.input.extend(vals);
            }

            if let Some(temperature) = req.temperature {
                r.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                r.max_output_tokens = Some(max_tokens as u64);
            }

            if let Some(output_schema) = req.output_schema {
                r.additional_parameters.text = Some(types::TextConfig::structured_output(
                    "structured_output".to_string(),
                    output_schema,
                ));
            }

            if !req.tools.is_empty() {
                r.tools = req
                    .tools
                    .into_iter()
                    .map(|v| {
                        let v = v.normalize_strict_parameters();
                        types::ToolDefinition::Function {
                            name: v.name,
                            description: if v.description.is_empty() {
                                None
                            } else {
                                Some(v.description)
                            },
                            parameters: v.parameters,
                            strict: v.strict.unwrap_or_default(),
                            defer_loading: None,
                        }
                    })
                    .collect::<Vec<_>>();
                r.tool_choice = Some(if req.tool_choice_required {
                    types::ToolChoice::required()
                } else {
                    types::ToolChoice::auto()
                });
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&r)
            {
                log::debug!(request = val; "Completion request");
            }

            let mut request = client.post("/responses").json(&r);
            if r.stream == Some(true) {
                request = request.header(ACCEPT, "text/event-stream");
            }
            let response = request.send().await.map_err(|err| {
                format!(
                    "Failed to send completion request, model: {}, error: {}",
                    r.model, err
                )
            })?;
            if response.status().is_success() {
                let res = if r.stream == Some(true) {
                    let events = read_sse_json_events(response, &r.model).await?;
                    responses_response_from_stream_events(events)?
                } else {
                    let text = response.text().await.map_err(|err| {
                        format!(
                            "Failed to read completion response, model: {}, error: {}",
                            r.model, err
                        )
                    })?;
                    match serde_json::from_str::<types::CompletionResponse>(&text) {
                        Ok(mut res) => {
                            res.parse_output();
                            res
                        }
                        Err(err) => {
                            return Err(format!(
                                "Invalid completion response, model: {}, error: {}, body: {}",
                                r.model, err, text
                            )
                            .into());
                        }
                    }
                };

                if log_enabled!(Debug) {
                    log::debug!(
                            model = r.model,
                            request:serde = r,
                            response:serde = res;
                            "Completion response");
                } else if res.maybe_failed() {
                    log::warn!(
                            request:serde = r,
                            response:serde = res;
                            "Completion maybe failed");
                }

                if skip_raw > 0 {
                    r.input.drain(0..skip_raw);
                }
                res.try_into(
                    r.input.into_iter().map(|v| json!(v)).collect(),
                    chat_history,
                )
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = r.model,
                    request:serde = r;
                    "Completion request failed: {status}, body: {msg}",
                );
                Err(format!("Completion failed, model: {}, error: {}", r.model, msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, json};

    #[test]
    fn response_v2_defaults_to_streaming_stateless_requests() {
        let model =
            Client::new("test-key", Some("http://localhost".into())).completion_model_v2("gpt-5.5");

        assert_eq!(model.default_request.stream, Some(true));
        assert_eq!(
            model.default_request.additional_parameters.store,
            Some(false)
        );
    }

    #[test]
    fn serializes_chat_completion_request_core_types() {
        let mut schema = Map::new();
        schema.insert("type".into(), Json::String("object".into()));

        let request = ChatCompletionRequest {
            model: "gpt-5.4".into(),
            messages: vec![json!(MessageInput {
                role: "developer".into(),
                content: "You are helpful.".into(),
                ..Default::default()
            })],
            audio: Some(ChatCompletionAudioParam {
                format: "mp3".into(),
                voice: ChatCompletionVoice::Name("alloy".into()),
            }),
            max_completion_tokens: Some(512),
            modalities: Some(vec![
                ChatCompletionModality::Text,
                ChatCompletionModality::Audio,
            ]),
            reasoning_effort: Some(ReasoningEffort::Low),
            response_format: Some(ResponseFormat::JsonSchema {
                json_schema: json!({
                    "name": "answer",
                    "description": null,
                    "schema": schema,
                    "strict": true,
                }),
            }),
            service_tier: Some(ServiceTier::Priority),
            stop: Some(Stop::Strings(vec!["END".into()])),
            stream_options: Some(ChatCompletionStreamOptions {
                include_obfuscation: Some(false),
                include_usage: Some(true),
            }),
            tool_choice: Some(ChatCompletionToolChoice::Object(
                ChatCompletionToolChoiceObject::AllowedTools {
                    allowed_tools: ChatCompletionAllowedTools {
                        mode: ChatCompletionAllowedToolsMode::Required,
                        tools: vec![ChatCompletionAllowedTool::Function {
                            function: NamedTool {
                                name: "lookup".into(),
                            },
                        }],
                    },
                },
            )),
            tools: vec![
                ToolDefinition::Function {
                    function: FunctionDefinition {
                        name: "lookup".into(),
                        description: "Look up a record".into(),
                        parameters: json!({
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": false
                        }),
                        strict: Some(true),
                    },
                },
                ToolDefinition::Custom {
                    custom: CustomToolDefinition {
                        name: "sql".into(),
                        description: Some("Run a SQL query".into()),
                        format: Some(CustomToolFormat::Text),
                    },
                },
            ],
            verbosity: Some(Verbosity::Low),
            ..Default::default()
        };

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["messages"][0]["role"], "developer");
        assert_eq!(value["modalities"], json!(["text", "audio"]));
        assert_eq!(value["audio"], json!({"format": "mp3", "voice": "alloy"}));
        assert_eq!(value["reasoning_effort"], "low");
        assert_eq!(value["response_format"]["type"], "json_schema");
        assert_eq!(value["response_format"]["json_schema"]["strict"], true);
        assert_eq!(value["service_tier"], "priority");
        assert_eq!(value["stream_options"]["include_usage"], true);
        assert_eq!(value["tool_choice"]["type"], "allowed_tools");
        assert_eq!(value["tool_choice"]["allowed_tools"]["mode"], "required");
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["function"]["strict"], true);
        assert_eq!(value["tools"][1]["type"], "custom");
        assert_eq!(value["tools"][1]["custom"]["format"]["type"], "text");
        assert_eq!(value["verbosity"], "low");
    }

    #[test]
    fn to_message_input_serializes_tool_calls_outside_content() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![
                ContentPart::Text {
                    text: "done".into(),
                },
                ContentPart::ToolCall {
                    name: "lookup".into(),
                    args: json!({"id": 1}),
                    call_id: Some("call_1".into()),
                },
            ],
            ..Default::default()
        };

        let values = to_message_inputs(&msg)
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(values.len(), 1);
        let value = &values[0];
        assert_eq!(value["role"], "assistant");
        assert_eq!(
            value["content"],
            json!([{ "type": "text", "text": "done" }])
        );
        assert_eq!(value["tool_calls"][0]["id"], "call_1");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            r#"{"id":1}"#
        );
    }

    #[test]
    fn to_message_inputs_splits_multiple_tool_outputs() {
        let msg = Message {
            role: "tool".into(),
            content: vec![
                ContentPart::ToolOutput {
                    name: "lookup".into(),
                    output: json!({"temperature": 20}),
                    call_id: Some("call_1".into()),
                    remote_id: None,
                },
                ContentPart::ToolOutput {
                    name: "lookup".into(),
                    output: json!({"temperature": 24}),
                    call_id: Some("call_2".into()),
                    remote_id: None,
                },
            ],
            ..Default::default()
        };

        let values = to_message_inputs(&msg)
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["role"], "tool");
        assert_eq!(values[0]["tool_call_id"], "call_1");
        assert_eq!(values[0]["content"], r#"{"temperature":20}"#);
        assert_eq!(values[1]["role"], "tool");
        assert_eq!(values[1]["tool_call_id"], "call_2");
        assert_eq!(values[1]["content"], r#"{"temperature":24}"#);
        assert!(values[0].get("tool_calls").is_none());
        assert!(values[1].get("tool_calls").is_none());
    }

    #[test]
    fn message_output_preserves_non_text_content_parts() {
        let msg: Message = serde_json::from_value::<MessageOutput>(json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "See attached"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "input_audio", "input_audio": {"data": "https://example.com/audio.mp3", "format": "mp3"}},
                {"type": "file", "file": {"file_id": "file_123", "filename": "notes.txt"}},
                {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
            ]
        }))
        .unwrap()
        .into();

        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.len(), 5);
        assert_eq!(
            msg.content[0],
            ContentPart::Text {
                text: "See attached".into()
            }
        );
        assert_eq!(
            msg.content[1],
            ContentPart::FileData {
                file_uri: "https://example.com/image.png".into(),
                mime_type: None,
            }
        );
        assert_eq!(
            msg.content[2],
            ContentPart::FileData {
                file_uri: "https://example.com/audio.mp3".into(),
                mime_type: Some("audio/mp3".into()),
            }
        );
        assert_eq!(
            msg.content[3],
            ContentPart::Any(json!({
                "type": "file",
                "file": {
                    "file_id": "file_123",
                    "filename": "notes.txt",
                },
            }))
        );
        assert_eq!(
            msg.content[4],
            ContentPart::Any(json!({
                "type": "video_url",
                "video_url": {"url": "https://example.com/video.mp4"},
            }))
        );
    }

    #[test]
    fn deserializes_chat_completion_tool_call_response() {
        let mut response: CompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1741569952,
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
                        }
                    }],
                    "refusal": null
                },
                "logprobs": null,
                "finish_reason": "tool_calls"
            }],
            "service_tier": "default",
            "system_fingerprint": "fp_123",
            "usage": {
                "prompt_tokens": 82,
                "completion_tokens": 17,
                "total_tokens": 99,
                "prompt_tokens_details": {"cached_tokens": 7, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 2,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }
        }))
        .unwrap();

        response.parse_output();
        assert!(!response.maybe_failed());
        assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 7);
        assert_eq!(response.usage.completion_tokens_details.reasoning_tokens, 2);

        let output = response
            .try_into(vec![json!({"role": "user"})], vec![])
            .unwrap();
        assert_eq!(output.usage.input_tokens, 82);
        assert_eq!(output.usage.output_tokens, 17);
        assert_eq!(output.usage.cached_tokens, 7);
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "get_current_weather");
        assert_eq!(
            output.tool_calls[0].args,
            json!({"location": "Boston", "unit": "celsius"})
        );
        assert_eq!(output.tool_calls[0].call_id.as_deref(), Some("call_1"));
        assert_eq!(output.raw_history.len(), 2);
    }

    #[test]
    fn aggregates_chat_completion_stream_chunks() {
        let chunks = vec![
            serde_json::from_value::<ChatCompletionStreamChunk>(json!({
                "id": "chatcmpl_stream_1",
                "object": "chat.completion.chunk",
                "created": 1741569955,
                "model": "gpt-5.4",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": null
                }],
                "usage": null
            }))
            .unwrap(),
            serde_json::from_value::<ChatCompletionStreamChunk>(json!({
                "id": "chatcmpl_stream_1",
                "object": "chat.completion.chunk",
                "created": 1741569955,
                "model": "gpt-5.4",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "lo",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"q\""}
                        }]
                    },
                    "finish_reason": null
                }]
            }))
            .unwrap(),
            serde_json::from_value::<ChatCompletionStreamChunk>(json!({
                "id": "chatcmpl_stream_1",
                "object": "chat.completion.chunk",
                "created": 1741569955,
                "model": "gpt-5.4",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": ":\"anda\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8
                }
            }))
            .unwrap(),
        ];

        let response = chat_completion_response_from_stream_chunks(chunks).unwrap();
        assert!(!response.maybe_failed());

        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Hello");
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "lookup");
        assert_eq!(output.tool_calls[0].args, json!({"q": "anda"}));
        assert_eq!(output.usage.input_tokens, 5);
        assert_eq!(output.usage.output_tokens, 3);
    }

    #[test]
    fn aggregates_responses_stream_completed_event() {
        let events = vec![
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "response.created",
                "response": {
                    "id": "resp_stream_1",
                    "created_at": 1741569956,
                    "model": "gpt-5.4",
                    "output": [],
                    "status": "in_progress",
                    "usage": null
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_stream_1",
                    "created_at": 1741569956,
                    "model": "gpt-5.4",
                    "output": [{
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi"}]
                    }],
                    "status": "completed",
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6
                    }
                }
            }))
            .unwrap(),
        ];

        let response = responses_response_from_stream_events(events).unwrap();
        assert!(!response.maybe_failed());

        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Hi");
        assert_eq!(output.usage.input_tokens, 4);
        assert_eq!(output.usage.output_tokens, 2);
    }

    #[test]
    fn aggregates_responses_stream_output_item_done_fallback() {
        let events = vec![
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "response.created",
                "response": {
                    "id": "resp_stream_2",
                    "created_at": 1741569957,
                    "model": "gpt-5.5",
                    "output": [],
                    "status": "in_progress",
                    "usage": null
                }
            }))
            .unwrap(),
            serde_json::from_value::<types::StreamEvent>(json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_2",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Streamed"}]
                }
            }))
            .unwrap(),
        ];

        let response = responses_response_from_stream_events(events).unwrap();
        assert_eq!(response.output.len(), 1);

        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Streamed");
    }

    #[test]
    fn deserializes_content_parts_custom_tool_calls_and_refusals() {
        let mut response: CompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl_2",
            "object": "chat.completion",
            "created": 1741569953,
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Use SQL"}],
                    "tool_calls": [{
                        "id": "call_custom",
                        "type": "custom",
                        "custom": {"name": "sql", "input": "select 1"}
                    }]
                },
                "logprobs": {
                    "content": [{
                        "token": "Use",
                        "bytes": [85, 115, 101],
                        "logprob": -0.1,
                        "top_logprobs": []
                    }],
                    "refusal": []
                },
                "finish_reason": "tool_calls"
            }],
            "usage": null
        }))
        .unwrap();

        response.parse_output();
        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.content, "Use SQL");
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "sql");
        assert_eq!(output.tool_calls[0].args, Json::String("select 1".into()));
        assert!(output.failed_reason.is_none());

        let mut response: CompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl_3",
            "object": "chat.completion",
            "created": 1741569954,
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "policy fail"}]
                },
                "finish_reason": "stop"
            }]
        }))
        .unwrap();

        response.parse_output();
        assert!(response.maybe_failed());
        let output = response.try_into(vec![], vec![]).unwrap();
        assert_eq!(output.failed_reason.as_deref(), Some("policy fail"));
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_openai() {
        let cli = Client::new(
            "sk-or-v1-0**",
            Some("https://openrouter.ai/api/v1".to_string()),
        )
        .completion_model_v2("openai/gpt-5.4-mini");
        let res = cli
            .completion(CompletionRequest {
                prompt: "What is 1+1?".to_string(),
                ..Default::default()
            })
            .await;
        println!("res: {:#?}", res);
    }
}
