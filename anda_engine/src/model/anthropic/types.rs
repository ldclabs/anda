use anda_core::{
    AgentOutput, BoxError, ByteBufB64, ContentPart, FunctionDefinition, Message as CoreMessage,
    Usage as ModelUsage,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashMap, str::FromStr};

use crate::unix_ms;

// https://platform.claude.com/docs/en/api/messages/create
#[derive(Debug)]
pub struct RequiredMessageParams {
    pub model: String,
    pub messages: Vec<Value>, // Vec<Message>
    pub max_tokens: u32,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

/// Prompt cache marker supported by Anthropic cacheable request blocks.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct CacheControlEphemeral {
    pub r#type: CacheControlType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheControlTtl>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum CacheControlType {
    #[serde(rename = "ephemeral")]
    Ephemeral,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum CacheControlTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

/// Top-level system prompt. The API accepts either a string or text blocks.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl From<String> for SystemPrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for SystemPrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<ContentBlock>> for SystemPrompt {
    fn from(value: Vec<ContentBlock>) -> Self {
        Self::Blocks(value)
    }
}

/// Parameters for creating a message
#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct CreateMessageParams {
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
    /// Input messages for the conversation
    pub messages: Vec<Value>, // Vec<Message>
    /// Model to use
    pub model: String,
    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Top-level prompt cache marker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlEphemeral>,
    /// Container identifier for reuse across requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// Geographic region for inference processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Output format and reasoning effort options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// Anthropic service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<RequestServiceTier>,
    /// Temperature for response generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Custom stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Tools that the model may use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// How the model should use tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Configuration for enabling Claude's extended thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    /// Request metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl From<RequiredMessageParams> for CreateMessageParams {
    fn from(required: RequiredMessageParams) -> Self {
        Self {
            model: required.model,
            messages: required.messages,
            max_tokens: required.max_tokens,
            ..Default::default()
        }
    }
}

/// Configuration options for model output.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct OutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<OutputEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<JsonOutputFormat>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputEffort {
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct JsonOutputFormat {
    pub schema: Value,
    pub r#type: JsonOutputFormatType,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum JsonOutputFormatType {
    #[serde(rename = "json_schema")]
    JsonSchema,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestServiceTier {
    Auto,
    StandardOnly,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct CitationsConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum TextCitation {
    #[serde(rename = "char_location")]
    CharLocation {
        cited_text: String,
        document_index: u32,
        document_title: String,
        end_char_index: u32,
        start_char_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    #[serde(rename = "page_location")]
    PageLocation {
        cited_text: String,
        document_index: u32,
        document_title: String,
        end_page_number: u32,
        start_page_number: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    #[serde(rename = "content_block_location")]
    ContentBlockLocation {
        cited_text: String,
        document_index: u32,
        document_title: String,
        end_block_index: u32,
        start_block_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    #[serde(rename = "web_search_result_location")]
    WebSearchResultLocation {
        cited_text: String,
        encrypted_index: String,
        title: String,
        url: String,
    },
    #[serde(rename = "search_result_location")]
    SearchResultLocation {
        cited_text: String,
        end_block_index: u32,
        search_result_index: u32,
        source: String,
        start_block_index: u32,
        title: String,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
    #[serde(rename = "url")]
    Url { url: String },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum DocumentSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
    #[serde(rename = "text")]
    Text { media_type: String, data: String },
    #[serde(rename = "content")]
    Content { content: DocumentSourceContent },
    #[serde(rename = "url")]
    Url { url: String },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum DocumentSourceContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ToolCaller {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825 { tool_id: String },
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120 { tool_id: String },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum WebSearchToolResultContent {
    Error(WebSearchToolResultError),
    Results(Vec<WebSearchResultBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct WebSearchToolResultError {
    pub r#type: String,
    pub error_code: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct WebSearchResultBlock {
    pub r#type: String,
    pub encrypted_content: String,
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum WebFetchToolResultContent {
    Error(WebFetchToolResultErrorBlock),
    Result(WebFetchBlock),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct WebFetchToolResultErrorBlock {
    pub r#type: String,
    pub error_code: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct WebFetchBlock {
    pub r#type: String,
    pub content: Box<ContentBlock>,
    pub url: String,
    pub retrieved_at: String,
}

/// Message in a conversation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Content of the message (either string or array of content blocks)
    #[serde(flatten)]
    pub content: MessageContent,
}

/// Role of a message sender
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Content of a message
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text { content: String },
    /// Structured content blocks
    Blocks { content: Vec<ContentBlock> },
}

/// Content block in a message
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<TextCitation>>,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    /// Document content
    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// Search result content
    #[serde(rename = "search_result")]
    SearchResult {
        content: Vec<ContentBlock>,
        source: String,
        title: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
    },
    /// Tool use content
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        caller: Option<ToolCaller>,
    },
    /// Tool result content
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// Thinking content
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// Redacted thinking
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    /// Server-side tool use content
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        caller: Option<ToolCaller>,
    },
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        content: WebSearchToolResultContent,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        caller: Option<ToolCaller>,
    },
    #[serde(rename = "web_fetch_tool_result")]
    WebFetchToolResult {
        content: WebFetchToolResultContent,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        caller: Option<ToolCaller>,
    },
    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult {
        content: Value,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    #[serde(rename = "bash_code_execution_tool_result")]
    BashCodeExecutionToolResult {
        content: Value,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    #[serde(rename = "text_editor_code_execution_tool_result")]
    TextEditorCodeExecutionToolResult {
        content: Value,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    #[serde(rename = "tool_search_tool_result")]
    ToolSearchToolResult {
        content: Value,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    #[serde(rename = "container_upload")]
    ContainerUpload {
        file_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    #[serde(untagged)]
    Any(Value),
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match &value {
            Value::Object(map)
                if matches!(
                    map.get("type").and_then(|t| t.as_str()),
                    Some(
                        "text"
                            | "image"
                            | "document"
                            | "search_result"
                            | "tool_use"
                            | "tool_result"
                            | "thinking"
                            | "redacted_thinking"
                            | "server_tool_use"
                            | "web_search_tool_result"
                            | "web_fetch_tool_result"
                            | "code_execution_tool_result"
                            | "bash_code_execution_tool_result"
                            | "text_editor_code_execution_tool_result"
                            | "tool_search_tool_result"
                            | "container_upload"
                    )
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type")]
                enum Helper {
                    #[serde(rename = "text")]
                    Text {
                        text: String,
                        cache_control: Option<CacheControlEphemeral>,
                        citations: Option<Vec<TextCitation>>,
                    },
                    #[serde(rename = "image")]
                    Image {
                        source: ImageSource,
                        cache_control: Option<CacheControlEphemeral>,
                    },
                    #[serde(rename = "document")]
                    Document {
                        source: DocumentSource,
                        cache_control: Option<CacheControlEphemeral>,
                        citations: Option<CitationsConfig>,
                        context: Option<String>,
                        title: Option<String>,
                    },
                    #[serde(rename = "search_result")]
                    SearchResult {
                        content: Vec<ContentBlock>,
                        source: String,
                        title: String,
                        cache_control: Option<CacheControlEphemeral>,
                        citations: Option<CitationsConfig>,
                    },
                    #[serde(rename = "tool_use")]
                    ToolUse {
                        id: String,
                        name: String,
                        input: Value,
                        cache_control: Option<CacheControlEphemeral>,
                        caller: Option<ToolCaller>,
                    },
                    #[serde(rename = "tool_result")]
                    ToolResult {
                        tool_use_id: String,
                        content: Option<ToolResultContent>,
                        cache_control: Option<CacheControlEphemeral>,
                        is_error: Option<bool>,
                    },
                    #[serde(rename = "thinking")]
                    Thinking { thinking: String, signature: String },
                    #[serde(rename = "redacted_thinking")]
                    RedactedThinking { data: String },
                    #[serde(rename = "server_tool_use")]
                    ServerToolUse {
                        id: String,
                        name: String,
                        input: Value,
                        cache_control: Option<CacheControlEphemeral>,
                        caller: Option<ToolCaller>,
                    },
                    #[serde(rename = "web_search_tool_result")]
                    WebSearchToolResult {
                        content: WebSearchToolResultContent,
                        tool_use_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                        caller: Option<ToolCaller>,
                    },
                    #[serde(rename = "web_fetch_tool_result")]
                    WebFetchToolResult {
                        content: WebFetchToolResultContent,
                        tool_use_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                        caller: Option<ToolCaller>,
                    },
                    #[serde(rename = "code_execution_tool_result")]
                    CodeExecutionToolResult {
                        content: Value,
                        tool_use_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                    },
                    #[serde(rename = "bash_code_execution_tool_result")]
                    BashCodeExecutionToolResult {
                        content: Value,
                        tool_use_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                    },
                    #[serde(rename = "text_editor_code_execution_tool_result")]
                    TextEditorCodeExecutionToolResult {
                        content: Value,
                        tool_use_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                    },
                    #[serde(rename = "tool_search_tool_result")]
                    ToolSearchToolResult {
                        content: Value,
                        tool_use_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                    },
                    #[serde(rename = "container_upload")]
                    ContainerUpload {
                        file_id: String,
                        cache_control: Option<CacheControlEphemeral>,
                    },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(h) => Ok(match h {
                        Helper::Text {
                            text,
                            cache_control,
                            citations,
                        } => ContentBlock::Text {
                            text,
                            cache_control,
                            citations,
                        },
                        Helper::Image {
                            source,
                            cache_control,
                        } => ContentBlock::Image {
                            source,
                            cache_control,
                        },
                        Helper::Document {
                            source,
                            cache_control,
                            citations,
                            context,
                            title,
                        } => ContentBlock::Document {
                            source,
                            cache_control,
                            citations,
                            context,
                            title,
                        },
                        Helper::SearchResult {
                            content,
                            source,
                            title,
                            cache_control,
                            citations,
                        } => ContentBlock::SearchResult {
                            content,
                            source,
                            title,
                            cache_control,
                            citations,
                        },
                        Helper::ToolUse {
                            id,
                            name,
                            input,
                            cache_control,
                            caller,
                        } => ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            cache_control,
                            caller,
                        },
                        Helper::ToolResult {
                            tool_use_id,
                            content,
                            cache_control,
                            is_error,
                        } => ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            cache_control,
                            is_error,
                        },
                        Helper::Thinking {
                            thinking,
                            signature,
                        } => ContentBlock::Thinking {
                            thinking,
                            signature,
                        },
                        Helper::RedactedThinking { data } => {
                            ContentBlock::RedactedThinking { data }
                        }
                        Helper::ServerToolUse {
                            id,
                            name,
                            input,
                            cache_control,
                            caller,
                        } => ContentBlock::ServerToolUse {
                            id,
                            name,
                            input,
                            cache_control,
                            caller,
                        },
                        Helper::WebSearchToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                            caller,
                        } => ContentBlock::WebSearchToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                            caller,
                        },
                        Helper::WebFetchToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                            caller,
                        } => ContentBlock::WebFetchToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                            caller,
                        },
                        Helper::CodeExecutionToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        } => ContentBlock::CodeExecutionToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        },
                        Helper::BashCodeExecutionToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        } => ContentBlock::BashCodeExecutionToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        },
                        Helper::TextEditorCodeExecutionToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        } => ContentBlock::TextEditorCodeExecutionToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        },
                        Helper::ToolSearchToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        } => ContentBlock::ToolSearchToolResult {
                            content,
                            tool_use_id,
                            cache_control,
                        },
                        Helper::ContainerUpload {
                            file_id,
                            cache_control,
                        } => ContentBlock::ContainerUpload {
                            file_id,
                            cache_control,
                        },
                    }),
                    Err(_) => Ok(ContentBlock::Any(value)),
                }
            }
            _ => Ok(ContentBlock::Any(value)),
        }
    }
}

/// Tool definition
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    /// Name of the tool
    pub name: String,
    /// Description of the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema for tool input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// Built-in/server tool type, if this is not a custom client tool.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlEphemeral>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eager_input_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_examples: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool choice configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ToolChoice {
    /// Let model choose whether to use tools
    #[serde(rename = "auto")]
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Model must use one of the provided tools
    #[serde(rename = "any")]
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Model must use a specific tool
    #[serde(rename = "tool")]
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Model must not use any tools
    #[serde(rename = "none")]
    None,
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self::Auto {
            disable_parallel_tool_use: None,
        }
    }

    pub fn any() -> Self {
        Self::Any {
            disable_parallel_tool_use: None,
        }
    }

    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool {
            name: name.into(),
            disable_parallel_tool_use: None,
        }
    }
}

/// Configuration for extended thinking
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Thinking {
    /// Must be at least 1024 tokens
    #[serde(default, skip_serializing_if = "is_zero")]
    pub budget_tokens: usize,
    pub r#type: ThinkingType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<ThinkingDisplay>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ThinkingType {
    #[serde(rename = "enabled")]
    Enabled,
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "adaptive")]
    Adaptive,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
}
/// Message metadata
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Metadata {
    /// External opaque user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Custom metadata fields
    #[serde(flatten)]
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Container {
    pub id: String,
    pub expires_at: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum StopDetails {
    #[serde(rename = "refusal")]
    Refusal {
        #[serde(skip_serializing_if = "Option::is_none")]
        category: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        explanation: Option<String>,
    },
}

/// Response from creating a message
#[derive(Debug, Deserialize, Serialize)]
pub struct CreateMessageResponse {
    /// Content blocks in the response
    pub content: Vec<ContentBlock>,
    /// Unique message identifier
    pub id: String,
    /// Container used by server-side tools, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Container>,
    /// Model that handled the request
    pub model: String,
    /// Role of the message (always "assistant")
    pub role: Role,
    /// Reason for stopping generation
    pub stop_reason: Option<StopReason>,
    /// Stop sequence that was generated
    pub stop_sequence: Option<String>,
    /// Structured stop details for refusals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<StopDetails>,
    /// Type of the message
    pub r#type: String,
    /// Usage statistics
    pub usage: Usage,
}

/// Reason for stopping message generation
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CacheCreation {
    #[serde(default)]
    pub ephemeral_1h_input_tokens: u32,
    #[serde(default)]
    pub ephemeral_5m_input_tokens: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ServerToolUsage {
    #[serde(default)]
    pub web_fetch_requests: u32,
    #[serde(default)]
    pub web_search_requests: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageServiceTier {
    Standard,
    Priority,
    Batch,
}

/// Token usage statistics
#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    /// Input tokens used
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,
    /// The number of input tokens used to create the cache entry.
    #[serde(default)]
    // The number of input tokens read from the cache.
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Output tokens used
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<UsageServiceTier>,
}

impl Message {
    /// Create a new message with simple text content
    pub fn new_text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: MessageContent::Text {
                content: text.into(),
            },
        }
    }

    /// Create a new message with content blocks
    pub fn new_blocks(role: Role, blocks: Vec<ContentBlock>) -> Self {
        Self {
            role,
            content: MessageContent::Blocks { content: blocks },
        }
    }
}

// Helper methods for content blocks
impl ContentBlock {
    /// Create a new text block
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
            citations: None,
        }
    }

    /// Create a new image block
    pub fn image(
        r#type: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<String>,
    ) -> Self {
        let source_type = r#type.into();
        let data = data.into();
        Self::Image {
            source: if source_type == "url" {
                ImageSource::Url { url: data }
            } else {
                ImageSource::Base64 {
                    media_type: media_type.into(),
                    data,
                }
            },
            cache_control: None,
        }
    }
}

// --- Conversions between Anthropic types and anda_core types ---

impl From<ContentPart> for ContentBlock {
    fn from(value: ContentPart) -> Self {
        match value {
            ContentPart::Text { text } => ContentBlock::Text {
                text,
                cache_control: None,
                citations: None,
            },
            ContentPart::Reasoning { text } => ContentBlock::Thinking {
                thinking: text,
                signature: String::new(),
            },
            ContentPart::FileData {
                file_uri,
                mime_type,
            } if (file_uri.starts_with("data:") || file_uri.starts_with("https://"))
                && mime_type
                    .as_deref()
                    .map(|v| v.starts_with("image/"))
                    .unwrap_or(false) =>
            {
                ContentBlock::Image {
                    source: ImageSource::Url { url: file_uri },
                    cache_control: None,
                }
            }
            ContentPart::FileData { file_uri, .. }
                if file_uri.starts_with("data:") || file_uri.starts_with("https://") =>
            {
                ContentBlock::Document {
                    source: DocumentSource::Url { url: file_uri },
                    cache_control: None,
                    citations: None,
                    context: None,
                    title: None,
                }
            }
            ContentPart::InlineData { mime_type, data } if mime_type.starts_with("image/") => {
                ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: mime_type,
                        data: data.to_base64(),
                    },
                    cache_control: None,
                }
            }
            ContentPart::InlineData { mime_type, data } => match String::from_utf8(data.0) {
                Ok(text) => ContentBlock::Document {
                    source: DocumentSource::Text {
                        media_type: mime_type,
                        data: text,
                    },
                    cache_control: None,
                    citations: None,
                    context: None,
                    title: None,
                },
                Err(v) => ContentBlock::Document {
                    source: DocumentSource::Base64 {
                        media_type: mime_type,
                        data: ByteBufB64(v.into_bytes()).to_base64(),
                    },
                    cache_control: None,
                    citations: None,
                    context: None,
                    title: None,
                },
            },
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => ContentBlock::ToolUse {
                id: call_id.unwrap_or_default(),
                name,
                input: args,
                cache_control: None,
                caller: None,
            },
            ContentPart::ToolOutput {
                name: _,
                output,
                call_id,
                ..
            } => ContentBlock::ToolResult {
                tool_use_id: call_id.unwrap_or_default(),
                content: Some(ToolResultContent::Text(match &output {
                    Value::String(s) => s.clone(),
                    _ => serde_json::to_string(&output).unwrap_or_default(),
                })),
                cache_control: None,
                is_error: None,
            },
            ContentPart::Any(json) => {
                serde_json::from_value(json.clone()).unwrap_or_else(|_| ContentBlock::Text {
                    text: serde_json::to_string(&json).unwrap_or_default(),
                    cache_control: None,
                    citations: None,
                })
            }
            other => ContentBlock::Text {
                text: serde_json::to_string(&other).unwrap_or_default(),
                cache_control: None,
                citations: None,
            },
        }
    }
}

impl From<ContentBlock> for ContentPart {
    fn from(value: ContentBlock) -> Self {
        match value {
            ContentBlock::Text { text, .. } => ContentPart::Text { text },
            ContentBlock::Image { source, .. } => match source {
                ImageSource::Base64 { media_type, data } => match ByteBufB64::from_str(&data) {
                    Ok(data) => ContentPart::InlineData {
                        mime_type: media_type,
                        data,
                    },
                    Err(_) => ContentPart::Any(json!({
                        "type": "image",
                        "source": { "type": "base64", "media_type": media_type, "data": data },
                    })),
                },
                ImageSource::Url { url } => ContentPart::FileData {
                    file_uri: url,
                    mime_type: None,
                },
            },
            ContentBlock::Document { source, .. } => match source {
                DocumentSource::Base64 { media_type, data } => match ByteBufB64::from_str(&data) {
                    Ok(data) => ContentPart::InlineData {
                        mime_type: media_type,
                        data,
                    },
                    Err(_) => ContentPart::Any(json!({
                        "type": "document",
                        "source": { "type": "base64", "media_type": media_type, "data": data },
                    })),
                },
                DocumentSource::Text { data, .. } => ContentPart::Text { text: data },
                DocumentSource::Content { content } => ContentPart::Any(json!({
                    "type": "document",
                    "source": { "type": "content", "content": content },
                })),
                DocumentSource::Url { url } => ContentPart::FileData {
                    file_uri: url,
                    mime_type: None,
                },
            },
            block @ ContentBlock::SearchResult { .. } => ContentPart::Any(json!(block)),
            ContentBlock::ServerToolUse {
                id, name, input, ..
            }
            | ContentBlock::ToolUse {
                id, name, input, ..
            } => ContentPart::ToolCall {
                name,
                args: input,
                call_id: if id.is_empty() { None } else { Some(id) },
            },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                let output = match content {
                    Some(ToolResultContent::Text(content)) => Value::String(content),
                    Some(ToolResultContent::Blocks(blocks)) => Value::Array(
                        blocks
                            .into_iter()
                            .map(|block| serde_json::to_value(block).unwrap_or(Value::Null))
                            .collect(),
                    ),
                    None => Value::Null,
                };

                ContentPart::ToolOutput {
                    name: String::new(),
                    output,
                    call_id: if tool_use_id.is_empty() {
                        None
                    } else {
                        Some(tool_use_id)
                    },
                    remote_id: None,
                }
            }
            ContentBlock::Thinking { thinking, .. } => ContentPart::Reasoning { text: thinking },
            ContentBlock::RedactedThinking { data } => ContentPart::Any(json!({
                "type": "redacted_thinking",
                "data": data,
            })),
            block @ (ContentBlock::WebSearchToolResult { .. }
            | ContentBlock::WebFetchToolResult { .. }
            | ContentBlock::CodeExecutionToolResult { .. }
            | ContentBlock::BashCodeExecutionToolResult { .. }
            | ContentBlock::TextEditorCodeExecutionToolResult { .. }
            | ContentBlock::ToolSearchToolResult { .. }
            | ContentBlock::ContainerUpload { .. }) => ContentPart::Any(json!(block)),
            ContentBlock::Any(value) => ContentPart::Any(value),
        }
    }
}

impl From<CoreMessage> for Message {
    fn from(msg: CoreMessage) -> Self {
        let role = match msg.role.as_str() {
            "assistant" | "model" => Role::Assistant,
            _ => Role::User,
        };
        let blocks: Vec<ContentBlock> = msg.content.into_iter().map(|v| v.into()).collect();
        if blocks.len() == 1
            && let Some(ContentBlock::Text { text, .. }) = blocks.first()
        {
            return Message {
                role,
                content: MessageContent::Text {
                    content: text.clone(),
                },
            };
        }
        Message {
            role,
            content: MessageContent::Blocks { content: blocks },
        }
    }
}

impl From<Message> for CoreMessage {
    fn from(msg: Message) -> Self {
        let role = match msg.role {
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
        };
        let content = match msg.content {
            MessageContent::Text { content } => vec![ContentPart::Text { text: content }],
            MessageContent::Blocks { content } => content.into_iter().map(|v| v.into()).collect(),
        };
        CoreMessage {
            role,
            content,
            ..Default::default()
        }
    }
}

impl From<FunctionDefinition> for Tool {
    fn from(def: FunctionDefinition) -> Self {
        Tool {
            name: def.name,
            description: if def.description.is_empty() {
                None
            } else {
                Some(def.description)
            },
            input_schema: Some(def.parameters),
            r#type: None,
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            eager_input_streaming: None,
            input_examples: None,
            strict: None,
        }
    }
}

impl CreateMessageResponse {
    pub fn try_into(
        self,
        raw_history: Vec<Value>,
        chat_history: Vec<CoreMessage>,
    ) -> Result<AgentOutput, BoxError> {
        let timestamp = unix_ms();
        let mut output = AgentOutput {
            raw_history,
            chat_history,
            model: Some(self.model.clone()),
            // Total input tokens in a request is the summation of input_tokens, cache_creation_input_tokens, and cache_read_input_tokens.
            usage: ModelUsage {
                input_tokens: (self.usage.input_tokens
                    + self.usage.cache_creation_input_tokens
                    + self.usage.cache_read_input_tokens) as u64,
                output_tokens: self.usage.output_tokens as u64,
                cached_tokens: self.usage.cache_read_input_tokens as u64,
                requests: 1,
            },
            ..Default::default()
        };

        if self.content.is_empty() {
            output.failed_reason = serde_json::to_string(&self.stop_reason).ok();
        } else {
            let content_parts: Vec<ContentPart> =
                self.content.iter().cloned().map(|v| v.into()).collect();
            output.raw_history.push(json!({
                "role": self.role,
                "content": self.content,
            }));

            let msg = CoreMessage {
                role: "assistant".to_string(),
                content: content_parts,
                name: Some(self.model),
                timestamp: Some(timestamp),
                ..Default::default()
            };

            match self.stop_reason {
                Some(StopReason::EndTurn)
                | Some(StopReason::StopSequence)
                | Some(StopReason::ToolUse) => {
                    output.content = msg.text().unwrap_or_default();
                    output.thoughts = msg.thoughts();
                    output.tool_calls = msg.tool_calls();
                }
                v => {
                    output.failed_reason = serde_json::to_string(&v).ok();
                }
            }
            output.chat_history.push(msg);
        }

        Ok(output)
    }

    pub fn maybe_failed(&self) -> bool {
        !matches!(
            self.stop_reason,
            Some(StopReason::EndTurn) | Some(StopReason::StopSequence) | Some(StopReason::ToolUse)
        )
    }
}

#[derive(Debug, Serialize, Default)]
pub struct CountMessageTokensParams {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
}

#[derive(Debug, Deserialize)]
pub struct CountMessageTokensResponse {
    pub input_tokens: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartContent },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaContent,
        usage: Option<Usage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: StreamError },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessageStartContent {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    #[serde(rename = "citations_delta")]
    CitationsDelta { citation: TextCitation },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessageDeltaContent {
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamError {
    pub r#type: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_block_deserialize_falls_back_to_any_for_unknown_or_invalid_known_blocks() {
        let unknown = json!({"type": "mystery_block", "foo": "bar"});
        let block: ContentBlock = serde_json::from_value(unknown.clone()).unwrap();
        assert_eq!(block, ContentBlock::Any(unknown));

        let malformed_known = json!({"type": "tool_use", "id": "toolu_1"});
        let block: ContentBlock = serde_json::from_value(malformed_known.clone()).unwrap();
        assert_eq!(block, ContentBlock::Any(malformed_known));
    }

    #[test]
    fn converts_messages_between_core_and_anthropic_shapes() {
        let anthropic: Message = CoreMessage {
            role: "assistant".to_string(),
            content: vec![ContentPart::Text {
                text: "hello".to_string(),
            }],
            ..Default::default()
        }
        .into();

        assert!(matches!(anthropic.role, Role::Assistant));
        assert_eq!(
            anthropic.content,
            MessageContent::Text {
                content: "hello".to_string()
            }
        );

        let core: CoreMessage = Message::new_blocks(
            Role::User,
            vec![
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "lookup".to_string(),
                    input: json!({"query": "anda"}),
                    cache_control: None,
                    caller: Some(ToolCaller::Direct),
                },
                ContentBlock::Thinking {
                    thinking: "plan".to_string(),
                    signature: "sig_1".to_string(),
                },
            ],
        )
        .into();

        assert_eq!(core.role, "user");
        assert!(matches!(
            &core.content[0],
            ContentPart::ToolCall {
                name,
                args,
                call_id: Some(call_id),
            } if name == "lookup" && call_id == "toolu_1" && args == &json!({"query": "anda"})
        ));
        assert!(matches!(
            &core.content[1],
            ContentPart::Reasoning { text } if text == "plan"
        ));
    }

    #[test]
    fn converts_content_parts_between_core_and_anthropic_blocks() {
        let image_part = ContentPart::InlineData {
            mime_type: "image/png".to_string(),
            data: ByteBufB64(vec![1, 2, 3, 4]),
        };
        let image_block: ContentBlock = image_part.clone().into();
        assert!(matches!(
            &image_block,
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
                ..
            } if media_type == "image/png" && data == &ByteBufB64(vec![1, 2, 3, 4]).to_base64()
        ));
        assert_eq!(ContentPart::from(image_block), image_part);

        let image_url_block: ContentBlock = ContentPart::FileData {
            file_uri: "https://example.com/image.png".to_string(),
            mime_type: Some("image/png".to_string()),
        }
        .into();
        assert!(matches!(
            image_url_block,
            ContentBlock::Image {
                source: ImageSource::Url { url },
                ..
            } if url == "https://example.com/image.png"
        ));

        let document_block = ContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: "not-base64".to_string(),
            },
            cache_control: None,
            citations: None,
            context: None,
            title: None,
        };
        let document_part: ContentPart = document_block.into();
        assert!(matches!(
            document_part,
            ContentPart::Any(value)
                if value["type"] == "document"
                && value["source"]["type"] == "base64"
                && value["source"]["data"] == "not-base64"
        ));
    }

    #[test]
    fn converts_non_remote_file_data_to_text_block() {
        let file_part = ContentPart::FileData {
            file_uri: "file:///tmp/report.pdf".to_string(),
            mime_type: Some("application/pdf".to_string()),
        };

        let block: ContentBlock = file_part.clone().into();
        match block {
            ContentBlock::Text { text, .. } => {
                let parsed: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(parsed, serde_json::to_value(&file_part).unwrap());
            }
            _ => panic!("non-remote FileData should fall back to Text"),
        }
    }

    #[test]
    fn try_into_builds_success_output_for_tool_use_stop_reason() {
        let response = CreateMessageResponse {
            content: vec![
                ContentBlock::text("Need lookup"),
                ContentBlock::Thinking {
                    thinking: "Check the tool".to_string(),
                    signature: "sig_1".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "lookup".to_string(),
                    input: json!({"query": "anda"}),
                    cache_control: None,
                    caller: None,
                },
            ],
            id: "msg_1".to_string(),
            container: None,
            model: "claude-sonnet-4-5".to_string(),
            role: Role::Assistant,
            stop_reason: Some(StopReason::ToolUse),
            stop_sequence: None,
            stop_details: None,
            r#type: "message".to_string(),
            usage: Usage {
                input_tokens: 10,
                cache_creation: None,
                cache_creation_input_tokens: 2,
                cache_read_input_tokens: 3,
                inference_geo: None,
                output_tokens: 4,
                server_tool_use: None,
                service_tier: None,
            },
        };

        assert!(!response.maybe_failed());

        let output = response.try_into(Vec::new(), Vec::new()).unwrap();
        assert_eq!(output.model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(output.content, "Need lookup");
        assert_eq!(output.thoughts.as_deref(), Some("Check the tool"));
        assert!(output.failed_reason.is_none());
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "lookup");
        assert_eq!(output.tool_calls[0].call_id.as_deref(), Some("toolu_1"));
        assert_eq!(output.usage.input_tokens, 15);
        assert_eq!(output.usage.output_tokens, 4);
        assert_eq!(output.usage.cached_tokens, 3);
        assert_eq!(output.chat_history.len(), 1);
        assert_eq!(output.raw_history.len(), 1);
    }

    #[test]
    fn try_into_marks_failed_outputs_for_empty_content_or_pause_turn() {
        let empty_response = CreateMessageResponse {
            content: Vec::new(),
            id: "msg_empty".to_string(),
            container: None,
            model: "claude-haiku-4-5".to_string(),
            role: Role::Assistant,
            stop_reason: Some(StopReason::Refusal),
            stop_sequence: None,
            stop_details: Some(StopDetails::Refusal {
                category: Some("safety".to_string()),
                explanation: Some("blocked".to_string()),
            }),
            r#type: "message".to_string(),
            usage: Usage {
                input_tokens: 1,
                cache_creation: None,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                inference_geo: None,
                output_tokens: 0,
                server_tool_use: None,
                service_tier: None,
            },
        };
        assert!(empty_response.maybe_failed());

        let empty_output = empty_response.try_into(Vec::new(), Vec::new()).unwrap();
        assert_eq!(empty_output.failed_reason.as_deref(), Some("\"refusal\""));
        assert!(empty_output.chat_history.is_empty());

        let paused_response = CreateMessageResponse {
            content: vec![ContentBlock::text("partial")],
            id: "msg_pause".to_string(),
            container: None,
            model: "claude-haiku-4-5".to_string(),
            role: Role::Assistant,
            stop_reason: Some(StopReason::PauseTurn),
            stop_sequence: None,
            stop_details: None,
            r#type: "message".to_string(),
            usage: Usage {
                input_tokens: 1,
                cache_creation: None,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                inference_geo: None,
                output_tokens: 1,
                server_tool_use: None,
                service_tier: None,
            },
        };
        assert!(paused_response.maybe_failed());

        let paused_output = paused_response.try_into(Vec::new(), Vec::new()).unwrap();
        assert_eq!(
            paused_output.failed_reason.as_deref(),
            Some("\"pause_turn\"")
        );
        assert_eq!(paused_output.chat_history.len(), 1);
        assert_eq!(paused_output.content, "");
    }

    #[test]
    fn converts_required_params_and_function_definitions_into_request_shapes() {
        let params: CreateMessageParams = RequiredMessageParams {
            model: "claude-opus-4-6".to_string(),
            messages: vec![json!({"role": "user", "content": "hi"})],
            max_tokens: 256,
        }
        .into();
        assert_eq!(params.model, "claude-opus-4-6");
        assert_eq!(params.max_tokens, 256);
        assert_eq!(params.messages.len(), 1);
        assert!(params.tools.is_none());

        let tool: Tool = FunctionDefinition {
            name: "lookup".to_string(),
            description: String::new(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
        .into();
        assert_eq!(tool.name, "lookup");
        assert!(tool.description.is_none());
        assert_eq!(
            tool.input_schema,
            Some(json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }))
        );
        assert!(tool.strict.is_none());
    }

    #[test]
    fn serializes_request_with_system_blocks_and_tool_choice_options() {
        let req = CreateMessageParams {
            max_tokens: 1024,
            messages: vec![json!({"role": "user", "content": "hi"})],
            model: "claude-opus-4-6".to_string(),
            system: Some(vec![ContentBlock::text("Today is 2026-05-16.")].into()),
            tool_choice: Some(ToolChoice::Tool {
                name: "lookup".to_string(),
                disable_parallel_tool_use: Some(true),
            }),
            thinking: Some(Thinking {
                budget_tokens: 0,
                r#type: ThinkingType::Adaptive,
                display: Some(ThinkingDisplay::Omitted),
            }),
            output_config: Some(OutputConfig {
                effort: Some(OutputEffort::Max),
                format: Some(JsonOutputFormat {
                    schema: json!({"type": "object"}),
                    r#type: JsonOutputFormatType::JsonSchema,
                }),
            }),
            ..Default::default()
        };

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["tool_choice"]["disable_parallel_tool_use"], true);
        assert_eq!(value["thinking"]["type"], "adaptive");
        assert!(value["thinking"].get("budget_tokens").is_none());
        assert_eq!(value["output_config"]["effort"], "max");
    }

    #[test]
    fn deserializes_response_with_server_tool_and_extended_usage() {
        let response: CreateMessageResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "container": {"id": "container_1", "expires_at": "2026-05-16T00:00:00Z"},
            "content": [{
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "anda"},
                "caller": {"type": "direct"}
            }],
            "stop_reason": "pause_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "output_tokens": 4,
                "cache_creation": {"ephemeral_5m_input_tokens": 2},
                "server_tool_use": {"web_search_requests": 1},
                "service_tier": "priority"
            }
        }))
        .unwrap();

        assert!(matches!(response.stop_reason, Some(StopReason::PauseTurn)));
        assert!(matches!(
            response.content.first(),
            Some(ContentBlock::ServerToolUse { name, .. }) if name == "web_search"
        ));
        assert_eq!(
            response
                .usage
                .cache_creation
                .unwrap()
                .ephemeral_5m_input_tokens,
            2
        );
        assert!(matches!(
            response.usage.server_tool_use,
            Some(ServerToolUsage {
                web_search_requests: 1,
                ..
            })
        ));
    }

    #[test]
    fn preserves_tool_result_block_content_as_tool_output_json() {
        let block: ContentBlock = serde_json::from_value(json!({
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [{"type": "text", "text": "ok"}],
            "is_error": false
        }))
        .unwrap();

        assert!(matches!(
            &block,
            ContentBlock::ToolResult {
                content: Some(ToolResultContent::Blocks(blocks)),
                is_error: Some(false),
                ..
            } if matches!(blocks.first(), Some(ContentBlock::Text { text, .. }) if text == "ok")
        ));

        let part: ContentPart = block.into();
        assert!(matches!(
            part,
            ContentPart::ToolOutput {
                output: Value::Array(values),
                call_id: Some(call_id),
                ..
            } if call_id == "toolu_1" && values[0]["type"] == "text"
        ));
    }
}
