use anda_core::{
    AgentOutput, BoxError, ByteBufB64, ContentPart, FunctionDefinition, Message as CoreMessage,
    Usage as ModelUsage,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::str::FromStr;

use crate::unix_ms;

// https://platform.claude.com/docs/en/api/messages/create
#[derive(Debug)]
pub struct RequiredMessageParams {
    pub model: String,
    pub messages: Vec<Value>, // Vec<Message>
    pub max_tokens: u32,
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
    pub system: Option<String>,
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
    Text { text: String },
    /// Image content
    #[serde(rename = "image")]
    Image { source: ImageSource },
    /// Tool use content
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result content
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    /// Thinking content
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// Redacted thinking
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
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
                            | "tool_use"
                            | "tool_result"
                            | "thinking"
                            | "redacted_thinking"
                    )
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type")]
                enum Helper {
                    #[serde(rename = "text")]
                    Text { text: String },
                    #[serde(rename = "image")]
                    Image { source: ImageSource },
                    #[serde(rename = "tool_use")]
                    ToolUse {
                        id: String,
                        name: String,
                        input: Value,
                    },
                    #[serde(rename = "tool_result")]
                    ToolResult {
                        tool_use_id: String,
                        content: String,
                    },
                    #[serde(rename = "thinking")]
                    Thinking { thinking: String, signature: String },
                    #[serde(rename = "redacted_thinking")]
                    RedactedThinking { data: String },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(h) => Ok(match h {
                        Helper::Text { text } => ContentBlock::Text { text },
                        Helper::Image { source } => ContentBlock::Image { source },
                        Helper::ToolUse { id, name, input } => {
                            ContentBlock::ToolUse { id, name, input }
                        }
                        Helper::ToolResult {
                            tool_use_id,
                            content,
                        } => ContentBlock::ToolResult {
                            tool_use_id,
                            content,
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
                    }),
                    Err(_) => Ok(ContentBlock::Any(value)),
                }
            }
            _ => Ok(ContentBlock::Any(value)),
        }
    }
}

/// Source of an image
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ImageSource {
    /// Type of image source
    pub r#type: String,
    /// Media type of the image
    pub media_type: String,
    /// Base64-encoded image data
    pub data: String,
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
    pub input_schema: serde_json::Value,
}

/// Tool choice configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ToolChoice {
    /// Let model choose whether to use tools
    #[serde(rename = "auto")]
    Auto,
    /// Model must use one of the provided tools
    #[serde(rename = "any")]
    Any,
    /// Model must use a specific tool
    #[serde(rename = "tool")]
    Tool { name: String },
    /// Model must not use any tools
    #[serde(rename = "none")]
    None,
}

/// Configuration for extended thinking
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Thinking {
    /// Must be at least 1024 tokens
    pub budget_tokens: usize,
    pub r#type: ThinkingType,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ThinkingType {
    #[serde(rename = "enabled")]
    Enabled,
}
/// Message metadata
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Metadata {
    /// Custom metadata fields
    #[serde(flatten)]
    pub fields: std::collections::HashMap<String, String>,
}

/// Response from creating a message
#[derive(Debug, Deserialize, Serialize)]
pub struct CreateMessageResponse {
    /// Content blocks in the response
    pub content: Vec<ContentBlock>,
    /// Unique message identifier
    pub id: String,
    /// Model that handled the request
    pub model: String,
    /// Role of the message (always "assistant")
    pub role: Role,
    /// Reason for stopping generation
    pub stop_reason: Option<StopReason>,
    /// Stop sequence that was generated
    pub stop_sequence: Option<String>,
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
    Refusal,
}

/// Token usage statistics
#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    /// Input tokens used
    #[serde(default)]
    pub input_tokens: u32,
    /// The number of input tokens used to create the cache entry.
    #[serde(default)]
    // The number of input tokens read from the cache.
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Output tokens used
    pub output_tokens: u32,
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
        Self::Text { text: text.into() }
    }

    /// Create a new image block
    pub fn image(
        r#type: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<String>,
    ) -> Self {
        Self::Image {
            source: ImageSource {
                r#type: r#type.into(),
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }
}

// --- Conversions between Anthropic types and anda_core types ---

impl From<ContentPart> for ContentBlock {
    fn from(value: ContentPart) -> Self {
        match value {
            ContentPart::Text { text } => ContentBlock::Text { text },
            ContentPart::Reasoning { text } => ContentBlock::Thinking {
                thinking: text,
                signature: String::new(),
            },
            ContentPart::InlineData { mime_type, data } => ContentBlock::Image {
                source: ImageSource {
                    r#type: "base64".to_string(),
                    media_type: mime_type,
                    data: data.to_string(),
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
            },
            ContentPart::ToolOutput {
                name: _,
                output,
                call_id,
                ..
            } => ContentBlock::ToolResult {
                tool_use_id: call_id.unwrap_or_default(),
                content: match &output {
                    Value::String(s) => s.clone(),
                    _ => serde_json::to_string(&output).unwrap_or_default(),
                },
            },
            ContentPart::Any(json) => {
                serde_json::from_value(json.clone()).unwrap_or_else(|_| ContentBlock::Text {
                    text: serde_json::to_string(&json).unwrap_or_default(),
                })
            }
            other => ContentBlock::Text {
                text: serde_json::to_string(&other).unwrap_or_default(),
            },
        }
    }
}

impl From<ContentBlock> for ContentPart {
    fn from(value: ContentBlock) -> Self {
        match value {
            ContentBlock::Text { text } => ContentPart::Text { text },
            ContentBlock::Image { source } => {
                if source.r#type == "base64" {
                    match ByteBufB64::from_str(&source.data) {
                        Ok(data) => ContentPart::InlineData {
                            mime_type: source.media_type,
                            data,
                        },
                        Err(_) => ContentPart::Any(json!({
                            "type": "image",
                            "source": source,
                        })),
                    }
                } else {
                    ContentPart::FileData {
                        file_uri: source.data,
                        mime_type: Some(source.media_type),
                    }
                }
            }
            ContentBlock::ToolUse { id, name, input } => ContentPart::ToolCall {
                name,
                args: input,
                call_id: if id.is_empty() { None } else { Some(id) },
            },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => ContentPart::ToolOutput {
                name: String::new(),
                output: Value::String(content),
                call_id: if tool_use_id.is_empty() {
                    None
                } else {
                    Some(tool_use_id)
                },
                remote_id: None,
            },
            ContentBlock::Thinking { thinking, .. } => ContentPart::Reasoning { text: thinking },
            ContentBlock::RedactedThinking { data } => ContentPart::Any(json!({
                "type": "redacted_thinking",
                "data": data,
            })),
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
            && let Some(ContentBlock::Text { text }) = blocks.first()
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
            input_schema: def.parameters,
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
