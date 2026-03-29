use anda_core::{AgentOutput, BoxError, ContentPart, Json, Message, Usage as ModelUsage};
use serde::{Deserialize, Serialize};
use serde_json::Map;

use crate::unix_ms;

/// The completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct CompletionRequest {
    /// Message inputs
    pub input: Vec<MessageItem>,
    /// The model name
    pub model: String,
    /// Instructions (also referred to as preamble, although in other APIs this would be the "system prompt")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The maximum number of output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Toggle to true for streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// The temperature. Set higher (up to a max of 1.0) for more creative responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Controls which (if any) tool is called by the model. "none", "auto", "required"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// The tools you want to use. Currently this is limited to functions, but will be expanded on in future.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Additional parameters
    #[serde(flatten)]
    pub additional_parameters: AdditionalParameters,
}

/// Additional parameters for the completion request type for OpenAI's Response API
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct AdditionalParameters {
    /// Whether or not a given model task should run in the background (ie a detached process).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// The text response format. This is where you would add structured outputs (if you want them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    /// What types of extra data you would like to include. This is mostly useless at the moment since the types of extra data to add is currently unsupported, but this will be coming soon!
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// `top_p`. Mutually exclusive with the `temperature` argument.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Whether or not the response should be truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationStrategy>,
    /// The username of the user (that you want to use).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Any additional metadata you'd like to add. This will additionally be returned by the response.
    #[serde(skip_serializing_if = "Map::is_empty", default)]
    pub metadata: serde_json::Map<String, Json>,
    /// Whether or not you want tool calls to run in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Previous response ID. If you are not sending a full conversation, this can help to track the message flow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Add thinking/reasoning to your response. The response will be emitted as a list member of the `output` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// The service tier you're using.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAIServiceTier>,
    /// Whether or not to store the response for later retrieval by API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// The standard response format from OpenAI's Responses API.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// The ID of a completion response.
    pub id: String,
    /// UNIX epoch in seconds.
    pub created_at: u64,
    #[serde(default)]
    pub output_text: String,
    /// Response error (optional)
    pub error: Option<ResponseError>,
    /// Incomplete response details (optional)
    pub incomplete_details: Option<IncompleteDetailsReason>,
    /// System prompt/preamble
    pub instructions: Option<String>,
    pub metadata: Option<Json>,
    /// The model name
    pub model: String,
    /// Token usage
    #[serde(default)]
    pub usage: ResponsesUsage,
    /// The model output (messages, etc will go here)
    pub output: Vec<Json>,
    #[serde(default)]
    pub parsed_output: Vec<MessageItem>,
    /// Tools
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// The status of the response.
    #[serde(default)]
    pub status: ResponseStatus,
    /// Additional parameters
    #[serde(flatten)]
    pub additional_parameters: AdditionalParameters,
}

impl CompletionResponse {
    pub fn parse_output(&mut self) {
        // If parsing fails, parsed_output will just be empty, and the raw output can be used instead.
        self.parsed_output = self
            .output
            .iter()
            .filter_map(|item| serde_json::from_value::<MessageItem>(item.clone()).ok())
            .collect();
    }

    pub fn try_into(
        mut self,
        raw_history: Vec<Json>,
        chat_history: Vec<Message>,
    ) -> Result<AgentOutput, BoxError> {
        let timestamp = unix_ms();
        let mut output = AgentOutput {
            raw_history,
            chat_history,
            usage: ModelUsage {
                input_tokens: self.usage.input_tokens,
                output_tokens: self.usage.output_tokens,
                requests: 1,
            },
            ..Default::default()
        };

        if let Some(error) = self.error {
            output.failed_reason = serde_json::to_string(&error).ok();
        } else {
            output.raw_history.append(&mut self.output);

            let (msg, failed_reason) = message_from(self.parsed_output);
            if let Some(mut msg) = msg {
                msg.name = Some(self.model);
                msg.timestamp = Some(timestamp);
                output.content = msg.text().unwrap_or_default();
                output.tool_calls = msg.tool_calls();
                output.chat_history.push(msg);
            }
            output.failed_reason = failed_reason;
        }

        Ok(output)
    }

    pub fn maybe_failed(&self) -> bool {
        self.error.is_some()
            || self.parsed_output.is_empty()
            || self.parsed_output.iter().any(|item| {
                if let MessageItem::Message { content, .. } = item {
                    content
                        .iter()
                        .any(|c| matches!(c, ContentItem::Refusal { .. }))
                } else {
                    false
                }
            })
    }
}

/// An input item for CompletionRequest.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageItem {
    Message {
        role: String,
        content: Vec<ContentItem>,
        // in output message
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        // in output message
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    FunctionCall {
        name: String,
        arguments: String, // JSON format
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    FunctionCallOutput {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: FunctionCallOutput,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ReasoningContent>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(untagged)]
    Any(Json),
}

impl<'de> Deserialize<'de> for MessageItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Json::deserialize(deserializer)?;
        match &value {
            Json::Object(map)
                if matches!(
                    map.get("type").and_then(|t| t.as_str()),
                    Some("message" | "function_call" | "function_call_output" | "reasoning")
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type", rename_all = "snake_case")]
                enum Helper {
                    Message {
                        role: String,
                        content: Vec<ContentItem>,
                        status: Option<String>,
                        id: Option<String>,
                    },
                    FunctionCall {
                        name: String,
                        arguments: String,
                        call_id: String,
                        id: Option<String>,
                        namespace: Option<String>,
                        status: Option<String>,
                    },
                    FunctionCallOutput {
                        id: Option<String>,
                        call_id: String,
                        output: FunctionCallOutput,
                        status: Option<String>,
                    },
                    Reasoning {
                        id: String,
                        summary: Vec<ReasoningSummary>,
                        content: Option<Vec<ReasoningContent>>,
                        encrypted_content: Option<String>,
                        status: Option<String>,
                    },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(h) => Ok(match h {
                        Helper::Message {
                            role,
                            content,
                            status,
                            id,
                        } => MessageItem::Message {
                            role,
                            content,
                            status,
                            id,
                        },
                        Helper::FunctionCall {
                            name,
                            arguments,
                            call_id,
                            id,
                            namespace,
                            status,
                        } => MessageItem::FunctionCall {
                            name,
                            arguments,
                            call_id,
                            id,
                            namespace,
                            status,
                        },
                        Helper::FunctionCallOutput {
                            id,
                            call_id,
                            output,
                            status,
                        } => MessageItem::FunctionCallOutput {
                            id,
                            call_id,
                            output,
                            status,
                        },
                        Helper::Reasoning {
                            id,
                            summary,
                            content,
                            encrypted_content,
                            status,
                        } => MessageItem::Reasoning {
                            id,
                            summary,
                            content,
                            encrypted_content,
                            status,
                        },
                    }),
                    Err(_) => Ok(MessageItem::Any(value)),
                }
            }
            _ => Ok(MessageItem::Any(value)),
        }
    }
}

pub fn message_into(msg: Message) -> Vec<MessageItem> {
    let mut rt: Vec<MessageItem> = Vec::new();
    let mut content: Vec<ContentItem> = Vec::new();

    for part in msg.content {
        match part {
            ContentPart::Text { text } => {
                content.push(ContentItem::Text { text });
            }
            ContentPart::Reasoning { text } => {
                if !content.is_empty() {
                    rt.push(MessageItem::Message {
                        role: msg.role.clone(),
                        content,
                        status: None,
                        id: None,
                    });
                    content = Vec::new();
                }
                rt.push(MessageItem::Reasoning {
                    id: "".to_string(),
                    summary: vec![ReasoningSummary::SummaryText { text }],
                    content: None,
                    encrypted_content: None,
                    status: None,
                });
            }
            ContentPart::FileData { file_uri, .. } => {
                content.push(ContentItem::File {
                    file_data: None,
                    file_url: Some(file_uri),
                    file_id: None,
                    filename: None,
                });
            }
            ContentPart::InlineData { mime_type, data } => content.push(ContentItem::File {
                file_data: Some(format!("data:{};base64,{}", mime_type, data)),
                file_url: None,
                file_id: None,
                filename: None,
            }),
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => {
                if !content.is_empty() {
                    rt.push(MessageItem::Message {
                        role: msg.role.clone(),
                        content,
                        status: None,
                        id: None,
                    });
                    content = Vec::new();
                }
                rt.push(MessageItem::FunctionCall {
                    name,
                    arguments: serde_json::to_string(&args).unwrap_or_default(),
                    call_id: call_id.unwrap_or_default(),
                    namespace: None,
                    id: None,
                    status: None,
                });
            }
            ContentPart::ToolOutput {
                output, call_id, ..
            } => {
                if !content.is_empty() {
                    rt.push(MessageItem::Message {
                        role: msg.role.clone(),
                        content,
                        status: None,
                        id: None,
                    });
                    content = Vec::new();
                }
                rt.push(MessageItem::FunctionCallOutput {
                    output: FunctionCallOutput::String(
                        serde_json::to_string(&output).unwrap_or_default(),
                    ),
                    call_id: call_id.unwrap_or_default(),
                    id: None,
                    status: None,
                });
            }
            v => content.push(ContentItem::Any(
                serde_json::to_value(v).unwrap_or(Json::Null),
            )),
        }
    }

    if !content.is_empty() {
        rt.push(MessageItem::Message {
            role: msg.role,
            content,
            status: None,
            id: None,
        });
    }

    rt
}

pub fn message_from(output: Vec<MessageItem>) -> (Option<Message>, Option<String>) {
    let mut msg = Message {
        role: "assistant".to_string(),
        content: Vec::new(),
        ..Default::default()
    };
    let mut failed_reason: Option<String> = None;
    for item in output {
        match item {
            MessageItem::Message { role, content, .. } => {
                msg.role = role;
                for c in content {
                    match c {
                        ContentItem::Text { text } => {
                            msg.content.push(ContentPart::Text { text });
                        }
                        ContentItem::Refusal { refusal } => {
                            failed_reason = Some(refusal);
                        }
                        ContentItem::OutputText { text } => {
                            msg.content.push(ContentPart::Text { text });
                        }
                        ContentItem::Image { image_url, .. } => {
                            msg.content.push(ContentPart::FileData {
                                file_uri: image_url,
                                mime_type: Some("image".to_string()),
                            });
                        }
                        ContentItem::Audio { input_audio } => {
                            msg.content.push(ContentPart::FileData {
                                file_uri: input_audio.data,
                                mime_type: Some(format!("audio/{}", input_audio.format)),
                            });
                        }
                        ContentItem::File {
                            file_data,
                            file_url,
                            ..
                        } => {
                            msg.content.push(ContentPart::FileData {
                                file_uri: file_url.or(file_data).unwrap_or_default(),
                                mime_type: None,
                            });
                        }
                        ContentItem::Any(json) => {
                            msg.content.push(ContentPart::Any(json));
                        }
                    }
                }
            }
            MessageItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                let args: Json = serde_json::from_str(&arguments).unwrap_or_default();
                msg.content.push(ContentPart::ToolCall {
                    name,
                    args,
                    call_id: Some(call_id),
                });
            }
            MessageItem::FunctionCallOutput {
                output, call_id, ..
            } => {
                let output = match output {
                    FunctionCallOutput::String(s) => {
                        serde_json::from_str(&s).unwrap_or(Json::String(s))
                    }
                    FunctionCallOutput::Items(items) => {
                        serde_json::to_value(items).unwrap_or(Json::Array(vec![]))
                    }
                };
                msg.content.push(ContentPart::ToolOutput {
                    name: "".to_string(),
                    output,
                    call_id: Some(call_id),
                    remote_id: None,
                });
            }
            MessageItem::Reasoning {
                summary, content, ..
            } => {
                for s in summary {
                    match s {
                        ReasoningSummary::SummaryText { text } => {
                            msg.content.push(ContentPart::Reasoning { text });
                        }
                    }
                }
                if let Some(content) = content {
                    for c in content {
                        match c {
                            ReasoningContent::ReasoningText { text } => {
                                msg.content.push(ContentPart::Reasoning { text });
                            }
                        }
                    }
                }
            }
            MessageItem::Any(json) => msg.content.push(ContentPart::Any(json)),
        }
    }

    if msg.content.is_empty() {
        (None, failed_reason)
    } else {
        (Some(msg), failed_reason)
    }
}

#[derive(Debug, Serialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum ContentItem {
    #[serde(rename = "input_text")]
    Text { text: String },

    #[serde(rename = "output_text")]
    OutputText { text: String },

    #[serde(rename = "refusal")]
    Refusal { refusal: String },

    #[serde(rename = "input_image")]
    Image {
        detail: String, // One of high, low, or auto. Defaults to auto
        image_url: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    #[serde(rename = "input_file")]
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    #[serde(rename = "input_audio")]
    Audio { input_audio: InputAudio },
    #[serde(untagged)]
    Any(Json),
}

impl<'de> Deserialize<'de> for ContentItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Json::deserialize(deserializer)?;
        match &value {
            Json::Object(map)
                if matches!(
                    map.get("type").and_then(|t| t.as_str()),
                    Some(
                        "input_text"
                            | "output_text"
                            | "refusal"
                            | "input_image"
                            | "input_file"
                            | "input_audio"
                    )
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type")]
                enum Helper {
                    #[serde(rename = "input_text")]
                    Text { text: String },
                    #[serde(rename = "output_text")]
                    OutputText { text: String },
                    #[serde(rename = "refusal")]
                    Refusal { refusal: String },
                    #[serde(rename = "input_image")]
                    Image {
                        detail: String,
                        image_url: String,
                        file_id: Option<String>,
                    },
                    #[serde(rename = "input_file")]
                    File {
                        file_data: Option<String>,
                        file_url: Option<String>,
                        file_id: Option<String>,
                        filename: Option<String>,
                    },
                    #[serde(rename = "input_audio")]
                    Audio { input_audio: InputAudio },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(h) => Ok(match h {
                        Helper::Text { text } => ContentItem::Text { text },
                        Helper::OutputText { text } => ContentItem::OutputText { text },
                        Helper::Refusal { refusal } => ContentItem::Refusal { refusal },
                        Helper::Image {
                            detail,
                            image_url,
                            file_id,
                        } => ContentItem::Image {
                            detail,
                            image_url,
                            file_id,
                        },
                        Helper::File {
                            file_data,
                            file_url,
                            file_id,
                            filename,
                        } => ContentItem::File {
                            file_data,
                            file_url,
                            file_id,
                            filename,
                        },
                        Helper::Audio { input_audio } => ContentItem::Audio { input_audio },
                    }),
                    Err(_) => Ok(ContentItem::Any(value)),
                }
            }
            _ => Ok(ContentItem::Any(value)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum FunctionCallOutputItem {
    #[serde(rename = "input_text")]
    Text { text: String },

    #[serde(rename = "input_image")]
    Image {
        detail: String, // One of high, low, or auto. Defaults to auto
        image_url: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    #[serde(rename = "input_file")]
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
pub enum FunctionCallOutput {
    Items(Vec<FunctionCallOutputItem>),
    String(String),
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
    SummaryText { text: String },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningContent {
    ReasoningText { text: String },
}

/// The definition of a tool response, repurposed for OpenAI's Responses API.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Parameters - this should be a JSON schema. Tools should additionally ensure an "additionalParameters" field has been added with the value set to false, as this is required if using OpenAI's strict mode (enabled by default).
    pub parameters: Json,
    /// Whether to use strict mode. Enabled by default as it allows for improved efficiency.
    pub strict: bool,
    /// The type of tool. This should always be "function".
    pub r#type: String,
    /// Tool description.
    pub description: String,
}

/// Token usage.
/// Token usage from the OpenAI Responses API generally shows the input tokens and output tokens (both with more in-depth details) as well as a total tokens field.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResponsesUsage {
    /// Input tokens
    pub input_tokens: u64,
    /// Output tokens
    pub output_tokens: u64,
    /// Total tokens used (for a given prompt)
    pub total_tokens: u64,
}

/// Occasionally, when using OpenAI's Responses API you may get an incomplete response. This struct holds the reason as to why it happened.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IncompleteDetailsReason {
    /// The reason for an incomplete [`CompletionResponse`].
    pub reason: String,
}

/// A response error from OpenAI's Response API.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
}

/// The response status as an enum (ensures type validation)
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    #[default]
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Queued,
    Incomplete,
}

/// The truncation strategy.
/// When using auto, if the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.
/// Otherwise, does nothing (and is disabled by default).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    Auto,
    #[default]
    Disabled,
}

/// The model output format configuration.
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextConfig {
    pub format: TextFormat,
}

impl TextConfig {
    pub(crate) fn structured_output<S>(name: S, schema: Json) -> Self
    where
        S: Into<String>,
    {
        Self {
            format: TextFormat::JsonSchema(StructuredOutputsInput {
                name: name.into(),
                schema,
                strict: true,
            }),
        }
    }
}

/// The text format (contained by [`TextConfig`]).
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    JsonSchema(StructuredOutputsInput),
    #[default]
    Text,
}

/// The inputs required for adding structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuredOutputsInput {
    /// The name of your schema.
    pub name: String,
    /// Your required output schema. It is recommended that you use the JsonSchema macro, which you can check out at <https://docs.rs/schemars/latest/schemars/trait.JsonSchema.html>.
    pub schema: Json,
    /// Enable strict output. If you are using your AI agent in a data pipeline or another scenario that requires the data to be absolutely fixed to a given schema, it is recommended to set this to true.
    pub strict: bool,
}

/// Add reasoning to a [`CompletionRequest`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reasoning {
    /// How much effort you want the model to put into thinking/reasoning.
    pub effort: ReasoningEffort,
    /// How much effort you want the model to put into writing the reasoning summary.
    pub summary: ReasoningSummaryLevel,
}

/// The billing service tier that will be used. On auto by default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
}

/// The amount of reasoning effort that will be used by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    XHigh,
}

/// The amount of effort that will go into a reasoning summary by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummaryLevel {
    #[default]
    Auto,
    Concise,
    Detailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::{ContentPart, Message};
    use serde_json::json;

    #[test]
    fn test_message_into_mixed_parts() {
        let msg = Message {
            role: "user".to_string(),
            content: vec![
                ContentPart::Text {
                    text: "hello".into(),
                },
                ContentPart::Reasoning {
                    text: "thinking...".into(),
                },
                ContentPart::ToolCall {
                    name: "sum".into(),
                    args: json!({ "x": 1 }),
                    call_id: Some("c1".into()),
                },
                ContentPart::ToolOutput {
                    name: "sum".into(),
                    output: json!({ "ok": true }),
                    call_id: Some("c1".into()),
                    remote_id: None,
                },
                ContentPart::FileData {
                    file_uri: "http://a/b".into(),
                    mime_type: None,
                },
            ],
            ..Default::default()
        };

        let items = message_into(msg);
        // Expected order:
        // 0) Message (accumulated content: Text)
        // 1) Reasoning
        // 2) FunctionCall
        // 3) FunctionCallOutput
        // 4) Message (accumulated content: File)
        assert_eq!(items.len(), 5);

        // Text
        if let MessageItem::Message { role, content, .. } = &items[0] {
            assert_eq!(role, "user");
            assert_eq!(content.len(), 1);

            // content[0] should be Text("hello")
            match &content[0] {
                ContentItem::Text { text } => assert_eq!(text, "hello"),
                _ => panic!("content[0] should be Text"),
            }
        } else {
            panic!("items[0] should be Message");
        }

        // Reasoning
        if let MessageItem::Reasoning { summary, .. } = &items[1] {
            assert_eq!(
                summary,
                &vec![ReasoningSummary::SummaryText {
                    text: "thinking...".into()
                }]
            );
        } else {
            panic!("items[1] should be Reasoning");
        }

        // FunctionCall
        if let MessageItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        } = &items[2]
        {
            assert_eq!(name, "sum");
            assert_eq!(arguments, r#"{"x":1}"#);
            assert_eq!(call_id, "c1");
        } else {
            panic!("items[2] should be FunctionCall");
        }

        // FunctionCallOutput
        if let MessageItem::FunctionCallOutput {
            output, call_id, ..
        } = &items[3]
        {
            assert_eq!(
                output,
                &FunctionCallOutput::String(r#"{"ok":true}"#.to_string())
            );
            assert_eq!(call_id, "c1");
        } else {
            panic!("items[3] should be FunctionCallOutput");
        }

        // Message with accumulated non-tool content
        if let MessageItem::Message { role, content, .. } = &items[4] {
            assert_eq!(role, "user");
            assert_eq!(content.len(), 1);

            // content[0] should be File with file_url Some("http://a/b")
            match &content[0] {
                ContentItem::File {
                    file_data,
                    file_url,
                    file_id,
                    filename,
                } => {
                    assert!(file_data.is_none());
                    assert_eq!(file_url.as_deref(), Some("http://a/b"));
                    assert!(file_id.is_none());
                    assert!(filename.is_none());
                }
                _ => panic!("content[0] should be File"),
            }
        } else {
            panic!("items[4] should be Message");
        }
    }

    #[test]
    fn test_message_from_composes_message_and_failed_reason() {
        let items = vec![
            MessageItem::Message {
                role: "assistant".into(),
                content: vec![
                    ContentItem::OutputText { text: "hi".into() },
                    ContentItem::Refusal {
                        refusal: "policy fail".into(),
                    },
                ],
                status: None,
                id: None,
            },
            MessageItem::FunctionCall {
                name: "f".into(),
                arguments: r#"{"x":1}"#.into(),
                call_id: "c1".into(),
                namespace: None,
                id: None,
                status: None,
            },
            MessageItem::FunctionCallOutput {
                output: FunctionCallOutput::String(r#"{"ok":true}"#.into()),
                call_id: "c1".into(),
                id: None,
                status: None,
            },
            MessageItem::Reasoning {
                id: "".into(),
                summary: vec![ReasoningSummary::SummaryText {
                    text: "think".into(),
                }],
                content: None,
                encrypted_content: None,
                status: None,
            },
        ];

        let (msg_opt, failed_reason) = message_from(items);
        assert_eq!(failed_reason.as_deref(), Some("policy fail"));

        let msg = msg_opt.expect("message should be Some");
        assert_eq!(msg.role, "assistant");
        // Expected order in composed Message:
        // Text("hi"), ToolCall, ToolOutput, Reasoning("think")
        assert_eq!(msg.content.len(), 4);

        // Text
        match &msg.content[0] {
            ContentPart::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("msg.content[0] should be Text"),
        }

        // ToolCall
        match &msg.content[1] {
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => {
                assert_eq!(name, "f");
                assert_eq!(args, &json!({"x": 1}));
                assert_eq!(call_id.as_deref(), Some("c1"));
            }
            _ => panic!("msg.content[1] should be ToolCall"),
        }

        // ToolOutput (name should be empty string per mapping)
        match &msg.content[2] {
            ContentPart::ToolOutput {
                name,
                output,
                call_id,
                remote_id,
            } => {
                assert_eq!(name, "");
                assert_eq!(output, &json!({"ok":true}));
                assert_eq!(call_id.as_deref(), Some("c1"));
                assert!(remote_id.is_none());
            }
            _ => panic!("msg.content[2] should be ToolOutput"),
        }

        // Reasoning
        match &msg.content[3] {
            ContentPart::Reasoning { text } => assert_eq!(text, "think"),
            _ => panic!("msg.content[3] should be Reasoning"),
        }

        // Test deserializing function_call_output with string output
        let s = r#"{"type":"function_call_output","call_id":"cid","output":"{\"ok\":true}"}"#;
        let v = serde_json::from_str::<MessageItem>(s).unwrap();
        let m = message_from(vec![v.clone()]).0.unwrap();
        assert_eq!(
            m,
            Message {
                role: "assistant".into(),
                content: vec![ContentPart::ToolOutput {
                    name: "".into(),
                    output: json!({"ok":true}),
                    call_id: Some("cid".into()),
                    remote_id: None,
                }],
                ..Default::default()
            }
        );

        // Test deserializing function_call_output with Items (array) output
        let s = r#"{"type":"function_call_output","call_id":"cid","output":[{"type":"input_text","text":"result"}]}"#;
        let v = serde_json::from_str::<MessageItem>(s).unwrap();
        if let MessageItem::FunctionCallOutput {
            output, call_id, ..
        } = &v
        {
            assert_eq!(
                output,
                &FunctionCallOutput::Items(vec![FunctionCallOutputItem::Text {
                    text: "result".into()
                }])
            );
            assert_eq!(call_id, "cid");
        } else {
            panic!("should be FunctionCallOutput");
        }
        let m = message_from(vec![v]).0.unwrap();
        assert_eq!(
            m,
            Message {
                role: "assistant".into(),
                content: vec![ContentPart::ToolOutput {
                    name: "".into(),
                    output: json!([{"type": "input_text", "text": "result"}]),
                    call_id: Some("cid".into()),
                    remote_id: None,
                }],
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_message_from_only_refusal_returns_none() {
        let items = vec![MessageItem::Message {
            role: "assistant".into(),
            content: vec![ContentItem::Refusal {
                refusal: "bad".into(),
            }],
            status: None,
            id: None,
        }];

        let (msg_opt, failed_reason) = message_from(items);
        assert!(msg_opt.is_none());
        assert_eq!(failed_reason.as_deref(), Some("bad"));
    }

    #[test]
    fn test_messageitem_serde_type_tags() {
        // Message
        let item = MessageItem::Message {
            role: "user".into(),
            content: vec![ContentItem::Text { text: "hi".into() }],
            status: None,
            id: None,
        };
        let s = serde_json::to_string(&item).unwrap();
        assert!(s.contains(r#""type":"message""#));
        // content items should use their own tags (e.g., input_text)
        assert!(s.contains(r#""type":"input_text""#));

        // FunctionCall
        let item = MessageItem::FunctionCall {
            name: "f".into(),
            arguments: r#"{"a":1}"#.into(),
            call_id: "cid".into(),
            id: None,
            namespace: None,
            status: None,
        };
        let s = serde_json::to_string(&item).unwrap();
        assert!(s.contains(r#""type":"function_call""#));

        // FunctionCallOutput with String output
        let item = MessageItem::FunctionCallOutput {
            output: FunctionCallOutput::String(r#"{"ok":true}"#.into()),
            call_id: "cid".into(),
            id: None,
            status: None,
        };
        let s = serde_json::to_string(&item).unwrap();
        assert!(s.contains(r#""type":"function_call_output""#));
        assert!(s.contains(r#""output":"{\"ok\":true}""#));

        // FunctionCallOutput with Items output
        let item = MessageItem::FunctionCallOutput {
            output: FunctionCallOutput::Items(vec![FunctionCallOutputItem::Text {
                text: "result".into(),
            }]),
            call_id: "cid".into(),
            id: None,
            status: None,
        };
        let s = serde_json::to_string(&item).unwrap();
        assert!(s.contains(r#""type":"function_call_output""#));
        assert!(s.contains(r#""output":[{"type":"input_text","text":"result"}]"#));

        // Reasoning
        let item = MessageItem::Reasoning {
            id: "".into(),
            summary: vec![ReasoningSummary::SummaryText { text: "t".into() }],
            content: None,
            encrypted_content: None,
            status: None,
        };
        let s = serde_json::to_string(&item).unwrap();
        assert!(s.contains(r#""type":"reasoning""#));
    }
}
