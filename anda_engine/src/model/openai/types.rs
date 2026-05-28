use anda_core::{
    AgentOutput, BoxError, ContentPart, Json, Message, Usage as ModelUsage,
    inline_data_from_data_url, normalize_strict_schema, part_to_data_url,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};

use crate::unix_ms;

fn default_true() -> bool {
    true
}

fn default_image_detail() -> String {
    "auto".to_string()
}

fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContentInput {
    Text(String),
    Items(Vec<ContentItem>),
}

impl From<MessageContentInput> for Vec<ContentItem> {
    fn from(value: MessageContentInput) -> Self {
        match value {
            MessageContentInput::Text(text) => vec![ContentItem::Text { text }],
            MessageContentInput::Items(items) => items,
        }
    }
}

fn deserialize_content_items<'de, D>(deserializer: D) -> Result<Vec<ContentItem>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(MessageContentInput::deserialize(deserializer)?.into())
}

fn is_response_output_role(role: &str) -> bool {
    role.eq_ignore_ascii_case("assistant")
}

fn normalize_message_content(role: &str, content: Vec<ContentItem>) -> Vec<ContentItem> {
    let is_output = is_response_output_role(role);
    content
        .into_iter()
        .filter_map(|item| match (is_output, item) {
            (true, ContentItem::Text { text }) | (true, ContentItem::OutputText { text }) => {
                Some(ContentItem::OutputText { text })
            }
            (true, ContentItem::Refusal { refusal }) => Some(ContentItem::Refusal { refusal }),
            (true, _) => None,
            (false, ContentItem::Text { text }) | (false, ContentItem::OutputText { text }) => {
                Some(ContentItem::Text { text })
            }
            (false, ContentItem::Refusal { refusal }) => Some(ContentItem::Text { text: refusal }),
            (false, item) => Some(item),
        })
        .collect()
}

fn push_message_item(rt: &mut Vec<MessageItem>, role: &str, content: &mut Vec<ContentItem>) {
    if content.is_empty() {
        return;
    }

    let content = normalize_message_content(role, std::mem::take(content));
    if !content.is_empty() {
        rt.push(MessageItem::Message {
            role: role.to_string(),
            content,
            status: None,
            id: None,
            phase: None,
        });
    }
}

fn normalize_message_item(item: MessageItem) -> Option<MessageItem> {
    match item {
        MessageItem::Message {
            role,
            content,
            status,
            id,
            phase,
        } => {
            let content = normalize_message_content(&role, content);
            if content.is_empty() {
                None
            } else {
                Some(MessageItem::Message {
                    role,
                    content,
                    status,
                    id,
                    phase,
                })
            }
        }
        MessageItem::Reasoning {
            encrypted_content: None,
            ..
        } => None,
        MessageItem::ItemReference { .. } => None,
        item => Some(item),
    }
}

pub(crate) fn raw_history_into(value: Json) -> Vec<MessageItem> {
    let item = serde_json::from_value::<MessageItem>(value.clone())
        .unwrap_or_else(|_| MessageItem::Any(value.clone()));
    match item {
        MessageItem::Any(value) => {
            if let Ok(msg) = serde_json::from_value::<Message>(value.clone())
                && !msg.role.is_empty()
            {
                message_into(msg)
                    .into_iter()
                    .filter_map(normalize_message_item)
                    .collect()
            } else {
                vec![MessageItem::Any(value)]
            }
        }
        item => normalize_message_item(item).into_iter().collect(),
    }
}

/// The completion request type for OpenAI's Response API: <https://developers.openai.com/api/reference/resources/responses/methods/create>
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
    /// Controls which (if any) tool is called by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// The tools you want to use.
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
    /// Context management configuration for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagement>>,
    /// Conversation state to attach this response to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ResponseConversationParam>,
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
    /// Maximum number of built-in tool calls processed in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u64>,
    /// Prompt template reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,
    /// Cache key used to improve prompt cache hit rates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    /// Stable safety identifier for the application end user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Streaming behavior options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Number of token logprobs to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ContextManagement {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_threshold: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ResponseConversationParam {
    Id(String),
    Object { id: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ResponseConversation {
    pub id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ResponsePrompt {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<std::collections::HashMap<String, PromptVariable>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum PromptVariable {
    String(String),
    Content(ContentItem),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

/// The standard response format from OpenAI's Responses API.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// The ID of a completion response.
    pub id: String,
    /// UNIX epoch in seconds.
    pub created_at: u64,
    #[serde(default, deserialize_with = "null_default")]
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
    #[serde(default, deserialize_with = "null_default")]
    pub usage: ResponsesUsage,
    /// The model output (messages, etc will go here)
    pub output: Vec<Json>,
    #[serde(skip, default)]
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
                cached_tokens: self.usage.input_tokens_details.cached_tokens,
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
                output.thoughts = msg.thoughts();
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

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated { response: CompletionResponse },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress { response: CompletionResponse },
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: CompletionResponse },
    #[serde(rename = "response.failed")]
    ResponseFailed { response: CompletionResponse },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response: CompletionResponse },
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        output_index: usize,
        item: MessageItem,
    },
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        output_index: usize,
        item: MessageItem,
    },
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ContentItem,
    },
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ContentItem,
    },
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(untagged)]
    Any(Json),
}

impl<'de> Deserialize<'de> for StreamEvent {
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
                        "response.created"
                            | "response.in_progress"
                            | "response.completed"
                            | "response.failed"
                            | "response.incomplete"
                            | "response.output_item.added"
                            | "response.output_item.done"
                            | "response.content_part.added"
                            | "response.content_part.done"
                            | "response.output_text.delta"
                            | "response.output_text.done"
                    )
                ) =>
            {
                #[allow(clippy::enum_variant_names)]
                #[derive(Deserialize)]
                #[serde(tag = "type")]
                enum Helper {
                    #[serde(rename = "response.created")]
                    ResponseCreated { response: CompletionResponse },
                    #[serde(rename = "response.in_progress")]
                    ResponseInProgress { response: CompletionResponse },
                    #[serde(rename = "response.completed")]
                    ResponseCompleted { response: CompletionResponse },
                    #[serde(rename = "response.failed")]
                    ResponseFailed { response: CompletionResponse },
                    #[serde(rename = "response.incomplete")]
                    ResponseIncomplete { response: CompletionResponse },
                    #[serde(rename = "response.output_item.added")]
                    ResponseOutputItemAdded {
                        output_index: usize,
                        item: MessageItem,
                    },
                    #[serde(rename = "response.output_item.done")]
                    ResponseOutputItemDone {
                        output_index: usize,
                        item: MessageItem,
                    },
                    #[serde(rename = "response.content_part.added")]
                    ResponseContentPartAdded {
                        item_id: String,
                        output_index: usize,
                        content_index: usize,
                        part: ContentItem,
                    },
                    #[serde(rename = "response.content_part.done")]
                    ResponseContentPartDone {
                        item_id: String,
                        output_index: usize,
                        content_index: usize,
                        part: ContentItem,
                    },
                    #[serde(rename = "response.output_text.delta")]
                    ResponseOutputTextDelta {
                        item_id: String,
                        output_index: usize,
                        content_index: usize,
                        delta: String,
                    },
                    #[serde(rename = "response.output_text.done")]
                    ResponseOutputTextDone {
                        item_id: String,
                        output_index: usize,
                        content_index: usize,
                        text: String,
                    },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(Helper::ResponseCreated { response }) => {
                        Ok(StreamEvent::ResponseCreated { response })
                    }
                    Ok(Helper::ResponseInProgress { response }) => {
                        Ok(StreamEvent::ResponseInProgress { response })
                    }
                    Ok(Helper::ResponseCompleted { response }) => {
                        Ok(StreamEvent::ResponseCompleted { response })
                    }
                    Ok(Helper::ResponseFailed { response }) => {
                        Ok(StreamEvent::ResponseFailed { response })
                    }
                    Ok(Helper::ResponseIncomplete { response }) => {
                        Ok(StreamEvent::ResponseIncomplete { response })
                    }
                    Ok(Helper::ResponseOutputItemAdded { output_index, item }) => {
                        Ok(StreamEvent::ResponseOutputItemAdded { output_index, item })
                    }
                    Ok(Helper::ResponseOutputItemDone { output_index, item }) => {
                        Ok(StreamEvent::ResponseOutputItemDone { output_index, item })
                    }
                    Ok(Helper::ResponseContentPartAdded {
                        item_id,
                        output_index,
                        content_index,
                        part,
                    }) => Ok(StreamEvent::ResponseContentPartAdded {
                        item_id,
                        output_index,
                        content_index,
                        part,
                    }),
                    Ok(Helper::ResponseContentPartDone {
                        item_id,
                        output_index,
                        content_index,
                        part,
                    }) => Ok(StreamEvent::ResponseContentPartDone {
                        item_id,
                        output_index,
                        content_index,
                        part,
                    }),
                    Ok(Helper::ResponseOutputTextDelta {
                        item_id,
                        output_index,
                        content_index,
                        delta,
                    }) => Ok(StreamEvent::ResponseOutputTextDelta {
                        item_id,
                        output_index,
                        content_index,
                        delta,
                    }),
                    Ok(Helper::ResponseOutputTextDone {
                        item_id,
                        output_index,
                        content_index,
                        text,
                    }) => Ok(StreamEvent::ResponseOutputTextDone {
                        item_id,
                        output_index,
                        content_index,
                        text,
                    }),
                    Err(_) => Ok(StreamEvent::Any(value)),
                }
            }
            _ => Ok(StreamEvent::Any(value)),
        }
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
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
    FileSearchCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        queries: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        results: Option<Vec<FileSearchResult>>,
    },
    ComputerCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pending_safety_checks: Vec<ComputerSafetyCheck>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        action: Option<Json>,
        #[serde(skip_serializing_if = "Option::is_none")]
        actions: Option<Vec<Json>>,
    },
    ComputerCallOutput {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: ComputerScreenshot,
        #[serde(skip_serializing_if = "Option::is_none")]
        acknowledged_safety_checks: Option<Vec<ComputerSafetyCheck>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    WebSearchCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        action: WebSearchAction,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    ToolSearchCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        arguments: Json,
        #[serde(skip_serializing_if = "Option::is_none")]
        execution: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    ToolSearchOutput {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<ToolDefinition>,
        #[serde(skip_serializing_if = "Option::is_none")]
        execution: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
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
    Compaction {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        encrypted_content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    ImageGenerationCall {
        id: String,
        result: String,
        status: String,
    },
    CodeInterpreterCall {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        container_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        outputs: Option<Vec<CodeInterpreterOutput>>,
        status: String,
    },
    LocalShellCall {
        id: String,
        action: LocalShellAction,
        call_id: String,
        status: String,
    },
    LocalShellCallOutput {
        id: String,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    ShellCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        action: ShellAction,
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        environment: Option<ShellEnvironment>,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    ShellCallOutput {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_output_length: Option<u64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        output: Vec<ShellCallOutputContent>,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    ApplyPatchCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        operation: ApplyPatchOperation,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    ApplyPatchCallOutput {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    McpCall {
        id: String,
        arguments: String,
        name: String,
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        approval_request_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    McpListTools {
        id: String,
        server_label: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<McpTool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    McpApprovalRequest {
        id: String,
        arguments: String,
        name: String,
        server_label: String,
    },
    McpApprovalResponse {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        approval_request_id: String,
        approve: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    CustomToolCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        input: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
    CustomToolCallOutput {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: FunctionCallOutput,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    CompactionTrigger,
    ItemReference {
        id: String,
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
                    Some(
                        "message"
                            | "file_search_call"
                            | "computer_call"
                            | "computer_call_output"
                            | "web_search_call"
                            | "function_call"
                            | "function_call_output"
                            | "tool_search_call"
                            | "tool_search_output"
                            | "reasoning"
                            | "compaction"
                            | "image_generation_call"
                            | "code_interpreter_call"
                            | "local_shell_call"
                            | "local_shell_call_output"
                            | "shell_call"
                            | "shell_call_output"
                            | "apply_patch_call"
                            | "apply_patch_call_output"
                            | "mcp_call"
                            | "mcp_list_tools"
                            | "mcp_approval_request"
                            | "mcp_approval_response"
                            | "custom_tool_call"
                            | "custom_tool_call_output"
                            | "compaction_trigger"
                            | "item_reference"
                    )
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type", rename_all = "snake_case")]
                enum Helper {
                    Message {
                        role: String,
                        #[serde(deserialize_with = "deserialize_content_items")]
                        content: Vec<ContentItem>,
                        status: Option<String>,
                        id: Option<String>,
                        phase: Option<MessagePhase>,
                    },
                    FileSearchCall {
                        id: Option<String>,
                        #[serde(default)]
                        queries: Vec<String>,
                        status: Option<String>,
                        results: Option<Vec<FileSearchResult>>,
                    },
                    ComputerCall {
                        id: Option<String>,
                        call_id: String,
                        #[serde(default)]
                        pending_safety_checks: Vec<ComputerSafetyCheck>,
                        status: Option<String>,
                        action: Option<Json>,
                        actions: Option<Vec<Json>>,
                    },
                    ComputerCallOutput {
                        id: Option<String>,
                        call_id: String,
                        output: ComputerScreenshot,
                        acknowledged_safety_checks: Option<Vec<ComputerSafetyCheck>>,
                        status: Option<String>,
                        created_by: Option<String>,
                    },
                    WebSearchCall {
                        id: Option<String>,
                        action: WebSearchAction,
                        status: Option<String>,
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
                        created_by: Option<String>,
                    },
                    ToolSearchCall {
                        id: Option<String>,
                        call_id: Option<String>,
                        arguments: Json,
                        execution: Option<String>,
                        status: Option<String>,
                        created_by: Option<String>,
                    },
                    ToolSearchOutput {
                        id: Option<String>,
                        call_id: Option<String>,
                        #[serde(default)]
                        tools: Vec<ToolDefinition>,
                        execution: Option<String>,
                        status: Option<String>,
                        created_by: Option<String>,
                    },
                    Reasoning {
                        id: String,
                        summary: Vec<ReasoningSummary>,
                        content: Option<Vec<ReasoningContent>>,
                        encrypted_content: Option<String>,
                        status: Option<String>,
                    },
                    Compaction {
                        id: Option<String>,
                        encrypted_content: String,
                        created_by: Option<String>,
                    },
                    ImageGenerationCall {
                        id: String,
                        result: String,
                        status: String,
                    },
                    CodeInterpreterCall {
                        id: String,
                        code: Option<String>,
                        container_id: String,
                        outputs: Option<Vec<CodeInterpreterOutput>>,
                        status: String,
                    },
                    LocalShellCall {
                        id: String,
                        action: LocalShellAction,
                        call_id: String,
                        status: String,
                    },
                    LocalShellCallOutput {
                        id: String,
                        output: String,
                        status: Option<String>,
                    },
                    ShellCall {
                        id: Option<String>,
                        action: ShellAction,
                        call_id: String,
                        environment: Option<ShellEnvironment>,
                        status: String,
                        created_by: Option<String>,
                    },
                    ShellCallOutput {
                        id: Option<String>,
                        call_id: String,
                        max_output_length: Option<u64>,
                        #[serde(default)]
                        output: Vec<ShellCallOutputContent>,
                        status: String,
                        created_by: Option<String>,
                    },
                    ApplyPatchCall {
                        id: Option<String>,
                        call_id: String,
                        operation: ApplyPatchOperation,
                        status: String,
                        created_by: Option<String>,
                    },
                    ApplyPatchCallOutput {
                        id: Option<String>,
                        call_id: String,
                        status: String,
                        output: Option<String>,
                        created_by: Option<String>,
                    },
                    McpCall {
                        id: String,
                        arguments: String,
                        name: String,
                        server_label: String,
                        approval_request_id: Option<String>,
                        error: Option<String>,
                        output: Option<String>,
                        status: Option<String>,
                    },
                    McpListTools {
                        id: String,
                        server_label: String,
                        #[serde(default)]
                        tools: Vec<McpTool>,
                        error: Option<String>,
                    },
                    McpApprovalRequest {
                        id: String,
                        arguments: String,
                        name: String,
                        server_label: String,
                    },
                    McpApprovalResponse {
                        id: Option<String>,
                        approval_request_id: String,
                        approve: bool,
                        reason: Option<String>,
                    },
                    CustomToolCall {
                        id: Option<String>,
                        call_id: String,
                        input: String,
                        name: String,
                        namespace: Option<String>,
                    },
                    CustomToolCallOutput {
                        id: Option<String>,
                        call_id: String,
                        output: FunctionCallOutput,
                        status: Option<String>,
                        created_by: Option<String>,
                    },
                    CompactionTrigger,
                    ItemReference {
                        id: String,
                    },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(h) => Ok(match h {
                        Helper::Message {
                            role,
                            content,
                            status,
                            id,
                            phase,
                        } => MessageItem::Message {
                            role,
                            content,
                            status,
                            id,
                            phase,
                        },
                        Helper::FileSearchCall {
                            id,
                            queries,
                            status,
                            results,
                        } => MessageItem::FileSearchCall {
                            id,
                            queries,
                            status,
                            results,
                        },
                        Helper::ComputerCall {
                            id,
                            call_id,
                            pending_safety_checks,
                            status,
                            action,
                            actions,
                        } => MessageItem::ComputerCall {
                            id,
                            call_id,
                            pending_safety_checks,
                            status,
                            action,
                            actions,
                        },
                        Helper::ComputerCallOutput {
                            id,
                            call_id,
                            output,
                            acknowledged_safety_checks,
                            status,
                            created_by,
                        } => MessageItem::ComputerCallOutput {
                            id,
                            call_id,
                            output,
                            acknowledged_safety_checks,
                            status,
                            created_by,
                        },
                        Helper::WebSearchCall { id, action, status } => {
                            MessageItem::WebSearchCall { id, action, status }
                        }
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
                            created_by,
                        } => MessageItem::FunctionCallOutput {
                            id,
                            call_id,
                            output,
                            status,
                            created_by,
                        },
                        Helper::ToolSearchCall {
                            id,
                            call_id,
                            arguments,
                            execution,
                            status,
                            created_by,
                        } => MessageItem::ToolSearchCall {
                            id,
                            call_id,
                            arguments,
                            execution,
                            status,
                            created_by,
                        },
                        Helper::ToolSearchOutput {
                            id,
                            call_id,
                            tools,
                            execution,
                            status,
                            created_by,
                        } => MessageItem::ToolSearchOutput {
                            id,
                            call_id,
                            tools,
                            execution,
                            status,
                            created_by,
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
                        Helper::Compaction {
                            id,
                            encrypted_content,
                            created_by,
                        } => MessageItem::Compaction {
                            id,
                            encrypted_content,
                            created_by,
                        },
                        Helper::ImageGenerationCall { id, result, status } => {
                            MessageItem::ImageGenerationCall { id, result, status }
                        }
                        Helper::CodeInterpreterCall {
                            id,
                            code,
                            container_id,
                            outputs,
                            status,
                        } => MessageItem::CodeInterpreterCall {
                            id,
                            code,
                            container_id,
                            outputs,
                            status,
                        },
                        Helper::LocalShellCall {
                            id,
                            action,
                            call_id,
                            status,
                        } => MessageItem::LocalShellCall {
                            id,
                            action,
                            call_id,
                            status,
                        },
                        Helper::LocalShellCallOutput { id, output, status } => {
                            MessageItem::LocalShellCallOutput { id, output, status }
                        }
                        Helper::ShellCall {
                            id,
                            action,
                            call_id,
                            environment,
                            status,
                            created_by,
                        } => MessageItem::ShellCall {
                            id,
                            action,
                            call_id,
                            environment,
                            status,
                            created_by,
                        },
                        Helper::ShellCallOutput {
                            id,
                            call_id,
                            max_output_length,
                            output,
                            status,
                            created_by,
                        } => MessageItem::ShellCallOutput {
                            id,
                            call_id,
                            max_output_length,
                            output,
                            status,
                            created_by,
                        },
                        Helper::ApplyPatchCall {
                            id,
                            call_id,
                            operation,
                            status,
                            created_by,
                        } => MessageItem::ApplyPatchCall {
                            id,
                            call_id,
                            operation,
                            status,
                            created_by,
                        },
                        Helper::ApplyPatchCallOutput {
                            id,
                            call_id,
                            status,
                            output,
                            created_by,
                        } => MessageItem::ApplyPatchCallOutput {
                            id,
                            call_id,
                            status,
                            output,
                            created_by,
                        },
                        Helper::McpCall {
                            id,
                            arguments,
                            name,
                            server_label,
                            approval_request_id,
                            error,
                            output,
                            status,
                        } => MessageItem::McpCall {
                            id,
                            arguments,
                            name,
                            server_label,
                            approval_request_id,
                            error,
                            output,
                            status,
                        },
                        Helper::McpListTools {
                            id,
                            server_label,
                            tools,
                            error,
                        } => MessageItem::McpListTools {
                            id,
                            server_label,
                            tools,
                            error,
                        },
                        Helper::McpApprovalRequest {
                            id,
                            arguments,
                            name,
                            server_label,
                        } => MessageItem::McpApprovalRequest {
                            id,
                            arguments,
                            name,
                            server_label,
                        },
                        Helper::McpApprovalResponse {
                            id,
                            approval_request_id,
                            approve,
                            reason,
                        } => MessageItem::McpApprovalResponse {
                            id,
                            approval_request_id,
                            approve,
                            reason,
                        },
                        Helper::CustomToolCall {
                            id,
                            call_id,
                            input,
                            name,
                            namespace,
                        } => MessageItem::CustomToolCall {
                            id,
                            call_id,
                            input,
                            name,
                            namespace,
                        },
                        Helper::CustomToolCallOutput {
                            id,
                            call_id,
                            output,
                            status,
                            created_by,
                        } => MessageItem::CustomToolCallOutput {
                            id,
                            call_id,
                            output,
                            status,
                            created_by,
                        },
                        Helper::CompactionTrigger => MessageItem::CompactionTrigger,
                        Helper::ItemReference { id } => MessageItem::ItemReference { id },
                    }),
                    Err(_) => Ok(MessageItem::Any(value)),
                }
            }
            _ => Ok(MessageItem::Any(value)),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessagePhase {
    Commentary,
    FinalAnswer,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct FileSearchResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Map<String, Json>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct ComputerSafetyCheck {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct ComputerScreenshot {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct WebSearchAction {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    pub fields: Map<String, Json>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct CodeInterpreterOutput {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    pub fields: Map<String, Json>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct LocalShellAction {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub env: Map<String, Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ShellAction {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ShellEnvironment {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    pub fields: Map<String, Json>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ShellCallOutputContent {
    pub outcome: Json,
    #[serde(default)]
    pub stderr: String,
    #[serde(default)]
    pub stdout: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct ApplyPatchOperation {
    #[serde(rename = "type")]
    pub r#type: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct McpTool {
    #[serde(default)]
    pub input_schema: Json,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
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
                push_message_item(&mut rt, &msg.role, &mut content);
                rt.push(MessageItem::Reasoning {
                    id: "".to_string(),
                    summary: vec![ReasoningSummary::SummaryText { text }],
                    content: None,
                    encrypted_content: None,
                    status: None,
                });
            }
            ContentPart::FileData {
                file_uri,
                mime_type,
            } if (file_uri.starts_with("data:") || file_uri.starts_with("https://"))
                && mime_type
                    .as_ref()
                    .map(|v| v.starts_with("image/"))
                    .unwrap_or(false) =>
            {
                content.push(ContentItem::Image {
                    detail: "auto".to_string(),
                    image_url: file_uri,
                    file_id: None,
                });
            }
            ContentPart::FileData { file_uri, .. }
                if file_uri.starts_with("data:") || file_uri.starts_with("https://") =>
            {
                content.push(ContentItem::File {
                    file_data: None,
                    file_url: Some(file_uri),
                    file_id: None,
                    filename: None,
                });
            }
            ContentPart::InlineData { mime_type, data } if mime_type.starts_with("image/") => {
                content.push(ContentItem::Image {
                    detail: "auto".to_string(),
                    image_url: part_to_data_url(&data, Some(&mime_type)),
                    file_id: None,
                })
            }
            ContentPart::InlineData { mime_type, data } if mime_type.starts_with("audio/") => {
                content.push(ContentItem::Audio {
                    input_audio: InputAudio {
                        format: mime_type.trim_start_matches("audio/").to_string(),
                        data: part_to_data_url(&data, Some(&mime_type)),
                    },
                })
            }
            ContentPart::InlineData { mime_type, data } => content.push(ContentItem::File {
                file_data: Some(part_to_data_url(&data, Some(&mime_type))),
                file_url: None,
                file_id: None,
                filename: None,
            }),
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => {
                push_message_item(&mut rt, &msg.role, &mut content);
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
                push_message_item(&mut rt, &msg.role, &mut content);
                rt.push(MessageItem::FunctionCallOutput {
                    output: FunctionCallOutput::String(
                        serde_json::to_string(&output).unwrap_or_default(),
                    ),
                    call_id: call_id.unwrap_or_default(),
                    id: None,
                    status: None,
                    created_by: None,
                });
            }
            v => content.push(ContentItem::Text {
                text: serde_json::to_string(&v).unwrap_or_default(),
            }),
        }
    }

    push_message_item(&mut rt, &msg.role, &mut content);

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
                            if let Some((data, mime_type)) = inline_data_from_data_url(&image_url) {
                                msg.content
                                    .push(ContentPart::InlineData { mime_type, data });
                            } else {
                                msg.content.push(ContentPart::FileData {
                                    file_uri: image_url,
                                    mime_type: None,
                                });
                            }
                        }
                        ContentItem::Audio { input_audio } => {
                            if let Some((data, mime_type)) =
                                inline_data_from_data_url(&input_audio.data)
                            {
                                msg.content
                                    .push(ContentPart::InlineData { mime_type, data });
                            } else {
                                msg.content.push(ContentPart::FileData {
                                    file_uri: input_audio.data,
                                    mime_type: None,
                                });
                            }
                        }
                        ContentItem::File {
                            file_data,
                            file_url,
                            file_id,
                            filename,
                        } => {
                            if let Some(data_url) = file_data.as_deref()
                                && let Some((data, mime_type)) = inline_data_from_data_url(data_url)
                            {
                                msg.content
                                    .push(ContentPart::InlineData { mime_type, data });
                            } else if let Some(data_url) = file_url.as_deref()
                                && let Some((data, mime_type)) = inline_data_from_data_url(data_url)
                            {
                                msg.content
                                    .push(ContentPart::InlineData { mime_type, data });
                            } else if let Some(file_uri) = file_url.clone() {
                                msg.content.push(ContentPart::FileData {
                                    file_uri,
                                    mime_type: None,
                                });
                            } else {
                                msg.content.push(ContentPart::Any(json!(ContentItem::File {
                                    file_data,
                                    file_url,
                                    file_id,
                                    filename,
                                })));
                            }
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
                    is_error: None,
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
            other => msg.content.push(ContentPart::Any(
                serde_json::to_value(other).unwrap_or(Json::Null),
            )),
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
        #[serde(default)]
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
                        #[serde(default = "default_image_detail")]
                        detail: String,
                        #[serde(default)]
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
        #[serde(default = "default_image_detail")]
        detail: String, // One of high, low, or auto. Defaults to auto
        #[serde(default)]
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

/// A tool definition accepted by OpenAI's Responses API.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function {
        name: String,
        #[serde(default)]
        parameters: Json,
        #[serde(default = "default_true")]
        strict: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
    },
    FileSearch {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        vector_store_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<Json>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<FileSearchRankingOptions>,
    },
    Computer,
    ComputerUsePreview {
        display_height: u32,
        display_width: u32,
        environment: String,
    },
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<WebSearchFilters>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    #[serde(rename = "web_search_2025_08_26")]
    WebSearch20250826 {
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<WebSearchFilters>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_tools: Option<McpAllowedTools>,
        #[serde(skip_serializing_if = "Option::is_none")]
        authorization: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        connector_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        headers: Option<Map<String, Json>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        require_approval: Option<Json>,
        #[serde(skip_serializing_if = "Option::is_none")]
        server_description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        server_url: Option<String>,
    },
    CodeInterpreter {
        container: CodeInterpreterContainer,
    },
    ImageGeneration {
        #[serde(skip_serializing_if = "Option::is_none")]
        action: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        background: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_fidelity: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_image_mask: Option<ImageInputMask>,
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        moderation: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_compression: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        partial_images: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quality: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<String>,
    },
    LocalShell,
    Shell {
        #[serde(skip_serializing_if = "Option::is_none")]
        environment: Option<Json>,
    },
    Custom {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<CustomToolInputFormat>,
    },
    Namespace {
        description: String,
        name: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<NamespaceToolDefinition>,
    },
    ToolSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        execution: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Json>,
    },
    WebSearchPreview {
        #[serde(skip_serializing_if = "Option::is_none")]
        search_content_types: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311 {
        #[serde(skip_serializing_if = "Option::is_none")]
        search_content_types: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    ApplyPatch,
    #[serde(untagged)]
    Any(Json),
}

impl<'de> Deserialize<'de> for ToolDefinition {
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
                        "function"
                            | "file_search"
                            | "computer"
                            | "computer_use_preview"
                            | "web_search"
                            | "web_search_2025_08_26"
                            | "mcp"
                            | "code_interpreter"
                            | "image_generation"
                            | "local_shell"
                            | "shell"
                            | "custom"
                            | "namespace"
                            | "tool_search"
                            | "web_search_preview"
                            | "web_search_preview_2025_03_11"
                            | "apply_patch"
                    )
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type", rename_all = "snake_case")]
                enum Helper {
                    Function {
                        name: String,
                        #[serde(default)]
                        parameters: Json,
                        #[serde(default = "default_true")]
                        strict: bool,
                        description: Option<String>,
                        defer_loading: Option<bool>,
                    },
                    FileSearch {
                        #[serde(default)]
                        vector_store_ids: Vec<String>,
                        filters: Option<Json>,
                        max_num_results: Option<u32>,
                        ranking_options: Option<FileSearchRankingOptions>,
                    },
                    Computer,
                    ComputerUsePreview {
                        display_height: u32,
                        display_width: u32,
                        environment: String,
                    },
                    WebSearch {
                        filters: Option<WebSearchFilters>,
                        search_context_size: Option<String>,
                        user_location: Option<UserLocation>,
                    },
                    #[serde(rename = "web_search_2025_08_26")]
                    WebSearch20250826 {
                        filters: Option<WebSearchFilters>,
                        search_context_size: Option<String>,
                        user_location: Option<UserLocation>,
                    },
                    Mcp {
                        server_label: String,
                        allowed_tools: Option<McpAllowedTools>,
                        authorization: Option<String>,
                        connector_id: Option<String>,
                        defer_loading: Option<bool>,
                        headers: Option<Map<String, Json>>,
                        require_approval: Option<Json>,
                        server_description: Option<String>,
                        server_url: Option<String>,
                    },
                    CodeInterpreter {
                        container: CodeInterpreterContainer,
                    },
                    ImageGeneration {
                        action: Option<String>,
                        background: Option<String>,
                        input_fidelity: Option<String>,
                        input_image_mask: Option<ImageInputMask>,
                        model: Option<String>,
                        moderation: Option<String>,
                        output_compression: Option<u8>,
                        output_format: Option<String>,
                        partial_images: Option<u8>,
                        quality: Option<String>,
                        size: Option<String>,
                    },
                    LocalShell,
                    Shell {
                        environment: Option<Json>,
                    },
                    Custom {
                        name: String,
                        defer_loading: Option<bool>,
                        description: Option<String>,
                        format: Option<CustomToolInputFormat>,
                    },
                    Namespace {
                        description: String,
                        name: String,
                        #[serde(default)]
                        tools: Vec<NamespaceToolDefinition>,
                    },
                    ToolSearch {
                        description: Option<String>,
                        execution: Option<String>,
                        parameters: Option<Json>,
                    },
                    WebSearchPreview {
                        search_content_types: Option<Vec<String>>,
                        search_context_size: Option<String>,
                        user_location: Option<UserLocation>,
                    },
                    #[serde(rename = "web_search_preview_2025_03_11")]
                    WebSearchPreview20250311 {
                        search_content_types: Option<Vec<String>>,
                        search_context_size: Option<String>,
                        user_location: Option<UserLocation>,
                    },
                    ApplyPatch,
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(h) => Ok(match h {
                        Helper::Function {
                            name,
                            parameters,
                            strict,
                            description,
                            defer_loading,
                        } => ToolDefinition::Function {
                            name,
                            parameters,
                            strict,
                            description,
                            defer_loading,
                        },
                        Helper::FileSearch {
                            vector_store_ids,
                            filters,
                            max_num_results,
                            ranking_options,
                        } => ToolDefinition::FileSearch {
                            vector_store_ids,
                            filters,
                            max_num_results,
                            ranking_options,
                        },
                        Helper::Computer => ToolDefinition::Computer,
                        Helper::ComputerUsePreview {
                            display_height,
                            display_width,
                            environment,
                        } => ToolDefinition::ComputerUsePreview {
                            display_height,
                            display_width,
                            environment,
                        },
                        Helper::WebSearch {
                            filters,
                            search_context_size,
                            user_location,
                        } => ToolDefinition::WebSearch {
                            filters,
                            search_context_size,
                            user_location,
                        },
                        Helper::WebSearch20250826 {
                            filters,
                            search_context_size,
                            user_location,
                        } => ToolDefinition::WebSearch20250826 {
                            filters,
                            search_context_size,
                            user_location,
                        },
                        Helper::Mcp {
                            server_label,
                            allowed_tools,
                            authorization,
                            connector_id,
                            defer_loading,
                            headers,
                            require_approval,
                            server_description,
                            server_url,
                        } => ToolDefinition::Mcp {
                            server_label,
                            allowed_tools,
                            authorization,
                            connector_id,
                            defer_loading,
                            headers,
                            require_approval,
                            server_description,
                            server_url,
                        },
                        Helper::CodeInterpreter { container } => {
                            ToolDefinition::CodeInterpreter { container }
                        }
                        Helper::ImageGeneration {
                            action,
                            background,
                            input_fidelity,
                            input_image_mask,
                            model,
                            moderation,
                            output_compression,
                            output_format,
                            partial_images,
                            quality,
                            size,
                        } => ToolDefinition::ImageGeneration {
                            action,
                            background,
                            input_fidelity,
                            input_image_mask,
                            model,
                            moderation,
                            output_compression,
                            output_format,
                            partial_images,
                            quality,
                            size,
                        },
                        Helper::LocalShell => ToolDefinition::LocalShell,
                        Helper::Shell { environment } => ToolDefinition::Shell { environment },
                        Helper::Custom {
                            name,
                            defer_loading,
                            description,
                            format,
                        } => ToolDefinition::Custom {
                            name,
                            defer_loading,
                            description,
                            format,
                        },
                        Helper::Namespace {
                            description,
                            name,
                            tools,
                        } => ToolDefinition::Namespace {
                            description,
                            name,
                            tools,
                        },
                        Helper::ToolSearch {
                            description,
                            execution,
                            parameters,
                        } => ToolDefinition::ToolSearch {
                            description,
                            execution,
                            parameters,
                        },
                        Helper::WebSearchPreview {
                            search_content_types,
                            search_context_size,
                            user_location,
                        } => ToolDefinition::WebSearchPreview {
                            search_content_types,
                            search_context_size,
                            user_location,
                        },
                        Helper::WebSearchPreview20250311 {
                            search_content_types,
                            search_context_size,
                            user_location,
                        } => ToolDefinition::WebSearchPreview20250311 {
                            search_content_types,
                            search_context_size,
                            user_location,
                        },
                        Helper::ApplyPatch => ToolDefinition::ApplyPatch,
                    }),
                    Err(_) => Ok(ToolDefinition::Any(value)),
                }
            }
            _ => Ok(ToolDefinition::Any(value)),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct FileSearchRankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_search: Option<FileSearchHybridSearch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct FileSearchHybridSearch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_weight: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct WebSearchFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct UserLocation {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum McpAllowedTools {
    Names(Vec<String>),
    Filter(McpToolFilter),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct McpToolFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum CodeInterpreterContainer {
    Id(String),
    Auto(CodeInterpreterContainerAuto),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct CodeInterpreterContainerAuto {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_policy: Option<Json>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct ImageInputMask {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CustomToolInputFormat {
    Text,
    Grammar { definition: String, syntax: String },
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NamespaceToolDefinition {
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Json>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
    },
    Custom {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<CustomToolInputFormat>,
    },
    #[serde(untagged)]
    Any(Json),
}

impl<'de> Deserialize<'de> for NamespaceToolDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Json::deserialize(deserializer)?;
        match &value {
            Json::Object(map)
                if matches!(
                    map.get("type").and_then(|t| t.as_str()),
                    Some("function" | "custom")
                ) =>
            {
                #[derive(Deserialize)]
                #[serde(tag = "type", rename_all = "snake_case")]
                enum Helper {
                    Function {
                        name: String,
                        parameters: Option<Json>,
                        strict: Option<bool>,
                        description: Option<String>,
                        defer_loading: Option<bool>,
                    },
                    Custom {
                        name: String,
                        defer_loading: Option<bool>,
                        description: Option<String>,
                        format: Option<CustomToolInputFormat>,
                    },
                }

                match serde_json::from_value::<Helper>(value.clone()) {
                    Ok(Helper::Function {
                        name,
                        parameters,
                        strict,
                        description,
                        defer_loading,
                    }) => Ok(NamespaceToolDefinition::Function {
                        name,
                        parameters,
                        strict,
                        description,
                        defer_loading,
                    }),
                    Ok(Helper::Custom {
                        name,
                        defer_loading,
                        description,
                        format,
                    }) => Ok(NamespaceToolDefinition::Custom {
                        name,
                        defer_loading,
                        description,
                        format,
                    }),
                    Err(_) => Ok(NamespaceToolDefinition::Any(value)),
                }
            }
            _ => Ok(NamespaceToolDefinition::Any(value)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Object(ToolChoiceObject),
    Any(Json),
}

impl ToolChoice {
    pub fn none() -> Self {
        Self::Mode(ToolChoiceMode::None)
    }

    pub fn auto() -> Self {
        Self::Mode(ToolChoiceMode::Auto)
    }

    pub fn required() -> Self {
        Self::Mode(ToolChoiceMode::Required)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceObject {
    AllowedTools {
        mode: ToolChoiceAllowedMode,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<Json>,
    },
    Function {
        name: String,
    },
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Custom {
        name: String,
    },
    ApplyPatch,
    Shell,
    FileSearch,
    WebSearchPreview,
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311,
    Computer,
    ComputerUsePreview,
    ComputerUse,
    ImageGeneration,
    CodeInterpreter,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceAllowedMode {
    Auto,
    Required,
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
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<TextVerbosity>,
}

impl TextConfig {
    pub(crate) fn structured_output<S>(name: S, schema: Json) -> Self
    where
        S: Into<String>,
    {
        Self {
            format: Some(TextFormat::JsonSchema(StructuredOutputsInput {
                name: name.into(),
                schema: normalize_strict_schema(schema),
                description: None,
                strict: Some(true),
            })),
            verbosity: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    Low,
    Medium,
    High,
}

/// The text format (contained by [`TextConfig`]).
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    JsonSchema(StructuredOutputsInput),
    JsonObject,
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
    /// Optional description of the response format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Enable strict output. If you are using your AI agent in a data pipeline or another scenario that requires the data to be absolutely fixed to a given schema, it is recommended to set this to true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Add reasoning to a [`CompletionRequest`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reasoning {
    /// How much effort you want the model to put into thinking/reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Deprecated reasoning summary option retained for API compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<ReasoningSummaryLevel>,
    /// How much effort you want the model to put into writing the reasoning summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryLevel>,
}

impl Default for Reasoning {
    fn default() -> Self {
        Self {
            effort: Some(ReasoningEffort::Medium),
            generate_summary: None,
            summary: Some(ReasoningSummaryLevel::Auto),
        }
    }
}

/// The billing service tier that will be used. On auto by default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

/// The amount of reasoning effort that will be used by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
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
                    is_error: None,
                    call_id: Some("c1".into()),
                    remote_id: None,
                },
                ContentPart::FileData {
                    file_uri: "https://a/b".into(),
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
                    assert_eq!(file_url.as_deref(), Some("https://a/b"));
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
    fn test_message_into_serializes_non_remote_file_data_as_text() {
        let file_part = ContentPart::FileData {
            file_uri: "file:///tmp/report.pdf".into(),
            mime_type: Some("application/pdf".into()),
        };
        let items = message_into(Message {
            role: "user".into(),
            content: vec![file_part.clone()],
            ..Default::default()
        });

        assert_eq!(items.len(), 1);
        match &items[0] {
            MessageItem::Message { role, content, .. } => {
                assert_eq!(role, "user");
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ContentItem::Text { text } => {
                        let parsed: Json = serde_json::from_str(text).unwrap();
                        assert_eq!(parsed, serde_json::to_value(&file_part).unwrap());
                    }
                    _ => panic!("content[0] should be Text"),
                }
            }
            _ => panic!("items[0] should be Message"),
        }
    }

    #[test]
    fn message_into_uses_output_text_for_assistant_history() {
        let items = message_into(Message {
            role: "assistant".into(),
            content: vec![ContentPart::Text {
                text: "done".into(),
            }],
            ..Default::default()
        });

        assert_eq!(items.len(), 1);
        match &items[0] {
            MessageItem::Message { role, content, .. } => {
                assert_eq!(role, "assistant");
                assert_eq!(
                    content,
                    &vec![ContentItem::OutputText {
                        text: "done".into()
                    }]
                );
            }
            _ => panic!("items[0] should be Message"),
        }

        let value = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(value["content"][0]["type"], "output_text");
    }

    #[test]
    fn raw_history_normalizes_core_and_legacy_response_messages() {
        let items = raw_history_into(json!({
            "role": "assistant",
            "content": [{"type": "Text", "text": "old core"}]
        }));
        assert!(matches!(
            items.first(),
            Some(MessageItem::Message { content, .. })
                if matches!(content.first(), Some(ContentItem::OutputText { text }) if text == "old core")
        ));

        let items = raw_history_into(json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "legacy responses"}]
        }));
        assert!(matches!(
            items.first(),
            Some(MessageItem::Message { content, .. })
                if matches!(content.first(), Some(ContentItem::OutputText { text }) if text == "legacy responses")
        ));

        let items = raw_history_into(json!({
            "type": "reasoning",
            "id": "rs_022c4f1e6736eb98016a0f16d145b48191950615b7ca9fdfd3",
            "summary": []
        }));
        assert!(items.is_empty());

        let items = raw_history_into(json!({
            "type": "item_reference",
            "id": "rs_022c4f1e6736eb98016a0f16d145b48191950615b7ca9fdfd3"
        }));
        assert!(items.is_empty());
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
                phase: None,
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
                created_by: None,
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
                is_error,
                call_id,
                remote_id,
            } => {
                assert_eq!(name, "");
                assert_eq!(output, &json!({"ok":true}));
                assert!(is_error.is_none());
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
                    is_error: None,
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
                    is_error: None,
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
            phase: None,
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
            phase: None,
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
            created_by: None,
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
            created_by: None,
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

    #[test]
    fn deserializes_response_with_function_tool_and_nullable_fields() {
        let mut response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1741294021,
            "status": "completed",
            "completed_at": 1741294022,
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.4",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_current_weather",
                "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}",
                "status": "completed"
            }],
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {
                "effort": null,
                "summary": null
            },
            "store": true,
            "text": {"format": {"type": "text"}},
            "tools": [{
                "type": "function",
                "description": "Get current weather",
                "name": "get_current_weather",
                "parameters": {"type": "object"},
                "strict": true
            }],
            "usage": {
                "input_tokens": 291,
                "output_tokens": 23,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 314
            },
            "metadata": {}
        }))
        .unwrap();

        response.parse_output();
        assert_eq!(response.usage.total_tokens, 314);
        assert_eq!(response.parsed_output.len(), 1);

        assert!(matches!(
            response.parsed_output.first(),
            Some(MessageItem::FunctionCall { name, call_id, .. })
                if name == "get_current_weather" && call_id == "call_1"
        ));
        assert!(matches!(
            response.tools.first(),
            Some(ToolDefinition::Function {
                name,
                description: Some(description),
                strict: true,
                ..
            }) if name == "get_current_weather" && description == "Get current weather"
        ));
        assert!(matches!(
            response.additional_parameters.reasoning,
            Some(Reasoning {
                effort: None,
                summary: None,
                ..
            })
        ));
    }

    #[test]
    fn serializes_request_with_core_tool_types_and_tool_choice() {
        let request = CompletionRequest {
            model: "gpt-5.4".into(),
            input: vec![MessageItem::Message {
                role: "user".into(),
                content: vec![ContentItem::Text {
                    text: "search docs".into(),
                }],
                status: None,
                id: None,
                phase: None,
            }],
            tool_choice: Some(ToolChoice::Object(ToolChoiceObject::Function {
                name: "lookup".into(),
            })),
            tools: vec![
                ToolDefinition::FileSearch {
                    vector_store_ids: vec!["vs_1".into()],
                    filters: Some(json!({"type": "eq", "key": "kind", "value": "doc"})),
                    max_num_results: Some(5),
                    ranking_options: None,
                },
                ToolDefinition::WebSearchPreview {
                    search_content_types: Some(vec!["text".into()]),
                    search_context_size: Some("low".into()),
                    user_location: None,
                },
                ToolDefinition::ToolSearch {
                    description: Some("load deferred tools".into()),
                    execution: Some("client".into()),
                    parameters: Some(json!({"type": "object"})),
                },
                ToolDefinition::ApplyPatch,
            ],
            ..Default::default()
        };

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["tool_choice"]["type"], "function");
        assert_eq!(value["tool_choice"]["name"], "lookup");
        assert_eq!(value["tools"][0]["type"], "file_search");
        assert_eq!(value["tools"][1]["type"], "web_search_preview");
        assert_eq!(value["tools"][2]["type"], "tool_search");
        assert_eq!(value["tools"][3]["type"], "apply_patch");

        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "lookup"}]
        }))
        .unwrap();
        assert!(matches!(
            choice,
            ToolChoice::Object(ToolChoiceObject::AllowedTools {
                mode: ToolChoiceAllowedMode::Required,
                ..
            })
        ));
    }

    #[test]
    fn deserializes_core_output_item_variants() {
        let items: Vec<MessageItem> = serde_json::from_value(json!([
            {
                "type": "web_search_call",
                "id": "ws_1",
                "action": {"type": "search", "queries": ["rust serde"]},
                "status": "completed"
            },
            {
                "type": "code_interpreter_call",
                "id": "ci_1",
                "code": null,
                "container_id": "ctr_1",
                "outputs": [{"type": "logs", "logs": "ok"}],
                "status": "completed"
            },
            {
                "type": "shell_call_output",
                "id": "shout_1",
                "call_id": "call_shell",
                "max_output_length": 2000,
                "output": [{
                    "outcome": {"type": "exit", "exit_code": 0},
                    "stdout": "done",
                    "stderr": ""
                }],
                "status": "completed"
            },
            {
                "type": "mcp_call",
                "id": "mcp_1",
                "arguments": "{\"q\":\"anda\"}",
                "name": "search",
                "server_label": "deepwiki",
                "status": "completed",
                "output": "result"
            },
            {
                "type": "custom_tool_call_output",
                "id": "cto_1",
                "call_id": "call_custom",
                "output": "ok",
                "status": "completed"
            },
            {
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "ciphertext"
            }
        ]))
        .unwrap();

        assert!(matches!(items[0], MessageItem::WebSearchCall { .. }));
        assert!(matches!(items[1], MessageItem::CodeInterpreterCall { .. }));
        assert!(matches!(items[2], MessageItem::ShellCallOutput { .. }));
        assert!(matches!(items[3], MessageItem::McpCall { .. }));
        assert!(matches!(items[4], MessageItem::CustomToolCallOutput { .. }));
        assert!(matches!(items[5], MessageItem::Compaction { .. }));

        let (message, failed_reason) = message_from(items);
        assert!(failed_reason.is_none());
        let message = message.expect("unmapped tool items should be preserved as Any content");
        assert_eq!(message.content.len(), 6);
        assert!(matches!(message.content.first(), Some(ContentPart::Any(_))));
    }

    #[test]
    fn deserializes_stream_events_with_nullable_response_usage() {
        let event: StreamEvent = serde_json::from_value(json!({
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 1741290958,
                "status": "in_progress",
                "error": null,
                "incomplete_details": null,
                "instructions": "You are helpful.",
                "model": "gpt-5.4",
                "output": [],
                "parallel_tool_calls": true,
                "reasoning": {"effort": null, "summary": null},
                "text": {"format": {"type": "text"}},
                "tools": [],
                "usage": null,
                "metadata": {}
            }
        }))
        .unwrap();

        match event {
            StreamEvent::ResponseCreated { response } => {
                assert_eq!(response.id, "resp_1");
                assert_eq!(response.usage.total_tokens, 0);
            }
            _ => panic!("expected response.created"),
        }

        let event: StreamEvent = serde_json::from_value(json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        }))
        .unwrap();
        assert!(matches!(
            event,
            StreamEvent::ResponseContentPartAdded {
                part: ContentItem::OutputText { .. },
                ..
            }
        ));

        let event: StreamEvent = serde_json::from_value(json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi"
        }))
        .unwrap();
        assert!(matches!(
            event,
            StreamEvent::ResponseOutputTextDelta { delta, .. } if delta == "Hi"
        ));
    }
}
