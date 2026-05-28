use anda_core::{
    AgentOutput, BoxError, ByteBufB64, ContentPart, FunctionDefinition, Message,
    Usage as ModelUsage,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{fmt, str::FromStr};

use crate::unix_ms;

fn is_zero(value: &u32) -> bool {
    *value == 0
}

// https://ai.google.dev/api/generate-content
// https://googleapis.github.io/js-genai/release_docs/index.html
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    /// Optional. Developer set system instruction(s). Currently, text only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contents: Vec<Value>, // Vec<Content>

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<SafetySetting>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Local SDK switch: use the streaming REST endpoint and aggregate chunks.
    #[serde(default, skip)]
    pub stream: bool,

    pub generation_config: GenerationConfig,
}

/// Response from the model supporting multiple candidate responses.
///
/// Safety ratings and content filtering are reported for both prompt in
/// GenerateContentResponse.prompt_feedback and for each candidate in
/// finishReason and in safetyRatings.
///
/// The API:
/// - Returns either all requested candidates or none of them
/// - Returns no candidates at all only if there was something wrong with the
///   prompt (check promptFeedback)
/// - Reports feedback on each candidate in finishReason and safetyRatings.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,

    #[serde(default)]
    pub usage_metadata: UsageMetadata,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_status: Option<ModelStatus>,
}

impl GenerateContentResponse {
    pub fn try_into(
        mut self,
        raw_history: Vec<Value>,
        chat_history: Vec<Message>,
    ) -> Result<AgentOutput, BoxError> {
        let timestamp = unix_ms();
        let mut output = AgentOutput {
            raw_history,
            chat_history,
            model: self.model_version.clone(),
            usage: ModelUsage {
                input_tokens: self.usage_metadata.prompt_token_count as u64,
                output_tokens: (self.usage_metadata.candidates_token_count
                    + self.usage_metadata.thoughts_token_count
                    + self.usage_metadata.tool_use_prompt_token_count)
                    as u64,
                cached_tokens: self.usage_metadata.cached_content_token_count as u64,
                requests: 1,
            },
            ..Default::default()
        };

        if let Some(feedback) = self.prompt_feedback {
            output.failed_reason = serde_json::to_string(&feedback).ok();
        } else {
            let candidate = self.candidates.pop().ok_or("No completion choice")?;

            if candidate.content.parts.is_empty() {
                output.failed_reason = serde_json::to_string(&candidate.finish_reason).ok();
            } else {
                output.raw_history.push(json!(&candidate.content));
                let mut msg: Message = candidate.content.into();
                msg.name = self.model_version;
                msg.timestamp = Some(timestamp);

                match candidate.finish_reason {
                    Some(FinishReason::Stop) => {
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
        }

        Ok(output)
    }

    pub fn maybe_failed(&self) -> bool {
        self.prompt_feedback.is_some()
            || !self.candidates.iter().any(|candidate| {
                matches!(candidate.finish_reason.as_ref(), Some(FinishReason::Stop))
            })
    }
}

/// A response candidate generated from the model.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// Generated content returned from the model.
    pub content: Content,
    /// The reason why the model stopped generating tokens. If empty, the model
    /// has not stopped generating tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// List of ratings for the safety of a response candidate. There is at most
    /// one rating per category.
    #[serde(alias = "satefyRatings", skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    /// Citation information for model-generated candidate.
    ///
    /// This field may be populated with recitation information for any text
    /// included in the content. These are passages that are "recited" from
    /// copyrighted material in the foundational LLM's training data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>,
    /// Token count for this candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u32>,
    /// Attribution information for sources that contributed to a grounded answer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grounding_attributions: Vec<GroundingAttribution>,
    /// Grounding metadata for the candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_metadata: Option<GroundingMetadata>,
    /// Average log probability score of the candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprobs: Option<f64>,
    /// Log-likelihood scores for the response tokens and top tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs_result: Option<LogprobsResult>,
    /// Metadata related to URL context retrieval tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_context_metadata: Option<UrlContextMetadata>,
    /// Index of the candidate in the list of response candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// Details the reason why the model stopped generating tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    // Optional. The producer of the content. Must be either 'user' or 'model'.
    // Useful to set for multi-turn conversations, otherwise can be left blank or unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,

    #[serde(default)]
    pub parts: Vec<Part>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// whether or not the part is a reasoning/thinking text or not
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    /// an opaque sig for the thought so it can be reused - is a base64 string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,

    #[serde(flatten)]
    pub data: PartKind,
}

impl From<ContentPart> for Part {
    fn from(value: ContentPart) -> Self {
        match value {
            ContentPart::Text { text } => Part {
                data: PartKind::Text(text),
                ..Default::default()
            },
            ContentPart::Reasoning { text } => Part {
                thought: Some(true),
                data: PartKind::Text(text),
                ..Default::default()
            },
            ContentPart::FileData {
                mime_type,
                file_uri,
            } if file_uri.starts_with("data:") || file_uri.starts_with("https://") => Part {
                data: PartKind::FileData {
                    file_uri,
                    mime_type,
                },
                ..Default::default()
            },
            ContentPart::InlineData { mime_type, data } => Part {
                data: PartKind::InlineData {
                    mime_type,
                    data: data.to_base64(),
                },
                ..Default::default()
            },
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => Part {
                data: PartKind::FunctionCall {
                    name,
                    args: Some(args),
                    id: call_id,
                },
                ..Default::default()
            },
            ContentPart::ToolOutput {
                name,
                output,
                is_error,
                call_id,
                ..
            } => Part {
                data: PartKind::FunctionResponse {
                    name,
                    response: if is_error.map_or(false, |b| b) {
                        FunctionResponseValue {
                            error: Some(output),
                            ..Default::default()
                        }
                    } else {
                        FunctionResponseValue {
                            output: Some(output),
                            ..Default::default()
                        }
                    },
                    id: call_id,
                    will_continue: None,
                    scheduling: None,
                    parts: None,
                },
                ..Default::default()
            },
            ContentPart::Any(json) => {
                serde_json::from_value(json.clone()).unwrap_or_else(|_| Part {
                    data: PartKind::Text(serde_json::to_string(&json).unwrap_or_default()),
                    ..Default::default()
                })
            }
            _ => Part {
                data: PartKind::Text(serde_json::to_string(&value).unwrap_or_default()),
                ..Default::default()
            },
        }
    }
}

impl From<Part> for ContentPart {
    fn from(mut value: Part) -> Self {
        match value.data {
            PartKind::Text(text) if value.thought == Some(true) => ContentPart::Reasoning { text },
            PartKind::Text(text) => ContentPart::Text { text },
            PartKind::FileData {
                file_uri,
                mime_type,
            } => ContentPart::FileData {
                file_uri,
                mime_type,
            },
            PartKind::InlineData { mime_type, data } => match ByteBufB64::from_str(data.as_str()) {
                Ok(data) => ContentPart::InlineData { mime_type, data },
                Err(_) => ContentPart::Any(json!({
                    "type": "InlineData",
                    "mimeType": mime_type,
                    "data": data,
                })),
            },
            PartKind::FunctionCall { name, args, id } => ContentPart::ToolCall {
                name,
                args: args.unwrap_or_default(),
                call_id: id,
            },
            PartKind::FunctionResponse {
                name, response, id, ..
            } => ContentPart::ToolOutput {
                name,
                output: response.to_output(),
                is_error: response.error.as_ref().map(|_| true),
                call_id: id,
                remote_id: None,
            },
            PartKind::Any(val) => ContentPart::Any(val),
            _ => {
                value.thought_signature = None;
                ContentPart::Any(json!(value.data))
            }
        }
    }
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum PartKind {
    CodeExecutionResult {
        outcome: String,
        output: String,
    },
    ExecutableCode {
        language: String,
        code: String,
    },
    FileData {
        file_uri: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    FunctionCall {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        args: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    FunctionResponse {
        name: String,
        response: FunctionResponseValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        will_continue: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scheduling: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parts: Option<Vec<Value>>,
    },
    InlineData {
        mime_type: String,
        data: String,
    },
    Text(String),
    #[serde(untagged)]
    Any(Value),
}

impl<'de> Deserialize<'de> for PartKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        // Text variant is just a string under "text" key (handled by flatten in Part)
        if let Value::String(s) = &value {
            return Ok(PartKind::Text(s.clone()));
        }
        if let Value::Object(map) = &value {
            // Detect variant by checking which key exists
            if map.contains_key("codeExecutionResult")
                || map.contains_key("executableCode")
                || map.contains_key("fileData")
                || map.contains_key("functionCall")
                || map.contains_key("functionResponse")
                || map.contains_key("inlineData")
                || map.contains_key("text")
            {
                #[derive(Deserialize)]
                #[serde(rename_all = "camelCase", rename_all_fields = "camelCase")]
                enum Helper {
                    CodeExecutionResult {
                        outcome: String,
                        output: String,
                    },
                    ExecutableCode {
                        language: String,
                        code: String,
                    },
                    FileData {
                        file_uri: String,
                        mime_type: Option<String>,
                    },
                    FunctionCall {
                        name: String,
                        args: Option<Value>,
                        id: Option<String>,
                    },
                    FunctionResponse {
                        name: String,
                        response: FunctionResponseValue,
                        id: Option<String>,
                        will_continue: Option<bool>,
                        scheduling: Option<String>,
                        parts: Option<Vec<Value>>,
                    },
                    InlineData {
                        mime_type: String,
                        data: String,
                    },
                    Text(String),
                }

                if let Ok(h) = serde_json::from_value::<Helper>(value.clone()) {
                    return Ok(match h {
                        Helper::CodeExecutionResult { outcome, output } => {
                            PartKind::CodeExecutionResult { outcome, output }
                        }
                        Helper::ExecutableCode { language, code } => {
                            PartKind::ExecutableCode { language, code }
                        }
                        Helper::FileData {
                            file_uri,
                            mime_type,
                        } => PartKind::FileData {
                            file_uri,
                            mime_type,
                        },
                        Helper::FunctionCall { name, args, id } => {
                            PartKind::FunctionCall { name, args, id }
                        }
                        Helper::FunctionResponse {
                            name,
                            response,
                            id,
                            will_continue,
                            scheduling,
                            parts,
                        } => PartKind::FunctionResponse {
                            name,
                            response,
                            id,
                            will_continue,
                            scheduling,
                            parts,
                        },
                        Helper::InlineData { mime_type, data } => {
                            PartKind::InlineData { mime_type, data }
                        }
                        Helper::Text(text) => PartKind::Text(text),
                    });
                }
            }
        }
        Ok(PartKind::Any(value))
    }
}

// The function response in JSON object format. Use "output" key to specify function output and "error" key to specify error details (if any)
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponseValue {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,

    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl FunctionResponseValue {
    pub fn to_output(&self) -> Value {
        if let Some(output) = &self.output {
            output.clone()
        } else if self.error.is_some() {
            json!(self)
        } else {
            Value::Object(self.extra.clone())
        }
    }
}

impl Default for PartKind {
    fn default() -> Self {
        Self::Text(String::default())
    }
}

impl From<Message> for Content {
    fn from(msg: Message) -> Self {
        Self {
            role: Some(Role::from(msg.role.as_str())),
            parts: msg.content.into_iter().map(|v| v.into()).collect(),
        }
    }
}

impl From<Content> for Message {
    fn from(content: Content) -> Self {
        Self {
            role: content.role.unwrap_or_default().to_string(),
            content: content.parts.into_iter().map(|v| v.into()).collect(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    User,
    Model,
}

impl From<&str> for Role {
    fn from(value: &str) -> Self {
        match value {
            "user" | "tool" => Role::User,
            _ => Role::Model,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Model => write!(f, "assistant"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum Tool {
    FunctionDeclaration {
        function_declarations: Vec<FunctionDeclaration>,
    },

    CodeExecution {
        code_execution: CodeExecution,
    },

    /// Enable Google Search grounding with (optional) dynamic retrieval settings.
    GoogleSearchRetrieval {
        google_search_retrieval: GoogleSearchRetrieval,
    },

    /// Enable Google Search tool.
    GoogleSearch {
        google_search: GoogleSearch,
    },

    /// Enable URL context tool.
    UrlContext {
        url_context: UrlContext,
    },
}

impl From<Vec<FunctionDefinition>> for Tool {
    fn from(tools: Vec<FunctionDefinition>) -> Self {
        Self::FunctionDeclaration {
            function_declarations: tools
                .into_iter()
                .map(|v| FunctionDeclaration {
                    name: v.name,
                    description: v.description,
                    parameters: Some(v.parameters),
                    response: None,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecution {}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GoogleSearch {}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UrlContext {}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GoogleSearchRetrieval {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_retrieval_config: Option<DynamicRetrievalConfig>,

    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRetrievalConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_threshold: Option<f64>,

    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            function_calling_config: FunctionCallingConfig {
                mode: FunctionCallingMode::Auto,
                allowed_function_names: None,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: FunctionCallingMode,
    /// A set of function names that, when provided, limits the functions the
    /// model will call.
    ///
    /// This should only be set when the Mode is ANY. Function names should match
    /// [FunctionDeclaration.name]. With mode set to ANY, model will predict a
    /// function call from the set of function names provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Defines the execution behavior for function calling by defining the execution
/// mode.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    /// Unspecified function calling mode. This value should not be used.
    #[default]
    ModeUnspecified,
    /// Default model behavior, model decides to predict either a function call
    /// or a natural language response.
    Auto,
    /// Model is constrained to always predicting a function call only. If
    /// "allowedFunctionNames" are set, the predicted function call will be
    /// limited to any one of "allowedFunctionNames", else the predicted
    /// function call will be any one of the provided "functionDeclarations".
    Any,
    /// Model will not predict any function call. Model behavior is same as when
    /// not passing any function declarations.
    None,
}

/// Gemini API Configuration options for model generation and outputs. Not all parameters are
/// configurable for every model. From [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop
    /// at the first appearance of a stop_sequence. The stop sequence will not be included as part of the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// MIME type of the generated candidate text. Supported MIME types are:
    ///     - text/plain:  (default) Text output
    ///     - application/json: JSON response in the response candidates.
    ///     - text/x.enum: ENUM as a string response in the response candidates.
    /// Refer to the docs for a list of all supported text MIME types
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,

    /// Output schema of the generated candidate text. Schemas must be a subset of the OpenAPI schema and can be
    /// objects, primitives or arrays. If set, a compatible responseMimeType must also  be set. Compatible MIME
    /// types: application/json: Schema for JSON response. Refer to the JSON text generation guide for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,

    /// Alternative JSON Schema format accepted by the Gemini API.
    #[serde(
        rename = "_responseJsonSchema",
        skip_serializing_if = "Option::is_none"
    )]
    pub response_json_schema_compat: Option<Value>,

    /// JSON Schema format accepted by the Gemini API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<Modality>>,

    /// Number of generated responses to return. Currently, this value can only be set to 1. If
    /// unset, this will default to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,

    /// The maximum number of tokens to include in a response candidate. Note: The default value varies by model, see
    /// the Model.output_token_limit attribute of the Model returned from the getModel function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,

    /// Controls the randomness of the output. Note: The default value varies by model, see the Model.temperature
    /// attribute of the Model returned from the getModel function. Values can range from [0.0, 2.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// The maximum cumulative probability of tokens to consider when sampling. The model uses combined Top-k and
    /// Top-p (nucleus) sampling. Tokens are sorted based on their assigned probabilities so that only the most
    /// likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while
    /// Nucleus sampling limits the number of tokens based on the cumulative probability. Note: The default value
    /// varies by Model and is specified by theModel.top_p attribute returned from the getModel function. An empty
    /// topK attribute indicates that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// The maximum number of tokens to consider when sampling. Gemini models use Top-p (nucleus) sampling or a
    /// combination of Top-k and nucleus sampling. Top-k sampling considers the set of topK most probable tokens.
    /// Models running with nucleus sampling don't allow topK setting. Note: The default value varies by Model and is
    /// specified by theModel.top_p attribute returned from the getModel function. An empty topK attribute indicates
    /// that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Seed used in decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,

    /// Presence penalty applied to the next token's logprobs if the token has already been seen in the response.
    /// This penalty is binary on/off and not dependent on the number of times the token is used (after the first).
    /// Use frequencyPenalty for a penalty that increases with each use. A positive penalty will discourage the use
    /// of tokens that have already been used in the response, increasing the vocabulary. A negative penalty will
    /// encourage the use of tokens that have already been used in the response, decreasing the vocabulary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been
    /// seen in the response so far. A positive penalty will discourage the use of tokens that have already been
    /// used, proportional to the number of times the token has been used: The more a token is used, the more
    /// difficult it is for the  model to use that token again increasing the vocabulary of responses. Caution: A
    /// negative penalty will encourage the model to reuse tokens proportional to the number of times the token has
    /// been used. Small negative values will reduce the vocabulary of a response. Larger negative values will cause
    /// the model to  repeating a common token until it hits the maxOutputTokens limit: "...the the the the the...".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// If true, export the logprobs results in response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_logprobs: Option<bool>,

    /// Only valid if responseLogprobs=True. This sets the number of top logprobs to return at each decoding step in
    /// [Candidate.logprobs_result].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_enhanced_civic_answers: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<SpeechConfig>,

    /// Configuration for thinking/reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<ImageConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<MediaResolution>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: Some(1.0),
            max_output_tokens: Some(65535),
            stop_sequences: None,
            response_mime_type: None,
            response_schema: None,
            response_json_schema_compat: None,
            response_json_schema: None,
            response_modalities: None,
            candidate_count: None,
            top_p: None,
            top_k: None,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_logprobs: None,
            logprobs: None,
            enable_enhanced_civic_answers: None,
            speech_config: None,
            thinking_config: None,
            image_config: None,
            media_resolution: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_stage: Option<ModelStage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub retirement_time: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelStage {
    #[default]
    ModelStageUnspecified,
    UnstableExperimental,
    Experimental,
    Preview,
    Stable,
    Legacy,
    Deprecated,
    Retired,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Modality {
    #[default]
    ModalityUnspecified,
    Text,
    Image,
    Audio,
    Video,
    Document,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingAttribution {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<AttributionSourceId>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AttributionSourceId {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_passage: Option<GroundingPassageId>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_retriever_chunk: Option<SemanticRetrieverChunk>,

    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingPassageId {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passage_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_index: Option<u32>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticRetrieverChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grounding_chunks: Vec<GroundingChunk>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grounding_supports: Vec<GroundingSupport>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub web_search_queries: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_search_queries: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_entry_point: Option<SearchEntryPoint>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_metadata: Option<RetrievalMetadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_maps_widget_context_token: Option<String>,

    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_blob: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web: Option<Web>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<Image>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieved_context: Option<RetrievedContext>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub maps: Option<Maps>,

    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Web {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Image {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RetrievedContext {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub custom_metadata: Vec<CustomMetadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_search_store: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CustomMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub string_value: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub string_list_value: Option<StringList>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric_value: Option<f64>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StringList {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Maps {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub place_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub place_answer_sources: Option<PlaceAnswerSources>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PlaceAnswerSources {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub review_snippets: Vec<ReviewSnippet>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReviewSnippet {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_maps_uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grounding_chunk_indices: Vec<u32>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub confidence_scores: Vec<f64>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rendered_parts: Vec<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<Segment>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_index: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_dynamic_retrieval_score: Option<f64>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LogprobsResult {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_candidates: Vec<TopCandidates>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chosen_candidates: Vec<LogprobCandidate>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_probability_sum: Option<f64>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TopCandidates {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<LogprobCandidate>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LogprobCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_id: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_probability: Option<f64>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UrlContextMetadata {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub url_metadata: Vec<UrlMetadata>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UrlMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieved_url: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_retrieval_status: Option<UrlRetrievalStatus>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UrlRetrievalStatus {
    #[default]
    UrlRetrievalStatusUnspecified,
    UrlRetrievalStatusSuccess,
    UrlRetrievalStatusError,
    UrlRetrievalStatusPaywall,
    UrlRetrievalStatusUnsafe,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModalityTokenCount {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modality: Option<Modality>,

    #[serde(default, skip_serializing_if = "is_zero")]
    pub token_count: u32,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_config: Option<VoiceConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_speaker_voice_config: Option<MultiSpeakerVoiceConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prebuilt_voice_config: Option<PrebuiltVoiceConfig>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrebuiltVoiceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_name: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MultiSpeakerVoiceConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speaker_voice_configs: Vec<SpeakerVoiceConfig>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerVoiceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_config: Option<VoiceConfig>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MediaResolution {
    #[default]
    MediaResolutionUnspecified,
    MediaResolutionLow,
    MediaResolutionMedium,
    MediaResolutionHigh,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    #[default]
    Unspecified,
    Standard,
    Flex,
    Priority,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    #[default]
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
    Off,
}

/// A set of the feedback metadata the prompt specified in [GenerateContentRequest.contents](GenerateContentRequest).
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    /// Optional. If set, the prompt was blocked and no candidates are returned. Rephrase the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<BlockReason>,
    /// Ratings for safety of the prompt. There is at most one rating per category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

/// Reason why a prompt was blocked by the model
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockReason {
    /// Default value. This value is unused.
    #[default]
    BlockReasonUnspecified,
    /// Prompt was blocked due to safety reasons. Inspect safetyRatings to understand which safety category blocked it.
    Safety,
    /// Prompt was blocked due to unknown reasons.
    Other,
    /// Prompt was blocked due to the terms which are included from the terminology blocklist.
    Blocklist,
    /// Prompt was blocked due to prohibited content.
    ProhibitedContent,
    /// Candidates blocked due to unsafe image generation content.
    ImageSafety,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u32,

    #[serde(default)]
    pub total_token_count: u32,

    #[serde(default)]
    pub candidates_token_count: u32,

    #[serde(default)]
    pub thoughts_token_count: u32,

    #[serde(default)]
    pub tool_use_prompt_token_count: u32,

    #[serde(default)]
    pub cached_content_token_count: u32,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_tokens_details: Vec<ModalityTokenCount>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_tokens_details: Vec<ModalityTokenCount>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates_tokens_details: Vec<ModalityTokenCount>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_use_prompt_tokens_details: Vec<ModalityTokenCount>,
}

/// Config for thinking features.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Indicates whether to include thoughts in the response. If true, thoughts
    /// are returned only when available.
    pub include_thoughts: bool,
    /// The number of thoughts tokens that the model should generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    /// Controls the maximum depth of the model's internal reasoning process before it produces a response. If not specified, the default is HIGH. Recommended for Gemini 3 or later models. Use with earlier models results in an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ThinkingLevel {
    #[default]
    ThinkingLevelUnspecified,
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    #[serde(default)]
    pub citation_sources: Vec<CitationSource>,
}

/// CitationSource
///
/// A citation to a source for a portion of a specific response.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    /// Start of segment of the response that is attributed to this source.
    /// Index indicates the start of the segment, measured in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u32>,
    /// End of the attributed segment, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<u32>,
    /// URI that is attributed as a source for a portion of the text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// License for the GitHub project that is attributed as a source for
    /// segment. License info is required for code citations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

/// Safety rating for a piece of content.
///
/// The safety rating contains the category of harm and the harm probability
/// level in that category for a piece of content. Content is classified for
/// safety across a number of harm categories and the probability of the harm
/// classification is included here.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// The category for this rating.
    pub category: HarmCategory,
    /// The probability of harm for this content.
    pub probability: HarmProbability,
    /// Was this content blocked because of this rating?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>,
}

pub type SatisfyRating = SafetyRating;

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmProbability {
    /// Default value. This value is unused.
    #[default]
    HarmProbabilityUnspecified,
    /// Content has a negligible chance of being unsafe.
    Negligible,
    /// Content has a low chance of being unsafe.
    Low,
    /// Content has a medium chance of being unsafe.
    Medium,
    /// Content has a high chance of being unsafe.
    High,
}

// HarmCategory
//
// The category of a rating.
//
// These categories cover various kinds of harms that developers may wish to
// adjust.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    #[default]
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexual,
    HarmCategorySexually,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
    HarmCategoryCivicIntegrity,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    /// Default value. This value is unused.
    #[default]
    FinishReasonUnspecified,
    /// Natural stop point of the model or provided stop sequence.
    Stop,
    /// The maximum number of tokens as specified in the request was reached.
    MaxTokens,
    /// The response candidate content was flagged for safety reasons.
    Safety,
    /// The response candidate content was flagged for recitation reasons.
    Recitation,
    /// The response candidate content was flagged for using an unsupported
    /// language.
    Language,
    /// Unknown reason.
    Other,
    /// Token generation stopped because the content contains forbidden terms.
    Blocklist,
    /// Token generation stopped for potentially containing prohibited content.
    ProhibitedContent,
    /// Token generation stopped because the content potentially contains
    /// Sensitive Personally Identifiable Information (SPII).
    Spii,
    /// The function call generated by the model is invalid.
    MalformedFunctionCall,
    /// Token generation stopped because generated images contain safety
    /// violations.
    ImageSafety,
    /// Image generation stopped because generated images have prohibited content.
    ImageProhibitedContent,
    /// Image generation stopped because of another issue.
    ImageOther,
    /// The model was expected to generate an image, but none was generated.
    NoImage,
    /// Image generation stopped due to recitation.
    ImageRecitation,
    /// Model generated a tool call but no tools were enabled in the request.
    UnexpectedToolCall,
    /// Model called too many tools consecutively.
    TooManyToolCalls,
    /// Request has at least one thought signature missing.
    MissingThoughtSignature,
    /// Finished due to malformed response.
    MalformedResponse,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_content_part() {
        // Test Text variant
        let text_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::Text("Hello world".to_string()),
        };
        let json_value = serde_json::to_value(&text_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "text": "Hello world"
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, text_part);

        // Test Text with thought metadata
        let thought_text_part = Part {
            thought: Some(true),
            thought_signature: Some("base64signature".to_string()),
            data: PartKind::Text("This is a thought".to_string()),
        };
        let json_value = serde_json::to_value(&thought_text_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "thought": true,
                "thoughtSignature": "base64signature",
                "text": "This is a thought"
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, thought_text_part);

        // Test FunctionCall variant
        let function_call_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::FunctionCall {
                name: "get_weather".to_string(),
                args: Some(json!({"location": "Shanghai"})),
                id: Some("call_123".to_string()),
            },
        };
        let json_value = serde_json::to_value(&function_call_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Shanghai"},
                    "id": "call_123"
                }
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, function_call_part);

        // Test FunctionResponse variant
        let function_response_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::FunctionResponse {
                name: "get_weather".to_string(),
                response: FunctionResponseValue {
                    output: Some(json!({"temperature": "25°C", "condition": "sunny"})),
                    ..Default::default()
                },
                id: Some("call_123".to_string()),
                will_continue: Some(false),
                scheduling: None,
                parts: None,
            },
        };
        let json_value = serde_json::to_value(&function_response_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"output":{"temperature": "25°C", "condition": "sunny"}},
                    "id": "call_123",
                    "willContinue": false
                }
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, function_response_part);

        // Test InlineData variant
        let inline_data_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::InlineData {
                mime_type: "image/jpeg".to_string(),
                data: "base64encodedimagedata".to_string(),
            },
        };
        let json_value = serde_json::to_value(&inline_data_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": "base64encodedimagedata"
                }
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, inline_data_part);

        // Test FileData variant
        let file_data_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::FileData {
                file_uri: "gs://my-bucket/image.jpg".to_string(),
                mime_type: Some("image/jpeg".to_string()),
            },
        };
        let json_value = serde_json::to_value(&file_data_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "fileData": {
                    "fileUri": "gs://my-bucket/image.jpg",
                    "mimeType": "image/jpeg"
                }
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, file_data_part);

        // Test ExecutableCode variant
        let executable_code_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::ExecutableCode {
                language: "python".to_string(),
                code: "print('Hello, World!')".to_string(),
            },
        };
        let json_value = serde_json::to_value(&executable_code_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "executableCode": {
                    "language": "python",
                    "code": "print('Hello, World!')"
                }
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, executable_code_part);

        // Test CodeExecutionResult variant
        let code_result_part = Part {
            thought: None,
            thought_signature: None,
            data: PartKind::CodeExecutionResult {
                outcome: "success".to_string(),
                output: "Hello, World!".to_string(),
            },
        };
        let json_value = serde_json::to_value(&code_result_part).unwrap();
        assert_eq!(
            json_value,
            json!({
                "codeExecutionResult": {
                    "outcome": "success",
                    "output": "Hello, World!"
                }
            })
        );
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, code_result_part);

        // Test default ContentPart
        let default_part = Part::default();
        let json_value = serde_json::to_value(&default_part).unwrap();
        assert_eq!(json_value, json!({"text": ""}));
        let deserialized: Part = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized, default_part);

        // Test TryFrom<Value> for ContentPart with string
        // let string_value = json!("Simple text");
        // let content_part: Part = Part::try_from(string_value.clone()).unwrap();
        // assert_eq!(content_part.data, PartKind::Text("Simple text".to_string()));
        // assert_eq!(content_part.thought, None);
        // assert_eq!(content_part.thought_signature, None);

        // let val = into_parts(string_value.clone()).unwrap();
        // assert_eq!(val, vec![content_part.clone()]);

        // // Test TryFrom<Value> for ContentPart with complex object
        // let complex_value = json!({
        //     "thought": true,
        //     "thoughtSignature": "abc123",
        //     "functionCall": {
        //         "name": "test_function",
        //         "args": {"param": "value"}
        //     }
        // });
        // let content_part2: Part = Part::try_from(complex_value.clone()).unwrap();
        // assert_eq!(content_part2.thought, Some(true));
        // assert_eq!(content_part2.thought_signature, Some("abc123".to_string()));
        // if let PartKind::FunctionCall { name, args, id: _ } = &content_part2.data {
        //     assert_eq!(name, "test_function");
        //     assert_eq!(args, &Some(json!({"param": "value"})));
        // } else {
        //     panic!("Expected FunctionCall variant");
        // }

        // let val = into_parts(complex_value.clone()).unwrap();
        // assert_eq!(val, vec![content_part2.clone()]);

        // let val = into_parts(json!(vec![string_value, complex_value])).unwrap();
        // assert_eq!(val, vec![content_part, content_part2]);
    }

    #[test]
    fn test_non_remote_file_data_falls_back_to_text_part() {
        let file_part = ContentPart::FileData {
            file_uri: "file:///tmp/report.pdf".to_string(),
            mime_type: Some("application/pdf".to_string()),
        };

        let part: Part = file_part.clone().into();
        match part.data {
            PartKind::Text(text) => {
                let parsed: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(parsed, serde_json::to_value(&file_part).unwrap());
            }
            _ => panic!("non-remote FileData should fall back to Text"),
        }
    }

    #[test]
    fn test_generate_content_request_serde() {
        let mut req = GenerateContentRequest::default();
        req.contents.push(json!({
            "role": "user",
            "parts": [{"text": "Hello"}]
        }));
        req.safety_settings.push(SafetySetting {
            category: HarmCategory::HarmCategoryHarassment,
            threshold: HarmBlockThreshold::BlockOnlyHigh,
        });
        req.cached_content = Some("cachedContents/abc".to_string());
        req.service_tier = Some(ServiceTier::Priority);
        req.store = Some(false);
        req.generation_config.response_modalities = Some(vec![Modality::Text, Modality::Image]);
        req.generation_config.response_json_schema = Some(json!({"type": "object"}));
        req.generation_config.seed = Some(42);
        req.generation_config.enable_enhanced_civic_answers = Some(true);
        req.generation_config.speech_config = Some(SpeechConfig {
            voice_config: Some(VoiceConfig {
                prebuilt_voice_config: Some(PrebuiltVoiceConfig {
                    voice_name: Some("Kore".to_string()),
                }),
            }),
            language_code: Some("en-US".to_string()),
            ..Default::default()
        });
        req.generation_config.image_config = Some(ImageConfig {
            aspect_ratio: Some("16:9".to_string()),
            image_size: Some("1K".to_string()),
        });
        req.generation_config.media_resolution = Some(MediaResolution::MediaResolutionHigh);

        let json_value = serde_json::to_value(&req).unwrap();
        assert_eq!(json_value["cachedContent"], "cachedContents/abc");
        assert_eq!(json_value["serviceTier"], "priority");
        assert_eq!(json_value["store"], false);
        assert_eq!(
            json_value["safetySettings"],
            json!([{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH" }])
        );
        assert_eq!(
            json_value["generationConfig"]["responseModalities"],
            json!(["TEXT", "IMAGE"])
        );
        assert_eq!(
            json_value["generationConfig"]["responseJsonSchema"],
            json!({"type": "object"})
        );
        assert_eq!(json_value["generationConfig"]["seed"], 42);
        assert_eq!(
            json_value["generationConfig"]["enableEnhancedCivicAnswers"],
            true
        );
        assert_eq!(
            json_value["generationConfig"]["speechConfig"],
            json!({
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
                "languageCode": "en-US"
            })
        );
        assert_eq!(
            json_value["generationConfig"]["imageConfig"],
            json!({"aspectRatio": "16:9", "imageSize": "1K"})
        );
        assert_eq!(
            json_value["generationConfig"]["mediaResolution"],
            "MEDIA_RESOLUTION_HIGH"
        );

        let deserialized: GenerateContentRequest = serde_json::from_value(json_value).unwrap();
        assert_eq!(
            deserialized.safety_settings[0].category,
            HarmCategory::HarmCategoryHarassment
        );
        assert_eq!(deserialized.service_tier, Some(ServiceTier::Priority));
        assert_eq!(
            deserialized.generation_config.response_modalities,
            Some(vec![Modality::Text, Modality::Image])
        );
    }

    #[test]
    fn test_generate_content_response_core_metadata_serde() {
        let json_value = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Grounded answer"}]
                },
                "finishReason": "STOP",
                "safetyRatings": [{
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability": "LOW",
                    "blocked": false
                }],
                "citationMetadata": {
                    "citationSources": [{
                        "startIndex": 0,
                        "endIndex": 8,
                        "uri": "https://example.com",
                        "license": "MIT"
                    }]
                },
                "tokenCount": 12,
                "groundingAttributions": [{
                    "sourceId": {
                        "groundingPassage": {"passageId": "p1", "partIndex": 0}
                    },
                    "content": {"parts": [{"text": "source text"}]}
                }],
                "groundingMetadata": {
                    "groundingChunks": [{
                        "web": {"uri": "https://example.com", "title": "Example"}
                    }, {
                        "retrievedContext": {
                            "customMetadata": [{
                                "key": "kind",
                                "stringListValue": {"values": ["doc", "faq"]},
                                "numericValue": 0.8
                            }],
                            "uri": "fileSearchStores/1/documents/a",
                            "title": "Doc",
                            "text": "snippet",
                            "fileSearchStore": "fileSearchStores/1"
                        }
                    }, {
                        "maps": {
                            "uri": "https://maps.example/place",
                            "title": "Cafe",
                            "text": "Open now",
                            "placeId": "places/abc",
                            "placeAnswerSources": {
                                "reviewSnippets": [{
                                    "reviewId": "r1",
                                    "googleMapsUri": "https://maps.example/review",
                                    "title": "Helpful review"
                                }]
                            }
                        }
                    }],
                    "groundingSupports": [{
                        "groundingChunkIndices": [0],
                        "confidenceScores": [0.9],
                        "renderedParts": [0],
                        "segment": {
                            "partIndex": 0,
                            "startIndex": 0,
                            "endIndex": 8,
                            "text": "Grounded"
                        }
                    }],
                    "webSearchQueries": ["grounding query"],
                    "imageSearchQueries": ["image query"],
                    "searchEntryPoint": {
                        "renderedContent": "<div></div>",
                        "sdkBlob": "W10="
                    },
                    "retrievalMetadata": {"googleSearchDynamicRetrievalScore": 0.75},
                    "googleMapsWidgetContextToken": "maps-token"
                },
                "avgLogprobs": -0.12,
                "logprobsResult": {
                    "topCandidates": [{
                        "candidates": [{"token": "Hello", "tokenId": 1, "logProbability": -0.1}]
                    }],
                    "chosenCandidates": [{"token": "Hello", "tokenId": 1, "logProbability": -0.1}],
                    "logProbabilitySum": -0.1
                },
                "urlContextMetadata": {
                    "urlMetadata": [{
                        "retrievedUrl": "https://example.com",
                        "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                    }]
                },
                "index": 0,
                "finishMessage": "done"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "cachedContentTokenCount": 2,
                "candidatesTokenCount": 5,
                "toolUsePromptTokenCount": 1,
                "thoughtsTokenCount": 3,
                "totalTokenCount": 19,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
                "cacheTokensDetails": [{"modality": "DOCUMENT", "tokenCount": 2}],
                "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
                "toolUsePromptTokensDetails": [{"modality": "TEXT", "tokenCount": 1}]
            },
            "modelVersion": "gemini-2.5-flash",
            "responseId": "response-1",
            "modelStatus": {
                "modelStage": "STABLE",
                "retirementTime": "2026-12-31T00:00:00Z",
                "message": "ok"
            }
        });

        let response: GenerateContentResponse = serde_json::from_value(json_value.clone()).unwrap();
        assert_eq!(
            response.model_status.as_ref().unwrap().model_stage.as_ref(),
            Some(&ModelStage::Stable)
        );
        assert_eq!(
            response.usage_metadata.prompt_tokens_details[0]
                .modality
                .as_ref(),
            Some(&Modality::Text)
        );

        let candidate = &response.candidates[0];
        assert_eq!(candidate.finish_reason.as_ref(), Some(&FinishReason::Stop));
        assert_eq!(
            &candidate.safety_ratings.as_ref().unwrap()[0].probability,
            &HarmProbability::Low
        );
        assert_eq!(
            candidate.grounding_attributions[0]
                .source_id
                .as_ref()
                .unwrap()
                .grounding_passage
                .as_ref()
                .unwrap()
                .passage_id
                .as_deref(),
            Some("p1")
        );
        assert_eq!(
            candidate
                .grounding_metadata
                .as_ref()
                .unwrap()
                .grounding_chunks
                .len(),
            3
        );
        assert_eq!(
            candidate
                .logprobs_result
                .as_ref()
                .unwrap()
                .chosen_candidates[0]
                .token
                .as_deref(),
            Some("Hello")
        );
        assert_eq!(
            candidate
                .url_context_metadata
                .as_ref()
                .unwrap()
                .url_metadata[0]
                .url_retrieval_status
                .as_ref(),
            Some(&UrlRetrievalStatus::UrlRetrievalStatusSuccess)
        );
        assert_eq!(candidate.finish_message.as_deref(), Some("done"));

        assert_eq!(serde_json::to_value(response).unwrap(), json_value);
    }

    #[test]
    fn test_prompt_feedback_and_legacy_safety_alias() {
        let response: GenerateContentResponse = serde_json::from_value(json!({
            "promptFeedback": {
                "blockReason": "IMAGE_SAFETY",
                "safetyRatings": [{
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "HIGH",
                    "blocked": true
                }]
            },
            "usageMetadata": {}
        }))
        .unwrap();
        assert!(response.candidates.is_empty());
        let feedback = response.prompt_feedback.unwrap();
        assert_eq!(
            feedback.block_reason.as_ref(),
            Some(&BlockReason::ImageSafety)
        );
        assert_eq!(
            &feedback.safety_ratings.unwrap()[0].category,
            &HarmCategory::HarmCategoryDangerousContent
        );

        let candidate: Candidate = serde_json::from_value(json!({
            "content": {"parts": [{"text": "legacy"}]},
            "satefyRatings": [{
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "probability": "MEDIUM"
            }]
        }))
        .unwrap();
        assert_eq!(
            &candidate.safety_ratings.unwrap()[0].category,
            &HarmCategory::HarmCategoryHateSpeech
        );
    }

    #[test]
    fn test_tool_serde() {
        let tool = Tool::FunctionDeclaration {
            function_declarations: vec![FunctionDeclaration {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                })),
                response: None,
            }],
        };

        let json_value = serde_json::to_value(&tool).unwrap();
        assert_eq!(
            json_value,
            json!({
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                ]
            })
        );

        let deserialized: Tool = serde_json::from_value(json_value).unwrap();
        assert_eq!(
            serde_json::to_value(deserialized).unwrap(),
            serde_json::to_value(tool).unwrap()
        );

        let code_tool = Tool::CodeExecution {
            code_execution: CodeExecution {},
        };
        let json_value = serde_json::to_value(&code_tool).unwrap();
        assert_eq!(json_value, json!({ "codeExecution": {} }));
        let deserialized: Tool = serde_json::from_value(json_value).unwrap();
        assert_eq!(
            serde_json::to_value(deserialized).unwrap(),
            serde_json::to_value(code_tool).unwrap()
        );

        let search_retrieval_tool = Tool::GoogleSearchRetrieval {
            google_search_retrieval: GoogleSearchRetrieval {
                dynamic_retrieval_config: Some(DynamicRetrievalConfig {
                    mode: Some("MODE_DYNAMIC".to_string()),
                    dynamic_threshold: Some(0.3),
                    extra: Map::new(),
                }),
                extra: Map::new(),
            },
        };
        let json_value = serde_json::to_value(&search_retrieval_tool).unwrap();
        assert_eq!(
            json_value,
            json!({
                "googleSearchRetrieval": {
                    "dynamicRetrievalConfig": {
                        "mode": "MODE_DYNAMIC",
                        "dynamicThreshold": 0.3
                    }
                }
            })
        );
        let deserialized: Tool = serde_json::from_value(json_value).unwrap();
        assert_eq!(
            serde_json::to_value(deserialized).unwrap(),
            serde_json::to_value(search_retrieval_tool).unwrap()
        );

        let search_tool = Tool::GoogleSearch {
            google_search: GoogleSearch {},
        };
        let json_value = serde_json::to_value(&search_tool).unwrap();
        assert_eq!(json_value, json!({ "googleSearch": {} }));
        let deserialized: Tool = serde_json::from_value(json_value).unwrap();
        assert_eq!(
            serde_json::to_value(deserialized).unwrap(),
            serde_json::to_value(search_tool).unwrap()
        );

        let url_context_tool = Tool::UrlContext {
            url_context: UrlContext {},
        };
        let json_value = serde_json::to_value(&url_context_tool).unwrap();
        assert_eq!(json_value, json!({ "urlContext": {} }));
        let deserialized: Tool = serde_json::from_value(json_value).unwrap();
        assert_eq!(
            serde_json::to_value(deserialized).unwrap(),
            serde_json::to_value(url_context_tool).unwrap()
        );
    }

    #[test]
    fn test_tool_from_function_definitions() {
        let defs = vec![FunctionDefinition {
            name: "sum".to_string(),
            description: "Sum two integers".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }),
            strict: None,
        }];

        let tool: Tool = defs.into();
        let json_value = serde_json::to_value(&tool).unwrap();
        assert_eq!(
            json_value,
            json!({
                "functionDeclarations": [
                    {
                        "name": "sum",
                        "description": "Sum two integers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"}
                            },
                            "required": ["a", "b"]
                        }
                    }
                ]
            })
        );
    }
}
