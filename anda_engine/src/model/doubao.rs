//! Doubao Seed API client implementation for Anda Engine
//!
//! This module provides integration with Doubao's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, ContentPart,
    FunctionDefinition, Json, Message, Resource, Usage as ModelUsage,
};
use log::{Level::Debug, log_enabled};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{CompletionFeaturesDyn, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

// ================================================================
// Main Doubao Client https://www.volcengine.com/docs/82379/1494384?lang=zh
// ================================================================
const API_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/v3";
pub static DOUBAO_SEED2_LITE: &str = "doubao-seed-2-0-lite-260215";

/// Doubao API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new Doubao client instance with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - Doubao API key for authentication
    ///
    /// # Returns
    /// Configured Doubao client instance
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
                .expect("Doubao reqwest client should build"),
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

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url).bearer_auth(&self.api_key)
    }

    /// Creates a new completion model instance using the default Doubao model
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(
            self.clone(),
            if model.is_empty() {
                DOUBAO_SEED2_LITE
            } else {
                model
            },
        )
    }
}

/// Token usage statistics from Doubao API responses
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    /// Number of tokens used in the prompt
    pub prompt_tokens: usize,
    /// Number of tokens used in the completion
    pub completion_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} completion tokens: {}",
            self.prompt_tokens, self.completion_tokens
        )
    }
}

/// Completion response from Doubao API
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (typically "chat.completion")
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of completion choices
    pub choices: Vec<Choice>,
    /// Token usage statistics
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    fn try_into(
        mut self,
        raw_history: Vec<Json>,
        chat_history: Vec<Message>,
    ) -> Result<AgentOutput, BoxError> {
        let mut output = AgentOutput {
            raw_history,
            chat_history,
            usage: self
                .usage
                .as_ref()
                .map(|u| ModelUsage {
                    input_tokens: u.prompt_tokens as u64,
                    output_tokens: u.completion_tokens as u64,
                    requests: 1,
                })
                .unwrap_or_default(),
            ..Default::default()
        };

        let choice = self.choices.pop().ok_or("No completion choice")?;
        if !matches!(choice.finish_reason.as_str(), "stop" | "tool_calls") {
            output.failed_reason = Some(choice.finish_reason);
        } else {
            output.raw_history.push(json!(&choice.message));
            let timestamp = unix_ms();
            let mut msg: Message = choice.message.into();
            msg.name = Some(self.model);
            msg.timestamp = Some(timestamp);
            output.content = msg.text().unwrap_or_default();
            output.tool_calls = msg.tool_calls();
            output.chat_history.push(msg);
        }

        Ok(output)
    }

    fn maybe_failed(&self) -> bool {
        !self.choices.iter().any(|choice| {
            matches!(choice.finish_reason.as_str(), "stop" | "tool_calls")
                && (!choice.message.content.is_empty() || choice.message.tool_calls.is_some())
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessageInput {
    pub role: String,

    pub content: Json,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

fn to_message_input(msg: &Message) -> MessageInput {
    let mut arr: Vec<Json> = Vec::new();
    let mut res = MessageInput {
        role: msg.role.clone(),
        content: Json::Null,
        tool_call_id: None,
    };
    for content in msg.content.iter() {
        match content {
            ContentPart::Text { text } => arr.push(json!({
                "type": "text",
                "text": text,
            })),
            ContentPart::ToolOutput {
                output, call_id, ..
            } => {
                arr.push(serde_json::to_string(output).unwrap_or_default().into());
                res.tool_call_id = call_id.clone();
            }
            ContentPart::InlineData { data, mime_type } => {
                match mime_type.as_str() {
                    mt if mt.starts_with("image") => {
                        arr.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": data.to_string(),
                            },
                        }));
                    }
                    mt if mt.starts_with("video") => {
                        arr.push(json!({
                            "type": "video_url",
                            "video_url": {
                                "url": data.to_string(),
                            },
                        }));
                    }
                    _ => {}
                };
            }
            v => arr.push(json!(v)),
        }
    }
    res.content = Json::Array(arr);
    res
}

/// Individual completion choice from Doubao API
#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: MessageOutput,
    pub finish_reason: String,
}

/// Output message structure from Doubao API
#[derive(Debug, Deserialize, Serialize)]
pub struct MessageOutput {
    pub role: String,
    #[serde(default)]
    pub content: String,

    // 模型处理问题的思维链内容，仅深度推理模型支持返回此字段
    #[serde(default)]
    pub reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

impl From<MessageOutput> for Message {
    fn from(msg: MessageOutput) -> Self {
        let mut content = Vec::new();
        if !msg.content.is_empty() {
            content.push(ContentPart::Text { text: msg.content });
        }
        if let Some(tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                content.push(ContentPart::ToolCall {
                    name: tc.function.name,
                    args: serde_json::from_str(&tc.function.arguments).unwrap_or_default(),
                    call_id: Some(tc.id),
                });
            }
        }
        Self {
            role: msg.role,
            content,
            ..Default::default()
        }
    }
}

/// Tool call output structure from Doubao API
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolCallOutput {
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

impl From<FunctionDefinition> for ToolDefinition {
    fn from(f: FunctionDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: f,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

/// Completion model wrapper for Doubao API
#[derive(Clone)]
pub struct CompletionModel {
    /// Doubao client instance
    client: Client,
    /// Model identifier
    pub model: String,
    // enabled， disabled，auto
    thinking: Option<String>,
    // minimal，low，medium，high
    reasoning_effort: Option<String>,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - Doubao client instance
    /// * `model` - Model identifier string
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            thinking: None,
            reasoning_effort: None,
        }
    }

    /// Sets thinking mode for the model, which can be enabled, disabled, or auto.
    /// Default is enabled.
    pub fn with_thinking(mut self, thinking: Option<String>) -> Self {
        self.thinking = thinking;
        self
    }

    /// Sets reasoning effort level for the model, which can be minimal, low, medium, or high.
    /// Default is medium.
    pub fn with_reasoning_effort(mut self, reasoning_effort: Option<String>) -> Self {
        self.reasoning_effort = reasoning_effort;
        self
    }
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
        let thinking = self.thinking.clone();
        let reasoning_effort = self.reasoning_effort.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut raw_history: Vec<Json> = Vec::new();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                raw_history.push(json!(MessageInput {
                    role: "system".into(),
                    content: req.instructions.clone().into(),
                    tool_call_id: None,
                }));
            };

            raw_history.append(&mut req.raw_history);
            let skip_raw = raw_history.len();

            for msg in req.chat_history {
                raw_history.push(json!(to_message_input(&msg)));
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                raw_history.push(json!(to_message_input(&msg)));
                chat_history.push(msg);
            }

            let mut content = req.content;
            if !req.prompt.is_empty() {
                content.push(req.prompt.into());
            }
            if !content.is_empty() {
                let msg = Message {
                    role: req.role.unwrap_or_else(|| "user".to_string()),
                    content,
                    timestamp: Some(timestamp),
                    ..Default::default()
                };

                raw_history.push(json!(to_message_input(&msg)));
                chat_history.push(msg);
            }

            let mut body = json!({
                "model": model,
                "messages": &raw_history,
            });

            let body = body.as_object_mut().unwrap();
            if let Some(temperature) = req.temperature {
                // Doubao temperature is in range [0, 1]
                body.insert("temperature".to_string(), Json::from(temperature.min(1.0)));
            }

            if let Some(max_tokens) = req.max_output_tokens {
                body.insert("max_tokens".to_string(), Json::from(max_tokens));
            }

            if let Some(output_schema) = req.output_schema {
                body.insert("response_format".to_string(), output_schema);
            }

            if let Some(stop) = req.stop {
                body.insert("stop".to_string(), Json::from(stop));
            }

            if !req.tools.is_empty() {
                body.insert(
                    "tools".to_string(),
                    json!(
                        req.tools
                            .into_iter()
                            .map(ToolDefinition::from)
                            .collect::<Vec<_>>()
                    ),
                );
                body.insert(
                    "tool_choice".to_string(),
                    if req.tool_choice_required {
                        Json::from("required")
                    } else {
                        Json::from("auto")
                    },
                );
            };

            if let Some(thinking) = thinking {
                body.insert(
                    "thinking".to_string(),
                    json!({
                        "type": thinking
                    }),
                );
            }

            if let Some(reasoning_effort) = reasoning_effort {
                body.insert("reasoning_effort".to_string(), reasoning_effort.into());
            }

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&body)
            {
                log::debug!(request = val; "Doubao completions request");
            }

            let response = client.post("/chat/completions").json(body).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                match serde_json::from_str::<CompletionResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            log::debug!(
                                request:serde = body,
                                response:serde = res;
                                "Doubao completions response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                request:serde = body,
                                response:serde = res;
                                "completions maybe failed");
                        }
                        if skip_raw > 0 {
                            raw_history.drain(0..skip_raw);
                        }
                        res.try_into(raw_history, chat_history)
                    }
                    Err(err) => {
                        Err(format!("Doubao completions error: {}, body: {}", err, text).into())
                    }
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    request:serde = body;
                    "completions request failed: {status}, body: {msg}",
                );
                Err(format!("Doubao completions error: {}", msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_doubao() {}
}
