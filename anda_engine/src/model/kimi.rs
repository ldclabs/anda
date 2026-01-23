//! Kimi Moonshot API client implementation for Anda Engine
//!
//! This module provides integration with Kimi's API, including:
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
// Main Kimi Client
// ================================================================
const API_BASE_URL: &str = "https://api.moonshot.cn/v1";
pub static KIMI_K2: &str = "kimi-k2-0711-preview";

/// Kimi API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new Kimi client instance with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - Kimi API key for authentication
    ///
    /// # Returns
    /// Configured Kimi client instance
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
                .expect("Kimi reqwest client should build"),
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

    /// Creates a new completion model instance using the default Kimi model
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), if model.is_empty() { KIMI_K2 } else { model })
    }
}

/// Token usage statistics from Kimi API responses
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

/// Completion response from Kimi API
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
                && (choice.message.content.is_some() || choice.message.tool_calls.is_some())
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

fn to_message_input(msg: &Message) -> Vec<MessageInput> {
    let mut res = Vec::new();
    for content in msg.content.iter() {
        match content {
            ContentPart::Text { text } => res.push(MessageInput {
                role: msg.role.clone(),
                content: text.clone().into(),
                tool_call_id: None,
            }),
            ContentPart::ToolOutput {
                output, call_id, ..
            } => res.push(MessageInput {
                role: msg.role.clone(),
                content: serde_json::to_string(output).unwrap_or_default().into(),
                tool_call_id: call_id.clone(),
            }),
            ContentPart::FileData {
                file_uri,
                mime_type,
            } => res.push(MessageInput {
                role: msg.role.clone(),
                content: match mime_type.clone().unwrap_or_default().as_str() {
                    mt if mt.starts_with("image") => {
                        json!({
                            "type": "input_image",
                            "image_url":  {
                                "url": file_uri,
                            },
                        })
                    }
                    _ => serde_json::to_string(content).unwrap_or_default().into(),
                },
                tool_call_id: None,
            }),
            ContentPart::InlineData { data, mime_type } => res.push(MessageInput {
                role: msg.role.clone(),
                content: match mime_type.as_str() {
                    mt if mt.starts_with("image") => {
                        json!({
                            "type": "input_image",
                            "image_url":  {
                                "url": data,
                            },
                        })
                    }
                    _ => json!({
                        "type": "file",
                        "file":  {
                            "file_data": data,
                        },
                    }),
                },
                tool_call_id: None,
            }),
            // TODO: handle other content parts
            v => res.push(MessageInput {
                role: msg.role.clone(),
                content: serde_json::to_string(v).unwrap_or_default().into(),
                tool_call_id: None,
            }),
        }
    }
    res
}

/// Individual completion choice from Kimi API
#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: MessageOutput,
    pub finish_reason: String,
}

/// Output message structure from Kimi API
#[derive(Debug, Deserialize, Serialize)]
pub struct MessageOutput {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

impl From<MessageOutput> for Message {
    fn from(msg: MessageOutput) -> Self {
        let mut content = Vec::new();
        if let Some(text) = msg.content {
            content.push(ContentPart::Text { text });
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

/// Tool call output structure from Kimi API
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

/// Completion model wrapper for Kimi API
#[derive(Clone)]
pub struct CompletionModel {
    /// Kimi client instance
    client: Client,
    /// Model identifier
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - Kimi client instance
    /// * `model` - Model identifier string
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
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
                let val = to_message_input(&msg);
                for v in val {
                    raw_history.push(serde_json::to_value(&v)?);
                }
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                let val = to_message_input(&msg);
                for v in val {
                    raw_history.push(serde_json::to_value(&v)?);
                }
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

                let val = to_message_input(&msg);
                for v in val {
                    raw_history.push(serde_json::to_value(&v)?);
                }
                chat_history.push(msg);
            }

            let mut body = json!({
                "model": model,
                "messages": &raw_history,
            });

            let body = body.as_object_mut().unwrap();
            if let Some(temperature) = req.temperature {
                // Kimi temperature is in range [0, 1]
                body.insert("temperature".to_string(), Json::from(temperature / 2.0));
            }

            if let Some(max_tokens) = req.max_output_tokens {
                body.insert("max_tokens".to_string(), Json::from(max_tokens));
            }

            if req.output_schema.is_some() {
                // DeepSeek only supports `{"type": "json_object"}`
                body.insert(
                    "response_format".to_string(),
                    json!({"type": "json_object"}),
                );
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

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&body)
            {
                log::debug!(request = val; "Kimi completions request");
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
                                "Kimi completions response");
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
                        Err(format!("Kimi completions error: {}, body: {}", err, text).into())
                    }
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    request:serde = body;
                    "completions request failed: {status}, body: {msg}",
                );
                Err(format!("Kimi completions error: {}", msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_kimi() {}
}
