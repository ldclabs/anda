//! OpenAI API client implementation for Anda Engine
//!
//! This module provides integration with OpenAI's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Embedding model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, ContentPart, FunctionDefinition, Json,
    Message, Usage as ModelUsage, part_to_data_url,
};
use log::{Level::Debug, log_enabled};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

pub mod types;

use super::{CompletionFeaturesDyn, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

// ================================================================
// Main OpenAI Client
// ================================================================
const API_BASE_URL: &str = "https://api.openai.com/v1";

/// Default completion model to use if not specified
pub const DEFAULT_COMPLETION_MODEL: &str = "gpt-5.4-mini";

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
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: usize,
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

    #[serde(default)]
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

            if let Some(refusal) = &msg.refusal {
                output.failed_reason = Some(refusal.clone());
            }

            let mut msg: Message = msg.into();
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
                && choice
                    .parsed_message
                    .as_ref()
                    .map(|m| !m.content.is_empty() || m.tool_calls.is_some())
                    .unwrap_or(false)
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
                if msg.content.len() == 1 {
                    res.content = serde_json::to_string(output).unwrap_or_default().into();
                    res.tool_call_id = call_id.clone();
                    return res;
                }
                arr.push(json!({
                    "type": "text",
                    "text": serde_json::to_string(output).unwrap_or_default(),
                }));
                res.tool_call_id = call_id.clone();
            }
            ContentPart::FileData {
                file_uri,
                mime_type,
            } => match mime_type.clone().unwrap_or_default().as_str() {
                mt if mt.starts_with("image") => arr.push(json!({
                    "type": "image_url",
                    "image_url":  {
                        "url": file_uri,
                    },
                })),
                _ => arr.push(json!({
                    "type": "file",
                    "file":  {
                        "file_data": file_uri,
                    },
                })),
            },
            ContentPart::InlineData { data, mime_type } => {
                match mime_type.as_str() {
                    mt if mt.starts_with("image") => {
                        arr.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": part_to_data_url(data, Some(mime_type)),
                            },
                        }));
                    }
                    mt if mt.starts_with("video") => {
                        arr.push(json!({
                            "type": "video_url",
                            "video_url": {
                                "url": part_to_data_url(data, Some(mime_type)),
                            },
                        }));
                    }
                    mt if mt.starts_with("audio") => {
                        arr.push(json!({
                            "type": "input_audio",
                            "input_audio": {
                                "data": part_to_data_url(data, Some(mime_type)),
                                "format": if mt.contains("wav") { "wav" } else { "mp3" },
                            },
                        }));
                    }
                    _ => arr.push(json!({
                        "type": "file",
                        "file":  {
                            "file_data": data,
                        },
                    })),
                };
            }
            v => arr.push(json!(v)),
        }
    }
    res.content = Json::Array(arr);
    res
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Json,
    pub finish_reason: String,

    #[serde(skip)]
    pub parsed_message: Option<MessageOutput>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessageOutput {
    pub role: String,

    #[serde(default)]
    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,

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

/// Completion model implementation for OpenAI API
#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
    extra_body: Option<HashMap<String, Json>>,
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
            extra_body: None,
        }
    }

    /// Sets extra body parameters for the model.
    pub fn with_extra_body(mut self, extra_body: HashMap<String, Json>) -> Self {
        self.extra_body = Some(extra_body);
        self
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        let extra_body = self.extra_body.clone();

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
                raw_history.push(serde_json::to_value(to_message_input(&msg))?);
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                raw_history.push(serde_json::to_value(to_message_input(&msg))?);
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

                raw_history.push(serde_json::to_value(to_message_input(&msg))?);
                chat_history.push(msg);
            }

            let mut body = json!({
                "model": model,
                "messages": &raw_history,
            });

            let body = body.as_object_mut().unwrap();
            if let Some(temperature) = req.temperature {
                body.insert("temperature".to_string(), Json::from(temperature));
            }

            if let Some(max_tokens) = req.max_output_tokens {
                body.insert("max_completion_tokens".to_string(), Json::from(max_tokens));
            }

            if let Some(output_schema) = req.output_schema {
                body.insert(
                    "response_format".to_string(),
                    json!({ "type": "json_schema", "json_schema": output_schema }),
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

            if let Some(extra_body) = extra_body {
                for (k, v) in extra_body.into_iter() {
                    body.insert(k, v);
                }
            }

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&body)
            {
                log::debug!(request = val; "OpenAI completions request");
            }

            let response = client.post("/chat/completions").json(body).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                match serde_json::from_str::<CompletionResponse>(&text) {
                    Ok(mut res) => {
                        res.parse_output();
                        if log_enabled!(Debug) {
                            log::debug!(
                                model = model,
                                request:serde = body,
                                response:serde = res;
                                "Completion response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                model = model,
                                request:serde = body,
                                response:serde = res;
                                "Completion maybe failed");
                        }
                        if skip_raw > 0 {
                            raw_history.drain(0..skip_raw);
                        }
                        res.try_into(raw_history, chat_history)
                    }
                    Err(err) => Err(format!(
                        "Invalid completion response, model: {}, error: {}, body: {}",
                        model, err, text
                    )
                    .into()),
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = model,
                    request:serde = body;
                    "Completion request failed: {status}, body: {msg}",
                );
                Err(format!("Completion failed, model: {}, error: {}", model, msg).into())
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
                additional_parameters: types::AdditionalParameters {
                    store: Some(false),
                    reasoning: Some(types::Reasoning::default()),
                    ..Default::default()
                },
                ..Default::default()
            },
            model: model.to_string(),
        }
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
        let mut oreq = self.default_request.clone();
        oreq.model = self.model.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut raw_history: Vec<Json> = Vec::new();
            let mut chat_history: Vec<Message> = Vec::new();

            if !req.instructions.is_empty() {
                oreq.instructions = Some(req.instructions);
            };

            oreq.input
                .extend(req.raw_history.into_iter().map(types::MessageItem::Any));

            for msg in req.chat_history {
                let vals = types::message_into(msg);
                for val in vals {
                    raw_history.push(serde_json::to_value(&val)?);
                    oreq.input.push(val);
                }
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let vals = types::message_into(msg);
                for val in vals {
                    raw_history.push(serde_json::to_value(&val)?);
                    oreq.input.push(val);
                }
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

                chat_history.push(msg.clone());
                let vals = types::message_into(msg);
                for val in vals {
                    raw_history.push(serde_json::to_value(&val)?);
                    oreq.input.push(val);
                }
            }

            if let Some(temperature) = req.temperature {
                oreq.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                oreq.max_output_tokens = Some(max_tokens as u64);
            }

            if let Some(output_schema) = req.output_schema {
                oreq.additional_parameters.text = Some(types::TextConfig::structured_output(
                    "structured_output".to_string(),
                    output_schema,
                ));
            }

            if !req.tools.is_empty() {
                oreq.tools = req
                    .tools
                    .into_iter()
                    .map(|v| types::ToolDefinition {
                        r#type: "function".to_string(),
                        name: v.name,
                        description: v.description,
                        parameters: v.parameters,
                        strict: v.strict.unwrap_or_default(),
                    })
                    .collect::<Vec<_>>();
                oreq.tool_choice = Some(if req.tool_choice_required {
                    "required".to_string()
                } else {
                    "auto".to_string()
                });
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&oreq)
            {
                log::debug!(request = val; "Completion request");
            }

            let response = client.post("/responses").json(&oreq).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                match serde_json::from_str::<types::CompletionResponse>(&text) {
                    Ok(mut res) => {
                        res.parse_output();

                        if log_enabled!(Debug) {
                            log::debug!(
                                model = oreq.model,
                                request:serde = oreq,
                                response:serde = res;
                                "Completion response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                request:serde = oreq,
                                response:serde = res;
                                "Completion maybe failed");
                        }

                        res.try_into(raw_history, chat_history)
                    }
                    Err(err) => Err(format!(
                        "Invalid completion response, model: {}, error: {}, body: {}",
                        oreq.model, err, text
                    )
                    .into()),
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    model = oreq.model,
                    request:serde = oreq;
                    "Completion request failed: {status}, body: {msg}",
                );
                Err(format!("Completion failed, model: {}, error: {}", oreq.model, msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
