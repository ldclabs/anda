//! OpenAI API client and Anda integration
//!
use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, FunctionDefinition, MessageInput,
    ToolCall, CONTENT_TYPE_JSON,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use super::CompletionFeaturesDyn;
use crate::APP_USER_AGENT;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEKSEEK_API_BASE_URL: &str = "https://api.deepseek.com";
static DEEKSEEK_MODEL: &str = "deepseek-chat";

#[derive(Clone)]
pub struct Client {
    endpoint: String,
    http: reqwest::Client,
}

impl Client {
    /// Create a new DeepSeek client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            endpoint: DEEKSEEK_API_BASE_URL.to_string(),
            http: reqwest::Client::builder()
                .use_rustls_tls()
                .https_only(true)
                .http2_keep_alive_interval(Some(Duration::from_secs(25)))
                .http2_keep_alive_timeout(Duration::from_secs(15))
                .http2_keep_alive_while_idle(true)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(60))
                .user_agent(APP_USER_AGENT)
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    let ct: http::HeaderValue = CONTENT_TYPE_JSON.parse().unwrap();
                    headers.insert(http::header::CONTENT_TYPE, ct.clone());
                    headers.insert(http::header::ACCEPT, ct);
                    headers.insert(
                        http::header::AUTHORIZATION,
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("DeepSeek reqwest client should build"),
        }
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url)
    }

    /// Create a completion model.
    pub fn completion_model(&self) -> CompletionModel {
        CompletionModel::new(self.clone(), DEEKSEEK_MODEL)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    fn try_into(mut self) -> Result<AgentOutput, BoxError> {
        let choice = self.choices.pop().ok_or("No completion choice")?;
        let mut output = AgentOutput {
            content: choice.message.content,
            tool_calls: choice.message.tool_calls.map(|tools| {
                tools
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        args: tc.function.arguments,
                        result: None,
                    })
                    .collect()
            }),
            ..Default::default()
        };

        if !matches!(choice.finish_reason.as_str(), "stop" | "tool_calls") {
            output.failed_reason = Some(choice.finish_reason);
        }
        if let Some(refusal) = choice.message.refusal {
            output.failed_reason = Some(refusal);
        }

        Ok(output)
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: MessageOutput,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct MessageOutput {
    pub role: String,
    #[serde(default)]
    pub content: String,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            // Add system to chat history (if available)
            let mut full_history = if let Some(system) = &req.system {
                vec![MessageInput {
                    role: "system".into(),
                    content: system.clone(),
                    ..Default::default()
                }]
            } else {
                vec![]
            };

            // Extend existing chat history
            full_history.append(&mut req.chat_history);

            // Add context documents to chat history
            full_history.push(MessageInput {
                role: "user".into(),
                content: req.prompt,
                ..Default::default()
            });

            let mut body = json!({
                "model": model,
                "messages": full_history,
                "temperature": req.temperature,
            });
            let body = body.as_object_mut().unwrap();

            if let Some(max_tokens) = req.max_tokens {
                body.insert("max_tokens".to_string(), Value::from(max_tokens));
            }

            if req.response_format.is_some() {
                // DeepSeek only supports `{"type": "json_object"}`
                body.insert(
                    "response_format".to_string(),
                    json!({"type": "json_object"}),
                );
            }

            if let Some(stop) = req.stop {
                body.insert("stop".to_string(), Value::from(stop));
            }

            if !req.tools.is_empty() {
                body.insert(
                    "tools".to_string(),
                    json!(req
                        .tools
                        .into_iter()
                        .map(ToolDefinition::from)
                        .collect::<Vec<_>>()),
                );
                body.insert(
                    "tool_choice".to_string(),
                    if req.tool_choice_required {
                        Value::from("required")
                    } else {
                        Value::from("auto")
                    },
                );
            };

            let response = client.post("/chat/completions").json(body).send().await?;
            if response.status().is_success() {
                match response.json::<CompletionResponse>().await {
                    Ok(res) => res.try_into(),
                    Err(err) => Err(format!("DeepSeek completions error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("DeepSeek completions error: {}", msg).into())
            }
        })
    }
}
