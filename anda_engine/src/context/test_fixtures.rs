//! Shared test doubles for the context and runner test suites.
//!
//! Scripted [`CompletionFeaturesDyn`] completers, echo/fail tools and agents,
//! and small builders used by `context::agent` and `context::runner` tests.

use anda_core::{
    Agent, AgentOutput, BoxError,
    CompletionRequest, ContentPart, Function, FunctionDefinition, Message, Resource, Tool,
    ToolCall, ToolOutput, Usage,
};
use candid::Principal;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::context::base::BaseCtx;
use crate::context::engine::{AgentInfo, EngineCard, RemoteEngines};
use crate::context::AgentCtx;
use crate::model::CompletionFeaturesDyn;

#[derive(Clone, Debug)]
pub(crate) struct AlwaysFailCompleter;

impl CompletionFeaturesDyn for AlwaysFailCompleter {
    fn model_name(&self) -> String {
        "always_fail".to_string()
    }

    fn completion(
        &self,
        _req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Ok(AgentOutput {
            failed_reason: Some("primary failed".to_string()),
            ..Default::default()
        })))
    }
}

/// Completer that echoes prompt as content, no tool calls.
#[derive(Clone, Debug)]
pub(crate) struct EchoCompleter;

impl CompletionFeaturesDyn for EchoCompleter {
    fn model_name(&self) -> String {
        "echo".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        let content = if req.prompt.is_empty() {
            req.content
                .iter()
                .map(|part| match part {
                    anda_core::ContentPart::Text { text }
                    | anda_core::ContentPart::Reasoning { text } => text.clone(),
                    _ => serde_json::to_string(part).unwrap_or_default(),
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            req.prompt.clone()
        };

        Box::pin(futures::future::ready(Ok(AgentOutput {
            content,
            usage: Usage {
                input_tokens: 5,
                output_tokens: 10,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

/// Completer that returns tool calls on the first call, then echoes on subsequent calls.
#[derive(Clone, Debug)]
pub(crate) struct ToolCallCompleter {
    pub(crate) tool_calls: Vec<ToolCall>,
}

impl CompletionFeaturesDyn for ToolCallCompleter {
    fn model_name(&self) -> String {
        "tool_call".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        // If the request role is "tool", it means we already executed tools,
        // so respond with final content.
        let role = req.role.as_deref().unwrap_or("");
        if role == "tool" {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "tool_result_processed".to_string(),
                usage: Usage {
                    input_tokens: 3,
                    output_tokens: 6,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        let tool_calls = self.tool_calls.clone();
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: String::new(),
            tool_calls,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ToolChainUntilFollowUpCompleter {
    pub(crate) requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for ToolChainUntilFollowUpCompleter {
    fn model_name(&self) -> String {
        "tool_chain_until_follow_up".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        let request_index = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(req.clone());
            requests.len()
        };
        let saw_follow_up = req.content.iter().any(|part| {
            matches!(
                part,
                ContentPart::Text { text }
                    if text == "follow up while tool chain is pending"
            )
        });

        if saw_follow_up {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "follow_up_seen_with_tool_result".to_string(),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "chain"}),
                call_id: Some(format!("chain_call_{request_index}")),
                result: None,
                remote_id: None,
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoveryCompleter {
    pub(crate) requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for DiscoveryCompleter {
    fn model_name(&self) -> String {
        "discovery".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().unwrap().push(req.clone());

        if req.role.as_deref() == Some("tool")
            && req.content.iter().any(|part| {
                matches!(
                    part,
                    ContentPart::ToolOutput { name, .. } if name == "echo_tool"
                )
            })
        {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "echo tool used after discovery".to_string(),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        if req.role.as_deref() == Some("tool")
            && req
                .tools
                .iter()
                .any(|tool| tool.name.as_str() == "echo_tool")
        {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "after-select"}),
                    call_id: Some("call_echo_after_select".into()),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        if req.role.as_deref() == Some("tool") {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "tools_select".to_string(),
                    args: json!({
                        "tools": ["echo_tool"],
                        "query": "",
                        "limit": 0
                    }),
                    call_id: Some("select_echo_tool_again".into()),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "tools_select".to_string(),
                args: json!({
                    "tools": ["echo_tool"],
                    "query": "",
                    "limit": 0
                }),
                call_id: Some("select_echo_tool".into()),
                result: None,
                remote_id: None,
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoveryCompactionCompleter {
    pub(crate) requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for DiscoveryCompactionCompleter {
    fn model_name(&self) -> String {
        "discovery_compaction".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().unwrap().push(req.clone());

        let prompt = if req.prompt.is_empty() {
            req.content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                        text.clone()
                    }
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            req.prompt.clone()
        };

        if prompt.trim() == super::COMPACTION_PROMPT.trim() {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "compacted handoff".to_string(),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        if req.role.as_deref() != Some("tool")
            && req
                .tools
                .iter()
                .any(|tool| tool.name.as_str() == "echo_tool")
        {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "after-handoff"}),
                    call_id: Some("call_echo_after_handoff".into()),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "tools_select".to_string(),
                args: json!({
                    "tools": ["echo_tool"],
                    "query": "",
                    "limit": 0
                }),
                call_id: Some("select_echo_tool".into()),
                result: None,
                remote_id: None,
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

/// Completer that returns agent calls.
#[derive(Clone, Debug)]
pub(crate) struct AgentCallCompleter {
    pub(crate) agent_name: String,
}

impl CompletionFeaturesDyn for AgentCallCompleter {
    fn model_name(&self) -> String {
        "agent_call".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        let role = req.role.as_deref().unwrap_or("");
        if role == "tool" {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "agent_result_processed".to_string(),
                usage: Usage {
                    input_tokens: 2,
                    output_tokens: 4,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        let agent_name = self.agent_name.clone();
        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: agent_name,
                args: json!({"prompt": "subagent task"}),
                call_id: Some("agent_call_1".into()),
                result: None,
                remote_id: None,
            }],
            usage: Usage {
                input_tokens: 8,
                output_tokens: 16,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

/// Completer that returns an Err (not failed_reason, but actual error).
#[derive(Clone, Debug)]
pub(crate) struct ErrorCompleter;

impl CompletionFeaturesDyn for ErrorCompleter {
    fn model_name(&self) -> String {
        "error".to_string()
    }

    fn completion(
        &self,
        _req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Err("model error".into())))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ToolResultErrorCompleter;

impl CompletionFeaturesDyn for ToolResultErrorCompleter {
    fn model_name(&self) -> String {
        "tool_result_error".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        if req.role.as_deref() == Some("tool") {
            return Box::pin(futures::future::ready(Err("model error".into())));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "hello"}),
                call_id: Some("call_1".into()),
                result: None,
                remote_id: None,
            }],
            raw_history: vec![json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo_tool",
                        "arguments": "{\"input\":\"hello\"}"
                    }
                }]
            })],
            ..Default::default()
        })))
    }
}

/// Like [`ToolResultErrorCompleter`] but the first turn also populates the
/// normalized `chat_history` with the assistant tool call, as real provider
/// adapters do. This exercises variant A of the discard bug: after the tool
/// round executes and the follow-up model call fails, the visible history
/// holds an unanswered `ToolCall`.
#[derive(Clone, Debug)]
pub(crate) struct ToolResultErrorWithHistoryCompleter;

impl CompletionFeaturesDyn for ToolResultErrorWithHistoryCompleter {
    fn model_name(&self) -> String {
        "tool_result_error_with_history".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        if req.role.as_deref() == Some("tool") {
            return Box::pin(futures::future::ready(Err("model error".into())));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "hello"}),
                call_id: Some("call_1".into()),
                result: None,
                remote_id: None,
            }],
            chat_history: vec![Message {
                role: "assistant".to_string(),
                content: vec![ContentPart::ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "hello"}),
                    call_id: Some("call_1".into()),
                }],
                ..Default::default()
            }],
            raw_history: vec![json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo_tool",
                        "arguments": "{\"input\":\"hello\"}"
                    }
                }]
            })],
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ToolCallHistoryCompleter {
    pub(crate) requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for ToolCallHistoryCompleter {
    fn model_name(&self) -> String {
        "tool_call_history".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().unwrap().push(req.clone());
        let text = req
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if req.prompt == "start tool" || text.contains("start tool") {
            let call = ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "stop"}),
                call_id: Some("call_stop_test".to_string()),
                result: None,
                remote_id: None,
            };
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![call.clone()],
                chat_history: vec![
                    Message {
                        role: "user".to_string(),
                        content: vec![ContentPart::Text {
                            text: "start tool".to_string(),
                        }],
                        ..Default::default()
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: vec![
                            ContentPart::Text {
                                text: "planning before tool".to_string(),
                            },
                            ContentPart::ToolCall {
                                name: call.name,
                                args: call.args,
                                call_id: call.call_id,
                            },
                        ],
                        ..Default::default()
                    },
                ],
                raw_history: vec![
                    json!({
                        "role": "assistant",
                        "content": "planning before tool",
                        "tool_calls": [{
                            "id": "call_stop_test",
                            "type": "function",
                            "function": {
                                "name": "echo_tool",
                                "arguments": "{\"input\":\"stop\"}"
                            }
                        }]
                    }),
                    json!({
                        "type": "function_call",
                        "call_id": "call_stop_test",
                        "name": "echo_tool",
                        "arguments": "{\"input\":\"stop\"}"
                    }),
                ],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: "continued".to_string(),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

/// Completer that waits forever (for cancellation tests).
#[derive(Clone, Debug)]
pub(crate) struct SlowCompleter;

impl CompletionFeaturesDyn for SlowCompleter {
    fn model_name(&self) -> String {
        "slow".to_string()
    }

    fn completion(
        &self,
        _req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(async {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            Ok(AgentOutput::default())
        })
    }
}

/// Completer that completes after a short delay, exercising stream pending state.
#[derive(Clone, Debug)]
pub(crate) struct DelayedEchoCompleter;

impl CompletionFeaturesDyn for DelayedEchoCompleter {
    fn model_name(&self) -> String {
        "delayed_echo".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(AgentOutput {
                content: req.prompt,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RawHistoryToolCallCompleter {
    pub(crate) requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for RawHistoryToolCallCompleter {
    fn model_name(&self) -> String {
        "raw_history_tool_call".to_string()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().unwrap().push(req.clone());

        if req.role.as_deref() == Some("user") {
            return Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "steered".to_string(),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })));
        }

        Box::pin(futures::future::ready(Ok(AgentOutput {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "raw history"}),
                call_id: Some("raw_call".into()),
                result: None,
                remote_id: None,
            }],
            raw_history: vec![
                json!({
                    "role": "assistant",
                    "content": "planning tool call",
                    "tool_calls": [{
                        "id": "raw_call",
                        "type": "function",
                        "function": {
                            "name": "echo_tool",
                            "arguments": "{\"input\":\"raw history\"}"
                        }
                    }],
                    "reasoning": "keep this reasoning"
                }),
                json!({"type": "function_call", "call_id": "raw_call"}),
            ],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RecordingCompleter {
    pub(crate) name: String,
    pub(crate) requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl CompletionFeaturesDyn for RecordingCompleter {
    fn model_name(&self) -> String {
        self.name.clone()
    }

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().unwrap().push(req.clone());
        let content = if req.prompt.is_empty() {
            req.content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                        text.clone()
                    }
                    _ => serde_json::to_string(part).unwrap_or_default(),
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            req.prompt
        };

        Box::pin(futures::future::ready(Ok(AgentOutput {
            content,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })))
    }
}

// ── Helper tool ──

pub(crate) struct EchoTool;

#[derive(Debug, Deserialize)]
pub(crate) struct EchoToolArgs {
    #[serde(default)]
    pub(crate) input: String,
}

impl Tool<BaseCtx> for EchoTool {
    type Args = EchoToolArgs;
    type Output = String;

    fn name(&self) -> String {
        "echo_tool".to_string()
    }

    fn description(&self) -> String {
        "Echoes input back".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "echo_tool".to_string(),
            description: "Echoes input back".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<String>, BoxError> {
        Ok(ToolOutput {
            output: format!("echoed:{}", args.input),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })
    }
}

/// A tool that always fails.
pub(crate) struct FailTool;

#[derive(Debug, Deserialize)]
pub(crate) struct FailToolArgs {}

impl Tool<BaseCtx> for FailTool {
    type Args = FailToolArgs;
    type Output = String;

    fn name(&self) -> String {
        "fail_tool".to_string()
    }

    fn description(&self) -> String {
        "Always fails".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "fail_tool".to_string(),
            description: "Always fails".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        _args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<String>, BoxError> {
        Err("tool execution failed".into())
    }
}

// ── Helper agent ──

pub(crate) struct EchoAgent;

impl Agent<AgentCtx> for EchoAgent {
    fn name(&self) -> String {
        "echo_agent".to_string()
    }

    fn description(&self) -> String {
        "Echoes prompt back".to_string()
    }

    async fn run(
        &self,
        _ctx: AgentCtx,
        prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        Ok(AgentOutput {
            content: format!("agent_echoed:{}", prompt),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        })
    }
}

pub(crate) struct FailAgent;

impl Agent<AgentCtx> for FailAgent {
    fn name(&self) -> String {
        "fail_agent".to_string()
    }

    fn description(&self) -> String {
        "Always fails".to_string()
    }

    async fn run(
        &self,
        _ctx: AgentCtx,
        _prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        Err("agent execution failed".into())
    }
}

pub(crate) fn function(name: &str, description: &str, tags: &[&str]) -> Function {
    Function {
        definition: FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters: json!({"type": "object"}),
            ..Default::default()
        },
        supported_resource_tags: tags.iter().map(|tag| tag.to_string()).collect(),
    }
}

pub(crate) fn resource(id: u64, tags: &[&str]) -> Resource {
    Resource {
        _id: id,
        name: format!("resource-{id}"),
        tags: tags.iter().map(|tag| tag.to_string()).collect(),
        ..Default::default()
    }
}

pub(crate) fn dynamic_remote_engines() -> RemoteEngines {
    let mut engines = BTreeMap::new();
    engines.insert(
        "dyn".to_string(),
        EngineCard {
            id: Principal::self_authenticating([9; 32]),
            info: AgentInfo {
                handle: "Dynamic".to_string(),
                endpoint: "https://dynamic.example".to_string(),
                ..Default::default()
            },
            agents: vec![function("chat", "Chat remotely", &["md"])],
            tools: vec![function("lookup", "Lookup remotely", &["text"])],
        },
    );
    RemoteEngines { engines }
}

