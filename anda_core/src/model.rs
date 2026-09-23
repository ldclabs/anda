//! Core data models shared by agents, tools, and model adapters.
//!
//! The types in this module form the data contract between Anda runtimes,
//! model providers, agents, tools, and clients. They cover:
//! - agent and tool inputs/outputs ([`AgentInput`], [`AgentOutput`], [`ToolInput`], [`ToolOutput`]);
//! - chat messages and multimodal content ([`Message`], [`ContentPart`]);
//! - function-call metadata ([`FunctionDefinition`], [`ToolCall`]);
//! - request metadata and usage accounting ([`RequestMeta`], [`Usage`]);
//! - prompt documents and completion requests ([`Document`], [`Documents`], [`CompletionRequest`]).

use candid::Principal;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, json};
use std::collections::HashMap;

use crate::{Json, json::normalize_strict_schema};
pub use ic_auth_types::{ByteArrayB64, ByteBufB64, Xid};

mod completion;
mod content;
mod document;
mod resource;
mod text;

pub use completion::*;
pub use content::*;
pub use document::*;
pub use resource::*;
pub use text::*;

/// Request sent to an agent for processing.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentInput {
    /// Agent name. When empty, the runtime selects its default agent.
    pub name: String,

    /// User prompt or task message for the agent.
    pub prompt: String,

    /// The resources to process by the agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,

    /// The topics for the agent request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topics: Option<Vec<String>>,

    /// Metadata for the agent request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<RequestMeta>,
}

impl AgentInput {
    /// Creates a new agent input with the given name and prompt.
    pub fn new(name: String, prompt: String) -> Self {
        Self {
            name,
            prompt,
            resources: Vec::new(),
            topics: None,
            meta: None,
        }
    }
}

/// Parsed command prefix from an agent prompt.
///
/// Empty prompts and `/ping` (with or without arguments) are treated as
/// lightweight health checks. A leading slash with no command name (`/`,
/// `/ arg`) and prompts without a leading slash are plain user prompts. Other
/// slash-prefixed prompts keep the original prompt while exposing the
/// ASCII-lowercased command name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PromptCommand {
    /// Empty prompt or `/ping`.
    #[default]
    Ping,
    /// Prompt text without a command prefix.
    Plain {
        /// Original prompt text.
        prompt: String,
    },
    /// Slash-prefixed command and the original prompt text.
    Command {
        /// ASCII-lowercased command name without the leading slash.
        command: String,
        /// Original prompt text.
        prompt: String,
    },
}

impl From<String> for PromptCommand {
    fn from(prompt: String) -> Self {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return Self::Ping;
        }

        let Some(stripped) = trimmed.strip_prefix('/') else {
            return Self::Plain { prompt };
        };
        let command_end = stripped.find(char::is_whitespace).unwrap_or(stripped.len());
        let command = stripped[..command_end].to_ascii_lowercase();

        // A leading slash with no command name (`/`, `/ arg`) is a plain prompt.
        if command.is_empty() {
            return Self::Plain { prompt };
        }

        // `/ping` is a health check regardless of any trailing arguments, so it
        // resolves the same way whether or not arguments follow.
        if command == "ping" {
            return Self::Ping;
        }

        Self::Command { command, prompt }
    }
}

impl PromptCommand {
    /// Returns the argument text after the slash command prefix.
    ///
    /// If this command was built manually with a prompt that does not contain a matching slash
    /// prefix, the trimmed prompt is treated as the argument.
    pub fn command_argument(&self) -> Option<&str> {
        let Self::Command { command, prompt } = self else {
            return None;
        };

        let trimmed = prompt.trim();
        let Some(stripped) = trimmed.strip_prefix('/') else {
            return Some(trimmed);
        };

        let command_end = stripped.find(char::is_whitespace).unwrap_or(stripped.len());
        if !stripped[..command_end].eq_ignore_ascii_case(command) {
            return Some(trimmed);
        }

        Some(stripped[command_end..].trim())
    }
}

/// Output produced by an agent execution.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentOutput {
    /// Final visible content from the agent. It may be empty.
    pub content: String,

    /// Optional intermediate reasoning text returned by providers that expose it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts: Option<String>,

    /// The usage statistics for the agent execution.
    pub usage: Usage,

    /// The usage statistics for each tool called by the agent.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tools_usage: HashMap<String, Usage>,

    /// Failure reason if execution failed. `None` indicates success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_reason: Option<String>,

    /// Tool calls returned by the LLM function calling.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,

    /// The history of the conversation.
    ///
    /// The provider-neutral, persistable view (see [`ContentPart`]). This is what callers
    /// store and replay; it deliberately carries no provider-specific intermediate state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chat_history: Vec<Message>,

    /// Provider-specific conversation history used internally by model adapters.
    ///
    /// Carries the provider's own message JSON verbatim so per-turn opaque state (Anthropic
    /// `thinking.signature`, Gemini `thoughtSignature`) survives a reasoning round without
    /// being modelled in [`ContentPart`]. The runner appends this onto
    /// [`CompletionRequest::raw_history`](crate::model::CompletionRequest::raw_history) for
    /// the next turn; see that field for the full contract.
    ///
    /// Scoped to one in-process round: `#[serde(skip)]` keeps it off the wire and the engine
    /// clears it at the RPC boundary, so it is never part of the stable engine API response
    /// and never reaches a persisted conversation.
    #[serde(skip)]
    pub raw_history: Vec<Json>,

    /// A collection of artifacts generated by the agent during the execution of the task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Resource>,

    /// The conversation ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<u64>,

    /// The session ID for the agent execution, if applicable.
    /// This is used to correlate related conversations or executions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,

    /// The model used by the agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Partial agent output serialized when metadata must be preserved.
///
/// This compact shape is used when an [`AgentOutput`] is converted into a tool
/// output and cannot be represented as just the final content string or JSON
/// value.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PartialAgentOutput {
    /// Final visible content from the agent.
    pub content: String,

    /// Optional intermediate reasoning text returned by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts: Option<String>,

    /// Failure reason if execution failed. `None` indicates success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_reason: Option<String>,

    /// The conversation ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<u64>,

    /// The session ID for the agent execution, if applicable.
    /// This is used to correlate related conversations or executions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,

    /// The model used by the agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl AgentOutput {
    /// Converts an agent result into a JSON tool output.
    ///
    /// If the agent produced metadata such as thoughts, failure information,
    /// conversation IDs, or model labels, the output is wrapped as
    /// [`PartialAgentOutput`]. Otherwise the final content is parsed as JSON
    /// when possible and falls back to a JSON string.
    pub fn into_tool_output(self) -> ToolOutput<Json> {
        let AgentOutput {
            content,
            thoughts,
            usage,
            tools_usage,
            failed_reason,
            artifacts,
            conversation,
            session,
            model,
            ..
        } = self;
        // Treat a blank failure reason as success so the tool output never
        // carries a contradictory `is_error = false` beside an empty
        // `failed_reason`.
        let failed_reason = failed_reason.filter(|reason| !reason.trim().is_empty());
        let has_metadata = thoughts.is_some()
            || failed_reason.is_some()
            || conversation.is_some()
            || session.is_some()
            || model.is_some();

        let is_error = failed_reason.as_ref().map(|_| true);
        let output = if has_metadata {
            json!(PartialAgentOutput {
                content,
                thoughts,
                failed_reason,
                conversation,
                session,
                model,
            })
        } else {
            serde_json::from_str::<Json>(&content).unwrap_or(Json::String(content))
        };

        ToolOutput {
            output,
            is_error,
            artifacts,
            usage,
            tools_usage,
        }
    }
}

/// Request sent to a tool for processing.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolInput<T> {
    /// Tool name.
    pub name: String,

    /// Tool arguments.
    pub args: T,

    /// The resources to process by the tool.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,

    /// The metadata for the tool request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<RequestMeta>,
}

impl<T> ToolInput<T> {
    /// Creates a new tool input with the given name and arguments.
    pub fn new(name: String, args: T) -> Self {
        Self {
            name,
            args,
            resources: Vec::new(),
            meta: None,
        }
    }
}

/// Output produced by a tool execution.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolOutput<T> {
    /// The output from the tool.
    pub output: T,

    /// Indicates if the tool execution resulted in an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,

    /// A collection of artifacts generated by the tool execution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Resource>,

    /// The usage statistics for the tool execution.
    pub usage: Usage,

    /// The usage statistics for each tool called by the agent.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tools_usage: HashMap<String, Usage>,
}

impl<T> ToolOutput<T> {
    /// Creates a new tool output with the given output value.
    pub fn new(output: T) -> Self {
        Self {
            output,
            is_error: None,
            artifacts: Vec::new(),
            usage: Usage::default(),
            tools_usage: HashMap::new(),
        }
    }
}

/// Metadata attached to an agent or tool request.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RequestMeta {
    /// The target engine principal for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<Principal>,

    /// User identifier supplied by the request context.
    /// Note: This is not verified and should not be used as a trusted identifier.
    /// For example, if triggered by a bot of X platform, this might be the username
    /// of the user interacting with the bot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Extra metadata key-value pairs.
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Json>,
}

impl RequestMeta {
    /// Gets an extra metadata value by key and deserializes it to the specified type.
    pub fn get_extra_as<T>(&self, key: &str) -> Option<T>
    where
        T: DeserializeOwned,
    {
        self.extra
            .get(key)
            .and_then(|value| T::deserialize(value).ok())
    }
}

/// Usage statistics for an agent, model, or tool execution.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    /// Input tokens sent to the LLM.
    pub input_tokens: u64,

    /// Output tokens received from the LLM.
    pub output_tokens: u64,

    /// cached tokens used in the execution.
    #[serde(default)]
    pub cached_tokens: u64,

    /// Number of requests made to models, agents, or tools.
    pub requests: u64,
}

impl Usage {
    /// Accumulates the usage statistics from another usage object.
    pub fn accumulate(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cached_tokens = self.cached_tokens.saturating_add(other.cached_tokens);
        self.requests = self.requests.saturating_add(other.requests);
    }
}

/// Tool call requested by an LLM or returned by a tool execution pipeline.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    /// Tool function name.
    pub name: String,

    /// Tool function arguments.
    pub args: Json,

    /// Tool result populated by the agent runtime when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<ToolOutput<Json>>,

    /// Provider-specific tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,

    /// Remote engine principal that executed the tool, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_id: Option<Principal>,
}

/// Represents a function definition with its metadata.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Function {
    /// Definition of the function.
    pub definition: FunctionDefinition,

    /// Resource tags supported by this function.
    pub supported_resource_tags: Vec<String>,
}

/// Defines a callable function with its metadata and schema.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FunctionDefinition {
    /// Name of the function.
    pub name: String,

    /// Description of what the function does.
    pub description: String,

    /// JSON schema defining the function's parameters.
    pub parameters: Json,

    /// Whether the model should strictly follow the parameter schema when calling the function.
    ///
    /// Provider support and the accepted JSON Schema subset vary by model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl FunctionDefinition {
    /// Modifies the function name with a prefix.
    pub fn name_with_prefix(mut self, prefix: &str) -> Self {
        self.name = format!("{}{}", prefix, self.name);
        self
    }

    /// Normalizes strict parameter schemas before sending them to providers.
    pub fn normalize_strict_parameters(mut self) -> Self {
        if self.strict.unwrap_or_default() {
            self.parameters = normalize_strict_schema(self.parameters);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::resource;

    #[test]
    fn test_agent_and_tool_constructors_default_optional_fields() {
        let agent = AgentInput::new("planner".into(), "summarize this".into());
        assert_eq!(agent.name, "planner");
        assert_eq!(agent.prompt, "summarize this");
        assert!(agent.resources.is_empty());
        assert!(agent.topics.is_none());
        assert!(agent.meta.is_none());

        let tool = ToolInput::new("sum".into(), json!({"x": 1, "y": 2}));
        assert_eq!(tool.name, "sum");
        assert_eq!(tool.args, json!({"x": 1, "y": 2}));
        assert!(tool.resources.is_empty());
        assert!(tool.meta.is_none());

        let output = ToolOutput::new(json!("ok"));
        assert_eq!(output.output, json!("ok"));
        assert!(output.artifacts.is_empty());
        assert_eq!(output.usage.requests, 0);
        assert!(output.tools_usage.is_empty());
    }

    #[test]
    fn test_prompt_command_from_string_variants() {
        assert_eq!(PromptCommand::from("".to_string()), PromptCommand::Ping);
        assert_eq!(
            PromptCommand::from("  /PING  ".to_string()),
            PromptCommand::Ping
        );
        assert_eq!(
            PromptCommand::from("hello".to_string()),
            PromptCommand::Plain {
                prompt: "hello".into(),
            }
        );
        assert_eq!(
            PromptCommand::from("/Status  show details".to_string()),
            PromptCommand::Command {
                command: "status".into(),
                prompt: "/Status  show details".into(),
            }
        );
        assert_eq!(
            PromptCommand::from("/help".to_string()),
            PromptCommand::Command {
                command: "help".into(),
                prompt: "/help".into(),
            }
        );

        // `/ping` resolves to `Ping` regardless of trailing arguments.
        assert_eq!(
            PromptCommand::from("/ping now".to_string()),
            PromptCommand::Ping
        );

        // A slash with no command name is a plain prompt, not an empty command.
        assert_eq!(
            PromptCommand::from("/".to_string()),
            PromptCommand::Plain { prompt: "/".into() }
        );
        assert_eq!(
            PromptCommand::from("/ arg".to_string()),
            PromptCommand::Plain {
                prompt: "/ arg".into(),
            }
        );

        let stop = PromptCommand::from("/stop  停止当前任务，保留会话".to_string());
        assert_eq!(stop.command_argument(), Some("停止当前任务，保留会话"));

        let manual = PromptCommand::Command {
            command: "cancel".into(),
            prompt: "取消当前任务".into(),
        };
        assert_eq!(manual.command_argument(), Some("取消当前任务"));

        assert_eq!(PromptCommand::Ping.command_argument(), None);

        // Command names are ASCII-lowercased, consistent with `command_argument`.
        let command = PromptCommand::from("/ÄRGER now".to_string());
        assert_eq!(
            command,
            PromptCommand::Command {
                command: "Ärger".into(),
                prompt: "/ÄRGER now".into(),
            }
        );
        assert_eq!(command.command_argument(), Some("now"));
    }

    #[test]
    fn test_agent_output_into_tool_output_handles_json_plain_text_and_metadata() {
        let mut tools_usage = HashMap::new();
        tools_usage.insert(
            "sum".into(),
            Usage {
                requests: 1,
                ..Default::default()
            },
        );

        let output = AgentOutput {
            content: r#"{"ok":true}"#.into(),
            usage: Usage {
                input_tokens: 2,
                output_tokens: 1,
                requests: 1,
                ..Default::default()
            },
            tools_usage: tools_usage.clone(),
            artifacts: vec![resource(7, &["text"])],
            ..Default::default()
        }
        .into_tool_output();
        assert_eq!(output.output, json!({"ok": true}));
        assert_eq!(output.artifacts.len(), 1);
        assert_eq!(output.artifacts[0]._id, 7);
        assert_eq!(output.usage.input_tokens, 2);
        assert_eq!(output.tools_usage.get("sum").unwrap().requests, 1);

        let output = AgentOutput {
            content: "not-json".into(),
            thoughts: Some("thinking".into()),
            session: Some("session-1".into()),
            model: Some("test-model".into()),
            ..Default::default()
        }
        .into_tool_output();
        assert_eq!(
            output.output,
            json!({
                "content": "not-json",
                "thoughts": "thinking",
                "session": "session-1",
                "model": "test-model"
            })
        );

        let output = AgentOutput {
            content: "still-not-json".into(),
            ..Default::default()
        }
        .into_tool_output();
        assert_eq!(output.output, json!("still-not-json"));
    }

    #[test]
    fn test_agent_output_into_tool_output_normalizes_blank_failed_reason() {
        // A blank failure reason is neither an error nor serialized metadata.
        let output = AgentOutput {
            content: r#"{"ok":true}"#.into(),
            failed_reason: Some("   ".into()),
            ..Default::default()
        }
        .into_tool_output();
        assert_eq!(output.is_error, None);
        assert_eq!(output.output, json!({"ok": true}));

        // A real failure reason still marks an error and is preserved.
        let output = AgentOutput {
            content: "boom".into(),
            failed_reason: Some("boom".into()),
            ..Default::default()
        }
        .into_tool_output();
        assert_eq!(output.is_error, Some(true));
        assert_eq!(output.output.get("failed_reason").unwrap(), &json!("boom"));
    }

    #[test]
    fn test_request_meta_get_extra_as_and_usage_accumulate() {
        let mut extra = Map::new();
        extra.insert("numbers".into(), json!([1, 2, 3]));
        extra.insert("flag".into(), json!(true));
        let meta = RequestMeta {
            extra,
            ..Default::default()
        };

        assert_eq!(
            meta.get_extra_as::<Vec<u64>>("numbers"),
            Some(vec![1, 2, 3])
        );
        assert_eq!(meta.get_extra_as::<bool>("flag"), Some(true));
        assert_eq!(meta.get_extra_as::<String>("missing"), None);

        let mut usage = Usage {
            input_tokens: u64::MAX - 1,
            output_tokens: 2,
            cached_tokens: 3,
            requests: u64::MAX,
        };
        let other = Usage {
            input_tokens: 10,
            output_tokens: 5,
            cached_tokens: u64::MAX,
            requests: 1,
        };
        usage.accumulate(&other);

        assert_eq!(usage.input_tokens, u64::MAX);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cached_tokens, u64::MAX);
        assert_eq!(usage.requests, u64::MAX);
    }

    #[test]
    fn test_request_meta_extra_flatten_serde() {
        // empty extra should not serialize
        let meta = RequestMeta {
            engine: None,
            user: None,
            extra: Map::new(),
        };
        let v = serde_json::to_value(&meta).unwrap();
        assert_eq!(v, serde_json::json!({}));

        // extra should be flattened into the top-level object
        let mut extra = Map::new();
        extra.insert("foo".into(), serde_json::json!("bar"));
        extra.insert("n".into(), serde_json::json!(1));
        extra.insert("obj".into(), serde_json::json!({"x": true}));

        let meta2 = RequestMeta {
            engine: Some(Principal::from_text("aaaaa-aa").unwrap()),
            user: Some("alice".into()),
            extra,
        };

        let v2 = serde_json::to_value(&meta2).unwrap();
        assert_eq!(v2.get("engine").unwrap(), "aaaaa-aa");
        assert_eq!(v2.get("user").unwrap(), "alice");
        assert_eq!(v2.get("foo").unwrap(), "bar");
        assert_eq!(v2.get("n").unwrap(), 1);
        assert_eq!(v2.get("obj").unwrap(), &serde_json::json!({"x": true}));
        assert!(v2.get("extra").is_none());

        // deserialization: unknown fields go into extra
        let input = serde_json::json!({
            "engine": "aaaaa-aa",
            "user": "bob",
            "k1": "v1",
            "k2": 2,
            "nested": {"a": 1}
        });
        let back: RequestMeta = serde_json::from_value(input).unwrap();
        assert_eq!(back.engine.unwrap().to_text(), "aaaaa-aa");
        assert_eq!(back.user.as_deref(), Some("bob"));
        assert_eq!(back.extra.get("k1").unwrap(), "v1");
        assert_eq!(back.extra.get("k2").unwrap(), 2);
        assert_eq!(
            back.extra.get("nested").unwrap(),
            &serde_json::json!({"a": 1})
        );

        // round-trip (field-by-field)
        let back2: RequestMeta = serde_json::from_value(v2).unwrap();
        assert_eq!(back2.engine.unwrap().to_text(), "aaaaa-aa");
        assert_eq!(back2.user.as_deref(), Some("alice"));
        assert_eq!(back2.extra.get("foo").unwrap(), "bar");
        assert_eq!(back2.extra.get("n").unwrap(), 1);
        assert_eq!(
            back2.extra.get("obj").unwrap(),
            &serde_json::json!({"x": true})
        );
    }

    #[test]
    fn test_function_definition_helpers() {
        let definition = FunctionDefinition {
            name: "search".into(),
            description: "Find documents".into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
        .name_with_prefix("tool_");
        assert_eq!(definition.name, "tool_search");
        assert_eq!(definition.description, "Find documents");
    }
}
