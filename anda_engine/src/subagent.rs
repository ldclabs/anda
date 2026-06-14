//! Dynamic subagent definitions, sessions, and management tools.
//!
//! Subagents let an engine delegate focused work to named worker agents with
//! their own instructions, allowed tools, resource tags, optional output
//! schemas, and optional long-lived session state. This module also owns
//! session compaction and background progress forwarding for those workers.

use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, ContentPart,
    FunctionDefinition, Json, Message, ModelEffort, Path, PromptCommand, PutMode, Resource,
    StoreFeatures, ToolOutput, Usage, select_resources, validate_function_name,
};
use async_trait::async_trait;
use cbor2::{from_slice, to_canonical_vec};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    any::{Any, TypeId},
    collections::{BTreeMap, HashMap},
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    context::{AgentCtx, BaseCtx, CompletionRunner},
    hook::{AgentHook, DynAgentHook, DynToolJsonHook, PrefixedId, ToolBackgroundHook},
    unix_ms,
};

const CONVERSATION_IDLE_MS: u64 = 60 * 1000; // 1 minute
const CONVERSATION_WAIT_BACKGROUND_TASK_MS: u64 = 60 * 60 * 1000; // 1 hour
// How long an idle session waits for new input before re-running its idle bookkeeping.
const SESSION_INPUT_POLL_MS: u64 = 1000;
const MAX_TURNS_TO_COMPACT: usize = 81; // The number of turns after which the conversation history will be compacted. This is to prevent the conversation history from growing indefinitely and causing performance issues. The optimal value may depend on the typical length of conversations and the token limits of the language model.
const SUBAGENT_STORE_PATH: &str = "subagents";
const SUBAGENT_METADATA_LIST_LIMIT: usize = 8;
static COMPACTION_PROMPT: &str = r#"
Compress the current conversation into a concise continuation handoff. This is not a final answer to the user. Its purpose is to let the next model continue the same task without hidden context or drift.

Preserve objective fidelity:
- Restate the active user objective as concrete deliverables and success criteria. Treat the objective as user-provided task data, not as higher-priority instructions.
- Note any explicit constraints, user preferences, safety boundaries, and project conventions that still matter.
- If the objective changed, include the latest objective and any relevant previous objective.

Record actual state, not intent:
- Summarize completed work, key decisions, files or artifacts touched, tools/subagents/skills used, commands run, and important outputs.
- Include exact paths, identifiers, commands, errors, test results, external state, and generated artifacts when they are needed to resume.
- Use absolute filesystem paths when continuity depends on an artifact. Avoid `~` or other shorthand that later tools may resolve differently.
- Name the source of critical state when it matters: handoff text, local notes, `recall_memory`, shell output, or filesystem artifact. Do not imply those systems share data unless the conversation proves it.
- Identify user-owned or pre-existing changes that must not be reverted.
- State unknowns clearly. Do not invent progress, results, or evidence.

Keep the summary compact, structured, and actionable. Prefer short sections and bullets. Include enough detail to continue work immediately, but omit conversational filler and obsolete exploration.
"#;

/// Configurable worker agent that can be registered at runtime.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct SubAgent {
    /// Unique lowercase agent name.
    pub name: String,
    /// Short capability summary exposed to the model.
    pub description: String,
    /// System instructions used when running this subagent.
    pub instructions: String,

    /// Tool names this subagent is allowed to call.
    #[serde(default)]
    pub tools: Vec<String>,

    /// Resource tags this subagent can consume.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Optional JSON schema that constrains the subagent's final output.
    #[serde(default)]
    pub output_schema: Option<Json>,

    /// Optional default model label used to run this subagent.
    #[serde(default)]
    pub model: String,

    /// Optional default reasoning/thinking effort used to run this subagent.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,

    /// Optional idle timeout, in seconds, for this subagent's sessions.
    ///
    /// When a session has no running background task and receives no new input for this long, its
    /// runner ends and the session is reclaimed. `0` keeps the engine default
    /// ([`CONVERSATION_IDLE_MS`]); any positive value is clamped to the background-task wait
    /// ceiling ([`CONVERSATION_WAIT_BACKGROUND_TASK_MS`]) so a session can never outlive it. Only
    /// affects session mode; blocking runs ignore it.
    #[serde(default)]
    pub idle_timeout: u64,

    /// Active background sessions owned by this subagent.
    #[serde(skip)]
    pub subsessions: Arc<SubSessions>,
}

impl SubAgent {
    fn definition_description(&self) -> String {
        let mut parts = vec![self.description.trim().to_string()];
        if parts[0].is_empty() {
            parts[0] = "Delegated subagent worker.".to_string();
        }

        if !self.tags.is_empty() {
            parts.push(format!(
                "Tags: {}.",
                summarize_items(&self.tags, SUBAGENT_METADATA_LIST_LIMIT)
            ));
        }

        if !self.tools.is_empty() {
            parts.push(format!(
                "Allowed tools: {}.",
                summarize_items(&self.tools, SUBAGENT_METADATA_LIST_LIMIT)
            ));
        }

        if self.output_schema.is_some() {
            parts.push("Returns structured output.".to_string());
        }

        if let Some(model) = selected_model_label(&self.model) {
            parts.push(format!("Default model label: {model}."));
        }

        if let Some(effort) = self.effort {
            parts.push(format!("Default effort: {effort}."));
        }

        if self.idle_timeout > 0 {
            parts.push(format!("Session idle timeout: {}s.", self.idle_timeout));
        }

        let sessions = self.subsessions.active_session_ids();
        if !sessions.is_empty() {
            parts.push(format!(
                "Active sessions: {}.",
                summarize_items(&sessions, SUBAGENT_METADATA_LIST_LIMIT)
            ));
        }

        parts.join(" ")
    }
}

fn summarize_items(items: &[String], limit: usize) -> String {
    let mut summary = items
        .iter()
        .take(limit)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");

    if items.len() > limit {
        if !summary.is_empty() {
            summary.push_str(", ");
        }
        summary.push_str(&format!("and {} more", items.len() - limit));
    }

    summary
}

fn selected_model_label(model: &str) -> Option<String> {
    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_ascii_lowercase())
    }
}

/// Resolves a subagent's configured idle timeout (in seconds) into the per-session idle window in
/// milliseconds. `0` keeps the engine default; any positive value is clamped to
/// `[1s, CONVERSATION_WAIT_BACKGROUND_TASK_MS]` so a session can neither expire instantly nor
/// outlive the hard background-task wait ceiling.
fn resolve_idle_timeout_ms(idle_timeout_secs: u64) -> u64 {
    if idle_timeout_secs == 0 {
        CONVERSATION_IDLE_MS
    } else {
        idle_timeout_secs
            .saturating_mul(1000)
            .clamp(1000, CONVERSATION_WAIT_BACKGROUND_TASK_MS)
    }
}

/// Arguments used to run a subagent.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SubAgentArgs {
    /// Task prompt passed to the subagent.
    pub prompt: String,

    /// Optional session ID for non-blocking session mode.
    ///
    /// When this field is empty, the subagent runs normally and returns the final output only after
    /// completion. When a session ID is provided, the subagent runs in the background and returns
    /// immediately with the normalized session ID. Progress and final output are delivered through
    /// [`AgentHook::on_background_progress`] and [`AgentHook::on_background_end`].
    ///
    /// Session mode keeps the subagent conversation alive across calls with the same session ID.
    /// Follow-up prompts, tool results, and background task results are accumulated into that
    /// conversation, allowing the subagent to preserve state across invocations. Session IDs are
    /// case-insensitive and scoped to each subagent. A missing session is created automatically.
    ///
    /// Subagents that need asynchronous tools should use session mode so background tool results can
    /// be fed back into later steps. In normal mode, asynchronous tool results cannot be delivered
    /// back into the completed subagent run.
    #[serde(default)]
    pub session: String,

    /// Optional model label for this subagent run. When empty, the subagent default is used.
    #[serde(default)]
    pub model: String,

    /// Optional reasoning/thinking effort for this subagent run.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,
}

impl SubAgentArgs {
    /// Parses tool-call arguments from the routed prompt string.
    ///
    /// A bare string is a plain blocking prompt. A JSON object whose keys all belong to
    /// [`SubAgentArgs`] must deserialize successfully, so invalid structured arguments surface as
    /// an error instead of silently running with the raw JSON as the prompt. Any other JSON
    /// payload is treated as task data for the subagent.
    fn from_prompt(prompt: String) -> Result<Self, BoxError> {
        if !prompt.trim_start().starts_with('{') {
            return Ok(Self {
                prompt,
                ..Default::default()
            });
        }

        match serde_json::from_str::<Json>(&prompt) {
            Ok(Json::Object(args))
                if args.keys().all(|key| {
                    matches!(key.as_str(), "prompt" | "session" | "model" | "effort")
                }) =>
            {
                serde_json::from_value(Json::Object(args))
                    .map_err(|err| format!("invalid subagent arguments: {err}").into())
            }
            _ => Ok(Self {
                prompt,
                ..Default::default()
            }),
        }
    }
}

/// Arguments accepted by the subagent manager tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubAgentManagerArgs {
    /// Operation to perform. Defaults to creating or updating a subagent.
    #[serde(default = "default_manager_operation")]
    pub operation: String,

    #[serde(default)]
    /// Subagent name to create, update, remove, or inspect.
    pub name: String,

    #[serde(default)]
    /// Short capability summary for the subagent.
    pub description: String,

    #[serde(default)]
    /// System instructions stored on the subagent.
    pub instructions: String,

    #[serde(default)]
    /// Tool names allowed for the subagent.
    pub tools: Vec<String>,

    #[serde(default)]
    /// Resource tags the subagent can consume.
    pub tags: Vec<String>,

    #[serde(default, deserialize_with = "deserialize_optional_json_schema")]
    /// Optional JSON schema for the subagent's final output.
    pub output_schema: Option<Json>,

    /// Optional default model label used to run this subagent.
    #[serde(default)]
    pub model: String,

    /// Optional default reasoning/thinking effort used to run this subagent.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,

    /// Optional idle timeout, in seconds, for this subagent's sessions. `0` keeps the engine
    /// default. See [`SubAgent::idle_timeout`].
    #[serde(default)]
    pub idle_timeout: u64,

    /// Optional task to run immediately after creating or updating the subagent.
    #[serde(default)]
    pub task: String,

    /// Optional session ID passed to the subagent when `task` is provided.
    #[serde(default)]
    pub session: String,

    /// Persist the subagent to storage so it remains available after restart.
    #[serde(default)]
    pub persist: bool,
}

fn default_manager_operation() -> String {
    "upsert".to_string()
}

impl Default for SubAgentManagerArgs {
    fn default() -> Self {
        Self {
            operation: default_manager_operation(),
            name: String::new(),
            description: String::new(),
            instructions: String::new(),
            tools: Vec::new(),
            tags: Vec::new(),
            output_schema: None,
            model: String::new(),
            effort: None,
            idle_timeout: 0,
            task: String::new(),
            session: String::new(),
            persist: false,
        }
    }
}

fn deserialize_optional_json_schema<'de, D>(deserializer: D) -> Result<Option<Json>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<Json>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Json::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                Ok(None)
            } else {
                serde_json::from_str(value)
                    .map(Some)
                    .map_err(serde::de::Error::custom)
            }
        }
        Json::Null => Ok(None),
        value => Ok(Some(value)),
    }
}

fn deserialize_optional_model_effort<'de, D>(
    deserializer: D,
) -> Result<Option<ModelEffort>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<Json>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Json::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                Ok(None)
            } else {
                serde_json::from_value(Json::String(value.to_ascii_lowercase()))
                    .map(Some)
                    .map_err(serde::de::Error::custom)
            }
        }
        Json::Null => Ok(None),
        value => serde_json::from_value(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

impl SubAgentManagerArgs {
    fn from_prompt(prompt: String) -> Result<Self, BoxError> {
        serde_json::from_str::<Self>(&prompt)
            .map_err(|err| format!("subagent manager expects JSON arguments: {err}").into())
    }

    fn into_subagent(self) -> (SubAgent, Option<String>, String, bool) {
        let task = self.task.trim().to_string();
        let task = if task.is_empty() { None } else { Some(task) };
        let session = self.session.trim().to_ascii_lowercase();
        let persist = self.persist;
        let agent = SubAgent {
            name: self.name.trim().to_string(),
            description: self.description,
            instructions: self.instructions,
            tools: self.tools,
            tags: self.tags,
            output_schema: self.output_schema,
            model: self.model.trim().to_string(),
            effort: self.effort,
            idle_timeout: self.idle_timeout,
            ..Default::default()
        };

        (agent, task, session, persist)
    }
}

#[derive(Default, Clone)]
struct SubAgentInput {
    command: PromptCommand,
    resources: Vec<Resource>,
    usage: Usage,
    model: Option<String>,
    effort: Option<ModelEffort>,
}

/// Metadata tracked for a background task running inside a subagent session.
#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct BackgroundTaskInfo {
    /// Subagent that owns the background task.
    pub agent_name: String,
    /// Tool name when the background task was started by a tool call.
    pub tool_name: Option<String>,
    /// Last progress message forwarded to the parent agent.
    pub progress_message: Option<String>,

    /// Cumulative usage already forwarded into the session, used to convert the cumulative
    /// usage carried by background agent outputs into deltas.
    #[serde(default)]
    pub reported_usage: Usage,
}

/// Long-lived conversation session for a subagent.
pub struct SubSession {
    id: String,
    agent: String,
    sender: tokio::sync::mpsc::Sender<SubAgentInput>,
    // task_id -> BackgroundTaskInfo
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
    active_at: AtomicU64,
    // Idle window in ms before an input-less, background-task-less session is reclaimed.
    idle_timeout_ms: u64,
}

fn resources_into_content(resources: Vec<Resource>) -> Vec<ContentPart> {
    resources
        .into_iter()
        .filter_map(|resource| ContentPart::try_from(resource).ok())
        .collect()
}

fn prompt_and_resources_into_content(
    prompt: String,
    resources: &mut Vec<Resource>,
) -> Vec<ContentPart> {
    let mut content = Vec::new();
    if !prompt.is_empty() {
        content.push(prompt.into());
    }
    content.extend(resources_into_content(std::mem::take(resources)));
    content
}

struct SubSessionRunner {
    session: Arc<SubSession>,
    agent_hook: Option<DynAgentHook>,
    runner: CompletionRunner,
    last_output: Option<AgentOutput>,
    /// Artifacts rescued from runners that were replaced during context compaction. They are
    /// merged back into the session's final output.
    carried_artifacts: Vec<Resource>,
    /// Set when the session decided to terminate; the runner then finishes the remaining queued
    /// inputs and exits at the next idle boundary instead of waiting for more input.
    closing: bool,
}

impl SubSessionRunner {
    fn with_session(&self, mut output: AgentOutput) -> AgentOutput {
        if output.session.is_none() {
            output.session = Some(self.session.id.clone());
        }

        output
    }

    fn has_observable_output(output: &AgentOutput) -> bool {
        !output.content.is_empty()
            || output.thoughts.is_some()
            || output.failed_reason.is_some()
            || !output.tool_calls.is_empty()
            || !output.chat_history.is_empty()
            || !output.artifacts.is_empty()
            || output.conversation.is_some()
            || output.model.is_some()
            || output.usage.requests > 0
            || !output.tools_usage.is_empty()
    }

    // Visible signals only. Usage is excluded because carried-over usage from compaction makes
    // it always non-zero, which must not let an empty finalize output shadow visible content.
    fn has_reportable_output(output: &AgentOutput) -> bool {
        !output.content.is_empty()
            || output.thoughts.is_some()
            || output.failed_reason.is_some()
            || !output.tool_calls.is_empty()
            || !output.artifacts.is_empty()
    }

    fn has_progress_signal(output: &AgentOutput) -> bool {
        !output.content.is_empty() || output.failed_reason.is_some()
    }

    fn latest_output(&mut self) -> AgentOutput {
        let output = self
            .last_output
            .take()
            .or_else(|| self.runner.last_output().cloned())
            .unwrap_or_default();

        self.with_session(output)
    }

    fn merge_carried_artifacts(&mut self, output: &mut AgentOutput) {
        if !self.carried_artifacts.is_empty() {
            let mut artifacts = std::mem::take(&mut self.carried_artifacts);
            artifacts.append(&mut output.artifacts);
            output.artifacts = artifacts;
        }
    }

    async fn finalize_output(&mut self) -> AgentOutput {
        let fallback = self.latest_output();

        let mut output = match self.runner.finalize(None).await {
            Ok(output) => {
                let output = self.with_session(output);
                if Self::has_reportable_output(&output) || !Self::has_observable_output(&fallback) {
                    output
                } else {
                    fallback
                }
            }
            Err(err) => {
                if Self::has_observable_output(&fallback) {
                    fallback
                } else {
                    self.with_session(AgentOutput {
                        failed_reason: Some(err.to_string()),
                        ..Default::default()
                    })
                }
            }
        };

        self.merge_carried_artifacts(&mut output);
        output
    }

    fn record_failed_output(&mut self, failed_reason: impl Into<String>) -> String {
        let mut failed_reason = failed_reason.into();
        if failed_reason.trim().is_empty() {
            failed_reason = "subagent session cancelled".to_string();
        }

        let mut output = self.latest_output();
        output.content.clear();
        output.thoughts = None;
        output.failed_reason = Some(failed_reason.clone());

        self.last_output = Some(output);
        failed_reason
    }

    async fn emit_progress(&self, output: AgentOutput) {
        if let Some(hook) = &self.agent_hook {
            hook.on_background_progress(self.runner.ctx(), self.session.id.clone(), output)
                .await;
        }
    }

    /// Summarizes the current conversation into a single handoff message and swaps in a fresh
    /// runner seeded with that summary, discarding the bloated history while preserving the
    /// session's accumulated usage, tool usage, and artifacts.
    ///
    /// Must only be called at an idle boundary (no pending tool calls or queued input), so the
    /// summarization turn does not strand an unanswered tool-call requirement.
    async fn compact(&mut self) -> Result<(), BoxError> {
        // Captured before clearing tools so the replacement runner restores the base toolset.
        let handoff_req = self.runner.req().clone();
        // Drop tools so the summarization turn cannot spawn more tool calls.
        self.runner.set_tools(Vec::new());

        let mut output = match self
            .runner
            .finalize(Some(COMPACTION_PROMPT.to_string()))
            .await
        {
            Ok(output) => output,
            Err(err) => {
                let failed_reason = self.record_failed_output(err.to_string());
                return Err(failed_reason.into());
            }
        };

        if let Some(failed_reason) = output.failed_reason.clone() {
            self.last_output = Some(self.with_session(output));
            return Err(failed_reason.into());
        }

        // The old runner handed over the whole session's accumulated usage/tools_usage/artifacts on
        // finalize; rescue them first so nothing is lost even if the summary turns out unusable.
        let carried_usage = output.usage.clone();
        let carried_tools_usage = output.tools_usage.clone();
        self.carried_artifacts.append(&mut output.artifacts);

        // The summary becomes the sole surviving context. Refuse to replace the whole conversation
        // with an empty message, which would silently erase the task; fail loudly instead so the
        // parent can retry rather than letting the next turn run blind.
        let summary = if !output.content.trim().is_empty() {
            std::mem::take(&mut output.content)
        } else {
            match output.thoughts.as_deref().map(str::trim) {
                Some(thoughts) if !thoughts.is_empty() => thoughts.to_string(),
                _ => {
                    let failed_reason =
                        self.record_failed_output("context compaction produced an empty summary");
                    return Err(failed_reason.into());
                }
            }
        };

        // The summary seeds the next conversation as its first message. It lives in `chat_history`
        // for the first request and migrates into the runner's raw history on later turns.
        let compaction_msg = Message {
            role: "assistant".into(),
            content: vec![summary.into()],
            timestamp: Some(unix_ms()),
            ..Default::default()
        };

        let req = CompletionRequest {
            instructions: handoff_req.instructions,
            role: handoff_req.role,
            chat_history: vec![compaction_msg.clone()],
            tools: handoff_req.tools,
            output_schema: handoff_req.output_schema,
            model: handoff_req.model,
            effort: handoff_req.effort,
            ..Default::default()
        };

        self.runner = self
            .runner
            .ctx()
            .clone()
            .completion_iter(req, Vec::new())
            // Seed the reported chat history too, so the summary survives into the final output.
            .reserve_chat_history(vec![compaction_msg])
            .unbound();
        self.runner.accumulate(&carried_usage);
        self.runner.accumulate_tools_usage(&carried_tools_usage);
        Ok(())
    }

    // returns true if the conversation should continue to be active after processing the inputs, or false if it should be terminated
    async fn run(&mut self, inputs: Vec<SubAgentInput>) -> Result<bool, BoxError> {
        let mut cancellation_requested: Option<String> = None;
        if !inputs.is_empty() {
            self.session.active_at.store(unix_ms(), Ordering::SeqCst);
        }

        for mut input in inputs {
            // 累计来自于后台任务的工具使用情况
            self.runner.accumulate(&input.usage);

            if input.model.is_some() {
                self.runner.set_model(input.model.take());
            }

            if input.effort.is_some() {
                self.runner.set_effort(input.effort);
            }

            match input.command {
                PromptCommand::Ping => {
                    let content =
                        prompt_and_resources_into_content(String::new(), &mut input.resources);
                    if !content.is_empty() {
                        self.runner.follow_up_content(content);
                    }
                    continue;
                }
                PromptCommand::Plain { prompt } => {
                    let content = prompt_and_resources_into_content(prompt, &mut input.resources);
                    if !content.is_empty() {
                        self.runner.follow_up_content(content);
                    }
                }
                PromptCommand::Command { command, prompt } => match command.as_str() {
                    "stop" | "cancel" => {
                        cancellation_requested = Some(prompt);
                        break;
                    }
                    "steer" => {
                        let content =
                            prompt_and_resources_into_content(prompt, &mut input.resources);
                        if !content.is_empty() {
                            self.runner.steer_content(content);
                        }
                    }
                    _ => {
                        let content =
                            prompt_and_resources_into_content(prompt, &mut input.resources);
                        if !content.is_empty() {
                            self.runner.follow_up_content(content);
                        }
                    }
                },
            }
        }

        if let Some(failed_reason) = cancellation_requested {
            let failed_reason = self.record_failed_output(failed_reason);
            return Err(failed_reason.into());
        }

        match self.runner.next().await {
            Ok(None) => {
                if self.closing || self.runner.is_done() {
                    return Ok(false);
                }

                let now_ms = unix_ms();

                let idle = now_ms.saturating_sub(self.session.active_at.load(Ordering::SeqCst));
                let has_background_tasks = !self.session.background_tasks.read().is_empty();

                if (idle > self.session.idle_timeout_ms && !has_background_tasks)
                    || (idle > CONVERSATION_WAIT_BACKGROUND_TASK_MS && has_background_tasks)
                {
                    return Ok(false);
                }

                if needs_compaction(&self.runner) {
                    // 上下文过长，先进行一次压缩总结，用压缩后的 handoff 替换 runner，再继续后续的处理
                    // 压缩只在 idle 边界触发：只有 idle 时才没有 pending tool call，中途压缩会丢掉未应答的 tool 要求。
                    // 但副作用是：一个持续 tool-loop、从不产出非 tool 回复的超长任务，会在到达 idle 前就把 context 撑爆、先撞模型硬上限。
                    self.compact().await?;
                }

                Ok(true)
            }

            Ok(Some(mut res)) => {
                let now_ms = unix_ms();
                self.session.active_at.store(now_ms, Ordering::SeqCst);
                res.session = Some(self.session.id.clone());
                let is_done = self.runner.is_done() || res.failed_reason.is_some();
                self.last_output = Some(res.clone());
                if !is_done && Self::has_progress_signal(&res) {
                    self.emit_progress(res).await;
                }
                Ok(!is_done)
            }

            Err(err) => {
                let failed_reason = self.record_failed_output(err.to_string());
                Err(failed_reason.into())
            }
        }
    }
}

impl SubSession {
    /// Closes the session input side.
    pub fn close(self: Arc<Self>) {
        // no things to do for now
    }

    /// Converts the cumulative usage reported by a background agent into a delta against what was
    /// already forwarded for `task_id`, so the session runner does not double-count usage when it
    /// accumulates progress and final outputs.
    fn take_usage_delta(&self, task_id: &str, current: &Usage, ended: bool) -> Usage {
        let mut tasks = self.background_tasks.write();
        let reported = if ended {
            tasks
                .remove(task_id)
                .map(|info| info.reported_usage)
                .unwrap_or_default()
        } else {
            let info = tasks.entry(task_id.to_string()).or_default();
            let reported = info.reported_usage.clone();
            // Keep the watermark monotonic even if a failure output carries empty usage.
            info.reported_usage = Usage {
                input_tokens: current.input_tokens.max(reported.input_tokens),
                output_tokens: current.output_tokens.max(reported.output_tokens),
                cached_tokens: current.cached_tokens.max(reported.cached_tokens),
                requests: current.requests.max(reported.requests),
            };
            reported
        };

        Usage {
            input_tokens: current.input_tokens.saturating_sub(reported.input_tokens),
            output_tokens: current.output_tokens.saturating_sub(reported.output_tokens),
            cached_tokens: current.cached_tokens.saturating_sub(reported.cached_tokens),
            requests: current.requests.saturating_sub(reported.requests),
        }
    }
}

#[async_trait]
impl AgentHook for SubSession {
    async fn on_background_start(
        &self,
        ctx: &AgentCtx,
        session_id: &str,
        _req: &CompletionRequest,
    ) {
        self.background_tasks.write().insert(
            session_id.to_string(),
            BackgroundTaskInfo {
                agent_name: ctx.base.agent.clone(),
                tool_name: None,
                progress_message: None,
                reported_usage: Usage::default(),
            },
        );
    }

    async fn on_background_progress(
        &self,
        _ctx: &AgentCtx,
        session_id: String,
        output: AgentOutput,
    ) {
        // Background agent outputs carry cumulative usage; forward only the delta.
        let usage = self.take_usage_delta(&session_id, &output.usage, false);
        let prompt = if !output.content.is_empty() {
            format!(
                "Subagent session {session_id} intermediate output:\n\n{}",
                output.content
            )
        } else if let Some(failed_reason) = output.failed_reason {
            format!("Subagent session {session_id} failed with reason: {failed_reason}")
        } else {
            format!("Subagent session {session_id} completed")
        };
        self.sender
            .send(SubAgentInput {
                command: PromptCommand::Plain { prompt },
                resources: vec![],
                usage,
                model: None,
                effort: None,
            })
            .await
            .ok();
    }

    async fn on_background_end(&self, _ctx: &AgentCtx, session_id: String, output: AgentOutput) {
        let usage = self.take_usage_delta(&session_id, &output.usage, true);
        let prompt = if !output.content.is_empty() {
            format!(
                "Subagent session {session_id} final output:\n\n{}",
                output.content
            )
        } else if let Some(failed_reason) = output.failed_reason {
            format!("Subagent session {session_id} failed with reason: {failed_reason}")
        } else {
            format!("Subagent session {session_id} completed")
        };
        self.sender
            .send(SubAgentInput {
                command: PromptCommand::Plain { prompt },
                resources: vec![],
                usage,
                model: None,
                effort: None,
            })
            .await
            .ok();
    }
}

#[async_trait]
impl ToolBackgroundHook for SubSession {
    async fn on_background_start(&self, ctx: &BaseCtx, task_id: &str, _args: Json) {
        let pid = PrefixedId::from_str(task_id).ok();
        self.background_tasks.write().insert(
            task_id.to_string(),
            BackgroundTaskInfo {
                agent_name: ctx.agent.clone(),
                tool_name: pid.map(|p| p.prefix),
                progress_message: None,
                reported_usage: Usage::default(),
            },
        );
    }

    async fn on_background_progress(
        &self,
        _ctx: &BaseCtx,
        task_id: String,
        output: ToolOutput<Json>,
    ) {
        let mut tasks = self.background_tasks.write();
        if let Some(info) = tasks.get_mut(&task_id) {
            info.progress_message = serde_json::to_string(&output.output).ok();
        }
    }

    async fn on_background_end(&self, _ctx: &BaseCtx, task_id: String, output: ToolOutput<Json>) {
        {
            self.background_tasks.write().remove(&task_id);
        }

        self.sender
            .send(SubAgentInput {
                command: PromptCommand::Plain {
                    prompt: format!(
                        "Background task {task_id} completed:\n\n{}",
                        serde_json::to_string(&output.output).unwrap_or_default()
                    ),
                },
                usage: output.usage,
                resources: output.artifacts,
                model: None,
                effort: None,
            })
            .await
            .ok();
    }
}

/// Registry of active subagent sessions for one subagent definition.
pub struct SubSessions {
    sessions: RwLock<BTreeMap<String, Arc<SubSession>>>,
}

impl Default for SubSessions {
    fn default() -> Self {
        Self {
            sessions: RwLock::new(BTreeMap::new()),
        }
    }
}

impl SubSessions {
    /// Inserts or replaces a session by ID.
    pub fn insert_session(&self, sess: Arc<SubSession>) {
        self.sessions.write().insert(sess.id.clone(), sess);
    }

    /// Atomically claims the session ID for `sess`.
    ///
    /// Returns `None` when `sess` was inserted, or `Some(existing)` when another active session
    /// already owns the ID, so concurrent callers join the same conversation instead of spawning
    /// duplicate runners.
    pub fn try_insert_session(&self, sess: Arc<SubSession>) -> Option<Arc<SubSession>> {
        let mut sessions = self.sessions.write();
        if let Some(existing) = sessions.get(&sess.id)
            && !existing.sender.is_closed()
        {
            return Some(existing.clone());
        }

        sessions.insert(sess.id.clone(), sess);
        None
    }

    /// Removes the session only if the registry still holds this exact instance, so a finished
    /// runner cannot remove a newer session that reused the same ID.
    pub fn remove_session_if(&self, sess: &Arc<SubSession>) {
        let removed = {
            let mut sessions = self.sessions.write();
            match sessions.get(&sess.id) {
                Some(existing) if Arc::ptr_eq(existing, sess) => sessions.remove(&sess.id),
                _ => None,
            }
        };

        if let Some(removed) = removed {
            removed.close();
        }
    }

    /// Returns IDs for sessions whose runners are still active.
    pub fn active_session_ids(&self) -> Vec<String> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed());
        sessions.keys().cloned().collect()
    }

    /// Returns an active session by ID.
    pub fn get_session(&self, id: &str) -> Option<Arc<SubSession>> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed());
        sessions.get(id).cloned()
    }

    /// Removes a session by ID and closes its input channel.
    pub fn remove_session(&self, id: &str) {
        if let Some(subsession) = self.sessions.write().remove(id) {
            subsession.close();
        }
    }
}

impl Agent<AgentCtx> for SubAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.definition_description(),
            parameters: json!({
                "type": "object",
                "description": "Run this subagent as a focused worker process. The caller acts as the scheduler: use blocking mode for short one-shot work, or session mode for long-running, parallel, asynchronous, or follow-up work. Keep each prompt as a self-contained handoff.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Self-contained task handoff for this subagent. Include objective, context/resources, constraints, dependencies, expected deliverable, success criteria, and what progress/final output should contain. To control an already-running session, send a control command here instead of a task: `/steer <guidance>` to adjust course mid-run, or `/stop <reason>` to end it."
                    },
                    "session": {
                        "type": "string",
                        "description": "Optional case-insensitive session ID. Leave empty for blocking one-shot work. Provide a stable ID for non-blocking, parallel, asynchronous, or follow-up work; reuse it to continue the same conversation.",
                        "default": ""
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model label for this subagent run. Leave empty to use the subagent default model or the caller context model.",
                        "default": ""
                    },
                    "effort": {
                        "type": ["string", "null"],
                        "enum": ["minimal", "low", "medium", "high", "max", null],
                        "description": "Optional reasoning/thinking effort for this subagent run. Use null to keep the subagent default.",
                        "default": null
                    },
                },
                "required": ["prompt", "session", "model", "effort"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.tools.clone()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.tags.clone()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let agent_hook = ctx.base.get_state::<DynAgentHook>();

        let (prompt, resources) = if let Some(hook) = &agent_hook {
            hook.before_agent_run(&ctx, prompt, resources).await?
        } else {
            (prompt, resources)
        };

        let args = SubAgentArgs::from_prompt(prompt)?;
        let model = selected_model_label(&args.model).or_else(|| selected_model_label(&self.model));
        let effort = args.effort.or(self.effort);

        let session_id = args.session.trim().to_ascii_lowercase();
        if session_id.is_empty() {
            if args.prompt.trim().is_empty() && resources.is_empty() {
                return Err("prompt cannot be empty".into());
            }

            let rt = ctx
                .completion(
                    CompletionRequest {
                        instructions: self.instructions.clone(),
                        prompt: args.prompt,
                        content: resources_into_content(resources),
                        tools: ctx.definitions(Some(&self.tools)).await,
                        output_schema: self.output_schema.clone(),
                        model,
                        effort,
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .await?;

            if let Some(hook) = &agent_hook {
                return hook.after_agent_run(&ctx, rt).await;
            }
            return Ok(rt);
        }

        let agent = self.name();
        let mut input = SubAgentInput {
            command: PromptCommand::from(args.prompt),
            resources,
            usage: Usage::default(),
            model,
            effort,
        };

        let subsessions = self.subsessions.clone();
        // Join the active session when one exists, otherwise atomically claim the session ID.
        // The bounded loop resolves races with concurrent callers using the same session ID, so
        // two callers can never spawn duplicate runners for one session.
        let mut claimed: Option<(Arc<SubSession>, tokio::sync::mpsc::Receiver<SubAgentInput>)> =
            None;
        for _ in 0..8 {
            if let Some(session) = subsessions.get_session(&session_id) {
                // Join existing conversation session if it's active
                match session.sender.send(input).await {
                    Ok(_) => {
                        let rt = AgentOutput {
                            content: format!(
                                "prompt queued for subagent {} session {}. Progress and final output will be pushed through the hooks.",
                                session.agent, session_id
                            ),
                            session: Some(session_id.clone()),
                            ..Default::default()
                        };
                        if let Some(hook) = &agent_hook {
                            return hook.after_agent_run(&ctx, rt).await;
                        }
                        return Ok(rt);
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to enqueue prompt for subagent {} session {}: receiver closed",
                            session.agent,
                            session_id,
                        );
                        subsessions.remove_session_if(&session);
                        input = err.0;
                        continue;
                    }
                }
            }

            // No active session, so control commands have nothing to act on. Report the session
            // state instead of starting a new session or returning a misleading error.
            let inactive_op = match &input.command {
                PromptCommand::Ping if input.resources.is_empty() => Some("ping"),
                PromptCommand::Command { command, .. }
                    if matches!(command.as_str(), "stop" | "cancel") =>
                {
                    Some("cancel")
                }
                _ => None,
            };
            if let Some(op) = inactive_op {
                let rt = AgentOutput {
                    content: format!(
                        "subagent {agent} session {session_id} is not active (it may have finished or expired); nothing to {op}. Call again with a non-empty prompt to start a new session."
                    ),
                    session: Some(session_id.clone()),
                    ..Default::default()
                };
                if let Some(hook) = &agent_hook {
                    return hook.after_agent_run(&ctx, rt).await;
                }
                return Ok(rt);
            }

            let (sender, rx) = tokio::sync::mpsc::channel::<SubAgentInput>(42);
            let candidate = Arc::new(SubSession {
                id: session_id.clone(),
                agent: agent.clone(),
                sender,
                background_tasks: Arc::new(RwLock::new(HashMap::new())),
                active_at: AtomicU64::new(unix_ms()),
                idle_timeout_ms: resolve_idle_timeout_ms(self.idle_timeout),
            });
            match subsessions.try_insert_session(candidate.clone()) {
                None => {
                    claimed = Some((candidate, rx));
                    break;
                }
                // Lost the claim to a concurrent caller; join that session on the next pass.
                Some(_) => continue,
            }
        }

        let Some((session, mut rx)) = claimed else {
            return Err(format!(
                "subagent {agent} session {session_id} is restarting concurrently, please retry"
            )
            .into());
        };

        // The session ID is claimed; start a new session runner with this prompt.
        let SubAgentInput {
            command,
            resources,
            model: input_model,
            effort: input_effort,
            ..
        } = input;

        let prompt = match command {
            // Empty pings and stop/cancel were answered above; a resource-only ping starts the
            // session with the resources as content.
            PromptCommand::Ping => String::new(),
            PromptCommand::Plain { prompt } => prompt,
            PromptCommand::Command { prompt, .. } => prompt,
        };

        let rt = AgentOutput {
            content: format!(
                "subagent {} is running in the background with session mode (session: {}). The output will be pushed to you through the hooks.",
                session.agent, session.id
            ),
            session: Some(session.id.clone()),
            ..Default::default()
        };

        let req = CompletionRequest {
            instructions: self.instructions.clone(),
            prompt,
            content: resources_into_content(resources),
            tools: ctx.definitions(Some(&self.tools)).await,
            output_schema: self.output_schema.clone(),
            model: input_model,
            effort: input_effort,
            ..Default::default()
        };

        let rt = if let Some(hook) = &agent_hook {
            match hook.after_agent_run(&ctx, rt).await {
                Ok(rt) => rt,
                Err(err) => {
                    // The session never started; release the claimed session ID.
                    subsessions.remove_session_if(&session);
                    return Err(err);
                }
            }
        } else {
            rt
        };

        ctx.base.set_state(DynAgentHook::new(session.clone()));
        ctx.base.set_state(DynToolJsonHook::new(session.clone()));

        if let Some(hook) = &agent_hook {
            hook.on_background_start(&ctx, &session.id, &req).await;
        }

        let runner = ctx.clone().completion_iter(req, vec![]).unbound();
        tokio::spawn(async move {
            let mut runner = SubSessionRunner {
                session: session.clone(),
                agent_hook,
                runner,
                last_output: None,
                carried_artifacts: Vec::new(),
                closing: false,
            };

            let mut pending: Vec<SubAgentInput> = Vec::new();
            loop {
                let mut inputs = std::mem::take(&mut pending);
                while let Ok(input) = rx.try_recv() {
                    inputs.push(input);
                }

                if inputs.is_empty() && !runner.closing && runner.runner.is_idle() {
                    // Wait for input so new prompts are processed without polling latency; the
                    // timeout keeps the idle-timeout bookkeeping in `run` ticking.
                    match tokio::time::timeout(
                        std::time::Duration::from_millis(SESSION_INPUT_POLL_MS),
                        rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(input)) => {
                            inputs.push(input);
                            while let Ok(input) = rx.try_recv() {
                                inputs.push(input);
                            }
                        }
                        // The channel cannot close while the session holds its own sender.
                        Ok(None) => {}
                        Err(_) => {}
                    }
                }

                match runner.run(inputs).await {
                    Ok(true) => {
                        // continue the subsession
                    }
                    Ok(false) => {
                        if !runner.closing {
                            runner.closing = true;
                            // Stop accepting new prompts, then finish anything that was queued
                            // (and already acknowledged to callers) before the channel closed.
                            rx.close();
                            while let Ok(input) = rx.try_recv() {
                                pending.push(input);
                            }
                            if !pending.is_empty() {
                                continue;
                            }
                        }

                        let output = runner.finalize_output().await;
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(runner.runner.ctx(), session.id.clone(), output)
                                .await;
                        }
                        break;
                    }
                    Err(err) => {
                        let mut output = runner.latest_output();
                        runner.merge_carried_artifacts(&mut output);
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(runner.runner.ctx(), session.id.clone(), output)
                                .await;
                        }
                        log::error!("Error processing session {}: {:?}", session.id, err);
                        break;
                    }
                }
            }

            subsessions.remove_session_if(&session);
        });

        Ok(rt)
    }
}

/// Object-safe registry interface for groups of subagents.
pub trait SubAgentSet: Send + Sync {
    /// Converts the registry into [`Any`] for downcasting.
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Checks if a subagent with the given lowercase name exists.
    fn contains_lowercase(&self, lowercase_name: &str) -> bool;

    /// Retrieves a subagent by lowercase name.
    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent>;

    /// Returns definitions for all or specified agents.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// - Vec<[`FunctionDefinition`]>: Vector of agent definitions.
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Selects and returns resources relevant to the specified subagent name from the provided list.
    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource>;
}

/// Tool and registry for creating, updating, loading, and running subagents.
pub struct SubAgentManager {
    agents: RwLock<BTreeMap<String, SubAgent>>,
    models: Vec<String>,
}

impl Default for SubAgentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentManager {
    /// Function name used when registering the manager tool.
    pub const NAME: &'static str = "subagents_manager";

    /// Creates an empty subagent manager.
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(BTreeMap::new()),
            models: Vec::new(),
        }
    }

    /// Sets the model labels allowed for managed subagents.
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        self.models = models;
        self
    }

    fn store_prefix() -> Path {
        Path::from(SUBAGENT_STORE_PATH)
    }

    fn store_path(name: &str) -> Path {
        Path::from(format!("{SUBAGENT_STORE_PATH}/{name}"))
    }

    /// Loads persisted subagents from engine storage.
    pub async fn load(&self, ctx: AgentCtx) -> Result<(), BoxError> {
        let offset = Path::from("");
        let prefix = Self::store_prefix();
        let agents = match ctx.root.store_list(Some(&prefix), &offset).await {
            Ok(agents) => agents,
            Err(err) => {
                log::warn!("failed to list persisted subagents: {err}");
                return Ok(());
            }
        };

        // One corrupted or unreadable entry must not prevent the other subagents from loading.
        for meta in agents {
            let data = match ctx.root.store_get(&meta.location).await {
                Ok((data, _)) => data,
                Err(err) => {
                    log::warn!("failed to read persisted subagent {}: {err}", meta.location);
                    continue;
                }
            };

            match from_slice::<SubAgent>(&data[..]) {
                Ok(mut agent) => {
                    let name = agent.name.to_ascii_lowercase();
                    self.preserve_runtime_state(&name, &mut agent);
                    self.agents.write().insert(name, agent);
                }
                Err(err) => {
                    log::warn!(
                        "failed to decode persisted subagent {}: {err}",
                        meta.location
                    );
                }
            }
        }

        Ok(())
    }

    fn preserve_runtime_state(&self, name: &str, agent: &mut SubAgent) {
        if let Some(existing) = self.agents.read().get(name) {
            agent.subsessions = existing.subsessions.clone();
        }
    }

    fn catalog(&self) -> Json {
        let agents = self.agents.read().values().cloned().collect::<Vec<_>>();
        let subagents = agents
            .into_iter()
            .map(|agent| {
                let name = agent.name.to_ascii_lowercase();
                let callable = format!("SA_{name}");
                let has_output_schema = agent.output_schema.is_some();
                let active_sessions = agent.subsessions.active_session_ids();
                let model = selected_model_label(&agent.model);
                json!({
                    "name": name,
                    "callable": callable,
                    "description": agent.description,
                    "tools": agent.tools,
                    "tags": agent.tags,
                    "has_output_schema": has_output_schema,
                    "model": model,
                    "effort": agent.effort,
                    "idle_timeout": agent.idle_timeout,
                    "active_sessions": active_sessions,
                })
            })
            .collect::<Vec<_>>();

        json!({
            "result": "listed",
            "count": subagents.len(),
            "subagents": subagents,
            "hint": "Use SA_<name> for delegated work. Use a stable session ID for long-running, parallel, asynchronous, or follow-up tasks."
        })
    }

    /// Creates or updates a subagent. The name is normalised to lowercase and validated. If an agent with the same name exists, it will be overwritten.
    pub async fn upsert(&self, ctx: AgentCtx, mut agent: SubAgent) -> Result<(), BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;
        self.preserve_runtime_state(&name, &mut agent);

        let data = to_canonical_vec(&agent)?;
        self.agents.write().insert(name.clone(), agent);

        ctx.root
            .store_put(&Self::store_path(&name), PutMode::Overwrite, data.into())
            .await?;
        Ok(())
    }

    /// Creates or updates an in-memory subagent without writing it to the store.
    pub fn upsert_temporary(&self, mut agent: SubAgent) -> Result<String, BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;
        self.preserve_runtime_state(&name, &mut agent);

        self.agents.write().insert(name.clone(), agent);
        Ok(name)
    }

    fn description_text(&self) -> String {
        if self.models.is_empty() {
            "Scheduler control plane for reusable subagents. Use it to list available workers, create or update focused helpers with stable instructions and restricted toolsets, optionally run an initial delegated task, and optionally persist useful helpers for future sessions and restarts. Temporary subagents are callable immediately as `SA_<name>`.".to_string()
        } else {
            format!(
                "Scheduler control plane for reusable subagents with model-aware routing. Use it to list available workers, create or update focused helpers with stable instructions and restricted toolsets, optionally run an initial delegated task, and optionally persist useful helpers for future sessions and restarts. Temporary subagents are callable immediately as `SA_<name>`. This manager supports the following models for routing decisions: {}.",
                self.models.join(", ")
            )
        }
    }

    fn manager_definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: Self::NAME.to_string(),
            description: self.description_text(),
            parameters: json!({
                "type": "object",
                "description": "List the subagent registry, or create/update a subagent configuration, optionally run it immediately, and optionally persist it for future reuse.",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["upsert", "list"],
                        "description": "Use `list` to inspect registered subagents and active sessions. Use `upsert` to create or update a worker and optionally run a delegated task.",
                        "default": "upsert"
                    },
                    "name": {
                        "type": "string",
                        "description": "For operation=upsert, the unique callable subagent name. Must be lowercase snake_case, start with a letter, contain only letters, digits, or underscores, and be no longer than 64 characters. The subagent becomes callable as SA_<name>. For operation=list, use an empty string."
                    },
                    "description": {
                        "type": "string",
                        "description": "For operation=upsert, the routing description shown when models decide whether to call this subagent. State when it should be used and what outcome it produces. For operation=list, use an empty string."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "For operation=upsert, durable system-style instructions for the subagent. Define its role, scope, workflow, constraints, decision rules, and expected output style. Write reusable guidance, not a one-off task prompt. For operation=list, use an empty string."
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional whitelist of tool names the subagent may use. Include only the minimum tools it needs. Leave empty to create a no-tool subagent.",
                        "default": []
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of resource tags relevant to this subagent, such as 'image', 'text', or 'audio'. Resources with matching tags are processed when the subagent is called.",
                        "default": []
                    },
                    "output_schema": {
                        "type": ["string", "null"],
                        "description": "Optional JSON schema encoded as a JSON string that the subagent's output must conform to. Use null for unstructured text output.",
                        "default": null
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional default model label used to run this subagent. Leave empty to use the caller context model. For operation=list, use an empty string.",
                        "default": ""
                    },
                    "effort": {
                        "type": ["string", "null"],
                        "enum": ["minimal", "low", "medium", "high", "max", null],
                        "description": "Optional default reasoning/thinking effort used to run this subagent. Use null to leave the selected model's default effort unchanged. For operation=list, use null.",
                        "default": null
                    },
                    "idle_timeout": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional idle timeout in seconds for this subagent's sessions. A session with no running background task ends after this much inactivity. Use 0 to keep the engine default; larger values are capped at the background-task wait ceiling. Tune it up for sessions you will revisit after gaps, down to reclaim idle workers sooner. For operation=list, use 0.",
                        "default": 0
                    },
                    "task": {
                        "type": "string",
                        "description": "Optional immediate task handoff to run with the newly created or updated subagent. Include objective, context/resources, constraints, dependencies, expected deliverable, and success criteria. Leave empty to only create/update or when operation=list.",
                        "default": ""
                    },
                    "session": {
                        "type": "string",
                        "description": "Optional session ID for the immediate task. Leave empty for blocking one-shot mode. Provide a stable ID for non-blocking, parallel, asynchronous, or follow-up work with hook-delivered progress and final output.",
                        "default": ""
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Set true to save or update this subagent for future calls and restarts. Leave false to keep it temporary in the current engine process.",
                        "default": false
                    }
                },
                "required": ["operation", "name", "description", "instructions", "tools", "tags", "output_schema", "model", "effort", "idle_timeout", "task", "session", "persist"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn configure(
        &self,
        ctx: AgentCtx,
        args: SubAgentManagerArgs,
    ) -> Result<(String, SubAgent, Option<String>, String, bool), BoxError> {
        let (mut agent, task, session, persist) = args.into_subagent();
        let name = agent.name.to_ascii_lowercase();
        self.preserve_runtime_state(&name, &mut agent);

        if persist {
            self.upsert(ctx, agent.clone()).await?;
        } else {
            self.upsert_temporary(agent.clone())?;
        }

        Ok((name, agent, task, session, persist))
    }
}

impl Agent<AgentCtx> for SubAgentManager {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description_text()
    }

    fn definition(&self) -> FunctionDefinition {
        self.manager_definition()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        vec!["*".to_string()]
    }

    async fn init(&self, ctx: AgentCtx) -> Result<(), BoxError> {
        self.load(ctx).await
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let args = SubAgentManagerArgs::from_prompt(prompt)?;
        let operation = args.operation.trim().to_ascii_lowercase();
        if matches!(operation.as_str(), "list" | "status" | "catalog") {
            return Ok(AgentOutput {
                content: self.catalog().to_string(),
                ..Default::default()
            });
        }

        if !matches!(operation.as_str(), "" | "upsert" | "create" | "update") {
            return Err(format!("unsupported subagent manager operation: {operation}").into());
        }

        let (name, agent, task, session, persist) = self.configure(ctx.clone(), args).await?;
        let callable = format!("SA_{name}");
        let subagent = json!({
            "result": if persist { "persisted" } else { "created" },
            "name": name,
            "callable": callable,
            "persisted": persist,
            "model": selected_model_label(&agent.model),
            "effort": agent.effort,
            "active_sessions": agent.subsessions.active_session_ids(),
            "hint": "Call the subagent by this callable name. Use a stable session ID for long-running, parallel, asynchronous, or follow-up tasks. If a temporary subagent proves useful, call subagents_manager again with persist=true to save it."
        });
        let Some(task) = task else {
            return Ok(AgentOutput {
                content: subagent.to_string(),
                ..Default::default()
            });
        };

        let prompt = serde_json::to_string(&SubAgentArgs {
            prompt: task,
            session,
            model: String::new(),
            effort: None,
        })?;

        match agent.run(ctx.child(&name, &name)?, prompt, resources).await {
            Ok(mut rt) => {
                rt.content = json!({
                    "subagent": subagent,
                    "output": rt.content,
                })
                .to_string();
                Ok(rt)
            }
            Err(err) => Ok(AgentOutput {
                content: subagent.to_string(),
                failed_reason: Some(format!("Subagent run error: {err}")),
                ..Default::default()
            }),
        }
    }
}

impl SubAgentSet for SubAgentManager {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.agents.read().contains_key(lowercase_name)
    }

    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent> {
        self.agents.read().get(lowercase_name).cloned()
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        match names {
            None => self
                .agents
                .read()
                .values()
                .map(|agent| agent.definition())
                .collect(),
            Some(names) => {
                let agents = self.agents.read();
                names
                    .iter()
                    .filter_map(|name| {
                        agents
                            .get(&name.to_ascii_lowercase())
                            .map(|agent| agent.definition())
                    })
                    .collect()
            }
        }
    }

    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        self.agents
            .read()
            .get(&name.to_ascii_lowercase())
            .map(|agent| {
                let supported_tags = agent.supported_resource_tags();
                select_resources(resources, &supported_tags)
            })
            .unwrap_or_default()
    }
}

/// Type-indexed collection of subagent registries.
pub struct SubAgentSetManager {
    sets: RwLock<BTreeMap<TypeId, Arc<dyn SubAgentSet>>>,
}

impl Default for SubAgentSetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentSetManager {
    /// Creates an empty collection of subagent registries.
    pub fn new() -> Self {
        Self {
            sets: RwLock::new(BTreeMap::new()),
        }
    }

    /// Inserts a typed subagent registry and returns the previous registry of the same type.
    pub fn insert<T: SubAgentSet + Sized + 'static>(&self, set: Arc<T>) -> Option<Arc<T>> {
        let type_id = TypeId::of::<T>();
        self.sets
            .write()
            .insert(type_id, set)
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
    }

    /// Returns a typed subagent registry when one has been inserted.
    pub fn get<T: SubAgentSet + Sized + 'static>(&self) -> Option<Arc<T>> {
        let type_id = TypeId::of::<T>();
        self.sets
            .read()
            .get(&type_id)
            .and_then(|boxed| boxed.clone().into_any().downcast::<T>().ok())
    }
}

impl SubAgentSet for SubAgentSetManager {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.sets
            .read()
            .values()
            .any(|set| set.contains_lowercase(lowercase_name))
    }

    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent> {
        for set in self.sets.read().values() {
            if let Some(agent) = set.get_lowercase(lowercase_name) {
                return Some(agent);
            }
        }
        None
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        self.sets
            .read()
            .values()
            .flat_map(|set| set.definitions(names))
            .collect()
    }

    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        for set in self.sets.read().values() {
            let selected = set.select_resources(name, resources);
            if !selected.is_empty() {
                return selected;
            }
        }

        Vec::new()
    }
}

/// Returns true when a session runner should compact its conversation history.
pub fn needs_compaction(runner: &CompletionRunner) -> bool {
    let current_usage = runner.current_usage();
    let context_window = runner.model().context_window as u64;
    let threshold = if context_window == 0 {
        100_000
    } else {
        context_window.saturating_mul(8).saturating_div(10).max(1)
    };

    current_usage.input_tokens >= threshold || runner.turns() >= MAX_TURNS_TO_COMPACT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use crate::model::{CompletionFeaturesDyn, Model, Models};
    use anda_core::BoxPinFut;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;

    #[derive(Clone, Debug)]
    struct EchoCompleter;

    impl CompletionFeaturesDyn for EchoCompleter {
        fn model_name(&self) -> String {
            "echo".to_string()
        }

        fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            let content = request_text(&req);
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct UsageCompleter {
        input_tokens: u64,
    }

    impl CompletionFeaturesDyn for UsageCompleter {
        fn model_name(&self) -> String {
            "usage".to_string()
        }

        fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            let content = request_text(&req);
            let input_tokens = self.input_tokens;
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content,
                usage: Usage {
                    input_tokens,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct RecordingCompactionCompleter {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for RecordingCompactionCompleter {
        fn model_name(&self) -> String {
            "recording-compaction".to_string()
        }

        fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            self.requests.lock().push(req.clone());

            let prompt = request_text(&req);
            let history = req
                .chat_history
                .iter()
                .filter_map(Message::text)
                .collect::<Vec<_>>()
                .join(" | ");

            let content = if prompt.trim() == COMPACTION_PROMPT.trim() {
                "compacted handoff".to_string()
            } else if history.is_empty() {
                prompt.clone()
            } else {
                format!("history={history}; input={prompt}")
            };

            let input_tokens = if prompt.trim() == COMPACTION_PROMPT.trim() {
                1
            } else {
                100_000
            };

            Box::pin(futures::future::ready(Ok(AgentOutput {
                content,
                usage: Usage {
                    input_tokens,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct EmptyCompactionCompleter;

    impl CompletionFeaturesDyn for EmptyCompactionCompleter {
        fn model_name(&self) -> String {
            "empty-compaction".to_string()
        }

        fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            let prompt = request_text(&req);
            let is_compaction = prompt.trim() == COMPACTION_PROMPT.trim();
            Box::pin(futures::future::ready(Ok(AgentOutput {
                // Normal turns echo their input and push usage over the compaction threshold; the
                // compaction turn returns no usable summary at all.
                content: if is_compaction { String::new() } else { prompt },
                usage: Usage {
                    input_tokens: if is_compaction { 1 } else { 100_000 },
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct RecordingRequestCompleter {
        name: &'static str,
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for RecordingRequestCompleter {
        fn model_name(&self) -> String {
            self.name.to_string()
        }

        fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            let content = request_text(&req);
            self.requests.lock().push(req);
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

    #[derive(Clone, Debug)]
    struct ErrorCompleter;

    impl CompletionFeaturesDyn for ErrorCompleter {
        fn model_name(&self) -> String {
            "error".to_string()
        }

        fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Err("model failed".into())))
        }
    }

    #[derive(Clone, Debug)]
    struct ToolCallProgressCompleter;

    impl CompletionFeaturesDyn for ToolCallProgressCompleter {
        fn model_name(&self) -> String {
            "tool-call-progress".to_string()
        }

        fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![anda_core::ToolCall {
                    name: "long_running_tool".to_string(),
                    args: json!({}),
                    result: None,
                    call_id: Some("call-1".to_string()),
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

    // Emits visible narration alongside a tool call, so the step carries a progress signal while
    // tool calls keep the runner busy (non-idle).
    #[derive(Clone, Debug)]
    struct NarratingToolCallCompleter;

    impl CompletionFeaturesDyn for NarratingToolCallCompleter {
        fn model_name(&self) -> String {
            "narrating-tool-call".to_string()
        }

        fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "searching now".to_string(),
                tool_calls: vec![anda_core::ToolCall {
                    name: "long_running_tool".to_string(),
                    args: json!({}),
                    result: None,
                    call_id: Some("call-1".to_string()),
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

    #[derive(Clone, Default)]
    struct RecordingAgentHook {
        progress: Arc<Mutex<Vec<(String, AgentOutput)>>>,
    }

    impl RecordingAgentHook {
        fn progress_events(&self) -> Vec<(String, AgentOutput)> {
            self.progress.lock().clone()
        }
    }

    #[derive(Clone, Default)]
    struct FailingAfterAgentHook {
        starts: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Default)]
    struct EmptySubAgentSet;

    impl SubAgentSet for EmptySubAgentSet {
        fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
            self
        }

        fn contains_lowercase(&self, _lowercase_name: &str) -> bool {
            false
        }

        fn get_lowercase(&self, _lowercase_name: &str) -> Option<SubAgent> {
            None
        }

        fn definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
            Vec::new()
        }

        fn select_resources(&self, _name: &str, _resources: &mut Vec<Resource>) -> Vec<Resource> {
            Vec::new()
        }
    }

    #[async_trait]
    impl AgentHook for FailingAfterAgentHook {
        async fn after_agent_run(
            &self,
            _ctx: &AgentCtx,
            _output: AgentOutput,
        ) -> Result<AgentOutput, BoxError> {
            Err("after hook rejected output".into())
        }

        async fn on_background_start(
            &self,
            _ctx: &AgentCtx,
            session_id: &str,
            _req: &CompletionRequest,
        ) {
            self.starts.lock().push(session_id.to_string());
        }
    }

    #[async_trait]
    impl AgentHook for RecordingAgentHook {
        async fn on_background_progress(
            &self,
            _ctx: &AgentCtx,
            session_id: String,
            output: AgentOutput,
        ) {
            self.progress.lock().push((session_id, output));
        }
    }

    fn request_text(req: &CompletionRequest) -> String {
        if !req.prompt.is_empty() {
            return req.prompt.clone();
        }

        Message {
            content: req.content.clone(),
            ..Default::default()
        }
        .text()
        .unwrap_or_default()
    }

    async fn recv_subagent_prompt(rx: &mut tokio::sync::mpsc::Receiver<SubAgentInput>) -> String {
        match rx.recv().await.unwrap().command {
            PromptCommand::Plain { prompt } => prompt,
            other => panic!("unexpected command: {other:?}"),
        }
    }

    fn resource(id: u64, tags: &[&str]) -> Resource {
        Resource {
            _id: id,
            name: format!("resource-{id}"),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn subagent_metadata_args_and_content_helpers_cover_edge_inputs() {
        assert_eq!(
            summarize_items(&["a".to_string(), "b".to_string(), "c".to_string()], 1),
            "a, and 2 more"
        );
        assert_eq!(selected_model_label("  Pro "), Some("pro".to_string()));
        assert_eq!(selected_model_label("  "), None);

        let agent = SubAgent {
            name: "meta_worker".to_string(),
            description: "  ".to_string(),
            tools: (0..10).map(|idx| format!("tool_{idx}")).collect(),
            tags: (0..10).map(|idx| format!("tag_{idx}")).collect(),
            output_schema: Some(json!({"type": "object"})),
            model: " Pro ".to_string(),
            effort: Some(ModelEffort::Max),
            idle_timeout: 120,
            ..Default::default()
        };
        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        agent.subsessions.insert_session(Arc::new(SubSession {
            id: "Job-A".to_string(),
            agent: "meta_worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        }));
        let definition = agent.definition();
        assert!(
            definition
                .description
                .starts_with("Delegated subagent worker.")
        );
        assert!(definition.description.contains("Tags: tag_0"));
        assert!(definition.description.contains("and 2 more"));
        assert!(
            definition
                .description
                .contains("Returns structured output.")
        );
        assert!(definition.description.contains("Default model label: pro."));
        assert!(definition.description.contains("Default effort: max."));
        assert!(
            definition
                .description
                .contains("Session idle timeout: 120s.")
        );
        assert!(definition.description.contains("Active sessions: Job-A."));
        assert_eq!(Agent::<AgentCtx>::description(&agent), "  ");
        assert_eq!(
            Agent::<AgentCtx>::tool_dependencies(&agent),
            (0..10).map(|idx| format!("tool_{idx}")).collect::<Vec<_>>()
        );
        assert_eq!(
            Agent::<AgentCtx>::supported_resource_tags(&agent),
            (0..10).map(|idx| format!("tag_{idx}")).collect::<Vec<_>>()
        );

        let args = SubAgentArgs::from_prompt(
            json!({
                "prompt": "structured",
                "session": "CASE",
                "model": "Flash",
                "effort": null
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(args.prompt, "structured");
        assert_eq!(args.effort, None);
        assert_eq!(
            SubAgentArgs::from_prompt("plain".to_string())
                .unwrap()
                .prompt,
            "plain"
        );

        // Invalid structured arguments must surface as errors instead of silently degrading
        // into a blocking run with the raw JSON as the prompt.
        assert!(
            SubAgentArgs::from_prompt(json!({"prompt": "task", "effort": "ultra"}).to_string())
                .is_err()
        );
        // JSON payloads that are not subagent arguments are task data, not arguments.
        let data_prompt = json!({"city": "Reykjavik", "population": 139000}).to_string();
        let args = SubAgentArgs::from_prompt(data_prompt.clone()).unwrap();
        assert_eq!(args.prompt, data_prompt);
        assert!(args.session.is_empty());
        // Text that merely starts with '{' is also a plain prompt.
        let text_prompt = "{\"a\":1} and {\"b\":2} differ, explain why".to_string();
        let args = SubAgentArgs::from_prompt(text_prompt.clone()).unwrap();
        assert_eq!(args.prompt, text_prompt);

        let manager_args = SubAgentManagerArgs::from_prompt(
            json!({
                "output_schema": {"type": "object"},
                "effort": null,
                "idle_timeout": 90,
                "task": "  run task  ",
                "session": "  Thread  ",
                "model": "  Pro  ",
                "persist": true
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(manager_args.output_schema, Some(json!({"type": "object"})));
        assert_eq!(manager_args.effort, None);
        assert_eq!(manager_args.idle_timeout, 90);
        let (agent, task, session, persist) = manager_args.into_subagent();
        assert_eq!(task.as_deref(), Some("run task"));
        assert_eq!(session, "thread");
        assert_eq!(agent.model, "Pro");
        assert_eq!(agent.idle_timeout, 90);
        assert!(persist);
        assert!(SubAgentManagerArgs::from_prompt("not json".to_string()).is_err());

        let manager_args = SubAgentManagerArgs::from_prompt(
            json!({
                "output_schema": "{\"type\":\"array\"}",
                "effort": " HIGH "
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(manager_args.output_schema, Some(json!({"type": "array"})));
        assert_eq!(manager_args.effort, Some(ModelEffort::High));

        let manager_args = SubAgentManagerArgs::from_prompt(
            json!({
                "output_schema": "",
                "effort": ""
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(manager_args.output_schema, None);
        assert_eq!(manager_args.effort, None);

        let manager_args = SubAgentManagerArgs::from_prompt(
            json!({
                "output_schema": null,
                "effort": null
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(manager_args.output_schema, None);
        assert_eq!(manager_args.effort, None);

        assert!(
            SubAgentManagerArgs::from_prompt(json!({"output_schema": "{"}).to_string()).is_err()
        );
        assert!(
            SubAgentManagerArgs::from_prompt(json!({"effort": {"bad": true}}).to_string()).is_err()
        );

        let mut resources = vec![
            Resource {
                blob: Some(b"text resource".to_vec().into()),
                ..Default::default()
            },
            Resource {
                uri: Some("file://image.png".to_string()),
                mime_type: Some("image/png".to_string()),
                ..Default::default()
            },
            Resource::default(),
        ];
        let content = prompt_and_resources_into_content("prompt".to_string(), &mut resources);
        assert_eq!(content.len(), 3);
        assert!(resources.is_empty());
        assert!(resources_into_content(vec![Resource::default()]).is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagent_managers_cover_status_errors_persistence_and_set_routing() {
        let ctx = EngineBuilder::new()
            .with_model(Model::with_completer(Arc::new(EchoCompleter)))
            .mock_ctx();
        let manager =
            SubAgentManager::new().with_models(vec!["flash".to_string(), "pro".to_string()]);
        assert!(manager.description().contains("flash, pro"));
        assert_eq!(manager.supported_resource_tags(), vec!["*".to_string()]);
        assert_eq!(manager.name(), SubAgentManager::NAME);

        let unsupported = Agent::<AgentCtx>::run(
            &manager,
            ctx.clone(),
            serde_json::to_string(&SubAgentManagerArgs {
                operation: "delete".to_string(),
                ..Default::default()
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap_err();
        assert!(unsupported.to_string().contains("unsupported"));

        let created = Agent::<AgentCtx>::run(
            &manager,
            ctx.clone(),
            serde_json::to_string(&SubAgentManagerArgs {
                operation: "create".to_string(),
                name: "router".to_string(),
                description: "Routes tagged resources.".to_string(),
                instructions: "Echo tasks.".to_string(),
                tags: vec!["text".to_string()],
                persist: true,
                ..Default::default()
            })
            .unwrap(),
            vec![resource(9, &["text"])],
        )
        .await
        .unwrap();
        let created_json: Json = serde_json::from_str(&created.content).unwrap();
        assert_eq!(created_json["result"], json!("persisted"));
        assert_eq!(created_json["callable"], json!("SA_router"));

        let status = Agent::<AgentCtx>::run(
            &manager,
            ctx.clone(),
            serde_json::to_string(&SubAgentManagerArgs {
                operation: "status".to_string(),
                ..Default::default()
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(&status.content).unwrap()["count"],
            json!(1)
        );

        let filtered = manager.definitions(Some(&["ROUTER".to_string(), "missing".to_string()]));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "router");

        let mut resources = vec![resource(1, &["image"]), resource(2, &["text"])];
        let selected = SubAgentSet::select_resources(&manager, "ROUTER", &mut resources);
        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            resources
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert!(SubAgentSet::select_resources(&manager, "missing", &mut resources).is_empty());
        resources.clear();
        assert!(SubAgentSet::select_resources(&manager, "router", &mut resources).is_empty());

        let set_manager = Arc::new(SubAgentSetManager::default());
        assert!(set_manager.definitions(None).is_empty());
        assert!(!set_manager.contains_lowercase("router"));
        assert!(set_manager.get_lowercase("router").is_none());
        let downcast = set_manager.clone().into_any();
        assert!(downcast.downcast_ref::<SubAgentSetManager>().is_some());
        assert!(set_manager.insert(Arc::new(manager)).is_none());
        assert!(set_manager.contains_lowercase("router"));
        assert!(set_manager.get::<SubAgentManager>().is_some());
        assert!(set_manager.get_lowercase("router").is_some());
        assert_eq!(
            set_manager.definitions(Some(&["router".to_string()])).len(),
            1
        );
        let mut resources = vec![resource(3, &["text"]), resource(4, &["audio"])];
        let selected =
            SubAgentSet::select_resources(set_manager.as_ref(), "router", &mut resources);
        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![3]
        );

        let empty_set_manager = Arc::new(SubAgentSetManager::default());
        assert!(
            empty_set_manager
                .insert(Arc::new(EmptySubAgentSet))
                .is_none()
        );
        assert!(!empty_set_manager.contains_lowercase("router"));
        assert!(empty_set_manager.get_lowercase("router").is_none());
        assert!(empty_set_manager.definitions(None).is_empty());
        let mut resources = vec![resource(5, &["text"])];
        assert!(
            SubAgentSet::select_resources(empty_set_manager.as_ref(), "router", &mut resources)
                .is_empty()
        );
        assert_eq!(
            resources
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![5]
        );
    }

    #[test]
    fn resolve_idle_timeout_ms_applies_default_and_clamps() {
        // 0 keeps the engine default.
        assert_eq!(resolve_idle_timeout_ms(0), CONVERSATION_IDLE_MS);
        // A positive value converts seconds to milliseconds.
        assert_eq!(resolve_idle_timeout_ms(30), 30_000);
        // Anything above the background-task ceiling is clamped down to it.
        assert_eq!(
            resolve_idle_timeout_ms(u64::MAX),
            CONVERSATION_WAIT_BACKGROUND_TASK_MS
        );
        assert_eq!(
            resolve_idle_timeout_ms(CONVERSATION_WAIT_BACKGROUND_TASK_MS / 1000 + 60),
            CONVERSATION_WAIT_BACKGROUND_TASK_MS
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_honors_configured_idle_timeout() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        // Builds a runner whose session uses the given idle window, seeds it with one step, then
        // rewinds `active_at` by a second to simulate the window elapsing.
        async fn aged_runner(
            ctx: &AgentCtx,
            idle_timeout_ms: u64,
        ) -> (Arc<SubSession>, SubSessionRunner) {
            let (sender, _rx) = tokio::sync::mpsc::channel(4);
            let session = Arc::new(SubSession {
                id: "session-1".to_string(),
                agent: "worker".to_string(),
                sender,
                background_tasks: Arc::new(RwLock::new(HashMap::new())),
                active_at: AtomicU64::new(unix_ms()),
                idle_timeout_ms,
            });
            let mut runner = SubSessionRunner {
                session: session.clone(),
                agent_hook: None,
                runner: ctx
                    .clone()
                    .completion_iter(
                        CompletionRequest {
                            prompt: "seed task".to_string(),
                            ..Default::default()
                        },
                        Vec::new(),
                    )
                    .unbound(),
                last_output: None,
                carried_artifacts: Vec::new(),
                closing: false,
            };
            // First step consumes the seed and keeps the session active.
            assert!(runner.run(Vec::new()).await.unwrap());
            session
                .active_at
                .store(unix_ms().saturating_sub(1000), Ordering::SeqCst);
            (session, runner)
        }

        // A tiny window: ~1s of inactivity exceeds it, so the idle session is reclaimed.
        let (_session, mut tight) = aged_runner(&ctx, 5).await;
        assert!(!tight.run(Vec::new()).await.unwrap());

        // A generous window: the same ~1s of inactivity stays well within it, so the session
        // survives. This proves the boundary tracks the configured value, not merely "active_at is
        // old".
        let (_session, mut roomy) = aged_runner(&ctx, 60_000).await;
        assert!(roomy.run(Vec::new()).await.unwrap());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn needs_compaction_respects_usage_threshold() {
        let low_model = Model::with_completer(Arc::new(UsageCompleter {
            input_tokens: 99_999,
        }));
        let low_ctx = EngineBuilder::new().with_model(low_model).mock_ctx();
        let mut low_runner = low_ctx.completion_iter(
            CompletionRequest {
                prompt: "below threshold".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );
        low_runner.next().await.unwrap().unwrap();
        assert_eq!(low_runner.current_usage().input_tokens, 99_999);
        assert!(!needs_compaction(&low_runner));

        let high_model = Model::with_completer(Arc::new(UsageCompleter {
            input_tokens: 100_000,
        }));
        let high_ctx = EngineBuilder::new().with_model(high_model).mock_ctx();
        let mut high_runner = high_ctx.completion_iter(
            CompletionRequest {
                prompt: "at threshold".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );
        high_runner.next().await.unwrap().unwrap();
        assert_eq!(high_runner.current_usage().input_tokens, 100_000);
        assert!(needs_compaction(&high_runner));

        let mut small_context_model =
            Model::with_completer(Arc::new(UsageCompleter { input_tokens: 800 }));
        small_context_model.context_window = 1_000;
        let small_context_ctx = EngineBuilder::new()
            .with_model(small_context_model)
            .mock_ctx();
        let mut small_context_runner = small_context_ctx.completion_iter(
            CompletionRequest {
                prompt: "near small context limit".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );
        small_context_runner.next().await.unwrap().unwrap();
        assert!(needs_compaction(&small_context_runner));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn needs_compaction_triggers_at_turn_limit() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let mut runner = ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "turn-0".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound();

        for turn in 0..MAX_TURNS_TO_COMPACT {
            if turn > 0 {
                runner.follow_up(format!("turn-{turn}"));
            }

            let output = runner.next().await.unwrap().unwrap();
            assert_eq!(output.content, format!("turn-{turn}"));

            if turn + 1 < MAX_TURNS_TO_COMPACT {
                assert!(runner.next().await.unwrap().is_none());
                assert!(!needs_compaction(&runner));
            }
        }

        assert_eq!(runner.turns(), MAX_TURNS_TO_COMPACT);
        assert!(needs_compaction(&runner));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_compacts_context_and_continues_from_handoff() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingCompactionCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let hook = Arc::new(RecordingAgentHook::default());

        let req = CompletionRequest {
            instructions: "Keep working".to_string(),
            prompt: "seed task".to_string(),
            role: Some("user".to_string()),
            tools: vec![FunctionDefinition {
                name: "lookup".to_string(),
                description: "Lookup data.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                strict: Some(true),
            }],
            output_schema: Some(json!({
                "type": "object",
                "additionalProperties": false
            })),
            model: Some("flash".to_string()),
            effort: Some(ModelEffort::High),
            ..Default::default()
        };

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "compactor".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: Some(DynAgentHook::new(hook.clone())),
            runner: ctx.clone().completion_iter(req, Vec::new()).unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert_eq!(
            runner.last_output.as_ref().map(|v| v.content.as_str()),
            Some("seed task")
        );

        let role_before_compaction = runner.runner.req().role.clone();
        let output_schema_before_compaction = runner.runner.req().output_schema.clone();
        let model_before_compaction = runner.runner.req().model.clone();
        let effort_before_compaction = runner.runner.req().effort;

        assert!(runner.run(Vec::new()).await.unwrap());

        // Usage accumulated before compaction is carried into the replacement runner.
        assert_eq!(runner.runner.total_usage().input_tokens, 100_001);
        assert_eq!(runner.runner.total_usage().requests, 2);

        let progress = hook.progress_events();
        assert_eq!(progress.len(), 1);
        assert_eq!(progress[0].0, "session-1");
        assert_eq!(progress[0].1.content, "seed task");

        let recorded = requests.lock().clone();
        assert_eq!(recorded.len(), 2);
        assert_eq!(request_text(&recorded[0]), "seed task");
        assert_eq!(request_text(&recorded[1]).trim(), COMPACTION_PROMPT.trim());
        assert!(recorded[1].tools.is_empty());

        assert_eq!(runner.runner.chat_history().len(), 1);
        assert_eq!(runner.runner.chat_history()[0].role, "assistant");
        assert_eq!(
            runner.runner.chat_history()[0].text().as_deref(),
            Some("compacted handoff")
        );
        assert_eq!(runner.runner.req().instructions, "Keep working");
        assert_eq!(runner.runner.req().role, role_before_compaction);
        assert_eq!(
            runner.runner.req().output_schema,
            output_schema_before_compaction
        );
        assert_eq!(runner.runner.req().model, model_before_compaction);
        assert_eq!(runner.runner.req().effort, effort_before_compaction);
        assert_eq!(runner.runner.req().tools.len(), 1);
        assert_eq!(runner.runner.req().tools[0].name, "lookup");
        assert!(runner.runner.req().prompt.is_empty());
        assert!(runner.runner.req().content.is_empty());
        assert_eq!(
            runner
                .last_output
                .as_ref()
                .map(|output| output.content.as_str()),
            Some("seed task")
        );

        assert!(
            runner
                .run(vec![SubAgentInput {
                    command: PromptCommand::Plain {
                        prompt: "continue after compaction".to_string(),
                    },
                    resources: Vec::new(),
                    usage: Usage::default(),
                    model: None,
                    effort: None,
                }])
                .await
                .unwrap()
        );

        let resumed = runner.last_output.as_ref().unwrap();
        assert_eq!(resumed.session.as_deref(), Some("session-1"));
        assert_eq!(
            resumed.content,
            "history=compacted handoff; input=continue after compaction"
        );

        let recorded = requests.lock().clone();
        assert_eq!(recorded.len(), 3);
        assert_eq!(
            recorded[2]
                .chat_history
                .iter()
                .filter_map(Message::text)
                .collect::<Vec<_>>(),
            vec!["compacted handoff".to_string()]
        );
        assert_eq!(request_text(&recorded[2]), "continue after compaction");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_fails_when_compaction_summary_is_empty() {
        let model = Model::with_completer(Arc::new(EmptyCompactionCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            instructions: "Keep working".to_string(),
            prompt: "seed task".to_string(),
            ..Default::default()
        };

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "compactor".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
            // A previously rescued artifact must survive even when compaction fails.
            carried_artifacts: vec![resource(5, &["artifact"])],
            closing: false,
        };

        // First step seeds the conversation and pushes usage over the compaction threshold.
        assert!(runner.run(Vec::new()).await.unwrap());

        // The next idle step triggers compaction. An empty summary must fail loudly instead of
        // replacing the entire conversation with an empty handoff message.
        let err = runner.run(Vec::new()).await.unwrap_err();
        assert!(err.to_string().contains("empty summary"), "{err}");

        let mut output = runner.latest_output();
        runner.merge_carried_artifacts(&mut output);
        assert_eq!(
            output.failed_reason.as_deref(),
            Some("context compaction produced an empty summary")
        );
        assert_eq!(output.session.as_deref(), Some("session-1"));
        assert!(output.content.is_empty());
        assert_eq!(
            output
                .artifacts
                .iter()
                .map(|artifact| artifact._id)
                .collect::<Vec<_>>(),
            vec![5]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_emits_progress_for_signal_steps_without_waiting_for_idle() {
        // A step that carries visible narration emits progress immediately, even while tool calls
        // keep the runner busy (i.e. without waiting for an idle boundary).
        let model = Model::with_completer(Arc::new(NarratingToolCallCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let hook = Arc::new(RecordingAgentHook::default());

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: Some(DynAgentHook::new(hook.clone())),
            runner: ctx
                .completion_iter(
                    CompletionRequest {
                        prompt: "start long work".to_string(),
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert!(!runner.runner.is_idle());

        let progress = hook.progress_events();
        assert_eq!(progress.len(), 1);
        assert_eq!(progress[0].0, "session-1");
        assert_eq!(progress[0].1.session.as_deref(), Some("session-1"));
        assert_eq!(progress[0].1.content, "searching now");
        assert_eq!(progress[0].1.tool_calls.len(), 1);
        assert_eq!(
            runner
                .last_output
                .as_ref()
                .map(|output| output.tool_calls.len()),
            Some(1)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_filters_signalless_tool_call_steps_from_progress() {
        // A pure tool-call step carries no visible signal; it must be filtered out so the parent
        // is not flooded with mechanical "still calling tools" noise. The step is still tracked in
        // last_output for the eventual final report.
        let model = Model::with_completer(Arc::new(ToolCallProgressCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let hook = Arc::new(RecordingAgentHook::default());

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: Some(DynAgentHook::new(hook.clone())),
            runner: ctx
                .completion_iter(
                    CompletionRequest {
                        prompt: "start long work".to_string(),
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert!(!runner.runner.is_idle());

        assert!(hook.progress_events().is_empty());
        assert_eq!(
            runner
                .last_output
                .as_ref()
                .map(|output| output.tool_calls.len()),
            Some(1)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_accepts_resource_only_follow_up() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingRequestCompleter {
            name: "recorder",
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "seed task".to_string(),
            ..Default::default()
        };

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert_eq!(runner.last_output.as_ref().unwrap().content, "seed task");

        assert!(
            runner
                .run(vec![SubAgentInput {
                    command: PromptCommand::Ping,
                    resources: vec![Resource {
                        blob: Some(b"resource follow-up".to_vec().into()),
                        ..Default::default()
                    }],
                    usage: Usage::default(),
                    model: None,
                    effort: None,
                }])
                .await
                .unwrap()
        );

        assert_eq!(
            runner.last_output.as_ref().unwrap().content,
            "resource follow-up"
        );
        let recorded = requests.lock().clone();
        assert_eq!(recorded.len(), 2);
        assert!(recorded[1].prompt.is_empty());
        assert_eq!(request_text(&recorded[1]), "resource follow-up");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_handles_control_inputs_model_effort_and_finalize_errors() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingRequestCompleter {
            name: "recorder",
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx
                .completion_iter(
                    CompletionRequest {
                        prompt: "seed task".to_string(),
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert!(
            runner
                .run(vec![SubAgentInput {
                    command: PromptCommand::Command {
                        command: "steer".to_string(),
                        prompt: "correct course".to_string(),
                    },
                    resources: Vec::new(),
                    usage: Usage {
                        input_tokens: 3,
                        output_tokens: 4,
                        cached_tokens: 1,
                        requests: 1,
                    },
                    model: Some("analysis".to_string()),
                    effort: Some(ModelEffort::Low),
                }])
                .await
                .unwrap()
        );
        assert_eq!(
            runner.last_output.as_ref().unwrap().session.as_deref(),
            Some("session-1")
        );

        assert!(
            runner
                .run(vec![SubAgentInput {
                    command: PromptCommand::Command {
                        command: "note".to_string(),
                        prompt: "/note custom follow-up".to_string(),
                    },
                    resources: Vec::new(),
                    usage: Usage::default(),
                    model: None,
                    effort: None,
                }])
                .await
                .unwrap()
        );

        let recorded = requests.lock().clone();
        assert_eq!(recorded.len(), 3);
        assert_eq!(request_text(&recorded[0]), "seed task");
        assert_eq!(request_text(&recorded[1]), "correct course");
        assert_eq!(recorded[1].model.as_deref(), Some("analysis"));
        assert_eq!(recorded[1].effort, Some(ModelEffort::Low));
        assert_eq!(request_text(&recorded[2]), "/note custom follow-up");

        let error_ctx = EngineBuilder::new()
            .with_model(Model::with_completer(Arc::new(ErrorCompleter)))
            .mock_ctx();
        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-err".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });
        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: error_ctx
                .completion_iter(
                    CompletionRequest {
                        prompt: "will fail".to_string(),
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };
        let output = runner.finalize_output().await;
        assert_eq!(output.failed_reason.as_deref(), Some("model failed"));
        assert_eq!(output.session.as_deref(), Some("session-err"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_finalizes_with_latest_visible_output_after_compaction() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingCompactionCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            instructions: "Keep working".to_string(),
            prompt: "seed task".to_string(),
            ..Default::default()
        };

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "compactor".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert!(runner.run(Vec::new()).await.unwrap());
        assert_eq!(
            runner.runner.chat_history()[0].text().as_deref(),
            Some("compacted handoff")
        );

        runner.carried_artifacts.push(resource(11, &["artifact"]));
        let output = runner.finalize_output().await;
        assert_eq!(output.content, "seed task");
        assert_eq!(output.session.as_deref(), Some("session-1"));
        // Artifacts rescued from the pre-compaction runner survive into the final output.
        assert_eq!(
            output
                .artifacts
                .iter()
                .map(|artifact| artifact._id)
                .collect::<Vec<_>>(),
            vec![11]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_keeps_latest_output_for_final_end_after_progress() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "seed task".to_string(),
            ..Default::default()
        };

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert_eq!(
            runner
                .last_output
                .as_ref()
                .map(|output| output.content.as_str()),
            Some("seed task")
        );

        let progress_output = runner.last_output.take().unwrap();
        assert_eq!(progress_output.content, "seed task");

        let final_output = runner.finalize_output().await;
        assert_eq!(final_output.content, "seed task");
        assert_eq!(final_output.session.as_deref(), Some("session-1"));
        assert!(final_output.failed_reason.is_none());
        assert!(runner.runner.is_done());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_output_classification_and_failure_defaults_are_covered() {
        assert!(!SubSessionRunner::has_observable_output(
            &AgentOutput::default()
        ));

        for output in [
            AgentOutput {
                thoughts: Some("thinking".to_string()),
                ..Default::default()
            },
            AgentOutput {
                failed_reason: Some("failed".to_string()),
                ..Default::default()
            },
            AgentOutput {
                tool_calls: vec![anda_core::ToolCall {
                    name: "lookup".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            AgentOutput {
                chat_history: vec![Message {
                    role: "assistant".to_string(),
                    content: vec!["history".to_string().into()],
                    ..Default::default()
                }],
                ..Default::default()
            },
            AgentOutput {
                artifacts: vec![Resource {
                    _id: 42,
                    ..Default::default()
                }],
                ..Default::default()
            },
            AgentOutput {
                conversation: Some(7),
                ..Default::default()
            },
            AgentOutput {
                model: Some("flash".to_string()),
                ..Default::default()
            },
            AgentOutput {
                usage: Usage {
                    requests: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
            AgentOutput {
                tools_usage: HashMap::from([(
                    "lookup".to_string(),
                    Usage {
                        requests: 1,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            },
        ] {
            assert!(SubSessionRunner::has_observable_output(&output));
        }

        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx
                .completion_iter(
                    CompletionRequest {
                        prompt: "seed task".to_string(),
                        ..Default::default()
                    },
                    Vec::new(),
                )
                .unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert_eq!(
            runner.record_failed_output(""),
            "subagent session cancelled"
        );
        let output = runner.latest_output();
        assert_eq!(
            output.failed_reason.as_deref(),
            Some("subagent session cancelled")
        );
        assert_eq!(output.session.as_deref(), Some("session-1"));
        assert!(output.content.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_runner_reports_cancel_failure_as_latest_output() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "seed task".to_string(),
            ..Default::default()
        };

        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        let session = Arc::new(SubSession {
            id: "session-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
            carried_artifacts: Vec::new(),
            closing: false,
        };

        assert!(runner.run(Vec::new()).await.unwrap());

        let err = runner
            .run(vec![SubAgentInput {
                command: PromptCommand::Command {
                    command: "cancel".to_string(),
                    prompt: "stop because requested".to_string(),
                },
                resources: Vec::new(),
                usage: Usage::default(),
                model: None,
                effort: None,
            }])
            .await
            .unwrap_err();

        assert_eq!(err.to_string(), "stop because requested");
        let final_output = runner.latest_output();
        assert_eq!(
            final_output.failed_reason.as_deref(),
            Some("stop because requested")
        );
        assert_eq!(final_output.session.as_deref(), Some("session-1"));
        assert!(final_output.content.is_empty());
    }

    #[test]
    fn subagent_definition_guides_self_contained_prompts() {
        let agent = SubAgent {
            name: "research_assistant".to_string(),
            description: "Handles recurring research tasks with concise synthesis.".to_string(),
            instructions: "Research carefully and synthesize findings.".to_string(),
            tools: vec!["google_web_search".to_string()],
            model: "pro".to_string(),
            effort: Some(ModelEffort::High),
            ..Default::default()
        };

        let definition = agent.definition();

        assert_eq!(definition.name, "research_assistant");
        assert!(
            definition
                .description
                .starts_with("Handles recurring research tasks with concise synthesis.")
        );
        assert!(
            definition
                .description
                .contains("Allowed tools: google_web_search.")
        );
        assert!(definition.description.contains("Default model label: pro."));
        assert!(definition.description.contains("Default effort: high."));
        assert_eq!(
            definition.parameters["description"],
            json!(
                "Run this subagent as a focused worker process. The caller acts as the scheduler: use blocking mode for short one-shot work, or session mode for long-running, parallel, asynchronous, or follow-up work. Keep each prompt as a self-contained handoff."
            )
        );
        assert!(
            definition.parameters["properties"]["prompt"]["description"]
                .as_str()
                .unwrap()
                .contains("Self-contained task handoff")
        );
        assert_eq!(
            definition.parameters["properties"]["session"]["default"],
            json!("")
        );
        assert_eq!(
            definition.parameters["properties"]["model"]["default"],
            json!("")
        );
        assert_eq!(
            definition.parameters["properties"]["effort"]["enum"],
            json!(["minimal", "low", "medium", "high", "max", null])
        );
        assert_eq!(definition.parameters["additionalProperties"], json!(false));
    }

    #[test]
    fn subagents_manager_definition_uses_strict_safe_output_schema() {
        let definition = SubAgentManager::new()
            .definition()
            .normalize_strict_parameters();

        assert_eq!(
            definition.parameters["properties"]["output_schema"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            definition.parameters["properties"]["operation"]["default"],
            json!("upsert")
        );
        assert_eq!(
            definition.parameters["properties"]["model"]["default"],
            json!("")
        );
        assert_eq!(
            definition.parameters["properties"]["effort"]["type"],
            json!(["string", "null"])
        );
        assert!(
            definition.parameters["required"]
                .as_array()
                .unwrap()
                .contains(&json!("operation"))
        );
        assert_eq!(definition.parameters["additionalProperties"], json!(false));
    }

    #[test]
    fn subagents_manager_args_accept_json_encoded_output_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" }
            },
            "required": ["summary"],
            "additionalProperties": false
        });

        let args = SubAgentManagerArgs::from_prompt(
            json!({
                "name": "structured_helper",
                "description": "Creates structured output.",
                "instructions": "Return structured output.",
                "tools": [],
                "tags": [],
                "output_schema": serde_json::to_string(&schema).unwrap(),
                "model": "Pro",
                "effort": "HIGH",
                "task": "",
                "session": "",
                "persist": false
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(args.operation, "upsert");
        assert_eq!(args.output_schema, Some(schema));
        assert_eq!(args.model, "Pro");
        assert_eq!(args.effort, Some(ModelEffort::High));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagent_run_allows_model_and_effort_selection() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let models = Arc::new(Models::default());
        models.set_model(Model::with_completer(Arc::new(EchoCompleter)));
        models.set(
            "analysis".to_string(),
            Model::with_completer(Arc::new(RecordingRequestCompleter {
                name: "analysis-model",
                requests: requests.clone(),
            })),
        );
        let ctx = EngineBuilder::new().with_models(models).mock_ctx();
        let agent = SubAgent {
            name: "analysis_helper".to_string(),
            description: "Runs analysis work.".to_string(),
            instructions: "Analyze carefully.".to_string(),
            ..Default::default()
        };

        let output = agent
            .run(
                ctx,
                serde_json::to_string(&SubAgentArgs {
                    prompt: "inspect this".to_string(),
                    session: String::new(),
                    model: "analysis".to_string(),
                    effort: Some(ModelEffort::High),
                })
                .unwrap(),
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(output.content, "inspect this");
        assert_eq!(output.model.as_deref(), Some("analysis-model"));
        let recorded = requests.lock().clone();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].model.as_deref(), Some("analysis"));
        assert_eq!(recorded[0].effort, Some(ModelEffort::High));
    }

    #[test]
    fn subagents_manager_upsert_preserves_active_sessions() {
        let manager = SubAgentManager::new();
        let agent = SubAgent {
            name: "worker".to_string(),
            description: "Handles delegated work.".to_string(),
            instructions: "Work carefully.".to_string(),
            ..Default::default()
        };
        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        agent.subsessions.insert_session(Arc::new(SubSession {
            id: "job-1".to_string(),
            agent: "worker".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        }));

        manager.upsert_temporary(agent).unwrap();
        manager
            .upsert_temporary(SubAgent {
                name: "worker".to_string(),
                description: "Updated routing description.".to_string(),
                instructions: "Use the updated workflow.".to_string(),
                ..Default::default()
            })
            .unwrap();

        let loaded = manager.get_lowercase("worker").unwrap();
        assert_eq!(loaded.description, "Updated routing description.");
        assert_eq!(loaded.subsessions.active_session_ids(), vec!["job-1"]);
        assert!(loaded.subsessions.get_session("job-1").is_some());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagent_run_accepts_plain_prompts_and_structured_session_args() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let agent = SubAgent {
            name: "echo_helper".to_string(),
            description: "Echoes input.".to_string(),
            instructions: "Echo the prompt.".to_string(),
            ..Default::default()
        };

        let output = agent
            .run(ctx.clone(), "plain task".to_string(), Vec::new())
            .await
            .unwrap();
        assert_eq!(output.content, "plain task");

        let output = agent
            .run(
                ctx,
                serde_json::to_string(&SubAgentArgs {
                    prompt: "session task".to_string(),
                    session: "ThreadA".to_string(),
                    model: String::new(),
                    effort: None,
                })
                .unwrap(),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.session.as_deref(), Some("threada"));
        assert!(output.content.contains("session mode"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagent_session_does_not_emit_start_hook_when_ack_hook_fails() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let hook = Arc::new(FailingAfterAgentHook::default());
        ctx.base.set_state(DynAgentHook::new(hook.clone()));

        let agent = SubAgent {
            name: "echo_helper".to_string(),
            description: "Echoes input.".to_string(),
            instructions: "Echo the prompt.".to_string(),
            ..Default::default()
        };

        let err = agent
            .run(
                ctx,
                serde_json::to_string(&SubAgentArgs {
                    prompt: "session task".to_string(),
                    session: "ThreadA".to_string(),
                    model: String::new(),
                    effort: None,
                })
                .unwrap(),
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.to_string(), "after hook rejected output");
        assert!(hook.starts.lock().is_empty());
        assert!(agent.subsessions.active_session_ids().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_background_hooks_forward_outputs_and_manage_registry() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let session = Arc::new(SubSession {
            id: "session-1".into(),
            agent: "worker".into(),
            sender: tx,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });
        let ctx = EngineBuilder::new().mock_ctx();

        let sessions = SubSessions::default();
        sessions.insert_session(session.clone());
        assert_eq!(sessions.active_session_ids(), vec!["session-1".to_string()]);
        assert!(sessions.get_session("session-1").is_some());

        AgentHook::on_background_start(
            session.as_ref(),
            &ctx,
            "child-session",
            &CompletionRequest::default(),
        )
        .await;
        assert_eq!(
            session
                .background_tasks
                .read()
                .get("child-session")
                .unwrap()
                .agent_name,
            "Mocker"
        );

        AgentHook::on_background_progress(
            session.as_ref(),
            &ctx,
            "child-session".into(),
            AgentOutput {
                content: "partial".into(),
                usage: Usage {
                    requests: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await;
        assert!(
            recv_subagent_prompt(&mut rx)
                .await
                .contains("intermediate output")
        );

        AgentHook::on_background_progress(
            session.as_ref(),
            &ctx,
            "child-session".into(),
            AgentOutput {
                failed_reason: Some("bad".into()),
                ..Default::default()
            },
        )
        .await;
        assert!(
            recv_subagent_prompt(&mut rx)
                .await
                .contains("failed with reason")
        );

        AgentHook::on_background_progress(
            session.as_ref(),
            &ctx,
            "child-session".into(),
            AgentOutput::default(),
        )
        .await;
        assert!(recv_subagent_prompt(&mut rx).await.contains("completed"));

        AgentHook::on_background_end(
            session.as_ref(),
            &ctx,
            "child-session".into(),
            AgentOutput {
                content: "final".into(),
                ..Default::default()
            },
        )
        .await;
        assert!(
            !session
                .background_tasks
                .read()
                .contains_key("child-session")
        );
        assert!(recv_subagent_prompt(&mut rx).await.contains("final output"));

        ToolBackgroundHook::on_background_start(
            session.as_ref(),
            &ctx.base,
            "fetch:task-1",
            json!({"url": "https://example.test"}),
        )
        .await;
        assert_eq!(
            session
                .background_tasks
                .read()
                .get("fetch:task-1")
                .unwrap()
                .tool_name
                .as_deref(),
            Some("fetch")
        );

        ToolBackgroundHook::on_background_progress(
            session.as_ref(),
            &ctx.base,
            "fetch:task-1".into(),
            ToolOutput::new(json!({"status": "half"})),
        )
        .await;
        assert_eq!(
            session
                .background_tasks
                .read()
                .get("fetch:task-1")
                .unwrap()
                .progress_message
                .as_deref(),
            Some(r#"{"status":"half"}"#)
        );

        ToolBackgroundHook::on_background_start(
            session.as_ref(),
            &ctx.base,
            "unprefixed",
            Json::Null,
        )
        .await;
        assert!(
            session
                .background_tasks
                .read()
                .get("unprefixed")
                .unwrap()
                .tool_name
                .is_none()
        );

        ToolBackgroundHook::on_background_end(
            session.as_ref(),
            &ctx.base,
            "fetch:task-1".into(),
            ToolOutput {
                output: json!({"done": true}),
                artifacts: vec![resource(7, &["artifact"])],
                usage: Usage {
                    requests: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await;
        assert!(!session.background_tasks.read().contains_key("fetch:task-1"));
        let forwarded = rx.recv().await.unwrap();
        assert!(matches!(forwarded.command, PromptCommand::Plain { .. }));
        assert_eq!(forwarded.resources.len(), 1);
        assert_eq!(forwarded.usage.requests, 1);

        sessions.remove_session("session-1");
        assert!(sessions.get_session("session-1").is_none());

        let (closed_tx, closed_rx) = tokio::sync::mpsc::channel(1);
        let closed = Arc::new(SubSession {
            id: "closed".into(),
            agent: "worker".into(),
            sender: closed_tx,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        });
        drop(closed_rx);
        sessions.insert_session(closed);
        assert!(sessions.active_session_ids().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagents_manager_agent_entrypoint_uses_structured_args() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let manager: Arc<SubAgentManager> = ctx.subagents.get().unwrap();

        let output = Agent::<AgentCtx>::run(
            manager.as_ref(),
            ctx,
            serde_json::to_string(&SubAgentManagerArgs {
                name: "agent_helper".to_string(),
                description: "Created through the agent entrypoint.".to_string(),
                instructions: "Echo agent entrypoint tasks.".to_string(),
                task: "agent task".to_string(),
                ..Default::default()
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();
        let content: Json = serde_json::from_str(&output.content).unwrap();

        assert_eq!(content["subagent"]["result"], json!("created"));
        assert_eq!(content["output"], json!("agent task"));
        assert!(manager.get_lowercase("agent_helper").is_some());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagents_manager_lists_registry_and_active_sessions() {
        let ctx = EngineBuilder::new().mock_ctx();
        let manager: Arc<SubAgentManager> = ctx.subagents.get().unwrap();
        let agent = SubAgent {
            name: "planner".to_string(),
            description: "Plans delegated work.".to_string(),
            instructions: "Break work into concrete steps.".to_string(),
            tools: vec!["tools_select".to_string()],
            tags: vec!["planning".to_string()],
            model: "flash".to_string(),
            effort: Some(ModelEffort::Low),
            ..Default::default()
        };
        let (sender, _rx) = tokio::sync::mpsc::channel(4);
        agent.subsessions.insert_session(Arc::new(SubSession {
            id: "plan-1".to_string(),
            agent: "planner".to_string(),
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
            idle_timeout_ms: 0,
        }));
        manager.upsert_temporary(agent).unwrap();

        let output = Agent::<AgentCtx>::run(
            manager.as_ref(),
            ctx,
            serde_json::to_string(&SubAgentManagerArgs {
                operation: "list".to_string(),
                ..Default::default()
            })
            .unwrap(),
            Vec::new(),
        )
        .await
        .unwrap();
        let content: Json = serde_json::from_str(&output.content).unwrap();

        assert_eq!(content["result"], json!("listed"));
        assert_eq!(content["count"], json!(1));
        assert_eq!(content["subagents"][0]["name"], json!("planner"));
        assert_eq!(content["subagents"][0]["callable"], json!("SA_planner"));
        assert_eq!(content["subagents"][0]["model"], json!("flash"));
        assert_eq!(content["subagents"][0]["effort"], json!("low"));
        assert_eq!(
            content["subagents"][0]["active_sessions"],
            json!(["plan-1"])
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn load_restores_all_persisted_subagents() {
        let ctx = EngineBuilder::new().mock_ctx();
        let tool: Arc<SubAgentManager> = ctx.subagents.get().unwrap();

        let agents = vec![
            SubAgent {
                name: "Research_Assistant".to_string(),
                description: "Handles recurring research tasks with concise synthesis.".to_string(),
                instructions: "Research carefully and synthesize findings.".to_string(),
                tools: vec!["google_web_search".to_string()],
                tags: vec!["research".to_string()],
                ..Default::default()
            },
            SubAgent {
                name: "code_reviewer".to_string(),
                description: "Reviews code for correctness and risks.".to_string(),
                instructions: "Review code changes and summarize findings.".to_string(),
                tools: vec!["read_file".to_string(), "grep_search".to_string()],
                tags: vec!["code".to_string(), "review".to_string()],
                model: "pro".to_string(),
                effort: Some(ModelEffort::Medium),
                idle_timeout: 300,
                ..Default::default()
            },
            SubAgent {
                name: "writer_helper".to_string(),
                description: "Drafts concise written content.".to_string(),
                instructions: "Write clearly and keep the response concise.".to_string(),
                tags: vec!["writing".to_string()],
                output_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" }
                    },
                    "required": ["summary"],
                    "additionalProperties": false
                })),
                ..Default::default()
            },
        ];

        for agent in agents.clone() {
            tool.upsert(ctx.clone(), agent).await.unwrap();
        }

        let legacy_agent = SubAgent {
            name: "legacy_agent".to_string(),
            description: "Stored outside the subagents directory.".to_string(),
            instructions: "This should not be loaded by SubAgentManager::load.".to_string(),
            ..Default::default()
        };
        ctx.store_put(
            &Path::from("legacy_agent"),
            PutMode::Overwrite,
            to_canonical_vec(&legacy_agent).unwrap().into(),
        )
        .await
        .unwrap();

        let stored = ctx
            .store_list(Some(&SubAgentManager::store_prefix()), &Path::from(""))
            .await
            .unwrap();
        assert_eq!(stored.len(), agents.len());
        assert!(stored.iter().all(|meta| {
            meta.location
                .prefix_match(&SubAgentManager::store_prefix())
                .is_some()
        }));

        for meta in &stored {
            let (data, _) = ctx.store_get(&meta.location).await.unwrap();
            let loaded = from_slice::<SubAgent>(&data[..]).unwrap();
            assert!(agents.iter().any(|agent| agent.name == loaded.name));
        }

        let reloaded = SubAgentManager::new();
        reloaded.load(ctx).await.unwrap();

        assert_eq!(reloaded.definitions(None).len(), agents.len());
        assert!(reloaded.get_lowercase("legacy_agent").is_none());

        for expected in agents {
            let loaded = reloaded
                .get_lowercase(&expected.name.to_ascii_lowercase())
                .unwrap();

            assert_eq!(loaded.name, expected.name);
            assert_eq!(loaded.description, expected.description);
            assert_eq!(loaded.instructions, expected.instructions);
            assert_eq!(loaded.tools, expected.tools);
            assert_eq!(loaded.tags, expected.tags);
            assert_eq!(loaded.output_schema, expected.output_schema);
            assert_eq!(loaded.model, expected.model);
            assert_eq!(loaded.effort, expected.effort);
            assert_eq!(loaded.idle_timeout, expected.idle_timeout);
        }
    }

    fn test_session(id: &str) -> (Arc<SubSession>, tokio::sync::mpsc::Receiver<SubAgentInput>) {
        let (sender, rx) = tokio::sync::mpsc::channel(16);
        (
            Arc::new(SubSession {
                id: id.to_string(),
                agent: "worker".to_string(),
                sender,
                background_tasks: Arc::new(RwLock::new(HashMap::new())),
                active_at: AtomicU64::new(unix_ms()),
                idle_timeout_ms: 0,
            }),
            rx,
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsession_agent_hook_forwards_usage_deltas() {
        let (session, mut rx) = test_session("parent");
        let ctx = EngineBuilder::new().mock_ctx();

        AgentHook::on_background_start(
            session.as_ref(),
            &ctx,
            "child",
            &CompletionRequest::default(),
        )
        .await;

        // Background agent outputs carry cumulative usage; the session must receive deltas.
        AgentHook::on_background_progress(
            session.as_ref(),
            &ctx,
            "child".into(),
            AgentOutput {
                content: "step-1".into(),
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 10,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            },
        )
        .await;
        let input = rx.recv().await.unwrap();
        assert_eq!(input.usage.input_tokens, 100);
        assert_eq!(input.usage.output_tokens, 10);
        assert_eq!(input.usage.requests, 1);

        AgentHook::on_background_progress(
            session.as_ref(),
            &ctx,
            "child".into(),
            AgentOutput {
                content: "step-2".into(),
                usage: Usage {
                    input_tokens: 250,
                    output_tokens: 25,
                    cached_tokens: 0,
                    requests: 3,
                },
                ..Default::default()
            },
        )
        .await;
        let input = rx.recv().await.unwrap();
        assert_eq!(input.usage.input_tokens, 150);
        assert_eq!(input.usage.output_tokens, 15);
        assert_eq!(input.usage.requests, 2);

        // A failure output with empty usage must not reset the watermark.
        AgentHook::on_background_progress(
            session.as_ref(),
            &ctx,
            "child".into(),
            AgentOutput {
                failed_reason: Some("transient".into()),
                ..Default::default()
            },
        )
        .await;
        let input = rx.recv().await.unwrap();
        assert_eq!(input.usage.input_tokens, 0);
        assert_eq!(input.usage.requests, 0);

        AgentHook::on_background_end(
            session.as_ref(),
            &ctx,
            "child".into(),
            AgentOutput {
                content: "done".into(),
                usage: Usage {
                    input_tokens: 300,
                    output_tokens: 30,
                    cached_tokens: 0,
                    requests: 4,
                },
                ..Default::default()
            },
        )
        .await;
        let input = rx.recv().await.unwrap();
        assert_eq!(input.usage.input_tokens, 50);
        assert_eq!(input.usage.output_tokens, 5);
        assert_eq!(input.usage.requests, 1);
        assert!(session.background_tasks.read().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subsessions_claim_and_conditional_remove_handle_races() {
        let sessions = SubSessions::default();
        let (first, _rx1) = test_session("job");
        assert!(sessions.try_insert_session(first.clone()).is_none());

        // A concurrent claim joins the active session instead of replacing it.
        let (second, _rx2) = test_session("job");
        let existing = sessions.try_insert_session(second.clone()).unwrap();
        assert!(Arc::ptr_eq(&existing, &first));

        // A stale runner must not remove a session it no longer owns.
        sessions.remove_session_if(&second);
        assert!(sessions.get_session("job").is_some());
        sessions.remove_session_if(&first);
        assert!(sessions.get_session("job").is_none());

        // A closed session is replaced by a fresh claim.
        let (closed, closed_rx) = test_session("job");
        drop(closed_rx);
        sessions.insert_session(closed);
        let (fresh, _rx3) = test_session("job");
        assert!(sessions.try_insert_session(fresh.clone()).is_none());
        assert!(Arc::ptr_eq(&sessions.get_session("job").unwrap(), &fresh));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagent_control_commands_report_inactive_sessions() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let agent = SubAgent {
            name: "echo_helper".to_string(),
            description: "Echoes input.".to_string(),
            instructions: "Echo the prompt.".to_string(),
            ..Default::default()
        };

        for prompt in ["/stop finish now", "/cancel", "", "/ping"] {
            let output = agent
                .run(
                    ctx.clone(),
                    serde_json::to_string(&SubAgentArgs {
                        prompt: prompt.to_string(),
                        session: "Ghost".to_string(),
                        model: String::new(),
                        effort: None,
                    })
                    .unwrap(),
                    Vec::new(),
                )
                .await
                .unwrap();
            assert_eq!(output.session.as_deref(), Some("ghost"));
            assert!(output.content.contains("not active"), "{}", output.content);
            assert!(output.failed_reason.is_none());
            assert!(agent.subsessions.active_session_ids().is_empty());
        }

        // A blocking run still rejects an empty prompt explicitly.
        let err = agent
            .run(ctx, "".to_string(), Vec::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("prompt cannot be empty"));
    }
}
