use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, ContentPart,
    FunctionDefinition, Json, Message, ModelEffort, Path, PromptCommand, PutMode, Resource,
    StoreFeatures, ToolOutput, Usage, select_resources, validate_function_name,
};
use async_trait::async_trait;
use ciborium::from_reader;
use ic_auth_types::deterministic_cbor_into_vec;
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

const CONVERSATION_IDLE_MS: u64 = 10 * 60 * 1000; // 10 minutes
const CONVERSATION_WAIT_BACKGROUND_TASK_MS: u64 = 60 * 60 * 1000; // 1 hour
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

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct SubAgent {
    pub name: String,
    pub description: String,
    pub instructions: String,

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default)]
    pub output_schema: Option<Json>,

    /// Optional default model label used to run this subagent.
    #[serde(default)]
    pub model: String,

    /// Optional default reasoning/thinking effort used to run this subagent.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,

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

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SubAgentArgs {
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
    fn from_prompt(prompt: String) -> Self {
        serde_json::from_str::<Self>(&prompt).unwrap_or(Self {
            prompt,
            session: String::new(),
            model: String::new(),
            effort: None,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubAgentManagerArgs {
    /// Operation to perform. Defaults to creating or updating a subagent.
    #[serde(default = "default_manager_operation")]
    pub operation: String,

    #[serde(default)]
    pub name: String,

    #[serde(default)]
    pub description: String,

    #[serde(default)]
    pub instructions: String,

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default, deserialize_with = "deserialize_optional_json_schema")]
    pub output_schema: Option<Json>,

    /// Optional default model label used to run this subagent.
    #[serde(default)]
    pub model: String,

    /// Optional default reasoning/thinking effort used to run this subagent.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,

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

#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct BackgroundTaskInfo {
    pub agent_name: String,
    pub tool_name: Option<String>,
    pub progress_message: Option<String>,
}

pub struct SubSession {
    id: String,
    agent: String,
    sender: tokio::sync::mpsc::Sender<SubAgentInput>,
    // task_id -> BackgroundTaskInfo
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
    active_at: AtomicU64,
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

    fn has_reportable_output(output: &AgentOutput) -> bool {
        !output.content.is_empty()
            || output.thoughts.is_some()
            || output.failed_reason.is_some()
            || !output.tool_calls.is_empty()
            || !output.artifacts.is_empty()
            || output.usage.requests > 0
            || !output.tools_usage.is_empty()
    }

    fn latest_output(&mut self) -> AgentOutput {
        let output = self
            .last_output
            .take()
            .or_else(|| self.runner.last_output().cloned())
            .unwrap_or_default();

        self.with_session(output)
    }

    async fn finalize_output(&mut self) -> AgentOutput {
        let fallback = self.latest_output();

        match self.runner.finalize(None).await {
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
        }
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
                let now_ms = unix_ms();

                let idle = now_ms.saturating_sub(self.session.active_at.load(Ordering::SeqCst));
                let has_background_tasks = !self.session.background_tasks.read().is_empty();

                if idle > CONVERSATION_IDLE_MS && !has_background_tasks
                    || (idle > CONVERSATION_WAIT_BACKGROUND_TASK_MS && has_background_tasks)
                {
                    return Ok(false);
                }

                if let Some(hook) = &self.agent_hook
                    && let Some(last_output) = self.last_output.take()
                {
                    // 仅当空闲下来时响应 progress，避免过于频繁地推送中间结果
                    hook.on_background_progress(
                        self.runner.ctx(),
                        self.session.id.clone(),
                        last_output,
                    )
                    .await;
                }

                if needs_compaction(&self.runner) {
                    // 上下文过长，先进行一次压缩总结，更新conversation状态和历史消息，再继续后续的处理
                    let handoff_req = self.runner.req().clone();
                    self.runner.set_tools(Vec::new());
                    let output = match self
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
                        let output = self.with_session(output);
                        self.last_output = Some(output);
                        return Err(failed_reason.into());
                    }
                    // 前一轮压缩总结的内容作为新 conversation 的第一条消息，继续后续的交互
                    let now_ms = unix_ms();
                    let compaction_msg = Message {
                        role: "assistant".into(),
                        content: vec![output.content.into()],
                        timestamp: Some(now_ms),
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
                        .reserve_chat_history(vec![compaction_msg])
                        .unbound();
                    return Ok(true);
                }

                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                Ok(true)
            }

            Ok(Some(mut res)) => {
                let now_ms = unix_ms();
                self.session.active_at.store(now_ms, Ordering::SeqCst);
                res.session = Some(self.session.id.clone());
                let is_done = self.runner.is_done() || res.failed_reason.is_some();
                self.last_output = Some(res);
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
    pub fn close(self: Arc<Self>) {
        // no things to do for now
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
            },
        );
    }

    async fn on_background_progress(
        &self,
        _ctx: &AgentCtx,
        session_id: String,
        output: AgentOutput,
    ) {
        let prompt = if !output.content.is_empty() {
            format!(
                "Subagent session {session_id} intermediate output:\n\n{}",
                output.content
            )
        } else if let Some(failed_reason) = output.failed_reason {
            format!(
                "Subagent session {session_id} failed with reason: {:?}",
                failed_reason
            )
        } else {
            format!("Subagent session {session_id} completed")
        };
        self.sender
            .send(SubAgentInput {
                command: PromptCommand::Plain { prompt },
                resources: vec![],
                usage: output.usage,
                model: None,
                effort: None,
            })
            .await
            .ok();
    }

    async fn on_background_end(&self, _ctx: &AgentCtx, session_id: String, output: AgentOutput) {
        {
            self.background_tasks.write().remove(&session_id);
        }

        let prompt = if !output.content.is_empty() {
            format!(
                "Subagent session {session_id} final output:\n\n{}",
                output.content
            )
        } else if let Some(failed_reason) = output.failed_reason {
            format!(
                "Subagent session {session_id} failed with reason: {:?}",
                failed_reason
            )
        } else {
            format!("Subagent session {session_id} completed")
        };
        self.sender
            .send(SubAgentInput {
                command: PromptCommand::Plain { prompt },
                resources: vec![],
                usage: output.usage,
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
    pub fn insert_session(&self, sess: Arc<SubSession>) {
        self.sessions.write().insert(sess.id.clone(), sess);
    }

    pub fn active_session_ids(&self) -> Vec<String> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed());
        sessions.keys().cloned().collect()
    }

    pub fn get_session(&self, id: &str) -> Option<Arc<SubSession>> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed());
        sessions.get(id).cloned()
    }

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
                        "description": "Self-contained task handoff for this subagent. Include objective, context/resources, constraints, dependencies, expected deliverable, success criteria, and what progress/final output should contain."
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

        let args = SubAgentArgs::from_prompt(prompt);
        let model = selected_model_label(&args.model).or_else(|| selected_model_label(&self.model));
        let effort = args.effort.or(self.effort);

        let session_id = args.session.trim().to_ascii_lowercase();
        if session_id.is_empty() {
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
                        "Failed to enqueue prompt for processing session {}",
                        session_id,
                    );
                    subsessions.remove_session(&session_id);
                    input = err.0;
                }
            }
        }

        // If the conversation session is not active, start a new session and process the prompt
        let SubAgentInput {
            command,
            resources,
            model: input_model,
            effort: input_effort,
            ..
        } = input;

        let prompt = match command {
            PromptCommand::Ping if resources.is_empty() => {
                return Err("prompt cannot be empty".into());
            }
            PromptCommand::Ping => String::new(),
            PromptCommand::Plain { prompt } => prompt,
            PromptCommand::Command { command, prompt } => match command.as_str() {
                "stop" | "cancel" => {
                    return Err("prompt cannot be empty".into());
                }
                _ => prompt,
            },
        };

        let (sender, mut rx) = tokio::sync::mpsc::channel::<SubAgentInput>(42);
        let session = Arc::new(SubSession {
            id: session_id,
            agent,
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(unix_ms()),
        });

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
            hook.after_agent_run(&ctx, rt).await?
        } else {
            rt
        };

        ctx.base.set_state(DynAgentHook::new(session.clone()));
        ctx.base.set_state(DynToolJsonHook::new(session.clone()));

        subsessions.insert_session(session.clone());
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
            };

            loop {
                let mut inputs = Vec::new();

                while let Ok(input) = rx.try_recv() {
                    inputs.push(input);
                }

                match runner.run(inputs).await {
                    Ok(true) => {
                        // continue the subsession
                    }
                    Ok(false) => {
                        let output = runner.finalize_output().await;
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(runner.runner.ctx(), session.id.clone(), output)
                                .await;
                        }
                        break;
                    }
                    Err(err) => {
                        let output = runner.latest_output();
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(runner.runner.ctx(), session.id.clone(), output)
                                .await;
                        }
                        log::error!("Error processing session {}: {:?}", session.id, err);
                        break;
                    }
                }
            }

            subsessions.remove_session(&session.id);
        });

        Ok(rt)
    }
}

pub trait SubAgentSet: Send + Sync {
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
    pub const NAME: &'static str = "subagents_manager";

    pub fn new() -> Self {
        Self {
            agents: RwLock::new(BTreeMap::new()),
            models: Vec::new(),
        }
    }

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

    pub async fn load(&self, ctx: AgentCtx) -> Result<(), BoxError> {
        let offset = Path::from("");
        let prefix = Self::store_prefix();
        if let Ok(agents) = ctx.root.store_list(Some(&prefix), &offset).await {
            for meta in agents {
                let (data, _) = ctx.root.store_get(&meta.location).await?;
                if let Ok(mut agent) = from_reader::<SubAgent, _>(&data[..]) {
                    let name = agent.name.to_ascii_lowercase();
                    self.preserve_runtime_state(&name, &mut agent);
                    self.agents.write().insert(name, agent);
                }
            }
        };

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

        let data = deterministic_cbor_into_vec(&agent)?;
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
                "required": ["operation", "name", "description", "instructions", "tools", "tags", "output_schema", "model", "effort", "task", "session", "persist"],
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

pub struct SubAgentSetManager {
    sets: RwLock<BTreeMap<TypeId, Arc<dyn SubAgentSet>>>,
}

impl Default for SubAgentSetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentSetManager {
    pub fn new() -> Self {
        Self {
            sets: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn insert<T: SubAgentSet + Sized + 'static>(&self, set: Arc<T>) -> Option<Arc<T>> {
        let type_id = TypeId::of::<T>();
        self.sets
            .write()
            .insert(type_id, set)
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
    }

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
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: Some(DynAgentHook::new(hook.clone())),
            runner: ctx.clone().completion_iter(req, Vec::new()).unbound(),
            last_output: None,
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
        assert!(runner.last_output.is_none());

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
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
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
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
        };

        assert!(runner.run(Vec::new()).await.unwrap());
        assert!(runner.run(Vec::new()).await.unwrap());
        assert_eq!(
            runner.runner.chat_history()[0].text().as_deref(),
            Some("compacted handoff")
        );

        let output = runner.finalize_output().await;
        assert_eq!(output.content, "seed task");
        assert_eq!(output.session.as_deref(), Some("session-1"));
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
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
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
        });

        let mut runner = SubSessionRunner {
            session,
            agent_hook: None,
            runner: ctx.completion_iter(req, Vec::new()).unbound(),
            last_output: None,
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
            deterministic_cbor_into_vec(&legacy_agent).unwrap().into(),
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
            let loaded = from_reader::<SubAgent, _>(&data[..]).unwrap();
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
        }
    }
}
