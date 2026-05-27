use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, ContentPart,
    FunctionDefinition, Json, Message, Path, PromptCommand, PutMode, Resource, StoreFeatures,
    ToolOutput, Usage, prompt_with_resources, select_resources, validate_function_name,
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

    #[serde(skip)]
    pub subsessions: Arc<SubSessions>,
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
}

impl SubAgentArgs {
    fn from_prompt(prompt: String) -> Self {
        serde_json::from_str::<Self>(&prompt).unwrap_or(Self {
            prompt,
            session: String::new(),
        })
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SubAgentManagerArgs {
    pub name: String,
    pub description: String,
    pub instructions: String,

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default, deserialize_with = "deserialize_optional_json_schema")]
    pub output_schema: Option<Json>,

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
            name: self.name,
            description: self.description,
            instructions: self.instructions,
            tools: self.tools,
            tags: self.tags,
            output_schema: self.output_schema,
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

struct SubSessionRunner {
    session: Arc<SubSession>,
    agent_hook: Option<DynAgentHook>,
    runner: CompletionRunner,
    last_output: Option<AgentOutput>,
}

impl SubSessionRunner {
    // returns true if the conversation should continue to be active after processing the inputs, or false if it should be terminated
    async fn run(&mut self, inputs: Vec<SubAgentInput>) -> Result<bool, BoxError> {
        let mut cancellation_requested: Option<String> = None;
        if !inputs.is_empty() {
            self.session.active_at.store(unix_ms(), Ordering::SeqCst);
        }

        for mut input in inputs {
            // 累计来自于后台任务的工具使用情况
            self.runner.accumulate(&input.usage);

            match input.command {
                PromptCommand::Ping => {
                    // PING from the user to keep the conversation alive.
                    continue;
                }
                PromptCommand::Plain { prompt } => {
                    self.runner
                        .follow_up(prompt_with_resources(prompt, &mut input.resources));
                }
                PromptCommand::Command { command, prompt } => match command.as_str() {
                    "stop" | "cancel" => {
                        cancellation_requested = Some(prompt);
                        break;
                    }
                    "steer" => {
                        self.runner
                            .steer(prompt_with_resources(prompt, &mut input.resources));
                    }
                    _ => {
                        self.runner
                            .follow_up(prompt_with_resources(prompt, &mut input.resources));
                    }
                },
            }
        }

        if let Some(failed_reason) = cancellation_requested {
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
                    let output = self
                        .runner
                        .finalize(Some(COMPACTION_PROMPT.to_string()))
                        .await?;

                    if let Some(failed_reason) = output.failed_reason {
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

                    let req = self.runner.req();
                    let req = CompletionRequest {
                        instructions: req.instructions.clone(),
                        role: req.role.clone(),
                        chat_history: vec![compaction_msg.clone()],
                        tools: req.tools.clone(),
                        output_schema: req.output_schema.clone(),
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
                let failed_reason = err.to_string();
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
            description: self.description(),
            parameters: json!({
                "type": "object",
                "description": "Run this subagent on a focused task. Leave session empty for normal blocking mode, or provide a session ID for non-blocking session mode with hook-delivered progress and final output.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task for this subagent. Include the objective, relevant context, constraints, preferred workflow or deliverable, and any success criteria needed to complete the work."
                    },
                    "session": {
                        "type": "string",
                        "description": "Optional case-insensitive session ID. Leave empty for normal blocking mode. Provide a stable ID to run asynchronously, continue the same subagent conversation across calls, and receive progress/final output through hooks.",
                        "default": ""
                    },
                },
                "required": ["prompt", "session"],
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

        let session_id = args.session.trim().to_ascii_lowercase();
        if session_id.is_empty() {
            let rt = ctx
                .completion(
                    CompletionRequest {
                        instructions: self.instructions.clone(),
                        prompt: args.prompt,
                        content: resources
                            .into_iter()
                            .map(ContentPart::try_from)
                            .collect::<Result<Vec<_>, _>>()
                            .ok()
                            .unwrap_or_default(),
                        tools: ctx.definitions(Some(&self.tools)).await,
                        output_schema: self.output_schema.clone(),
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
        };

        let subsessions = self.subsessions.clone();
        if let Some(session) = subsessions.get_session(&session_id) {
            // Join existing conversation session if it's active
            match session.sender.send(input).await {
                Ok(_) => {
                    let rt = AgentOutput {
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
            command, resources, ..
        } = input;

        let prompt = match command {
            PromptCommand::Ping => return Err("prompt cannot be empty".into()),
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
                "subagent {} is running in the background with session mode. The output will be pushed to you through the hooks.",
                session.agent
            ),
            session: Some(session.id.clone()),
            ..Default::default()
        };

        let req = CompletionRequest {
            instructions: self.instructions.clone(),
            prompt,
            content: resources
                .into_iter()
                .map(ContentPart::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()
                .unwrap_or_default(),
            tools: ctx.definitions(Some(&self.tools)).await,
            output_schema: self.output_schema.clone(),
            ..Default::default()
        };

        if let Some(hook) = &agent_hook {
            hook.on_background_start(&ctx, &session.id, &req).await;
        }

        let rt = if let Some(hook) = &agent_hook {
            hook.after_agent_run(&ctx, rt).await?
        } else {
            rt
        };

        ctx.base.set_state(DynAgentHook::new(session.clone()));
        ctx.base.set_state(DynToolJsonHook::new(session.clone()));

        subsessions.insert_session(session.clone());
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
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(
                                runner.runner.ctx(),
                                session.id.clone(),
                                runner.last_output.take().unwrap_or_default(),
                            )
                            .await;
                        }
                        break;
                    }
                    Err(err) => {
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(
                                runner.runner.ctx(),
                                session.id.clone(),
                                runner.last_output.take().unwrap_or_default(),
                            )
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
        }
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
                if let Ok(agent) = from_reader::<SubAgent, _>(&data[..]) {
                    self.agents
                        .write()
                        .insert(agent.name.to_ascii_lowercase(), agent);
                }
            }
        };

        Ok(())
    }

    /// Creates or updates a subagent. The name is normalised to lowercase and validated. If an agent with the same name exists, it will be overwritten.
    pub async fn upsert(&self, ctx: AgentCtx, agent: SubAgent) -> Result<(), BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;

        let data = deterministic_cbor_into_vec(&agent)?;
        self.agents.write().insert(name.clone(), agent);

        ctx.root
            .store_put(&Self::store_path(&name), PutMode::Overwrite, data.into())
            .await?;
        Ok(())
    }

    /// Creates or updates an in-memory subagent without writing it to the store.
    pub fn upsert_temporary(&self, agent: SubAgent) -> Result<String, BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;

        self.agents.write().insert(name.clone(), agent);
        Ok(name)
    }

    fn description_text() -> String {
        "Create, update, optionally run, and optionally persist reusable subagents. Use this when a task would benefit from a focused temporary helper with stable instructions or a restricted toolset. By default the subagent is temporary for the current engine process and can be called immediately as `SA_<name>`. If it proves useful, call this again with `persist: true` to save or update it for future sessions and restarts.".to_string()
    }

    fn manager_definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: Self::NAME.to_string(),
            description: Self::description_text(),
            parameters: json!({
                "type": "object",
                "description": "Create or update a subagent configuration, optionally run it immediately, and optionally persist it for future reuse.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique callable subagent name. Must be lowercase snake_case, start with a letter, contain only letters, digits, or underscores, and be no longer than 64 characters. The subagent becomes callable as SA_<name>."
                    },
                    "description": {
                        "type": "string",
                        "description": "Short routing description shown when models decide whether to call this subagent. State when it should be used and what outcome it produces."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Durable system-style instructions for the subagent. Define its role, scope, workflow, constraints, decision rules, and expected output style. Write reusable guidance, not a one-off task prompt."
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
                    "task": {
                        "type": "string",
                        "description": "Optional immediate task to run with the newly created or updated subagent. Leave empty to only create or update the subagent.",
                        "default": ""
                    },
                    "session": {
                        "type": "string",
                        "description": "Optional session ID for the immediate task. Leave empty for normal blocking mode. Provide a stable ID for non-blocking session mode with hook-delivered progress and final output.",
                        "default": ""
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Set true to save or update this subagent for future calls and restarts. Leave false to keep it temporary in the current engine process.",
                        "default": false
                    }
                },
                "required": ["name", "description", "instructions", "tools", "tags", "output_schema", "task", "session", "persist"],
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
        let (agent, task, session, persist) = args.into_subagent();
        let name = agent.name.to_ascii_lowercase();

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
        Self::description_text()
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
        let (name, agent, task, session, persist) = self.configure(ctx.clone(), args).await?;
        let callable = format!("SA_{name}");
        let subagent = json!({
            "result": if persist { "persisted" } else { "created" },
            "name": name,
            "callable": callable,
            "persisted": persist,
            "hint": "Call the subagent by this callable name. If a temporary subagent proves useful, call subagents_manager again with persist=true to save it."
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
    let threshold = runner
        .model()
        .context_window
        .saturating_mul(8)
        .saturating_div(10)
        .max(100_000) as u64;

    current_usage.input_tokens >= threshold || runner.turns() >= MAX_TURNS_TO_COMPACT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use crate::model::{CompletionFeaturesDyn, Model};
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

    #[derive(Clone, Default)]
    struct RecordingAgentHook {
        progress: Arc<Mutex<Vec<(String, AgentOutput)>>>,
    }

    impl RecordingAgentHook {
        fn progress_events(&self) -> Vec<(String, AgentOutput)> {
            self.progress.lock().clone()
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
            output_schema: Some(json!({
                "type": "object",
                "additionalProperties": false
            })),
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

        assert!(runner.run(Vec::new()).await.unwrap());

        let progress = hook.progress_events();
        assert_eq!(progress.len(), 1);
        assert_eq!(progress[0].0, "session-1");
        assert_eq!(progress[0].1.content, "seed task");

        let recorded = requests.lock().clone();
        assert_eq!(recorded.len(), 2);
        assert_eq!(request_text(&recorded[0]), "seed task");
        assert_eq!(request_text(&recorded[1]).trim(), COMPACTION_PROMPT.trim());

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

    #[test]
    fn subagent_definition_guides_self_contained_prompts() {
        let agent = SubAgent {
            name: "research_assistant".to_string(),
            description: "Handles recurring research tasks with concise synthesis.".to_string(),
            instructions: "Research carefully and synthesize findings.".to_string(),
            tools: vec!["google_web_search".to_string()],
            ..Default::default()
        };

        let definition = agent.definition();

        assert_eq!(definition.name, "research_assistant");
        assert_eq!(
            definition.description,
            "Handles recurring research tasks with concise synthesis."
        );
        assert_eq!(
            definition.parameters["description"],
            json!(
                "Run this subagent on a focused task. Leave session empty for normal blocking mode, or provide a session ID for non-blocking session mode with hook-delivered progress and final output."
            )
        );
        assert_eq!(
            definition.parameters["properties"]["session"]["default"],
            json!("")
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
                "task": "",
                "session": "",
                "persist": false
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(args.output_schema, Some(schema));
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
        }
    }
}
