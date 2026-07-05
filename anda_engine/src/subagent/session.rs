use super::*;

#[derive(Default, Clone)]
pub(super) struct SubAgentInput {
    pub(super) command: PromptCommand,
    pub(super) resources: Vec<Resource>,
    pub(super) usage: Usage,
    pub(super) model: Option<String>,
    pub(super) effort: Option<ModelEffort>,
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
    /// Whether this task belonged to a stopped session task and should no longer be forwarded.
    #[serde(default)]
    pub stopped: bool,
}

/// Live progress snapshot for a subagent session.
///
/// The session runner refreshes this after each step so the parent agent can poll a session's
/// current state through `/status` (or the manager catalog) without waiting for hook callbacks.
#[derive(Debug, Default, Clone, Serialize)]
pub struct SubSessionStatus {
    /// Cumulative token usage across every turn in this session.
    pub usage: Usage,
    /// Per-tool token usage accumulated by the session runner.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub tools_usage: HashMap<String, Usage>,
    /// Number of completed model turns.
    pub turns: usize,
    /// Resolved model label currently driving the session, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// True while the runner is mid-task (an in-flight request or pending tool calls); false when
    /// it is idle and waiting for the next prompt.
    pub busy: bool,
    /// Latest visible output text from the subagent, truncated for a compact status view.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_progress: Option<String>,
}

/// Maximum number of bytes retained for [`SubSessionStatus::last_progress`].
pub(super) const STATUS_PROGRESS_MAX_BYTES: usize = 2000;

/// Long-lived conversation session for a subagent.
pub struct SubSession {
    pub(super) id: String,
    pub(super) agent: String,
    pub(super) sender: tokio::sync::mpsc::Sender<SubAgentInput>,
    // task_id -> BackgroundTaskInfo
    pub(super) background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
    pub(super) active_at: AtomicU64,
    // Wall-clock ms when this session's runner started, used to report elapsed run time.
    pub(super) created_at: u64,
    // Idle window in ms before an input-less, background-task-less session is reclaimed.
    pub(super) idle_timeout_ms: u64,
    // Live progress snapshot refreshed after each runner step so the parent can poll status
    // without waiting for hook callbacks.
    pub(super) status: RwLock<SubSessionStatus>,
    // Persistent conversation record for this session when conversation recording is enabled.
    pub(super) conversation: AtomicU64,
}

pub(super) fn resources_into_content(resources: Vec<Resource>) -> Vec<ContentPart> {
    resources
        .into_iter()
        .filter_map(|resource| ContentPart::try_from(resource).ok())
        .collect()
}

/// Extracts a short, human-readable progress line from an output for the live status snapshot.
/// Prefers visible content, then a failure reason, then reasoning thoughts; truncates with the
/// shared [`truncate_utf8_to_max_bytes`] helper (grapheme-cluster-safe) to keep the report compact.
pub(super) fn progress_text(output: &AgentOutput) -> Option<String> {
    let text = if !output.content.is_empty() {
        output.content.clone()
    } else if let Some(reason) = &output.failed_reason {
        format!("failed: {reason}")
    } else {
        output.thoughts.clone()?
    };

    let mut text = text.trim().to_string();
    if text.is_empty() {
        return None;
    }

    if truncate_utf8_to_max_bytes(&mut text, STATUS_PROGRESS_MAX_BYTES).is_some() {
        text.push('…');
    }
    Some(text)
}

pub(super) fn prompt_and_resources_into_content(
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

pub(super) struct SubSessionRunner {
    pub(super) session: Arc<SubSession>,
    pub(super) agent_hook: Option<DynAgentHook>,
    pub(super) runner: CompletionRunner,
    pub(super) conversation: Option<SubAgentConversationLog>,
    pub(super) last_output: Option<AgentOutput>,
    /// Artifacts rescued from runners that were replaced during context compaction. They are
    /// merged back into the session's final output.
    pub(super) carried_artifacts: Vec<Resource>,
    /// Set when the session decided to terminate; the runner then finishes the remaining queued
    /// inputs and exits at the next idle boundary instead of waiting for more input.
    pub(super) closing: bool,
}

impl SubSessionRunner {
    pub(super) fn with_session(&self, mut output: AgentOutput) -> AgentOutput {
        if output.session.is_none() {
            output.session = Some(self.session.id.clone());
        }

        output
    }

    pub(super) fn has_observable_output(output: &AgentOutput) -> bool {
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
    pub(super) fn has_reportable_output(output: &AgentOutput) -> bool {
        !output.content.is_empty()
            || output.thoughts.is_some()
            || output.failed_reason.is_some()
            || !output.tool_calls.is_empty()
            || !output.artifacts.is_empty()
    }

    pub(super) fn has_progress_signal(output: &AgentOutput) -> bool {
        !output.content.is_empty() || output.failed_reason.is_some()
    }

    /// Refreshes the session's live status snapshot from the runner's current state, so a `/status`
    /// poll reflects the latest usage, turn count, and visible progress without waiting for hooks.
    pub(super) fn sync_status(&self) {
        let last_progress = self
            .last_output
            .as_ref()
            .or_else(|| self.runner.last_output())
            .and_then(progress_text);
        let model = self
            .runner
            .last_output()
            .and_then(|output| output.model.clone())
            .or_else(|| self.runner.req().model.clone());

        self.session.record_status(SubSessionStatus {
            usage: self.runner.total_usage().clone(),
            tools_usage: self.runner.tools_usage().clone(),
            turns: self.runner.turns(),
            model,
            busy: !self.runner.is_idle(),
            last_progress,
        });
    }

    pub(super) fn latest_output(&mut self) -> AgentOutput {
        let output = self
            .last_output
            .take()
            .or_else(|| self.runner.last_output().cloned())
            .unwrap_or_default();

        self.with_session(output)
    }

    pub(super) fn merge_carried_artifacts(&mut self, output: &mut AgentOutput) {
        if !self.carried_artifacts.is_empty() {
            let mut artifacts = std::mem::take(&mut self.carried_artifacts);
            artifacts.append(&mut output.artifacts);
            output.artifacts = artifacts;
        }
    }

    pub(super) async fn finalize_output(&mut self) -> AgentOutput {
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

    pub(super) fn record_failed_output(&mut self, failed_reason: impl Into<String>) -> String {
        let mut failed_reason = failed_reason.into();
        if failed_reason.trim().is_empty() {
            failed_reason = DEFAULT_CANCEL_REASON.to_string();
        }

        let mut output = self.latest_output();
        output.content.clear();
        output.thoughts = None;
        output.failed_reason = Some(failed_reason.clone());

        self.last_output = Some(output);
        failed_reason
    }

    pub(super) async fn stop_current_task(&mut self, reason: impl Into<String>) {
        self.session.stop_background_tasks();

        let reason = reason.into();
        let content = if reason.trim().is_empty() {
            DEFAULT_STOP_REASON.to_string()
        } else {
            format!("Subagent session stopped: {}", reason.trim())
        };

        let output = self.with_session(AgentOutput {
            content,
            ..Default::default()
        });
        let mut output = self.runner.stop_current_task(output);
        if let Some(conversation) = &mut self.conversation {
            conversation
                .record_output(&mut output, ConversationStatus::Idle)
                .await;
        }
        self.last_output = Some(output.clone());
        self.emit_progress(output).await;
    }

    pub(super) async fn emit_progress(&self, output: AgentOutput) {
        if let Some(hook) = &self.agent_hook {
            hook.on_background_progress(self.runner.ctx(), self.session.id.clone(), output)
                .await;
        }
    }

    /// Summarizes the current conversation into a single handoff message and swaps in a fresh
    /// runner seeded with that summary, discarding the bloated history while preserving the
    /// session's accumulated usage, tool usage, and artifacts.
    ///
    /// Pending tool calls are executed before summarization, so the compaction turn does not
    /// strand an unanswered tool-call requirement.
    pub(super) async fn compact(&mut self) -> Result<(), BoxError> {
        let (runner, mut output) = match self.runner.handoff(None).await {
            Ok((runner, output)) => (runner, output),
            Err(err) => {
                let failed_reason = self.record_failed_output(err.to_string());
                return Err(failed_reason.into());
            }
        };

        // The old runner handed over the whole session's accumulated usage/tools_usage/artifacts on
        // finalize; rescue them first so nothing is lost even if the summary turns out unusable.
        self.runner = runner;
        self.runner.accumulate(&output.usage);
        self.runner.accumulate_tools_usage(&output.tools_usage);
        self.carried_artifacts.append(&mut output.artifacts);
        if let Some(conversation) = &mut self.conversation {
            conversation.reset_runner_history_cursor();
        }
        // Compaction is real work: refresh the activity clock so the idle-timeout check on the
        // turn that follows does not mistake the session for stale.
        self.session.active_at.store(unix_ms(), Ordering::SeqCst);
        Ok(())
    }

    // returns true if the conversation should continue to be active after processing the inputs, or false if it should be terminated
    pub(super) async fn run(&mut self, inputs: Vec<SubAgentInput>) -> Result<bool, BoxError> {
        let mut stop_requested: Option<String> = None;
        let mut cancellation_requested: Option<String> = None;
        if !inputs.is_empty() {
            self.session.active_at.store(unix_ms(), Ordering::SeqCst);
        }

        // Accumulate all follow-up/steer content for this batch instead of queueing it
        // input-by-input. Background results arrive as separate inputs and are drained into a
        // single run() call, so a batch can be far larger than any single input. Queueing each one
        // immediately defeated compaction: only the first input was size-checked, because attaching
        // it made the runner report not-idle and the rest bypassed the check. Sizing the whole
        // batch up front lets idle compaction run before the content is attached.
        let mut follow_up_batch: Vec<ContentPart> = Vec::new();
        let mut steer_batch: Vec<ContentPart> = Vec::new();

        for mut input in inputs {
            // Accumulate tool usage reported by background tasks.
            self.runner.accumulate(&input.usage);

            if input.model.is_some() {
                self.runner.set_model(input.model.take());
            }

            if input.effort.is_some() {
                self.runner.set_effort(input.effort);
            }

            if let PromptCommand::Command { command, .. } = &input.command {
                match command.as_str() {
                    "stop" => {
                        stop_requested = Some(
                            input
                                .command
                                .command_argument()
                                .unwrap_or_default()
                                .to_string(),
                        );
                        break;
                    }
                    "cancel" => {
                        cancellation_requested = Some(
                            input
                                .command
                                .command_argument()
                                .unwrap_or_default()
                                .to_string(),
                        );
                        break;
                    }
                    _ => {}
                }
            }

            match input.command {
                PromptCommand::Ping => {
                    follow_up_batch.extend(prompt_and_resources_into_content(
                        String::new(),
                        &mut input.resources,
                    ));
                    continue;
                }
                PromptCommand::Plain { prompt } => {
                    follow_up_batch.extend(prompt_and_resources_into_content(
                        prompt,
                        &mut input.resources,
                    ));
                }
                PromptCommand::Command { command, prompt } => match command.as_str() {
                    "steer" => {
                        steer_batch.extend(prompt_and_resources_into_content(
                            prompt,
                            &mut input.resources,
                        ));
                    }
                    _ => {
                        follow_up_batch.extend(prompt_and_resources_into_content(
                            prompt,
                            &mut input.resources,
                        ));
                    }
                },
            }
        }

        if let Some(failed_reason) = cancellation_requested {
            let failed_reason = self.record_failed_output(failed_reason);
            if let Some(conversation) = &mut self.conversation
                && let Some(output) = &mut self.last_output
            {
                conversation
                    .record_output(output, ConversationStatus::Cancelled)
                    .await;
            }
            return Err(failed_reason.into());
        }

        if let Some(reason) = stop_requested {
            self.stop_current_task(reason).await;
            return Ok(true);
        }

        // Compact (if needed) before attaching the batch, accounting for its estimated size, then
        // queue it. Running unconditionally also covers the case where the committed history grew
        // over the threshold without any new input this round.
        if self.runner.needs_compaction_with(|| {
            estimated_content_tokens(&follow_up_batch)
                .saturating_add(estimated_content_tokens(&steer_batch))
                .saturating_add(
                    self.runner
                        .steering_message_iter()
                        .map(|c| c.estimated_tokens() as u64)
                        .sum(),
                )
                .saturating_add(
                    self.runner
                        .follow_up_message_iter()
                        .map(|c| c.estimated_tokens() as u64)
                        .sum(),
                )
        }) {
            match self.compact().await {
                Ok(()) => {}
                Err(err) => {
                    if let Some(conversation) = &mut self.conversation {
                        conversation.record_failure(err.to_string()).await;
                    }
                    return Err(err);
                }
            }
        }
        let has_queued_work = !follow_up_batch.is_empty() || !steer_batch.is_empty();
        if !follow_up_batch.is_empty() {
            self.runner.follow_up_content(follow_up_batch);
        }
        if !steer_batch.is_empty() {
            self.runner.steer_content(steer_batch);
        }

        if let Some(conversation) = &mut self.conversation
            && (has_queued_work || !self.runner.is_idle())
        {
            conversation.mark_status(ConversationStatus::Working).await;
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

                if let Some(conversation) = &mut self.conversation {
                    conversation.mark_status(ConversationStatus::Idle).await;
                }
                self.runner.prune_req_raw_history();
                Ok(true)
            }

            Ok(Some(mut res)) => {
                let now_ms = unix_ms();
                self.session.active_at.store(now_ms, Ordering::SeqCst);
                res.session = Some(self.session.id.clone());
                let is_done = self.runner.is_done() || res.failed_reason.is_some();
                if let Some(conversation) = &mut self.conversation {
                    let status = if res.failed_reason.is_some() {
                        ConversationStatus::Failed
                    } else if is_done {
                        ConversationStatus::Completed
                    } else {
                        ConversationStatus::Working
                    };
                    conversation.record_output(&mut res, status).await;
                }
                self.last_output = Some(res.clone());
                if !is_done && Self::has_progress_signal(&res) {
                    self.emit_progress(res).await;
                }
                Ok(!is_done)
            }

            Err(err) => {
                let failed_reason = self.record_failed_output(err.to_string());
                if let Some(conversation) = &mut self.conversation
                    && let Some(output) = &mut self.last_output
                {
                    conversation
                        .record_output(output, ConversationStatus::Failed)
                        .await;
                }
                Err(failed_reason.into())
            }
        }
    }
}

impl SubSession {
    /// Creates a new session handle. `created_at` and `active_at` are stamped with the current
    /// time and the live status snapshot starts empty.
    pub(super) fn new(
        id: String,
        agent: String,
        sender: tokio::sync::mpsc::Sender<SubAgentInput>,
        idle_timeout_ms: u64,
    ) -> Self {
        let now = unix_ms();
        Self {
            id,
            agent,
            sender,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_at: AtomicU64::new(now),
            created_at: now,
            idle_timeout_ms,
            status: RwLock::new(SubSessionStatus::default()),
            conversation: AtomicU64::new(0),
        }
    }

    pub(super) fn set_conversation_id(&self, conversation: u64) {
        self.conversation.store(conversation, Ordering::SeqCst);
    }

    pub(super) fn conversation_id(&self) -> Option<u64> {
        match self.conversation.load(Ordering::SeqCst) {
            0 => None,
            id => Some(id),
        }
    }

    /// Overwrites the live progress snapshot with the runner's latest state.
    pub(super) fn record_status(&self, status: SubSessionStatus) {
        *self.status.write() = status;
    }

    /// Renders a synchronous, read-only status report for this session: elapsed run time, idle
    /// time, the latest progress snapshot, and any active background tasks. Used by the `/status`
    /// control command and the manager catalog so the parent can poll progress without waiting for
    /// hook callbacks.
    pub(super) fn detail(&self) -> Json {
        let now = unix_ms();
        let active_at = self.active_at.load(Ordering::SeqCst);
        let status = self.status.read().clone();
        let background_tasks = self
            .background_tasks
            .read()
            .iter()
            .filter(|(_, info)| !info.stopped)
            .map(|(task_id, info)| {
                json!({
                    "task_id": task_id,
                    "agent": info.agent_name,
                    "tool": info.tool_name,
                    "progress": info.progress_message,
                })
            })
            .collect::<Vec<_>>();

        json!({
            "session": self.id,
            "conversation": self.conversation_id(),
            "agent": self.agent,
            "active": true,
            "busy": status.busy,
            "running_ms": now.saturating_sub(self.created_at),
            "idle_ms": now.saturating_sub(active_at),
            "turns": status.turns,
            "model": status.model,
            "usage": status.usage,
            "tools_usage": status.tools_usage,
            "background_tasks": background_tasks,
            "last_progress": status.last_progress,
        })
    }

    /// Closes the session.
    ///
    /// This is a no-op today: the input channel closes on its own once every `sender` clone is
    /// dropped, and the runner exits at its next idle boundary. The method is kept as the single
    /// place to hook explicit teardown (e.g. cancelling the runner) if that becomes necessary.
    pub fn close(self: Arc<Self>) {}

    pub(super) fn stop_background_tasks(&self) {
        for info in self.background_tasks.write().values_mut() {
            info.stopped = true;
        }
    }

    /// Converts the cumulative usage reported by a background agent into a delta against what was
    /// already forwarded for `task_id`, so the session runner does not double-count usage when it
    /// accumulates progress and final outputs.
    pub(super) fn take_usage_delta(
        &self,
        task_id: &str,
        current: &Usage,
        ended: bool,
    ) -> Option<Usage> {
        let mut tasks = self.background_tasks.write();
        let reported = if ended {
            let info = tasks.remove(task_id).unwrap_or_default();
            if info.stopped {
                return None;
            }
            info.reported_usage
        } else {
            let info = tasks.entry(task_id.to_string()).or_default();
            if info.stopped {
                return None;
            }
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

        Some(Usage {
            input_tokens: current.input_tokens.saturating_sub(reported.input_tokens),
            output_tokens: current.output_tokens.saturating_sub(reported.output_tokens),
            cached_tokens: current.cached_tokens.saturating_sub(reported.cached_tokens),
            requests: current.requests.saturating_sub(reported.requests),
        })
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
                stopped: false,
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
        let Some(usage) = self.take_usage_delta(&session_id, &output.usage, false) else {
            return;
        };
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
        let Some(usage) = self.take_usage_delta(&session_id, &output.usage, true) else {
            return;
        };
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
                stopped: false,
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
            if info.stopped {
                return;
            }
            info.progress_message = serde_json::to_string(&output.output).ok();
        }
    }

    async fn on_background_end(&self, _ctx: &BaseCtx, task_id: String, output: ToolOutput<Json>) {
        let stopped = {
            self.background_tasks
                .write()
                .remove(&task_id)
                .map(|info| info.stopped)
                .unwrap_or(false)
        };
        if stopped {
            return;
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

    /// Returns a live status report for each active session: elapsed run time, idle time, token
    /// usage, turn count, latest progress, and active background tasks.
    pub fn session_details(&self) -> Vec<Json> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed());
        sessions.values().map(|sess| sess.detail()).collect()
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
