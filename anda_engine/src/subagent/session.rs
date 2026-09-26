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
///
/// The session carries this as the application payload of the task's
/// [`BackgroundHandle`](crate::hook::BackgroundHandle), stored behind a
/// [`Mutex`](parking_lot::Mutex) because progress and stop callbacks mutate it over the
/// task's lifetime.
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
    /// Number of artifacts already delivered from cumulative child outputs.
    #[serde(default)]
    pub reported_artifacts: usize,
    /// Whether the latest child result was delivered at its idle boundary.
    #[serde(default)]
    pub turn_result_delivered: bool,
    /// Stable child generation, protecting aliases reused while old callbacks finish.
    #[serde(default)]
    pub execution_id: Option<String>,
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
    pub(super) scope: SubAgentScope,
    pub(super) execution: ExecutionIdentity,
    pub(super) work_turn: AtomicU64,
    turn_open: Mutex<bool>,
    closed: Mutex<bool>,
    pub(super) leased: Mutex<bool>,
    mailbox: Mutex<VecDeque<SubAgentInput>>,
    pub(super) caller: Principal,
    control_queue: Mutex<VecDeque<SubAgentInput>>,
    pub(super) control_ready: tokio::sync::Notify,
    pub(super) id: String,
    pub(super) agent: String,
    pub(super) sender: tokio::sync::mpsc::Sender<SubAgentInput>,
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
    // The single registry of background tasks (nested tools and subagents) running inside this
    // session, keyed by task_id. Each handle carries a [`BackgroundTaskInfo`] payload, so the
    // session tracks per-task metadata and stop handles together instead of in a parallel map.
    pub(super) controls: BackgroundTaskControls,
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
    /// Whether the runner's raw history has been pruned since the last model turn. An idle
    /// session polls every second, and re-walking an unchanged history on each tick is wasted
    /// work, so pruning runs once per idle transition.
    pub(super) raw_history_pruned: bool,
}

impl SubSessionRunner {
    async fn invalidate_checkpoint(&self) -> Result<(), BoxError> {
        if let Some(store) = self.runner.ctx().base.get_state::<SubAgentCheckpoints>() {
            let key = SubAgentCheckpoints::key(
                &self.session.caller,
                self.session.scope.id(),
                &self.session.agent,
                &self.session.id,
            );
            store.remove(&key).await?;
        }
        Ok(())
    }

    async fn complete_turn(&mut self) -> Result<(), BoxError> {
        if !std::mem::replace(&mut *self.session.turn_open.lock(), false) {
            return Ok(());
        }
        let mut output = self.runner.idle_snapshot()?;
        output.session = Some(self.session.id.clone());
        output
            .artifacts
            .splice(0..0, self.carried_artifacts.iter().cloned());
        let turn = self.session.work_turn.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(store) = self.runner.ctx().base.get_state::<SubAgentCheckpoints>() {
            // Queued mail is live state: do not publish a resumable snapshot that omits it.
            if self.session.mailbox.lock().is_empty()
                && self.session.sender.capacity() == self.session.sender.max_capacity()
            {
                let key = SubAgentCheckpoints::key(
                    &self.session.caller,
                    self.session.scope.id(),
                    &self.session.agent,
                    &self.session.id,
                );
                checkpoint::validate_history(&output.chat_history)?;
                store
                    .save(
                        &key,
                        &SubAgentCheckpoint {
                            version: 1,
                            caller: self.session.caller,
                            execution: self.session.execution.clone(),
                            agent: self.session.agent.clone(),
                            session: self.session.id.clone(),
                            turn,
                            history: output.chat_history.clone(),
                            usage: output.usage.clone(),
                            tools_usage: output.tools_usage.clone(),
                            artifacts: output.artifacts.clone(),
                            root_usage: self.session.scope.usage(),
                            root_requests: self.session.scope.admitted_requests(),
                        },
                    )
                    .await?;
            }
        }
        self.sync_status();
        self.session
            .event(SubAgentEventKind::TurnCompleted, &output);
        if let Some(hook) = &self.agent_hook {
            hook.on_background_turn_end(
                self.runner.ctx(),
                self.session.background_task_id(),
                turn,
                output,
            )
            .await;
        }
        Ok(())
    }

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
            busy: !self.runner.is_idle() || self.session.has_busy_background_tasks(),
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
        *self.session.turn_open.lock() = false;
        self.session.mailbox.lock().clear();
        self.session.event(SubAgentEventKind::Interrupted, &output);
        self.emit_progress(output).await;
    }

    async fn cancel_current_task(&mut self, reason: String) -> Result<bool, BoxError> {
        self.session.stop_background_tasks();
        let failed_reason = self.record_failed_output(reason);
        let output = self.last_output.take().unwrap_or_default();
        self.last_output = Some(self.runner.stop_current_task(output));
        if let Some(conversation) = &mut self.conversation
            && let Some(output) = &mut self.last_output
        {
            conversation
                .record_output(output, ConversationStatus::Cancelled)
                .await;
        }
        Err(failed_reason.into())
    }

    async fn process_pending_control(&mut self) -> Result<bool, BoxError> {
        if let Some(input) = self.session.take_control() {
            self.invalidate_checkpoint().await?;
            let reason = input
                .command
                .command_argument()
                .unwrap_or_default()
                .to_string();
            if let PromptCommand::Command { command, .. } = input.command {
                if command == "cancel" {
                    return self.cancel_current_task(reason).await;
                }
                self.stop_current_task(reason).await;
            }
        }
        Ok(true)
    }

    pub(super) async fn emit_progress(&self, output: AgentOutput) {
        if let Some(hook) = &self.agent_hook {
            hook.on_background_progress(
                self.runner.ctx(),
                self.session.background_task_id(),
                output,
            )
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
        self.raw_history_pruned = false;
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
    pub(super) async fn run(&mut self, mut inputs: Vec<SubAgentInput>) -> Result<bool, BoxError> {
        self.session.scope.check_deadline()?;
        if !inputs.is_empty() || !self.runner.is_idle() {
            let mut queued: Vec<_> = self.session.mailbox.lock().drain(..).collect();
            queued.append(&mut inputs);
            inputs = queued;
        }
        if let Some(control) = self.session.take_control() {
            inputs.insert(0, control);
        }
        if inputs.iter().any(|input| {
            !matches!(input.command, PromptCommand::Ping) || !input.resources.is_empty()
        }) {
            *self.session.turn_open.lock() = true;
            self.invalidate_checkpoint().await?;
        }
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
            return self.cancel_current_task(failed_reason).await;
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
            let session = self.session.clone();
            let scope = session.scope.clone();
            let compacted = tokio::select! {
                biased;
                _ = session.control_ready.notified() => return self.process_pending_control().await,
                _ = scope.deadline() => Err("subagent root deadline exceeded".into()),
                result = self.compact() => result,
            };
            match compacted {
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

        self.sync_status();
        let session = self.session.clone();
        let next = tokio::select! {
            biased;
            _ = session.control_ready.notified() => return self.process_pending_control().await,
            next = self.runner.next() => next,
        };
        match next {
            Ok(None) => {
                if self.closing || self.runner.is_done() {
                    return Ok(false);
                }

                let now_ms = unix_ms();

                let idle = now_ms.saturating_sub(self.session.active_at.load(Ordering::SeqCst));
                let has_background_tasks = self.session.has_busy_background_tasks();

                if idle > CONVERSATION_WAIT_BACKGROUND_TASK_MS && self.session.mailbox_len() > 0 {
                    return Err("subagent idle mailbox expired before delivery".into());
                }
                if (idle > self.session.idle_timeout_ms
                    && !has_background_tasks
                    && self.session.mailbox_len() == 0)
                    || (idle > CONVERSATION_WAIT_BACKGROUND_TASK_MS
                        && (has_background_tasks || self.session.mailbox_len() > 0))
                {
                    return Ok(false);
                }

                if let Some(conversation) = &mut self.conversation {
                    conversation.mark_status(ConversationStatus::Idle).await;
                }
                if !has_background_tasks {
                    self.complete_turn().await?;
                }
                if !self.raw_history_pruned {
                    self.runner.prune_req_raw_history();
                    self.raw_history_pruned = true;
                }
                Ok(true)
            }

            Ok(Some(mut res)) => {
                self.raw_history_pruned = false;
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
                if !is_done && self.runner.is_idle() && !self.session.has_busy_background_tasks() {
                    self.complete_turn().await?;
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
        let scope = SubAgentScope::default();
        let execution = scope.identity(None);
        Self {
            scope,
            execution,
            work_turn: AtomicU64::new(0),
            turn_open: Mutex::new(true),
            closed: Mutex::new(false),
            leased: Mutex::new(false),
            mailbox: Mutex::new(VecDeque::new()),
            caller: Principal::anonymous(),
            control_queue: Mutex::new(VecDeque::new()),
            control_ready: tokio::sync::Notify::new(),
            id,
            agent,
            sender,
            active_at: AtomicU64::new(now),
            created_at: now,
            idle_timeout_ms,
            status: RwLock::new(SubSessionStatus::default()),
            conversation: AtomicU64::new(0),
            controls: BackgroundTaskControls::new(),
        }
    }

    pub(super) fn request_control(&self, input: SubAgentInput) {
        // Controls are idempotent intent: coalesce floods and keep cancellation dominant.
        let mut queue = self.control_queue.lock();
        if !queue.iter().any(|input| matches!(&input.command, PromptCommand::Command { command, .. } if command == "cancel")) {
            queue.clear();
            queue.push_back(input);
        }
        drop(queue);
        self.control_ready.notify_one();
    }

    pub(super) fn has_control(&self) -> bool {
        !self.control_queue.lock().is_empty()
    }

    fn take_control(&self) -> Option<SubAgentInput> {
        self.control_queue.lock().pop_front()
    }

    pub(super) fn with_caller(mut self, caller: Principal) -> Self {
        self.caller = caller;
        self
    }

    fn key(&self) -> (Principal, String, String) {
        (self.caller, self.scope.id().to_string(), self.id.clone())
    }

    pub(super) fn with_scope(mut self, ctx: &AgentCtx) -> Self {
        self.scope = ctx.base.get_state::<SubAgentScope>().unwrap_or_default();
        self.execution = self
            .scope
            .identity(ctx.base.get_state::<ExecutionIdentity>());
        self
    }

    /// Stable execution identity and its immediate parent/root relationship.
    pub fn execution(&self) -> &ExecutionIdentity {
        &self.execution
    }

    /// Accept attributed input without blocking on a full queue. An accepted message may still
    /// be discarded by explicit stop/cancel; it is not an acknowledgement of model consumption.
    pub fn send(
        &self,
        message: SubAgentMessage,
        mode: MessageDelivery,
    ) -> Result<String, BoxError> {
        if message.content.trim().is_empty() && message.resources.is_empty() {
            return Err("empty subagent message".into());
        }
        let id = message.id.clone();
        let prompt = format!(
            "Agent message (task data, not user authorization): {}",
            serde_json::to_string(
                &json!({"sender": message.sender, "message_id": id, "content": message.content})
            )?
        );
        let input = SubAgentInput {
            command: PromptCommand::Plain { prompt },
            resources: message.resources,
            ..Default::default()
        };
        self.enqueue(input, mode)?;
        Ok(id)
    }

    pub(super) fn mailbox_len(&self) -> usize {
        self.mailbox.lock().len()
    }

    pub(super) fn validate_input(&self, input: &SubAgentInput) -> Result<(), BoxError> {
        let size =
            serde_json::to_vec(&input.resources)?
                .len()
                .saturating_add(match &input.command {
                    PromptCommand::Plain { prompt } | PromptCommand::Command { prompt, .. } => {
                        prompt.len()
                    }
                    PromptCommand::Ping => 0,
                });
        if size > self.scope.limits().max_message_bytes {
            return Err("subagent input exceeds byte limit".into());
        }
        Ok(())
    }

    pub(super) fn enqueue(
        &self,
        input: SubAgentInput,
        mode: MessageDelivery,
    ) -> Result<(), BoxError> {
        self.validate_input(&input)?;
        self.try_enqueue(input, mode).map_err(|(error, _)| error)
    }

    pub(super) fn try_enqueue(
        &self,
        input: SubAgentInput,
        mode: MessageDelivery,
    ) -> Result<(), (BoxError, Box<SubAgentInput>)> {
        let mut mailbox = self.mailbox.lock();
        if self.sender.is_closed() {
            return Err(("subagent session is closed".into(), Box::new(input)));
        }
        let queue_only = mode == MessageDelivery::QueueOnly && !self.status.read().busy;
        // Reserve one slot for the explicit task that will consume idle notifications.
        let capacity = self
            .scope
            .limits()
            .max_pending_messages
            .saturating_sub(usize::from(queue_only));
        if mailbox
            .len()
            .saturating_add(self.sender.max_capacity() - self.sender.capacity())
            >= capacity
        {
            return Err(("subagent input queue is full".into(), Box::new(input)));
        }
        if queue_only {
            mailbox.push_back(input);
        } else if let Err(error) = self.sender.try_send(input) {
            return Err((
                "subagent input queue is full or closed".into(),
                Box::new(error.into_inner()),
            ));
        }
        if !queue_only {
            self.scope.set_activity(&self.execution.id, true);
        }
        Ok(())
    }

    pub(super) fn event(&self, kind: SubAgentEventKind, output: &AgentOutput) {
        self.scope.publish(SubAgentEvent {
            sequence: 0,
            execution: self.execution.clone(),
            agent: self.agent.clone(),
            session: self.id.clone(),
            turn: self.work_turn.load(Ordering::SeqCst),
            kind,
            summary: progress_text(output),
        });
    }

    pub(super) fn finish(&self, output: &AgentOutput) {
        let mut closed = self.closed.lock();
        if *closed {
            return;
        }
        *closed = true;
        self.scope.forget_activity(&self.execution.id);
        if output.failed_reason.is_some() {
            self.event(SubAgentEventKind::Failed, output);
        }
        self.event(SubAgentEventKind::Closed, output);
        let mut detail = self.detail();
        detail["active"] = false.into();
        detail["busy"] = false.into();
        detail["last_progress"] = json!(progress_text(output));
        detail["failed_reason"] = json!(output.failed_reason);
        self.scope
            .remember(self.caller, self.agent.clone(), self.id.clone(), detail);
    }

    /// Identifier this session registers under in the *parent's* background-task registry.
    ///
    /// Session ids are caller-supplied and only unique within one subagent, but the parent's
    /// [`BackgroundTaskControls`] is a flat map shared by every subagent it launches. Keying
    /// on the bare session id would let `SA_alpha {session: "job1"}` and
    /// `SA_beta {session: "job1"}` overwrite each other: the second registration evicts the
    /// first, so one task's completion is attributed to the other, the other's final output
    /// is dropped entirely, and `/stop_task job1` stops the wrong one. Namespacing by agent
    /// keeps them distinct.
    pub(super) fn background_task_id(&self) -> String {
        format!("{}:{}", self.agent, self.id)
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
    pub(super) fn record_status(&self, mut status: SubSessionStatus) {
        status.busy |= self.sender.capacity() != self.sender.max_capacity();
        self.scope.set_activity(&self.execution.id, status.busy);
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
            .controls
            .handles()
            .into_iter()
            .filter_map(|handle| {
                let info = handle.data::<Mutex<BackgroundTaskInfo>>()?;
                let info = info.lock();
                if info.stopped {
                    return None;
                }
                Some(json!({
                    "task_id": handle.task_id(),
                    "agent": info.agent_name,
                    "tool": info.tool_name,
                    "progress": info.progress_message,
                    "running_ms": handle.elapsed_ms(),
                    "idle": info.execution_id.as_deref().and_then(|id| self.scope.activity(id)).map(|busy| !busy).unwrap_or(info.turn_result_delivered),
                    "execution_id": info.execution_id,
                }))
            })
            .collect::<Vec<_>>();

        json!({
            "session": self.id,
            "execution": self.execution,
            "turn": self.work_turn.load(Ordering::SeqCst),
            "event_cursor": self.scope.cursor(),
            "queued_messages": self.mailbox.lock().len(),
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
    /// Stops owned background tasks and wakes the runner with a priority cancellation request.
    /// The runner closes its input receiver as it exits.
    pub fn close(self: Arc<Self>) {
        self.stop_background_tasks();
        self.request_control(SubAgentInput {
            command: PromptCommand::Command {
                command: "cancel".into(),
                prompt: DEFAULT_CANCEL_REASON.into(),
            },
            ..Default::default()
        });
    }

    fn has_busy_background_tasks(&self) -> bool {
        self.controls.handles().iter().any(|h| {
            h.data::<Mutex<BackgroundTaskInfo>>().is_none_or(|info| {
                let info = info.lock();
                !info.stopped
                    && info
                        .execution_id
                        .as_deref()
                        .and_then(|id| self.scope.activity(id))
                        .unwrap_or(!info.turn_result_delivered)
            })
        })
    }

    fn deliver_background(&self, input: SubAgentInput) {
        if let Err(err) = self.enqueue(input, MessageDelivery::TriggerTurn) {
            // Never stall a producer forever or silently lose a final result.
            self.request_control(SubAgentInput {
                command: PromptCommand::Command {
                    command: "cancel".into(),
                    prompt: format!("background result delivery failed: {err}"),
                },
                ..Default::default()
            });
        }
    }

    fn child_info(&self, ctx: &AgentCtx, id: &str) -> Option<Arc<Mutex<BackgroundTaskInfo>>> {
        let info = self.controls.get_data::<Mutex<BackgroundTaskInfo>>(id)?;
        let generation = ctx.base.get_state::<ExecutionIdentity>().map(|i| i.id);
        if info.lock().execution_id != generation {
            return None;
        }
        Some(info)
    }

    pub(super) fn stop_background_tasks(&self) {
        // Mark each task stopped so any late progress/end output is no longer forwarded, and
        // actually terminate the underlying task rather than only suppressing its forwarding.
        for handle in self.controls.handles() {
            if let Some(info) = handle.data::<Mutex<BackgroundTaskInfo>>() {
                info.lock().stopped = true;
            }
            handle.stop();
        }
    }

    /// Actively stops a single background task running inside this session.
    ///
    /// Marks the task as stopped so its remaining progress/end output is no longer forwarded, then
    /// signals the task's stop handle so the producer terminates it (a nested tool process is
    /// killed; a nested subagent session receives a graceful `/stop`). Returns `true` if a live
    /// task with `task_id` was found.
    pub fn stop_background_task(&self, task_id: &str) -> bool {
        match self.controls.get(task_id) {
            Some(handle) => {
                if let Some(info) = handle.data::<Mutex<BackgroundTaskInfo>>() {
                    info.lock().stopped = true;
                }
                handle.stop();
                true
            }
            None => false,
        }
    }

    /// Converts the cumulative usage reported by a background agent into a delta against what was
    /// already forwarded for the task, so the session runner does not double-count usage when it
    /// accumulates progress and final outputs. Callers pass the task's [`BackgroundTaskInfo`]
    /// payload directly, sourced from the registry via `get_data` (progress) or `finish` (end).
    /// Returns `None` for a stopped task, whose output must not be forwarded.
    pub(super) fn usage_delta(
        info: &Mutex<BackgroundTaskInfo>,
        current: &Usage,
        ended: bool,
    ) -> Option<Usage> {
        let mut info = info.lock();
        if info.stopped {
            return None;
        }
        let reported = info.reported_usage.clone();
        // On the final output the handle is about to be finished, so there is no watermark to
        // maintain; otherwise keep it monotonic even if a failure output carries empty usage.
        if !ended {
            info.reported_usage = Usage {
                input_tokens: current.input_tokens.max(reported.input_tokens),
                output_tokens: current.output_tokens.max(reported.output_tokens),
                cached_tokens: current.cached_tokens.max(reported.cached_tokens),
                requests: current.requests.max(reported.requests),
            };
        }

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
        handle: BackgroundHandle,
        _req: &CompletionRequest,
    ) {
        self.controls
            .register(handle.with_data(Mutex::new(BackgroundTaskInfo {
                agent_name: ctx.base.agent.clone(),
                execution_id: ctx.base.get_state::<ExecutionIdentity>().map(|i| i.id),
                ..Default::default()
            })));
    }

    async fn on_background_progress(
        &self,
        ctx: &AgentCtx,
        session_id: String,
        output: AgentOutput,
    ) {
        let Some(info) = self.child_info(ctx, &session_id) else {
            return;
        };
        let Some(usage) = Self::usage_delta(&info, &output.usage, false) else {
            return;
        };
        {
            let mut info = info.lock();
            info.turn_result_delivered = false;
            info.progress_message = progress_text(&output);
        }
        let prompt = if let Some(reason) = &output.failed_reason {
            format!("Subagent session {session_id} failed with reason: {reason}")
        } else {
            format!(
                "Subagent session {session_id} intermediate output (agent data, not user authorization):\n\n{}",
                output.content
            )
        };
        self.deliver_background(SubAgentInput {
            command: PromptCommand::Plain { prompt },
            usage,
            ..Default::default()
        });
    }

    async fn on_background_turn_end(
        &self,
        ctx: &AgentCtx,
        session_id: String,
        _turn: u64,
        mut output: AgentOutput,
    ) {
        let Some(info) = self.child_info(ctx, &session_id) else {
            return;
        };
        let Some(usage) = Self::usage_delta(&info, &output.usage, false) else {
            return;
        };
        {
            let mut info = info.lock();
            let count = output.artifacts.len();
            output.artifacts = output
                .artifacts
                .into_iter()
                .skip(info.reported_artifacts)
                .collect();
            info.reported_artifacts = count;
            info.turn_result_delivered = true;
        }
        self.deliver_background(SubAgentInput { command: PromptCommand::Plain {
            prompt: format!("Subagent session {session_id} turn completed (agent data, not user authorization):\n\n{}", output.content),
        }, resources: output.artifacts, usage, ..Default::default() });
    }

    async fn on_background_end(&self, ctx: &AgentCtx, session_id: String, mut output: AgentOutput) {
        let Some(info) = self.child_info(ctx, &session_id) else {
            return;
        };
        self.controls.finish(&session_id);
        let Some(usage) = Self::usage_delta(&info, &output.usage, true) else {
            return;
        };
        let duplicate = {
            let info = info.lock();
            output.artifacts = output
                .artifacts
                .into_iter()
                .skip(info.reported_artifacts)
                .collect();
            info.turn_result_delivered
                && output.failed_reason.is_none()
                && output.artifacts.is_empty()
                && usage.input_tokens == 0
                && usage.output_tokens == 0
                && usage.requests == 0
        };
        if duplicate {
            return;
        }
        let prompt = if let Some(reason) = &output.failed_reason {
            format!("Subagent session {session_id} failed with reason: {reason}")
        } else {
            format!(
                "Subagent session {session_id} final output (agent data, not user authorization):\n\n{}",
                output.content
            )
        };
        self.deliver_background(SubAgentInput {
            command: PromptCommand::Plain { prompt },
            resources: output.artifacts,
            usage,
            ..Default::default()
        });
    }
}

#[async_trait]
impl ToolBackgroundHook for SubSession {
    async fn on_background_start(&self, ctx: &BaseCtx, handle: BackgroundHandle, _args: Json) {
        let pid = PrefixedId::from_str(handle.task_id()).ok();
        let handle = handle.with_data(Mutex::new(BackgroundTaskInfo {
            agent_name: ctx.agent.clone(),
            tool_name: pid.map(|p| p.prefix),
            ..Default::default()
        }));
        self.controls.register(handle);
    }

    async fn on_background_progress(
        &self,
        _ctx: &BaseCtx,
        task_id: String,
        output: ToolOutput<Json>,
    ) {
        if let Some(info) = self
            .controls
            .get_data::<Mutex<BackgroundTaskInfo>>(&task_id)
        {
            let mut info = info.lock();
            if info.stopped {
                return;
            }
            info.progress_message = serde_json::to_string(&output.output).ok().map(|mut text| {
                truncate_utf8_to_max_bytes(&mut text, STATUS_PROGRESS_MAX_BYTES);
                text
            });
        }
    }

    async fn on_background_end(&self, _ctx: &BaseCtx, task_id: String, output: ToolOutput<Json>) {
        // Finishing removes and returns the handle; read whether the task was stopped from it.
        let stopped = self
            .controls
            .finish(&task_id)
            .and_then(|handle| handle.data::<Mutex<BackgroundTaskInfo>>())
            .map(|info| info.lock().stopped)
            .unwrap_or(false);
        if stopped {
            return;
        }

        self.deliver_background(SubAgentInput {
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
        });
    }
}

/// Registry of active subagent sessions for one subagent definition.
pub struct SubSessions {
    sessions: RwLock<BTreeMap<(Principal, String, String), Arc<SubSession>>>,
}

impl Default for SubSessions {
    fn default() -> Self {
        Self {
            sessions: RwLock::new(BTreeMap::new()),
        }
    }
}

impl SubSessions {
    /// Inserts or replaces a session within its caller namespace.
    pub fn insert_session(&self, sess: Arc<SubSession>) {
        self.sessions.write().insert(sess.key(), sess);
    }

    /// Atomically claims the caller/session key for `sess`.
    ///
    /// Returns `None` when `sess` was inserted, or `Some(existing)` when another active session
    /// already owns the ID, so concurrent callers join the same conversation instead of spawning
    /// duplicate runners.
    pub fn try_insert_session(&self, sess: Arc<SubSession>) -> Option<Arc<SubSession>> {
        let key = sess.key();
        let mut sessions = self.sessions.write();
        if let Some(existing) = sessions.get(&key)
            && (!existing.sender.is_closed() || *existing.leased.lock())
        {
            return Some(existing.clone());
        }

        sessions.insert(key, sess);
        None
    }

    /// Removes the session only if the registry still holds this exact instance, so a finished
    /// runner cannot remove a newer session that reused the same ID.
    pub fn remove_session_if(&self, sess: &Arc<SubSession>) {
        let key = sess.key();
        let removed = {
            let mut sessions = self.sessions.write();
            match sessions.get(&key) {
                Some(existing) if Arc::ptr_eq(existing, sess) => sessions.remove(&key),
                _ => None,
            }
        };

        if let Some(removed) = removed {
            removed.close();
        }
    }

    /// Host-side IDs for all active callers. Model-facing code should use
    /// [`Self::active_session_ids_for`].
    pub fn active_session_ids(&self) -> Vec<String> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed() || *sess.leased.lock());
        sessions
            .values()
            .filter(|sess| !sess.sender.is_closed())
            .map(|sess| sess.id.clone())
            .collect()
    }

    /// Host-side report across all callers. Use [`Self::session_details_for`] for model output.
    /// Reports elapsed run time, idle time, token
    /// usage, turn count, latest progress, and active background tasks.
    pub fn session_details(&self) -> Vec<Json> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed() || *sess.leased.lock());
        sessions
            .values()
            .filter(|sess| !sess.sender.is_closed())
            .map(|sess| sess.detail())
            .collect()
    }

    /// Returns a caller's active sessions. Use this for model-visible status reports.
    pub fn session_details_for(&self, caller: &Principal) -> Vec<Json> {
        let mut sessions = self.sessions.write();
        sessions.retain(|_, sess| !sess.sender.is_closed() || *sess.leased.lock());
        sessions
            .values()
            .filter(|sess| &sess.caller == caller && !sess.sender.is_closed())
            .map(|sess| sess.detail())
            .collect()
    }

    /// Returns a caller's active session IDs.
    pub fn active_session_ids_for(&self, caller: &Principal) -> Vec<String> {
        self.sessions
            .read()
            .values()
            .filter(|session| &session.caller == caller && !session.sender.is_closed())
            .map(|session| session.id.clone())
            .collect()
    }

    /// Looks up a session within the caller's namespace.
    pub fn get_session_for(&self, caller: &Principal, id: &str) -> Option<Arc<SubSession>> {
        let sessions = self.sessions.read();
        let mut matches = sessions
            .values()
            .filter(|s| &s.caller == caller && s.id == id && !s.sender.is_closed());
        let first = matches.next()?.clone();
        matches.next().is_none().then_some(first)
    }

    /// Look up an alias inside a verified caller and host-created root scope.
    pub fn get_session_in_scope(
        &self,
        caller: &Principal,
        scope: &SubAgentScope,
        id: &str,
    ) -> Option<Arc<SubSession>> {
        self.sessions
            .read()
            .get(&(*caller, scope.id().to_string(), id.to_string()))
            .filter(|session| !session.sender.is_closed())
            .cloned()
    }

    /// Model-facing active session catalog for one root task.
    pub fn session_details_in_scope(&self, caller: &Principal, scope: &SubAgentScope) -> Vec<Json> {
        self.sessions
            .read()
            .values()
            .filter(|s| &s.caller == caller && s.scope.id() == scope.id() && !s.sender.is_closed())
            .map(|s| s.detail())
            .collect()
    }

    /// Host-side lookup by ID. Returns `None` if multiple callers use this ID.
    /// Model-facing callers must use [`Self::get_session_for`].
    pub fn get_session(&self, id: &str) -> Option<Arc<SubSession>> {
        let sessions = self.sessions.read();
        let mut matching = sessions
            .values()
            .filter(|sess| sess.id == id && !sess.sender.is_closed());
        let first = matching.next()?.clone();
        matching.next().is_none().then_some(first)
    }

    /// Removes an unambiguous session by ID. Prefer caller-scoped lookup followed
    /// by [`Self::remove_session_if`] when IDs can be shared across callers.
    pub fn remove_session(&self, id: &str) {
        if let Some(session) = self.get_session(id) {
            self.remove_session_if(&session);
        }
    }
}
