use super::*;

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
    /// (60 seconds); any positive value is clamped to the background-task wait
    /// ceiling (1 hour) so a session can never outlive it. Only
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
                        "description": "Self-contained task handoff for this subagent. Include objective, context/resources, constraints, dependencies, expected deliverable, success criteria, and what progress/final output should contain. To control or inspect an already-running session, send a control command here instead of a task: `/status` to fetch the session's live progress (elapsed time, token usage, turns, active background tasks) synchronously without disturbing the run, `/steer <guidance>` to adjust course mid-run, `/stop_task <task_id>` to stop one specific background task (a task_id from `/status`) while leaving the session and its other background tasks running, `/stop <reason>` to stop the current task and keep the session idle, or `/cancel <reason>` to end the session runner."
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

            let input_resources = resources.clone();
            let req = CompletionRequest {
                instructions: self.instructions.clone(),
                prompt: args.prompt,
                content: resources_into_content(resources),
                tools: ctx.definitions(Some(&self.tools)).await,
                output_schema: self.output_schema.clone(),
                model,
                effort,
                ..Default::default()
            };

            let rt = if let Some(recorder) = ctx.base.get_state::<SubAgentConversationRecorder>() {
                let mut conversation = recorder
                    .start(&ctx, self, "blocking", None, &req, input_resources)
                    .await?;
                let mut runner = ctx.clone().completion_iter(req, Vec::new());
                let mut last: Option<AgentOutput> = None;

                loop {
                    match runner.next().await {
                        Ok(Some(mut output)) => {
                            let status = if output.failed_reason.is_some() {
                                ConversationStatus::Failed
                            } else if runner.is_done() {
                                ConversationStatus::Completed
                            } else {
                                ConversationStatus::Working
                            };
                            conversation.record_output(&mut output, status).await;
                            let terminal = runner.is_done() || output.failed_reason.is_some();
                            last = Some(output);
                            if terminal {
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(err) => {
                            conversation.record_failure(err.to_string()).await;
                            return Err(err);
                        }
                    }
                }

                match last {
                    Some(output) => output,
                    None => {
                        let reason = "completion runner returned no output".to_string();
                        conversation.record_failure(reason.clone()).await;
                        return Err(reason.into());
                    }
                }
            } else {
                ctx.completion(req, Vec::new()).await?
            };

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

        // `/status` is a read-only poll: report the session's live snapshot synchronously without
        // enqueuing anything into the runner, so the parent can check progress (elapsed time, token
        // usage, turns, background tasks) without waiting for hook callbacks.
        if let PromptCommand::Command { command, .. } = &input.command
            && command == "status"
        {
            let rt = match subsessions.get_session(&session_id) {
                Some(session) => AgentOutput {
                    content: session.detail().to_string(),
                    conversation: session.conversation_id(),
                    session: Some(session_id.clone()),
                    ..Default::default()
                },
                None => AgentOutput {
                    content: json!({
                        "session": session_id,
                        "agent": agent,
                        "active": false,
                        "note": "session is not active (it may have finished or expired); call again with a non-empty prompt to start a new session."
                    })
                    .to_string(),
                    session: Some(session_id.clone()),
                    ..Default::default()
                },
            };
            if let Some(hook) = &agent_hook {
                return hook.after_agent_run(&ctx, rt).await;
            }
            return Ok(rt);
        }

        // `/stop_task <task_id>` stops a single background task running inside the session (a nested
        // tool process or nested subagent) without disturbing the session's own turn or its other
        // background tasks. Handled synchronously so the parent gets immediate found/not-found
        // feedback, mirroring `/status`.
        if let PromptCommand::Command { command, .. } = &input.command
            && command == "stop_task"
        {
            let task_id = input
                .command
                .command_argument()
                .unwrap_or_default()
                .trim()
                .to_string();
            let rt = match subsessions.get_session(&session_id) {
                Some(session) if task_id.is_empty() => AgentOutput {
                    content: format!(
                        "subagent {agent} session {session_id}: /stop_task requires a task_id argument (see the session's background_tasks in /status)."
                    ),
                    conversation: session.conversation_id(),
                    session: Some(session_id.clone()),
                    ..Default::default()
                },
                Some(session) => {
                    let stopped = session.stop_background_task(&task_id);
                    let content = if stopped {
                        format!(
                            "Stopped background task {task_id} in subagent {agent} session {session_id}."
                        )
                    } else {
                        format!(
                            "No active background task {task_id} in subagent {agent} session {session_id} (it may have finished or the id is unknown)."
                        )
                    };
                    AgentOutput {
                        content,
                        conversation: session.conversation_id(),
                        session: Some(session_id.clone()),
                        ..Default::default()
                    }
                }
                None => AgentOutput {
                    content: format!(
                        "subagent {agent} session {session_id} is not active (it may have finished or expired); nothing to stop."
                    ),
                    session: Some(session_id.clone()),
                    ..Default::default()
                },
            };
            if let Some(hook) = &agent_hook {
                return hook.after_agent_run(&ctx, rt).await;
            }
            return Ok(rt);
        }

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
                            conversation: session.conversation_id(),
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
                PromptCommand::Command { command, .. } if command == "stop" => Some("stop"),
                PromptCommand::Command { command, .. } if command == "cancel" => Some("cancel"),
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
            let candidate = Arc::new(SubSession::new(
                session_id.clone(),
                agent.clone(),
                sender,
                resolve_idle_timeout_ms(self.idle_timeout),
            ));
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

        let input_resources = resources.clone();
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

        let mut conversation =
            if let Some(recorder) = ctx.base.get_state::<SubAgentConversationRecorder>() {
                match recorder
                    .start(
                        &ctx,
                        self,
                        "session",
                        Some(&session.id),
                        &req,
                        input_resources,
                    )
                    .await
                {
                    Ok(conversation) => {
                        session.set_conversation_id(conversation.id());
                        Some(conversation)
                    }
                    Err(err) => {
                        subsessions.remove_session_if(&session);
                        return Err(err);
                    }
                }
            } else {
                None
            };

        let rt = AgentOutput {
            content: format!(
                "subagent {} is running in the background with session mode (session: {}). The output will be pushed to you through the hooks.",
                session.agent, session.id
            ),
            conversation: conversation.as_ref().map(SubAgentConversationLog::id),
            session: Some(session.id.clone()),
            ..Default::default()
        };

        let rt = if let Some(hook) = &agent_hook {
            match hook.after_agent_run(&ctx, rt).await {
                Ok(rt) => rt,
                Err(err) => {
                    if let Some(conversation) = &mut conversation {
                        conversation.record_failure(err.to_string()).await;
                    }
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

        // Per-session stop token handed to the parent subscriber. It is a child of the session
        // context token but is NOT the token driving the runner, so cancelling it does not hard
        // abort an in-flight model request. Instead a bridge task translates cancellation into a
        // graceful `/stop`, matching the semantics of the `/stop` control command.
        let session_token = ctx.base.cancellation_token().child_token();
        if let Some(hook) = &agent_hook {
            let handle = BackgroundHandle::new(&session.id, session_token.clone());
            hook.on_background_start(&ctx, handle, &req).await;
        }

        {
            // Deliver a graceful `/stop` to this session when its stop handle fires. The task is
            // released when `session_token` is cancelled (either by the parent stopping the task or
            // by the runner cancelling it on exit), so it never outlives the session.
            let stop_sender = session.sender.clone();
            let stop_token = session_token.clone();
            tokio::spawn(async move {
                stop_token.cancelled().await;
                let _ = stop_sender
                    .send(SubAgentInput {
                        command: PromptCommand::Command {
                            command: "stop".to_string(),
                            prompt: String::new(),
                        },
                        ..Default::default()
                    })
                    .await;
            });
        }

        let runner = ctx.clone().completion_iter(req, vec![]).unbound();
        tokio::spawn(async move {
            let mut runner = SubSessionRunner {
                session: session.clone(),
                agent_hook,
                runner,
                conversation,
                last_output: None,
                carried_artifacts: Vec::new(),
                closing: false,
            };

            // Publish an initial snapshot so a `/status` poll right after launch reports the
            // session as running even before the first step completes.
            runner.sync_status();

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

                let result = runner.run(inputs).await;
                // Refresh the live snapshot after every step so a concurrent `/status` poll sees
                // the latest usage, turn count, and progress.
                runner.sync_status();
                match result {
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

                        let mut output = runner.finalize_output().await;
                        if let Some(conversation) = &mut runner.conversation {
                            let status = if output.failed_reason.is_some() {
                                ConversationStatus::Failed
                            } else {
                                ConversationStatus::Completed
                            };
                            conversation.record_output(&mut output, status).await;
                        }
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(runner.runner.ctx(), session.id.clone(), output)
                                .await;
                        }
                        break;
                    }
                    Err(err) => {
                        let mut output = runner.latest_output();
                        runner.merge_carried_artifacts(&mut output);
                        if let Some(conversation) = &mut runner.conversation {
                            let status = if conversation.conversation.status
                                == ConversationStatus::Cancelled
                            {
                                ConversationStatus::Cancelled
                            } else {
                                ConversationStatus::Failed
                            };
                            conversation.record_output(&mut output, status).await;
                        }
                        if let Some(hook) = &runner.agent_hook {
                            hook.on_background_end(runner.runner.ctx(), session.id.clone(), output)
                                .await;
                        }
                        log::error!("Error processing session {}: {:?}", session.id, err);
                        break;
                    }
                }
            }

            // Release the stop-bridge task; the session is finished so its stop token is moot.
            session_token.cancel();
            subsessions.remove_session_if(&session);
        });

        Ok(rt)
    }
}
