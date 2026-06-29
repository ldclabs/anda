use super::*;

/// Persistent conversation recorder used by subagents when installed in context state.
///
/// Engines that do not configure this recorder keep the existing in-memory-only subagent
/// behavior. When configured, each blocking subagent call and each background subagent session is
/// stored as a [`Conversation`] for operational visibility and later audit.
#[derive(Clone, Debug)]
pub struct SubAgentConversationRecorder {
    conversations: Conversations,
}

impl SubAgentConversationRecorder {
    /// Creates a recorder backed by the shared conversation store.
    pub fn new(conversations: Conversations) -> Self {
        Self { conversations }
    }

    pub(super) async fn start(
        &self,
        ctx: &AgentCtx,
        agent: &SubAgent,
        mode: &'static str,
        session: Option<&str>,
        req: &CompletionRequest,
        resources: Vec<Resource>,
    ) -> Result<SubAgentConversationLog, BoxError> {
        let now_ms = unix_ms();
        let session = session.map(str::to_string);
        let mut conversation = Conversation {
            user: ctx.base.caller,
            messages: initial_request_messages(req, now_ms),
            resources,
            status: ConversationStatus::Working,
            period: now_ms / 3600 / 1000,
            created_at: now_ms,
            updated_at: now_ms,
            label: Some(format!("subagent:{}", agent.name.to_ascii_lowercase())),
            extra: Some(json!({
                "kind": "subagent",
                "subagent": agent.name.to_ascii_lowercase(),
                "session": session,
                "mode": mode,
                "parent_agent": ctx.root.agent.clone(),
                "context_agent": ctx.base.agent.clone(),
                "model": req.model.clone(),
                "effort": req.effort,
                "tools": agent.tools.clone(),
                "tags": agent.tags.clone(),
                "has_output_schema": agent.output_schema.is_some(),
            })),
            ..Default::default()
        };

        let id = self
            .conversations
            .add_conversation(ConversationRef::from(&conversation))
            .await?;
        conversation._id = id;

        Ok(SubAgentConversationLog {
            recorder: self.clone(),
            conversation,
            persisted_runner_history_len: 0,
            replace_initial_input: true,
        })
    }
}

pub(super) struct SubAgentConversationLog {
    recorder: SubAgentConversationRecorder,
    pub(super) conversation: Conversation,
    persisted_runner_history_len: usize,
    replace_initial_input: bool,
}

impl SubAgentConversationLog {
    pub(super) fn id(&self) -> u64 {
        self.conversation._id
    }

    pub(super) fn reset_runner_history_cursor(&mut self) {
        self.persisted_runner_history_len = 0;
        self.replace_initial_input = false;
    }

    pub(super) async fn mark_status(&mut self, status: ConversationStatus) {
        if self.conversation.status == status {
            return;
        }

        self.conversation.status = status;
        self.conversation.updated_at = unix_ms();
        self.persist().await;
    }

    pub(super) async fn record_output(
        &mut self,
        output: &mut AgentOutput,
        status: ConversationStatus,
    ) {
        append_runner_history(
            &mut self.conversation,
            &output.chat_history,
            &mut self.persisted_runner_history_len,
            &mut self.replace_initial_input,
        );
        self.conversation.status = status;
        self.conversation.usage = output.usage.clone();
        self.conversation.artifacts = output.artifacts.clone();
        self.conversation.updated_at = unix_ms();
        self.conversation.failed_reason = output.failed_reason.clone();
        merge_output_metadata(&mut self.conversation, output);
        self.persist().await;
        output.conversation = Some(self.conversation._id);
    }

    pub(super) async fn record_failure(&mut self, reason: String) {
        self.conversation.status = ConversationStatus::Failed;
        self.conversation.failed_reason = Some(reason);
        self.conversation.updated_at = unix_ms();
        self.persist().await;
    }

    async fn persist(&self) {
        match self.conversation.to_changes() {
            Ok(changes) => {
                if let Err(err) = self
                    .recorder
                    .conversations
                    .update_conversation(self.conversation._id, changes)
                    .await
                {
                    log::warn!(
                        "failed to update subagent conversation {}: {err}",
                        self.conversation._id
                    );
                }
            }
            Err(err) => {
                log::warn!(
                    "failed to encode subagent conversation {} changes: {err}",
                    self.conversation._id
                );
            }
        }
    }
}

fn initial_request_messages(req: &CompletionRequest, timestamp: u64) -> Vec<Json> {
    let mut content = req.content.clone();
    if !req.prompt.is_empty() {
        content.insert(0, req.prompt.clone().into());
    }

    if content.is_empty() {
        Vec::new()
    } else {
        vec![json!(Message {
            role: req.role.clone().unwrap_or_else(|| "user".to_string()),
            content,
            timestamp: Some(timestamp),
            ..Default::default()
        })]
    }
}

fn append_runner_history(
    conversation: &mut Conversation,
    chat_history: &[Message],
    persisted_runner_history_len: &mut usize,
    replace_existing: &mut bool,
) {
    if chat_history.is_empty() {
        return;
    }

    if *replace_existing {
        conversation.messages.clear();
        *replace_existing = false;
    }

    // Runner output is cumulative only for the current runner. After compaction, the replacement
    // runner starts from the handoff summary, so a shorter incoming history means a new runner's
    // full visible history should be appended rather than treated as a duplicate prefix.
    let incoming_len = chat_history.len();
    let new_messages = if incoming_len >= *persisted_runner_history_len {
        chat_history[*persisted_runner_history_len..].to_vec()
    } else {
        chat_history.to_vec()
    };
    conversation.append_messages(new_messages);
    *persisted_runner_history_len = incoming_len;
}

fn merge_output_metadata(conversation: &mut Conversation, output: &AgentOutput) {
    let mut extra = conversation.extra.take().unwrap_or_else(|| json!({}));
    let Some(map) = extra.as_object_mut() else {
        conversation.extra = Some(extra);
        return;
    };

    if let Some(model) = &output.model {
        map.insert("model".to_string(), model.clone().into());
    }
    if let Some(session) = &output.session {
        map.insert("session".to_string(), session.clone().into());
    }
    if !output.tools_usage.is_empty() {
        map.insert(
            "tools_usage".to_string(),
            serde_json::to_value(&output.tools_usage).unwrap_or_default(),
        );
    }

    conversation.extra = Some(extra);
}
