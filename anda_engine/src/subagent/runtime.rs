//! Root-scoped identity, bounded observation, and atomic resource admission.
use super::*;
use std::time::Duration;

/// Host policy shared by a root execution and all of its workers.
#[derive(Clone, Debug)]
pub struct SubAgentLimits {
    /// Maximum resident background sessions across all worker definitions.
    pub max_sessions: usize,
    /// Maximum simultaneous model requests, including root and compaction requests.
    pub max_parallel_requests: usize,
    /// Maximum admitted model requests over this scope's lifetime.
    pub max_requests: Option<u64>,
    /// Input plus output token budget. In-flight responses may overshoot this limit.
    pub max_tokens: Option<u64>,
    /// Absolute wall-clock deadline in Unix milliseconds, including idle time.
    pub deadline_ms: Option<u64>,
    /// Maximum encoded bytes per input, including resource payloads.
    pub max_message_bytes: usize,
    /// Maximum pending inputs per session.
    pub max_pending_messages: usize,
    /// Maximum retained events and terminal snapshots per scope.
    pub max_events: usize,
    /// Terminal snapshot retention in seconds.
    pub terminal_retention_secs: u64,
}
impl Default for SubAgentLimits {
    fn default() -> Self {
        Self {
            max_sessions: 64,
            max_parallel_requests: 8,
            max_requests: None,
            max_tokens: None,
            deadline_ms: None,
            max_message_bytes: 64 * 1024,
            max_pending_messages: 42,
            max_events: 128,
            terminal_retention_secs: 3600,
        }
    }
}

/// Stable worker instance identity, independent of its reusable definition and session alias.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExecutionIdentity {
    /// Opaque instance identifier.
    pub id: String,
    /// Host-created root execution scope.
    pub root_id: String,
    /// Immediate delegating worker, if any.
    pub parent_id: Option<String>,
}

/// Lifecycle events distinguish a completed work turn from a closed session.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SubAgentEventKind {
    /// Initial input accepted.
    Started,
    /// A completed work turn; the session may accept more work.
    TurnCompleted,
    /// Current work was explicitly stopped, leaving the session reusable.
    Interrupted,
    /// Execution failed or was cancelled.
    Failed,
    /// The session has finished; observer callbacks may still be completing cleanup.
    Closed,
}
/// Compact event; full conversation history never enters the event channel.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubAgentEvent {
    /// Monotonic sequence within the scope.
    pub sequence: u64,
    /// Worker identity.
    pub execution: ExecutionIdentity,
    /// Reusable worker name.
    pub agent: String,
    /// Caller-provided session alias (empty for blocking calls).
    pub session: String,
    /// Completed work-turn counter, distinct from model request count.
    pub turn: u64,
    /// Transition type.
    pub kind: SubAgentEventKind,
    /// Bounded visible result or failure preview.
    pub summary: Option<String>,
}
/// Result of an event read or bounded wait.
#[derive(Clone, Debug, Serialize)]
pub struct SubAgentEvents {
    /// Last scope sequence observed; pass this to the next read.
    pub cursor: u64,
    /// Whether earlier events were evicted before this read.
    pub lagged: bool,
    /// Whether the wait expired without satisfying its condition.
    pub timed_out: bool,
    /// Matching retained events in order.
    pub events: Vec<SubAgentEvent>,
}
/// Condition used by [`SubAgentScope::wait`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitMode {
    /// Return on any matching event.
    Any,
    /// Return after every specified execution has a terminal turn or session event.
    All,
}
/// Whether input may start an idle worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageDelivery {
    /// Keep input queued until the worker is active or receives a follow-up task.
    QueueOnly,
    /// Start work if idle; otherwise deliver at a safe runner boundary.
    TriggerTurn,
}
/// Attributed worker data. The envelope is not a new user authorization.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubAgentMessage {
    /// Origin assigned by the host or sending execution.
    pub sender: String,
    /// Unique acceptance identifier. Acceptance does not imply model consumption.
    pub id: String,
    /// Message text.
    pub content: String,
    /// Explicit attachments; never include hidden conversation history.
    #[serde(default)]
    pub resources: Vec<Resource>,
}

#[derive(Default)]
struct ScopeState {
    owner: Option<Principal>,
    activity: HashMap<String, bool>,
    sessions: usize,
    requests_in_flight: usize,
    requests_admitted: u64,
    usage: Usage,
    sequence: u64,
    events: VecDeque<SubAgentEvent>,
    terminals: BTreeMap<(Principal, String, String), (u64, Json)>,
}
struct ScopeInner {
    id: String,
    limits: SubAgentLimits,
    state: Mutex<ScopeState>,
    changed: tokio::sync::watch::Sender<u64>,
}
/// Host-created task scope. Install a clone in context state to continue the same root task.
///
/// Engine entry contexts get independent scopes by default. Never derive a scope ID from
/// model input or untrusted request metadata. A scope must belong to one caller.
#[derive(Clone)]
pub struct SubAgentScope(Arc<ScopeInner>);
impl Default for SubAgentScope {
    fn default() -> Self {
        Self::new(SubAgentLimits::default())
    }
}
impl SubAgentScope {
    /// Creates a fresh scope with an opaque random ID.
    pub fn new(limits: SubAgentLimits) -> Self {
        Self::restore(format!("root_{:032x}", rand::random::<u128>()), limits)
    }
    /// Reattaches a trusted persisted root identity under current host limits.
    /// Budget counters must be restored separately with [`Self::restore_usage`].
    pub fn restore(id: String, limits: SubAgentLimits) -> Self {
        let (changed, _) = tokio::sync::watch::channel(0);
        Self(Arc::new(ScopeInner {
            id,
            limits,
            state: Mutex::new(ScopeState::default()),
            changed,
        }))
    }
    /// Root execution ID.
    pub fn id(&self) -> &str {
        &self.0.id
    }
    /// Current host limits.
    pub fn limits(&self) -> &SubAgentLimits {
        &self.0.limits
    }
    /// Restore a trusted aggregate budget watermark, without counting child reports again.
    pub fn restore_usage(&self, usage: Usage, admitted_requests: u64) {
        let mut state = self.0.state.lock();
        state.usage.input_tokens = state.usage.input_tokens.max(usage.input_tokens);
        state.usage.output_tokens = state.usage.output_tokens.max(usage.output_tokens);
        state.usage.cached_tokens = state.usage.cached_tokens.max(usage.cached_tokens);
        state.usage.requests = state.usage.requests.max(usage.requests);
        state.requests_admitted = state.requests_admitted.max(admitted_requests);
    }
    /// Aggregate usage from actual inference responses, including compaction.
    pub fn usage(&self) -> Usage {
        self.0.state.lock().usage.clone()
    }
    /// Number of model attempts admitted, including failed/cancelled attempts.
    pub fn admitted_requests(&self) -> u64 {
        self.0.state.lock().requests_admitted
    }
    pub(crate) fn bind_caller(&self, caller: Principal) -> Result<(), BoxError> {
        let mut state = self.0.state.lock();
        if state.owner.is_some_and(|owner| owner != caller) {
            return Err("subagent scope belongs to another caller".into());
        }
        state.owner = Some(caller);
        Ok(())
    }
    pub(super) fn set_activity(&self, id: &str, busy: bool) {
        self.0.state.lock().activity.insert(id.into(), busy);
    }
    pub(super) fn activity(&self, id: &str) -> Option<bool> {
        self.0.state.lock().activity.get(id).copied()
    }
    pub(super) fn forget_activity(&self, id: &str) {
        self.0.state.lock().activity.remove(id);
    }
    pub(super) fn identity(&self, parent: Option<ExecutionIdentity>) -> ExecutionIdentity {
        ExecutionIdentity {
            id: format!("exec_{:032x}", rand::random::<u128>()),
            root_id: self.id().into(),
            parent_id: parent.filter(|p| p.root_id == self.id()).map(|p| p.id),
        }
    }
    pub(crate) async fn deadline(&self) {
        match self.limits().deadline_ms {
            Some(deadline) => {
                tokio::time::sleep(Duration::from_millis(deadline.saturating_sub(unix_ms()))).await
            }
            None => std::future::pending::<()>().await,
        }
    }
    /// Latest observation cursor without copying events.
    pub fn cursor(&self) -> u64 {
        self.0.state.lock().sequence
    }
    pub(crate) fn check_deadline(&self) -> Result<(), BoxError> {
        if self
            .limits()
            .deadline_ms
            .is_some_and(|deadline| unix_ms() >= deadline)
        {
            return Err("subagent root deadline exceeded".into());
        }
        Ok(())
    }
    pub(super) fn reserve_session(&self) -> Result<ScopePermit, BoxError> {
        self.check_deadline()?;
        let mut state = self.0.state.lock();
        if state.sessions >= self.limits().max_sessions {
            return Err("subagent resident session limit reached".into());
        }
        state.sessions += 1;
        Ok(ScopePermit {
            scope: self.clone(),
            session: true,
        })
    }
    pub(crate) fn admit_request(&self) -> Result<ScopePermit, BoxError> {
        self.check_deadline()?;
        let mut state = self.0.state.lock();
        if self
            .limits()
            .max_requests
            .is_some_and(|n| state.requests_admitted >= n)
            || self.limits().max_tokens.is_some_and(|n| {
                state
                    .usage
                    .input_tokens
                    .saturating_add(state.usage.output_tokens)
                    >= n
            })
        {
            return Err("subagent root model budget exhausted".into());
        }
        if state.requests_in_flight >= self.limits().max_parallel_requests {
            return Err("subagent parallel model request limit reached".into());
        }
        state.requests_in_flight += 1;
        state.requests_admitted = state.requests_admitted.saturating_add(1);
        Ok(ScopePermit {
            scope: self.clone(),
            session: false,
        })
    }
    pub(crate) fn record_usage(&self, usage: &Usage) {
        self.0.state.lock().usage.accumulate(usage);
    }
    pub(super) fn publish(&self, mut event: SubAgentEvent) {
        let mut state = self.0.state.lock();
        state.sequence += 1;
        event.sequence = state.sequence;
        if let Some(summary) = &mut event.summary {
            truncate_utf8_to_max_bytes(summary, 2000);
        }
        state.events.push_back(event);
        while state.events.len() > self.limits().max_events.clamp(1, 4096) {
            state.events.pop_front();
        }
        self.0.changed.send_replace(state.sequence);
    }
    /// Read bounded events after a cursor. Empty targets selects the whole root tree.
    pub fn events(&self, after: u64, targets: &[String]) -> SubAgentEvents {
        let state = self.0.state.lock();
        SubAgentEvents {
            cursor: state.sequence,
            lagged: after > state.sequence
                || state
                    .events
                    .front()
                    .is_some_and(|e| after.saturating_add(1) < e.sequence),
            timed_out: false,
            events: state
                .events
                .iter()
                .filter(|e| {
                    e.sequence > after && (targets.is_empty() || targets.contains(&e.execution.id))
                })
                .cloned()
                .collect(),
        }
    }
    /// Wait without polling. Subscribes before reading to avoid lost wakeups.
    /// Waits are capped at 60 seconds; cancellation returns an error. `All` requires targets.
    pub async fn wait(
        &self,
        after: u64,
        targets: &[String],
        mode: WaitMode,
        timeout: Duration,
        cancellation: anda_core::CancellationToken,
    ) -> Result<SubAgentEvents, BoxError> {
        if mode == WaitMode::All && targets.is_empty() {
            return Err("wait-all requires execution IDs".into());
        }
        let mut changes = self.0.changed.subscribe();
        let deadline = tokio::time::Instant::now() + timeout.min(Duration::from_secs(60));
        loop {
            let result = self.events(after, targets);
            let ready = match mode {
                WaitMode::Any => !result.events.is_empty(),
                WaitMode::All => targets.iter().all(|id| {
                    result
                        .events
                        .iter()
                        .any(|e| &e.execution.id == id && e.kind != SubAgentEventKind::Started)
                }),
            };
            if ready || result.lagged {
                return Ok(result);
            }
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Err("subagent wait cancelled".into()),
                _ = tokio::time::sleep_until(deadline) => return Ok(SubAgentEvents { timed_out: true, ..self.events(after, targets) }),
                _ = changes.changed() => {}
            }
        }
    }
    pub(super) fn remember(&self, caller: Principal, agent: String, session: String, detail: Json) {
        let mut state = self.0.state.lock();
        state
            .terminals
            .insert((caller, agent, session), (unix_ms(), detail));
        while state.terminals.len() > self.limits().max_events.clamp(1, 4096) {
            let oldest = state
                .terminals
                .iter()
                .min_by_key(|(_, (at, _))| *at)
                .map(|(key, _)| key.clone())
                .unwrap();
            state.terminals.remove(&oldest);
        }
    }
    pub(super) fn terminal(&self, caller: Principal, agent: &str, session: &str) -> Option<Json> {
        let mut state = self.0.state.lock();
        state.terminals.retain(|_, (at, _)| {
            unix_ms().saturating_sub(*at)
                <= self.limits().terminal_retention_secs.saturating_mul(1000)
        });
        state
            .terminals
            .get(&(caller, agent.into(), session.into()))
            .map(|(_, detail)| detail.clone())
    }
}

pub(crate) struct ScopePermit {
    scope: SubAgentScope,
    session: bool,
}
impl Drop for ScopePermit {
    fn drop(&mut self) {
        let mut state = self.scope.0.state.lock();
        if self.session {
            state.sessions -= 1;
        } else {
            state.requests_in_flight -= 1;
        }
    }
}
