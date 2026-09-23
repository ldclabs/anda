//! MCP tool routing: local name mapping, call rounds, and result adaptation.
//!
//! Maps remote MCP tool names onto collision-free Anda-facing names, drives one
//! `tools/call` through the `2026-07-28` intermediate answers (MRTR
//! `input_required` rounds, task handles) until a final result, and adapts
//! [`CallToolResult`] into the audited [`ToolOutput`] envelope.

use anda_core::{BoxError, CancellationToken, FunctionDefinition, Json, ToolOutput, Usage};
use rmcp::{
    Peer, RoleClient,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, CancelTaskParams, ContentBlock,
        CreateTaskResult, DEFAULT_MRTR_MAX_ROUNDS, GetTaskParams, InputRequiredResult, TaskPayload,
    },
};
use serde_json::json;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    time::Duration,
};

use super::McpTasksConfig;
use tokio::time::Instant;

pub(crate) const TOOL_REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
const TASK_CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

/// How many times to re-derive a local tool name before giving up on a collision.
pub(crate) const MAX_LOCAL_NAME_ATTEMPTS: usize = 8;

/// Pause between MRTR rounds that carry only `requestState`, i.e. the server
/// asking to be polled rather than asking for input.
pub(crate) const MRTR_STATE_ROUND_DELAY: Duration = Duration::from_millis(200);

/// Poll interval used when a task suggests none, plus the bounds applied to a
/// server-suggested one. A remote server controls `pollIntervalMs`, so it is
/// clamped instead of trusted.
pub(crate) const TASK_POLL_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const TASK_POLL_INTERVAL_MIN: Duration = Duration::from_millis(250);
pub(crate) const TASK_POLL_INTERVAL_MAX: Duration = Duration::from_secs(10);

/// Default ceiling on how long one tool call waits for a task to finish.
pub(crate) const DEFAULT_TASK_MAX_WAIT_SECS: u64 = 300;

/// Hard ceiling on a configured `max_wait_secs`.
///
/// A tool call blocks for the whole wait, so a day is already far past anything
/// sane; the bound also keeps the poll deadline from overflowing `Instant`.
pub(crate) const MAX_TASK_MAX_WAIT_SECS: u64 = 24 * 60 * 60;

/// Drives one `tools/call` until the server produces a result.
///
/// Before `2026-07-28` that took a single round trip. The revision adds two
/// intermediate answers: an MRTR `input_required` result (SEP-2322) and a task
/// handle (SEP-2663). Both are resolved here so the caller still sees one
/// [`CallToolResult`].
pub(crate) async fn call_tool_rounds(
    route: &McpToolRoute,
    peer: &Peer<RoleClient>,
    mut params: CallToolRequestParams,
    tasks: Option<&McpTasksConfig>,
    cancellation: &CancellationToken,
) -> Result<CallToolResult, BoxError> {
    for _ in 0..DEFAULT_MRTR_MAX_ROUNDS {
        let response = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err("MCP tool call cancelled".into()),
            response = tokio::time::timeout(TOOL_REQUEST_TIMEOUT, peer.call_tool_once(params.clone())) => {
                response.map_err(|_| format!("MCP tool {} request timed out", route.name))??
            }
        };
        match response {
            CallToolResponse::Complete(result) => return Ok(result),
            CallToolResponse::InputRequired(result) => {
                // A round carrying actual `inputRequests` wants Sampling,
                // Elicitation, or Roots, none of which this host advertises. Report
                // it as a tool-level error the model can act on instead of failing
                // the turn. A round with only `requestState` is the server asking to
                // be polled: echo the state back and continue.
                if result
                    .input_requests
                    .as_ref()
                    .is_some_and(|requests| !requests.is_empty())
                {
                    return Ok(input_required_error(route, &result));
                }
                let Some(request_state) = result.request_state else {
                    return Err(format!(
                        "MCP tool {} returned an input_required result with neither \
                         input requests nor request state",
                        route.name
                    )
                    .into());
                };
                params.request_state = Some(request_state);
                params.input_responses = None;
                tokio::time::sleep(MRTR_STATE_ROUND_DELAY).await;
            }
            CallToolResponse::Task(task) => {
                return await_task(route, peer, task, tasks, cancellation).await;
            }
            other => {
                return Err(format!(
                    "MCP tool {} returned an unsupported response: {other:?}",
                    route.name
                )
                .into());
            }
        }
    }

    Err(format!(
        "MCP tool {} did not complete within {DEFAULT_MRTR_MAX_ROUNDS} input_required rounds",
        route.name
    )
    .into())
}

/// Polls a SEP-2663 task to a terminal state and returns its tool result.
///
/// The task is cancelled best-effort whenever this host walks away from it, so
/// an abandoned task does not keep running on the server.
async fn await_task(
    route: &McpToolRoute,
    peer: &Peer<RoleClient>,
    created: CreateTaskResult,
    tasks: Option<&McpTasksConfig>,
    cancellation: &CancellationToken,
) -> Result<CallToolResult, BoxError> {
    let task_id = created.task.task_id.clone();
    let mut cleanup = TaskCleanup {
        peer: peer.clone(),
        task_id: task_id.clone(),
        armed: true,
    };
    let Some(tasks) = tasks else {
        cleanup.armed = false;
        cancel_task(peer, &task_id).await;
        return Err(format!(
            "MCP tool {} returned a task handle, but the tasks extension is not enabled \
             for server {}",
            route.name, route.server_id
        )
        .into());
    };

    let max_wait = tasks.max_wait();
    let deadline = Instant::now() + max_wait;
    let mut interval = task_poll_interval(created.task.poll_interval_ms);
    loop {
        let poll = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                cleanup.armed = false;
                cancel_task(peer, &task_id).await;
                return Err("MCP task cancelled".into());
            }
            poll = tokio::time::timeout_at(deadline, async {
                tokio::time::sleep(interval).await;
                peer.get_task(GetTaskParams::new(task_id.clone())).await
            }) => poll,
        };
        let task = match poll {
            Ok(result) => result?.task,
            Err(_) => {
                cleanup.armed = false;
                cancel_task(peer, &task_id).await;
                return Err(format!(
                    "MCP tool {} task {task_id} did not finish within {}s",
                    route.name,
                    max_wait.as_secs()
                )
                .into());
            }
        };
        interval = task_poll_interval(task.task.poll_interval_ms);
        match task.payload {
            TaskPayload::Working => continue,
            TaskPayload::Completed { result } => {
                cleanup.armed = false;
                // The payload mirrors the result of the original request, so it
                // deserializes as the `tools/call` result it stands in for.
                return serde_json::from_value(Json::Object(result)).map_err(|err| {
                    format!(
                        "MCP tool {} returned an unreadable task result: {err}",
                        route.name
                    )
                    .into()
                });
            }
            TaskPayload::Failed { error } => {
                cleanup.armed = false;
                return Err(format!(
                    "MCP tool {} task {task_id} failed: {}",
                    route.name,
                    Json::Object(error)
                )
                .into());
            }
            TaskPayload::Cancelled => {
                cleanup.armed = false;
                return Err(format!("MCP tool {} task {task_id} was cancelled", route.name).into());
            }
            TaskPayload::InputRequired { input_requests } => {
                cleanup.armed = false;
                // Same reasoning as the MRTR round above: nothing here can answer a
                // sampling, elicitation, or roots request.
                cancel_task(peer, &task_id).await;
                return Ok(unsupported_input_error(
                    route,
                    input_requests.keys().map(String::as_str),
                ));
            }
            _ => {
                return Err(format!(
                    "MCP tool {} task {task_id} reported an unsupported status",
                    route.name
                )
                .into());
            }
        }
    }
}

// Also cancel a remote task when a parent runner drops the polling future.
struct TaskCleanup {
    peer: Peer<RoleClient>,
    task_id: String,
    armed: bool,
}

impl Drop for TaskCleanup {
    fn drop(&mut self) {
        if self.armed
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let peer = self.peer.clone();
            let task_id = self.task_id.clone();
            runtime.spawn(async move {
                cancel_task(&peer, &task_id).await;
            });
        }
    }
}

/// One Anda-facing route to an MCP tool.
#[derive(Debug, Clone)]
pub struct McpToolRoute {
    /// Anda-facing tool name.
    pub name: String,
    /// Configured MCP server id.
    pub server_id: String,
    /// Original MCP tool name.
    pub remote_name: String,
    /// Model-facing function definition.
    pub definition: FunctionDefinition,
}

/// Clamps a server-suggested `tasks/get` poll interval into a sane range.
pub(crate) fn task_poll_interval(poll_interval_ms: Option<u64>) -> Duration {
    poll_interval_ms
        .map(Duration::from_millis)
        .unwrap_or(TASK_POLL_INTERVAL)
        .clamp(TASK_POLL_INTERVAL_MIN, TASK_POLL_INTERVAL_MAX)
}

/// Abandons a task this host will not wait for, so the server can release it.
async fn cancel_task(peer: &Peer<RoleClient>, task_id: &str) {
    match tokio::time::timeout(
        TASK_CANCEL_TIMEOUT,
        peer.cancel_task(CancelTaskParams::new(task_id)),
    )
    .await
    {
        Ok(Ok(_)) => {}
        Ok(Err(err)) => log::debug!("MCP task {task_id} could not be cancelled: {err}"),
        Err(_) => log::debug!("MCP task {task_id} cancellation acknowledgement timed out"),
    }
}

pub(crate) fn input_required_error(
    route: &McpToolRoute,
    result: &InputRequiredResult,
) -> CallToolResult {
    let keys = result
        .input_requests
        .iter()
        .flat_map(|requests| requests.keys().map(String::as_str));
    unsupported_input_error(route, keys)
}

/// Tool-level error for a server round this host cannot answer.
///
/// Anda advertises neither Sampling, Elicitation, nor Roots, so an MRTR round
/// asking for them is a dead end. Returning it as a failed tool result — rather
/// than an error that aborts the turn — lets the model choose another path.
pub(crate) fn unsupported_input_error<'a>(
    route: &McpToolRoute,
    request_keys: impl Iterator<Item = &'a str>,
) -> CallToolResult {
    let keys: Vec<&str> = request_keys.collect();
    let requested = if keys.is_empty() {
        String::new()
    } else {
        format!(" (requests: {})", keys.join(", "))
    };
    CallToolResult::error(vec![ContentBlock::text(format!(
        "MCP tool {} on server {} requires client-side input{requested}, which this host \
         does not provide: sampling, elicitation, and roots are not supported. Call the tool \
         with complete arguments, or use a different tool.",
        route.remote_name, route.server_id
    ))])
}

pub(crate) fn mcp_result_to_tool_output(
    route: &McpToolRoute,
    result: CallToolResult,
) -> ToolOutput<Json> {
    let mut output = ToolOutput::new(json!({
        "server_id": route.server_id,
        "tool": route.remote_name,
        "structured_content": result.structured_content,
        "content": result.content,
        "_meta": result.meta,
    }));
    output.is_error = result.is_error;
    output.usage = Usage {
        requests: 1,
        ..Usage::default()
    };
    output
}

pub(crate) fn sanitize_name_part(input: &str) -> String {
    let mut out = String::new();
    let mut previous_underscore = false;
    for c in input.chars() {
        let c = c.to_ascii_lowercase();
        let valid = matches!(c, 'a'..='z' | '0'..='9');
        if valid {
            out.push(c);
            previous_underscore = false;
        } else if !previous_underscore {
            out.push('_');
            previous_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    let mut normalized = if trimmed.is_empty() {
        "x".to_string()
    } else {
        trimmed
    };
    if !normalized
        .chars()
        .next()
        .is_some_and(|c: char| c.is_ascii_lowercase())
    {
        normalized.insert(0, 'x');
    }
    normalized
}

pub(crate) fn shorten_with_hash(base: &str, key: &str) -> String {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let suffix = format!("{:08x}", hasher.finish() as u32);
    let max_prefix = 64usize.saturating_sub(suffix.len() + 1);
    let mut prefix = base.chars().take(max_prefix).collect::<String>();
    prefix = prefix.trim_end_matches('_').to_string();
    format!("{}_{}", prefix, suffix)
}
