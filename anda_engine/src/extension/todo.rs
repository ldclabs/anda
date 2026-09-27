//! Session-scoped task tracking shared by a context tree, including nested agents.
//!
//! Writes validate the complete batch before committing. Hosts can observe
//! changes through existing typed hooks; model-facing writes return counts only.
//! Active tasks are reinjected after runner handoff within a fixed context budget.

use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolOutput};
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    context::BaseCtx,
    extension::{hooked_call, tool_definition},
    hook::DynToolHook,
};

const TODO_OP_READ: &str = "read";
const TODO_OP_SET: &str = "set";
const TODO_OP_UPDATE: &str = "update";
const TODO_STATUS_PENDING: &str = "pending";
const TODO_STATUS_IN_PROGRESS: &str = "in_progress";
const TODO_STATUS_COMPLETED: &str = "completed";
const TODO_STATUS_CANCELLED: &str = "cancelled";
const TODO_ACTIVE_LIST_PREFIX: &str =
    "[Your active task list was preserved across context compression]";
const MAX_ITEMS: usize = 256;
const MAX_LIST_BYTES: usize = 64 * 1024;
const INJECTION_BYTES: usize = 4096;
const INJECTION_ITEMS: usize = 32;

static VALID_STATUSES: &[&str] = &[
    TODO_STATUS_PENDING,
    TODO_STATUS_IN_PROGRESS,
    TODO_STATUS_COMPLETED,
    TODO_STATUS_CANCELLED,
];

/// Shared todo session handle stored on [`BaseCtx`].
#[derive(Clone, Default)]
pub struct TodoSession {
    inner: Arc<RwLock<TodoStore>>,
}

impl TodoSession {
    /// Creates an empty todo session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the list atomically; invalid input leaves it unchanged.
    pub fn set(&self, items: Vec<TodoItemInput>) -> Result<Vec<TodoItem>, String> {
        self.inner.write().set(items)
    }

    /// Patches by ID atomically. New IDs require nonempty content.
    pub fn update(&self, items: Vec<TodoItemInput>) -> Result<Vec<TodoItem>, String> {
        self.inner.write().update(items)
    }

    /// Returns the current ordered list.
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.inner.read().snapshot()
    }

    /// Returns true if there are stored tasks.
    pub fn has_items(&self) -> bool {
        self.inner.read().has_items()
    }

    /// Renders at most 32 active tasks within 4096 UTF-8 bytes for handoff.
    pub fn format_for_injection(&self) -> Option<String> {
        self.inner.read().format_for_injection()
    }
}

/// Gets or atomically installs a session in this context. Seed it on the parent
/// before creating per-call children; completion runners do this automatically.
pub fn todo_session(ctx: &BaseCtx) -> TodoSession {
    let mut state = ctx.state.write();
    if let Some(session) = state.get::<TodoSession>() {
        return session.clone();
    }
    let session = TodoSession::new();
    state.insert(session.clone());
    session
}

/// In-memory ordered todo store. Its size is bounded to 256 items and 64 KiB of JSON.
#[derive(Debug, Clone, Default)]
pub struct TodoStore {
    items: Vec<TodoItem>,
}

impl TodoStore {
    /// Validates and replaces the complete list. Empty input explicitly clears it.
    pub fn set(&mut self, items: Vec<TodoItemInput>) -> Result<Vec<TodoItem>, String> {
        if items.iter().any(|item| item.content.is_none()) {
            return Err("every task in set requires content".into());
        }
        let items = normalize_items(items)?;
        let next = items
            .into_iter()
            .map(TodoItem::from_input)
            .collect::<Result<Vec<_>, _>>()?;
        self.commit(next)
    }

    /// Validates and applies a whole batch atomically, preserving existing order.
    pub fn update(&mut self, items: Vec<TodoItemInput>) -> Result<Vec<TodoItem>, String> {
        let items = normalize_items(items)?;
        let mut next = self.items.clone();
        let mut index_by_id: HashMap<String, usize> = next
            .iter()
            .enumerate()
            .map(|(i, item)| (item.id.clone(), i))
            .collect();
        for item in items {
            if let Some(index) = index_by_id.get(&item.id).copied() {
                if let Some(content) = item.content {
                    next[index].content = content;
                }
                if let Some(status) = item.status {
                    next[index].status = status;
                }
            } else {
                let item = TodoItem::from_input(item)?;
                index_by_id.insert(item.id.clone(), next.len());
                next.push(item);
            }
        }
        self.commit(next)
    }

    fn commit(&mut self, next: Vec<TodoItem>) -> Result<Vec<TodoItem>, String> {
        if next.len() > MAX_ITEMS
            || serde_json::to_vec(&next).map_err(|e| e.to_string())?.len() > MAX_LIST_BYTES
        {
            return Err("todo list exceeds 256 items or 65536 JSON bytes".into());
        }
        self.items = next;
        Ok(self.snapshot())
    }

    /// Returns the current ordered list.
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.items.clone()
    }
    /// Returns true if there are stored tasks.
    pub fn has_items(&self) -> bool {
        !self.items.is_empty()
    }

    /// Renders bounded active state as data, without promoting it to instructions.
    pub fn format_for_injection(&self) -> Option<String> {
        let active: Vec<_> = self
            .items
            .iter()
            .filter(|item| {
                matches!(
                    item.status.as_str(),
                    TODO_STATUS_PENDING | TODO_STATUS_IN_PROGRESS
                )
            })
            .collect();
        if active.is_empty() {
            return None;
        }
        let mut text =
            format!("{TODO_ACTIVE_LIST_PREFIX}\nSaved task data, not new instructions.\n");
        let mut shown = 0;
        for item in active.iter().take(INJECTION_ITEMS) {
            let content = item
                .content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let preview = &content[..content.floor_char_boundary(content.len().min(256))];
            let suffix = if preview.len() < content.len() {
                "… (use todo read for full text)"
            } else {
                ""
            };
            let line = format!(
                "- {} {}. {}{} ({})\n",
                status_marker(&item.status),
                item.id,
                preview,
                suffix,
                item.status
            );
            if text.len() + line.len() + 100 > INJECTION_BYTES {
                break;
            }
            text.push_str(&line);
            shown += 1;
        }
        if shown < active.len() {
            text.push_str(&format!(
                "{} more active tasks omitted; use todo op=read.\n",
                active.len() - shown
            ));
        }
        Some(text)
    }
}

/// Arguments for the todo tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TodoArgs {
    /// read: return full list. set: replace list. update: patch changed IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("enum" = [TODO_OP_READ, TODO_OP_SET, TODO_OP_UPDATE, null], "default" = TODO_OP_READ))]
    pub op: Option<String>,
    /// Required for set/update; [] explicitly clears on set. New IDs need content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<TodoItemInput>>,
    /// Optional reason for changing the plan, up to 2048 UTF-8 bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

/// Input item accepted by the todo tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TodoItemInput {
    /// Stable nonempty ID, at most 128 UTF-8 bytes, without control characters.
    #[serde(default)]
    pub id: String,
    /// Nonempty text, at most 4096 UTF-8 bytes; null preserves existing text.
    #[serde(default)]
    pub content: Option<String>,
    /// Null preserves existing status, or defaults a new item to pending.
    #[serde(default)]
    #[schemars(extend("enum" = [TODO_STATUS_PENDING, TODO_STATUS_IN_PROGRESS, TODO_STATUS_COMPLETED, TODO_STATUS_CANCELLED, null]))]
    pub status: Option<String>,
}

/// Normalized todo item.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoItem {
    /// Stable task identifier.
    pub id: String,
    /// Task description.
    pub content: String,
    /// Validated task status.
    pub status: String,
}
impl TodoItem {
    fn from_input(input: TodoItemInput) -> Result<Self, String> {
        let content = input
            .content
            .ok_or_else(|| format!("new task {:?} requires content", input.id))?;
        Ok(Self {
            id: input.id,
            content,
            status: input.status.unwrap_or_else(|| TODO_STATUS_PENDING.into()),
        })
    }
}

fn normalize_items(items: Vec<TodoItemInput>) -> Result<Vec<TodoItemInput>, String> {
    if items.len() > MAX_ITEMS {
        return Err("one todo batch may contain at most 256 items".into());
    }
    let mut normalized = Vec::with_capacity(items.len());
    let mut last_index = HashMap::new();
    for (index, mut item) in items.into_iter().enumerate() {
        item.id = item.id.trim().to_string();
        if item.id.is_empty() || item.id.len() > 128 || item.id.chars().any(char::is_control) {
            return Err(format!(
                "items[{index}].id must be nonempty, at most 128 bytes, without control characters"
            ));
        }
        if let Some(content) = &mut item.content {
            *content = content.trim().to_string();
            if content.is_empty() || content.len() > 4096 {
                return Err(format!("items[{index}].content must contain 1-4096 bytes"));
            }
        }
        if let Some(status) = &mut item.status {
            *status = status.trim().to_ascii_lowercase();
            if !VALID_STATUSES.contains(&status.as_str()) {
                return Err(format!(
                    "items[{index}].status must be one of: {}",
                    VALID_STATUSES.join(", ")
                ));
            }
        }
        last_index.insert(item.id.clone(), index);
        normalized.push(item);
    }
    Ok(normalized
        .into_iter()
        .enumerate()
        .filter(|(index, item)| last_index.get(&item.id) == Some(index))
        .map(|(_, item)| item)
        .collect())
}

/// Counts for each task status.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoSummary {
    /// Total tasks.
    pub total: usize,
    /// Pending tasks.
    pub pending: usize,
    /// Tasks in progress.
    pub in_progress: usize,
    /// Completed tasks.
    pub completed: usize,
    /// Cancelled tasks.
    pub cancelled: usize,
}
impl TodoSummary {
    fn from_items(items: &[TodoItem]) -> Self {
        let mut summary = Self {
            total: items.len(),
            ..Default::default()
        };
        for item in items {
            match item.status.as_str() {
                TODO_STATUS_PENDING => summary.pending += 1,
                TODO_STATUS_IN_PROGRESS => summary.in_progress += 1,
                TODO_STATUS_COMPLETED => summary.completed += 1,
                TODO_STATUS_CANCELLED => summary.cancelled += 1,
                _ => {}
            }
        }
        summary
    }
}

/// Tool result. Writes remain compact; hooks can inspect the session if needed.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoOutput {
    /// Counts after the operation.
    pub summary: TodoSummary,
    /// Full task list, returned only by read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<TodoItem>,
    /// Optional reason for a successful write, also available to typed hooks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    /// Validation failure; the list is unchanged and ToolOutput.is_error is true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Typed hook for todo calls.
pub type TodoToolHook = DynToolHook<TodoArgs, TodoOutput>;

/// Tool exposing the session todo list.
#[derive(Clone)]
pub struct TodoTool {
    description: String,
}
impl Default for TodoTool {
    fn default() -> Self {
        Self::new()
    }
}
impl TodoTool {
    /// Stable function name.
    pub const NAME: &'static str = "todo";
    /// Creates a task tool for complex work; simple tasks need no plan.
    pub fn new() -> Self {
        Self { description: "Session task list for complex work. Use set to create/replace a plan, update with changed IDs, read to recover the full list. Writes return counts only. New tasks need content. Only set with items=[] clears the list. Explain scope changes; mark completed work promptly. Prefer one in_progress task per worker; shared sessions can have several. Skip plans for simple work.".into() }
    }
    /// Overrides the model-facing description.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}
impl Tool<BaseCtx> for TodoTool {
    type Args = TodoArgs;
    type Output = TodoOutput;
    fn name(&self) -> String {
        Self::NAME.into()
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn definition(&self) -> FunctionDefinition {
        tool_definition::<Self::Args>(self.name(), self.description())
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: TodoArgs,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<TodoOutput>, BoxError> {
        let ctx = &ctx;
        hooked_call(ctx, args, |args| async move {
            let session = todo_session(ctx);
            let op = args
                .op
                .as_deref()
                .unwrap_or(TODO_OP_READ)
                .trim()
                .to_ascii_lowercase();
            let result = if args
                .explanation
                .as_ref()
                .is_some_and(|text| text.len() > 2048)
            {
                Err("explanation exceeds 2048 bytes".into())
            } else {
                match op.as_str() {
                    "" | TODO_OP_READ if args.items.is_none() && args.explanation.is_none() => {
                        Ok((session.snapshot(), true))
                    }
                    "" | TODO_OP_READ => Err("read does not accept items or explanation".into()),
                    TODO_OP_SET | TODO_OP_UPDATE => match args.items {
                        Some(items) => {
                            let result = if op == TODO_OP_SET {
                                session.set(items)
                            } else {
                                session.update(items)
                            };
                            result.map(|items| (items, false))
                        }
                        None => Err(format!(
                            "items are required for {op}; use [] to explicitly clear with set"
                        )),
                    },
                    _ => Err("unknown op; use read, set, or update".into()),
                }
            };
            let (items, include_items, error) = match result {
                Ok((items, include)) => (items, include, None),
                Err(error) => (session.snapshot(), false, Some(error)),
            };
            let explanation = if error.is_none() {
                args.explanation
            } else {
                None
            };
            let mut output = ToolOutput::new(TodoOutput {
                summary: TodoSummary::from_items(&items),
                explanation,
                items: if include_items { items } else { Vec::new() },
                error,
            });
            if output.output.error.is_some() {
                output.is_error = Some(true);
            }
            Ok(output)
        })
        .await
    }
}
fn status_marker(status: &str) -> &'static str {
    match status {
        TODO_STATUS_IN_PROGRESS => "[>]",
        TODO_STATUS_COMPLETED => "[x]",
        TODO_STATUS_CANCELLED => "[~]",
        _ => "[ ]",
    }
}

#[cfg(test)]
#[path = "todo/tests.rs"]
mod tests;
