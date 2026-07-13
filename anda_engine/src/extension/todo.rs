//! Todo extension primitives.
//!
//! This module provides:
//! - an in-memory todo store shared through [`BaseCtx`] state,
//! - tool input/output types for reading and updating the list,
//! - and the public tool entrypoint ([`TodoTool`]).
//!
//! The todo list is session-scoped rather than durable. Repeated calls in the
//! same context tree, including subagents spawned from that session, see the
//! same ordered task list.

use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolOutput};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};

use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

const TODO_OP_READ: &str = "read";
const TODO_OP_SET: &str = "set";
const TODO_OP_UPDATE: &str = "update";
const TODO_STATUS_PENDING: &str = "pending";
const TODO_STATUS_IN_PROGRESS: &str = "in_progress";
const TODO_STATUS_COMPLETED: &str = "completed";
const TODO_STATUS_CANCELLED: &str = "cancelled";
const TODO_EMPTY_CONTENT: &str = "(no description)";
const TODO_ACTIVE_LIST_PREFIX: &str =
    "[Your active task list was preserved across context compression]";
const TODO_MARKER_PENDING: &str = "[ ]";
const TODO_MARKER_IN_PROGRESS: &str = "[>]";
const TODO_MARKER_COMPLETED: &str = "[x]";
const TODO_MARKER_CANCELLED: &str = "[~]";

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

    /// Replaces the session list and returns the full normalized snapshot.
    pub fn set(&self, items: Vec<TodoItemInput>) -> Vec<TodoItem> {
        self.inner.write().set(items)
    }

    /// Applies partial updates and returns the full normalized snapshot.
    pub fn update(&self, items: Vec<TodoItemInput>) -> Vec<TodoItem> {
        self.inner.write().update(items)
    }

    /// Returns the current ordered task list.
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.inner.read().snapshot()
    }

    /// Returns true when at least one task is stored.
    pub fn has_items(&self) -> bool {
        self.inner.read().has_items()
    }

    /// Formats active tasks for reinjection after context compression.
    pub fn format_for_injection(&self) -> Option<String> {
        self.inner.read().format_for_injection()
    }
}

/// Returns the current todo session for the context, creating one on demand.
pub fn todo_session(ctx: &BaseCtx) -> TodoSession {
    if let Some(session) = ctx.get_state::<TodoSession>() {
        return session;
    }

    let session = TodoSession::new();
    let _ = ctx.set_state(session.clone());
    session
}

/// In-memory ordered todo store.
#[derive(Debug, Clone, Default)]
pub struct TodoStore {
    items: Vec<TodoItem>,
}

impl TodoStore {
    /// Replaces the current list, then returns the full snapshot.
    pub fn set(&mut self, items: Vec<TodoItemInput>) -> Vec<TodoItem> {
        self.items = Self::dedupe_by_id(items)
            .into_iter()
            .filter(|item| !item.id.trim().is_empty())
            .map(TodoItem::from_input)
            .collect();
        self.snapshot()
    }

    /// Applies partial updates by id, then returns the full snapshot.
    pub fn update(&mut self, items: Vec<TodoItemInput>) -> Vec<TodoItem> {
        let items = Self::dedupe_by_id(items);
        let mut index_by_id: HashMap<String, usize> = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| (item.id.clone(), index))
            .collect();

        for item in items {
            let item_id = item.id.trim().to_string();
            if item_id.is_empty() {
                continue;
            }

            if let Some(index) = index_by_id.get(&item_id).copied() {
                let current = &mut self.items[index];
                if let Some(content) = item.content.as_deref().map(str::trim)
                    && !content.is_empty()
                {
                    current.content = content.to_string();
                }

                if let Some(status) = item.status.as_deref() {
                    current.status = normalize_status(Some(status));
                }
            } else {
                let validated = TodoItem::from_input(item);
                index_by_id.insert(validated.id.clone(), self.items.len());
                self.items.push(validated);
            }
        }

        self.snapshot()
    }

    /// Returns a copy of the current ordered todo list.
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.items.clone()
    }

    /// Returns true when the store contains at least one item.
    pub fn has_items(&self) -> bool {
        !self.items.is_empty()
    }

    /// Renders the active todo items for prompt reinjection after compression.
    pub fn format_for_injection(&self) -> Option<String> {
        if self.items.is_empty() {
            return None;
        }

        let active_items: Vec<&TodoItem> = self
            .items
            .iter()
            .filter(|item| {
                matches!(
                    item.status.as_str(),
                    TODO_STATUS_PENDING | TODO_STATUS_IN_PROGRESS
                )
            })
            .collect();

        if active_items.is_empty() {
            return None;
        }

        let mut lines = Vec::with_capacity(active_items.len() + 1);
        lines.push(TODO_ACTIVE_LIST_PREFIX.to_string());
        for item in active_items {
            lines.push(format!(
                "- {} {}. {} ({})",
                status_marker(&item.status),
                item.id,
                item.content,
                item.status
            ));
        }

        Some(lines.join("\n"))
    }

    fn dedupe_by_id(items: Vec<TodoItemInput>) -> Vec<TodoItemInput> {
        let mut last_index = HashMap::new();
        for (index, item) in items.iter().enumerate() {
            last_index.insert(todo_dedupe_key(item), index);
        }

        let mut indexes: Vec<usize> = last_index.into_values().collect();
        indexes.sort_unstable();

        indexes
            .into_iter()
            .map(|index| items[index].clone())
            .collect()
    }
}

/// Arguments for the todo tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoArgs {
    /// Operation to perform: read, set, or update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    /// Task items for set/update. Omit or null for read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<TodoItemInput>>,
}

/// Input item accepted by the todo tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoItemInput {
    /// Unique todo identifier.
    #[serde(default)]
    pub id: String,
    /// Human-readable todo description.
    #[serde(default)]
    pub content: Option<String>,
    /// Current todo status.
    #[serde(default)]
    pub status: Option<String>,
}

/// Normalized todo item returned by the tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoItem {
    /// Unique todo identifier.
    pub id: String,
    /// Human-readable todo description.
    pub content: String,
    /// Current todo status.
    pub status: String,
}

impl TodoItem {
    fn from_input(input: TodoItemInput) -> Self {
        let id = input.id.trim();
        let content = input.content.as_deref().unwrap_or_default().trim();

        Self {
            id: id.to_string(),
            content: if content.is_empty() {
                TODO_EMPTY_CONTENT.to_string()
            } else {
                content.to_string()
            },
            status: normalize_status(input.status.as_deref()),
        }
    }
}

/// Summary counts returned with the todo list.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoSummary {
    /// Total number of tasks.
    pub total: usize,
    /// Number of pending tasks.
    pub pending: usize,
    /// Number of tasks currently in progress.
    pub in_progress: usize,
    /// Number of completed tasks.
    pub completed: usize,
    /// Number of cancelled tasks.
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

/// Output returned by the todo tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoOutput {
    /// Summary counts for the current task list.
    pub summary: TodoSummary,
    /// Full task list. Present only for read operations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<TodoItem>,
}

/// Typed hook for todo tool calls.
pub type TodoToolHook = DynToolHook<TodoArgs, TodoOutput>;

/// Tool implementation that exposes the session todo list to the agent.
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
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "todo";

    /// Creates a todo tool with the default behavioral guidance.
    pub fn new() -> Self {
        Self {
            description: concat!(
                "Session task list for complex work. Use op=set once to create ",
                "or replace a plan; use op=update for progress with only changed ",
                "ids; use op=read only when you need the full list again. ",
                "Writes return counts only; reads return items. Keep one ",
                "in_progress item, and mark work completed as soon as it is done."
            )
            .to_string(),
        }
    }

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

impl Tool<BaseCtx> for TodoTool {
    type Args = TodoArgs;
    type Output = TodoOutput;

    fn name(&self) -> String {
        Self::NAME.to_string()
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
                "properties": {
                    "op": {
                        "type": ["string", "null"],
                        "enum": [TODO_OP_READ, TODO_OP_SET, TODO_OP_UPDATE, null],
                        "description": "read: return full list. set: replace list. update: patch only changed ids.",
                        "default": TODO_OP_READ
                    },
                    "items": {
                        "description": "Items for set/update. Use null for read. For update, include only changed ids; null content/status keeps existing values.",
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable short id"
                                },
                                "content": {
                                    "type": ["string", "null"],
                                    "description": "Task text; null keeps existing text on update"
                                },
                                "status": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        TODO_STATUS_PENDING,
                                        TODO_STATUS_IN_PROGRESS,
                                        TODO_STATUS_COMPLETED,
                                        TODO_STATUS_CANCELLED,
                                        null
                                    ],
                                    "description": "Task status; null keeps existing status on update"
                                }
                            },
                            "required": ["id", "content", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["op", "items"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let hook = ctx.get_state::<TodoToolHook>();
        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let session = todo_session(&ctx);
        let op = normalize_op(args.op.as_deref());
        let (items, include_items) = match op {
            TODO_OP_SET => (session.set(args.items.unwrap_or_default()), false),
            TODO_OP_UPDATE => (session.update(args.items.unwrap_or_default()), false),
            _ => (session.snapshot(), true),
        };

        let output = TodoOutput {
            summary: TodoSummary::from_items(&items),
            items: if include_items { items } else { Vec::new() },
        };

        if let Some(hook) = &hook {
            return hook.after_tool_call(&ctx, ToolOutput::new(output)).await;
        }

        Ok(ToolOutput::new(output))
    }
}

fn todo_dedupe_key(item: &TodoItemInput) -> String {
    let id = item.id.trim();
    if id.is_empty() {
        String::new()
    } else {
        id.to_string()
    }
}

fn normalize_op(op: Option<&str>) -> &'static str {
    let op = op
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| TODO_OP_READ.to_string());

    match op.as_str() {
        TODO_OP_SET => TODO_OP_SET,
        TODO_OP_UPDATE => TODO_OP_UPDATE,
        _ => TODO_OP_READ,
    }
}

fn normalize_status(status: Option<&str>) -> String {
    let status = status
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| TODO_STATUS_PENDING.to_string());

    if VALID_STATUSES.contains(&status.as_str()) {
        status
    } else {
        TODO_STATUS_PENDING.to_string()
    }
}

fn status_marker(status: &str) -> &'static str {
    match status {
        TODO_STATUS_PENDING => TODO_MARKER_PENDING,
        TODO_STATUS_IN_PROGRESS => TODO_MARKER_IN_PROGRESS,
        TODO_STATUS_COMPLETED => TODO_MARKER_COMPLETED,
        TODO_STATUS_CANCELLED => TODO_MARKER_CANCELLED,
        _ => TODO_MARKER_PENDING,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;

    fn input(id: &str, content: Option<&str>, status: Option<&str>) -> TodoItemInput {
        TodoItemInput {
            id: id.to_string(),
            content: content.map(ToString::to_string),
            status: status.map(ToString::to_string),
        }
    }

    fn mock_ctx() -> BaseCtx {
        EngineBuilder::new().mock_ctx().base
    }

    #[test]
    fn set_dedupes_normalizes_and_skips_empty_ids() {
        let mut store = TodoStore::default();

        let items = store.set(vec![
            input("1", Some("draft plan"), Some(TODO_STATUS_PENDING)),
            input("1", Some("final plan"), Some(TODO_STATUS_COMPLETED)),
            input("", Some(""), Some("invalid")),
        ]);

        assert_eq!(
            items,
            vec![TodoItem {
                id: "1".to_string(),
                content: "final plan".to_string(),
                status: TODO_STATUS_COMPLETED.to_string(),
            }]
        );
    }

    #[test]
    fn update_patches_existing_items_and_preserves_order() {
        let mut store = TodoStore::default();
        store.set(vec![
            input("1", Some("draft"), Some(TODO_STATUS_PENDING)),
            input("2", Some("implement"), Some(TODO_STATUS_PENDING)),
        ]);

        let items = store.update(vec![
            input(
                "2",
                Some("implement todo tool"),
                Some(TODO_STATUS_IN_PROGRESS),
            ),
            input("3", Some("write tests"), Some(TODO_STATUS_PENDING)),
            input("", Some("ignored"), Some(TODO_STATUS_COMPLETED)),
            input(
                "3",
                Some("write tests thoroughly"),
                Some(TODO_STATUS_COMPLETED),
            ),
        ]);

        assert_eq!(
            items,
            vec![
                TodoItem {
                    id: "1".to_string(),
                    content: "draft".to_string(),
                    status: TODO_STATUS_PENDING.to_string(),
                },
                TodoItem {
                    id: "2".to_string(),
                    content: "implement todo tool".to_string(),
                    status: TODO_STATUS_IN_PROGRESS.to_string(),
                },
                TodoItem {
                    id: "3".to_string(),
                    content: "write tests thoroughly".to_string(),
                    status: TODO_STATUS_COMPLETED.to_string(),
                },
            ]
        );
    }

    #[test]
    fn injection_format_only_includes_active_items() {
        let mut store = TodoStore::default();
        store.set(vec![
            input("1", Some("plan"), Some(TODO_STATUS_PENDING)),
            input("2", Some("build"), Some(TODO_STATUS_IN_PROGRESS)),
            input("3", Some("done"), Some(TODO_STATUS_COMPLETED)),
            input("4", Some("skip"), Some(TODO_STATUS_CANCELLED)),
        ]);

        let injected = store.format_for_injection().unwrap();
        assert!(injected.contains(TODO_ACTIVE_LIST_PREFIX));
        assert!(injected.contains("- [ ] 1. plan (pending)"));
        assert!(injected.contains("- [>] 2. build (in_progress)"));
        assert!(!injected.contains("done"));
        assert!(!injected.contains("skip"));
    }

    #[tokio::test]
    async fn tool_call_persists_session_state() {
        let ctx = mock_ctx();
        let tool = TodoTool::new();

        let first = tool
            .call(
                ctx.clone(),
                TodoArgs {
                    op: Some(TODO_OP_SET.to_string()),
                    items: Some(vec![input("1", Some("plan"), Some(TODO_STATUS_PENDING))]),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(first.output.summary.total, 1);
        assert!(first.output.items.is_empty());
        assert!(todo_session(&ctx).has_items());

        let second = tool
            .call(ctx.clone(), TodoArgs::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(
            second.output.items,
            vec![TodoItem {
                id: "1".to_string(),
                content: "plan".to_string(),
                status: TODO_STATUS_PENDING.to_string(),
            }]
        );

        let third = tool
            .call(
                ctx.clone(),
                TodoArgs {
                    op: Some(TODO_OP_UPDATE.to_string()),
                    items: Some(vec![TodoItemInput {
                        id: "1".to_string(),
                        content: Some("plan carefully".to_string()),
                        status: None,
                    }]),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(third.output.summary.pending, 1);
        assert!(third.output.items.is_empty());

        let fourth = tool
            .call(
                ctx.clone(),
                TodoArgs {
                    op: Some(TODO_OP_READ.to_string()),
                    items: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(fourth.output.items[0].content, "plan carefully");
    }

    #[tokio::test]
    async fn session_state_persists_across_per_call_child_contexts() {
        // Mirror how the completion runner drives tools: a long-lived parent
        // context seeds the session, and each tool invocation runs on a fresh
        // `child_base` context that snapshot-copies parent state. Without the
        // seeded session each child would create its own empty store and writes
        // would be lost between calls.
        let parent = EngineBuilder::new().mock_ctx();
        parent.base.set_state(TodoSession::new());
        let tool = TodoTool::new();

        let write_ctx = parent.child_base("todo").unwrap();
        let first = tool
            .call(
                write_ctx,
                TodoArgs {
                    op: Some(TODO_OP_SET.to_string()),
                    items: Some(vec![input("1", Some("plan"), Some(TODO_STATUS_PENDING))]),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(first.output.summary.total, 1);

        // A distinct child context (as produced by a later tool call) must
        // observe the write performed by the previous one.
        let read_ctx = parent.child_base("todo").unwrap();
        let second = tool
            .call(
                read_ctx,
                TodoArgs {
                    op: Some(TODO_OP_READ.to_string()),
                    items: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            second.output.items,
            vec![TodoItem {
                id: "1".to_string(),
                content: "plan".to_string(),
                status: TODO_STATUS_PENDING.to_string(),
            }]
        );
    }

    #[test]
    fn definition_schema_avoids_anyof() {
        let definition = TodoTool::new().definition();

        assert!(
            definition.parameters["properties"]["items"]
                .get("anyOf")
                .is_none()
        );
        assert_eq!(
            definition.parameters["properties"]["items"]["type"],
            json!(["array", "null"])
        );
        assert_eq!(
            definition.parameters["properties"]["op"]["enum"],
            json!([TODO_OP_READ, TODO_OP_SET, TODO_OP_UPDATE, null])
        );
        assert_eq!(
            definition.parameters["properties"]["items"]["items"]["properties"]["content"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            definition.parameters["properties"]["items"]["items"]["properties"]["status"]["enum"],
            json!([
                TODO_STATUS_PENDING,
                TODO_STATUS_IN_PROGRESS,
                TODO_STATUS_COMPLETED,
                TODO_STATUS_CANCELLED,
                null
            ])
        );
        assert_eq!(definition.parameters["required"], json!(["op", "items"]));
    }
}
