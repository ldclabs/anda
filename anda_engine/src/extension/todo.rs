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
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

const TODO_STATUS_PENDING: &str = "pending";
const TODO_STATUS_IN_PROGRESS: &str = "in_progress";
const TODO_STATUS_COMPLETED: &str = "completed";
const TODO_STATUS_CANCELLED: &str = "cancelled";
const TODO_EMPTY_ID: &str = "?";
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write(&self, todos: Vec<TodoItemInput>, merge: bool) -> Vec<TodoItem> {
        self.inner.write().write(todos, merge)
    }

    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.inner.read().snapshot()
    }

    pub fn has_items(&self) -> bool {
        self.inner.read().has_items()
    }

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
    /// Writes todos using replace or merge semantics, then returns the full list.
    pub fn write(&mut self, todos: Vec<TodoItemInput>, merge: bool) -> Vec<TodoItem> {
        let todos = Self::dedupe_by_id(todos);

        if !merge {
            self.items = todos.into_iter().map(TodoItem::from_input).collect();
            return self.snapshot();
        }

        let mut existing: HashMap<String, TodoItem> = self
            .items
            .iter()
            .cloned()
            .map(|item| (item.id.clone(), item))
            .collect();

        for todo in todos {
            let item_id = todo.id.trim().to_string();
            if item_id.is_empty() {
                continue;
            }

            if let Some(item) = existing.get_mut(&item_id) {
                if let Some(content) = todo.content.as_deref().map(str::trim)
                    && !content.is_empty()
                {
                    item.content = content.to_string();
                }

                if let Some(status) = todo.status.as_deref() {
                    item.status = normalize_status(Some(status));
                }
            } else {
                let validated = TodoItem::from_input(todo);
                existing.insert(validated.id.clone(), validated.clone());
                self.items.push(validated);
            }
        }

        let mut seen = HashSet::new();
        self.items = self
            .items
            .iter()
            .filter_map(|item| {
                let current = existing
                    .get(&item.id)
                    .cloned()
                    .unwrap_or_else(|| item.clone());
                if seen.insert(current.id.clone()) {
                    Some(current)
                } else {
                    None
                }
            })
            .collect();

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

    fn dedupe_by_id(todos: Vec<TodoItemInput>) -> Vec<TodoItemInput> {
        let mut last_index = HashMap::new();
        for (index, item) in todos.iter().enumerate() {
            last_index.insert(todo_dedupe_key(item), index);
        }

        let mut indexes: Vec<usize> = last_index.into_values().collect();
        indexes.sort_unstable();

        indexes
            .into_iter()
            .map(|index| todos[index].clone())
            .collect()
    }
}

/// Arguments for the todo tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TodoArgs {
    /// Task items to write. Omit to read the current list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub todos: Option<Vec<TodoItemInput>>,
    /// Whether writes should merge into the existing list by id.
    #[serde(default)]
    pub merge: bool,
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
            id: if id.is_empty() {
                TODO_EMPTY_ID.to_string()
            } else {
                id.to_string()
            },
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
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
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
    pub todos: Vec<TodoItem>,
    pub summary: TodoSummary,
}

pub type TodoToolHook = DynToolHook<TodoArgs, ToolOutput<TodoOutput>>;

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
                "Manage your task list for the current session. Use for complex tasks ",
                "with 3+ steps or when the user provides multiple tasks. ",
                "This list is shared across the current agent and its subagents ",
                "within the same session/context tree. ",
                "Call with no parameters to read the current list.\n\n",
                "Writing:\n",
                "- Provide 'todos' array to create/update items\n",
                "- merge=false (default): replace the entire list with a fresh plan\n",
                "- merge=true: update existing items by id, add any new ones\n\n",
                "Each item: {id: string, content: string, ",
                "status: pending|in_progress|completed|cancelled}\n",
                "List order is priority. Only ONE item in_progress at a time.\n",
                "Mark items completed immediately when done. If something fails, ",
                "cancel it and add a revised item.\n\n",
                "Always returns the full current list."
            )
            .to_string(),
        }
    }

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
                    "todos": {
                        "type": "array",
                        "description": "Task items to write. Omit to read current list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique item identifier"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Task description"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": VALID_STATUSES,
                                    "description": "Current status"
                                }
                            },
                            "required": ["id", "content", "status"],
                            "additionalProperties": false
                        }
                    },
                    "merge": {
                        "type": "boolean",
                        "description": "true: update existing items by id, add new ones. false (default): replace the entire list.",
                        "default": false
                    }
                },
                "required": [],
                "additionalProperties": false
            }),
            strict: None,
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
        let items = if let Some(todos) = args.todos {
            session.write(todos, args.merge)
        } else {
            session.snapshot()
        };

        let output = TodoOutput {
            summary: TodoSummary::from_items(&items),
            todos: items,
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
        TODO_EMPTY_ID.to_string()
    } else {
        id.to_string()
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
    fn replace_mode_dedupes_and_normalizes_items() {
        let mut store = TodoStore::default();

        let items = store.write(
            vec![
                input("1", Some("draft plan"), Some(TODO_STATUS_PENDING)),
                input("1", Some("final plan"), Some(TODO_STATUS_COMPLETED)),
                input("", Some(""), Some("invalid")),
            ],
            false,
        );

        assert_eq!(
            items,
            vec![
                TodoItem {
                    id: "1".to_string(),
                    content: "final plan".to_string(),
                    status: TODO_STATUS_COMPLETED.to_string(),
                },
                TodoItem {
                    id: TODO_EMPTY_ID.to_string(),
                    content: TODO_EMPTY_CONTENT.to_string(),
                    status: TODO_STATUS_PENDING.to_string(),
                },
            ]
        );
    }

    #[test]
    fn merge_mode_updates_existing_items_and_preserves_order() {
        let mut store = TodoStore::default();
        store.write(
            vec![
                input("1", Some("draft"), Some(TODO_STATUS_PENDING)),
                input("2", Some("implement"), Some(TODO_STATUS_PENDING)),
            ],
            false,
        );

        let items = store.write(
            vec![
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
            ],
            true,
        );

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
        store.write(
            vec![
                input("1", Some("plan"), Some(TODO_STATUS_PENDING)),
                input("2", Some("build"), Some(TODO_STATUS_IN_PROGRESS)),
                input("3", Some("done"), Some(TODO_STATUS_COMPLETED)),
                input("4", Some("skip"), Some(TODO_STATUS_CANCELLED)),
            ],
            false,
        );

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
                    todos: Some(vec![input("1", Some("plan"), Some(TODO_STATUS_PENDING))]),
                    merge: false,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(first.output.summary.total, 1);
        assert!(todo_session(&ctx).has_items());

        let second = tool
            .call(ctx.clone(), TodoArgs::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(second.output.todos, first.output.todos);

        let third = tool
            .call(
                ctx.clone(),
                TodoArgs {
                    todos: Some(vec![TodoItemInput {
                        id: "1".to_string(),
                        content: Some("plan carefully".to_string()),
                        status: None,
                    }]),
                    merge: true,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(third.output.summary.pending, 1);
        assert_eq!(third.output.todos[0].content, "plan carefully");
    }
}
