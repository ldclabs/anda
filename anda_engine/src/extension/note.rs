//! Persistent note tool for agent-scoped self-memory.
//!
//! This module provides:
//! - a durable note store backed by [`StoreFeatures`],
//! - tool input/output types for reading and mutating notes,
//! - and the public tool entrypoint ([`NoteTool`]).
//!
//! Notes are scoped to the current agent path. Since tools run under the
//! calling agent's context tree, one agent's notes are not visible to another.

use anda_core::{
    BoxError, FunctionDefinition, Path, PutMode, Resource, StoreFeatures, Tool, ToolOutput,
};
use ciborium::from_reader;
use ic_auth_types::deterministic_cbor_into_vec;
use object_store::Error as ObjectStoreError;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;

use crate::{
    context::{AgentCtx, BaseCtx},
    hook::{DynToolHook, ToolHook},
};

const NOTE_ACTION_READ: &str = "read";
const NOTE_ACTION_ADD: &str = "add";
const NOTE_ACTION_REPLACE: &str = "replace";
const NOTE_ACTION_REMOVE: &str = "remove";
const NOTE_STORE_PATH: &str = "notes";
const NOTE_CHAR_LIMIT: usize = 8192;
const NOTE_MATCH_PREVIEW_LIMIT: usize = 80;
const NOTE_EMPTY_CONTENT: &str = "content cannot be empty";
const NOTE_EMPTY_OLD_TEXT: &str = "old_text cannot be empty";
const NOTE_ENTRY_DELIMITER: &str = "\n---\n";

static VALID_ACTIONS: &[&str] = &[
    NOTE_ACTION_READ,
    NOTE_ACTION_ADD,
    NOTE_ACTION_REPLACE,
    NOTE_ACTION_REMOVE,
];

/// Arguments accepted by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteArgs {
    /// Action to perform. Omit or use `read` to retrieve current notes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Note content for `add` and `replace`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Unique substring used to identify a note for `replace` and `remove`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_text: Option<String>,
}

/// Output returned by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteOutput {
    pub success: bool,
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matches: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
struct NoteStore {
    notes: Vec<String>,
}

impl NoteStore {
    fn usage(&self, char_limit: usize) -> String {
        let current = self.char_count();
        let pct = ((current * 100) / char_limit).min(100);
        format!("{pct}% - {current}/{char_limit} chars")
    }

    fn char_count(&self) -> usize {
        joined_len(&self.notes)
    }

    fn success_output(&self, message: Option<String>, char_limit: Option<usize>) -> NoteOutput {
        NoteOutput {
            success: true,
            notes: self.notes.clone(),
            usage: char_limit.map(|limit| self.usage(limit)),
            message,
            error: None,
            matches: None,
        }
    }

    fn failure_output(
        &self,
        error: String,
        matches: Option<Vec<String>>,
        char_limit: Option<usize>,
    ) -> NoteOutput {
        NoteOutput {
            success: false,
            notes: self.notes.clone(),
            usage: char_limit.map(|limit| self.usage(limit)),
            message: None,
            error: Some(error),
            matches,
        }
    }

    fn add(&mut self, content: String, char_limit: usize) -> NoteOutput {
        let content = content.trim();
        if content.is_empty() {
            return self.failure_output(NOTE_EMPTY_CONTENT.to_string(), None, Some(char_limit));
        }

        if self.notes.iter().any(|entry| entry == content) {
            return self.success_output(
                Some("Entry already exists (no duplicate added).".into()),
                Some(char_limit),
            );
        }

        let mut next = self.notes.clone();
        next.push(content.to_string());
        if let Err(error) = validate_note_size(&next) {
            return self.failure_output(error, None, Some(char_limit));
        }

        self.notes = next;
        self.success_output(Some("Entry added.".into()), Some(char_limit))
    }

    fn replace(&mut self, old_text: String, content: String, char_limit: usize) -> NoteOutput {
        let old_text = old_text.trim();
        if old_text.is_empty() {
            return self.failure_output(NOTE_EMPTY_OLD_TEXT.to_string(), None, Some(char_limit));
        }

        let content = content.trim();
        if content.is_empty() {
            return self.failure_output(NOTE_EMPTY_CONTENT.to_string(), None, Some(char_limit));
        }

        let matches = self.find_matches(old_text);
        let Some(index) = resolve_single_match(&matches) else {
            if matches.is_empty() {
                return self.failure_output(
                    format!("No entry matched {:?}.", old_text),
                    None,
                    Some(char_limit),
                );
            }

            return self.failure_output(
                format!("Multiple entries matched {:?}. Be more specific.", old_text),
                Some(preview_matches(&matches)),
                Some(char_limit),
            );
        };

        let mut next = self.notes.clone();
        next[index] = content.to_string();
        if let Err(error) = validate_note_size(&next) {
            return self.failure_output(error, None, Some(char_limit));
        }

        self.notes = next;
        self.success_output(Some("Entry replaced.".into()), Some(char_limit))
    }

    fn remove(&mut self, old_text: String, char_limit: usize) -> NoteOutput {
        let old_text = old_text.trim();
        if old_text.is_empty() {
            return self.failure_output(NOTE_EMPTY_OLD_TEXT.to_string(), None, Some(char_limit));
        }

        let matches = self.find_matches(old_text);
        let Some(index) = resolve_single_match(&matches) else {
            if matches.is_empty() {
                return self.failure_output(
                    format!("No entry matched {:?}.", old_text),
                    None,
                    Some(char_limit),
                );
            }

            return self.failure_output(
                format!("Multiple entries matched {:?}. Be more specific.", old_text),
                Some(preview_matches(&matches)),
                Some(char_limit),
            );
        };

        self.notes.remove(index);
        self.success_output(Some("Entry removed.".into()), Some(char_limit))
    }

    fn find_matches(&self, needle: &str) -> Vec<(usize, String)> {
        self.notes
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.contains(needle))
            .map(|(index, entry)| (index, entry.clone()))
            .collect()
    }
}

pub type NoteToolHook = DynToolHook<NoteArgs, NoteOutput>;

/// Tool implementation that exposes a persistent agent-scoped note store.
#[derive(Clone)]
pub struct NoteTool {
    char_limit: usize,
    description: String,
}

impl Default for NoteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl NoteTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "note";

    /// Creates a note tool with the default behavioral guidance.
    pub fn new() -> Self {
        Self {
            char_limit: NOTE_CHAR_LIMIT,
            description: concat!(
                "Manage persistent notes for the current agent only. ",
                "These notes are stored durably and are isolated by agent, ",
                "so other agents cannot read them. ",
                "Call with no parameters or action=read to read the current notes.\n\n",
                "Writing:\n",
                "- action=add with content: append a new note\n",
                "- action=replace with old_text and content: replace one matching note\n",
                "- action=remove with old_text: remove one matching note\n\n",
                "Use short unique substrings for old_text. ",
                "Exact duplicate notes are ignored. ",
                "The store is bounded to prevent unbounded growth.\n\n",
                "Always returns the full current note list and usage."
            )
            .to_string(),
        }
    }

    pub fn with_char_limit(mut self, char_limit: usize) -> Self {
        self.char_limit = char_limit;
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    fn store_path(agent: &str) -> Path {
        Path::from(format!("{NOTE_STORE_PATH}:{agent}"))
    }

    async fn load_store(ctx: &BaseCtx) -> Result<NoteStore, BoxError> {
        match ctx.store_get(&Self::store_path(&ctx.agent)).await {
            Ok((data, _)) => Ok(from_reader(&data[..])?),
            Err(err) if is_missing_store_object(err.as_ref()) => Ok(NoteStore::default()),
            Err(err) => Err(err),
        }
    }

    async fn save_store(ctx: &BaseCtx, store: &NoteStore) -> Result<(), BoxError> {
        let data = deterministic_cbor_into_vec(store)?;
        ctx.store_put(
            &Self::store_path(&ctx.agent),
            PutMode::Overwrite,
            data.into(),
        )
        .await?;
        Ok(())
    }
}

/// Public entrypoint for loading notes outside of the tool call interface, e.g. in agent.
pub async fn load_notes(ctx: &AgentCtx) -> Option<NoteOutput> {
    let base_ctx = ctx.child_base(NoteTool::NAME).ok()?;
    NoteTool::load_store(&base_ctx)
        .await
        .ok()
        .map(|store| store.success_output(None, None))
}

impl Tool<BaseCtx> for NoteTool {
    type Args = NoteArgs;
    type Output = NoteOutput;

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
                    "action": {
                        "type": "string",
                        "enum": VALID_ACTIONS,
                        "description": "Action to perform. Omit or use read to return the current notes.",
                        "default": NOTE_ACTION_READ
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content for add and replace."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Unique substring identifying the note to replace or remove."
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
        let hook = ctx.get_state::<NoteToolHook>();
        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let mut store = Self::load_store(&ctx).await?;
        let action = args
            .action
            .as_deref()
            .map(|value| value.trim().to_ascii_lowercase())
            .unwrap_or_else(|| NOTE_ACTION_READ.to_string());

        let output = match action.as_str() {
            NOTE_ACTION_READ => store.success_output(None, Some(self.char_limit)),
            NOTE_ACTION_ADD => match args.content {
                Some(content) => {
                    let output = store.add(content, self.char_limit);
                    if output.success {
                        Self::save_store(&ctx, &store).await?;
                    }
                    output
                }
                None => store.failure_output(
                    "content is required for add".into(),
                    None,
                    Some(self.char_limit),
                ),
            },
            NOTE_ACTION_REPLACE => match (args.old_text, args.content) {
                (Some(old_text), Some(content)) => {
                    let output = store.replace(old_text, content, self.char_limit);
                    if output.success {
                        Self::save_store(&ctx, &store).await?;
                    }
                    output
                }
                (None, _) => store.failure_output(
                    "old_text is required for replace".into(),
                    None,
                    Some(self.char_limit),
                ),
                (_, None) => store.failure_output(
                    "content is required for replace".into(),
                    None,
                    Some(self.char_limit),
                ),
            },
            NOTE_ACTION_REMOVE => match args.old_text {
                Some(old_text) => {
                    let output = store.remove(old_text, self.char_limit);
                    if output.success {
                        Self::save_store(&ctx, &store).await?;
                    }
                    output
                }
                None => store.failure_output(
                    "old_text is required for remove".into(),
                    None,
                    Some(self.char_limit),
                ),
            },
            _ => store.failure_output(
                format!(
                    "Unknown action {:?}. Use one of: {}.",
                    action,
                    VALID_ACTIONS.join(", ")
                ),
                None,
                Some(self.char_limit),
            ),
        };

        let output = ToolOutput::new(output);
        if let Some(hook) = &hook {
            return hook.after_tool_call(&ctx, output).await;
        }

        Ok(output)
    }
}

fn validate_note_size(notes: &[String]) -> Result<(), String> {
    let current = joined_len(notes);
    if current > NOTE_CHAR_LIMIT {
        return Err(format!(
            "Notes use {current}/{NOTE_CHAR_LIMIT} chars. Shorten the new content or remove older notes first."
        ));
    }

    Ok(())
}

fn joined_len(notes: &[String]) -> usize {
    if notes.is_empty() {
        0
    } else {
        notes.join(NOTE_ENTRY_DELIMITER).chars().count()
    }
}

fn resolve_single_match(matches: &[(usize, String)]) -> Option<usize> {
    if matches.is_empty() {
        return None;
    }

    if matches.len() == 1 {
        return Some(matches[0].0);
    }

    let unique: HashSet<&str> = matches.iter().map(|(_, entry)| entry.as_str()).collect();
    if unique.len() == 1 {
        return Some(matches[0].0);
    }

    None
}

fn preview_matches(matches: &[(usize, String)]) -> Vec<String> {
    matches
        .iter()
        .map(|(_, entry)| {
            if entry.chars().count() > NOTE_MATCH_PREVIEW_LIMIT {
                format!(
                    "{}...",
                    entry
                        .chars()
                        .take(NOTE_MATCH_PREVIEW_LIMIT)
                        .collect::<String>()
                )
            } else {
                entry.clone()
            }
        })
        .collect()
}

fn is_missing_store_object(err: &(dyn std::error::Error + 'static)) -> bool {
    err.downcast_ref::<ObjectStoreError>()
        .is_some_and(|err| matches!(err, ObjectStoreError::NotFound { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::AgentCtx, engine::EngineBuilder};

    fn agent_ctx(name: &str) -> AgentCtx {
        EngineBuilder::new()
            .mock_ctx()
            .child(name, name)
            .expect("create child agent ctx")
    }

    fn note_ctx(name: &str) -> BaseCtx {
        agent_ctx(name)
            .child_base(NoteTool::NAME)
            .expect("create note tool ctx")
    }

    #[test]
    fn store_add_replace_remove_match_memory_style() {
        let mut store = NoteStore::default();

        let added = store.add("remember the release checklist".into(), NOTE_CHAR_LIMIT);
        assert!(added.success);
        assert_eq!(
            store.notes,
            vec!["remember the release checklist".to_string()]
        );

        let duplicate = store.add("remember the release checklist".into(), NOTE_CHAR_LIMIT);
        assert!(duplicate.success);
        assert_eq!(store.notes.len(), 1);

        let replaced = store.replace(
            "release".into(),
            "remember the launch checklist".into(),
            NOTE_CHAR_LIMIT,
        );
        assert!(replaced.success);
        assert_eq!(
            store.notes,
            vec!["remember the launch checklist".to_string()]
        );

        let removed = store.remove("launch".into(), NOTE_CHAR_LIMIT);
        assert!(removed.success);
        assert!(store.notes.is_empty());
    }

    #[test]
    fn store_reports_ambiguous_matches() {
        let mut store = NoteStore {
            notes: vec![
                "remember alpha release".to_string(),
                "remember alpha rollout".to_string(),
            ],
        };

        let output = store.replace("alpha".into(), "new note".into(), NOTE_CHAR_LIMIT);
        assert!(!output.success);
        assert_eq!(store.notes.len(), 2);
        assert_eq!(output.matches.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn tool_reads_empty_store_before_first_write() {
        let tool = NoteTool::new();
        let output = tool
            .call(note_ctx("writer"), NoteArgs::default(), Vec::new())
            .await
            .unwrap();

        assert!(output.output.success);
        assert!(output.output.notes.is_empty());
    }

    #[tokio::test]
    async fn tool_persists_notes_across_calls_for_same_agent() {
        let tool = NoteTool::new();
        let ctx = note_ctx("writer");

        let first = tool
            .call(
                ctx.clone(),
                NoteArgs {
                    action: Some(NOTE_ACTION_ADD.to_string()),
                    content: Some("remember to tag releases".to_string()),
                    old_text: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(first.output.success);
        assert_eq!(
            first.output.notes,
            vec!["remember to tag releases".to_string()]
        );

        let second = tool
            .call(ctx.clone(), NoteArgs::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(second.output.notes, first.output.notes);

        let third = tool
            .call(
                ctx,
                NoteArgs {
                    action: Some(NOTE_ACTION_REPLACE.to_string()),
                    content: Some("remember to tag stable releases".to_string()),
                    old_text: Some("tag releases".to_string()),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(third.output.success);
        assert_eq!(
            third.output.notes,
            vec!["remember to tag stable releases".to_string()]
        );
    }

    #[tokio::test]
    async fn tool_storage_is_isolated_between_agents() {
        let tool = NoteTool::new();

        let writer = tool
            .call(
                note_ctx("writer"),
                NoteArgs {
                    action: Some(NOTE_ACTION_ADD.to_string()),
                    content: Some("writer only note".to_string()),
                    old_text: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(writer.output.success);
        assert_eq!(writer.output.notes, vec!["writer only note".to_string()]);

        let reviewer = tool
            .call(note_ctx("reviewer"), NoteArgs::default(), Vec::new())
            .await
            .unwrap();
        assert!(reviewer.output.success);
        assert!(reviewer.output.notes.is_empty());
    }
}
