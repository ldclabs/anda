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
use std::collections::{HashMap, HashSet};

use crate::{
    context::{AgentCtx, BaseCtx},
    hook::{DynToolHook, ToolHook},
};

const NOTE_OP_READ: &str = "read";
const NOTE_OP_SET: &str = "set";
const NOTE_OP_UPSERT: &str = "upsert";
const NOTE_OP_DELETE: &str = "delete";
const LEGACY_NOTE_STORE_PATH: &str = "notes";
const NOTE_CHAR_LIMIT: usize = 16384;
const NOTE_ENTRY_DELIMITER: &str = "\n---\n";

static VALID_OPS: &[&str] = &[NOTE_OP_READ, NOTE_OP_SET, NOTE_OP_UPSERT, NOTE_OP_DELETE];

/// Arguments accepted by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteArgs {
    /// Operation to perform: read, set, upsert, or delete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    /// Items for set/upsert/delete. Omit or null for read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<NoteItemInput>>,
}

/// Input item accepted by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteItemInput {
    /// Stable short id used for future updates/deletes.
    #[serde(default)]
    pub id: String,
    /// Note content for set/upsert. Use null for delete.
    #[serde(default)]
    pub content: Option<String>,
}

/// Normalized persistent note item.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteItem {
    /// Stable short note identifier.
    pub id: String,
    /// Note content.
    pub content: String,
}

/// Compact note store usage summary.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteSummary {
    /// Number of stored notes.
    pub total: usize,
    /// Character count after notes are joined for prompt use.
    pub chars: usize,
    /// Configured character limit, when one is enforced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Output returned by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NoteOutput {
    /// Whether the requested operation succeeded.
    pub success: bool,
    /// Store usage summary after the operation.
    pub summary: NoteSummary,
    /// Full note list. Present only for read operations and `load_notes`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<NoteItem>,
    /// Human-readable error message for failed operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
struct NoteStore {
    items: Vec<NoteItem>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
struct LegacyNoteStore {
    notes: Vec<String>,
}

impl NoteStore {
    fn set(&mut self, items: Vec<NoteItemInput>, char_limit: usize) -> Result<bool, String> {
        let next = normalize_note_items(items)?;
        validate_note_size(&next, char_limit)?;

        let changed = self.items != next;
        self.items = next;
        Ok(changed)
    }

    fn upsert(&mut self, items: Vec<NoteItemInput>, char_limit: usize) -> Result<bool, String> {
        let updates = normalize_note_items(items)?;
        if updates.is_empty() {
            return Ok(false);
        }

        let mut next = self.items.clone();
        let mut index_by_id: HashMap<String, usize> = next
            .iter()
            .enumerate()
            .map(|(index, item)| (item.id.clone(), index))
            .collect();

        for item in updates {
            if let Some(index) = index_by_id.get(&item.id).copied() {
                next[index] = item;
            } else {
                index_by_id.insert(item.id.clone(), next.len());
                next.push(item);
            }
        }

        validate_note_size(&next, char_limit)?;
        let changed = self.items != next;
        self.items = next;
        Ok(changed)
    }

    fn delete(&mut self, items: Vec<NoteItemInput>) -> Result<bool, String> {
        let ids = normalize_note_ids(items)?;
        if ids.is_empty() {
            return Ok(false);
        }

        let ids: HashSet<String> = ids.into_iter().collect();
        let old_len = self.items.len();
        self.items.retain(|item| !ids.contains(&item.id));
        Ok(self.items.len() != old_len)
    }

    fn output(&self, success: bool, include_items: bool, char_limit: Option<usize>) -> NoteOutput {
        NoteOutput {
            success,
            summary: self.summary(char_limit),
            items: if include_items {
                self.items.clone()
            } else {
                Vec::new()
            },
            error: None,
        }
    }

    fn error_output(&self, error: String, char_limit: Option<usize>) -> NoteOutput {
        NoteOutput {
            success: false,
            summary: self.summary(char_limit),
            items: Vec::new(),
            error: Some(error),
        }
    }

    fn summary(&self, char_limit: Option<usize>) -> NoteSummary {
        NoteSummary {
            total: self.items.len(),
            chars: joined_len(&self.items),
            limit: char_limit,
        }
    }
}

/// Typed hook for note tool calls.
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
                "Persistent notes for the current agent only. Use op=upsert ",
                "with stable short ids to add or update only changed notes; ",
                "op=delete removes by id; op=set replaces all notes; op=read ",
                "returns the full list. Writes return only summary counts."
            )
            .to_string(),
        }
    }

    /// Sets the maximum total note content length accepted by write operations.
    pub fn with_char_limit(mut self, char_limit: usize) -> Self {
        self.char_limit = char_limit;
        self
    }

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    fn store_path(agent: &str) -> Path {
        Path::from(agent)
    }

    fn legacy_store_path(agent: &str) -> Path {
        Path::from(format!("{LEGACY_NOTE_STORE_PATH}:{agent}"))
    }

    async fn load_store(ctx: &BaseCtx) -> Result<NoteStore, BoxError> {
        match ctx.store_get(&Self::store_path(&ctx.agent)).await {
            Ok((data, _)) => Ok(from_reader(&data[..])?),
            Err(err) if is_missing_store_object(err.as_ref()) => Ok(NoteStore::default()),
            Err(err) => Err(err),
        }
    }

    async fn load_legacy_store(ctx: &BaseCtx) -> Result<LegacyNoteStore, BoxError> {
        match ctx.store_get(&Self::legacy_store_path(&ctx.agent)).await {
            Ok((data, _)) => Ok(from_reader(&data[..])?),
            Err(err) if is_missing_store_object(err.as_ref()) => Ok(LegacyNoteStore::default()),
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
        .map(|store| store.output(true, true, None))
}

/// Loads notes from the pre note store without mutating the current store.
pub async fn load_notes_from_legacy(ctx: &AgentCtx) -> Option<NoteOutput> {
    let mut base_ctx = ctx.child_base(NoteTool::NAME).ok()?;
    base_ctx.path = "t:note".into();
    NoteTool::load_legacy_store(&base_ctx)
        .await
        .ok()
        .map(legacy_store_output)
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
                    "op": {
                        "type": ["string", "null"],
                        "enum": [
                            NOTE_OP_READ,
                            NOTE_OP_SET,
                            NOTE_OP_UPSERT,
                            NOTE_OP_DELETE,
                            null
                        ],
                        "description": "read: return notes. set: replace all. upsert: add/update changed ids. delete: remove ids.",
                        "default": NOTE_OP_READ
                    },
                    "items": {
                        "type": ["array", "null"],
                        "description": "Items for set/upsert/delete. Use null for read. Delete only needs id with content=null.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable short note id"
                                },
                                "content": {
                                    "type": ["string", "null"],
                                    "description": "Note content for set/upsert; null for delete"
                                }
                            },
                            "required": ["id", "content"],
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
        let hook = ctx.get_state::<NoteToolHook>();
        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let mut store = Self::load_store(&ctx).await?;
        let op = normalize_op(args.op.as_deref());
        let items = args.items;

        let (output, changed) = match op.as_deref() {
            Some(NOTE_OP_READ) => (store.output(true, true, Some(self.char_limit)), false),
            Some(NOTE_OP_SET) => match items {
                Some(items) => match store.set(items, self.char_limit) {
                    Ok(changed) => (store.output(true, false, Some(self.char_limit)), changed),
                    Err(error) => (store.error_output(error, Some(self.char_limit)), false),
                },
                None => (
                    store.error_output("items are required for set".into(), Some(self.char_limit)),
                    false,
                ),
            },
            Some(NOTE_OP_UPSERT) => match items {
                Some(items) => match store.upsert(items, self.char_limit) {
                    Ok(changed) => (store.output(true, false, Some(self.char_limit)), changed),
                    Err(error) => (store.error_output(error, Some(self.char_limit)), false),
                },
                None => (
                    store.error_output(
                        "items are required for upsert".into(),
                        Some(self.char_limit),
                    ),
                    false,
                ),
            },
            Some(NOTE_OP_DELETE) => match items {
                Some(items) => match store.delete(items) {
                    Ok(changed) => (store.output(true, false, Some(self.char_limit)), changed),
                    Err(error) => (store.error_output(error, Some(self.char_limit)), false),
                },
                None => (
                    store.error_output(
                        "items are required for delete".into(),
                        Some(self.char_limit),
                    ),
                    false,
                ),
            },
            Some(op) => (
                store.error_output(
                    format!("Unknown op {:?}. Use one of: {}.", op, VALID_OPS.join(", ")),
                    Some(self.char_limit),
                ),
                false,
            ),
            None => (store.output(true, true, Some(self.char_limit)), false),
        };

        if changed {
            Self::save_store(&ctx, &store).await?;
        }

        let output = ToolOutput::new(output);
        if let Some(hook) = &hook {
            return hook.after_tool_call(&ctx, output).await;
        }

        Ok(output)
    }
}

fn normalize_op(op: Option<&str>) -> Option<String> {
    op.map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .or_else(|| Some(NOTE_OP_READ.to_string()))
}

fn legacy_store_output(store: LegacyNoteStore) -> NoteOutput {
    let items = store
        .notes
        .into_iter()
        .enumerate()
        .filter_map(|(index, content)| {
            let content = content.trim();
            if content.is_empty() {
                return None;
            }

            Some(NoteItem {
                id: format!("legacy_{}", index + 1),
                content: content.to_string(),
            })
        })
        .collect();

    NoteStore { items }.output(true, true, None)
}

fn normalize_note_items(items: Vec<NoteItemInput>) -> Result<Vec<NoteItem>, String> {
    let mut last_index: HashMap<String, usize> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let id = item.id.trim();
        if id.is_empty() {
            return Err(format!("items[{index}].id cannot be empty"));
        }
        last_index.insert(id.to_string(), index);
    }

    let mut indexes: Vec<usize> = last_index.into_values().collect();
    indexes.sort_unstable();

    indexes
        .into_iter()
        .map(|index| {
            let item = &items[index];
            let id = item.id.trim();
            let Some(content) = item.content.as_deref() else {
                return Err(format!("items[{index}].content is required"));
            };
            let content = content.trim();
            if content.is_empty() {
                return Err(format!("items[{index}].content cannot be empty"));
            }

            Ok(NoteItem {
                id: id.to_string(),
                content: content.to_string(),
            })
        })
        .collect()
}

fn normalize_note_ids(items: Vec<NoteItemInput>) -> Result<Vec<String>, String> {
    let mut last_index: HashMap<String, usize> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let id = item.id.trim();
        if id.is_empty() {
            return Err(format!("items[{index}].id cannot be empty"));
        }
        last_index.insert(id.to_string(), index);
    }

    let mut indexes: Vec<usize> = last_index.into_values().collect();
    indexes.sort_unstable();
    Ok(indexes
        .into_iter()
        .map(|index| items[index].id.trim().to_string())
        .collect())
}

fn validate_note_size(items: &[NoteItem], char_limit: usize) -> Result<(), String> {
    let current = joined_len(items);
    if current > char_limit {
        return Err(format!(
            "Notes use {current}/{char_limit} chars. Shorten new content or delete older notes first."
        ));
    }

    Ok(())
}

fn joined_len(items: &[NoteItem]) -> usize {
    if items.is_empty() {
        return 0;
    }

    let delimiter_len = NOTE_ENTRY_DELIMITER.chars().count() * (items.len() - 1);
    let item_len = items
        .iter()
        .map(|item| item.id.chars().count() + item.content.chars().count() + 2)
        .sum::<usize>();
    item_len + delimiter_len
}

fn is_missing_store_object(err: &(dyn std::error::Error + 'static)) -> bool {
    err.downcast_ref::<ObjectStoreError>()
        .is_some_and(|err| matches!(err, ObjectStoreError::NotFound { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::AgentCtx, engine::EngineBuilder};
    use async_trait::async_trait;
    use std::sync::Arc;

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

    fn input(id: &str, content: Option<&str>) -> NoteItemInput {
        NoteItemInput {
            id: id.to_string(),
            content: content.map(ToString::to_string),
        }
    }

    fn item(id: &str, content: &str) -> NoteItem {
        NoteItem {
            id: id.to_string(),
            content: content.to_string(),
        }
    }

    struct MutatingHook;

    #[async_trait]
    impl ToolHook<NoteArgs, NoteOutput> for MutatingHook {
        async fn before_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut args: NoteArgs,
        ) -> Result<NoteArgs, BoxError> {
            args.op = Some(NOTE_OP_UPSERT.to_string());
            args.items = Some(vec![input("hook", Some("hook inserted note"))]);
            Ok(args)
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut output: ToolOutput<NoteOutput>,
        ) -> Result<ToolOutput<NoteOutput>, BoxError> {
            output.output.summary.limit = Some(1);
            Ok(output)
        }
    }

    #[test]
    fn store_set_upsert_delete_by_stable_id() {
        let mut store = NoteStore::default();

        assert!(
            store
                .set(
                    vec![
                        input("release", Some("remember release checklist")),
                        input("release", Some("remember launch checklist")),
                    ],
                    NOTE_CHAR_LIMIT,
                )
                .unwrap()
        );
        assert_eq!(
            store.items,
            vec![item("release", "remember launch checklist")]
        );

        assert!(
            store
                .upsert(
                    vec![
                        input("release", Some("remember stable release tags")),
                        input("review", Some("prefer focused review notes")),
                    ],
                    NOTE_CHAR_LIMIT,
                )
                .unwrap()
        );
        assert_eq!(
            store.items,
            vec![
                item("release", "remember stable release tags"),
                item("review", "prefer focused review notes"),
            ]
        );

        assert!(store.delete(vec![input("release", None)]).unwrap());
        assert_eq!(
            store.items,
            vec![item("review", "prefer focused review notes")]
        );
        assert!(!store.delete(vec![input("missing", None)]).unwrap());
    }

    #[test]
    fn store_reports_validation_and_limit_errors() {
        let mut store = NoteStore::default();

        assert_eq!(
            store
                .set(vec![input(" ", Some("content"))], NOTE_CHAR_LIMIT)
                .unwrap_err(),
            "items[0].id cannot be empty"
        );
        assert_eq!(
            store
                .upsert(vec![input("alpha", None)], NOTE_CHAR_LIMIT)
                .unwrap_err(),
            "items[0].content is required"
        );
        assert_eq!(
            store
                .upsert(vec![input("alpha", Some(" "))], NOTE_CHAR_LIMIT)
                .unwrap_err(),
            "items[0].content cannot be empty"
        );
        assert_eq!(
            store.delete(vec![input("", None)]).unwrap_err(),
            "items[0].id cannot be empty"
        );

        let oversized = "x".repeat(33);
        assert!(
            store
                .set(vec![input("a", Some(&oversized))], 32)
                .unwrap_err()
                .contains("32")
        );
    }

    #[tokio::test]
    async fn tool_reads_empty_store_before_first_write() {
        let tool = NoteTool::new();
        let output = tool
            .call(note_ctx("writer"), NoteArgs::default(), Vec::new())
            .await
            .unwrap();

        assert!(output.output.success);
        assert!(output.output.items.is_empty());
        assert_eq!(output.output.summary.total, 0);
    }

    #[tokio::test]
    async fn tool_persists_items_and_write_outputs_are_compact() {
        let tool = NoteTool::new();
        let ctx = note_ctx("writer");

        let first = tool
            .call(
                ctx.clone(),
                NoteArgs {
                    op: Some(NOTE_OP_UPSERT.to_string()),
                    items: Some(vec![input("release", Some("remember to tag releases"))]),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(first.output.success);
        assert_eq!(first.output.summary.total, 1);
        assert!(first.output.items.is_empty());

        let second = tool
            .call(ctx.clone(), NoteArgs::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(
            second.output.items,
            vec![item("release", "remember to tag releases")]
        );

        let third = tool
            .call(
                ctx,
                NoteArgs {
                    op: Some(NOTE_OP_UPSERT.to_string()),
                    items: Some(vec![input(
                        "release",
                        Some("remember to tag stable releases"),
                    )]),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(third.output.success);
        assert!(third.output.items.is_empty());
        assert_eq!(third.output.summary.total, 1);
    }

    #[tokio::test]
    async fn tool_storage_is_isolated_between_agents() {
        let tool = NoteTool::new();

        let writer = tool
            .call(
                note_ctx("writer"),
                NoteArgs {
                    op: Some(NOTE_OP_UPSERT.to_string()),
                    items: Some(vec![input("owner", Some("writer only note"))]),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(writer.output.success);
        assert_eq!(writer.output.summary.total, 1);

        let reviewer = tool
            .call(note_ctx("reviewer"), NoteArgs::default(), Vec::new())
            .await
            .unwrap();
        assert!(reviewer.output.success);
        assert!(reviewer.output.items.is_empty());
    }

    #[tokio::test]
    async fn tool_reports_validation_errors_and_unknown_ops_without_persisting() {
        let tool = NoteTool::default()
            .with_char_limit(32)
            .with_description("custom note description".to_string());
        assert_eq!(tool.description(), "custom note description");
        let ctx = note_ctx("validation");

        let missing_items = tool
            .call(
                ctx.clone(),
                NoteArgs {
                    op: Some(NOTE_OP_UPSERT.to_string()),
                    items: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            missing_items.output.error.as_deref(),
            Some("items are required for upsert")
        );
        assert_eq!(missing_items.output.summary.limit, Some(32));

        let missing_content = tool
            .call(
                ctx.clone(),
                NoteArgs {
                    op: Some(NOTE_OP_SET.to_string()),
                    items: Some(vec![input("entry", None)]),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            missing_content.output.error.as_deref(),
            Some("items[0].content is required")
        );

        let unknown = tool
            .call(
                ctx.clone(),
                NoteArgs {
                    op: Some("archive".to_string()),
                    items: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(
            unknown
                .output
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Unknown op"))
        );

        let read = tool
            .call(ctx, NoteArgs::default(), Vec::new())
            .await
            .unwrap();
        assert!(read.output.items.is_empty());
    }

    #[tokio::test]
    async fn tool_hooks_and_load_notes_use_agent_scoped_store() {
        let engine_ctx = EngineBuilder::new().mock_ctx();
        let agent = engine_ctx
            .child("hooked", "hooked")
            .expect("create child agent ctx");
        let ctx = agent.child_base(NoteTool::NAME).unwrap();
        ctx.set_state(NoteToolHook::new(Arc::new(MutatingHook)));

        let tool = NoteTool::new();
        let output = tool
            .call(
                ctx,
                NoteArgs {
                    op: Some(NOTE_OP_READ.to_string()),
                    items: None,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(output.output.success);
        assert_eq!(output.output.summary.limit, Some(1));

        let loaded = load_notes(&agent).await.unwrap();
        assert!(loaded.success);
        assert_eq!(loaded.items, vec![item("hook", "hook inserted note")]);
        assert_eq!(loaded.summary.limit, None);
    }

    #[tokio::test]
    async fn load_notes_from_legacy_reads_old_store_without_touching_v2() {
        let agent = agent_ctx("legacy");
        let mut ctx = agent.child_base(NoteTool::NAME).unwrap();
        ctx.path = "t:note".into();
        let legacy = LegacyNoteStore {
            notes: vec![
                "remember old release process".to_string(),
                "prefer concise persisted notes".to_string(),
            ],
        };
        ctx.store_put(
            &NoteTool::legacy_store_path(&ctx.agent),
            PutMode::Overwrite,
            deterministic_cbor_into_vec(&legacy).unwrap().into(),
        )
        .await
        .unwrap();

        let loaded = load_notes_from_legacy(&agent).await.unwrap();
        assert!(loaded.success);
        assert_eq!(
            loaded.items,
            vec![
                item("legacy_1", "remember old release process"),
                item("legacy_2", "prefer concise persisted notes"),
            ]
        );
        assert_eq!(loaded.summary.total, 2);
        assert_eq!(loaded.summary.limit, None);

        let current = load_notes(&agent).await.unwrap();
        assert!(current.items.is_empty());
    }

    #[test]
    fn definition_schema_avoids_anyof() {
        let definition = NoteTool::new().definition();

        assert!(
            definition.parameters["properties"]["op"]
                .get("anyOf")
                .is_none()
        );
        assert_eq!(
            definition.parameters["properties"]["op"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            definition.parameters["properties"]["op"]["enum"],
            json!([
                NOTE_OP_READ,
                NOTE_OP_SET,
                NOTE_OP_UPSERT,
                NOTE_OP_DELETE,
                null
            ])
        );
        assert_eq!(
            definition.parameters["properties"]["items"]["items"]["required"],
            json!(["id", "content"])
        );
        assert_eq!(definition.parameters["required"], json!(["op", "items"]));
    }
}
