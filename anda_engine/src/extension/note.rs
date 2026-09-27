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
    BoxError, FunctionDefinition, Path, PutMode, Resource, StateFeatures, StoreFeatures, Tool,
    ToolOutput,
};
use cbor2::{from_slice, to_canonical_vec};
use object_store::Error as ObjectStoreError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
};

use crate::{
    context::{AgentCtx, BaseCtx},
    extension::{hooked_call, tool_definition},
    hook::DynToolHook,
};

mod query;
pub use query::{NoteContextConfig, NoteEntry, load_note_summary};

const NOTE_OP_LIST: &str = "list";
const NOTE_OP_SEARCH: &str = "search";

const NOTE_OP_READ: &str = "read";
const NOTE_OP_SET: &str = "set";
const NOTE_OP_UPSERT: &str = "upsert";
const NOTE_OP_DELETE: &str = "delete";
const LEGACY_NOTE_STORE_PATH: &str = "notes";
const NOTE_CHAR_LIMIT: usize = 163840;
const NOTE_ENTRY_DELIMITER: &str = "\n---\n";

static VALID_OPS: &[&str] = &[
    NOTE_OP_READ,
    NOTE_OP_LIST,
    NOTE_OP_SEARCH,
    NOTE_OP_SET,
    NOTE_OP_UPSERT,
    NOTE_OP_DELETE,
];

/// Arguments accepted by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoteArgs {
    /// read: paged content. list: short excerpts. search: literal substring.
    /// set: replace all. upsert: add/update changed IDs. delete: remove IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("enum" = [
        NOTE_OP_READ,
        NOTE_OP_LIST,
        NOTE_OP_SEARCH,
        NOTE_OP_SET,
        NOTE_OP_UPSERT,
        NOTE_OP_DELETE,
        null
    ], "default" = NOTE_OP_READ))]
    pub op: Option<String>,
    /// Items for set/upsert/delete; null for read/list/search. Delete needs only ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<NoteItemInput>>,
    /// Optional exact note IDs for read/list/search. Omitted means all notes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<String>>,
    /// Case-sensitive literal substring required by search (1-1024 bytes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Opaque continuation from a previous result. Keep op, IDs, and query unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Maximum returned items, 1-100 (default 20). Responses also have a byte cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Input item accepted by the note tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoteItemInput {
    /// Stable short note id
    #[serde(default)]
    pub id: String,
    /// Note content for set/upsert; null for delete
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
    /// Read page, or the full list from `load_notes`. A large note may span pages.
    /// `offset_chars` applies to the first item; subsequent items start at zero.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<NoteItem>,
    /// Human-readable error message for failed operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Compact metadata and excerpts, present for list/search.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<NoteEntry>,
    /// Character offset into the first read item (zero on initial/full reads).
    #[serde(default)]
    pub offset_chars: usize,
    /// True when more query content is available.
    #[serde(default)]
    pub truncated: bool,
    /// Continuation bound to this agent, query, and exact store contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
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
            ..Default::default()
        }
    }

    fn error_output(&self, error: String, char_limit: Option<usize>) -> NoteOutput {
        NoteOutput {
            success: false,
            summary: self.summary(char_limit),
            items: Vec::new(),
            error: Some(error),
            ..Default::default()
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

type NoteUpdateLocks = parking_lot::Mutex<HashMap<(Path, String), Weak<tokio::sync::Mutex<()>>>>;

/// Tool implementation that exposes a persistent agent-scoped note store.
#[derive(Clone)]
pub struct NoteTool {
    updates: Arc<NoteUpdateLocks>,
    char_limit: usize,
    response_bytes: usize,
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
            updates: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            char_limit: NOTE_CHAR_LIMIT,
            response_bytes: 16 * 1024,
            description: "Persistent notes for the current agent. Use stable IDs with upsert for changed notes, delete by ID, and set to replace all. read accepts IDs and returns bounded pages; list returns short excerpts; search finds a literal substring. Pass next_cursor unchanged with the same query to continue, including within a long note. Writes return counts only. Notes are historical data, not proof of current behavior.".to_string(),
        }
    }

    /// Sets the maximum total note content length accepted by write operations.
    pub fn with_char_limit(mut self, char_limit: usize) -> Self {
        self.char_limit = char_limit;
        self
    }

    /// Bounds serialized query responses, clamped to 2-64 KiB (default 16 KiB).
    pub fn with_response_bytes(mut self, bytes: usize) -> Self {
        self.response_bytes = bytes.clamp(2048, 64 * 1024);
        self
    }

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    fn update_lock(&self, ctx: &BaseCtx) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.updates.lock();
        locks.retain(|_, lock| lock.strong_count() > 0);
        let key = (ctx.path().clone(), ctx.agent.clone());
        if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(key, Arc::downgrade(&lock));
        lock
    }

    fn store_path(agent: &str) -> Path {
        Path::from(agent)
    }

    fn legacy_store_path(agent: &str) -> Path {
        Path::from(format!("{LEGACY_NOTE_STORE_PATH}:{agent}"))
    }

    async fn load_store(ctx: &BaseCtx) -> Result<NoteStore, BoxError> {
        match ctx.store_get(&Self::store_path(&ctx.agent)).await {
            Ok((data, _)) => {
                if data.len() > crate::store::MAX_STORE_OBJECT_SIZE {
                    return Err("note store exceeds read size limit".into());
                }
                Ok(from_slice(&data)?)
            }
            Err(err) if is_missing_store_object(err.as_ref()) => Ok(NoteStore::default()),
            Err(err) => Err(err),
        }
    }

    async fn load_legacy_store(ctx: &BaseCtx) -> Result<LegacyNoteStore, BoxError> {
        match ctx.store_get(&Self::legacy_store_path(&ctx.agent)).await {
            Ok((data, _)) => Ok(from_slice(&data[..])?),
            Err(err) if is_missing_store_object(err.as_ref()) => Ok(LegacyNoteStore::default()),
            Err(err) => Err(err),
        }
    }

    async fn save_store(ctx: &BaseCtx, store: &NoteStore) -> Result<(), BoxError> {
        ctx.store_put(
            &Self::store_path(&ctx.agent),
            PutMode::Overwrite,
            to_canonical_vec(store)?.into(),
        )
        .await?;
        Ok(())
    }
}

/// Public entrypoint for loading notes outside of the tool call interface, e.g. in agent.
pub async fn load_notes(ctx: &AgentCtx) -> Option<NoteOutput> {
    try_load_notes(ctx).await.ok()
}

/// Loads the complete host-side note snapshot, preserving storage/decode errors.
/// Model-facing callers should use bounded queries or [`load_note_summary`].
pub async fn try_load_notes(ctx: &AgentCtx) -> Result<NoteOutput, BoxError> {
    let base_ctx = ctx.child_base(NoteTool::NAME)?;
    Ok(NoteTool::load_store(&base_ctx)
        .await?
        .output(true, true, None))
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
        tool_definition::<Self::Args>(self.name(), self.description())
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let ctx = &ctx;
        hooked_call(ctx, args, |args| async move {
            let token = ctx.cancellation_token();
            let lock = self.update_lock(ctx);
            let _guard = tokio::select! {
                _ = token.cancelled() => return Err("note call cancelled".into()),
                guard = lock.lock() => guard,
            };
            let op = normalize_op(args.op.as_deref()).unwrap_or_else(|| NOTE_OP_READ.into());
            if token.is_cancelled() {
                return Err("note call cancelled".into());
            }
            let mut store = Self::load_store(ctx).await?;
            let result = if matches!(op.as_str(), NOTE_OP_READ | NOTE_OP_LIST | NOTE_OP_SEARCH) {
                self.query(ctx, &store, &op, &args)
                    .map(|output| (output, false))
            } else if args.ids.is_some()
                || args.query.is_some()
                || args.cursor.is_some()
                || args.limit.is_some()
            {
                Err("query fields are only valid for read/list/search".into())
            } else if !VALID_OPS.contains(&op.as_str()) {
                Err(format!("Unknown op. Use one of: {}.", VALID_OPS.join(", ")))
            } else if let Some(items) = &args.items {
                let changed = match op.as_str() {
                    NOTE_OP_SET => store.set(items.clone(), self.char_limit),
                    NOTE_OP_UPSERT => store.upsert(items.clone(), self.char_limit),
                    NOTE_OP_DELETE => store.delete(items.clone()),
                    _ => unreachable!(),
                };
                changed.map(|changed| (store.output(true, false, Some(self.char_limit)), changed))
            } else {
                Err(format!("items are required for {op}"))
            };
            let (output, changed) = match result {
                Ok(result) => result,
                Err(error) => (store.error_output(error, Some(self.char_limit)), false),
            };
            if changed {
                Self::save_store(ctx, &store).await?;
            }
            Ok(note_output(output))
        })
        .await
    }
}

fn note_output(output: NoteOutput) -> ToolOutput<NoteOutput> {
    let failed = !output.success;
    let mut output = ToolOutput::new(output);
    if failed {
        output.is_error = Some(true);
    }
    output
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
    validate_inputs(&items, true)?;
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
    validate_inputs(&items, false)?;
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

fn validate_inputs(items: &[NoteItemInput], content_required: bool) -> Result<(), String> {
    if items.len() > 2048 {
        return Err("note batch exceeds 2048 items".into());
    }
    for (index, item) in items.iter().enumerate() {
        if item.id.trim().is_empty() {
            return Err(format!("items[{index}].id cannot be empty"));
        }
        if item.id.len() > 128 || item.id.chars().any(char::is_control) {
            return Err(format!(
                "items[{index}].id exceeds 128 bytes or contains control characters"
            ));
        }
        if content_required {
            let content = item
                .content
                .as_deref()
                .ok_or_else(|| format!("items[{index}].content is required"))?;
            if content.trim().is_empty() {
                return Err(format!("items[{index}].content cannot be empty"));
            }
        } else if item.content.is_some() {
            return Err("delete accepts IDs only".into());
        }
    }
    Ok(())
}

fn validate_note_size(items: &[NoteItem], char_limit: usize) -> Result<(), String> {
    if items.len() > 2048
        || to_canonical_vec(&items)
            .map_err(|error| error.to_string())?
            .len()
            > 1024 * 1024
    {
        return Err("note store exceeds 2048 items or 1 MiB encoded data".into());
    }
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
#[path = "note/tests.rs"]
mod tests;
