//! Persistent memory tools and conversation storage.
//!
//! This module stores conversations, resources, artifacts, and memory-management
//! commands for agents. Conversation metadata is stored in AndaDB collections
//! with BTree and BM25 indexes, while higher-level memory operations are exposed
//! through KIP (Knowledge Interaction Protocol) tools backed by the Cognitive
//! Nexus.
//!
//! The KIP, resource, and conversation tools report the same capability group
//! via [`memory_tool_group_info`], so the discovery layer presents them to the
//! model as one persistent-memory bundle.

use anda_cognitive_nexus::{
    CognitiveNexus, SpaceDraft,
    governance::{AuthContext, rows::principal_class, store::PrincipalDraft},
    nexus::{DEFAULT_SPACE, Session},
};
use anda_core::{
    BoxError, ContentPart, Document, Documents, FunctionDefinition, Message, Resource, ResourceRef,
    StateFeatures, Tool, ToolGroupInfo, ToolOutput, Usage, Xid,
};
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::AndaDB,
    error::DBError,
    index::BTree,
    query::{Filter, Query, RangeQuery, Search},
};
use anda_db_schema::{AndaDBSchema, Ft, Fv, Json};
use anda_db_tfs::jieba_tokenizer;
use anda_kip::{
    ErrorObject, Execution, KIP_FUNCTION_DEFINITION, KIP_READONLY_FUNCTION_DEFINITION, KipError,
    KipErrorCode, Operation, OperationResult, ReadBinding, Request, RequestOptions, Response,
    SpaceSelector, TopLevelStatus, execute_request, execute_request_readonly,
};
use async_trait::async_trait;
use candid::Principal;
use cbor2::cbor;
use ic_auth_types::ByteBufB64;
use moka::future::Cache;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{Arc, LazyLock},
};

use crate::{
    context::BaseCtx,
    extension::{fetch::FetchWebResourcesTool, hooked_call, tool_definition},
    rfc3339_datetime, unix_ms,
};

/// Stable id of the persistent memory capability group.
pub const MEMORY_TOOL_GROUP_ID: &str = "memory";

/// Returns the shared [`ToolGroupInfo`] for the persistent memory tools.
///
/// The KIP execution, resource, and conversation tools all report this so the
/// registry presents them as one bundle. The registry fills in the member list
/// from the tools actually registered.
pub fn memory_tool_group_info() -> ToolGroupInfo {
    ToolGroupInfo {
        id: MEMORY_TOOL_GROUP_ID.to_string(),
        title: "Persistent memory".to_string(),
        description: "Store and recall long-term knowledge and past conversations from the agent's persistent memory (Cognitive Nexus + conversation store).".to_string(),
        instructions: Some(
            "These tools share one persistent memory backend. Use the KIP execution tool (`execute_kip` / `execute_kip_readonly`) to query or update the Cognitive Nexus knowledge graph; prefer the read-only variant for retrieval. Use `list_previous_conversations` and `search_conversations` to recall past dialogue, `get_resource_content` to fetch a stored resource by id, and `memory_api` for unified conversation management (stop/steer/follow-up/delete, logs). Retrieve context before acting, and write durable facts back so they survive across sessions.".to_string(),
        ),
    }
}

/// Default KIP tool function definition used by [`MemoryManagement`].
///
/// `anda_kip` ships the agent-facing definition alongside the protocol it
/// describes, so the tool a model sees and the envelope the engine executes
/// stay in step across protocol revisions.
pub static FUNCTION_DEFINITION: LazyLock<FunctionDefinition> =
    LazyLock::new(|| serde_json::from_value(KIP_FUNCTION_DEFINITION.clone()).unwrap());

/// Default read-only KIP tool function definition used by [`MemoryReadonly`].
///
/// Not the writable definition with a different name: KIP 2.0's read-only entry
/// point offers no write vocabulary and no `execution` modes to choose between,
/// so reusing the writable schema would advertise a batch mode the read path
/// has no use for.
pub static READONLY_FUNCTION_DEFINITION: LazyLock<FunctionDefinition> =
    LazyLock::new(|| serde_json::from_value(KIP_READONLY_FUNCTION_DEFINITION.clone()).unwrap());

/// One entry of [`KipArgs::operations`].
///
/// The tool schema accepts a bare command string as well as an operation
/// object, because a model batching three reads should not have to spell out
/// three objects to do it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum KipOperation {
    /// Just the command text.
    Command(String),
    /// The full operation, with its own `op_id` and parameters.
    Operation(Box<Operation>),
}

impl From<KipOperation> for Operation {
    fn from(op: KipOperation) -> Self {
        match op {
            KipOperation::Command(command) => Operation::new(command),
            KipOperation::Operation(operation) => *operation,
        }
    }
}

/// The model-facing arguments of `execute_kip` and `execute_kip_readonly`.
///
/// KIP 2.0's bundled tool definitions let a model send a single `command`, or an
/// `operations` array whose items may be bare strings, and put `dry_run` at the
/// top level. That is deliberately easier to write than the wire [`Request`],
/// which carries the protocol tag, nests `dry_run` under `options`, and rejects
/// unknown fields. This type is the bridge, so what the model was shown and what
/// the engine executes are the same thing.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KipArgs {
    /// A single complete KIP command. Mutually exclusive with `operations`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,

    /// Several KIP commands in one round-trip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operations: Option<Vec<KipOperation>>,

    /// How the operations relate to one another.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<Execution>,

    /// The read coordinate to bind every operation to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read: Option<ReadBinding>,

    /// Values bound into the commands' `:placeholders`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Map<String, Json>>,

    /// Validate and plan without committing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,
}

impl KipArgs {
    /// Builds the read-only tool envelope. Its bundled schema exposes no
    /// transaction mode, so a batch defaults to independent reads. Raw wire
    /// requests and the writable tool keep KIP's explicit-mode requirement.
    #[allow(clippy::result_large_err)]
    pub fn into_readonly_request(self) -> Result<Request, KipError> {
        let mut request = self.into_request()?;
        if request.operations.len() > 1 && request.execution.is_none() {
            request.execution = Some(Execution::new(anda_kip::ExecutionMode::Independent));
        }
        Ok(request)
    }

    /// Builds the wire envelope these arguments describe.
    ///
    /// The `command` / `operations` exclusion is checked here rather than left
    /// to [`Request::validate`], because an envelope built from both would
    /// silently run only one of them.
    // `KipError` is the protocol's own error type and carries its registry code,
    // hint and structured detail; boxing it here would make this one function
    // disagree with every other KIP signature in the crate.
    #[allow(clippy::result_large_err)]
    pub fn into_request(self) -> Result<Request, KipError> {
        let operations = match (self.command, self.operations) {
            (Some(command), None) => vec![Operation::new(command)],
            (None, Some(operations)) => operations.into_iter().map(Operation::from).collect(),
            (Some(_), Some(_)) => {
                return Err(KipError::invalid_request_envelope(
                    "send either a single `command` or an `operations` batch, never both",
                ));
            }
            (None, None) => {
                return Err(KipError::invalid_request_envelope(
                    "send a `command` or an `operations` batch",
                ));
            }
        };

        Ok(Request {
            operations,
            execution: self.execution,
            read: self.read,
            parameters: self.parameters,
            options: self.dry_run.map(|dry_run| RequestOptions {
                dry_run: Some(dry_run),
                ..Default::default()
            }),
            ..Default::default()
        })
    }
}

/// Wraps one already-computed value as a single-operation KIP response.
///
/// The conversation and resource tools answer from AndaDB rather than from the
/// Nexus, but share the KIP response envelope so a caller handles one output
/// shape across the whole memory bundle.
fn kip_page(result: Json, next_cursor: Option<String>) -> Response {
    Response {
        results: vec![OperationResult::ok(result)],
        next_cursor,
        ..Default::default()
    }
}

/// Reads a single-operation response as a plain result or an error.
///
/// A KIP failure lives at the operation level for an ordinary error and at the
/// request level only for an envelope error, so reading just one of them would
/// turn half of the failures into an empty success.
#[allow(clippy::result_large_err)] // see `KipArgs::into_request`
fn kip_result(response: Response) -> Result<Json, KipError> {
    let error = response
        .error
        .or_else(|| response.results.first().and_then(|r| r.error.clone()));
    match error {
        Some(error) => Err(kip_error(error)),
        None => Ok(response
            .results
            .into_iter()
            .next()
            .and_then(|r| r.result)
            .unwrap_or(Json::Null)),
    }
}

/// Reads the Person Concept a caller's principal keys.
const CALLER_BY_KEY: &str =
    r#"FIND(?person) WHERE { ?person CONCEPT {type: "Person", key: :key} } LIMIT 1"#;

/// The first row of a `FIND` result, or `Json::Null` when nothing matched.
///
/// A miss is not an error here: "this Space holds no such Concept yet" is an
/// ordinary answer, and raising it would make an empty memory look like a
/// broken one.
fn first_row(result: Json) -> Json {
    match result {
        Json::Array(rows) => rows.into_iter().next().unwrap_or(Json::Null),
        other => other,
    }
}

/// Recovers a typed [`KipError`] from a wire [`ErrorObject`].
///
/// An unregistered code becomes `InternalError` rather than being dropped: the
/// message still says what happened, and inventing a category from a code this
/// build does not know would misreport whether a retry can help.
fn kip_error(error: ErrorObject) -> KipError {
    let ErrorObject {
        code,
        message,
        hint,
        details,
        ..
    } = error;
    let code = code
        .parse::<KipErrorCode>()
        .unwrap_or(KipErrorCode::InternalError);
    let mut err = KipError::new(code, message);
    if let Some(hint) = hint {
        err = err.with_hint(hint);
    }
    if let Some(details) = details {
        err = err.with_details(details);
    }
    err
}

/// Conversation record stored in the memory database.
///
/// Schema version: 4
#[derive(Debug, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct Conversation {
    /// Unique collection identifier assigned by AndaDB.
    pub _id: u64,

    /// Principal that owns the conversation.
    #[field_type = "Bytes"]
    pub user: Principal,

    /// Optional thread identifier used to group related conversation turns.
    #[field_type = "Option<Bytes>"]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread: Option<Xid>,

    /// Serialized chat messages accumulated for the conversation.
    pub messages: Vec<Json>,

    /// The request resources used by the agent to process the conversation.
    pub resources: Vec<Resource>,

    /// A collection of artifacts generated by the agent during the execution of the task.
    pub artifacts: Vec<Resource>,

    /// Current lifecycle state of the conversation.
    #[field_type = "Text"]
    pub status: ConversationStatus,

    /// Failure reason recorded when the conversation did not complete.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_reason: Option<String>,

    /// The LLM usage statistics for the conversation.
    #[field_type = "Map<String, U64>"]
    pub usage: Usage,

    /// Messages queued to interrupt the agent mid-run.
    ///
    /// They are delivered after the current tool execution and skip remaining
    /// pending tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steering_messages: Option<Vec<String>>,

    /// Follow-up messages queued for the agent's next safe user turn.
    ///
    /// They are delivered with the current pending tool-call results when they
    /// finish, or at the next idle boundary when no tools are pending. Steering
    /// still takes priority.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub follow_up_messages: Option<Vec<String>>,

    /// The child conversation ID, if this conversation has been continued. Should not be updated after set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child: Option<u64>,

    /// The ancestor conversation IDs, ordered from root to parent.
    /// Should not be updated after creation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ancestors: Option<Vec<u64>>,

    /// An optional label for the conversation, which can be used for categorization or retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,

    /// Extra information for future extensions.
    ///
    /// This field is not indexed and should not be used for filtering or
    /// searching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,

    /// The period when the conversation was created, in hours (timestamp / 3600 / 1000).
    /// It is used to index the conversation for faster retrieval by time.
    pub period: u64,

    /// The timestamp when the conversation was created, in milliseconds.
    pub created_at: u64,

    /// The timestamp when the conversation was updated, in milliseconds.
    pub updated_at: u64,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            _id: 0,
            user: Principal::anonymous(),
            thread: None,
            messages: Vec::new(),
            resources: Vec::new(),
            artifacts: Vec::new(),
            status: ConversationStatus::default(),
            failed_reason: None,
            usage: Usage::default(),
            steering_messages: None,
            follow_up_messages: None,
            child: None,
            ancestors: None,
            label: None,
            extra: None,
            period: 0,
            created_at: 0,
            updated_at: 0,
        }
    }
}

impl Conversation {
    /// Appends messages to the serialized chat history.
    pub fn append_messages(&mut self, message: Vec<Message>) {
        self.messages.extend(message.into_iter().map(|v| json!(v)));
    }

    /// Converts mutable conversation fields into AndaDB update values.
    pub fn to_changes(&self) -> Result<BTreeMap<String, Fv>, BoxError> {
        self.to_changes_with_inbound_queues(true)
    }

    /// Builds runner-owned changes without overwriting concurrently queued steering/follow-ups.
    ///
    /// The memory API owns those inbound queue fields, while a running conversation recorder
    /// normally holds a snapshot that never loaded them. Omitting the fields keeps the database's
    /// current queues intact; all other mutable state, including clearing a stale failure reason,
    /// is still persisted.
    pub(crate) fn to_runner_changes(&self) -> Result<BTreeMap<String, Fv>, BoxError> {
        self.to_changes_with_inbound_queues(false)
    }

    fn to_changes_with_inbound_queues(
        &self,
        include_inbound_queues: bool,
    ) -> Result<BTreeMap<String, Fv>, BoxError> {
        let messages = cbor!(self.messages).map_err(|err| format!("encode messages: {err}"))?;
        let resources = cbor!(self.resources).map_err(|err| format!("encode resources: {err}"))?;
        let artifacts = cbor!(self.artifacts).map_err(|err| format!("encode artifacts: {err}"))?;
        let usage = cbor!(self.usage).map_err(|err| format!("encode usage: {err}"))?;
        let mut changes = BTreeMap::from([
            (
                "messages".to_string(),
                Fv::array_from(messages, &[Ft::Json])?,
            ),
            (
                "resources".to_string(),
                Fv::array_from(resources, &[Resource::field_type()])?,
            ),
            (
                "artifacts".to_string(),
                Fv::array_from(artifacts, &[Resource::field_type()])?,
            ),
            ("status".to_string(), Fv::Text(self.status.to_string())),
            (
                "usage".to_string(),
                Fv::map_from(usage, &BTreeMap::from([("*".into(), Ft::U64)]))?,
            ),
            ("updated_at".to_string(), Fv::U64(self.updated_at)),
            (
                "label".to_string(),
                if let Some(label) = self.label.clone() {
                    label.into()
                } else {
                    Fv::Null
                },
            ),
            (
                "extra".to_string(),
                if let Some(extra) = self.extra.clone() {
                    extra.into()
                } else {
                    Fv::Null
                },
            ),
        ]);

        if include_inbound_queues {
            changes.insert(
                "steering_messages".to_string(),
                self.steering_messages.clone().map_or(Fv::Null, Into::into),
            );
            changes.insert(
                "follow_up_messages".to_string(),
                self.follow_up_messages.clone().map_or(Fv::Null, Into::into),
            );
        }

        if let Some(child) = self.child {
            changes.insert("child".to_string(), Fv::U64(child));
        }
        // Clear a stale reason on a successful update; otherwise a conversation that failed
        // once keeps reporting that reason after a later turn succeeds.
        changes.insert(
            "failed_reason".to_string(),
            match &self.failed_reason {
                Some(reason) => Fv::Text(reason.clone()),
                None => Fv::Null,
            },
        );
        Ok(changes)
    }

    /// Builds an incremental delta from borrowed conversation data.
    pub fn to_delta(&self, messages_offset: usize, artifacts_offset: usize) -> ConversationDelta {
        ConversationDelta {
            _id: self._id,
            messages: self
                .messages
                .iter()
                .skip(messages_offset)
                .cloned()
                .collect(),
            artifacts: self
                .artifacts
                .iter()
                .skip(artifacts_offset)
                .cloned()
                .collect(),
            status: self.status.clone(),
            usage: self.usage.clone(),
            failed_reason: self.failed_reason.clone(),
            updated_at: self.updated_at,
            child: self.child,
        }
    }

    /// Builds an incremental delta while consuming the conversation.
    pub fn into_delta(self, messages_offset: usize, artifacts_offset: usize) -> ConversationDelta {
        ConversationDelta {
            _id: self._id,
            messages: self.messages.into_iter().skip(messages_offset).collect(),
            artifacts: self.artifacts.into_iter().skip(artifacts_offset).collect(),
            status: self.status,
            usage: self.usage,
            failed_reason: self.failed_reason,
            updated_at: self.updated_at,
            child: self.child,
        }
    }
}

/// Message view used when embedding conversations for search and recall.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct PrunedMessage {
    /// Message role.
    pub role: String,

    /// Visible or reasoning content retained after pruning.
    pub content: Vec<ContentPart>,

    /// Optional participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Optional sender principal string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Optional RFC 3339 timestamp string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl PrunedMessage {
    /// Prunes non-visible content from a message and converts metadata to strings.
    pub fn try_from(mut msg: Message) -> Option<Self> {
        msg.prune_content();
        Some(Self {
            role: msg.role,
            content: msg.content,
            name: msg.name,
            user: msg.user.map(|u| u.to_string()),
            timestamp: msg.timestamp.and_then(rfc3339_datetime),
        })
    }
}

impl From<Conversation> for Document {
    fn from(conversation: Conversation) -> Self {
        let mut metadata = BTreeMap::from([
            ("_id".to_string(), conversation._id.into()),
            ("type".to_string(), "Conversation".into()),
            ("user".to_string(), conversation.user.to_string().into()),
            ("status".to_string(), conversation.status.to_string().into()),
        ]);
        if let Some(created_at) = rfc3339_datetime(conversation.created_at) {
            metadata.insert("created_at".to_string(), created_at.into());
        }
        if let Some(updated_at) = rfc3339_datetime(conversation.updated_at) {
            metadata.insert("updated_at".to_string(), updated_at.into());
        }
        if let Some(thread) = conversation.thread {
            metadata.insert("thread".to_string(), thread.to_string().into());
        }
        if let Some(label) = conversation.label {
            metadata.insert("label".to_string(), label.into());
        }
        let message: Vec<PrunedMessage> = conversation
            .messages
            .iter()
            .filter_map(|v| {
                serde_json::from_value::<Message>(v.clone())
                    .ok()
                    .and_then(PrunedMessage::try_from)
            })
            .collect();
        Self {
            content: serde_json::to_value(message).unwrap_or_default(),
            metadata,
        }
    }
}

/// Borrowed view of a conversation for insertion into AndaDB.
#[derive(Debug, Serialize)]
pub struct ConversationRef<'a> {
    /// Conversation ID. `0` lets AndaDB assign a new ID.
    pub _id: u64,
    /// Principal that owns the conversation.
    pub user: &'a Principal,
    /// Optional thread identifier.
    pub thread: Option<&'a Xid>,
    /// Serialized message history.
    pub messages: &'a [Json],
    /// Input resources attached to the conversation.
    pub resources: &'a [Resource],
    /// Artifacts produced by the conversation.
    pub artifacts: &'a [Resource],
    /// Current conversation lifecycle status.
    pub status: &'a ConversationStatus,
    /// Accumulated model and tool usage.
    pub usage: &'a Usage,
    /// Pending steering messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steering_messages: &'a Option<Vec<String>>,
    /// Pending follow-up messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub follow_up_messages: &'a Option<Vec<String>>,
    /// Optional conversation label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: &'a Option<String>,
    /// Extra unindexed metadata.
    pub extra: &'a Option<Json>,
    /// Hour bucket used for expiration scans.
    pub period: u64,
    /// Creation timestamp in milliseconds.
    pub created_at: u64,
    /// Last update timestamp in milliseconds.
    pub updated_at: u64,
    /// Child continuation conversation ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child: &'a Option<u64>,
    /// Ancestor conversation IDs from root to parent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ancestors: &'a Option<Vec<u64>>,
}

impl<'a> From<&'a Conversation> for ConversationRef<'a> {
    fn from(conversation: &'a Conversation) -> Self {
        Self {
            _id: conversation._id,
            user: &conversation.user,
            thread: conversation.thread.as_ref(),
            messages: &conversation.messages,
            resources: &conversation.resources,
            artifacts: &conversation.artifacts,
            status: &conversation.status,
            usage: &conversation.usage,
            steering_messages: &conversation.steering_messages,
            follow_up_messages: &conversation.follow_up_messages,
            label: &conversation.label,
            extra: &conversation.extra,
            period: conversation.period,
            created_at: conversation.created_at,
            updated_at: conversation.updated_at,
            child: &conversation.child,
            ancestors: &conversation.ancestors,
        }
    }
}

/// Lightweight conversation state returned by management APIs.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConversationState {
    /// Conversation ID.
    pub _id: u64,
    /// Current lifecycle status.
    pub status: ConversationStatus,
}

impl From<&ConversationRef<'_>> for ConversationState {
    fn from(conversation: &ConversationRef<'_>) -> Self {
        Self {
            _id: conversation._id,
            status: conversation.status.clone(),
        }
    }
}

impl From<&Conversation> for ConversationState {
    fn from(conversation: &Conversation) -> Self {
        Self {
            _id: conversation._id,
            status: conversation.status.clone(),
        }
    }
}

/// A delta of a conversation since a given offset, used for incremental fetching of conversation messages.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConversationDelta {
    /// Conversation ID.
    pub _id: u64,
    /// The new messages since the given offset. The offset is determined by the client and is not stored in the database. It is used to support incremental fetching of conversation messages.
    pub messages: Vec<Json>,
    /// New artifacts since the requested offset.
    pub artifacts: Vec<Resource>,
    /// Current lifecycle status.
    pub status: ConversationStatus,
    /// Accumulated usage at the time the delta was read.
    pub usage: Usage,
    /// Failure reason when the conversation failed.
    pub failed_reason: Option<String>,
    /// Last update timestamp in milliseconds.
    pub updated_at: u64,
    /// Child continuation conversation ID, when present.
    pub child: Option<u64>,
}

/// Conversation lifecycle state.
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConversationStatus {
    /// Conversation has been submitted but not yet picked up by a runner.
    #[default]
    Submitted,
    /// Runner is actively processing the conversation.
    Working,
    /// Runner is idle and can accept follow-up input.
    Idle,
    /// Conversation finished successfully.
    Completed,
    /// Conversation was cancelled.
    Cancelled,
    /// Conversation failed.
    Failed,
}

impl fmt::Display for ConversationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversationStatus::Submitted => write!(f, "submitted"),
            ConversationStatus::Working => write!(f, "working"),
            ConversationStatus::Idle => write!(f, "idle"),
            ConversationStatus::Completed => write!(f, "completed"),
            ConversationStatus::Cancelled => write!(f, "cancelled"),
            ConversationStatus::Failed => write!(f, "failed"),
        }
    }
}

/// Storage wrapper for the conversations collection.
#[derive(Debug, Clone)]
pub struct Conversations {
    /// Underlying AndaDB collection.
    conversations: Arc<Collection>,
}

#[async_trait]
impl crate::subagent::ConversationRecords for Conversations {
    async fn create(&self, conversation: ConversationRef<'_>) -> Result<u64, BoxError> {
        Ok(self.add_conversation(conversation).await?)
    }

    async fn update(&self, conversation: &Conversation) -> Result<(), BoxError> {
        let changes = conversation.to_runner_changes()?;
        self.update_conversation(conversation._id, changes).await?;
        Ok(())
    }
}

impl Conversations {
    /// Opens or creates the conversations collection and indexes.
    pub async fn connect(db: Arc<AndaDB>, name: String) -> Result<Self, BoxError> {
        let mut schema = Conversation::schema()?;
        schema.with_version(4);

        let conversations = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name,
                    description: "conversations collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["user"]).await?;
                    collection.create_btree_index_nx(&["thread"]).await?;
                    collection.create_btree_index_nx(&["period"]).await?;
                    collection
                        .create_bm25_index_nx(&["messages", "resources", "artifacts"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(Self { conversations })
    }

    /// Adds a conversation and flushes the collection.
    pub async fn add_conversation(
        &self,
        conversation: ConversationRef<'_>,
    ) -> Result<u64, DBError> {
        let id = self.conversations.add_from(&conversation).await?;
        self.conversations.flush(unix_ms()).await?;
        Ok(id)
    }

    /// Updates selected conversation fields and flushes the collection.
    pub async fn update_conversation(
        &self,
        id: u64,
        fields: BTreeMap<String, Fv>,
    ) -> Result<(), DBError> {
        self.conversations.update(id, fields).await?;
        self.conversations.flush(unix_ms()).await?;
        Ok(())
    }

    /// Retrieves a conversation by ID.
    pub async fn get_conversation(&self, id: u64) -> Result<Conversation, DBError> {
        self.conversations.get_as(id).await
    }

    /// Deletes a conversation by ID and returns whether it existed.
    pub async fn delete_conversation(&self, id: u64) -> Result<bool, DBError> {
        let doc = self.conversations.remove(id).await?;
        self.conversations.flush(unix_ms()).await?;
        Ok(doc.is_some())
    }

    /// Retrieves the user's conversations matching `ids`.
    ///
    /// The underlying query limit is capped at 1000 IDs per call.
    pub async fn batch_get_conversations(
        &self,
        user: &Principal,
        ids: Vec<u64>,
    ) -> Result<Vec<Conversation>, BoxError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // `limit: None` falls back to the database default (10); request the full batch instead.
        let limit = ids.len();
        let filter = Some(Filter::And(vec![
            Box::new(Filter::Field((
                "_id".to_string(),
                RangeQuery::Include(ids.into_iter().map(Fv::U64).collect()),
            ))),
            Box::new(Filter::Field((
                "user".to_string(),
                RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
            ))),
        ]));

        let rt: Vec<Conversation> = self
            .conversations
            .search_as(Query {
                search: None,
                filter,
                limit: Some(limit),
            })
            .await?;
        Ok(rt)
    }

    /// Lists the user's conversations, newest first, with cursor-based pagination.
    pub async fn list_conversations_by_user(
        &self,
        user: &Principal,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<Conversation>, Option<String>), BoxError> {
        // 0 means "no limit" to the database, and an empty page would panic below; clamp instead.
        let limit = limit.unwrap_or(10).clamp(1, 100);
        let cursor = match BTree::from_cursor::<u64>(&cursor)? {
            Some(cursor) => cursor,
            None => self.conversations.max_document_id() + 1,
        };
        let filter = Filter::And(vec![
            Box::new(Filter::Field((
                "user".to_string(),
                RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
            ))),
            Box::new(Filter::Field((
                "_id".to_string(),
                RangeQuery::Lt(Fv::U64(cursor)),
            ))),
        ]);

        // `query_last_ids` returns the newest page for any filter shape; the IDs
        // come back ascending, so reverse them for newest-first output.
        let mut ids = self
            .conversations
            .query_last_ids(filter, Some(limit))
            .await?;
        ids.reverse();
        let rt = materialize_conversation_ids(&self.conversations, ids).await?;
        // The page holds the newest matching conversations; the next cursor is the smallest ID,
        // so the following page fetches strictly older ones.
        let cursor = if rt.len() >= limit {
            rt.iter()
                .map(|conversation| conversation._id)
                .min()
                .and_then(|id| BTree::to_cursor(&id))
        } else {
            None
        };
        Ok((rt, cursor))
    }

    /// Searches a user's conversations with the BM25 conversation index.
    pub async fn search_conversations(
        &self,
        user: &Principal,
        query: String,
        limit: Option<usize>,
    ) -> Result<Vec<Conversation>, BoxError> {
        let limit = limit.unwrap_or(10).clamp(1, 100);
        let rt = self
            .conversations
            .search_as(Query {
                search: Some(Search {
                    text: Some(query),
                    logical_search: true,
                    ..Default::default()
                }),
                filter: Some(Filter::Field((
                    "user".to_string(),
                    RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
                ))),
                limit: Some(limit),
            })
            .await?;
        Ok(rt)
    }

    /// Deletes all conversations created before `timestamp` (in milliseconds).
    pub async fn delete_expired_conversations(&self, timestamp: u64) -> Result<u64, BoxError> {
        let period = timestamp / 3600 / 1000;
        let mut count = 0u64;
        loop {
            let ids = next_expired_batch(&self.conversations, period).await?;
            if ids.is_empty() {
                break;
            }

            let mut removed = 0u64;
            for id in ids {
                if matches!(self.conversations.remove(id).await, Ok(Some(_))) {
                    removed += 1;
                }
            }
            count += removed;
            if removed == 0 {
                // Nothing was removable; stop instead of spinning on undeletable documents.
                break;
            }
        }

        self.conversations.flush(unix_ms()).await?;
        Ok(count)
    }
}

// Load sequentially so every ID returned by the index is materialized in the same order.
async fn materialize_conversation_ids(
    conversations: &Collection,
    ids: Vec<u64>,
) -> Result<Vec<Conversation>, DBError> {
    let mut results = Vec::with_capacity(ids.len());
    for id in ids {
        match conversations.get_as(id).await {
            Ok(conversation) => results.push(conversation),
            Err(DBError::NotFound { .. }) => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(results)
}

/// The maximum number of expired conversations fetched per deletion batch. The database caps
/// query limits at 1000.
const DELETE_EXPIRED_BATCH: usize = 1000;

/// Returns the next batch of conversation IDs whose `period` is older than `period`.
async fn next_expired_batch(conversations: &Collection, period: u64) -> Result<Vec<u64>, BoxError> {
    let ids = conversations
        .search_ids(Query {
            search: None,
            filter: Some(Filter::Field((
                "period".to_string(),
                RangeQuery::Lt(Fv::U64(period)),
            ))),
            limit: Some(DELETE_EXPIRED_BATCH),
        })
        .await?;
    Ok(ids)
}

/// How the KIP tools bind a call to an identity and a MemorySpace.
///
/// `anda_cognitive_nexus` authorizes a bare `&CognitiveNexus` as the engine's
/// own system Principal — a real authorization, not a bypass, but one that
/// makes *every* caller the owner of everything. That is right for the embedded
/// case, where the process that opened the database is the owner, and wrong the
/// moment one engine serves more than one caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tenancy {
    /// One brain, owned by the process. Every caller runs as the system
    /// Principal against the default Space.
    #[default]
    Shared,

    /// Each verified caller runs as its own Principal, in a MemorySpace it
    /// owns.
    ///
    /// Isolation comes from ownership rather than from Grants the host has to
    /// write and keep correct: a Space's owner is authorized in it, and nobody
    /// is authorized in a Space they do not own. A caller therefore reaches
    /// its own memory and no one else's, with no policy to get wrong.
    ///
    /// The Principal and Space are created on the caller's first KIP call. The
    /// agent's own knowledge stays in the default Space, which the host-level
    /// helpers ([`MemoryManagement::query`], [`MemoryManagement::execute`],
    /// [`MemoryManagement::describe_primer`]) still reach as the system
    /// Principal.
    PerCallerSpace,
}

/// The Governance Principal a caller authenticates as.
///
/// Namespaced like the engine's own `kip:principal:system` so a caller's id
/// cannot collide with a Principal the host registered by another route.
fn tenant_principal(caller: &Principal) -> String {
    format!("kip:principal:caller:{caller}")
}

/// The MemorySpace a caller owns under [`Tenancy::PerCallerSpace`].
fn tenant_space(caller: &Principal) -> String {
    format!("kip:space:caller:{caller}")
}

/// High-level memory manager for conversations, resources, and the Cognitive Nexus.
#[derive(Debug, Clone)]
pub struct MemoryManagement {
    /// Shared Cognitive Nexus used for KIP execution.
    nexus: Arc<CognitiveNexus>,
    /// Conversation collection.
    conversations: Arc<Collection>,
    /// Resource collection.
    resources: Arc<Collection>,
    /// Function definition exposed for the writable KIP tool.
    kip_function_definitions: FunctionDefinition,
    /// How a caller's KIP calls are bound to identity and Space.
    tenancy: Tenancy,
    /// Callers this process has already provisioned, so the common case does
    /// not re-read the Principal, Space and Schema Environment on every call.
    /// Bounded, because the miss is only a few idempotent lookups.
    provisioned: Cache<Principal, ()>,
}

// Preserve KIP's native structured error across the public host/tenant API.
#[allow(clippy::result_large_err)]
impl MemoryManagement {
    /// Opens or creates all memory collections and connects them to a nexus.
    pub async fn connect(db: Arc<AndaDB>, nexus: Arc<CognitiveNexus>) -> Result<Self, BoxError> {
        let conversations = Conversations::connect(db.clone(), "conversations".to_string())
            .await?
            .conversations;

        let schema = Resource::schema()?;
        let resources = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "resources".to_string(),
                    description: "Resources collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["tags"]).await?;
                    collection.create_btree_index_nx(&["hash"]).await?;
                    collection.create_btree_index_nx(&["mime_type"]).await?;
                    collection
                        .create_bm25_index_nx(&["name", "description", "metadata"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(Self {
            nexus,
            conversations,
            resources,
            kip_function_definitions: FUNCTION_DEFINITION.clone(),
            tenancy: Tenancy::default(),
            provisioned: Cache::builder().max_capacity(10_000).build(),
        })
    }

    /// Chooses how callers are bound to identity and Space.
    ///
    /// Defaults to [`Tenancy::Shared`], which is the behavior of an engine that
    /// never called this.
    pub fn with_tenancy(mut self, tenancy: Tenancy) -> Self {
        self.tenancy = tenancy;
        self
    }

    /// How this engine binds callers.
    pub fn tenancy(&self) -> Tenancy {
        self.tenancy
    }

    /// Overrides the writable KIP tool definition.
    pub fn with_kip_function_definitions(mut self, def: FunctionDefinition) -> Self {
        self.kip_function_definitions = def;
        self
    }

    /// Returns the shared Cognitive Nexus handle.
    pub fn nexus(&self) -> Arc<CognitiveNexus> {
        self.nexus.clone()
    }

    /// Views the conversations collection through the shared [`Conversations`] API.
    fn as_conversations(&self) -> Conversations {
        Conversations {
            conversations: self.conversations.clone(),
        }
    }

    /// Returns the largest conversation document ID currently known.
    pub fn max_conversation_id(&self) -> u64 {
        self.conversations.max_document_id()
    }

    /// Binds one request to the session and Space a caller runs in, provisioning
    /// them if needed.
    ///
    /// The caller binding replaces any Space already present on `request`, so a
    /// tenant cannot select another caller's Space through a raw envelope.
    ///
    /// Under [`Tenancy::Shared`] this is the engine's own system session, which
    /// is what a bare `&CognitiveNexus` would have authorized as anyway.
    ///
    /// Under [`Tenancy::PerCallerSpace`] a verified caller gets a Principal and
    /// a Space of its own, created on first use. An *anonymous* caller does not:
    /// [`StateFeatures::caller`] returns the anonymous Principal precisely when
    /// nothing was verified, and giving every unverified request one shared
    /// Space would pool strangers' memories into it. It runs as the Nexus's
    /// anonymous Principal on the default Space instead, where default deny
    /// gives it nothing until a Space policy says otherwise.
    pub async fn session_for(
        &self,
        caller: &Principal,
        request: &mut Request,
    ) -> Result<Session, KipError> {
        let (session, space) = self.binding(caller).await?;
        request.space = space;
        Ok(session)
    }

    /// The identity a caller runs as, and the Space it runs in.
    ///
    /// One decision returning both, because they are only correct together: a
    /// caller authenticated as itself but pointed at another caller's Space is
    /// precisely the failure [`Tenancy::PerCallerSpace`] exists to prevent, and
    /// deciding them apart is how the two drift.
    async fn binding(
        &self,
        caller: &Principal,
    ) -> Result<(Session, Option<SpaceSelector>), KipError> {
        match self.tenancy {
            Tenancy::Shared => Ok((self.nexus.system_session(), None)),
            Tenancy::PerCallerSpace if caller == &Principal::anonymous() => {
                Ok((self.nexus.session(AuthContext::anonymous()), None))
            }
            Tenancy::PerCallerSpace => {
                self.provision(caller).await?;
                let session = self.nexus.session(
                    AuthContext {
                        auth_method: "ic-auth".to_string(),
                        ..AuthContext::principal(tenant_principal(caller))
                    }
                    .with_client("anda_engine"),
                );
                let space = SpaceSelector {
                    id: Some(tenant_space(caller)),
                    uri: None,
                };
                Ok((session, Some(space)))
            }
        }
    }

    /// Creates the Principal and MemorySpace a caller owns, once per process.
    ///
    /// Every step is idempotent, so a cache miss after a restart re-runs them
    /// harmlessly. The Space inherits whatever Schema Environment is in force in
    /// the default Space: a fresh Space would otherwise hold Core alone, and a
    /// host that installed its own vocabulary would find the same command
    /// working for the owner and failing for every tenant.
    async fn provision(&self, caller: &Principal) -> Result<(), KipError> {
        self.provisioned
            .try_get_with_by_ref(caller, async {
                let principal_id = tenant_principal(caller);
                let space_id = tenant_space(caller);
                self.nexus
                    .governance()
                    .ensure_principal(PrincipalDraft {
                        principal_id: principal_id.clone(),
                        // Recorded, never read by an authorization decision. A caller
                        // here is an ic-auth Principal, which may stand for a person or
                        // for another agent; the deployment is what knows which, and can
                        // correct the record through the Governance API.
                        principal_class: principal_class::HUMAN.to_string(),
                        display_name: caller.to_string(),
                        auth_provider: "ic-auth".to_string(),
                        auth_subject: caller.to_string(),
                    })
                    .await?;
                self.nexus
                    .store
                    .open_or_create_space(SpaceDraft {
                        space_id: space_id.clone(),
                        name: format!("Memory of {caller}"),
                        description: "A caller's own MemorySpace.".to_string(),
                        owner_principal: principal_id,
                        ..Default::default()
                    })
                    .await?;
                let environment = self.nexus.store.schema_environment(DEFAULT_SPACE).await?;
                self.nexus
                    .ensure_schema(&space_id, environment.lock)
                    .await?;
                Ok::<(), KipError>(())
            })
            .await
            .map_err(|err| (*err).clone())
    }

    /// Runs one read-only KIP command as the engine itself.
    ///
    /// Read-only is enforced on what the command parses to, so a mutation sent
    /// through here is refused before the engine sees it.
    ///
    /// This is the *host* acting, on the default Space, under every tenancy —
    /// use [`Self::query_as`] to read a caller's own memory.
    pub async fn query(
        &self,
        command: &str,
        parameters: Option<Map<String, Json>>,
    ) -> Result<Json, KipError> {
        let request = Request {
            parameters,
            ..Request::single(command)
        };
        kip_result(execute_request_readonly(&self.nexus.system_session(), &request).await)
    }

    /// Runs one state-changing KIP command as the engine itself.
    ///
    /// The host counterpart of [`Self::execute_as`]; see [`Self::query`].
    pub async fn execute(
        &self,
        command: &str,
        parameters: Option<Map<String, Json>>,
    ) -> Result<Json, KipError> {
        let request = Request {
            parameters,
            ..Request::single(command)
        };
        kip_result(execute_request(&self.nexus.system_session(), &request).await)
    }

    /// Runs one read-only KIP command in a caller's own memory.
    pub async fn query_as(
        &self,
        caller: &Principal,
        command: &str,
        parameters: Option<Map<String, Json>>,
    ) -> Result<Json, KipError> {
        let mut request = Request {
            parameters,
            ..Request::single(command)
        };
        let session = self.session_for(caller, &mut request).await?;
        kip_result(execute_request_readonly(&session, &request).await)
    }

    /// Runs one state-changing KIP command in a caller's own memory.
    pub async fn execute_as(
        &self,
        caller: &Principal,
        command: &str,
        parameters: Option<Map<String, Json>>,
    ) -> Result<Json, KipError> {
        let mut request = Request {
            parameters,
            ..Request::single(command)
        };
        let session = self.session_for(caller, &mut request).await?;
        kip_result(execute_request(&session, &request).await)
    }

    /// Describes the Cognitive Nexus primer.
    pub async fn describe_primer(&self) -> Result<Json, KipError> {
        self.query("DESCRIBE PRIMER", None).await
    }

    /// Describes the agent's own model of itself, or `null` when it holds none.
    ///
    /// KIP 1.x answered this from a `$system` Person node carrying the engine's
    /// principal. KIP 2.0 forbids that shape — `Person != PrincipalRecord`, and
    /// an identity written into cognitive content to stand in for a Principal is
    /// exactly the confusion the profile rules out — so self-knowledge lives in
    /// a `SelfModel` Concept, which is descriptive cognition and grants nothing.
    pub async fn describe_self(&self) -> Result<Json, KipError> {
        let found = self
            .query(
                r#"FIND(?self) WHERE { ?self {type: "SelfModel"} } LIMIT 1"#,
                None,
            )
            .await?;
        Ok(first_row(found))
    }

    /// Describes the caller identity stored in the nexus, or `null` when absent.
    ///
    /// The caller's principal is the Concept's `key`: a semantic handle for the
    /// person the agent is talking to, never a claim that this Concept can
    /// authenticate as that Principal.
    pub async fn describe_caller(&self, id: &Principal) -> Result<Json, KipError> {
        self.query_as(
            id,
            CALLER_BY_KEY,
            Some(Map::from_iter([("key".to_string(), id.to_string().into())])),
        )
        .await
        .map(first_row)
    }

    /// Gets or initializes the caller identity concept in the nexus.
    ///
    /// `UPSERT ... MATCH {type, key}` is the identity-bearing form: matching on
    /// the principal creates the Person once and resolves to the same one on
    /// every later turn, where a name-only match would mint a second Person for
    /// anyone who renamed themselves.
    pub async fn get_or_init_caller(
        &self,
        id: &Principal,
        name: Option<String>,
    ) -> Result<Json, KipError> {
        let key = id.to_string();
        if name.is_none() {
            let existing = self.describe_caller(id).await?;
            if !existing.is_null() {
                return Ok(existing);
            }
        }
        let name = name.unwrap_or_else(|| key.clone());
        let parameters = Map::from_iter([
            ("key".to_string(), key.clone().into()),
            ("name".to_string(), name.into()),
        ]);
        self.execute_as(
            id,
            r#"UPSERT CONCEPT ?person {
                 MATCH { type: "Person", key: :key }
                 SET FIELDS { name: :name }
               }"#,
            Some(parameters),
        )
        .await?;

        // Read back rather than returning the write receipt: callers want the
        // Person as it now stands, which on a match is not what this call sent.
        self.query_as(
            id,
            CALLER_BY_KEY,
            Some(Map::from_iter([("key".to_string(), key.into())])),
        )
        .await
        .map(first_row)
    }

    /// Adds one resource reference and flushes the resource collection.
    pub async fn add_resource(&self, resource: ResourceRef<'_>) -> Result<u64, DBError> {
        let id = self.resources.add_from(&resource).await?;
        self.resources.flush(unix_ms()).await?;
        Ok(id)
    }

    /// Adds resources when needed and returns resource references without blobs.
    pub async fn try_add_resources(
        &self,
        resources: &[Resource],
    ) -> Result<Vec<Resource>, BoxError> {
        let mut rs: Vec<Resource> = Vec::with_capacity(resources.len());
        let mut count = 0;
        for r in resources.iter() {
            let rf: ResourceRef = r.into();
            let id = if r._id > 0 {
                // Stored resources carry no owner, so only existence can be verified here;
                // rejecting unknown IDs keeps conversations free of dangling references.
                if !self.resources.contains(r._id) {
                    return Err(format!("resource {} does not exist", r._id).into());
                }
                r._id
            } else {
                match self.resources.add_from(&rf).await {
                    Ok(id) => {
                        count += 1;
                        id
                    }
                    Err(DBError::AlreadyExists { _id, .. }) => _id,
                    Err(err) => Err(err)?,
                }
            };

            let r2 = Resource {
                _id: id,
                blob: None,
                ..r.clone()
            };
            rs.push(r2)
        }

        if count > 0 {
            self.resources.flush(unix_ms()).await?;
        }

        Ok(rs)
    }

    /// Retrieves a resource by ID.
    ///
    /// Stored resources carry no owner, so this performs **no** access control. Every
    /// caller-facing path must go through [`MemoryManagement::get_resource_for`] instead.
    pub async fn get_resource(&self, id: u64) -> Result<Resource, DBError> {
        self.resources.get_as(id).await
    }

    /// Retrieves a resource on behalf of `caller`, enforcing ownership.
    ///
    /// Resources live in a single global collection with dense sequential IDs and no owner
    /// field, so ownership is established indirectly: the caller names a conversation, that
    /// conversation must belong to them, and the resource must belong to that conversation.
    /// Without the second half, any caller could read an arbitrary resource by pairing its ID
    /// with a conversation they own.
    pub async fn get_resource_for(
        &self,
        caller: &Principal,
        conversation: u64,
        id: u64,
    ) -> Result<Resource, BoxError> {
        let conversation = self.get_conversation(conversation).await?;
        if &conversation.user != caller {
            return Err("permission denied".into());
        }

        if !conversation
            .resources
            .iter()
            .chain(conversation.artifacts.iter())
            .any(|resource| resource._id == id)
        {
            return Err(format!(
                "permission denied: resource {id} does not belong to conversation {}",
                conversation._id
            )
            .into());
        }

        Ok(self.get_resource(id).await?)
    }

    /// Adds a conversation through the shared conversations API.
    pub async fn add_conversation(
        &self,
        conversation: ConversationRef<'_>,
    ) -> Result<u64, DBError> {
        self.as_conversations().add_conversation(conversation).await
    }

    /// Updates selected conversation fields through the shared conversations API.
    pub async fn update_conversation(
        &self,
        id: u64,
        fields: BTreeMap<String, Fv>,
    ) -> Result<(), DBError> {
        self.as_conversations()
            .update_conversation(id, fields)
            .await
    }

    /// Retrieves a conversation by ID.
    pub async fn get_conversation(&self, id: u64) -> Result<Conversation, DBError> {
        self.as_conversations().get_conversation(id).await
    }

    /// Deletes a conversation by ID.
    pub async fn delete_conversation(&self, id: u64) -> Result<bool, DBError> {
        self.as_conversations().delete_conversation(id).await
    }

    /// Lists a user's conversations with cursor-based pagination.
    pub async fn list_conversations_by_user(
        &self,
        user: &Principal,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<Conversation>, Option<String>), BoxError> {
        self.as_conversations()
            .list_conversations_by_user(user, cursor, limit)
            .await
    }

    /// Searches a user's conversations by text query.
    pub async fn search_conversations(
        &self,
        user: &Principal,
        query: String,
        limit: Option<usize>,
    ) -> Result<Vec<Conversation>, BoxError> {
        self.as_conversations()
            .search_conversations(user, query, limit)
            .await
    }

    /// Deletes all conversations created before `timestamp` (in milliseconds).
    ///
    /// Referenced resources are intentionally **not** deleted here. Resources are
    /// content-deduplicated (see [`MemoryManagement::try_add_resources`]): a single
    /// resource `_id` can be shared by several conversations, and resources carry
    /// no owner or reference count. Deleting a resource when one referencing
    /// conversation expires would break every other (possibly still-active)
    /// conversation that shares it, and there is no reverse index to check for
    /// remaining references. Reclaiming orphaned resources therefore requires a
    /// dedicated reference-counted GC pass and is left to callers.
    pub async fn delete_expired_conversations(&self, timestamp: u64) -> Result<u64, BoxError> {
        self.as_conversations()
            .delete_expired_conversations(timestamp)
            .await
    }
}

/// KIP tool for memory management
impl Tool<BaseCtx> for MemoryManagement {
    type Args = KipArgs;
    type Output = Response;

    fn name(&self) -> String {
        self.kip_function_definitions.name.clone()
    }

    fn description(&self) -> String {
        self.kip_function_definitions.description.clone()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(memory_tool_group_info())
    }

    fn definition(&self) -> FunctionDefinition {
        self.kip_function_definitions.clone()
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        request: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let caller = *ctx.caller();
        hooked_call(&ctx, request, |request| async move {
            let res = match request.into_request() {
                Ok(mut request) => match self.session_for(&caller, &mut request).await {
                    Ok(session) => execute_request(&session, &request).await,
                    Err(err) => Response::from(err),
                },
                Err(err) => Response::from(err),
            };
            Ok(kip_output(res))
        })
        .await
    }
}

/// Wraps a KIP response as a tool output.
///
/// Anything short of `succeeded` is flagged as an error, `partial` included: a
/// batch where one operation failed is not a clean result, and the per-operation
/// detail the model needs to tell which is already in the payload.
fn kip_output(res: Response) -> ToolOutput<Response> {
    ToolOutput {
        is_error: if res.status == TopLevelStatus::Succeeded {
            None
        } else {
            Some(true)
        },
        output: res,
        artifacts: Vec::new(),
        usage: Usage::default(),
        tools_usage: HashMap::new(),
    }
}

/// A read-only version of the KIP tool for memory management, which does not allow any modifications to the memory and is safe to use for retrieval operations.
#[derive(Debug, Clone)]
pub struct MemoryReadonly {
    memory: Arc<MemoryManagement>,
    kip_function_definitions: FunctionDefinition,
}

impl MemoryReadonly {
    /// Function name used when registering the read-only KIP tool.
    pub const NAME: &'static str = "execute_kip_readonly";

    /// Creates a new MemoryReadonly instance
    pub fn new(memory: Arc<MemoryManagement>) -> Self {
        Self {
            memory,
            kip_function_definitions: READONLY_FUNCTION_DEFINITION.clone(),
        }
    }

    /// Overrides the read-only KIP tool definition.
    pub fn with_kip_function_definitions(mut self, def: FunctionDefinition) -> Self {
        self.kip_function_definitions = def;
        self
    }
}

impl Tool<BaseCtx> for MemoryReadonly {
    type Args = KipArgs;
    type Output = Response;

    fn name(&self) -> String {
        self.kip_function_definitions.name.clone()
    }

    fn description(&self) -> String {
        self.kip_function_definitions.description.clone()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(memory_tool_group_info())
    }

    fn definition(&self) -> FunctionDefinition {
        self.kip_function_definitions.clone()
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        request: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        // Note: this tool shares `DynToolHook<KipArgs, Response>` with
        // `execute_kip` — the hook state is keyed by argument/output types. A
        // hook that must tell them apart can compare `BaseCtx::path()`, whose
        // last segment is `t_execute_kip_readonly` here and `t_execute_kip`
        // there.
        let caller = *ctx.caller();
        hooked_call(&ctx, request, |request| async move {
            let res = match request.into_readonly_request() {
                Ok(mut request) => match self.memory.session_for(&caller, &mut request).await {
                    Ok(session) => execute_request_readonly(&session, &request).await,
                    Err(err) => Response::from(err),
                },
                Err(err) => Response::from(err),
            };
            Ok(kip_output(res))
        })
        .await
    }
}

/// Arguments for "get_resource_content" tool
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct GetResourceContentArgs {
    /// The ID of the resource to get
    pub _id: u64,
    /// The ID of the conversation the resource belongs to
    pub conversation: u64,
}

/// Tool that retrieves the full content for a stored resource.
#[derive(Debug, Clone)]
pub struct GetResourceContentTool {
    memory: Arc<MemoryManagement>,
}

impl GetResourceContentTool {
    /// Function name used when registering the resource-content tool.
    pub const NAME: &'static str = "get_resource_content";

    /// Creates a new GetResourceContentTool instance
    pub fn new(memory: Arc<MemoryManagement>) -> Self {
        Self { memory }
    }
}

impl Tool<BaseCtx> for GetResourceContentTool {
    type Args = GetResourceContentArgs;
    type Output = Response;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Retrieves the full content of a stored resource by its ID, within a conversation you own that references it. Returns the content as plain text if UTF-8 encoded, or as a base64url-encoded string for binary data. If the resource has no local blob but has a URI, it will be fetched from the remote source.".to_string()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(memory_tool_group_info())
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
            let res = self
                .memory
                .get_resource_for(ctx.caller(), args.conversation, args._id)
                .await?;
            let text = match res.blob {
                Some(blob) => match String::from_utf8(blob.0) {
                    Ok(s) => s,
                    Err(e) => ByteBufB64(e.into_bytes()).to_string(),
                },
                None => match res.uri {
                    Some(uri) => FetchWebResourcesTool::fetch_as_text(ctx, &uri).await?,
                    None => Err(format!("Invalid resource {}, no blob or uri", args._id))?,
                },
            };

            Ok(ToolOutput::new(Response::ok(text.into())))
        })
        .await
    }
}

/// Arguments for "list_previous_conversations" tool
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct ListConversationsArgs {
    /// The cursor for pagination, returned from the previous call. Use an empty string for the first page.
    #[serde(default)]
    pub cursor: String,
    /// The maximum number of conversations to return, clamped to 1-100 (default: 0, which applies the built-in default of 10)
    #[serde(default)]
    pub limit: usize,
}

/// Tool that lists previous conversations for the current caller.
#[derive(Debug, Clone)]
pub struct ListConversationsTool {
    conversations: Conversations,
    description: String,
}

impl ListConversationsTool {
    /// Function name used when registering the conversation-list tool.
    pub const NAME: &'static str = "list_previous_conversations";

    /// Creates a new ListConversationsTool instance
    pub fn new(conversations: Conversations) -> Self {
        Self { conversations, description: "Lists the current user's previous conversations in reverse chronological order with cursor-based pagination. Returns conversation metadata including status, timestamps, messages, and usage statistics. Use the cursor parameter to paginate through older conversations.".to_string() }
    }

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

impl Tool<BaseCtx> for ListConversationsTool {
    type Args = ListConversationsArgs;
    type Output = Response;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(memory_tool_group_info())
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
            let (conversations, next_cursor) = self
                .conversations
                .list_conversations_by_user(
                    ctx.caller(),
                    if args.cursor.is_empty() {
                        None
                    } else {
                        Some(args.cursor)
                    },
                    if args.limit == 0 {
                        None
                    } else {
                        Some(args.limit)
                    },
                )
                .await?;
            let docs: Vec<Document> = conversations.into_iter().map(Document::from).collect();
            Ok(ToolOutput::new(kip_page(
                Documents::from(docs).to_string().into(),
                next_cursor,
            )))
        })
        .await
    }
}

/// Arguments for "search_conversations" tool
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct SearchConversationsArgs {
    /// The query string to search for in the conversation history.
    pub query: String,
    /// The maximum number of conversations to return, clamped to 1-100 (default: 0, which applies the built-in default of 10)
    #[serde(default)]
    pub limit: usize,
}

/// Tool that searches previous conversations for the current caller.
#[derive(Debug, Clone)]
pub struct SearchConversationsTool {
    conversations: Conversations,
    description: String,
}

impl SearchConversationsTool {
    /// Function name used when registering the conversation-search tool.
    pub const NAME: &'static str = "search_conversations";

    /// Creates a new SearchConversationsTool instance
    pub fn new(conversations: Conversations) -> Self {
        Self { conversations, description: "Performs a full-text search across the current user's conversation history using a query string. Searches through messages, resources, and artifacts to find relevant past conversations. Use this to recall specific topics, instructions, or context from previous interactions.".to_string() }
    }

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

impl Tool<BaseCtx> for SearchConversationsTool {
    type Args = SearchConversationsArgs;
    type Output = Response;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(memory_tool_group_info())
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
            let conversations = self
                .conversations
                .search_conversations(
                    ctx.caller(),
                    args.query,
                    if args.limit == 0 {
                        None
                    } else {
                        Some(args.limit)
                    },
                )
                .await?;

            let docs: Vec<Document> = conversations.into_iter().map(Document::from).collect();
            Ok(ToolOutput::new(kip_page(
                Documents::from(docs).to_string().into(),
                None,
            )))
        })
        .await
    }
}

/// Arguments for "memory_api" tool
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum MemoryToolArgs {
    /// Get one resource from a conversation owned by the caller.
    GetResource {
        /// The ID of the resource to get
        _id: u64,
        /// The ID of the conversation where the resource is located
        conversation: u64,
    },
    /// Get a conversation by ID
    GetConversation {
        /// The ID of the conversation to get
        _id: u64,
    },
    /// Get a conversation delta by message and artifact offsets.
    GetConversationDelta {
        /// The ID of the conversation to get
        _id: u64,
        /// The messages offset for the conversation delta
        messages_offset: usize,
        /// The artifacts offset for the conversation delta
        artifacts_offset: usize,
    },
    /// Stop an in-progress conversation.
    StopConversation {
        /// The ID of the conversation to stop
        _id: u64,
    },
    /// Interrupt a conversation with a steering message.
    SteerConversation {
        /// The ID of the conversation to steer
        _id: u64,
        /// The steering message to interrupt the agent mid-run, delivered after current tool execution, skips remaining tools.
        message: String,
    },
    /// Queue a follow-up message for the next safe turn.
    FollowUpConversation {
        /// The ID of the conversation to follow up
        _id: u64,
        /// The follow-up message to queue for the agent's next safe user turn.
        message: String,
    },
    /// Delete a conversation owned by the caller.
    DeleteConversation {
        /// The ID of the conversation to delete
        _id: u64,
    },
    /// List previous conversations
    ListPrevConversations {
        /// The cursor for pagination
        cursor: Option<String>,
        /// The limit for pagination, default to 10
        limit: Option<usize>,
    },
    /// Search conversations
    SearchConversations {
        /// The query string to search
        query: String,
        /// The max number of conversations to return, default to 10
        limit: Option<usize>,
    },
}

/// A tool for conversation API
#[derive(Debug, Clone)]
pub struct MemoryTool {
    memory: Arc<MemoryManagement>,
}

impl MemoryTool {
    /// Function name used when registering the unified memory API tool.
    pub const NAME: &'static str = "memory_api";

    /// Creates a new SearchConversationsTool instance
    pub fn new(memory: Arc<MemoryManagement>) -> Self {
        Self { memory }
    }
}

/// Hand-flattened schema for [`MemoryToolArgs`].
///
/// The argument type is an internally tagged enum, which `schemars` renders as
/// an `anyOf` of per-variant schemas — a shape strict function-calling
/// providers reject. This flattens every variant's fields into one closed
/// object with nullable fields plus the `type` discriminator; serde's tagged
/// deserialization accepts it unchanged. Keep the two in sync by hand.
fn memory_tool_schema() -> Json {
    json!({
        "type": "object",
        "description": "Select one memory API action with type, then provide the fields used by that action. Fields not used by the selected type should be null.",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "GetResource",
                    "GetConversation",
                    "GetConversationDelta",
                    "StopConversation",
                    "SteerConversation",
                    "FollowUpConversation",
                    "DeleteConversation",
                    "ListPrevConversations",
                    "SearchConversations"
                ],
                "description": "Memory API action to perform."
            },
            "_id": {
                "type": ["integer", "null"],
                "description": "Resource or conversation ID. Required for actions that target a single resource or conversation."
            },
            "conversation": {
                "type": ["integer", "null"],
                "description": "Conversation ID containing the resource. Required for GetResource."
            },
            "messages_offset": {
                "type": ["integer", "null"],
                "description": "Messages offset for GetConversationDelta. Use 0 for the first delta read."
            },
            "artifacts_offset": {
                "type": ["integer", "null"],
                "description": "Artifacts offset for GetConversationDelta. Use 0 for the first delta read."
            },
            "message": {
                "type": ["string", "null"],
                "description": "Message used by SteerConversation or FollowUpConversation."
            },
            "cursor": {
                "type": ["string", "null"],
                "description": "Pagination cursor for ListPrevConversations. Use null for the first page."
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum results for listing or searching conversations. Use null for the default."
            },
            "query": {
                "type": ["string", "null"],
                "description": "Search query for SearchConversations."
            }
        },
        "required": [
            "type",
            "_id",
            "conversation",
            "messages_offset",
            "artifacts_offset",
            "message",
            "cursor",
            "limit",
            "query"
        ],
        "additionalProperties": false
    })
}

impl Tool<BaseCtx> for MemoryTool {
    type Args = MemoryToolArgs;
    type Output = Response;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "A unified API for managing conversations and memory. Supports retrieving resources and conversation details, stopping or steering in-progress conversations, sending follow-up messages, deleting conversations, listing previous conversations with pagination, searching conversation history by keyword, and listing KIP command logs.".to_string()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(memory_tool_group_info())
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: memory_tool_schema(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let ctx = &ctx;
        hooked_call(ctx, args, |args| async move {
            match args {
                MemoryToolArgs::GetResource { _id, conversation } => {
                    let mut res = self
                        .memory
                        .get_resource_for(ctx.caller(), conversation, _id)
                        .await?;
                    if res.blob.is_none()
                        && let Some(uri) = &res.uri
                    {
                        res.blob = FetchWebResourcesTool::fetch_as_bytes(ctx, uri).await.ok();
                    }

                    Ok(ToolOutput::new(kip_page(json!(res), None)))
                }
                MemoryToolArgs::GetConversation { _id } => {
                    let conversation = self.memory.get_conversation(_id).await?;
                    if &conversation.user != ctx.caller() {
                        return Err("permission denied".into());
                    }

                    Ok(ToolOutput::new(kip_page(json!(conversation), None)))
                }
                MemoryToolArgs::GetConversationDelta {
                    _id,
                    messages_offset,
                    artifacts_offset,
                } => {
                    let conversation = self.memory.get_conversation(_id).await?;
                    if &conversation.user != ctx.caller() {
                        return Err("permission denied".into());
                    }

                    Ok(ToolOutput::new(kip_page(
                        json!(conversation.into_delta(messages_offset, artifacts_offset)),
                        None,
                    )))
                }
                MemoryToolArgs::StopConversation { _id } => {
                    let mut conversation = self.memory.get_conversation(_id).await?;
                    if &conversation.user != ctx.caller() {
                        return Err("permission denied".into());
                    }

                    if conversation.status == ConversationStatus::Working
                        || conversation.status == ConversationStatus::Submitted
                    {
                        conversation.status = ConversationStatus::Cancelled;
                        conversation.updated_at = unix_ms();
                        let changes = BTreeMap::from([
                            (
                                "status".to_string(),
                                Fv::Text(conversation.status.to_string()),
                            ),
                            ("updated_at".to_string(), Fv::U64(conversation.updated_at)),
                        ]);
                        self.memory.update_conversation(_id, changes).await?;
                    }

                    Ok(ToolOutput::new(kip_page(json!(conversation), None)))
                }
                MemoryToolArgs::SteerConversation { _id, message } => {
                    if message.trim().is_empty() {
                        return Err("steering message cannot be empty".into());
                    }

                    let mut conversation = self.memory.get_conversation(_id).await?;
                    if &conversation.user != ctx.caller() {
                        return Err("permission denied".into());
                    }

                    let steering_messages =
                        if let Some(msg) = conversation.steering_messages.clone() {
                            let mut msgs = msg;
                            msgs.push(message.clone());
                            msgs
                        } else {
                            vec![message.clone()]
                        };
                    conversation.steering_messages = Some(steering_messages.clone());
                    conversation.updated_at = unix_ms();
                    let changes = BTreeMap::from([
                        ("steering_messages".to_string(), steering_messages.into()),
                        ("updated_at".to_string(), Fv::U64(conversation.updated_at)),
                    ]);
                    self.memory.update_conversation(_id, changes).await?;

                    Ok(ToolOutput::new(kip_page(json!(conversation), None)))
                }
                MemoryToolArgs::FollowUpConversation { _id, message } => {
                    if message.trim().is_empty() {
                        return Err("follow-up message cannot be empty".into());
                    }

                    let mut conversation = self.memory.get_conversation(_id).await?;
                    if &conversation.user != ctx.caller() {
                        return Err("permission denied".into());
                    }

                    let follow_up_messages =
                        if let Some(msg) = conversation.follow_up_messages.clone() {
                            let mut msgs = msg;
                            msgs.push(message.clone());
                            msgs
                        } else {
                            vec![message.clone()]
                        };
                    conversation.follow_up_messages = Some(follow_up_messages.clone());
                    conversation.updated_at = unix_ms();
                    let changes = BTreeMap::from([
                        ("follow_up_messages".to_string(), follow_up_messages.into()),
                        ("updated_at".to_string(), Fv::U64(conversation.updated_at)),
                    ]);
                    self.memory.update_conversation(_id, changes).await?;

                    Ok(ToolOutput::new(kip_page(json!(conversation), None)))
                }
                MemoryToolArgs::DeleteConversation { _id } => {
                    let conversation = self.memory.get_conversation(_id).await?;
                    if &conversation.user != ctx.caller() {
                        return Err("permission denied".into());
                    }

                    let deleted = self.memory.delete_conversation(_id).await?;
                    Ok(ToolOutput::new(kip_page(
                        json!({ "deleted": deleted }),
                        None,
                    )))
                }
                MemoryToolArgs::ListPrevConversations { cursor, limit } => {
                    // Models often send "" instead of null for the first page.
                    let cursor = cursor.filter(|cursor| !cursor.is_empty());
                    let (conversations, next_cursor) = self
                        .memory
                        .list_conversations_by_user(ctx.caller(), cursor, limit)
                        .await?;

                    Ok(ToolOutput::new(kip_page(json!(conversations), next_cursor)))
                }
                MemoryToolArgs::SearchConversations { query, limit } => {
                    let conversations = self
                        .memory
                        .search_conversations(ctx.caller(), query, limit)
                        .await?;

                    Ok(ToolOutput::new(kip_page(json!(conversations), None)))
                }
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::{RemoteEngines, Web3SDK},
        store::{InMemory, Store},
    };
    use anda_core::CancellationToken;
    use anda_db::database::DBConfig;
    use std::collections::BTreeSet;

    fn principal(seed: u8) -> Principal {
        Principal::self_authenticating([seed; 32])
    }

    fn message(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentPart::Text {
                text: text.to_string(),
            }],
            user: Some(principal(7)),
            timestamp: Some(1_700_000_000_000),
            ..Default::default()
        }
    }

    fn resource(name: &str, blob: Option<&[u8]>) -> Resource {
        Resource {
            name: name.to_string(),
            tags: vec!["text".to_string()],
            description: Some(format!("{name} description")),
            mime_type: Some("text/plain".to_string()),
            blob: blob.map(|v| ByteBufB64(v.to_vec())),
            size: blob.map(|v| v.len() as u64),
            metadata: Some(Map::from_iter([("source".to_string(), json!("test"))])),
            ..Default::default()
        }
    }

    fn conversation(user: Principal, text: &str, period: u64) -> Conversation {
        let mut conversation = Conversation {
            user,
            thread: Some(Xid([period as u8; 12])),
            resources: vec![resource("input", Some(b"input text"))],
            artifacts: vec![resource("artifact", Some(b"artifact text"))],
            status: ConversationStatus::Working,
            failed_reason: Some("not finished".to_string()),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_tokens: 3,
                requests: 2,
            },
            steering_messages: Some(vec!["steer-one".to_string()]),
            follow_up_messages: Some(vec!["follow-one".to_string()]),
            child: Some(99),
            ancestors: Some(vec![1, 2]),
            label: Some("label-a".to_string()),
            extra: Some(json!({"priority": "high"})),
            period,
            created_at: 1_700_000_000_000,
            updated_at: 1_700_000_000_001,
            ..Default::default()
        };
        conversation.append_messages(vec![
            message("user", text),
            Message {
                role: "assistant".to_string(),
                content: vec![
                    ContentPart::Reasoning {
                        text: "thinking".to_string(),
                    },
                    ContentPart::ToolCall {
                        name: "lookup".to_string(),
                        args: json!({"q": text}),
                        call_id: Some("call-1".to_string()),
                    },
                ],
                timestamp: Some(1_700_000_000_002),
                ..Default::default()
            },
        ]);
        conversation
    }

    async fn test_db() -> Arc<AndaDB> {
        Arc::new(
            AndaDB::connect(Arc::new(InMemory::new()), DBConfig::default())
                .await
                .unwrap(),
        )
    }

    async fn test_conversations(name: &str) -> Conversations {
        Conversations::connect(test_db().await, name.to_string())
            .await
            .unwrap()
    }

    async fn test_memory() -> Arc<MemoryManagement> {
        test_memory_with(Tenancy::Shared).await
    }

    async fn test_memory_with(tenancy: Tenancy) -> Arc<MemoryManagement> {
        let db = test_db().await;
        let nexus = CognitiveNexus::connect(db.clone()).await.unwrap();
        // A Space that has activated nothing resolves Core alone, and Core declares
        // no Concept types at all, so the caller-identity helpers need the baseline
        // cognitive-memory profile in force before `Person` means anything.
        nexus
            .install_and_activate(
                &[(
                    "anda_engine",
                    anda_cognitive_nexus::profiles::COGNITIVE_MEMORY,
                )],
                anda_cognitive_nexus::nexus::DEFAULT_SPACE,
            )
            .await
            .unwrap();
        Arc::new(
            MemoryManagement::connect(db, Arc::new(nexus))
                .await
                .unwrap()
                .with_tenancy(tenancy),
        )
    }

    /// The single result of a KIP response that must have succeeded.
    fn kip_value(res: &Response) -> Json {
        assert_eq!(res.status, TopLevelStatus::Succeeded, "{:?}", res.error);
        res.first_result().cloned().unwrap_or(Json::Null)
    }

    fn test_ctx(caller: Principal) -> BaseCtx {
        BaseCtx::new(
            Principal::anonymous(),
            "engine".to_string(),
            "agent".to_string(),
            CancellationToken::new(),
            BTreeSet::new(),
            Arc::new(Web3SDK::not_implemented()),
            Store::new(Arc::new(InMemory::new())),
            Arc::new(RemoteEngines::new()),
        )
        .with_caller(caller)
    }

    #[test]
    fn test_conversation_status() {
        let chat = Conversation {
            status: ConversationStatus::Completed,
            ..Default::default()
        };
        let rt = ConversationStatus::Completed;
        println!("{}", rt);

        let rt = serde_json::to_string(&chat).unwrap();
        assert!(rt.contains(r#","status":"completed","#));
        let chat2: Conversation = serde_json::from_str(&rt).unwrap();
        assert_eq!(chat.status, chat2.status);

        let args = MemoryToolArgs::GetConversation { _id: 1 };
        let rt = serde_json::to_string(&args).unwrap();
        assert_eq!(rt, r#"{"type":"GetConversation","_id":1}"#);
        let args1: MemoryToolArgs = serde_json::from_str(&rt).unwrap();
        assert_eq!(args, args1);

        let strict_args: MemoryToolArgs = serde_json::from_value(json!({
            "type": "GetConversation",
            "_id": 1,
            "conversation": null,
            "messages_offset": null,
            "artifacts_offset": null,
            "message": null,
            "cursor": null,
            "limit": null,
            "query": null
        }))
        .unwrap();
        assert_eq!(strict_args, args);

        let schema = memory_tool_schema();
        let required = schema["required"].as_array().unwrap();
        let properties = schema["properties"].as_object().unwrap();
        assert_eq!(required.len(), properties.len());
        for key in properties.keys() {
            assert!(required.iter().any(|item| item.as_str() == Some(key)));
        }
    }

    #[test]
    fn conversation_conversions_preserve_public_state() {
        let user = principal(1);
        let conversation = conversation(user, "alpha memory topic", 17);

        let changes = conversation.to_changes().unwrap();
        assert!(matches!(changes.get("messages"), Some(Fv::Array(_))));
        assert!(matches!(changes.get("resources"), Some(Fv::Array(_))));
        assert!(matches!(changes.get("artifacts"), Some(Fv::Array(_))));
        assert!(matches!(
            changes.get("status"),
            Some(Fv::Text(v)) if v == "working"
        ));
        assert!(matches!(
            changes.get("steering_messages"),
            Some(Fv::Array(_))
        ));
        assert!(matches!(
            changes.get("follow_up_messages"),
            Some(Fv::Array(_))
        ));
        assert!(matches!(changes.get("label"), Some(Fv::Text(v)) if v == "label-a"));
        assert!(!matches!(changes.get("extra"), Some(Fv::Null) | None));
        assert!(matches!(changes.get("child"), Some(Fv::U64(99))));
        assert!(matches!(
            changes.get("failed_reason"),
            Some(Fv::Text(v)) if v == "not finished"
        ));

        let delta = conversation.to_delta(1, 1);
        assert_eq!(delta._id, conversation._id);
        assert_eq!(delta.messages.len(), 1);
        assert_eq!(delta.artifacts.len(), 0);
        assert_eq!(delta.status, ConversationStatus::Working);
        assert_eq!(delta.child, Some(99));

        let owned_delta = conversation.clone().into_delta(0, 0);
        assert_eq!(owned_delta.messages.len(), 2);
        assert_eq!(owned_delta.artifacts.len(), 1);

        let pruned = PrunedMessage::try_from(Message {
            role: "assistant".to_string(),
            content: vec![
                ContentPart::Text {
                    text: "visible".to_string(),
                },
                ContentPart::ToolCall {
                    name: "hidden".to_string(),
                    args: json!({}),
                    call_id: None,
                },
            ],
            user: Some(user),
            timestamp: Some(1_700_000_000_000),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(pruned.role, "assistant");
        assert_eq!(pruned.content.len(), 2);
        assert_eq!(pruned.user, Some(user.to_string()));
        assert_eq!(
            pruned.timestamp.as_deref(),
            Some("2023-11-14T22:13:20.000Z")
        );

        let doc = Document::from(conversation.clone());
        assert_eq!(doc.metadata["_id"], json!(0));
        assert_eq!(doc.metadata["type"], json!("Conversation"));
        assert_eq!(doc.metadata["user"], json!(user.to_string()));
        assert_eq!(doc.metadata["thread"], json!(Xid([17; 12]).to_string()));
        assert_eq!(doc.metadata["label"], json!("label-a"));
        assert!(doc.content.as_array().unwrap().len() >= 2);

        let conversation_ref = ConversationRef::from(&conversation);
        assert_eq!(conversation_ref.user, &user);
        assert_eq!(conversation_ref.thread, conversation.thread.as_ref());
        assert_eq!(conversation_ref.messages.len(), 2);
        assert_eq!(conversation_ref.child, &Some(99));
        assert_eq!(
            ConversationState::from(&conversation_ref).status,
            conversation.status
        );
        assert_eq!(ConversationState::from(&conversation)._id, conversation._id);

        let sparse = Conversation {
            user,
            updated_at: 1_700_000_000_010,
            ..Default::default()
        };
        let changes = sparse.to_runner_changes().unwrap();
        // The steer/follow-up queues belong to `memory_api`, so a writer that has no value
        // for them must omit the key entirely rather than clearing what the user queued.
        assert!(!changes.contains_key("steering_messages"));
        assert!(!changes.contains_key("follow_up_messages"));
        assert!(matches!(changes.get("label"), Some(Fv::Null)));
        assert!(matches!(changes.get("extra"), Some(Fv::Null)));
        assert!(!changes.contains_key("child"));
        // A cleared reason must be written so a stale failure does not survive a later success.
        assert!(matches!(changes.get("failed_reason"), Some(Fv::Null)));

        // The public full-snapshot conversion retains its original ability to clear queues.
        let full_changes = sparse.to_changes().unwrap();
        assert!(matches!(
            full_changes.get("steering_messages"),
            Some(Fv::Null)
        ));
        assert!(matches!(
            full_changes.get("follow_up_messages"),
            Some(Fv::Null)
        ));
    }

    #[test]
    fn conversation_status_display_and_strict_args_cover_all_variants() {
        let statuses = [
            (ConversationStatus::Submitted, "submitted"),
            (ConversationStatus::Working, "working"),
            (ConversationStatus::Idle, "idle"),
            (ConversationStatus::Completed, "completed"),
            (ConversationStatus::Cancelled, "cancelled"),
            (ConversationStatus::Failed, "failed"),
        ];
        for (status, expected) in statuses {
            assert_eq!(status.to_string(), expected);
        }

        let cases = vec![
            (
                json!({
                    "type": "GetResource",
                    "_id": 7,
                    "conversation": 3,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": null,
                    "cursor": null,
                    "limit": null,
                    "query": null
                }),
                MemoryToolArgs::GetResource {
                    _id: 7,
                    conversation: 3,
                },
            ),
            (
                json!({
                    "type": "GetConversationDelta",
                    "_id": 3,
                    "conversation": null,
                    "messages_offset": 1,
                    "artifacts_offset": 2,
                    "message": null,
                    "cursor": null,
                    "limit": null,
                    "query": null
                }),
                MemoryToolArgs::GetConversationDelta {
                    _id: 3,
                    messages_offset: 1,
                    artifacts_offset: 2,
                },
            ),
            (
                json!({
                    "type": "StopConversation",
                    "_id": 3,
                    "conversation": null,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": null,
                    "cursor": null,
                    "limit": null,
                    "query": null
                }),
                MemoryToolArgs::StopConversation { _id: 3 },
            ),
            (
                json!({
                    "type": "SteerConversation",
                    "_id": 3,
                    "conversation": null,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": "redirect",
                    "cursor": null,
                    "limit": null,
                    "query": null
                }),
                MemoryToolArgs::SteerConversation {
                    _id: 3,
                    message: "redirect".to_string(),
                },
            ),
            (
                json!({
                    "type": "FollowUpConversation",
                    "_id": 3,
                    "conversation": null,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": "continue",
                    "cursor": null,
                    "limit": null,
                    "query": null
                }),
                MemoryToolArgs::FollowUpConversation {
                    _id: 3,
                    message: "continue".to_string(),
                },
            ),
            (
                json!({
                    "type": "DeleteConversation",
                    "_id": 3,
                    "conversation": null,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": null,
                    "cursor": null,
                    "limit": null,
                    "query": null
                }),
                MemoryToolArgs::DeleteConversation { _id: 3 },
            ),
            (
                json!({
                    "type": "ListPrevConversations",
                    "_id": null,
                    "conversation": null,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": null,
                    "cursor": "abc",
                    "limit": 5,
                    "query": null
                }),
                MemoryToolArgs::ListPrevConversations {
                    cursor: Some("abc".to_string()),
                    limit: Some(5),
                },
            ),
            (
                json!({
                    "type": "SearchConversations",
                    "_id": null,
                    "conversation": null,
                    "messages_offset": null,
                    "artifacts_offset": null,
                    "message": null,
                    "cursor": null,
                    "limit": 4,
                    "query": "alpha"
                }),
                MemoryToolArgs::SearchConversations {
                    query: "alpha".to_string(),
                    limit: Some(4),
                },
            ),
        ];

        for (value, expected) in cases {
            let parsed: MemoryToolArgs = serde_json::from_value(value).unwrap();
            assert_eq!(parsed, expected);
        }
    }

    #[tokio::test]
    async fn conversations_collection_crud_search_and_expiry() {
        let conversations = test_conversations("conversation_test").await;
        let user = principal(2);
        let other_user = principal(3);

        let mut first = conversation(user, "alpha searchable topic", 1);
        first.status = ConversationStatus::Submitted;
        let first_id = conversations
            .add_conversation(ConversationRef::from(&first))
            .await
            .unwrap();

        let mut second = conversation(user, "beta searchable topic", 2);
        second.status = ConversationStatus::Completed;
        second.failed_reason = None;
        let second_id = conversations
            .add_conversation(ConversationRef::from(&second))
            .await
            .unwrap();

        let other = conversation(other_user, "alpha from another user", 3);
        let other_id = conversations
            .add_conversation(ConversationRef::from(&other))
            .await
            .unwrap();

        let fetched = conversations.get_conversation(first_id).await.unwrap();
        assert_eq!(fetched.user, user);
        assert_eq!(fetched.status, ConversationStatus::Submitted);

        let mut changes = fetched.to_changes().unwrap();
        changes.insert("status".to_string(), Fv::Text("idle".to_string()));
        conversations
            .update_conversation(first_id, changes)
            .await
            .unwrap();
        assert_eq!(
            conversations
                .get_conversation(first_id)
                .await
                .unwrap()
                .status,
            ConversationStatus::Idle
        );

        let batch = conversations
            .batch_get_conversations(&user, vec![first_id, other_id, second_id])
            .await
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|item| item.user == user));

        let (page, cursor) = conversations
            .list_conversations_by_user(&user, None, Some(1))
            .await
            .unwrap();
        assert_eq!(page.len(), 1);
        assert!(cursor.is_some());
        let (next_page, _) = conversations
            .list_conversations_by_user(&user, cursor, Some(10))
            .await
            .unwrap();
        assert_eq!(next_page.len(), 1);

        let search = conversations
            .search_conversations(&user, "alpha".to_string(), Some(10))
            .await
            .unwrap();
        assert!(search.iter().all(|item| item.user == user));

        assert!(!conversations.delete_conversation(999_999).await.unwrap());
        assert!(conversations.delete_conversation(other_id).await.unwrap());
        let deleted = conversations
            .delete_expired_conversations(2 * 3600 * 1000)
            .await
            .unwrap();
        assert_eq!(deleted, 1);
        assert!(conversations.get_conversation(first_id).await.is_err());
        assert!(conversations.get_conversation(second_id).await.is_ok());
    }

    #[tokio::test]
    async fn pagination_batch_and_expiry_handle_limits_robustly() {
        let conversations = test_conversations("pagination_test").await;
        let user = principal(8);

        let mut ids = Vec::new();
        for i in 0..12u64 {
            let item = conversation(user, &format!("topic {i}"), i + 1);
            ids.push(
                conversations
                    .add_conversation(ConversationRef::from(&item))
                    .await
                    .unwrap(),
            );
        }

        // batch_get must return all requested conversations, not the database default of 10.
        let batch = conversations
            .batch_get_conversations(&user, ids.clone())
            .await
            .unwrap();
        assert_eq!(batch.len(), 12);
        assert!(
            conversations
                .batch_get_conversations(&user, Vec::new())
                .await
                .unwrap()
                .is_empty()
        );

        // limit 0 is clamped instead of panicking or returning everything.
        let (page, _) = conversations
            .list_conversations_by_user(&user, None, Some(0))
            .await
            .unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0]._id, *ids.last().unwrap());

        // Pages are newest-first and the cursor walks backwards without overlap.
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let (page, next) = conversations
                .list_conversations_by_user(&user, cursor, Some(5))
                .await
                .unwrap();
            seen.extend(page.iter().map(|conversation| conversation._id));
            match next {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        assert_eq!(seen.len(), 12);
        assert!(
            seen.windows(2).all(|pair| pair[0] > pair[1]),
            "pages must be newest-first without duplicates: {seen:?}"
        );

        // Expired deletion must drain everything, not just the first database batch of 10.
        let memory = test_memory().await;
        let expiring_user = principal(9);
        for i in 0..12u64 {
            let item = conversation(expiring_user, &format!("expiring {i}"), i + 1);
            memory
                .add_conversation(ConversationRef::from(&item))
                .await
                .unwrap();
        }
        let deleted = memory
            .delete_expired_conversations(13 * 3600 * 1000)
            .await
            .unwrap();
        assert_eq!(deleted, 12);

        // Referencing a resource ID that does not exist is rejected.
        let err = memory
            .try_add_resources(&[Resource {
                _id: 999_999,
                ..resource("ghost", Some(b"ghost"))
            }])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn memory_management_resources_conversations_and_descriptions() {
        let memory = test_memory().await;
        let user = principal(4);

        assert_eq!(memory.name(), "execute_kip");
        assert!(
            memory
                .description()
                .contains("Knowledge Interaction Protocol")
        );
        let definition = memory.definition();
        assert_eq!(definition.name, "execute_kip");
        assert_eq!(definition.strict, None);
        assert_eq!(
            memory
                .as_ref()
                .clone()
                .with_kip_function_definitions(FunctionDefinition {
                    name: "custom_kip".to_string(),
                    description: "custom".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: Some(false),
                })
                .name(),
            "custom_kip"
        );

        let _ = memory.nexus();
        let primer = memory.describe_primer().await.unwrap();
        assert_eq!(primer["schema"]["environment_version"], json!(1));
        assert!(
            primer["schema"]["types"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r.as_str().is_some_and(|r| r.ends_with("/Person")))
        );
        // Nothing has written a SelfModel, and an absent one is an answer rather
        // than a failure.
        assert_eq!(memory.describe_self().await.unwrap(), json!(null));
        // An unknown caller is likewise absent, not an error.
        assert_eq!(memory.describe_caller(&user).await.unwrap(), json!(null));
        let caller = memory
            .get_or_init_caller(&user, Some("Ada".to_string()))
            .await
            .unwrap();
        assert_eq!(caller["key"], json!(user.to_string()));
        assert_eq!(caller["name"], json!("Ada"));
        let described = memory.describe_caller(&user).await.unwrap();
        assert_eq!(described["id"], caller["id"]);
        // A second call resolves the same Person rather than minting another.
        let again = memory.get_or_init_caller(&user, None).await.unwrap();
        assert_eq!(again["id"], caller["id"]);
        assert_eq!(again["name"], json!("Ada"));

        let added_resource = resource("resource-a", Some(b"hello resource"));
        let resource_id = memory
            .add_resource(ResourceRef::from(&added_resource))
            .await
            .unwrap();
        let fetched_resource = memory.get_resource(resource_id).await.unwrap();
        assert_eq!(fetched_resource.name, "resource-a");

        let existing = Resource {
            _id: resource_id,
            blob: Some(ByteBufB64(b"ignored blob".to_vec())),
            ..fetched_resource.clone()
        };
        let inserted = resource("resource-b", Some(b"new resource"));
        let normalized = memory
            .try_add_resources(&[existing, inserted])
            .await
            .unwrap();
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0]._id, resource_id);
        assert!(normalized.iter().all(|item| item.blob.is_none()));

        let mut stored = conversation(user, "gamma conversation", 1);
        stored.resources = normalized.clone();
        stored.artifacts = vec![Resource {
            _id: normalized[1]._id,
            ..normalized[1].clone()
        }];
        let conversation_id = memory
            .add_conversation(ConversationRef::from(&stored))
            .await
            .unwrap();
        assert!(memory.max_conversation_id() >= conversation_id);
        assert_eq!(
            memory.get_conversation(conversation_id).await.unwrap().user,
            user
        );

        let (listed, _) = memory
            .list_conversations_by_user(&user, None, Some(10))
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        let found = memory
            .search_conversations(&user, "gamma".to_string(), Some(10))
            .await
            .unwrap();
        assert!(found.iter().all(|item| item.user == user));

        let deleted = memory
            .delete_expired_conversations(2 * 3600 * 1000)
            .await
            .unwrap();
        assert_eq!(deleted, 1);
        assert!(memory.get_conversation(conversation_id).await.is_err());
        // Resources are content-deduplicated and may be shared across
        // conversations, so expiring a conversation must not delete them.
        assert!(memory.get_resource(resource_id).await.is_ok());
    }

    #[tokio::test]
    async fn memory_tools_share_one_capability_group() {
        use anda_core::ToolSet;

        let memory = test_memory().await;
        let conversations = Conversations {
            conversations: memory.conversations.clone(),
        };

        let mut tools = ToolSet::<BaseCtx>::new();
        tools.add(memory.clone()).unwrap();
        tools
            .add(Arc::new(MemoryReadonly::new(memory.clone())))
            .unwrap();
        tools
            .add(Arc::new(GetResourceContentTool::new(memory.clone())))
            .unwrap();
        tools
            .add(Arc::new(ListConversationsTool::new(conversations.clone())))
            .unwrap();
        tools
            .add(Arc::new(SearchConversationsTool::new(conversations)))
            .unwrap();
        tools
            .add(Arc::new(MemoryTool::new(memory.clone())))
            .unwrap();

        let groups = tools.groups();
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.id, MEMORY_TOOL_GROUP_ID);
        assert!(group.instructions.is_some());
        // Every registered memory tool, including the dynamically named KIP tool,
        // lands in the one bundle.
        for name in [
            memory.name(),
            MemoryReadonly::NAME.to_string(),
            GetResourceContentTool::NAME.to_string(),
            ListConversationsTool::NAME.to_string(),
            SearchConversationsTool::NAME.to_string(),
            MemoryTool::NAME.to_string(),
        ] {
            assert!(group.members.contains(&name), "missing member {name}");
        }
    }

    #[tokio::test]
    async fn memory_tools_expose_definitions_and_call_local_paths() {
        let memory = test_memory().await;
        let user = principal(5);
        let ctx = test_ctx(user);

        let text_resource = resource("text-resource", Some(b"plain text"));
        let text_id = memory
            .add_resource(ResourceRef::from(&text_resource))
            .await
            .unwrap();
        let binary_resource = resource("binary-resource", Some(&[0, 159, 146, 150]));
        let binary_id = memory
            .add_resource(ResourceRef::from(&binary_resource))
            .await
            .unwrap();

        let empty_resource = Resource {
            name: "empty".to_string(),
            tags: vec!["text".to_string()],
            ..Default::default()
        };
        let missing_id = memory
            .add_resource(ResourceRef::from(&empty_resource))
            .await
            .unwrap();
        // A resource nobody's conversation references, used to check the ownership guard.
        let unowned_id = memory
            .add_resource(ResourceRef::from(&resource("unowned", Some(b"secret"))))
            .await
            .unwrap();

        let mut stored = conversation(user, "delta memory api", 3);
        stored.resources = vec![
            Resource {
                _id: text_id,
                ..text_resource.clone()
            },
            Resource {
                _id: binary_id,
                ..binary_resource.clone()
            },
            Resource {
                _id: missing_id,
                ..empty_resource.clone()
            },
        ];
        stored.status = ConversationStatus::Working;
        let conversation_id = memory
            .add_conversation(ConversationRef::from(&stored))
            .await
            .unwrap();

        let readonly = MemoryReadonly::new(memory.clone());
        assert_eq!(readonly.name(), MemoryReadonly::NAME);
        assert!(readonly.description().contains("read-only"));
        // The KIP 2.0 schemas gate `command` against `operations` with `oneOf`,
        // which strict structured-output mode does not accept, so neither KIP tool
        // claims to be strict.
        assert_eq!(readonly.definition().strict, None);

        let get_content = GetResourceContentTool::new(memory.clone());
        assert_eq!(get_content.name(), GetResourceContentTool::NAME);
        assert!(get_content.description().contains("stored resource"));
        assert_eq!(get_content.definition().strict, Some(true));
        let output = get_content
            .call(
                ctx.clone(),
                GetResourceContentArgs {
                    _id: text_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(kip_value(&output.output), json!("plain text"));
        let output = get_content
            .call(
                ctx.clone(),
                GetResourceContentArgs {
                    _id: binary_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(
            kip_value(&output.output)
                .as_str()
                .unwrap()
                .starts_with("b64:AJ-Slg")
        );
        let err = get_content
            .call(
                ctx.clone(),
                GetResourceContentArgs {
                    _id: missing_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no blob or uri"));

        // Resource ids are dense and global, so naming one the conversation does not
        // reference must be refused rather than dumping another caller's blob.
        let err = get_content
            .call(
                ctx.clone(),
                GetResourceContentArgs {
                    _id: unowned_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        // A conversation owned by somebody else is refused before the resource is read.
        let err = get_content
            .call(
                test_ctx(principal(6)),
                GetResourceContentArgs {
                    _id: text_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        let list_tool = ListConversationsTool::new(Conversations {
            conversations: memory.conversations.clone(),
        })
        .with_description("custom list".to_string());
        assert_eq!(list_tool.name(), ListConversationsTool::NAME);
        assert_eq!(list_tool.description(), "custom list");
        assert_eq!(list_tool.definition().strict, Some(true));
        let output = list_tool
            .call(
                ctx.clone(),
                ListConversationsArgs {
                    cursor: String::new(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);
        let err = list_tool
            .call(
                ctx.clone(),
                ListConversationsArgs {
                    cursor: "not-a-valid-cursor".to_string(),
                    limit: 2,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty());

        let search_tool = SearchConversationsTool::new(Conversations {
            conversations: memory.conversations.clone(),
        })
        .with_description("custom search".to_string());
        assert_eq!(search_tool.name(), SearchConversationsTool::NAME);
        assert_eq!(search_tool.description(), "custom search");
        assert_eq!(search_tool.definition().strict, Some(true));
        let output = search_tool
            .call(
                ctx.clone(),
                SearchConversationsArgs {
                    query: "delta".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);
        let output = search_tool
            .call(
                ctx.clone(),
                SearchConversationsArgs {
                    query: "delta".to_string(),
                    limit: 3,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);

        let memory_tool = MemoryTool::new(memory.clone());
        assert_eq!(memory_tool.name(), MemoryTool::NAME);
        assert!(memory_tool.description().contains("managing conversations"));
        assert_eq!(memory_tool.definition().strict, Some(true));

        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::GetConversation {
                    _id: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);

        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::GetConversationDelta {
                    _id: conversation_id,
                    messages_offset: 1,
                    artifacts_offset: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        let delta: ConversationDelta = serde_json::from_value(kip_value(&output.output)).unwrap();
        assert_eq!(delta.messages.len(), 1);

        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::GetResource {
                    _id: text_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);

        // A resource that the conversation does not reference must be rejected even for the
        // conversation owner.
        let denied = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::GetResource {
                    _id: unowned_id,
                    conversation: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(denied.to_string().contains("does not belong"));

        // Models often send "" for the first page and 0 for the default limit.
        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::ListPrevConversations {
                    cursor: Some(String::new()),
                    limit: Some(0),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);

        for args in [
            MemoryToolArgs::GetResource {
                _id: text_id,
                conversation: conversation_id,
            },
            MemoryToolArgs::GetConversationDelta {
                _id: conversation_id,
                messages_offset: 0,
                artifacts_offset: 0,
            },
            MemoryToolArgs::StopConversation {
                _id: conversation_id,
            },
            MemoryToolArgs::SteerConversation {
                _id: conversation_id,
                message: "denied steering".to_string(),
            },
            MemoryToolArgs::FollowUpConversation {
                _id: conversation_id,
                message: "denied follow-up".to_string(),
            },
            MemoryToolArgs::DeleteConversation {
                _id: conversation_id,
            },
        ] {
            let denied = memory_tool
                .call(test_ctx(principal(6)), args, Vec::new())
                .await
                .unwrap_err();
            assert!(denied.to_string().contains("permission denied"));
        }

        memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::SteerConversation {
                    _id: conversation_id,
                    message: "new steering".to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::FollowUpConversation {
                    _id: conversation_id,
                    message: "new follow-up".to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        let updated = memory.get_conversation(conversation_id).await.unwrap();
        assert_eq!(updated.steering_messages.unwrap().len(), 2);
        assert_eq!(updated.follow_up_messages.unwrap().len(), 2);

        assert!(
            memory_tool
                .call(
                    ctx.clone(),
                    MemoryToolArgs::SteerConversation {
                        _id: conversation_id,
                        message: "  ".to_string(),
                    },
                    Vec::new(),
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("steering message cannot be empty")
        );
        assert!(
            memory_tool
                .call(
                    ctx.clone(),
                    MemoryToolArgs::FollowUpConversation {
                        _id: conversation_id,
                        message: String::new(),
                    },
                    Vec::new(),
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("follow-up message cannot be empty")
        );

        memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::StopConversation {
                    _id: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            memory
                .get_conversation(conversation_id)
                .await
                .unwrap()
                .status,
            ConversationStatus::Cancelled
        );
        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::StopConversation {
                    _id: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);

        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::ListPrevConversations {
                    cursor: None,
                    limit: Some(10),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);
        let output = memory_tool
            .call(
                ctx.clone(),
                MemoryToolArgs::SearchConversations {
                    query: "delta".to_string(),
                    limit: Some(10),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);

        let denied = memory_tool
            .call(
                test_ctx(principal(6)),
                MemoryToolArgs::GetConversation {
                    _id: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(denied.to_string().contains("permission denied"));

        let output = memory_tool
            .call(
                ctx,
                MemoryToolArgs::DeleteConversation {
                    _id: conversation_id,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(kip_value(&output.output)["deleted"], json!(true));
    }

    /// One caller's memory is another caller's `not_authorized`.
    #[tokio::test(flavor = "current_thread")]
    async fn per_caller_spaces_isolate_one_tenant_from_another() {
        let memory = test_memory_with(Tenancy::PerCallerSpace).await;
        let readonly = MemoryReadonly::new(memory.clone());
        let (alice, bob) = (principal(1), principal(2));

        let write = |who: Principal, name: &str| {
            let memory = memory.clone();
            let name = name.to_string();
            async move {
                memory
                    .call(
                        test_ctx(who),
                        KipArgs {
                            command: Some(
                                r#"CREATE CONCEPT ?e {
                                     TYPE "Event" NAME :name SET ATTRIBUTES { summary: :name }
                                   }"#
                                .to_string(),
                            ),
                            parameters: Some(Map::from_iter([("name".to_string(), name.into())])),
                            ..Default::default()
                        },
                        Vec::new(),
                    )
                    .await
                    .unwrap()
            }
        };
        let events_of = |who: Principal| {
            let readonly = readonly.clone();
            async move {
                let output = readonly
                    .call(
                        test_ctx(who),
                        KipArgs {
                            command: Some(r#"FIND(?e) WHERE { ?e {type: "Event"} }"#.to_string()),
                            ..Default::default()
                        },
                        Vec::new(),
                    )
                    .await
                    .unwrap();
                kip_value(&output.output)
                    .as_array()
                    .expect("FIND returns rows")
                    .len()
            }
        };

        // Each caller writes into a Space provisioned on first contact.
        assert_eq!(write(alice, "Alice ships v2").await.is_error, None);
        assert_eq!(write(bob, "Bob ships v3").await.is_error, None);

        // Each sees exactly its own — not two, which is what one shared brain
        // would have returned to both of them.
        assert_eq!(events_of(alice).await, 1);
        assert_eq!(events_of(bob).await, 1);

        // The isolation is ownership, so it holds for the identity too: the
        // Person Alice writes is not a Person Bob can find.
        memory
            .get_or_init_caller(&alice, Some("Alice".to_string()))
            .await
            .unwrap();
        assert_eq!(
            memory.describe_caller(&alice).await.unwrap()["name"],
            "Alice"
        );
        assert_eq!(memory.describe_caller(&bob).await.unwrap(), json!(null));

        // And the host still reaches the agent's own Space, which holds neither
        // tenant's writes.
        let shared = memory
            .query(r#"FIND(?e) WHERE { ?e {type: "Event"} }"#, None)
            .await
            .unwrap();
        assert!(shared.as_array().unwrap().is_empty());

        // An anonymous caller is unverified, so it gets no Space of its own and
        // default deny gives it nothing on the agent's.
        let output = readonly
            .call(
                test_ctx(Principal::anonymous()),
                KipArgs {
                    command: Some(r#"FIND(?e) WHERE { ?e {type: "Event"} }"#.to_string()),
                    ..Default::default()
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.is_error, Some(true));

        // A tenant cannot name its way out: the engine writes the Space over
        // whatever the envelope carried.
        let mut request = Request::single(r#"FIND(?e) WHERE { ?e {type: "Event"} }"#);
        request.space = Some(SpaceSelector {
            id: Some(tenant_space(&alice)),
            uri: None,
        });
        memory.session_for(&bob, &mut request).await.unwrap();
        assert_eq!(request.space.unwrap().id.unwrap(), tenant_space(&bob));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn concurrent_first_calls_share_tenant_provisioning() {
        let memory = test_memory_with(Tenancy::PerCallerSpace).await;
        let caller = principal(8);
        let mut first = Request::single("DESCRIBE PROTOCOL");
        let mut second = Request::single("DESCRIBE CAPABILITIES");

        let (first_session, second_session) = tokio::join!(
            memory.session_for(&caller, &mut first),
            memory.session_for(&caller, &mut second),
        );

        first_session.unwrap();
        second_session.unwrap();
        assert_eq!(first.space, second.space);
        assert_eq!(first.space.unwrap().id.unwrap(), tenant_space(&caller));
    }

    #[test]
    fn readonly_batch_default_does_not_relax_writable_wire_validation() {
        let args = KipArgs {
            operations: Some(vec![
                KipOperation::Command("DESCRIBE PROTOCOL".into()),
                KipOperation::Command("DESCRIBE CAPABILITIES".into()),
            ]),
            ..Default::default()
        };
        assert!(args.clone().into_request().unwrap().validate().is_err());
        let request = args.clone().into_readonly_request().unwrap();
        request.validate().unwrap();
        assert_eq!(
            request.execution_mode(),
            anda_kip::ExecutionMode::Independent
        );
        let explicit = KipArgs {
            execution: Some(Execution::new(anda_kip::ExecutionMode::Atomic)),
            ..args
        };
        assert_eq!(
            explicit.into_readonly_request().unwrap().execution_mode(),
            anda_kip::ExecutionMode::Atomic
        );
    }

    #[test]
    fn kip_adapter_rejects_unknown_fields_and_preserves_error_details() {
        let args = json!({
            "command": "CREATE CONCEPT ?c { TYPE \"Person\" NAME \"Ada\" }",
            "options": { "dry_run": true }
        });
        assert!(serde_json::from_value::<KipArgs>(args).is_err());

        let expected = json!({"family": "kql", "reason": "malformed"});
        let wire: ErrorObject =
            KipError::cursor_invalid("kql", "malformed", "invalid cursor").into();
        let recovered = kip_error(wire);
        assert_eq!(recovered.code, KipErrorCode::CursorInvalid);
        assert_eq!(recovered.details, Some(expected));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn kip_tools_run_the_2_0_envelope_and_hold_the_readonly_line() {
        let memory = test_memory().await;
        let ctx = test_ctx(principal(9));
        let readonly = MemoryReadonly::new(memory.clone());

        // The model-facing shape: one bare command, no envelope boilerplate.
        let output = memory
            .call(
                ctx.clone(),
                KipArgs {
                    command: Some("DESCRIBE PRIMER".to_string()),
                    ..Default::default()
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.is_error, None);
        assert!(kip_value(&output.output)["space"]["id"].is_string());

        // A batch of bare strings, which the schema also allows.
        let output = readonly
            .call(
                ctx.clone(),
                KipArgs {
                    operations: Some(vec![
                        KipOperation::Command("DESCRIBE PROTOCOL".to_string()),
                        KipOperation::Command("DESCRIBE CAPABILITIES".to_string()),
                    ]),
                    ..Default::default()
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.output.status, TopLevelStatus::Succeeded);
        assert_eq!(output.output.results.len(), 2);

        // A write lands through `execute_kip`, and parameters bind structurally.
        let output = memory
            .call(
                ctx.clone(),
                KipArgs {
                    command: Some(
                        r#"CREATE CONCEPT ?e {
                             TYPE "Event" NAME :name SET ATTRIBUTES { summary: :name }
                           }"#
                        .to_string(),
                    ),
                    parameters: Some(Map::from_iter([("name".to_string(), "Deploy v2".into())])),
                    ..Default::default()
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.is_error, None);

        // The same write through the read-only tool is refused on what the
        // command parses to, and reported as a tool error rather than silently
        // returning an empty result.
        let output = readonly
            .call(
                ctx.clone(),
                KipArgs {
                    command: Some(
                        r#"CREATE CONCEPT ?e {
                             TYPE "Event" NAME "Sneaky" SET ATTRIBUTES { summary: "Sneaky" }
                           }"#
                        .to_string(),
                    ),
                    ..Default::default()
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(output.is_error, Some(true));
        assert_eq!(output.output.status, TopLevelStatus::Failed);
        assert_eq!(
            output.output.error.as_ref().unwrap().code,
            KipErrorCode::ReadonlyViolation.name()
        );
        // ... and it did not run: only the first Event exists.
        let events = memory
            .query(r#"FIND(?e) WHERE { ?e {type: "Event"} }"#, None)
            .await
            .unwrap();
        assert_eq!(events.as_array().unwrap().len(), 1);

        // Neither half of the mutually exclusive pair is an executable request.
        for args in [
            KipArgs::default(),
            KipArgs {
                command: Some("DESCRIBE PRIMER".to_string()),
                operations: Some(vec![KipOperation::Command("DESCRIBE PROTOCOL".to_string())]),
                ..Default::default()
            },
        ] {
            let output = memory.call(ctx.clone(), args, Vec::new()).await.unwrap();
            assert_eq!(output.is_error, Some(true));
            assert_eq!(
                output.output.error.as_ref().unwrap().code,
                KipErrorCode::InvalidRequestEnvelope.name()
            );
        }
    }
}
