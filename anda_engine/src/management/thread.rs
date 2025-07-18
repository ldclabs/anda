use anda_core::Xid;
use anda_db_schema::{AndaDBSchema, FieldEntry, FieldType, Schema, SchemaError};
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Thread is a conversation session between Agents and users.
#[derive(Debug, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct Thread {
    /// The unique identifier for this resource in the Anda DB collection "threads".
    pub _id: u64,

    /// The unique identifier for the thread.
    #[field_type = "Bytes"]
    #[unique]
    pub id: Xid,

    /// The title of the thread.
    pub title: String,

    /// The description of the thread, can be generated by LLM from the conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The principal of the agent that created and serve the thread.
    #[field_type = "Bytes"]
    pub agent: Principal,

    /// The initiator of the thread, typically an agent or user principal.
    #[field_type = "Array<Bytes>"]
    pub managers: BTreeSet<Principal>,

    /// The participants of the thread.
    #[field_type = "Array<Bytes>"]
    pub participants: BTreeSet<Principal>,

    /// The messages count in the thread.
    pub messages: u64,

    /// The timestamp when the thread was created.
    pub created_at: u64,

    /// The timestamp when the thread was last updated.
    pub updated_at: u64,
}

impl Thread {
    /// Check if the given principal has permission to access the thread.
    pub fn has_permission(&self, id: &Principal) -> bool {
        &self.agent == id || self.managers.contains(id) || self.participants.contains(id)
    }
}
