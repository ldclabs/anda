//! Dynamic subagent definitions, sessions, and management tools.
//!
//! Subagents let an engine delegate focused work to named worker agents with
//! their own instructions, allowed tools, resource tags, optional output
//! schemas, and optional long-lived session state. This module also owns
//! session compaction and background progress forwarding for those workers.

use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionRequest, ContentPart, FunctionDefinition,
    Json, Message, ModelEffort, Path, PromptCommand, PutMode, Resource, StateFeatures,
    StoreFeatures, ToolOutput, Usage, select_resources, validate_function_name,
};
use async_trait::async_trait;
use cbor2::{from_slice, to_canonical_vec};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    any::{Any, TypeId},
    collections::{BTreeMap, BTreeSet, HashMap},
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    context::{AgentCtx, BaseCtx, CompletionRunner},
    hook::{
        AgentHook, BackgroundHandle, BackgroundTaskControls, DynAgentHook, DynToolJsonHook,
        PrefixedId, ToolBackgroundHook,
    },
    memory::{Conversation, ConversationRef, ConversationStatus, Conversations},
    truncate_utf8_to_max_bytes, unix_ms,
};

mod agent;
mod args;
mod conversation;
mod manager;
mod session;

pub use self::agent::SubAgent;
pub use self::args::{SubAgentArgs, SubAgentManagerArgs};
pub use self::conversation::SubAgentConversationRecorder;
pub use self::manager::{SubAgentManager, SubAgentSet, SubAgentSetManager};
pub use self::session::{BackgroundTaskInfo, SubSession, SubSessionStatus, SubSessions};

use self::args::deserialize_optional_model_effort;
use self::conversation::SubAgentConversationLog;
#[cfg(test)]
use self::session::{STATUS_PROGRESS_MAX_BYTES, progress_text, prompt_and_resources_into_content};
use self::session::{SubAgentInput, SubSessionRunner, resources_into_content};

const CONVERSATION_IDLE_MS: u64 = 60 * 1000; // 1 minute
const CONVERSATION_WAIT_BACKGROUND_TASK_MS: u64 = 60 * 60 * 1000; // 1 hour
// How long an idle session waits for new input before re-running its idle bookkeeping.
const SESSION_INPUT_POLL_MS: u64 = 1000;
const SUBAGENT_STORE_PATH: &str = "subagents";
const SUBAGENT_METADATA_LIST_LIMIT: usize = 8;
const DEFAULT_STOP_REASON: &str = "subagent session stopped";
const DEFAULT_CANCEL_REASON: &str = "subagent session cancelled";

fn summarize_items(items: &[String], limit: usize) -> String {
    let mut summary = items
        .iter()
        .take(limit)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");

    if items.len() > limit {
        if !summary.is_empty() {
            summary.push_str(", ");
        }
        summary.push_str(&format!("and {} more", items.len() - limit));
    }

    summary
}

fn selected_model_label(model: &str) -> Option<String> {
    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_ascii_lowercase())
    }
}

/// Resolves a subagent's configured idle timeout (in seconds) into the per-session idle window in
/// milliseconds. `0` keeps the engine default; any positive value is clamped to
/// `[1s, CONVERSATION_WAIT_BACKGROUND_TASK_MS]` so a session can neither expire instantly nor
/// outlive the hard background-task wait ceiling.
fn resolve_idle_timeout_ms(idle_timeout_secs: u64) -> u64 {
    if idle_timeout_secs == 0 {
        CONVERSATION_IDLE_MS
    } else {
        idle_timeout_secs
            .saturating_mul(1000)
            .clamp(1000, CONVERSATION_WAIT_BACKGROUND_TASK_MS)
    }
}

fn estimated_content_tokens(content: &[ContentPart]) -> u64 {
    content.iter().map(|c| c.estimated_tokens() as u64).sum()
}

#[cfg(test)]
mod tests;
