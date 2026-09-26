//! Resource and lifecycle policies for untrusted MCP peers.

use anda_core::BoxError;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Resource budgets for one MCP server. All sizes are UTF-8/wire bytes.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct McpLimits {
    /// Maximum pages in a single catalog enumeration.
    pub catalog_pages: usize,
    /// Maximum items in a single catalog enumeration, before filtering.
    pub catalog_items: usize,
    /// Maximum pagination cursor size.
    pub cursor_bytes: usize,
    /// Maximum serialized input or output schema size per tool.
    pub schema_bytes: usize,
    /// Maximum description size per tool.
    pub description_bytes: usize,
    /// Maximum combined server title, description, and instructions size.
    pub server_metadata_bytes: usize,
    /// Maximum incoming stdio line, HTTP JSON body, or SSE event size.
    pub message_bytes: usize,
    /// Maximum text sent to the model per tool result, excluding binary media.
    pub output_text_bytes: usize,
    /// Maximum decoded binary media retained in a model-facing tool result.
    pub output_media_bytes: usize,
    /// Maximum number of inline media blocks in one model-facing result.
    pub output_media_items: usize,
}

impl Default for McpLimits {
    fn default() -> Self {
        Self {
            catalog_pages: 100,
            catalog_items: 2_048,
            cursor_bytes: 64 * 1024,
            schema_bytes: 64 * 1024,
            description_bytes: 8 * 1024,
            server_metadata_bytes: 32 * 1024,
            message_bytes: 8 * 1024 * 1024,
            output_text_bytes: 32 * 1024,
            output_media_bytes: 5 * 1024 * 1024,
            output_media_items: 8,
        }
    }
}

impl McpLimits {
    pub(super) fn validate(&self) -> Result<(), BoxError> {
        if [
            self.catalog_pages,
            self.catalog_items,
            self.cursor_bytes,
            self.schema_bytes,
            self.description_bytes,
            self.server_metadata_bytes,
            self.message_bytes,
            self.output_text_bytes,
            self.output_media_bytes,
            self.output_media_items,
        ]
        .contains(&0)
            || self.output_text_bytes < 256
        {
            return Err(
                "MCP limits must be positive and output_text_bytes must be at least 256".into(),
            );
        }
        Ok(())
    }
}

/// Time budgets in seconds. Values must be between one second and one day.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct McpTimeouts {
    /// Connection establishment, including authentication and lifecycle fallback.
    pub setup_secs: u64,
    /// One complete catalog enumeration, including retries.
    pub list_secs: u64,
    /// One protocol request within a logical tool call.
    pub request_secs: u64,
    /// Entire tool call, including queueing, setup, MRTR, and task polling.
    pub call_secs: u64,
    /// Maximum wait for one application elicitation response.
    pub elicitation_secs: u64,
}

impl Default for McpTimeouts {
    fn default() -> Self {
        Self {
            setup_secs: 90,
            list_secs: 30,
            request_secs: 180,
            call_secs: 600,
            elicitation_secs: 300,
        }
    }
}

impl McpTimeouts {
    pub(super) fn validate(&self) -> Result<(), BoxError> {
        if [
            self.setup_secs,
            self.list_secs,
            self.request_secs,
            self.call_secs,
            self.elicitation_secs,
        ]
        .iter()
        .any(|n| !(1..=86_400).contains(n))
        {
            return Err("MCP timeouts must be between 1 and 86400 seconds".into());
        }
        Ok(())
    }

    pub(super) fn call(&self) -> Duration {
        Duration::from_secs(self.call_secs)
    }
}

/// Host-authorized concurrency policy. Remote annotations never grant permissions.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpConcurrency {
    /// Serialize all tools on this server (the safe default).
    #[default]
    Serial,
    /// Allow annotated read-only tools together; writes exclude every other call.
    ReadOnlyParallel,
    /// The host explicitly allows all tools to run concurrently.
    Parallel,
}

/// Startup behavior for an optional server.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpStartup {
    /// Wait for initial discovery before the engine becomes ready.
    #[default]
    Eager,
    /// Discover in the background. No tool is advertised before live discovery.
    Background,
}

/// Observable connection/catalog state. Reading it never starts a connection.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpServerStatus {
    /// No connection has been attempted, or the session was disconnected.
    #[default]
    Disconnected,
    /// A connection or initial catalog fetch is in progress.
    Connecting,
    /// A session exists but no live tool catalog has been published yet.
    Connected,
    /// A live catalog was published successfully.
    Ready,
    /// The application must complete authorization.
    AuthorizationRequired,
    /// Connection or catalog retrieval failed; an explicit refresh can retry.
    Failed,
}
