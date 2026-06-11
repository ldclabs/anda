//! Response payload types for the engine server.

use anda_core::Json;
use anda_engine::engine::AgentInfo;
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Discovery response returned by information endpoints.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformation {
    /// Engines exposed by this server.
    pub engines: Vec<AgentInfo>,
    /// Principal of the default engine.
    pub default_engine: Principal,
    /// Verified caller principal, or anonymous when unauthenticated.
    pub caller: Principal,
    /// Server start timestamp in milliseconds.
    pub start_time_ms: u64,
    /// Extra server metadata configured on [`crate::ServerBuilder`].
    pub extra_info: BTreeMap<String, Json>,
}
