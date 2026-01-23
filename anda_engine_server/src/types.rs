use anda_core::Json;
use anda_engine::engine::AgentInfo;
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformation {
    pub engines: Vec<AgentInfo>,
    pub default_engine: Principal,
    pub caller: Principal,
    pub start_time_ms: u64,
    pub extra_info: BTreeMap<String, Json>,
}
