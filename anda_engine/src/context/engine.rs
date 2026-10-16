//! Public engine card.
//!
//! [`EngineCard`] is the public description an engine returns from
//! [`Engine::information`](crate::engine::Engine::information): its identity,
//! metadata, and exported agent and tool definitions.

use anda_core::Function;
use candid::Principal;
use serde::{Deserialize, Serialize};

pub use anda_cloud_cdk::AgentInfo;

/// Information about the engine, including agent and tool definitions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EngineCard {
    /// The principal ID of the engine.
    pub id: Principal,
    /// Information about the agent, including name, description, and supported protocols.
    pub info: AgentInfo,
    /// Definitions for agents in the engine.
    pub agents: Vec<Function>,
    /// Definitions for tools in the engine.
    pub tools: Vec<Function>,
}
