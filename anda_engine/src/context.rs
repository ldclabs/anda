//! Execution context system for Anda agents and tools.
//!
//! Contexts carry the runtime capabilities exposed to [`anda_core`] agents and
//! tools: caller identity, request metadata, cancellation, scoped cache, object
//! storage, HTTP calls, Web3 signing, remote engine access, and per-session
//! state. Contexts are hierarchical so each agent or tool receives a scoped
//! namespace while sharing the same underlying runtime resources.

mod agent;
mod base;
mod cache;
mod engine;
mod tool;
mod web3;

pub use agent::*;
pub use base::*;
pub use engine::*;
pub use tool::*;
pub use web3::*;
