//! Web3 client implementations for Anda engine contexts.
//!
//! This crate provides concrete
//! `anda_engine::context::Web3ClientFeatures`
//! implementations. Everything is feature-gated so the default build pulls
//! no runtime dependencies:
//!
//! - `client`: the generic Web3 `client::Client` backed by `ic-agent`,
//!   `ic-cose`, and local key derivation (`crypto`) — for non-TEE
//!   environments.
//! - `tee`: the TEE gateway-backed `tee::TeeClient`, which pulls the
//!   `ic_tee_*` crates.
//! - `full`: enables both.
//!
//! The module names above are not intra-doc links because each is behind the
//! feature that gates it, so they do not resolve in a default-feature build.

/// Deterministic key derivation shared by the generic client.
#[cfg(feature = "client")]
pub mod crypto;

/// Generic Web3 client builder and runtime implementation.
#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "client")]
pub use client::*;

/// TEE gateway-backed Web3 client.
#[cfg(feature = "tee")]
pub mod tee;

#[cfg(feature = "_common")]
mod request;

#[cfg(feature = "_common")]
fn ecdsa_digest(message_hash: &[u8]) -> Result<&[u8; 32], anda_core::BoxError> {
    message_hash.try_into().map_err(|_| {
        "ECDSA message_hash must be a 32-byte digest, not a message or hex-encoded text".into()
    })
}
