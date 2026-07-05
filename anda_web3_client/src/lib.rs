//! Web3 client implementations for Anda engine contexts.
//!
//! This crate provides concrete
//! [`Web3ClientFeatures`](anda_engine::context::Web3ClientFeatures)
//! implementations. Everything is feature-gated so the default build pulls
//! neither `ic-agent` nor any `ic_tee_*` crate:
//!
//! - `client`: the generic Web3 [`client::Client`] backed by `ic-agent`,
//!   `ic-cose`, and local key derivation ([`crypto`]) — for non-TEE
//!   environments.
//! - `tee`: the TEE gateway-backed [`tee::TeeClient`], which pulls the
//!   `ic_tee_*` crates.
//! - `full`: enables both.

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
