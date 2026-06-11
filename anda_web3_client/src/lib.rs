//! Web3 client implementation for Anda engine contexts.
//!
//! This crate provides a concrete [`Web3ClientFeatures`](anda_engine::context::Web3ClientFeatures)
//! implementation backed by `ic-agent`, `ic-cose`, and the TEE gateway crypto
//! helpers.

/// Web3 client builder and runtime implementation.
pub mod client;

pub use client::*;
