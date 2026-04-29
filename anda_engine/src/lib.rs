//! Runtime implementation for the Anda agent framework.
//!
//! `anda_engine` turns the traits and data contracts from `anda_core` into a
//! runnable agent engine. It provides the execution context, model routing,
//! storage, hooks, memory tools, remote engine integration, and built-in
//! extensions used by Anda agents.
//!
//! # Main modules
//! - [`engine`]: engine construction, agent execution, tool calls, and remote
//!   engine metadata.
//! - [`context`]: runtime contexts passed to agents and tools, including cache,
//!   storage, HTTP, Web3, cancellation, and state features.
//! - [`model`]: model provider adapters and label-based model routing.
//! - [`extension`]: reusable tools such as filesystem, shell, fetch, notes,
//!   skills, todos, extraction, and search.
//! - [`memory`]: persistent conversation and resource memory backed by AndaDB
//!   and the Cognitive Nexus.
//! - [`store`]: object storage abstraction used by engine contexts.
//!
//! # Feature flags
//! - `sandbox`: enables the sandboxed shell runtime backed by `boxlite`.
//! - `full`: enables all optional runtime features currently provided by this
//!   crate.

use anda_core::Json;
use candid::Principal;
use chrono::prelude::*;
use rand::RngExt;

pub mod context;
pub mod engine;
pub mod extension;
pub mod hook;
pub mod management;
pub mod memory;
pub mod model;
pub mod store;

/// Returns the current Unix timestamp in milliseconds.
pub use structured_logger::unix_ms;

/// Generates cryptographically secure random bytes.
pub use ic_cose::rand_bytes;

/// This is used to represent unauthenticated or anonymous users in the system.
pub const ANONYMOUS: Principal = Principal::anonymous();

/// User agent string used by Anda Engine HTTP clients.
pub static APP_USER_AGENT: &str = concat!(
    "Mozilla/5.0 anda.bot ",
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
);

/// Generates a random number within the given range.
pub fn rand_number<T, R>(range: R) -> T
where
    T: rand::distr::uniform::SampleUniform,
    R: rand::distr::uniform::SampleRange<T>,
{
    let mut rng = rand::rng();
    rng.random_range(range)
}

/// Returns the current UTC datetime in RFC 3339 format with millisecond precision.
pub fn rfc3339_datetime_now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Converts a Unix timestamp in milliseconds to an RFC 3339 UTC datetime string.
pub fn rfc3339_datetime(now_ms: u64) -> Option<String> {
    let datetime = DateTime::<Utc>::from_timestamp_millis(now_ms as i64);
    datetime.map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

/// Sets the Unix timestamp in milliseconds for each JSON object in the vector.
pub fn json_set_unix_ms_timestamp(mut vals: Vec<Json>, timestamp_ms: u64) -> Vec<Json> {
    for val in vals.iter_mut() {
        if let Some(obj) = val.as_object_mut() {
            obj.insert("timestamp".into(), timestamp_ms.into());
        }
    }
    vals
}

/// Converts each JSON object's `timestamp` field from Unix milliseconds to RFC 3339.
pub fn json_convert_rfc3339_timestamp(mut vals: Vec<Json>) -> Vec<Json> {
    for val in vals.iter_mut() {
        if let Some(obj) = val.as_object_mut()
            && let Some(timestamp_ms) = obj.get("timestamp").and_then(Json::as_u64)
        {
            obj.insert("timestamp".into(), rfc3339_datetime(timestamp_ms).into());
        }
    }
    vals
}
