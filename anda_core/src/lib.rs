//! Core traits, data models, and utility helpers for the Anda agent framework.
//!
//! `anda_core` defines the stable interfaces shared by Anda runtimes,
//! agents, tools, model adapters, and clients. It intentionally keeps runtime
//! orchestration and provider integrations out of this crate; those pieces are
//! implemented by higher-level crates such as `anda_engine`.
//!
//! The main building blocks are:
//! - [`Agent`] and [`AgentSet`] for registering and running AI agents.
//! - [`Tool`] and [`ToolSet`] for type-safe tool implementations.
//! - [`BaseContext`] and [`AgentContext`] for execution capabilities.
//! - [`Message`], [`ContentPart`], [`CompletionRequest`], and related model
//!   types for LLM provider adapters.
//! - HTTP and CBOR/Candid RPC helpers for remote engine and canister calls.

use object_store::path::DELIMITER;
use std::{future::Future, pin::Pin};

pub mod agent;
pub mod context;
pub mod http;
pub mod json;
pub mod model;
pub mod tool;

pub use agent::*;
pub use context::*;
pub use http::*;
pub use json::*;
pub use model::*;
pub use tool::*;

/// A type alias for a boxed error that is thread-safe and sendable across threads.
/// This is commonly used as a return type for functions that can return various error types.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A type alias for a boxed future that is thread-safe and sendable across threads.
pub type BoxPinFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Returns a lowercase copy of an object-store path.
pub fn path_lowercase(path: &Path) -> Path {
    let mut path = path.to_string();
    path.make_ascii_lowercase();
    path.into()
}

/// Joins two object-store paths without percent-encoding and lowercases the result.
///
/// Root paths are treated as empty namespaces: joining root with `b` returns
/// `b`, and joining `a` with root returns `a`.
pub fn path_join(a: &Path, b: &Path) -> Path {
    let mut path = if a.is_root() {
        b.to_string()
    } else if b.is_root() {
        a.to_string()
    } else {
        format!("{}{}{}", a, DELIMITER, b)
    };
    path.make_ascii_lowercase();
    path.into()
}

/// Validates a single path component used in agent, tool, or user namespaces.
///
/// The value must be non-empty, must not contain the object-store path
/// delimiter, and must round-trip through [`Path`] without normalization.
pub fn validate_path_part(part: &str) -> Result<(), BoxError> {
    if part.is_empty() || part.contains(DELIMITER) || Path::from(part).as_ref() != part {
        return Err(format!("invalid path part: {}", part).into());
    }

    Ok(())
}

/// Validates an agent or tool function name.
///
/// # Rules
/// - Must not be empty
/// - Must not exceed 64 characters
/// - Must start with a lowercase letter
/// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_)
pub fn validate_function_name(name: &str) -> Result<(), BoxError> {
    if name.is_empty() {
        return Err("empty string".into());
    }

    if name.len() > 64 {
        return Err("string length exceeds the limit 64".into());
    }

    let mut iter = name.chars();
    if !matches!(iter.next(), Some('a'..='z')) {
        return Err("name must start with a lowercase letter".into());
    }

    for c in iter {
        if !matches!(c, 'a'..='z' | '0'..='9' | '_' ) {
            return Err(format!("invalid character: {}", c).into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_lowercase() {
        let a = Path::from("a/Foo");
        assert_eq!(path_lowercase(&a).as_ref(), "a/foo");
    }

    #[test]
    fn test_validate_path_part() {
        assert!(validate_path_part("foo").is_ok());
        assert!(validate_path_part("fOO").is_ok());
        assert!(validate_path_part("").is_err());
        assert!(validate_path_part("foo/").is_err());
        assert!(validate_path_part("/foo").is_err());
        assert!(validate_path_part("foo/bar").is_err());
        assert!(validate_path_part("foo/bar/").is_err());
    }
}
