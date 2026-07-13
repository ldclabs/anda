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

/// A type alias for a borrowed boxed future that is thread-safe and sendable.
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Returns a lowercase copy of an object-store path.
///
/// Lowercasing operates on the path's raw (already percent-encoded)
/// representation and re-parses it with [`Path::parse`], which does not
/// re-encode. Using `From<String>` here would percent-encode reserved
/// characters a second time (e.g. `%` -> `%25`), so it is avoided.
pub fn path_lowercase(path: &Path) -> Path {
    let mut raw = path.to_string();
    raw.make_ascii_lowercase();
    // The input comes from a valid `Path`, so `parse` (which treats the string
    // as already encoded) succeeds; fall back defensively just in case.
    Path::parse(&raw).unwrap_or_else(|_| Path::from(raw))
}

/// Joins two object-store paths without percent-encoding and lowercases the result.
///
/// Root paths are treated as empty namespaces: joining root with `b` returns
/// `b`, and joining `a` with root returns `a` (a root path yields no parts).
///
/// Segments are carried over verbatim (already percent-encoded), so the join is
/// idempotent: re-joining a namespace with a key taken from `store_list` output
/// does not double-encode reserved characters.
pub fn path_join(a: &Path, b: &Path) -> Path {
    path_lowercase(&Path::from_iter(a.parts().chain(b.parts())))
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
/// - Must not exceed 64 bytes
/// - Must start with a lowercase letter
/// - Can only contain: lowercase letters (a-z), digits (0-9), underscores (_), and hyphens (-)
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
        if !matches!(c, 'a'..='z' | '0'..='9' | '_' | '-') {
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
    fn test_path_join_handles_root() {
        let root = Path::default();
        let p = Path::from("a/b");
        assert_eq!(path_join(&root, &p).as_ref(), "a/b");
        assert_eq!(path_join(&p, &root).as_ref(), "a/b");
        assert_eq!(path_join(&root, &root).as_ref(), "");
        assert_eq!(path_join(&p, &Path::from("c")).as_ref(), "a/b/c");
    }

    #[test]
    fn test_path_join_is_idempotent_and_does_not_double_encode() {
        let ns = Path::from("agents/store");
        // `*` is percent-encoded by object_store (`a*b` -> `a%2Ab`).
        let key = Path::from("a*b");
        let joined = path_join(&ns, &key);

        // The reserved `%` in the encoded key is not itself re-encoded.
        assert!(!joined.as_ref().contains("%25"));

        // Re-joining an already-encoded key (e.g. a `store_list` location) with a
        // namespace must not double-encode it.
        let rejoined = path_join(&Path::default(), &joined);
        assert_eq!(joined, rejoined);
        assert_eq!(path_lowercase(&joined), joined);
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

    #[test]
    fn test_validate_function_name() {
        assert!(validate_function_name("foo").is_ok());
        assert!(validate_function_name("foo_bar9").is_ok());
        assert!(validate_function_name("foo-bar").is_ok());

        assert!(validate_function_name("").is_err());
        assert!(validate_function_name("9foo").is_err());
        assert!(validate_function_name("foo.bar").is_err());
        assert!(validate_function_name("foo/bar").is_err());
        assert!(validate_function_name(&"a".repeat(65)).is_err());
    }
}
