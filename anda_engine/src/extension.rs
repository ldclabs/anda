//! Built-in tool and agent extensions.
//!
//! Extensions are optional building blocks that can be registered with an
//! [`EngineBuilder`](crate::engine::EngineBuilder) or used directly in tests.
//! They cover common runtime needs such as structured extraction, web fetching,
//! workspace filesystem access, shell execution, notes, skills, todos, and
//! search.
//!
//! # Key Components
//! - [`extractor`]: structured extraction from unstructured text.
//! - [`fetch`]: signed HTTP fetching and resource loading.
//! - [`fs`]: workspace-scoped file read, write, search, and edit tools.
//! - [`google`]: Google search integration.
//! - [`note`]: lightweight per-agent note storage.
//! - [`shell`]: native or sandboxed command execution.
//! - [`skill`]: file-backed skill loading and lifecycle management.
//! - [`mod@todo`]: session-scoped task tracking for long-running agents.

pub mod extractor;
pub mod fetch;
pub mod fs;
pub mod google;
pub mod note;
pub mod shell;
pub mod skill;
pub mod todo;
