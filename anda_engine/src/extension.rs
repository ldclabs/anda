//! Built-in tool and agent extensions.
//!
//! Extensions are optional building blocks that can be registered with an
//! [`EngineBuilder`](crate::engine::EngineBuilder) or used directly in tests.
//! They cover common runtime needs such as web fetching, workspace filesystem
//! access, shell execution, notes, skills, and todos.
//!
//! # Key Components
//! - [`fetch`]: HTTP fetching and resource loading.
//! - [`fs`]: workspace-scoped file read, write, search, and edit tools.
//! - [`note`]: lightweight per-agent note storage.
//! - [`shell`]: shell command execution via a pluggable [`shell::Executor`]
//!   runtime (the host [`shell::NativeRuntime`] by default).
//! - [`skill`]: file-backed skill loading and lifecycle management.
//! - [`mod@todo`]: session-scoped task tracking for long-running agents.

pub mod fetch;
pub mod fs;
pub mod note;
pub mod shell;
pub mod skill;
pub mod todo;
