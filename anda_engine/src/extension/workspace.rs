//! Explicit tool bundles: coding uses shell plus patch, restricted file agents
//! can register filesystem tools without granting command execution.

use super::{
    fs::{ApplyPatchTool, EditFileTool, ReadFileTool, SearchFileTool, WriteFileTool},
    shell::{CustomEnv, Executor, NativeRuntime, ShellCommandTool, ShellSessionTool},
};
use crate::context::BaseCtx;
use anda_core::{BoxError, ToolSet};
use std::{path::PathBuf, sync::Arc};

/// Coding tools for a shared local workspace: shell, shell_session, apply_patch.
/// Install a ShellSessionScope in host context state before running the agent.
/// NativeRuntime is unrestricted unless explicitly configured with a sandbox.
pub fn coding_tools(
    runtime: Arc<NativeRuntime>,
    envs: Vec<CustomEnv>,
) -> Result<ToolSet<BaseCtx>, BoxError> {
    let mut tools = ToolSet::new();
    tools.add(Arc::new(ApplyPatchTool::new(runtime.workspace().clone())))?;
    tools.add(Arc::new(ShellCommandTool::new(runtime.clone(), envs)))?;
    tools.add(Arc::new(ShellSessionTool::new(runtime)))?;
    Ok(tools)
}

/// Read-only file tools without a shell or file mutation capability.
pub fn readonly_file_tools(workspace: PathBuf) -> Result<ToolSet<BaseCtx>, BoxError> {
    let mut tools = ToolSet::new();
    tools.add(Arc::new(ReadFileTool::new(workspace.clone())))?;
    tools.add(Arc::new(SearchFileTool::new(workspace)))?;
    Ok(tools)
}

/// The legacy four-tool filesystem bundle for hosts without a shell.
pub fn file_tools(workspace: PathBuf) -> Result<ToolSet<BaseCtx>, BoxError> {
    let mut tools = readonly_file_tools(workspace.clone())?;
    tools.add(Arc::new(EditFileTool::new(workspace.clone())))?;
    tools.add(Arc::new(WriteFileTool::new(workspace)))?;
    Ok(tools)
}
