//! Optional session shell protocol. Legacy shell arguments and results stay stable.

use super::{CustomEnv, DEFAULT_OUTPUT_BYTES, ExecOutput, Executor, MAX_OUTPUT_BYTES, ShellTool};
use crate::{
    context::BaseCtx,
    extension::{hooked_call, tool_definition},
};
use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolGroupInfo, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

/// Host-created conversation capability. Install in context state before creating
/// tool contexts. Do not derive this scope from untrusted request metadata.
#[derive(Clone, Debug)]
pub struct ShellSessionScope(pub(crate) u128);

impl ShellSessionScope {
    /// Creates an unguessable scope; clones can share sessions across turns.
    pub fn new() -> Self {
        Self(rand::random())
    }
}

impl Default for ShellSessionScope {
    fn default() -> Self {
        Self::new()
    }
}

/// Host limits applied before any model-provided execution options.
#[derive(Clone, Debug)]
pub struct SessionLimits {
    /// Maximum retained sessions, including running processes.
    pub max_sessions: usize,
    /// Maximum command lifetime, including background execution.
    pub max_runtime: Duration,
    /// Maximum model-visible stdout and stderr bytes combined.
    pub max_output_bytes: usize,
    /// Maximum bytes written to each session's combined raw log.
    pub max_log_bytes: usize,
    /// Retention after completion; cleanup occurs on the next session operation.
    pub retention: Duration,
    /// Whether commands may keep stdin open for later interaction.
    pub allow_stdin: bool,
    /// Whether commands may allocate a pseudoterminal on supported platforms.
    pub allow_pty: bool,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_sessions: 64,
            max_runtime: Duration::from_secs(600),
            max_output_bytes: MAX_OUTPUT_BYTES,
            max_log_bytes: 32 * 1024 * 1024,
            retention: Duration::from_secs(300),
            allow_stdin: false,
            allow_pty: false,
        }
    }
}

/// Model-visible session shell request. All optional limits are capped by the host.
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct CommandArgs {
    /// Shell command to execute.
    pub command: String,
    /// Configured environment keys to forward; values never appear in the schema.
    #[serde(default)]
    pub env_keys: Vec<String>,
    /// Relative or absolute working directory. Must resolve inside the configured root.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Return immediately with a session ID.
    #[serde(default)]
    pub background: bool,
    /// Foreground wait in milliseconds; default 10000, maximum 30000. This is not a runtime timeout.
    #[serde(default)]
    pub yield_time_ms: Option<u64>,
    /// Combined stdout/stderr preview budget in bytes; minimum 128, default 32768, capped by host limits.
    #[serde(default)]
    pub max_output_bytes: Option<usize>,
    /// Maximum process lifetime in milliseconds; capped by the host, including background execution.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Keep stdin open for shell_session write/close_stdin actions; requires host permission.
    #[serde(default)]
    pub stdin: bool,
    /// Allocate a PTY; requires host permission. PTY output combines stdout and stderr.
    #[serde(default)]
    pub tty: bool,
}

/// State of a supervised shell process.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandState {
    /// The process is still running.
    #[default]
    Running,
    /// The process exited, possibly with a nonzero exit code.
    Exited,
    /// Cancellation or an explicit stop terminated the process.
    Cancelled,
    /// The host or caller runtime deadline was reached.
    TimedOut,
    /// The process could not be started or supervised.
    Failed,
}

/// Structured session result; old output fields retain their names and meaning.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CommandOutput {
    /// Legacy workspace, PID, output streams and log path fields.
    #[serde(flatten)]
    pub output: ExecOutput,
    /// Opaque session identifier, scoped to the host-created conversation capability.
    pub task_id: String,
    /// Current process lifecycle state.
    pub state: CommandState,
    /// Numeric exit code, when available.
    pub exit_code: Option<i32>,
    /// Unix termination signal, when available.
    pub signal: Option<i32>,
    /// Bytes omitted from this response because of capture or preview limits.
    pub omitted_bytes: usize,
    /// Whether the raw log has captured all process output observed so far.
    pub log_complete: bool,
}

/// Operations on an existing shell session.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionAction {
    /// Wait briefly and drain newly available output.
    #[default]
    Poll,
    /// Send bounded input, then drain newly available output.
    Write,
    /// Close piped stdin (PTY callers can send the terminal's EOF character).
    CloseStdin,
    /// Cancel the process group and return its latest state.
    Stop,
    /// Read a bounded byte range from the session log.
    ReadLog,
    /// List sessions owned by this conversation and agent.
    List,
}

/// Request for shell_session; session IDs are capabilities, not operating-system PIDs.
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct SessionArgs {
    /// Operation to perform.
    #[schemars(with = "String", extend("enum" = ["poll", "write", "close_stdin", "stop", "read_log", "list"]))]
    pub action: SessionAction,
    /// Session ID returned by shell; omitted only for list.
    #[serde(default)]
    pub task_id: Option<String>,
    /// Text to write; accepted only for write, at most 65536 bytes.
    #[serde(default)]
    pub input: Option<String>,
    /// Wait for new output, at most 30000 milliseconds; default 1000.
    #[serde(default)]
    pub yield_time_ms: Option<u64>,
    /// Combined preview budget; minimum 128, capped by the host.
    #[serde(default)]
    pub max_output_bytes: Option<usize>,
    /// Byte offset in the raw log, used only for read_log.
    #[serde(default)]
    pub offset: u64,
}

/// Lightweight session listing entry; command text and environment are not disclosed.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionInfo {
    /// Session identifier.
    pub task_id: String,
    /// Current process state.
    pub state: CommandState,
}

/// Decoded preview of a byte range in the raw combined stdout/stderr log.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogChunk {
    /// Output decoded using the platform shell encoding.
    pub text: String,
    /// Next byte offset, independent of text character count.
    pub next_offset: u64,
    /// The current end of the retained log was reached (a running process may append).
    pub eof: bool,
    /// False if the log quota or an I/O error caused output loss.
    pub complete: bool,
}

/// Result of a session action.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SessionOutput {
    /// Present for process interaction operations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandOutput>,
    /// Present for list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sessions: Vec<SessionInfo>,
    /// Present for read_log.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<LogChunk>,
}

/// Execution capabilities advertised by an executor; unsupported options fail closed.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutorCapabilities {
    /// The executor supports the session protocol.
    pub sessions: bool,
    /// Interactive stdin is enabled by the host.
    pub stdin: bool,
    /// PTY allocation is enabled and supported.
    pub tty: bool,
    /// The executor enforces an external process sandbox.
    pub sandboxed: bool,
}

fn group() -> ToolGroupInfo {
    ToolGroupInfo { id: "shell_workspace".into(), title: "Shell execution".into(),
        description: "Execute commands and supervise their sessions.".into(),
        instructions: Some("Use shell for reading, searching, building and testing. Use the registered dedicated editing tool for file changes. If shell returns a running task_id, use shell_session to poll, stop or interact with it. A working directory is not a filesystem sandbox.".into()) }
}

/// Session-aware replacement for the legacy ShellTool. Register one, not both;
/// both use the stable name `shell`. Pair with ShellSessionTool using the same executor.
///
/// Argument hooks are typed on this tool's own types (`DynToolHook<CommandArgs,
/// CommandOutput>`), so an approval gate installed as `ShellToolHook` does not
/// intercept these calls; `ShellToolHook` and `DynToolJsonHook` only receive the
/// background events of commands that outlive the foreground wait.
#[derive(Clone)]
pub struct ShellCommandTool {
    shell: ShellTool,
}

impl ShellCommandTool {
    /// Builds a session-aware shell with the same explicit environment policy as ShellTool.
    pub fn new(runtime: Arc<dyn Executor>, envs: Vec<CustomEnv>) -> Self {
        Self {
            shell: ShellTool::new_with_custom_envs(runtime, envs, None),
        }
    }
}

impl Tool<BaseCtx> for ShellCommandTool {
    type Args = CommandArgs;
    type Output = CommandOutput;
    fn name(&self) -> String {
        ShellTool::NAME.into()
    }
    fn description(&self) -> String {
        format!(
            "{}\nReturns a scoped session ID for commands still running after the wait. Runtime deadlines also apply in the background.",
            self.shell.description
        )
    }
    fn group(&self) -> Option<ToolGroupInfo> {
        Some(group())
    }
    fn definition(&self) -> FunctionDefinition {
        let mut definition = tool_definition::<CommandArgs>(self.name(), self.description());
        definition.parameters["properties"]["env_keys"]["description"] =
            self.shell.env_keys_parameter_description().into();
        let caps = self.shell.runtime.capabilities();
        for (name, enabled) in [("stdin", caps.stdin), ("tty", caps.tty)] {
            if !enabled {
                definition.parameters["properties"][name]["description"] =
                    "Disabled by this executor; leave false.".into();
            }
        }
        definition
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: CommandArgs,
        _: Vec<Resource>,
    ) -> Result<ToolOutput<CommandOutput>, BoxError> {
        hooked_call(&ctx, args, |args| async {
            let env = self.shell.collect_shell_env_vars(&args.env_keys);
            let output = self
                .shell
                .runtime
                .execute_session(ctx.clone(), args, env)
                .await?;
            let is_error = matches!(output.state, CommandState::Failed | CommandState::TimedOut);
            Ok(ToolOutput {
                is_error: is_error.then_some(true),
                ..ToolOutput::new(output)
            })
        })
        .await
    }
}

/// Bounded, conversation-scoped process interaction tool.
#[derive(Clone)]
pub struct ShellSessionTool {
    runtime: Arc<dyn Executor>,
}

impl ShellSessionTool {
    /// Uses the same executor instance as ShellCommandTool.
    pub fn new(runtime: Arc<dyn Executor>) -> Self {
        Self { runtime }
    }
}

impl Tool<BaseCtx> for ShellSessionTool {
    type Args = SessionArgs;
    type Output = SessionOutput;
    fn name(&self) -> String {
        "shell_session".into()
    }
    fn description(&self) -> String {
        "Poll, list, stop, read logs, or send input to shell sessions owned by this conversation. Input and PTYs require host permission; polling does not extend a process deadline.".into()
    }
    fn group(&self) -> Option<ToolGroupInfo> {
        Some(group())
    }
    fn definition(&self) -> FunctionDefinition {
        tool_definition::<SessionArgs>(self.name(), self.description())
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: SessionArgs,
        _: Vec<Resource>,
    ) -> Result<ToolOutput<SessionOutput>, BoxError> {
        hooked_call(&ctx, args, |args| async {
            let output = self.runtime.interact_session(ctx.clone(), args).await?;
            let is_error = output.command.as_ref().is_some_and(|output| {
                matches!(output.state, CommandState::Failed | CommandState::TimedOut)
            });
            Ok(ToolOutput {
                is_error: is_error.then_some(true),
                ..ToolOutput::new(output)
            })
        })
        .await
    }
}

pub(super) fn budget(requested: Option<usize>, limits: &SessionLimits) -> usize {
    requested
        .unwrap_or(DEFAULT_OUTPUT_BYTES)
        .clamp(128, limits.max_output_bytes.clamp(128, MAX_OUTPUT_BYTES))
}
