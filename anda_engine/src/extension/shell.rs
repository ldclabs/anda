//! Shell extension primitives.
//!
//! This module defines:
//! - a runtime-agnostic command execution abstraction ([`Executor`]),
//! - tool input/output types ([`ExecArgs`], [`ExecOutput`]),
//! - and the public tool entrypoint ([`ShellTool`]).
//!
//! Runtime-specific implementations live in submodules:
//! - [`native`]: execute commands on the host OS.
//! - [`sandbox`] (feature-gated): execute commands in an isolated environment.

use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolOutput};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Output,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// Host runtime implementation.
pub mod native;

pub use native::NativeRuntime;

#[cfg(feature = "sandbox")]
/// Sandboxed runtime implementation.
pub mod sandbox;

use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

/// Maximum shell command execution time before kill.
pub const SHELL_TIMEOUT_SECS: u64 = 180;
/// Maximum output size in bytes (128KB).
pub const MAX_OUTPUT_BYTES: usize = 128 * 1024;

/// Runtime abstraction used by [`ShellTool`] to execute shell commands.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Return the human-readable name of this runtime environment.
    ///
    /// Used in logs and diagnostics (e.g., `"native"`, `"sandbox"`).
    fn name(&self) -> &str;

    /// Return the operating system name exposed by this runtime.
    ///
    /// By default, this uses the host compile-time OS constant.
    fn os(&self) -> &str {
        std::env::consts::OS
    }

    /// Return the runtime base working directory.
    ///
    /// The user-provided [`ExecArgs::work_dir`] is resolved relative to this path.
    fn work_dir(&self) -> &PathBuf;

    /// Return the temporary directory used by this runtime.
    ///
    /// Oversized command output may be persisted here and referenced by
    /// [`ExecOutput::raw_output_path`].
    fn temp_dir(&self) -> &PathBuf;

    /// Return the shell program name used by this runtime, if any.
    fn shell(&self) -> Option<&str>;

    /// Execute a shell command in this runtime environment, returning the output.
    async fn execute(
        &self,
        ctx: BaseCtx,
        input: ExecArgs,
        envs: HashMap<String, String>,
    ) -> Result<ExecOutput, BoxError>;
}

// /// Hook for receiving callbacks when a background command finishes.
// #[async_trait]
// pub trait ExecutorHook: Send + Sync {
//     /// Called before any command execution (foreground or background).
//     /// The execution will be cancelled if this returns an error.
//     async fn on_execution_start(&self, _ctx: &BaseCtx, _input: &ExecArgs) -> Result<(), BoxError> {
//         Ok(())
//     }

//     /// Called after any command execution (foreground or background).
//     async fn on_execution_end(&self, _ctx: &BaseCtx, _input: &ExecArgs, _output: &ExecOutput) {
//         // Default implementation does nothing.
//     }

//     /// Called after a background execution ends.
//     ///
//     /// The default implementation is a no-op.
//     async fn on_background_end(&self, _ctx: BaseCtx, _input: ExecArgs, _output: ExecOutput) {
//         // Default implementation does nothing.
//     }
// }

// #[derive(Clone)]
// pub struct DynExecutorHook {
//     inner: Arc<dyn ExecutorHook>,
// }

// impl DynExecutorHook {
//     pub fn new(inner: Arc<dyn ExecutorHook>) -> Self {
//         Self { inner }
//     }
// }

// #[async_trait]
// impl ExecutorHook for DynExecutorHook {
//     async fn on_execution_start(&self, ctx: &BaseCtx, input: &ExecArgs) -> Result<(), BoxError> {
//         self.inner.on_execution_start(ctx, input).await
//     }

//     async fn on_execution_end(&self, ctx: &BaseCtx, input: &ExecArgs, output: &ExecOutput) {
//         self.inner.on_execution_end(ctx, input, output).await
//     }

//     async fn on_background_end(&self, ctx: BaseCtx, input: ExecArgs, output: ExecOutput) {
//         self.inner.on_background_end(ctx, input, output).await
//     }
// }

/// Arguments for process execution
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ExecArgs {
    /// The shell command to execute
    pub command: String,
    /// The working directory to execute the command in (relative to runtime storage path)
    #[serde(default)]
    pub work_dir: String,
    /// Additional environment variable keys to set for the command
    #[serde(default)]
    pub env_keys: Vec<String>,
    /// Whether to run the command in the background (non-blocking)
    #[serde(default)]
    pub background: bool,
}

/// Normalized result returned by a shell execution.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ExecOutput {
    /// Process identifier, if provided by the runtime.
    pub process_id: Option<u32>,
    /// The status (exit code) of the process.
    pub exit_status: Option<String>,
    /// The data that the process wrote to stdout.
    pub stdout: Option<String>,
    /// The data that the process wrote to stderr.
    pub stderr: Option<String>,
    /// For large outputs, the full stdout/stderr content can be saved to a temporary file and the path returned here.
    pub raw_output_path: Option<String>,
}

enum RawOutputPersistence {
    NotNeeded,
    Saved(String),
    Failed(String),
}

impl ExecOutput {
    /// Build an [`ExecOutput`] from a process handle and optional captured output.
    ///
    /// If stdout/stderr exceed [`MAX_OUTPUT_BYTES`], full content is saved to
    /// `temp_dir` and the preview is truncated.
    pub async fn from_output(
        process_id: Option<u32>,
        output: Option<Output>,
        temp_dir: &Path,
    ) -> Self {
        let mut rt = ExecOutput {
            process_id,
            exit_status: output.as_ref().map(|o| o.status.to_string()),
            ..Default::default()
        };

        if let Some(output) = output {
            let stdout_overflow = output.stdout.len() > MAX_OUTPUT_BYTES;
            let stderr_overflow = output.stderr.len() > MAX_OUTPUT_BYTES;
            let raw_output = if stdout_overflow || stderr_overflow {
                match persist_raw_output(
                    temp_dir,
                    process_id,
                    stdout_overflow.then_some(output.stdout.as_slice()),
                    stderr_overflow.then_some(output.stderr.as_slice()),
                )
                .await
                {
                    Ok(Some(path)) => {
                        rt.raw_output_path = Some(path.clone());
                        RawOutputPersistence::Saved(path)
                    }
                    Ok(None) => RawOutputPersistence::NotNeeded,
                    Err(err) => RawOutputPersistence::Failed(err.to_string()),
                }
            } else {
                RawOutputPersistence::NotNeeded
            };

            if output.status.success() || !output.stdout.is_empty() {
                let stdout = format_output_preview(
                    "stdout",
                    String::from_utf8_lossy(&output.stdout).to_string(),
                    &raw_output,
                );
                rt.stdout = Some(stdout);
            }

            if !output.stderr.is_empty() {
                let stderr = format_output_preview(
                    "stderr",
                    String::from_utf8_lossy(&output.stderr).to_string(),
                    &raw_output,
                );
                rt.stderr = Some(stderr);
            }
        }
        rt
    }
}

pub type ShellToolHook = DynToolHook<ExecArgs, ExecOutput>;

/// Tool implementation that exposes shell command execution to the engine.
#[derive(Clone)]
pub struct ShellTool {
    runtime: Arc<dyn Executor>,
    envs: HashMap<String, String>,
    description: String,
}

impl ShellTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "shell";

    /// Create a shell tool with a runtime and a fixed environment map.
    pub fn new(runtime: Arc<dyn Executor>, envs: HashMap<String, String>) -> Self {
        let description = format!(
            "Execute a shell command in the workspace directory (Runtime: {}, OS: {}, Shell: {})",
            runtime.name(),
            runtime.os(),
            runtime.shell().unwrap_or("none")
        );

        Self {
            runtime,
            envs,
            description,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

impl Tool<BaseCtx> for ShellTool {
    type Args = ExecArgs;
    type Output = ExecOutput;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "The working directory to execute the command in (relative to runtime storage path)",
                        "default": ""
                    },
                    "env_keys": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Additional environment variable keys to set for the command",
                        "default": []
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Whether to run the command in the background (non-blocking)",
                        "default": false
                    }
                },
                "required": ["command"]
            }),
            strict: None,
        }
    }

    /// Execute a shell command and return structured output.
    ///
    /// # Arguments
    /// * `ctx` - Base context.
    /// * `args` - Command arguments including command string and execution mode.
    /// * `_resources` - Additional resources (currently unused).
    ///
    /// # Returns
    /// A [`ToolOutput`] containing [`ExecOutput`].
    ///
    /// Runtime failures and timeout events are converted into a successful tool
    /// call with `stderr` populated, so callers can handle command errors as
    /// regular tool output.
    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let hook = ctx.get_state::<ShellToolHook>();
        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };
        let command = args.command.clone();
        let envs: HashMap<String, String> = args
            .env_keys
            .iter()
            .filter_map(|key| self.envs.get(key).map(|value| (key.clone(), value.clone())))
            .collect();

        let result = tokio::time::timeout(
            Duration::from_secs(SHELL_TIMEOUT_SECS),
            self.runtime.execute(ctx.clone(), args, envs),
        )
        .await;

        let rt = match result {
            Ok(Ok(output)) => ToolOutput::new(output),
            Ok(Err(err)) => ToolOutput::new(ExecOutput {
                stderr: Some(format!(
                    "Failed to execute command: {command}, error: {err}"
                )),
                ..Default::default()
            }),
            Err(_) => ToolOutput::new(ExecOutput {
                stderr: Some(format!(
                    "Failed to execute command: {command}, error: timed out after {SHELL_TIMEOUT_SECS}s and was killed"
                )),
                ..Default::default()
            }),
        };

        if let Some(hook) = &hook {
            hook.after_tool_call(&ctx, rt).await
        } else {
            Ok(rt)
        }
    }
}

fn format_output_preview(
    stream_name: &str,
    mut text: String,
    raw_output: &RawOutputPersistence,
) -> String {
    if text.len() <= MAX_OUTPUT_BYTES {
        return text;
    }

    let detail = match raw_output {
        RawOutputPersistence::NotNeeded => String::new(),
        RawOutputPersistence::Saved(path) => format!("; full output saved to {path}"),
        RawOutputPersistence::Failed(err) => {
            format!("; failed to save full output to temp file: {err}")
        }
    };
    let max_preview_bytes = MAX_OUTPUT_BYTES.saturating_sub(detail.len() + 64);
    let cutoff = truncate_utf8_to_max_bytes(&mut text, max_preview_bytes).unwrap_or(text.len());
    text.push_str(&format!(
        "\n... [{stream_name} truncated at {cutoff} bytes{detail}]"
    ));
    text
}

async fn persist_raw_output(
    temp_dir: &Path,
    process_id: Option<u32>,
    stdout: Option<&[u8]>,
    stderr: Option<&[u8]>,
) -> std::io::Result<Option<String>> {
    if stdout.is_none() && stderr.is_none() {
        return Ok(None);
    }

    tokio::fs::create_dir_all(temp_dir).await?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let process_id = process_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let file_name = format!(
        "anda-shell-output-{process_id}-{timestamp}-{:016x}.log",
        rand::random::<u64>()
    );
    let path = temp_dir.join(file_name);

    tokio::fs::write(&path, build_raw_output_bytes(stdout, stderr)).await?;
    Ok(Some(path.display().to_string()))
}

fn build_raw_output_bytes(stdout: Option<&[u8]>, stderr: Option<&[u8]>) -> Vec<u8> {
    match (stdout, stderr) {
        (Some(stdout), None) => stdout.to_vec(),
        (None, Some(stderr)) => stderr.to_vec(),
        (Some(stdout), Some(stderr)) => {
            let mut content = Vec::with_capacity(stdout.len() + stderr.len() + 40);
            content.extend_from_slice(b"===== stdout =====\n");
            content.extend_from_slice(stdout);
            if !stdout.ends_with(b"\n") {
                content.push(b'\n');
            }
            content.extend_from_slice(b"\n===== stderr =====\n");
            content.extend_from_slice(stderr);
            if !stderr.ends_with(b"\n") {
                content.push(b'\n');
            }
            content
        }
        (None, None) => Vec::new(),
    }
}

fn truncate_utf8_to_max_bytes(text: &mut String, max_bytes: usize) -> Option<usize> {
    if text.len() <= max_bytes {
        return None;
    }
    let mut cutoff = max_bytes;
    while cutoff > 0 && !text.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    text.truncate(cutoff);
    Some(cutoff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::ExitStatus;

    struct TestTempDir(PathBuf);

    impl TestTempDir {
        async fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("anda-process-test-{:016x}", rand::random::<u64>()));
            tokio::fs::create_dir_all(&path).await.unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    fn success_status() -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;

        ExitStatus::from_raw(0)
    }

    #[cfg(windows)]
    fn success_status() -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;

        ExitStatus::from_raw(0)
    }

    #[tokio::test]
    async fn persists_large_stdout_to_temp_file() {
        let temp_dir = TestTempDir::new().await;
        let stdout = vec![b'a'; MAX_OUTPUT_BYTES + 128];
        let output = Output {
            status: success_status(),
            stdout: stdout.clone(),
            stderr: Vec::new(),
        };

        let result = ExecOutput::from_output(Some(7), Some(output), temp_dir.path()).await;

        let raw_output_path = result.raw_output_path.as_ref().unwrap();
        let raw_output = tokio::fs::read(raw_output_path).await.unwrap();
        assert_eq!(raw_output, stdout);
        assert!(
            result
                .stdout
                .as_ref()
                .unwrap()
                .contains("full output saved to")
        );
        assert!(result.stdout.as_ref().unwrap().len() <= MAX_OUTPUT_BYTES);
    }

    #[tokio::test]
    async fn persists_large_stdout_and_stderr_to_shared_temp_file() {
        let temp_dir = TestTempDir::new().await;
        let stdout = vec![b'o'; MAX_OUTPUT_BYTES + 64];
        let stderr = vec![b'e'; MAX_OUTPUT_BYTES + 96];
        let output = Output {
            status: success_status(),
            stdout: stdout.clone(),
            stderr: stderr.clone(),
        };

        let result = ExecOutput::from_output(Some(11), Some(output), temp_dir.path()).await;

        let raw_output_path = result.raw_output_path.as_ref().unwrap();
        let raw_output = tokio::fs::read(raw_output_path).await.unwrap();
        assert!(raw_output.starts_with(b"===== stdout =====\n"));
        assert!(
            raw_output
                .windows(stdout.len())
                .any(|window| window == stdout)
        );
        assert!(
            raw_output
                .windows(stderr.len())
                .any(|window| window == stderr)
        );
        assert!(
            result
                .stdout
                .as_ref()
                .unwrap()
                .contains("full output saved to")
        );
        assert!(
            result
                .stderr
                .as_ref()
                .unwrap()
                .contains("full output saved to")
        );
    }

    #[tokio::test]
    async fn keeps_small_output_inline() {
        let temp_dir = TestTempDir::new().await;
        let output = Output {
            status: success_status(),
            stdout: b"ok\n".to_vec(),
            stderr: b"warn\n".to_vec(),
        };

        let result = ExecOutput::from_output(Some(13), Some(output), temp_dir.path()).await;

        assert_eq!(result.raw_output_path, None);
        assert_eq!(result.stdout.as_deref(), Some("ok\n"));
        assert_eq!(result.stderr.as_deref(), Some("warn\n"));
    }
}
