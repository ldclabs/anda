//! Shell command execution extension.
//!
//! [`ShellTool`] exposes controlled command execution to agents. The tool is
//! runtime-agnostic: [`NativeRuntime`] runs commands on the host, while the
//! `sandbox` module (behind the `sandbox` feature) runs commands in an isolated
//! Boxlite environment.
//!
//! Command output is normalized into [`ExecOutput`]. Large stdout or stderr
//! streams are truncated in the tool response and written to a temporary file
//! referenced by [`ExecOutput::raw_output_path`]. Native commands receive only a
//! small allowlist of host environment variables plus explicit configured keys;
//! arbitrary process environment variables are not forwarded.

#[cfg(target_os = "windows")]
use anda_core::windows_code_page_encoding;
use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolOutput};
use async_trait::async_trait;
use encoding_rs::{Encoding, UTF_8};
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

use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

/// Maximum foreground shell command execution time before timeout handling.
pub const SHELL_TIMEOUT_SECS: u64 = 180;
/// Maximum native foreground shell runtime before moving the command to background execution.
pub const SHELL_AUTO_BACKGROUND_SECS: u64 = 42;
/// Maximum inline output size in bytes before writing a raw-output file.
pub const MAX_OUTPUT_BYTES: usize = 256 * 1024;

/// Environment variables safe to pass to shell commands.
/// Only functional variables are included — never API keys or secrets.
#[cfg(not(target_os = "windows"))]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "USER", "SHELL", "TMPDIR",
];

/// Environment variables safe to pass to shell commands on Windows.
/// Includes Windows-specific variables needed for cmd.exe and program resolution.
#[cfg(target_os = "windows")]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH",
    "PATHEXT",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TERM",
    "LANG",
    "USERNAME",
];

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
    /// Runtime implementations resolve command execution relative to this path.
    fn workspace(&self) -> &PathBuf;

    /// Return the shell program name used by this runtime, if any.
    fn shell(&self) -> &str;

    /// Execute a shell command in this runtime environment, returning the output.
    async fn execute(
        &self,
        ctx: BaseCtx,
        input: ExecArgs,
        envs: HashMap<String, String>,
    ) -> Result<ExecOutput, BoxError>;
}

/// Arguments for shell process execution.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ExecArgs {
    /// Shell command to execute.
    pub command: String,

    /// Additional custom environment variable keys to expose to the command.
    ///
    /// Values are taken from [`CustomEnv`] entries configured on [`ShellTool`].
    /// Variables marked [`CustomEnv::default`] are injected automatically and
    /// do not need to be listed here.
    #[serde(default)]
    pub env_keys: Vec<String>,

    /// Whether to start in background immediately.
    ///
    /// Native foreground commands that are still running after
    /// [`SHELL_AUTO_BACKGROUND_SECS`] are moved to background automatically.
    #[serde(default)]
    pub background: bool,
}

/// Normalized result returned by a shell execution.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ExecOutput {
    /// The working directory the command was executed.
    pub workspace: Option<String>,
    /// Process identifier, if provided by the runtime.
    pub process_id: Option<u32>,
    /// The status (exit code) of the process.
    pub exit_status: Option<String>,
    /// The data that the process wrote to stdout.
    pub stdout: Option<String>,
    /// The data that the process wrote to stderr.
    pub stderr: Option<String>,
    /// Path to the raw-output file when stdout or stderr exceeded the inline limit.
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
                    decode_shell_output(&output.stdout),
                    &raw_output,
                );
                rt.stdout = Some(stdout);
            }

            if !output.stderr.is_empty() {
                let stderr = format_output_preview(
                    "stderr",
                    decode_shell_output(&output.stderr),
                    &raw_output,
                );
                rt.stderr = Some(stderr);
            }
        }
        rt
    }
}

/// Typed hook for shell tool calls.
pub type ShellToolHook = DynToolHook<ExecArgs, ExecOutput>;

/// Configured environment variable available to shell commands.
///
/// The `key` and `description` are exposed to model providers so callers can
/// discover optional variables. The `value` is only injected into subprocesses
/// and is never included in tool descriptions or JSON schemas.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CustomEnv {
    /// Environment variable name.
    pub key: String,
    /// Environment variable value injected into the subprocess environment.
    pub value: String,
    /// Human-readable explanation of what this variable enables.
    #[serde(default)]
    pub description: String,
    /// Whether this variable is injected even when it is not requested.
    #[serde(default)]
    pub default: bool,
}

/// Tool implementation that exposes shell command execution to the engine.
#[derive(Clone)]
pub struct ShellTool {
    runtime: Arc<dyn Executor>,
    envs: Vec<CustomEnv>,
    name: String,
    description: String,
}

impl ShellTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "shell";

    /// Creates a shell tool with a runtime and a simple configured environment map.
    ///
    /// Each map entry becomes an optional [`CustomEnv`] without a description.
    /// Prefer [`Self::new_with_custom_envs`] when callers need to discover the
    /// available keys or when some variables should be injected by default.
    pub fn new(
        runtime: Arc<dyn Executor>,
        envs: HashMap<String, String>,
        name: Option<String>,
    ) -> Self {
        let envs = envs
            .into_iter()
            .map(|(key, value)| CustomEnv {
                key,
                value,
                description: String::new(),
                default: false,
            })
            .collect();
        Self::new_with_custom_envs(runtime, envs, name)
    }

    /// Creates a shell tool with documented custom environment variables.
    ///
    /// The custom environment list is not forwarded wholesale. Agents must request
    /// optional keys through [`ExecArgs::env_keys`], while variables with
    /// [`CustomEnv::default`] are injected automatically. Invalid environment
    /// variable names are ignored, and values are never exposed in tool metadata.
    pub fn new_with_custom_envs(
        runtime: Arc<dyn Executor>,
        envs: Vec<CustomEnv>,
        name: Option<String>,
    ) -> Self {
        let envs = normalize_custom_envs(envs);
        let name = name.unwrap_or_else(|| Self::NAME.to_string());
        let description = default_tool_description(runtime.as_ref(), &envs);

        Self {
            runtime,
            envs,
            name,
            description,
        }
    }

    /// Overrides the default tool description exposed to model providers.
    ///
    /// Custom environment metadata is appended so available keys remain
    /// discoverable, but values stay hidden.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = append_custom_env_description(description, &self.envs);
        self
    }

    fn env_keys_parameter_description(&self) -> String {
        let Some(custom_envs) = format_custom_env_description(&self.envs) else {
            return "Additional configured environment variable keys to set for the command"
                .to_string();
        };

        format!(
            "Optional custom environment variable keys to set for the command. Auto-injected keys are already included. Unknown or invalid keys are ignored. Available custom environment variables:\n{custom_envs}"
        )
    }

    fn collect_shell_env_vars(&self, env_keys: &[String]) -> HashMap<String, String> {
        let mut out = HashMap::new();
        if self.runtime.name() == "native" {
            // For native runtime, we allow safe environment variables from the host process
            for key in SAFE_ENV_VARS {
                let candidate = key.trim();
                if candidate.is_empty() || !is_valid_env_var_name(candidate) {
                    continue;
                }
                if let Ok(val) = std::env::var(candidate) {
                    out.insert(candidate.to_string(), val);
                }
            }
        }

        for env in self.envs.iter().filter(|env| env.default) {
            if !out.contains_key(&env.key) {
                out.insert(env.key.clone(), env.value.clone());
            }
        }

        for key in env_keys.iter() {
            let candidate = key.trim();
            if candidate.is_empty() || !is_valid_env_var_name(candidate) {
                continue;
            }
            if !out.contains_key(candidate)
                && let Some(env) = self.envs.iter().find(|env| env.key == candidate)
            {
                out.insert(candidate.to_string(), env.value.clone());
            }
        }
        out
    }
}

impl Tool<BaseCtx> for ShellTool {
    type Args = ExecArgs;
    type Output = ExecOutput;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        let env_keys_description = self.env_keys_parameter_description();
        let background_description = format!(
            "Whether to run the command in the background immediately (non-blocking). If false, native commands still running after {SHELL_AUTO_BACKGROUND_SECS} seconds are moved to the background automatically instead of returning a timeout error. New stdout/stderr output is pushed through background progress hooks as line-based progress: plain output is emitted as complete lines, multibyte character split boundaries are preserved, and terminal-style rewritten progress regions are normalized to their latest changed visible lines. The final output is pushed through background end hooks when the task completes."
        );

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
                    "env_keys": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": env_keys_description,
                        "default": []
                    },
                    "background": {
                        "type": "boolean",
                        "description": background_description,
                        "default": false
                    }
                },
                "required": ["command", "env_keys", "background"],
                "additionalProperties": false
            }),
            strict: Some(true),
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
        let envs = self.collect_shell_env_vars(&args.env_keys);

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

fn normalize_custom_envs(envs: Vec<CustomEnv>) -> Vec<CustomEnv> {
    let mut by_key = HashMap::new();
    for mut env in envs {
        env.key = env.key.trim().to_string();
        env.description = env.description.trim().to_string();
        if env.key.is_empty() || !is_valid_env_var_name(&env.key) {
            continue;
        }
        by_key.entry(env.key.clone()).or_insert(env);
    }

    let mut envs = by_key.into_values().collect::<Vec<_>>();
    envs.sort_by(|left, right| left.key.cmp(&right.key));
    envs
}

fn default_tool_description(runtime: &dyn Executor, envs: &[CustomEnv]) -> String {
    let description = format!(
        "Execute a shell command in the workspace directory (Runtime: {}, OS: {}, Shell: {})",
        runtime.name(),
        runtime.os(),
        runtime.shell()
    );

    append_custom_env_description(description, envs)
}

fn append_custom_env_description(mut description: String, envs: &[CustomEnv]) -> String {
    if let Some(custom_envs) = format_custom_env_description(envs) {
        description.push_str(
            "\n\nCustom environment variables are listed by key only; values are never exposed. Auto-injected variables are included in every command, and optional variables can be requested with env_keys.\n",
        );
        description.push_str(&custom_envs);
    }

    description
}

fn format_custom_env_description(envs: &[CustomEnv]) -> Option<String> {
    if envs.is_empty() {
        return None;
    }

    Some(
        envs.iter()
            .map(|env| {
                let mode = if env.default {
                    "auto-injected"
                } else {
                    "available via env_keys"
                };
                if env.description.is_empty() {
                    format!("- {} ({mode})", env.key)
                } else {
                    format!("- {} ({mode}): {}", env.key, env.description)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn is_valid_env_var_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
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

pub(crate) fn decode_shell_output(bytes: &[u8]) -> String {
    decode_shell_output_with_encoding(bytes, shell_output_encoding())
}

fn decode_shell_output_with_encoding(bytes: &[u8], fallback_encoding: &'static Encoding) -> String {
    if bytes.is_empty() {
        return String::new();
    }

    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.to_string();
    }

    let (text, _, _) = fallback_encoding.decode(bytes);
    text.into_owned()
}

pub(crate) fn complete_shell_output_prefix_len(bytes: &[u8]) -> usize {
    complete_shell_output_prefix_len_with_encoding(bytes, shell_output_encoding())
}

fn complete_shell_output_prefix_len_with_encoding(
    bytes: &[u8],
    encoding: &'static Encoding,
) -> usize {
    match encoding.name() {
        "UTF-8" => complete_utf8_prefix_len(bytes),
        "GBK" | "Big5" | "Shift_JIS" | "EUC-KR" => {
            complete_legacy_multibyte_prefix_len(bytes, encoding)
        }
        _ => bytes.len(),
    }
}

fn complete_utf8_prefix_len(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }

    let mut continuation_start = bytes.len();
    while continuation_start > 0 && is_utf8_continuation_byte(bytes[continuation_start - 1]) {
        continuation_start -= 1;
    }

    if continuation_start == 0 {
        return bytes.len();
    }

    let lead_index = if continuation_start == bytes.len() {
        bytes.len() - 1
    } else {
        continuation_start - 1
    };
    let required_len = utf8_sequence_len(bytes[lead_index]);
    if required_len > 1 && bytes.len() - lead_index < required_len {
        lead_index
    } else {
        bytes.len()
    }
}

fn is_utf8_continuation_byte(byte: u8) -> bool {
    byte & 0b1100_0000 == 0b1000_0000
}

fn utf8_sequence_len(byte: u8) -> usize {
    if byte & 0b1000_0000 == 0 {
        1
    } else if byte & 0b1110_0000 == 0b1100_0000 {
        2
    } else if byte & 0b1111_0000 == 0b1110_0000 {
        3
    } else if byte & 0b1111_1000 == 0b1111_0000 {
        4
    } else {
        1
    }
}

fn complete_legacy_multibyte_prefix_len(bytes: &[u8], encoding: &'static Encoding) -> usize {
    let mut len = 0;
    while len < bytes.len() {
        let byte = bytes[len];
        if legacy_multibyte_lead_byte(encoding, byte) {
            let Some(&trail) = bytes.get(len + 1) else {
                break;
            };
            if legacy_multibyte_trail_byte(encoding, trail) {
                len += 2;
            } else {
                len += 1;
            }
        } else {
            len += 1;
        }
    }
    len
}

fn legacy_multibyte_lead_byte(encoding: &'static Encoding, byte: u8) -> bool {
    match encoding.name() {
        "GBK" | "Big5" | "EUC-KR" => (0x81..=0xfe).contains(&byte),
        "Shift_JIS" => (0x81..=0x9f).contains(&byte) || (0xe0..=0xfc).contains(&byte),
        _ => false,
    }
}

fn legacy_multibyte_trail_byte(encoding: &'static Encoding, byte: u8) -> bool {
    match encoding.name() {
        "GBK" => (0x40..=0x7e).contains(&byte) || (0x80..=0xfe).contains(&byte),
        "Big5" => (0x40..=0x7e).contains(&byte) || (0xa1..=0xfe).contains(&byte),
        "EUC-KR" => {
            (0x41..=0x5a).contains(&byte)
                || (0x61..=0x7a).contains(&byte)
                || (0x81..=0xfe).contains(&byte)
        }
        "Shift_JIS" => (0x40..=0x7e).contains(&byte) || (0x80..=0xfc).contains(&byte),
        _ => false,
    }
}

fn shell_output_encoding() -> &'static Encoding {
    #[cfg(target_os = "windows")]
    {
        windows_code_page_encoding(windows_shell_code_page()).unwrap_or(UTF_8)
    }
    #[cfg(not(target_os = "windows"))]
    {
        UTF_8
    }
}

#[cfg(target_os = "windows")]
fn windows_shell_code_page() -> u32 {
    // `cmd.exe` built-ins such as `dir` emit bytes using the OEM code page
    // when stdout is piped. On Chinese Windows this is typically CP936 (GBK).
    unsafe { windows_sys::Win32::Globalization::GetOEMCP() }
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

/// Truncates a UTF-8 string to at most `max_bytes` without splitting a codepoint.
pub fn truncate_utf8_to_max_bytes(text: &mut String, max_bytes: usize) -> Option<usize> {
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
    use async_trait::async_trait;
    use std::process::ExitStatus;

    struct TestRuntime {
        name: &'static str,
        workspace: PathBuf,
    }

    impl TestRuntime {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                workspace: PathBuf::from("/tmp/anda-shell-test-workspace"),
            }
        }
    }

    #[async_trait]
    impl Executor for TestRuntime {
        fn name(&self) -> &str {
            self.name
        }

        fn workspace(&self) -> &PathBuf {
            &self.workspace
        }

        fn shell(&self) -> &str {
            "sh"
        }

        async fn execute(
            &self,
            _ctx: BaseCtx,
            _input: ExecArgs,
            _envs: HashMap<String, String>,
        ) -> Result<ExecOutput, BoxError> {
            unreachable!("test runtime does not execute commands")
        }
    }

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

    fn custom_env(key: &str, value: &str, description: &str, default: bool) -> CustomEnv {
        CustomEnv {
            key: key.to_string(),
            value: value.to_string(),
            description: description.to_string(),
            default,
        }
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

    #[test]
    fn shell_output_decoder_handles_gbk_and_preserves_utf8() {
        let gbk = [0xd6, 0xd0, 0xce, 0xc4, b'.', b't', b'x', b't', b'\n'];

        assert_eq!(
            decode_shell_output_with_encoding(&gbk, encoding_rs::GBK),
            "中文.txt\n"
        );
        assert_eq!(
            decode_shell_output_with_encoding("中文.txt\n".as_bytes(), encoding_rs::GBK),
            "中文.txt\n"
        );
        assert_eq!(
            complete_shell_output_prefix_len_with_encoding(&gbk[..1], encoding_rs::GBK),
            0
        );
        assert_eq!(
            complete_shell_output_prefix_len_with_encoding(&gbk[..3], encoding_rs::GBK),
            2
        );
        assert_eq!(
            complete_shell_output_prefix_len_with_encoding(&gbk, encoding_rs::GBK),
            gbk.len()
        );
    }

    #[test]
    fn collect_shell_env_vars_injects_safe_host_vars_for_native_runtime() {
        // Regression guard: the native runtime must report `name() == "native"` so safe host
        // env vars (PATH/HOME/...) are forwarded. A mismatched name silently dropped them.
        let tool = ShellTool::new(
            Arc::new(NativeRuntime::new(PathBuf::from(
                "/tmp/anda-native-env-test",
            ))),
            HashMap::new(),
            None,
        );

        let collected = tool.collect_shell_env_vars(&[]);

        // PATH is part of SAFE_ENV_VARS on every platform and is set in the test environment.
        if std::env::var_os("PATH").is_some() {
            assert!(
                collected.contains_key("PATH"),
                "native runtime should forward host PATH"
            );
        }
    }

    #[test]
    fn collect_shell_env_vars_uses_configured_keys_once() {
        let mut envs = HashMap::new();
        envs.insert("ANDA_TEST_ENV".to_string(), "configured".to_string());
        envs.insert("INVALID-NAME".to_string(), "ignored".to_string());
        let tool = ShellTool::new(Arc::new(TestRuntime::new("sandbox")), envs, None);

        let collected = tool.collect_shell_env_vars(&[
            " ANDA_TEST_ENV ".to_string(),
            "ANDA_TEST_ENV".to_string(),
            "INVALID-NAME".to_string(),
            "MISSING".to_string(),
        ]);

        assert_eq!(collected.len(), 1);
        assert_eq!(
            collected.get("ANDA_TEST_ENV").map(String::as_str),
            Some("configured")
        );
    }

    #[test]
    fn collect_shell_env_vars_auto_injects_default_custom_envs() {
        let tool = ShellTool::new_with_custom_envs(
            Arc::new(TestRuntime::new("sandbox")),
            vec![
                custom_env(
                    "ANDA_AUTO_ENV",
                    "auto",
                    "Always needed by the runtime",
                    true,
                ),
                custom_env(
                    "ANDA_OPTIONAL_ENV",
                    "optional",
                    "Only needed for optional operations",
                    false,
                ),
            ],
            None,
        );

        let collected = tool.collect_shell_env_vars(&[]);

        assert_eq!(collected.len(), 1);
        assert_eq!(
            collected.get("ANDA_AUTO_ENV").map(String::as_str),
            Some("auto")
        );
        assert!(!collected.contains_key("ANDA_OPTIONAL_ENV"));
    }

    #[test]
    fn collect_shell_env_vars_uses_requested_custom_envs() {
        let tool = ShellTool::new_with_custom_envs(
            Arc::new(TestRuntime::new("sandbox")),
            vec![
                custom_env(
                    "ANDA_AUTO_ENV",
                    "auto",
                    "Always needed by the runtime",
                    true,
                ),
                custom_env(
                    " ANDA_OPTIONAL_ENV ",
                    "optional",
                    "Only needed for optional operations",
                    false,
                ),
                custom_env("INVALID-NAME", "ignored", "Invalid key", true),
            ],
            None,
        );

        let collected = tool.collect_shell_env_vars(&[
            " ANDA_OPTIONAL_ENV ".to_string(),
            "ANDA_OPTIONAL_ENV".to_string(),
            "INVALID-NAME".to_string(),
            "MISSING".to_string(),
        ]);

        assert_eq!(collected.len(), 2);
        assert_eq!(
            collected.get("ANDA_AUTO_ENV").map(String::as_str),
            Some("auto")
        );
        assert_eq!(
            collected.get("ANDA_OPTIONAL_ENV").map(String::as_str),
            Some("optional")
        );
    }

    #[test]
    fn definition_describes_custom_envs_without_exposing_values() {
        let tool = ShellTool::new_with_custom_envs(
            Arc::new(TestRuntime::new("sandbox")),
            vec![
                custom_env("ANDA_AUTO_ENV", "auto-secret", "Always available", true),
                custom_env(
                    "ANDA_OPTIONAL_ENV",
                    "optional-secret",
                    "Request this for optional operations",
                    false,
                ),
                custom_env("INVALID-NAME", "ignored-secret", "Invalid key", true),
            ],
            None,
        );

        let description = <ShellTool as Tool<BaseCtx>>::description(&tool);
        assert!(description.contains("ANDA_AUTO_ENV"));
        assert!(description.contains("Always available"));
        assert!(description.contains("auto-injected"));
        assert!(description.contains("ANDA_OPTIONAL_ENV"));
        assert!(description.contains("Request this for optional operations"));
        assert!(description.contains("available via env_keys"));
        assert!(!description.contains("auto-secret"));
        assert!(!description.contains("optional-secret"));
        assert!(!description.contains("INVALID-NAME"));

        let definition = <ShellTool as Tool<BaseCtx>>::definition(&tool);
        let env_keys_description = definition.parameters["properties"]["env_keys"]["description"]
            .as_str()
            .unwrap();
        assert!(env_keys_description.contains("ANDA_OPTIONAL_ENV"));
        assert!(env_keys_description.contains("Request this for optional operations"));
        assert!(!env_keys_description.contains("optional-secret"));
    }

    #[test]
    fn custom_description_keeps_custom_env_metadata() {
        let tool = ShellTool::new_with_custom_envs(
            Arc::new(TestRuntime::new("sandbox")),
            vec![custom_env(
                "ANDA_OPTIONAL_ENV",
                "optional-secret",
                "Request this for optional operations",
                false,
            )],
            None,
        )
        .with_description("Run shell commands for tests".to_string());

        let description = <ShellTool as Tool<BaseCtx>>::description(&tool);
        assert!(description.starts_with("Run shell commands for tests"));
        assert!(description.contains("ANDA_OPTIONAL_ENV"));
        assert!(description.contains("Request this for optional operations"));
        assert!(!description.contains("optional-secret"));
    }
}
