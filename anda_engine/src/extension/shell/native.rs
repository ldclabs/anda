//! Native shell command executor.
//!
//! This runtime launches commands on the host operating system and optionally
//! streams long-running output through background hooks. It is intended for
//! trusted environments; sandboxed execution lives behind the `sandbox` feature.

use anda_core::{BoxError, StateFeatures, ToolOutput};
use async_trait::async_trait;
use ic_auth_types::Xid;
use serde_json::json;
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    process::{ExitStatus, Output, Stdio},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::{Child, Command},
    sync::Mutex as TokioMutex,
};

use super::{
    ExecArgs, ExecOutput, Executor, SHELL_AUTO_BACKGROUND_SECS, ShellToolHook,
    complete_shell_output_prefix_len, decode_shell_output,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolJsonHook, ToolBackgroundHook, ToolHook},
};

#[cfg(not(test))]
const BACKGROUND_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);
#[cfg(test)]
const BACKGROUND_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
const AUTO_BACKGROUND_AFTER: std::time::Duration =
    std::time::Duration::from_secs(SHELL_AUTO_BACKGROUND_SECS);
const OUTPUT_READ_CHUNK_BYTES: usize = 8192;
/// How long to wait for the output readers after the process exits. Descendant processes can
/// inherit the pipes and keep them open indefinitely; after this grace period the readers are
/// aborted and the output captured so far is used.
#[cfg(not(test))]
const OUTPUT_READER_GRACE: std::time::Duration = std::time::Duration::from_secs(2);
#[cfg(test)]
const OUTPUT_READER_GRACE: std::time::Duration = std::time::Duration::from_millis(100);
/// Maximum bytes kept in memory per output stream. When exceeded, the oldest bytes are dropped
/// so the tail of the output (where errors usually appear) is preserved.
const MAX_STREAM_BUFFER_BYTES: usize = 8 * 1024 * 1024;
/// Maximum bytes for a single background progress chunk pushed through hooks.
const MAX_PROGRESS_CHUNK_BYTES: usize = 16 * 1024;
/// Maximum terminal rows kept for progress rendering; older rows are scrolled out.
const MAX_TERMINAL_ROWS: usize = 4096;
/// Maximum terminal columns honored from cursor-positioning escape sequences.
const MAX_TERMINAL_COLUMNS: usize = 8192;
/// Maximum bytes a single rendered terminal line can grow to.
const MAX_TERMINAL_LINE_BYTES: usize = 64 * 1024;

type OutputBuffer = std::sync::Arc<TokioMutex<StreamBuffer>>;
type OutputReaderHandle = tokio::task::JoinHandle<std::io::Result<()>>;

/// In-memory capture of one output stream, bounded by [`MAX_STREAM_BUFFER_BYTES`].
///
/// Offsets used by progress tracking are absolute stream offsets: `trimmed` counts the bytes
/// already dropped from the head, and `data` holds the bytes from `trimmed` onwards.
#[derive(Default)]
struct StreamBuffer {
    data: Vec<u8>,
    trimmed: usize,
}

impl StreamBuffer {
    #[cfg(test)]
    fn from_bytes(data: Vec<u8>) -> Self {
        Self { data, trimmed: 0 }
    }

    fn append(&mut self, chunk: &[u8]) {
        self.data.extend_from_slice(chunk);
        if self.data.len() > MAX_STREAM_BUFFER_BYTES {
            // Drop down to 7/8 of the cap so trimming is amortized instead of per-append.
            let excess = self.data.len() - MAX_STREAM_BUFFER_BYTES / 8 * 7;
            self.data.drain(..excess);
            self.trimmed += excess;
        }
    }

    /// Total bytes ever written to this stream.
    fn total_len(&self) -> usize {
        self.trimmed + self.data.len()
    }

    /// Returns the captured bytes, prefixed with a marker when the head was dropped.
    fn into_bytes(self, stream_name: &str) -> Vec<u8> {
        if self.trimmed == 0 {
            return self.data;
        }

        let marker = format!(
            "[{} bytes of {stream_name} dropped: output exceeded the {} MiB in-memory buffer]\n",
            self.trimmed,
            MAX_STREAM_BUFFER_BYTES / 1024 / 1024,
        );
        let mut bytes = Vec::with_capacity(marker.len() + self.data.len());
        bytes.extend_from_slice(marker.as_bytes());
        bytes.extend_from_slice(&self.data);
        bytes
    }
}

struct RunningProcess {
    child: Child,
    stdout: OutputBuffer,
    stderr: OutputBuffer,
    stdout_reader: OutputReaderHandle,
    stderr_reader: OutputReaderHandle,
}

/// Native runtime — full access, runs on Mac/Linux/Docker/Raspberry Pi
pub struct NativeRuntime {
    workspace: PathBuf,
    temp_dir: PathBuf,
    insecure: bool,
    background_progress_interval: std::time::Duration,
    auto_background_after: std::time::Duration,
}

impl NativeRuntime {
    /// Builds the platform shell command used to execute a command string.
    pub fn build_shell_command(command: &str) -> std::process::Command {
        #[cfg(not(target_os = "windows"))]
        {
            let mut process = std::process::Command::new("sh");
            process.arg("-c").arg(command);
            process
        }

        #[cfg(target_os = "windows")]
        {
            let mut process = std::process::Command::new("cmd.exe");
            process.arg("/C").arg(command);
            process
        }
    }

    /// Creates a native runtime rooted at `workspace`.
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            workspace,
            temp_dir: std::env::temp_dir(),
            insecure: false,
            background_progress_interval: BACKGROUND_PROGRESS_INTERVAL,
            auto_background_after: AUTO_BACKGROUND_AFTER,
        }
    }

    /// Overrides the temporary directory used for raw output files.
    pub fn temp_dir(self, temp_dir: PathBuf) -> Self {
        Self { temp_dir, ..self }
    }

    /// Allows commands to inherit the process environment.
    pub fn insecure(self) -> Self {
        Self {
            insecure: true,
            ..self
        }
    }

    /// Sets how often background command progress is emitted.
    pub fn background_progress_interval(self, interval: std::time::Duration) -> Self {
        Self {
            background_progress_interval: interval,
            ..self
        }
    }

    /// Sets the foreground duration after which a command moves to background mode.
    pub fn auto_background_after(self, interval: std::time::Duration) -> Self {
        Self {
            auto_background_after: interval,
            ..self
        }
    }

    /// Executes a prepared native command and normalizes its output.
    pub async fn execute_command(
        &self,
        ctx: BaseCtx,
        tool_name: &str,
        command: std::process::Command,
        envs: HashMap<String, String>,
        args: Option<ExecArgs>,
    ) -> Result<ExecOutput, BoxError> {
        let args = args.unwrap_or_default();
        let hook = ctx.get_state::<ShellToolHook>();
        let workspace = ctx
            .meta()
            .get_extra_as::<String>("workspace")
            .map(PathBuf::from)
            .map(Cow::Owned)
            .unwrap_or_else(|| Cow::Borrowed(&self.workspace));
        let workspace_str = workspace.to_string_lossy().to_string();

        let mut cmd = Command::from(command);
        if !self.insecure {
            cmd.env_clear();
        }

        cmd.envs(envs);
        cmd.current_dir(workspace.as_ref());
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) => {
                return Ok(ExecOutput {
                    workspace: Some(workspace_str),
                    stderr: Some(format!("Failed to spawn process: {err}")),
                    ..Default::default()
                });
            }
        };
        let pid = child.id();
        let stdout = std::sync::Arc::new(TokioMutex::new(StreamBuffer::default()));
        let stderr = std::sync::Arc::new(TokioMutex::new(StreamBuffer::default()));
        let stdout_reader = spawn_output_reader(child.stdout.take(), stdout.clone());
        let stderr_reader = spawn_output_reader(child.stderr.take(), stderr.clone());
        let mut running = RunningProcess {
            child,
            stdout,
            stderr,
            stdout_reader,
            stderr_reader,
        };

        if !args.background {
            let status = {
                let wait = running.child.wait();
                tokio::pin!(wait);
                let auto_background = tokio::time::sleep(self.auto_background_after);
                tokio::pin!(auto_background);

                tokio::select! {
                    status = &mut wait => Some(status),
                    _ = &mut auto_background => None,
                }
            };

            if let Some(status) = status {
                return Ok(finalize_process_output(
                    pid,
                    &workspace_str,
                    status,
                    running.stdout,
                    running.stderr,
                    running.stdout_reader,
                    running.stderr_reader,
                    &self.temp_dir,
                )
                .await);
            }
        }

        let auto_started = !args.background;
        let task_id = format!("{}:{}", tool_name, Xid::new());
        let start_message = if auto_started {
            format!("Command is still running and was moved to background with task ID {task_id}")
        } else {
            format!("Background process started with task ID {task_id}")
        };
        // No `exit_status`: the process has not exited yet, and reporting a default success
        // status here misleads callers into treating the command as completed.
        let exec_output = ExecOutput {
            workspace: Some(workspace_str.clone()),
            process_id: pid,
            stdout: Some(start_message),
            ..Default::default()
        };
        let json_hook = ctx.get_state::<DynToolJsonHook>();
        if let Some(hook) = &json_hook {
            hook.on_background_start(&ctx, &task_id, json!(&args)).await;
        } else if let Some(hook) = &hook {
            hook.on_background_start(&ctx, &task_id, &args).await;
        }

        {
            let temp_dir = self.temp_dir.clone();
            let background_progress_interval = self.background_progress_interval;
            tokio::spawn(async move {
                let RunningProcess {
                    mut child,
                    stdout,
                    stderr,
                    stdout_reader,
                    stderr_reader,
                } = running;
                let mut stdout_progress = ProgressStreamState::default();
                let mut stderr_progress = ProgressStreamState::default();
                let mut interval = tokio::time::interval(background_progress_interval);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                interval.tick().await;

                let cancellation = ctx.cancellation_token();
                // `Child::wait` is cancel safe, so a fresh wait future per iteration is fine.
                let status = loop {
                    tokio::select! {
                        status = child.wait() => break status,
                        _ = cancellation.cancelled() => {
                            // The engine or request was cancelled; do not leave the process behind.
                            let _ = child.start_kill();
                            break child.wait().await;
                        }
                        _ = interval.tick() => {
                            if let Some((stdout_chunk, stderr_chunk)) = collect_progress_output(
                                &stdout,
                                &stderr,
                                &mut stdout_progress,
                                &mut stderr_progress,
                            ).await {
                                let exec_output = output_chunks_to_exec_output(
                                    pid,
                                    &workspace_str,
                                    stdout_chunk,
                                    stderr_chunk,
                                );
                                emit_background_progress(
                                    &ctx,
                                    &task_id,
                                    exec_output,
                                    json_hook.as_ref(),
                                    hook.as_ref(),
                                ).await;
                            }
                        }
                    }
                };

                let (final_progress, exec_output) = finalize_process_output_with_final_progress(
                    pid,
                    &workspace_str,
                    status,
                    stdout,
                    stderr,
                    stdout_reader,
                    stderr_reader,
                    &temp_dir,
                    &mut stdout_progress,
                    &mut stderr_progress,
                )
                .await;

                if let Some(exec_output) = final_progress {
                    emit_background_progress(
                        &ctx,
                        &task_id,
                        exec_output,
                        json_hook.as_ref(),
                        hook.as_ref(),
                    )
                    .await;
                }

                emit_background_end(
                    &ctx,
                    task_id,
                    exec_output,
                    json_hook.as_ref(),
                    hook.as_ref(),
                )
                .await;
            });
        }

        Ok(exec_output)
    }
}

#[async_trait]
impl Executor for NativeRuntime {
    fn name(&self) -> &str {
        // Runtime identity, surfaced in diagnostics and used by `ShellTool` to gate
        // host-environment injection (`collect_shell_env_vars`). This must stay "native";
        // the background task-id prefix is the *tool* name and is passed in separately below.
        "native"
    }

    fn workspace(&self) -> &PathBuf {
        &self.workspace
    }

    fn shell(&self) -> &str {
        #[cfg(not(target_os = "windows"))]
        {
            "sh"
        }

        #[cfg(target_os = "windows")]
        {
            "cmd.exe"
        }
    }

    async fn execute(
        &self,
        ctx: BaseCtx,
        input: ExecArgs,
        envs: HashMap<String, String>,
    ) -> Result<ExecOutput, BoxError> {
        let cmd = Self::build_shell_command(&input.command);
        // Background task IDs are prefixed with the tool name (not the runtime name).
        self.execute_command(ctx, super::ShellTool::NAME, cmd, envs, Some(input))
            .await
    }
}

#[derive(Default)]
struct ProgressStreamState {
    sent_len: usize,
    terminal: TerminalProgressState,
}

#[derive(Default)]
struct TerminalProgressState {
    lines: Vec<String>,
    cursor_row: usize,
    cursor: usize,
    rewrite_mode: bool,
    completed_lines: Vec<String>,
    dirty_rows: BTreeSet<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    Plain,
    Rewrite,
}

impl ProgressStreamState {
    /// `sent_len` is an absolute stream offset, so progress stays consistent even after the
    /// buffer dropped old bytes from its head.
    fn next_output(&mut self, output: &StreamBuffer) -> Option<String> {
        if output.total_len() <= self.sent_len {
            return None;
        }

        if self.sent_len < output.trimmed {
            // The unsent head was dropped by the buffer cap; continue from what is left.
            self.sent_len = output.trimmed;
        }

        let unread = &output.data[self.sent_len - output.trimmed..];
        let readable_len = complete_shell_output_prefix_len(unread);
        if readable_len == 0 {
            return None;
        }

        self.sent_len += readable_len;
        let text = decode_shell_output(&unread[..readable_len]);
        self.terminal.render(&text).map(cap_progress_chunk)
    }
}

/// Keeps the newest part of an oversized progress chunk so hooks are never flooded.
fn cap_progress_chunk(text: String) -> String {
    if text.len() <= MAX_PROGRESS_CHUNK_BYTES {
        return text;
    }

    let mut start = text.len() - MAX_PROGRESS_CHUNK_BYTES;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!(
        "[progress truncated to the last {} bytes]\n{}",
        text.len() - start,
        &text[start..]
    )
}

impl TerminalProgressState {
    fn render(&mut self, text: &str) -> Option<String> {
        if has_rewrite_control(text) {
            self.rewrite_mode = true;
        }
        let mode = if self.rewrite_mode {
            ProgressMode::Rewrite
        } else {
            ProgressMode::Plain
        };

        self.apply_text(text, mode);
        match mode {
            ProgressMode::Plain => self.take_completed_lines(),
            ProgressMode::Rewrite => self.take_dirty_lines(),
        }
    }

    fn apply_text(&mut self, text: &str, mode: ProgressMode) {
        self.ensure_cursor_line();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '\r' => self.cursor = 0,
                '\n' => self.newline(mode),
                '\x08' => self.move_cursor_left(),
                '\x1b' => {
                    self.apply_escape_sequence(&mut chars);
                }
                _ => self.write_char(ch),
            }
        }
    }

    fn take_completed_lines(&mut self) -> Option<String> {
        if self.completed_lines.is_empty() {
            return None;
        }
        let lines = std::mem::take(&mut self.completed_lines);
        let output = lines.join("\n");
        (!output.is_empty()).then_some(output)
    }

    fn take_dirty_lines(&mut self) -> Option<String> {
        if self.dirty_rows.is_empty() {
            return None;
        }
        let rows = std::mem::take(&mut self.dirty_rows);
        let lines = rows
            .into_iter()
            .filter_map(|row| self.lines.get(row))
            .map(|line| line.trim_end_matches(' '))
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();

        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    }

    fn newline(&mut self, mode: ProgressMode) {
        if mode == ProgressMode::Plain {
            let line = self.current_line().trim_end_matches(' ').to_string();
            self.completed_lines.push(line);
        } else {
            self.mark_dirty();
        }
        self.cursor_row += 1;
        self.cursor = 0;
        self.ensure_cursor_line();
    }

    fn write_char(&mut self, ch: char) {
        self.ensure_cursor_line();
        if self.cursor >= self.lines[self.cursor_row].len() {
            // Cap pathological single-line streams; the final output still uses the raw bytes,
            // this only bounds the progress rendering state.
            if self.lines[self.cursor_row].len() >= MAX_TERMINAL_LINE_BYTES {
                return;
            }
            self.lines[self.cursor_row].push(ch);
            self.cursor = self.lines[self.cursor_row].len();
            self.mark_dirty();
            return;
        }

        let end = self.lines[self.cursor_row][self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(idx, _)| self.cursor + idx)
            .unwrap_or(self.lines[self.cursor_row].len());
        let mut buf = [0; 4];
        self.lines[self.cursor_row].replace_range(self.cursor..end, ch.encode_utf8(&mut buf));
        self.cursor += ch.len_utf8();
        self.mark_dirty();
    }

    fn move_cursor_left(&mut self) {
        self.move_cursor_left_by(1);
    }

    fn move_cursor_left_by(&mut self, count: usize) {
        self.ensure_cursor_line();
        for _ in 0..count {
            if self.cursor == 0 {
                return;
            }
            self.cursor = self.lines[self.cursor_row][..self.cursor]
                .char_indices()
                .next_back()
                .map(|(idx, _)| idx)
                .unwrap_or(0);
        }
    }

    fn move_cursor_right_by(&mut self, count: usize) {
        self.ensure_cursor_line();
        for _ in 0..count {
            if self.cursor >= self.lines[self.cursor_row].len() {
                return;
            }
            self.cursor = self.lines[self.cursor_row][self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(idx, _)| self.cursor + idx)
                .unwrap_or(self.lines[self.cursor_row].len());
        }
    }

    fn move_cursor_up_by(&mut self, count: usize) {
        self.cursor_row = self.cursor_row.saturating_sub(count);
        self.clamp_cursor();
    }

    fn move_cursor_down_by(&mut self, count: usize) {
        self.cursor_row = self.cursor_row.saturating_add(count);
        self.ensure_cursor_line();
        self.clamp_cursor();
    }

    fn set_cursor_column(&mut self, column: usize) {
        self.ensure_cursor_line();
        // Clamp so untrusted escape sequences cannot allocate huge padding strings.
        let target_column = column.saturating_sub(1).min(MAX_TERMINAL_COLUMNS);
        let char_count = self.lines[self.cursor_row].chars().count();
        if target_column > char_count {
            self.lines[self.cursor_row].extend(std::iter::repeat_n(
                ' ',
                target_column.saturating_sub(char_count),
            ));
            self.mark_dirty();
        }
        self.cursor = byte_index_for_char_column(&self.lines[self.cursor_row], target_column);
    }

    fn set_cursor_position(&mut self, row: usize, column: usize) {
        self.cursor_row = row.saturating_sub(1);
        self.ensure_cursor_line();
        self.set_cursor_column(column);
    }

    fn clear_from_cursor(&mut self) {
        self.ensure_cursor_line();
        self.lines[self.cursor_row].truncate(self.cursor);
        self.mark_dirty();
    }

    fn clear_to_cursor(&mut self) {
        self.ensure_cursor_line();
        if self.cursor == 0 {
            return;
        }
        self.lines[self.cursor_row].replace_range(0..self.cursor, "");
        self.cursor = 0;
        self.mark_dirty();
    }

    fn clear_line(&mut self) {
        self.ensure_cursor_line();
        self.lines[self.cursor_row].clear();
        self.cursor = 0;
        self.mark_dirty();
    }

    fn clear_screen(&mut self) {
        self.lines.clear();
        self.cursor_row = 0;
        self.cursor = 0;
        self.completed_lines.clear();
        self.dirty_rows.clear();
        self.ensure_cursor_line();
    }

    fn apply_escape_sequence<I>(&mut self, chars: &mut std::iter::Peekable<I>)
    where
        I: Iterator<Item = char>,
    {
        match chars.peek() {
            Some('[') => {
                chars.next();
                let mut params = String::new();
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        self.apply_csi_sequence(&params, ch);
                        break;
                    }
                    params.push(ch);
                }
            }
            Some(']') => {
                chars.next();
                while let Some(ch) = chars.next() {
                    if ch == '\x07' {
                        break;
                    }
                    if ch == '\x1b' && matches!(chars.peek(), Some('\\')) {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    fn apply_csi_sequence(&mut self, params: &str, command: char) {
        let values = csi_params(params);
        match command {
            'A' => self.move_cursor_up_by(csi_param_or(&values, 0, 1)),
            'B' => self.move_cursor_down_by(csi_param_or(&values, 0, 1)),
            'C' => self.move_cursor_right_by(csi_param_or(&values, 0, 1)),
            'D' => self.move_cursor_left_by(csi_param_or(&values, 0, 1)),
            'G' => self.set_cursor_column(csi_param_or(&values, 0, 1)),
            'H' | 'f' => {
                self.set_cursor_position(csi_param_or(&values, 0, 1), csi_param_or(&values, 1, 1))
            }
            'J' if csi_param_or(&values, 0, 0) == 2 => self.clear_screen(),
            'K' => match csi_param_or(&values, 0, 0) {
                0 => self.clear_from_cursor(),
                1 => self.clear_to_cursor(),
                2 => self.clear_line(),
                _ => {}
            },
            _ => {}
        }
    }

    fn ensure_cursor_line(&mut self) {
        // Clamp before materializing rows so untrusted cursor escapes (e.g. `ESC[2000000000B`)
        // cannot allocate unbounded numbers of lines.
        if self.cursor_row >= MAX_TERMINAL_ROWS * 2 {
            self.cursor_row = MAX_TERMINAL_ROWS * 2 - 1;
        }
        while self.lines.len() <= self.cursor_row {
            self.lines.push(String::new());
        }
        self.trim_scrollback();
    }

    /// Drops the oldest rows once the screen exceeds [`MAX_TERMINAL_ROWS`], rebasing the cursor
    /// and dirty-row indexes. Long-running plain output would otherwise keep every printed line
    /// in memory for the lifetime of the process.
    fn trim_scrollback(&mut self) {
        if self.lines.len() <= MAX_TERMINAL_ROWS {
            return;
        }

        // Drop down to 3/4 of the cap so the drain cost is amortized across many lines.
        let excess = self.lines.len() - MAX_TERMINAL_ROWS / 4 * 3;
        self.lines.drain(..excess);
        self.cursor_row = self.cursor_row.saturating_sub(excess);
        self.dirty_rows = std::mem::take(&mut self.dirty_rows)
            .into_iter()
            .filter_map(|row| row.checked_sub(excess))
            .collect();
    }

    fn clamp_cursor(&mut self) {
        self.ensure_cursor_line();
        if self.cursor > self.lines[self.cursor_row].len() {
            self.cursor = self.lines[self.cursor_row].len();
        }
        while !self.lines[self.cursor_row].is_char_boundary(self.cursor) && self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    fn current_line(&mut self) -> &str {
        self.ensure_cursor_line();
        &self.lines[self.cursor_row]
    }

    fn mark_dirty(&mut self) {
        self.dirty_rows.insert(self.cursor_row);
    }
}

fn spawn_output_reader<R>(
    reader: Option<R>,
    output: OutputBuffer,
) -> tokio::task::JoinHandle<std::io::Result<()>>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let Some(mut reader) = reader else {
            return Ok(());
        };
        let mut chunk = [0; OUTPUT_READ_CHUNK_BYTES];
        loop {
            let len = reader.read(&mut chunk).await?;
            if len == 0 {
                return Ok(());
            }
            output.lock().await.append(&chunk[..len]);
        }
    })
}

#[allow(clippy::too_many_arguments)]
async fn finalize_process_output(
    process_id: Option<u32>,
    workspace: &str,
    status: std::io::Result<ExitStatus>,
    stdout: OutputBuffer,
    stderr: OutputBuffer,
    stdout_reader: OutputReaderHandle,
    stderr_reader: OutputReaderHandle,
    temp_dir: &Path,
) -> ExecOutput {
    finalize_process_output_with_final_progress(
        process_id,
        workspace,
        status,
        stdout,
        stderr,
        stdout_reader,
        stderr_reader,
        temp_dir,
        &mut ProgressStreamState::default(),
        &mut ProgressStreamState::default(),
    )
    .await
    .1
}

#[allow(clippy::too_many_arguments)]
async fn finalize_process_output_with_final_progress(
    process_id: Option<u32>,
    workspace: &str,
    status: std::io::Result<ExitStatus>,
    stdout: OutputBuffer,
    stderr: OutputBuffer,
    stdout_reader: OutputReaderHandle,
    stderr_reader: OutputReaderHandle,
    temp_dir: &Path,
    stdout_progress: &mut ProgressStreamState,
    stderr_progress: &mut ProgressStreamState,
) -> (Option<ExecOutput>, ExecOutput) {
    // Await both readers concurrently: each can block up to `OUTPUT_READER_GRACE` when a
    // descendant keeps the pipe open, so joining avoids doubling that wait.
    let (stdout_read_error, stderr_read_error) = tokio::join!(
        output_reader_error(stdout_reader, "stdout"),
        output_reader_error(stderr_reader, "stderr"),
    );
    let final_progress =
        collect_progress_output(&stdout, &stderr, stdout_progress, stderr_progress)
            .await
            .map(|(stdout_chunk, stderr_chunk)| {
                output_chunks_to_exec_output(process_id, workspace, stdout_chunk, stderr_chunk)
            });
    let stdout_bytes = std::mem::take(&mut *stdout.lock().await).into_bytes("stdout");
    let mut stderr_bytes = std::mem::take(&mut *stderr.lock().await).into_bytes("stderr");
    if let Some(err) = stdout_read_error {
        append_output_read_error(&mut stderr_bytes, err);
    }
    if let Some(err) = stderr_read_error {
        append_output_read_error(&mut stderr_bytes, err);
    }

    match status {
        Ok(status) => {
            let mut exec_output = ExecOutput::from_output(
                process_id,
                Some(Output {
                    status,
                    stdout: stdout_bytes,
                    stderr: stderr_bytes,
                }),
                temp_dir,
            )
            .await;
            exec_output.workspace = Some(workspace.to_string());
            (final_progress, exec_output)
        }
        Err(err) => {
            let mut error = format!("Failed to execute process: {err}").into_bytes();
            if !stderr_bytes.is_empty() {
                error.push(b'\n');
                error.extend_from_slice(&stderr_bytes);
            }
            (
                final_progress,
                output_bytes_to_exec_output(process_id, workspace, stdout_bytes, error),
            )
        }
    }
}

async fn collect_progress_output(
    stdout: &OutputBuffer,
    stderr: &OutputBuffer,
    stdout_progress: &mut ProgressStreamState,
    stderr_progress: &mut ProgressStreamState,
) -> Option<(String, String)> {
    let stdout_chunk = {
        let stdout = stdout.lock().await;
        stdout_progress.next_output(&stdout).unwrap_or_default()
    };

    let stderr_chunk = {
        let stderr = stderr.lock().await;
        stderr_progress.next_output(&stderr).unwrap_or_default()
    };

    if stdout_chunk.is_empty() && stderr_chunk.is_empty() {
        None
    } else {
        Some((stdout_chunk, stderr_chunk))
    }
}

fn has_rewrite_control(text: &str) -> bool {
    if text.contains(['\r', '\x08']) {
        return true;
    }

    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' || !matches!(chars.peek(), Some('[')) {
            continue;
        }
        chars.next();
        for ch in chars.by_ref() {
            if !('@'..='~').contains(&ch) {
                continue;
            }
            if matches!(ch, 'A' | 'B' | 'C' | 'D' | 'G' | 'H' | 'J' | 'K' | 'f') {
                return true;
            }
            break;
        }
    }

    false
}

fn byte_index_for_char_column(text: &str, column: usize) -> usize {
    text.char_indices()
        .nth(column)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn csi_params(params: &str) -> Vec<usize> {
    params
        .split(';')
        .filter_map(|part| {
            let digits = part
                .chars()
                .filter(|ch| ch.is_ascii_digit())
                .collect::<String>();
            if digits.is_empty() {
                None
            } else {
                digits.parse().ok()
            }
        })
        .collect()
}

fn csi_param_or(values: &[usize], index: usize, default: usize) -> usize {
    match values.get(index).copied() {
        Some(0) if default != 0 => default,
        Some(value) => value,
        None => default,
    }
}

async fn output_reader_error(
    mut handle: tokio::task::JoinHandle<std::io::Result<()>>,
    stream_name: &str,
) -> Option<String> {
    match tokio::time::timeout(OUTPUT_READER_GRACE, &mut handle).await {
        Ok(Ok(Ok(()))) => None,
        Ok(Ok(Err(err))) => Some(format!("Failed to read background {stream_name}: {err}")),
        Ok(Err(err)) => Some(format!(
            "Failed to join background {stream_name} reader: {err}"
        )),
        Err(_) => {
            // A descendant process inherited the pipe and keeps it open; use what was captured
            // instead of waiting indefinitely for EOF.
            handle.abort();
            Some(format!(
                "{stream_name} stayed open after the process exited (likely inherited by a background descendant); captured output may be incomplete"
            ))
        }
    }
}

fn append_output_read_error(stderr: &mut Vec<u8>, err: String) {
    if !stderr.is_empty() && !stderr.ends_with(b"\n") {
        stderr.push(b'\n');
    }
    stderr.extend_from_slice(err.as_bytes());
}

fn output_chunks_to_exec_output(
    process_id: Option<u32>,
    workspace: &str,
    stdout: String,
    stderr: String,
) -> ExecOutput {
    ExecOutput {
        workspace: Some(workspace.to_string()),
        process_id,
        stdout: (!stdout.is_empty()).then_some(stdout),
        stderr: (!stderr.is_empty()).then_some(stderr),
        ..Default::default()
    }
}

fn output_bytes_to_exec_output(
    process_id: Option<u32>,
    workspace: &str,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
) -> ExecOutput {
    output_chunks_to_exec_output(
        process_id,
        workspace,
        decode_shell_output(&stdout),
        decode_shell_output(&stderr),
    )
}

async fn emit_background_progress(
    ctx: &BaseCtx,
    task_id: &str,
    output: ExecOutput,
    json_hook: Option<&DynToolJsonHook>,
    hook: Option<&ShellToolHook>,
) {
    if let Some(hook) = json_hook {
        hook.on_background_progress(ctx, task_id.to_string(), ToolOutput::new(json!(output)))
            .await;
        return;
    }
    if let Some(hook) = hook {
        hook.on_background_progress(ctx, task_id.to_string(), ToolOutput::new(output))
            .await;
    }
}

async fn emit_background_end(
    ctx: &BaseCtx,
    task_id: String,
    output: ExecOutput,
    json_hook: Option<&DynToolJsonHook>,
    hook: Option<&ShellToolHook>,
) {
    if let Some(hook) = json_hook {
        hook.on_background_end(ctx, task_id, ToolOutput::new(json!(output)))
            .await;
        return;
    }
    if let Some(hook) = hook {
        hook.on_background_end(ctx, task_id, ToolOutput::new(output))
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use std::{
        path::Path,
        sync::{Arc, Mutex},
        time::Duration,
    };
    use tokio::sync::{mpsc, oneshot};

    struct TestTempDir(PathBuf);

    impl TestTempDir {
        async fn new(prefix: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("{prefix}-{:016x}", rand::random::<u64>()));
            tokio::fs::create_dir_all(&path).await.unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        async fn create_dir(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            tokio::fs::create_dir_all(&path).await.unwrap();
            path
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[allow(clippy::type_complexity)]
    struct TestHook {
        sender: Mutex<Option<oneshot::Sender<(String, ToolOutput<ExecOutput>)>>>,
    }

    impl TestHook {
        fn new(sender: oneshot::Sender<(String, ToolOutput<ExecOutput>)>) -> Self {
            Self {
                sender: Mutex::new(Some(sender)),
            }
        }
    }

    #[async_trait]
    impl ToolHook<ExecArgs, ExecOutput> for TestHook {
        async fn on_background_end(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            output: ToolOutput<ExecOutput>,
        ) {
            if let Some(sender) = self.sender.lock().unwrap().take() {
                let _ = sender.send((task_id, output));
            }
        }
    }

    #[allow(clippy::type_complexity)]
    struct ProgressHook {
        progress_sender: mpsc::UnboundedSender<(String, ToolOutput<ExecOutput>)>,
        end_sender: Mutex<Option<oneshot::Sender<(String, ToolOutput<ExecOutput>)>>>,
    }

    impl ProgressHook {
        fn new(
            progress_sender: mpsc::UnboundedSender<(String, ToolOutput<ExecOutput>)>,
            end_sender: oneshot::Sender<(String, ToolOutput<ExecOutput>)>,
        ) -> Self {
            Self {
                progress_sender,
                end_sender: Mutex::new(Some(end_sender)),
            }
        }
    }

    #[async_trait]
    impl ToolHook<ExecArgs, ExecOutput> for ProgressHook {
        async fn on_background_progress(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            output: ToolOutput<ExecOutput>,
        ) {
            let _ = self.progress_sender.send((task_id, output));
        }

        async fn on_background_end(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            output: ToolOutput<ExecOutput>,
        ) {
            if let Some(sender) = self.end_sender.lock().unwrap().take() {
                let _ = sender.send((task_id, output));
            }
        }
    }

    fn buf(bytes: &[u8]) -> StreamBuffer {
        StreamBuffer::from_bytes(bytes.to_vec())
    }

    fn foreground_command(runtime: &NativeRuntime, env_name: &str, output_file: &str) -> String {
        match runtime.shell() {
            "cmd.exe" => format!(
                "<nul set /p =%{env_name}% > {output_file} & <nul set /p =done & echo warn 1>&2"
            ),
            _ => format!(
                "printf '%s' \"${env_name}\" > {output_file}; printf '%s' 'done'; printf '%s' 'warn' >&2"
            ),
        }
    }

    fn windows_ping_delay_command() -> String {
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
        let system_root = system_root.trim_end_matches(['\\', '/']);
        format!(r"{system_root}\System32\ping.exe 127.0.0.1 -n 2 > nul")
    }

    fn background_command(runtime: &NativeRuntime) -> String {
        match runtime.shell() {
            "cmd.exe" => format!(
                "{} & <nul set /p =bg-out & echo bg-err 1>&2",
                windows_ping_delay_command()
            ),
            _ => "sleep 0.2; printf '%s' 'bg-out'; printf '%s' 'bg-err' >&2".to_string(),
        }
    }

    fn background_progress_command(runtime: &NativeRuntime) -> String {
        match runtime.shell() {
            "cmd.exe" => format!(
                "echo progress-out & echo progress-err 1>&2 & {} & <nul set /p =done",
                windows_ping_delay_command()
            ),
            _ => "printf '%s\n' 'progress-out'; printf '%s\n' 'progress-err' >&2; sleep 0.5; printf '%s' 'done'".to_string(),
        }
    }

    fn short_background_progress_command(runtime: &NativeRuntime) -> String {
        match runtime.shell() {
            "cmd.exe" => "echo progress-out & echo progress-err 1>&2".to_string(),
            _ => "printf '%s\n' 'progress-out'; printf '%s\n' 'progress-err' >&2".to_string(),
        }
    }

    #[test]
    fn progress_stream_waits_for_complete_utf8_sequence() {
        let mut state = ProgressStreamState::default();
        let mut output = vec![0xe4, 0xb8];

        assert_eq!(state.next_output(&buf(&output)), None);

        output.push(0xad);
        assert_eq!(state.next_output(&buf(&output)), None);

        output.push(b'\n');
        assert_eq!(state.next_output(&buf(&output)).as_deref(), Some("中"));
    }

    #[test]
    fn progress_stream_emits_complete_plain_lines() {
        let mut state = ProgressStreamState::default();
        let mut output = b"line 1\npartial".to_vec();

        assert_eq!(state.next_output(&buf(&output)).as_deref(), Some("line 1"));

        output.extend_from_slice(b" line\n");
        assert_eq!(
            state.next_output(&buf(&output)).as_deref(),
            Some("partial line")
        );
    }

    #[test]
    fn progress_stream_normalizes_rewritten_terminal_line() {
        let mut state = ProgressStreamState::default();

        assert_eq!(
            state.next_output(&buf(b"10%\r20%\r100%")).as_deref(),
            Some("100%")
        );
    }

    #[test]
    fn progress_stream_keeps_rewrite_mode_across_ticks() {
        let mut state = ProgressStreamState::default();
        let mut output = b"10%\r".to_vec();

        assert_eq!(state.next_output(&buf(&output)).as_deref(), Some("10%"));

        output.extend_from_slice(b"20%");
        assert_eq!(state.next_output(&buf(&output)).as_deref(), Some("20%"));
    }

    #[test]
    fn progress_stream_keeps_colored_plain_output_line_based() {
        let mut state = ProgressStreamState::default();
        let mut output = b"\x1b[31mred\x1b[0m".to_vec();

        assert_eq!(state.next_output(&buf(&output)), None);

        output.push(b'\n');
        assert_eq!(state.next_output(&buf(&output)).as_deref(), Some("red"));
    }

    #[test]
    fn progress_stream_handles_ansi_clear_line() {
        let mut state = ProgressStreamState::default();

        assert_eq!(
            state.next_output(&buf(b"abcdef\rxy\x1b[K")).as_deref(),
            Some("xy")
        );
    }

    #[test]
    fn progress_stream_handles_backspace_on_utf8_character() {
        let mut state = ProgressStreamState::default();

        assert_eq!(
            state.next_output(&buf("中\x08文".as_bytes())).as_deref(),
            Some("文")
        );
    }

    #[test]
    fn progress_stream_reports_all_changed_visible_progress_lines() {
        let mut state = ProgressStreamState::default();

        assert_eq!(
            state
                .next_output(&buf(
                    b"file-a 10%\nfile-b 20%\x1b[1A\rfile-a 90%\x1b[1B\rfile-b 80%"
                ))
                .as_deref(),
            Some("file-a 90%\nfile-b 80%")
        );
    }

    #[test]
    fn stream_buffer_caps_memory_and_progress_survives_trimming() {
        let mut buffer = StreamBuffer::default();
        let chunk = vec![b'a'; 1024 * 1024];
        for _ in 0..12 {
            buffer.append(&chunk);
        }
        assert!(buffer.data.len() <= MAX_STREAM_BUFFER_BYTES);
        assert_eq!(buffer.total_len(), 12 * 1024 * 1024);
        assert!(buffer.trimmed > 0);

        // Progress that fell behind the trimmed head skips forward instead of panicking.
        let mut state = ProgressStreamState::default();
        let mut buffer = StreamBuffer::default();
        buffer.append(b"first\n");
        assert_eq!(state.next_output(&buffer).as_deref(), Some("first"));
        buffer.trimmed = 100;
        buffer.data = b"later line\n".to_vec();
        assert_eq!(state.next_output(&buffer).as_deref(), Some("later line"));

        // The final bytes carry a marker when the head was dropped.
        let trimmed = StreamBuffer {
            data: b"tail".to_vec(),
            trimmed: 9,
        };
        let bytes = trimmed.into_bytes("stdout");
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("[9 bytes of stdout dropped"));
        assert!(text.ends_with("tail"));
        assert_eq!(
            StreamBuffer::from_bytes(b"plain".to_vec()).into_bytes("stdout"),
            b"plain"
        );
    }

    #[test]
    fn terminal_state_clamps_hostile_cursor_escapes() {
        // Huge cursor-down and absolute-position escapes must not allocate unbounded rows.
        let mut state = ProgressStreamState::default();
        state.next_output(&buf(b"x\x1b[2000000000Bdown\n"));
        assert!(state.terminal.lines.len() <= MAX_TERMINAL_ROWS);

        let mut state = ProgressStreamState::default();
        state.next_output(&buf(b"\x1b[2000000000;2000000000Hfar\n"));
        assert!(state.terminal.lines.len() <= MAX_TERMINAL_ROWS);

        // Huge column targets are clamped instead of materializing gigabytes of padding.
        let mut terminal = TerminalProgressState::default();
        terminal.set_cursor_column(2_000_000_000);
        assert!(terminal.lines[0].len() <= MAX_TERMINAL_COLUMNS);

        // A single line stops growing at the cap; rendering stays bounded.
        let mut terminal = TerminalProgressState::default();
        for _ in 0..(MAX_TERMINAL_LINE_BYTES + 16) {
            terminal.write_char('y');
        }
        assert_eq!(terminal.lines[0].len(), MAX_TERMINAL_LINE_BYTES);

        // Plain-mode scrollback is bounded: old rows are dropped, content still flows.
        let mut state = ProgressStreamState::default();
        let mut buffer = StreamBuffer::default();
        let mut emitted = 0usize;
        for i in 0..(MAX_TERMINAL_ROWS * 2) {
            buffer.append(format!("line {i}\n").as_bytes());
            if let Some(chunk) = state.next_output(&buffer) {
                emitted += chunk.lines().count();
            }
        }
        assert_eq!(emitted, MAX_TERMINAL_ROWS * 2);
        assert!(state.terminal.lines.len() <= MAX_TERMINAL_ROWS);
    }

    #[test]
    fn cap_progress_chunk_keeps_newest_tail() {
        let short = "ok".to_string();
        assert_eq!(cap_progress_chunk(short.clone()), short);

        let long = "异".repeat(MAX_PROGRESS_CHUNK_BYTES);
        let capped = cap_progress_chunk(long);
        assert!(capped.len() <= MAX_PROGRESS_CHUNK_BYTES + 64);
        assert!(capped.starts_with("[progress truncated to the last "));
        assert!(capped.ends_with('异'));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn output_reader_grace_aborts_readers_held_open_by_descendants() {
        // Simulates a pipe kept open by an orphaned descendant: the reader never finishes.
        let stuck_reader = tokio::spawn(async {
            std::future::pending::<()>().await;
            Ok::<(), std::io::Error>(())
        });
        let message = output_reader_error(stuck_reader, "stdout").await.unwrap();
        assert!(message.contains("stayed open after the process exited"));
    }

    #[test]
    fn new_initializes_paths_and_shell() {
        let runtime = NativeRuntime::new(PathBuf::from("/home/anda-native-runtime-tests"));

        assert_eq!(runtime.name(), "native");
        assert_eq!(
            runtime.workspace(),
            &PathBuf::from("/home/anda-native-runtime-tests")
        );
    }

    #[test]
    fn runtime_builders_and_low_level_output_helpers_cover_edge_paths() {
        let temp_dir = PathBuf::from("/tmp/anda-native-custom-temp");
        let runtime = NativeRuntime::new(PathBuf::from("/workspace"))
            .temp_dir(temp_dir.clone())
            .insecure()
            .background_progress_interval(Duration::from_millis(7))
            .auto_background_after(Duration::from_millis(9));

        assert_eq!(runtime.temp_dir, temp_dir);
        assert!(runtime.insecure);
        assert_eq!(
            runtime.background_progress_interval,
            Duration::from_millis(7)
        );
        assert_eq!(runtime.auto_background_after, Duration::from_millis(9));
        assert!(!windows_ping_delay_command().starts_with('"'));
        assert!(windows_ping_delay_command().contains(r"\System32\ping.exe"));

        assert_eq!(complete_shell_output_prefix_len(&[]), 0);
        assert_eq!(complete_shell_output_prefix_len("😀".as_bytes()), 4);
        assert_eq!(complete_shell_output_prefix_len(&[0xf0, 0x9f]), 0);
        assert!(has_rewrite_control("\x1b[2J"));
        assert!(!has_rewrite_control("\x1b[31mred"));
        assert_eq!(byte_index_for_char_column("a中b", 2), "a中".len());
        assert_eq!(csi_params("?25;0;12h"), vec![25, 0, 12]);
        assert_eq!(csi_param_or(&[0], 0, 1), 1);

        let mut stderr = b"first".to_vec();
        append_output_read_error(&mut stderr, "second".to_string());
        assert_eq!(String::from_utf8(stderr).unwrap(), "first\nsecond");

        let output =
            output_bytes_to_exec_output(Some(7), "/workspace", b"out".to_vec(), b"err".to_vec());
        assert_eq!(output.process_id, Some(7));
        assert_eq!(output.workspace.as_deref(), Some("/workspace"));
        assert_eq!(output.stdout.as_deref(), Some("out"));
        assert_eq!(output.stderr.as_deref(), Some("err"));
    }

    #[test]
    fn terminal_state_handles_cursor_motion_clears_screen_and_osc_sequences() {
        let mut terminal = TerminalProgressState::default();

        assert_eq!(
            terminal
                .render("abc\x1b[2GZ\x1b[1Kx\x1b[3G!\x1b]0;title\x07")
                .as_deref(),
            Some("x !")
        );
        assert_eq!(terminal.render("\x1b[2Jfresh").as_deref(), Some("fresh"));

        let mut terminal = TerminalProgressState::default();
        terminal.apply_text("ab\ncd", ProgressMode::Rewrite);
        terminal.set_cursor_position(1, 2);
        terminal.move_cursor_right_by(5);
        terminal.write_char('Z');
        terminal.move_cursor_down_by(3);
        terminal.write_char('x');
        terminal.move_cursor_up_by(2);
        terminal.clear_line();

        assert_eq!(terminal.lines[0], "abZ");
        assert!(terminal.lines.len() >= 4);
        assert_eq!(terminal.lines[3], "x");
        assert!(terminal.dirty_rows.contains(&0));
        assert!(terminal.dirty_rows.contains(&3));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn output_reader_and_finalize_helpers_report_read_join_and_wait_errors() {
        let empty_reader = spawn_output_reader::<tokio::io::Empty>(
            None,
            Arc::new(TokioMutex::new(StreamBuffer::default())),
        );
        assert!(output_reader_error(empty_reader, "stdout").await.is_none());

        let read_error =
            tokio::spawn(async { Err::<(), std::io::Error>(std::io::Error::other("read failed")) });
        assert!(
            output_reader_error(read_error, "stderr")
                .await
                .unwrap()
                .contains("Failed to read background stderr")
        );

        let join_error = tokio::spawn(async {
            panic!("join failed");
            #[allow(unreachable_code)]
            Ok::<(), std::io::Error>(())
        });
        assert!(
            output_reader_error(join_error, "stdout")
                .await
                .unwrap()
                .contains("Failed to join background stdout reader")
        );

        let stdout = Arc::new(TokioMutex::new(StreamBuffer::from_bytes(
            b"stdout".to_vec(),
        )));
        let stderr = Arc::new(TokioMutex::new(StreamBuffer::from_bytes(
            b"stderr".to_vec(),
        )));
        let ok_reader = tokio::spawn(async { Ok::<(), std::io::Error>(()) });
        let ok_reader_2 = tokio::spawn(async { Ok::<(), std::io::Error>(()) });
        let output = finalize_process_output(
            Some(99),
            "/workspace",
            Err(std::io::Error::other("wait failed")),
            stdout,
            stderr,
            ok_reader,
            ok_reader_2,
            Path::new("/tmp"),
        )
        .await;
        assert_eq!(output.process_id, Some(99));
        assert_eq!(output.workspace.as_deref(), Some("/workspace"));
        assert_eq!(output.stdout.as_deref(), Some("stdout"));
        assert!(
            output
                .stderr
                .as_deref()
                .is_some_and(|stderr| stderr.contains("wait failed") && stderr.contains("stderr"))
        );

        let stdout = Arc::new(TokioMutex::new(StreamBuffer::default()));
        let stderr = Arc::new(TokioMutex::new(StreamBuffer::default()));
        let read_error = tokio::spawn(async {
            Err::<(), std::io::Error>(std::io::Error::other("stdout failed"))
        });
        let ok_reader = tokio::spawn(async { Ok::<(), std::io::Error>(()) });
        let output = finalize_process_output(
            Some(100),
            "/workspace",
            Ok(ExitStatus::default()),
            stdout,
            stderr,
            read_error,
            ok_reader,
            Path::new("/tmp"),
        )
        .await;
        assert!(
            output
                .stderr
                .as_deref()
                .is_some_and(|stderr| stderr.contains("stdout failed"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_runs_foreground_command_with_envs_and_workspace() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-native-foreground").await;
        let nested_dir = workspace.create_dir("nested").await;
        let runtime = NativeRuntime::new(nested_dir.clone());
        let env_name = "ANDA_NATIVE_TEST_VALUE";
        let output_file = "env.txt";
        let mut envs = HashMap::new();
        envs.insert(env_name.to_string(), "secret-value".to_string());

        let output = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: foreground_command(&runtime, env_name, output_file),
                    ..Default::default()
                },
                envs,
            )
            .await
            .unwrap();

        let written = tokio::fs::read_to_string(nested_dir.join(output_file))
            .await
            .unwrap();
        assert_eq!(written.trim(), "secret-value");
        assert!(output.process_id.is_some());
        assert!(output.raw_output_path.is_none());
        assert_eq!(output.stdout.as_deref().map(str::trim), Some("done"));
        assert_eq!(output.stderr.as_deref().map(str::trim), Some("warn"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_reports_background_output_via_hook() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-native-background").await;
        let (sender, receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(TestHook::new(sender)));
        ctx.base.set_state(hook);
        let runtime = NativeRuntime::new(workspace.path().to_path_buf());
        let input = ExecArgs {
            command: background_command(&runtime),
            background: true,
            ..Default::default()
        };

        let output = runtime
            .execute(ctx.base, input.clone(), HashMap::new())
            .await
            .unwrap();

        assert!(output.process_id.is_some());
        assert!(output.exit_status.is_none());
        assert!(output.stdout.is_some());
        assert!(output.stderr.is_none());

        let (
            task_id,
            ToolOutput {
                output: hook_output,
                ..
            },
        ) = tokio::time::timeout(Duration::from_secs(5), receiver)
            .await
            .unwrap()
            .unwrap();

        assert!(task_id.contains("shell"));
        assert_eq!(hook_output.process_id, output.process_id);
        assert_eq!(hook_output.stdout.as_deref().map(str::trim), Some("bg-out"));
        assert_eq!(hook_output.stderr.as_deref().map(str::trim), Some("bg-err"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_reports_background_progress_via_hook() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-native-progress").await;
        let (progress_sender, mut progress_receiver) = mpsc::unbounded_channel();
        let (end_sender, end_receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(ProgressHook::new(progress_sender, end_sender)));
        ctx.base.set_state(hook);
        let runtime = NativeRuntime::new(workspace.path().to_path_buf());
        let input = ExecArgs {
            command: background_progress_command(&runtime),
            background: true,
            ..Default::default()
        };

        let output = runtime
            .execute(ctx.base, input.clone(), HashMap::new())
            .await
            .unwrap();

        assert!(output.process_id.is_some());
        assert!(output.exit_status.is_none());
        assert!(output.stdout.is_some());
        assert!(output.stderr.is_none());

        let progress_task_id = tokio::time::timeout(Duration::from_secs(5), async {
            let mut saw_stdout = false;
            let mut saw_stderr = false;
            loop {
                let (
                    task_id,
                    ToolOutput {
                        output: progress_output,
                        ..
                    },
                ) = progress_receiver.recv().await.unwrap();
                assert_eq!(progress_output.process_id, output.process_id);
                assert!(progress_output.exit_status.is_none());
                if progress_output
                    .stdout
                    .as_deref()
                    .is_some_and(|stdout| stdout.contains("progress-out"))
                {
                    saw_stdout = true;
                }
                if progress_output
                    .stderr
                    .as_deref()
                    .is_some_and(|stderr| stderr.contains("progress-err"))
                {
                    saw_stderr = true;
                }
                if saw_stdout && saw_stderr {
                    break task_id;
                }
            }
        })
        .await
        .unwrap();

        let (
            end_task_id,
            ToolOutput {
                output: hook_output,
                ..
            },
        ) = tokio::time::timeout(Duration::from_secs(5), end_receiver)
            .await
            .unwrap()
            .unwrap();

        assert!(progress_task_id.contains("shell"));
        assert_eq!(end_task_id, progress_task_id);
        assert_eq!(hook_output.process_id, output.process_id);
        assert!(
            hook_output
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.contains("progress-out") && stdout.contains("done"))
        );
        assert!(
            hook_output
                .stderr
                .as_deref()
                .is_some_and(|stderr| stderr.contains("progress-err"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_flushes_final_background_progress_before_end() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-native-final-progress").await;
        let (progress_sender, mut progress_receiver) = mpsc::unbounded_channel();
        let (end_sender, end_receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(ProgressHook::new(progress_sender, end_sender)));
        ctx.base.set_state(hook);
        let runtime = NativeRuntime::new(workspace.path().to_path_buf())
            .background_progress_interval(Duration::from_secs(60));
        let input = ExecArgs {
            command: short_background_progress_command(&runtime),
            background: true,
            ..Default::default()
        };

        let output = runtime
            .execute(ctx.base, input.clone(), HashMap::new())
            .await
            .unwrap();

        let (
            progress_task_id,
            ToolOutput {
                output: progress_output,
                ..
            },
        ) = tokio::time::timeout(Duration::from_secs(5), progress_receiver.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(progress_output.process_id, output.process_id);
        assert!(progress_output.exit_status.is_none());
        assert!(
            progress_output
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.contains("progress-out"))
        );
        assert!(
            progress_output
                .stderr
                .as_deref()
                .is_some_and(|stderr| stderr.contains("progress-err"))
        );

        let (
            end_task_id,
            ToolOutput {
                output: hook_output,
                ..
            },
        ) = tokio::time::timeout(Duration::from_secs(5), end_receiver)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(end_task_id, progress_task_id);
        assert_eq!(hook_output.process_id, output.process_id);
        assert!(
            hook_output
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.contains("progress-out"))
        );
        assert!(
            hook_output
                .stderr
                .as_deref()
                .is_some_and(|stderr| stderr.contains("progress-err"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_kills_background_process_on_cancellation() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-native-cancel").await;
        let (sender, receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(TestHook::new(sender)));
        ctx.base.set_state(hook);
        let runtime = NativeRuntime::new(workspace.path().to_path_buf());
        let command = match runtime.shell() {
            "cmd.exe" => {
                let system_root =
                    std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
                let system_root = system_root.trim_end_matches(['\\', '/']);
                format!(r"{system_root}\System32\ping.exe 127.0.0.1 -n 31 > nul")
            }
            _ => "sleep 30".to_string(),
        };

        let output = runtime
            .execute(
                ctx.base.clone(),
                ExecArgs {
                    command,
                    background: true,
                    ..Default::default()
                },
                HashMap::new(),
            )
            .await
            .unwrap();
        assert!(output.exit_status.is_none());

        ctx.base.cancellation_token().cancel();

        let (
            task_id,
            ToolOutput {
                output: hook_output,
                ..
            },
        ) = tokio::time::timeout(Duration::from_secs(5), receiver)
            .await
            .unwrap()
            .unwrap();

        assert!(task_id.contains("shell"));
        assert_eq!(hook_output.process_id, output.process_id);
        let exit_status = hook_output.exit_status.unwrap();
        assert!(!exit_status.contains("exit status: 0"), "{exit_status}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_auto_moves_long_running_foreground_to_background() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-native-auto-background").await;
        let (sender, receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(TestHook::new(sender)));
        ctx.base.set_state(hook);
        let runtime = NativeRuntime::new(workspace.path().to_path_buf())
            .auto_background_after(Duration::from_millis(100))
            .background_progress_interval(Duration::from_millis(100));
        let input = ExecArgs {
            command: background_progress_command(&runtime),
            ..Default::default()
        };

        let output = runtime
            .execute(ctx.base, input.clone(), HashMap::new())
            .await
            .unwrap();

        assert!(output.process_id.is_some());
        assert!(output.exit_status.is_none());
        assert!(
            output
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.contains("moved to background"))
        );
        assert!(output.stderr.is_none());

        let (
            task_id,
            ToolOutput {
                output: hook_output,
                ..
            },
        ) = tokio::time::timeout(Duration::from_secs(5), receiver)
            .await
            .unwrap()
            .unwrap();

        assert!(task_id.contains("shell"));
        assert_eq!(hook_output.process_id, output.process_id);
        assert!(
            hook_output
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.contains("progress-out") && stdout.contains("done"))
        );
        assert!(
            hook_output
                .stderr
                .as_deref()
                .is_some_and(|stderr| stderr.contains("progress-err"))
        );
    }
}
