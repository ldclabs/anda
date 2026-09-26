//! Scoped process sessions with bounded capture, logging, retention and interaction.

use super::*;
use crate::extension::shell::{
    CommandArgs, CommandOutput, CommandState, LogChunk, SessionAction, SessionArgs, SessionInfo,
    SessionLimits, SessionOutput, ShellSessionScope, command::budget, preview_output,
};
use anda_core::CancellationToken;
use parking_lot::Mutex;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncSeekExt, AsyncWrite, AsyncWriteExt},
    sync::Notify,
};

type Input = Box<dyn AsyncWrite + Send + Unpin>;

#[derive(Clone, PartialEq, Eq)]
struct Owner {
    engine: candid::Principal,
    caller: candid::Principal,
    agent: String,
    scope: u128,
}

impl Owner {
    fn for_ctx(ctx: &BaseCtx) -> Result<Self, BoxError> {
        let scope = ctx
            .get_state::<ShellSessionScope>()
            .ok_or("Session shell requires a host-created ShellSessionScope in context state")?;
        Ok(Self {
            engine: *ctx.engine_id(),
            caller: *ctx.caller(),
            agent: ctx.agent.clone(),
            scope: scope.0,
        })
    }
}

#[derive(Clone)]
struct Status {
    state: CommandState,
    code: Option<i32>,
    signal: Option<i32>,
    exit_status: Option<String>,
    finished: Option<Instant>,
}

struct Log {
    file: tokio::fs::File,
    path: PathBuf,
    bytes: usize,
    limit: usize,
    complete: bool,
    /// Stream of the last logged chunk (true for stderr); labels mark only switches.
    stream: Option<bool>,
}

impl Drop for Log {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct Entry {
    id: String,
    owner: Owner,
    workspace: String,
    pid: Option<u32>,
    tty: bool,
    token: CancellationToken,
    status: Mutex<Status>,
    stdout: OutputBuffer,
    stderr: OutputBuffer,
    input: TokioMutex<Option<Input>>,
    interaction: TokioMutex<(usize, usize)>,
    changed: Notify,
    log: Arc<TokioMutex<Log>>,
}

#[derive(Default)]
struct Registry {
    entries: HashMap<String, Arc<Entry>>,
    reserved: std::collections::HashSet<String>,
}

#[derive(Default)]
pub(super) struct SessionStore {
    registry: Mutex<Registry>,
}

impl Drop for SessionStore {
    fn drop(&mut self) {
        for entry in self.registry.get_mut().entries.values() {
            entry.token.cancel();
        }
    }
}

struct Reservation {
    store: Arc<SessionStore>,
    id: String,
}
impl Reservation {
    fn insert(&self, entry: Arc<Entry>) {
        let mut registry = self.store.registry.lock();
        registry.reserved.remove(&self.id);
        registry.entries.insert(self.id.clone(), entry);
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.store.registry.lock().reserved.remove(&self.id);
    }
}

impl SessionStore {
    fn prune(&self, limits: &SessionLimits) {
        self.registry.lock().entries.retain(|_, entry| {
            entry
                .status
                .lock()
                .finished
                .is_none_or(|at| at.elapsed() < limits.retention)
        });
    }

    fn reserve(
        self: &Arc<Self>,
        id: &str,
        limits: &SessionLimits,
    ) -> Result<Reservation, BoxError> {
        self.prune(limits);
        let mut registry = self.registry.lock();
        if registry.entries.len() + registry.reserved.len() >= limits.max_sessions {
            let oldest = registry
                .entries
                .iter()
                .filter_map(|(id, entry)| entry.status.lock().finished.map(|at| (id.clone(), at)))
                .min_by_key(|(_, at)| *at);
            if let Some((id, _)) = oldest {
                registry.entries.remove(&id);
            }
        }
        if registry.entries.len() + registry.reserved.len() >= limits.max_sessions {
            return Err(
                "Shell session limit reached; stop running sessions before starting another".into(),
            );
        }
        registry.reserved.insert(id.to_owned());
        Ok(Reservation {
            store: self.clone(),
            id: id.to_owned(),
        })
    }
}

impl NativeRuntime {
    /// Configures bounded session resources and opt-in interaction capabilities.
    pub fn session_limits(self, limits: SessionLimits) -> Self {
        Self {
            session_limits: limits,
            sessions: Default::default(),
            ..self
        }
    }

    pub(super) async fn start_session(
        &self,
        ctx: BaseCtx,
        args: CommandArgs,
        envs: HashMap<String, String>,
    ) -> Result<CommandOutput, BoxError> {
        let owner = Owner::for_ctx(&ctx)?;
        if args.command.len() > 128 * 1024 || args.env_keys.len() > 256 {
            return Err("Shell request exceeds command or environment-key limits".into());
        }
        let limits = &self.session_limits;
        if !(1..=256).contains(&limits.max_sessions)
            || limits.max_runtime.is_zero()
            || limits.max_runtime > Duration::from_secs(86400)
            || limits.max_log_bytes > 128 * 1024 * 1024
            || limits.max_output_bytes < 128
        {
            return Err("Invalid host session limits".into());
        }
        if args.max_output_bytes.is_some_and(|limit| limit < 128) {
            return Err("max_output_bytes must be at least 128".into());
        }
        if args.stdin && !limits.allow_stdin {
            return Err("Interactive stdin is disabled by the host".into());
        }
        if args.tty && (!limits.allow_pty || !cfg!(unix)) {
            return Err("PTY allocation is disabled or unsupported".into());
        }
        if ctx.cancellation_token().is_cancelled() {
            return Err("shell command cancelled".into());
        }
        let default_workspace = self.requested_workspace(&ctx).await.into_owned();
        let workspace = match &args.cwd {
            Some(path) => {
                fs::resolve_read_path(
                    &self.workspace,
                    &default_workspace.join(path).to_string_lossy(),
                )
                .await?
            }
            None => tokio::fs::canonicalize(default_workspace).await?,
        };
        if !workspace.is_dir() {
            return Err("Shell cwd must be a directory".into());
        }
        let runtime = args
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(limits.max_runtime)
            .min(limits.max_runtime);
        if runtime.is_zero() {
            return Err("timeout_ms must be positive".into());
        }
        let id = format!("shell:{:032x}", rand::random::<u128>());
        let reservation = self.sessions.reserve(&id, limits)?;
        tokio::fs::create_dir_all(&self.temp_dir).await?;
        let log_path = self
            .temp_dir
            .join(format!("anda-shell-session-{}.log", id.replace(':', "-")));
        let mut options = tokio::fs::OpenOptions::new();
        options.create_new(true).read(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let log = Arc::new(TokioMutex::new(Log {
            file: options.open(&log_path).await?,
            path: log_path,
            bytes: 0,
            limit: limits.max_log_bytes,
            complete: true,
            stream: None,
        }));

        let mut command = self.session_shell.command(&args.command);
        if let Some(policy) = &self.sandbox {
            command = policy.wrap(command, &workspace, args.tty)?;
        }
        command.current_dir(&workspace);
        if !self.insecure {
            command.env_clear();
        }
        command.envs(envs);
        let deadline = tokio::time::Instant::now() + runtime;
        let (mut child, stdout_source, stderr_source, input) =
            spawn(command, args.stdin, args.tty)?;
        let pid = child.id();
        let token = ctx.cancellation_token().child_token();
        let entry = Arc::new(Entry {
            id: id.clone(),
            owner,
            workspace: workspace.display().to_string(),
            pid,
            tty: args.tty,
            token,
            status: Mutex::new(Status {
                state: CommandState::Running,
                code: None,
                signal: None,
                exit_status: None,
                finished: None,
            }),
            stdout: Arc::new(TokioMutex::new(StreamBuffer::bounded(1024 * 1024))),
            stderr: Arc::new(TokioMutex::new(StreamBuffer::bounded(1024 * 1024))),
            input: TokioMutex::new(input),
            interaction: TokioMutex::new((0, 0)),
            changed: Notify::new(),
            log,
        });
        reservation.insert(entry.clone());
        let stdout_reader = reader(stdout_source, entry.clone(), false);
        let stderr_reader = reader(stderr_source, entry.clone(), true);
        let guard = ProcessGuard {
            pid,
            readers: [stdout_reader.abort_handle(), stderr_reader.abort_handle()],
            armed: true,
        };
        let hook = ctx.get_state::<ShellToolHook>();
        let json_hook = ctx.get_state::<DynToolJsonHook>();
        let legacy_args = ExecArgs {
            command: args.command.clone(),
            env_keys: args.env_keys.clone(),
            background: args.background,
        };
        let progress_interval = self
            .background_progress_interval
            .max(Duration::from_millis(10));
        let task = entry.clone();
        let wait = if args.background {
            0
        } else {
            args.yield_time_ms.unwrap_or(10_000).min(30_000)
        };
        // Shared by the foreground wait and the supervisor: like the legacy shell,
        // hooks only see commands still running once this wait has elapsed.
        let foreground_deadline = tokio::time::Instant::now() + Duration::from_millis(wait);
        let output_budget = budget(args.max_output_bytes, limits);
        let cancellation = ctx.cancellation_token();
        tokio::spawn(async move {
            let mut guard = guard;
            let mut backgrounded = false;
            let mut interval = tokio::time::interval(progress_interval);
            let mut stdout_progress = ProgressStreamState::default();
            let mut stderr_progress = ProgressStreamState::default();
            let mut state = CommandState::Exited;
            let status = loop {
                tokio::select! {
                    biased;
                    // First, so a command the foreground reported as running always
                    // pairs this start with the end event below.
                    _ = tokio::time::sleep_until(foreground_deadline), if !backgrounded => {
                        backgrounded = true;
                        interval.reset();
                        let handle = BackgroundHandle::new(&task.id, task.token.clone());
                        let start = async {
                            if let Some(hook) = &json_hook {
                                hook.on_background_start(&ctx, handle, json!(&args)).await;
                            } else if let Some(hook) = &hook {
                                hook.on_background_start(&ctx, handle, &legacy_args).await;
                            }
                        };
                        // Hook latency cannot indefinitely delay supervision or cancellation.
                        tokio::select! { biased; _ = tokio::time::timeout(Duration::from_secs(2), start) => (), _ = task.token.cancelled() => (), _ = tokio::time::sleep_until(deadline) => () }
                    }
                    _ = task.token.cancelled() => {
                        state = CommandState::Cancelled;
                        kill_process_group(task.pid); let _ = child.start_kill(); break child.wait().await;
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        state = CommandState::TimedOut;
                        kill_process_group(task.pid); let _ = child.start_kill(); break child.wait().await;
                    }
                    status = child.wait() => {
                        // The shell may exit while background descendants still own its pipes.
                        #[cfg(unix)]
                        kill_process_group(task.pid);
                        break status;
                    },
                    _ = interval.tick(), if backgrounded => {
                        if let Some((stdout, stderr)) = collect_progress_output(&task.stdout, &task.stderr, &mut stdout_progress, &mut stderr_progress).await {
                            let output = output_chunks_to_exec_output(task.pid, &task.workspace, stdout, stderr);
                            let progress = emit_background_progress(&ctx, &task.id, output, json_hook.as_ref(), hook.as_ref());
                            tokio::select! { _ = task.token.cancelled() => (), _ = tokio::time::sleep_until(deadline) => (), _ = tokio::time::timeout(Duration::from_secs(2), progress) => () }
                        }
                    }
                }
            };
            let (stdout_error, stderr_error) = tokio::join!(
                output_reader_error(stdout_reader, "stdout"),
                output_reader_error(stderr_reader, "stderr")
            );
            if stdout_error.is_some() || stderr_error.is_some() {
                task.log.lock().await.complete = false;
                for error in [stdout_error, stderr_error].into_iter().flatten() {
                    task.stderr.lock().await.append(error.as_bytes());
                }
            }
            task.input.lock().await.take();
            let (code, signal, exit_status) = match status {
                Ok(status) => {
                    #[cfg(unix)]
                    let signal = {
                        use std::os::unix::process::ExitStatusExt;
                        status.signal()
                    };
                    #[cfg(not(unix))]
                    let signal = None;
                    (status.code(), signal, Some(status.to_string()))
                }
                Err(error) => {
                    state = CommandState::Failed;
                    task.stderr
                        .lock()
                        .await
                        .append(error.to_string().as_bytes());
                    (None, None, None)
                }
            };
            *task.status.lock() = Status {
                state,
                code,
                signal,
                exit_status,
                finished: Some(Instant::now()),
            };
            task.changed.notify_waiters();
            guard.disarm();
            if !backgrounded {
                // The foreground call already returned this result to the caller.
                return;
            }
            let final_output = snapshot(&task, &mut (0, 0), super::super::MAX_OUTPUT_BYTES).await;
            let end = async {
                let is_error = matches!(
                    final_output.state,
                    CommandState::Failed | CommandState::TimedOut
                )
                .then_some(true);
                if let Some(hook) = &json_hook {
                    hook.on_background_end(
                        &ctx,
                        task.id.clone(),
                        ToolOutput {
                            is_error,
                            ..ToolOutput::new(json!(final_output))
                        },
                    )
                    .await;
                } else if let Some(hook) = &hook {
                    hook.on_background_end(
                        &ctx,
                        task.id.clone(),
                        ToolOutput {
                            is_error,
                            ..ToolOutput::new(final_output.output)
                        },
                    )
                    .await;
                }
            };
            let _ = tokio::time::timeout(Duration::from_secs(2), end).await;
        });
        let mut cursor = entry.interaction.lock().await;
        wait_output(&entry, &cursor, foreground_deadline, true, &cancellation).await?;
        Ok(snapshot(&entry, &mut cursor, output_budget).await)
    }

    pub(super) async fn session_action(
        &self,
        ctx: BaseCtx,
        args: SessionArgs,
    ) -> Result<SessionOutput, BoxError> {
        let owner = Owner::for_ctx(&ctx)?;
        if args.max_output_bytes.is_some_and(|limit| limit < 128) {
            return Err("max_output_bytes must be at least 128".into());
        }
        if args.input.is_some() && args.action != SessionAction::Write {
            return Err("input is accepted only by the write action".into());
        }
        self.sessions.prune(&self.session_limits);
        if args.action == SessionAction::List {
            let registry = self.sessions.registry.lock();
            let entries = &registry.entries;
            let mut sessions = entries
                .values()
                .filter(|entry| entry.owner == owner)
                .map(|entry| SessionInfo {
                    task_id: entry.id.clone(),
                    state: entry.status.lock().state,
                })
                .collect::<Vec<_>>();
            sessions.sort_by(|a, b| a.task_id.cmp(&b.task_id));
            return Ok(SessionOutput {
                sessions,
                ..Default::default()
            });
        }
        let id = args.task_id.as_deref().ok_or("task_id is required")?;
        let entry = self
            .sessions
            .registry
            .lock()
            .entries
            .get(id)
            .filter(|entry| entry.owner == owner)
            .cloned()
            .ok_or("Unknown or inaccessible shell session")?;
        let cancellation = ctx.cancellation_token();
        if cancellation.is_cancelled() {
            return Err("Session operation cancelled".into());
        }
        // Stop must interrupt an outstanding long poll instead of waiting for its lock.
        if args.action == SessionAction::Stop {
            entry.token.cancel();
        }
        let mut cursor = tokio::select! { biased; _ = cancellation.cancelled() => return Err("Session operation cancelled".into()), cursor = entry.interaction.lock() => cursor };
        let limit = budget(args.max_output_bytes, &self.session_limits);
        if args.action == SessionAction::ReadLog {
            let mut log = entry.log.lock().await;
            log.file.flush().await?;
            log.file.seek(std::io::SeekFrom::Start(args.offset)).await?;
            let mut bytes = vec![0; limit / 3];
            let count = log.file.read(&mut bytes).await?;
            log.file.seek(std::io::SeekFrom::End(0)).await?;
            // Leave a character split by the byte budget to the next read; a lone
            // trailing fragment is still returned so reads always make progress.
            let count = match complete_shell_output_prefix_len(&bytes[..count]) {
                0 => count,
                complete => complete,
            };
            bytes.truncate(count);
            return Ok(SessionOutput {
                log: Some(LogChunk {
                    text: decode_shell_output(&bytes),
                    next_offset: args.offset.saturating_add(count as u64),
                    eof: args.offset.saturating_add(count as u64) >= log.bytes as u64,
                    complete: log.complete,
                }),
                ..Default::default()
            });
        }
        match args.action {
            SessionAction::Write => {
                if !self.session_limits.allow_stdin {
                    return Err("Interactive stdin is disabled".into());
                }
                let input = args.input.as_deref().ok_or("input is required for write")?;
                if input.len() > 65536 {
                    return Err("stdin input exceeds 65536 bytes".into());
                }
                let mut stdin = entry.input.lock().await;
                let stdin = stdin.as_mut().ok_or("Session stdin is closed")?;
                let write = async {
                    stdin.write_all(input.as_bytes()).await?;
                    stdin.flush().await
                };
                tokio::select! {
                    _ = cancellation.cancelled() => return Err("Session input cancelled; some bytes may have been written".into()),
                    result = tokio::time::timeout(Duration::from_secs(2), write) => result.map_err(|_| "Session input timed out; some bytes may have been written")??,
                }
            }
            SessionAction::CloseStdin => {
                if entry.tty {
                    return Err("Use the PTY EOF character instead of close_stdin".into());
                }
                entry.input.lock().await.take();
            }
            SessionAction::Stop => entry.token.cancel(),
            SessionAction::Poll => (),
            SessionAction::ReadLog | SessionAction::List => unreachable!(),
        }
        let wait = args.yield_time_ms.unwrap_or(1000).min(30_000);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(wait);
        wait_output(&entry, &cursor, deadline, false, &cancellation).await?;
        Ok(SessionOutput {
            command: Some(snapshot(&entry, &mut cursor, limit).await),
            ..Default::default()
        })
    }
}

type Reader = Box<dyn AsyncRead + Send + Unpin>;

fn spawn(
    mut command: std::process::Command,
    stdin: bool,
    tty: bool,
) -> Result<(Child, Reader, Reader, Option<Input>), BoxError> {
    if tty {
        #[cfg(unix)]
        return super::pty::spawn(command);
        #[cfg(not(unix))]
        return Err("PTY unsupported on this platform".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
        .stdin(if stdin { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut command = Command::from(command);
    command.kill_on_drop(true);
    let mut child = command.spawn()?;
    let stdout = Box::new(child.stdout.take().ok_or("Missing stdout pipe")?);
    let stderr = Box::new(child.stderr.take().ok_or("Missing stderr pipe")?);
    let input = child.stdin.take().map(|input| Box::new(input) as Input);
    Ok((child, stdout, stderr, input))
}

fn reader(mut source: Reader, entry: Arc<Entry>, stderr: bool) -> OutputReaderHandle {
    tokio::spawn(async move {
        let mut bytes = [0; OUTPUT_READ_CHUNK_BYTES];
        loop {
            let count = source.read(&mut bytes).await?;
            if count == 0 {
                break;
            }
            let buffer = if stderr { &entry.stderr } else { &entry.stdout };
            buffer.lock().await.append(&bytes[..count]);
            entry.changed.notify_waiters();
            let mut log = entry.log.lock().await;
            let header = if log.stream == Some(stderr) {
                b"".as_slice()
            } else if stderr {
                b"\n[stderr]\n".as_slice()
            } else {
                b"\n[stdout]\n".as_slice()
            };
            if log.complete {
                let remaining = log.limit.saturating_sub(log.bytes);
                if header.len() + count > remaining {
                    log.complete = false;
                }
                if remaining >= header.len() {
                    let retained = count.min(remaining - header.len());
                    if log.file.write_all(header).await.is_err()
                        || log.file.write_all(&bytes[..retained]).await.is_err()
                    {
                        log.complete = false;
                    } else {
                        log.bytes += header.len() + retained;
                        log.stream = Some(stderr);
                    }
                }
            }
        }
        Ok(())
    })
}

/// Waits until the process finishes or the deadline passes. Unless `until_exit`
/// is set, newly captured output also ends the wait.
async fn wait_output(
    entry: &Entry,
    cursor: &(usize, usize),
    deadline: tokio::time::Instant,
    until_exit: bool,
    cancellation: &CancellationToken,
) -> Result<(), BoxError> {
    loop {
        let notified = entry.changed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if entry.status.lock().state != CommandState::Running
            || (!until_exit
                && (entry.stdout.lock().await.total_len() > cursor.0
                    || entry.stderr.lock().await.total_len() > cursor.1))
        {
            return Ok(());
        }
        tokio::select! { biased; _ = cancellation.cancelled() => return Err("Session operation cancelled".into()), _ = tokio::time::sleep_until(deadline) => return Ok(()), _ = notified => () }
    }
}

async fn snapshot(entry: &Entry, cursor: &mut (usize, usize), limit: usize) -> CommandOutput {
    async fn stream(buffer: &OutputBuffer, cursor: &mut usize, finished: bool) -> (String, usize) {
        let buffer = buffer.lock().await;
        let prefix = if *cursor < buffer.head.len() {
            &buffer.head[*cursor..]
        } else {
            &[]
        };
        let lost = buffer
            .trimmed
            .saturating_sub((*cursor).max(buffer.head.len()));
        let start = cursor.saturating_sub(buffer.trimmed).min(buffer.data.len());
        let bytes = &buffer.data[start..];
        // Keep incomplete multibyte characters for the next poll while streaming.
        let len = if finished {
            bytes.len()
        } else {
            complete_shell_output_prefix_len(bytes)
        };
        *cursor = buffer.trimmed + start + len;
        (
            format!(
                "{}{}",
                decode_shell_output(prefix),
                decode_shell_output(&bytes[..len])
            ),
            lost,
        )
    }
    let status = entry.status.lock().clone();
    let finished = status.state != CommandState::Running;
    let (stdout, lost_out) = stream(&entry.stdout, &mut cursor.0, finished).await;
    let (stderr, lost_err) = stream(&entry.stderr, &mut cursor.1, finished).await;
    let out_budget = if stderr.len() <= limit / 2 {
        limit - stderr.len()
    } else {
        (limit / 2).min(stdout.len())
    };
    let out = preview_output(&stdout, out_budget);
    let err = preview_output(&stderr, limit - out.text.len());
    let log = entry.log.lock().await;
    CommandOutput {
        output: ExecOutput {
            workspace: Some(entry.workspace.clone()),
            process_id: entry.pid,
            exit_status: status.exit_status.clone(),
            stdout: Some(out.text),
            stderr: (!err.text.is_empty()).then_some(err.text),
            raw_output_path: Some(log.path.display().to_string()),
        },
        task_id: entry.id.clone(),
        state: status.state,
        exit_code: status.code,
        signal: status.signal,
        omitted_bytes: lost_out + lost_err + out.omitted_bytes + err.omitted_bytes,
        log_complete: log.complete,
    }
}
