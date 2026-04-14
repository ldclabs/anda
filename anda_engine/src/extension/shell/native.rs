use anda_core::BoxError;
use async_trait::async_trait;
use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
};

use super::{DefaultExecutorHook, ExecArgs, ExecOutput, Executor, ExecutorHook};
use crate::context::BaseCtx;

/// Native runtime — full access, runs on Mac/Linux/Docker/Raspberry Pi
pub struct NativeRuntime {
    workdir: PathBuf,
    tempdir: PathBuf,
    shell: Option<ShellProgram>,
    hook: Arc<dyn ExecutorHook>,
}

impl NativeRuntime {
    pub fn new(workdir: PathBuf) -> Self {
        Self {
            shell: detect_native_shell(),
            workdir,
            tempdir: std::env::temp_dir(),
            hook: Arc::new(DefaultExecutorHook),
        }
    }

    pub fn with_hook(mut self, hook: Arc<dyn ExecutorHook>) -> Self {
        self.hook = hook;
        self
    }

    #[cfg(test)]
    fn test(shell: Option<ShellProgram>, workdir: PathBuf) -> Self {
        Self {
            shell,
            workdir,
            tempdir: std::env::temp_dir(),
            hook: Arc::new(DefaultExecutorHook),
        }
    }
}

#[async_trait]
impl Executor for NativeRuntime {
    fn name(&self) -> &str {
        "native"
    }

    fn work_dir(&self) -> &PathBuf {
        &self.workdir
    }

    fn temp_dir(&self) -> &PathBuf {
        &self.tempdir
    }

    fn shell(&self) -> Option<&str> {
        self.shell.as_ref().map(|s| s.kind.as_str())
    }

    async fn execute(
        &self,
        ctx: BaseCtx,
        input: ExecArgs,
        envs: HashMap<String, String>,
    ) -> Result<ExecOutput, BoxError> {
        let shell = self.shell.as_ref().ok_or_else(|| missing_shell_error())?;
        self.hook.on_execution_start(&ctx, &input).await?;

        let mut cmd = tokio::process::Command::new(&shell.program);
        shell.add_shell_args(&mut cmd, &input.command);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.env_clear();
        cmd.envs(envs);
        cmd.current_dir(self.workdir.join(&input.workdir));
        cmd.kill_on_drop(true);

        let child = cmd.spawn()?;
        let pid = child.id();
        if !input.background {
            let temp_dir = self.temp_dir();
            match child.wait_with_output().await {
                Ok(output) => {
                    let exec_output = ExecOutput::from_output(pid, Some(output), temp_dir).await;
                    self.hook.on_execution_end(&ctx, &input, &exec_output).await;
                    return Ok(exec_output);
                }
                Err(err) => {
                    let exec_output = ExecOutput {
                        process_id: pid,
                        stderr: Some(format!("Failed to execute background process: {err}")),
                        ..Default::default()
                    };
                    self.hook.on_execution_end(&ctx, &input, &exec_output).await;
                    return Ok(exec_output);
                }
            }
        }

        let temp_dir = self.temp_dir();
        let exec_output = ExecOutput::from_output(pid, None, temp_dir).await;
        self.hook.on_execution_end(&ctx, &input, &exec_output).await;

        {
            let hook = self.hook.clone();
            let temp_dir = temp_dir.clone();
            tokio::spawn(async move {
                match child.wait_with_output().await {
                    Ok(output) => {
                        let exec_output =
                            ExecOutput::from_output(pid, Some(output), &temp_dir).await;
                        hook.on_background_end(ctx, input, exec_output).await;
                    }
                    Err(err) => {
                        let exec_output = ExecOutput {
                            process_id: pid,
                            stderr: Some(format!("Failed to execute background process: {err}")),
                            ..Default::default()
                        };
                        hook.on_background_end(ctx, input, exec_output).await;
                    }
                }
            });
        }

        Ok(exec_output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ShellProgram {
    pub(super) kind: ShellKind,
    pub(super) program: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShellKind {
    Sh,
    Bash,
    Pwsh,
    PowerShell,
    Cmd,
}

impl fmt::Display for ShellKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl ShellKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            ShellKind::Sh => "sh",
            ShellKind::Bash => "bash",
            ShellKind::Pwsh => "pwsh",
            ShellKind::PowerShell => "powershell",
            ShellKind::Cmd => "cmd",
        }
    }
}

impl ShellProgram {
    pub(super) fn add_shell_args(&self, process: &mut tokio::process::Command, command: &str) {
        match self.kind {
            ShellKind::Sh | ShellKind::Bash => {
                process.arg("-c").arg(command);
            }
            ShellKind::Pwsh | ShellKind::PowerShell => {
                process
                    .arg("-NoLogo")
                    .arg("-NoProfile")
                    .arg("-NonInteractive")
                    .arg("-Command")
                    .arg(command);
            }
            ShellKind::Cmd => {
                process.arg("/C").arg(command);
            }
        }
    }
}

pub(super) fn detect_native_shell() -> Option<ShellProgram> {
    #[cfg(target_os = "windows")]
    {
        let comspec = std::env::var_os("COMSPEC").map(PathBuf::from);
        detect_native_shell_with(true, |name| which::which(name).ok(), comspec)
    }
    #[cfg(not(target_os = "windows"))]
    {
        detect_native_shell_with(false, |name| which::which(name).ok(), None)
    }
}

fn detect_native_shell_with<F>(
    is_windows: bool,
    mut resolve: F,
    comspec: Option<PathBuf>,
) -> Option<ShellProgram>
where
    F: FnMut(&str) -> Option<PathBuf>,
{
    if is_windows {
        for (name, kind) in [
            ("bash", ShellKind::Bash),
            ("sh", ShellKind::Sh),
            ("pwsh", ShellKind::Pwsh),
            ("powershell", ShellKind::PowerShell),
            ("cmd", ShellKind::Cmd),
            ("cmd.exe", ShellKind::Cmd),
        ] {
            if let Some(program) = resolve(name) {
                // Windows may expose `C:\Windows\System32\bash.exe`, a legacy
                // WSL launcher that executes commands inside Linux userspace.
                // That breaks native Windows commands like `ipconfig`.
                if name == "bash" && is_windows_wsl_bash_launcher(&program) {
                    continue;
                }
                return Some(ShellProgram { kind, program });
            }
        }
        if let Some(program) = comspec {
            return Some(ShellProgram {
                kind: ShellKind::Cmd,
                program,
            });
        }
        return None;
    }

    for (name, kind) in [("sh", ShellKind::Sh), ("bash", ShellKind::Bash)] {
        if let Some(program) = resolve(name) {
            return Some(ShellProgram { kind, program });
        }
    }
    None
}

fn is_windows_wsl_bash_launcher(program: &Path) -> bool {
    let normalized = program
        .to_string_lossy()
        .replace('/', "\\")
        .to_ascii_lowercase();
    normalized.ends_with("\\windows\\system32\\bash.exe")
        || normalized.ends_with("\\windows\\sysnative\\bash.exe")
}

pub(super) fn missing_shell_error() -> BoxError {
    #[cfg(target_os = "windows")]
    {
        "Native runtime could not find a usable shell (tried: bash, sh, pwsh, powershell, cmd). \
         Install Git Bash or PowerShell and ensure it is available on PATH."
            .into()
    }
    #[cfg(not(target_os = "windows"))]
    {
        "Native runtime could not find a usable shell (tried: sh, bash). \
         Install a POSIX shell and ensure it is available on PATH."
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use std::{ffi::OsStr, io::ErrorKind, sync::Mutex, time::Duration};
    use tokio::sync::oneshot;

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

    struct TestHook {
        sender: Mutex<Option<oneshot::Sender<(ExecArgs, ExecOutput)>>>,
    }

    impl TestHook {
        fn new(sender: oneshot::Sender<(ExecArgs, ExecOutput)>) -> Self {
            Self {
                sender: Mutex::new(Some(sender)),
            }
        }
    }

    #[async_trait]
    impl ExecutorHook for TestHook {
        async fn on_background_end(&self, _ctx: BaseCtx, input: ExecArgs, output: ExecOutput) {
            if let Some(sender) = self.sender.lock().unwrap().take() {
                let _ = sender.send((input, output));
            }
        }
    }

    fn shell_for_tests() -> ShellProgram {
        detect_native_shell().expect("expected a usable shell for NativeRuntime tests")
    }

    fn args_of(process: &tokio::process::Command) -> Vec<String> {
        process.as_std().get_args().map(os_str_to_string).collect()
    }

    fn os_str_to_string(value: &OsStr) -> String {
        value.to_string_lossy().into_owned()
    }

    fn foreground_command(kind: ShellKind, env_name: &str, output_file: &str) -> String {
        match kind {
            ShellKind::Sh | ShellKind::Bash => format!(
                "printf '%s' \"${env_name}\" > {output_file}; printf '%s' 'done'; printf '%s' 'warn' >&2"
            ),
            ShellKind::Pwsh | ShellKind::PowerShell => format!(
                "[System.IO.File]::WriteAllText('{output_file}', $env:{env_name}); [Console]::Out.Write('done'); [Console]::Error.Write('warn')"
            ),
            ShellKind::Cmd => format!(
                "<nul set /p =%{env_name}% > {output_file} & <nul set /p =done & echo warn 1>&2"
            ),
        }
    }

    fn background_command(kind: ShellKind) -> String {
        match kind {
            ShellKind::Sh | ShellKind::Bash => {
                "sleep 0.2; printf '%s' 'bg-out'; printf '%s' 'bg-err' >&2".to_string()
            }
            ShellKind::Pwsh | ShellKind::PowerShell => {
                "Start-Sleep -Milliseconds 200; [Console]::Out.Write('bg-out'); [Console]::Error.Write('bg-err')"
                    .to_string()
            }
            ShellKind::Cmd => {
                "ping 127.0.0.1 -n 2 > nul & <nul set /p =bg-out & echo bg-err 1>&2".to_string()
            }
        }
    }

    #[test]
    fn new_initializes_paths_and_shell() {
        let runtime = NativeRuntime::new(PathBuf::from("/home/anda-native-runtime-tests"));

        assert_eq!(runtime.name(), "native");
        assert_eq!(
            runtime.work_dir(),
            &PathBuf::from("/home/anda-native-runtime-tests")
        );
        assert_eq!(runtime.temp_dir(), &std::env::temp_dir());
        assert_eq!(
            runtime.shell(),
            detect_native_shell()
                .as_ref()
                .map(|shell| shell.kind.as_str())
        );
    }

    #[test]
    fn add_shell_args_matches_shell_kind() {
        for (kind, expected) in [
            (ShellKind::Sh, vec!["-c", "echo test"]),
            (ShellKind::Bash, vec!["-c", "echo test"]),
            (
                ShellKind::Pwsh,
                vec![
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "echo test",
                ],
            ),
            (
                ShellKind::PowerShell,
                vec![
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "echo test",
                ],
            ),
            (ShellKind::Cmd, vec!["/C", "echo test"]),
        ] {
            let shell = ShellProgram {
                kind,
                program: PathBuf::from(kind.as_str()),
            };
            let mut process = tokio::process::Command::new(&shell.program);

            shell.add_shell_args(&mut process, "echo test");

            assert_eq!(args_of(&process), expected);
        }
    }

    #[test]
    fn detects_posix_shell_prefers_sh() {
        let mut calls = Vec::new();
        let shell = detect_native_shell_with(
            false,
            |name| {
                calls.push(name.to_string());
                match name {
                    "sh" => Some(PathBuf::from("/bin/sh")),
                    "bash" => Some(PathBuf::from("/bin/bash")),
                    _ => None,
                }
            },
            Some(PathBuf::from("ignored")),
        );

        assert_eq!(calls, vec!["sh"]);
        assert_eq!(
            shell,
            Some(ShellProgram {
                kind: ShellKind::Sh,
                program: PathBuf::from("/bin/sh"),
            })
        );
    }

    #[test]
    fn detects_posix_shell_falls_back_to_bash() {
        let mut calls = Vec::new();
        let shell = detect_native_shell_with(
            false,
            |name| {
                calls.push(name.to_string());
                (name == "bash").then(|| PathBuf::from("/bin/bash"))
            },
            None,
        );

        assert_eq!(calls, vec!["sh", "bash"]);
        assert_eq!(
            shell,
            Some(ShellProgram {
                kind: ShellKind::Bash,
                program: PathBuf::from("/bin/bash"),
            })
        );
    }

    #[test]
    fn detects_windows_shell_skips_wsl_bash_launcher() {
        let mut calls = Vec::new();
        let shell = detect_native_shell_with(
            true,
            |name| {
                calls.push(name.to_string());
                match name {
                    "bash" => Some(PathBuf::from(r"C:\Windows\System32\bash.exe")),
                    "sh" => Some(PathBuf::from(r"C:\Program Files\Git\bin\sh.exe")),
                    _ => None,
                }
            },
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
        );

        assert_eq!(calls, vec!["bash", "sh"]);
        assert_eq!(
            shell,
            Some(ShellProgram {
                kind: ShellKind::Sh,
                program: PathBuf::from(r"C:\Program Files\Git\bin\sh.exe"),
            })
        );
    }

    #[test]
    fn detects_windows_shell_falls_back_to_comspec() {
        let shell = detect_native_shell_with(
            true,
            |_| None,
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
        );

        assert_eq!(
            shell,
            Some(ShellProgram {
                kind: ShellKind::Cmd,
                program: PathBuf::from(r"C:\Windows\System32\cmd.exe"),
            })
        );
    }

    #[test]
    fn recognizes_windows_wsl_bash_launchers() {
        assert!(is_windows_wsl_bash_launcher(Path::new(
            r"C:\Windows\System32\bash.exe"
        )));
        assert!(is_windows_wsl_bash_launcher(Path::new(
            r"C:/Windows/Sysnative/bash.exe"
        )));
        assert!(!is_windows_wsl_bash_launcher(Path::new(
            r"C:\Program Files\Git\bin\bash.exe"
        )));
    }

    #[test]
    fn missing_shell_error_message_mentions_supported_shells() {
        let message = missing_shell_error().to_string();

        assert!(message.contains("Native runtime could not find a usable shell"));
        #[cfg(target_os = "windows")]
        assert!(message.contains("bash, sh, pwsh, powershell, cmd"));
        #[cfg(not(target_os = "windows"))]
        assert!(message.contains("sh, bash"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_returns_error_when_shell_is_missing() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workdir = TestTempDir::new("anda-native-no-shell").await;
        let runtime = NativeRuntime::test(None, workdir.path().to_path_buf());

        let err = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: "echo ignored".to_string(),
                    ..Default::default()
                },
                HashMap::new(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.to_string(), missing_shell_error().to_string());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_runs_foreground_command_with_envs_and_workdir() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workdir = TestTempDir::new("anda-native-foreground").await;
        let nested_dir = workdir.create_dir("nested").await;
        let shell = shell_for_tests();
        let runtime = NativeRuntime::test(Some(shell.clone()), workdir.path().to_path_buf());
        let env_name = "ANDA_NATIVE_TEST_VALUE";
        let output_file = "env.txt";
        let mut envs = HashMap::new();
        envs.insert(env_name.to_string(), "secret-value".to_string());

        let output = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: foreground_command(shell.kind, env_name, output_file),
                    workdir: "nested".to_string(),
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
    async fn execute_returns_spawn_error_for_missing_workdir() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workdir = TestTempDir::new("anda-native-missing-workdir").await;
        let shell = shell_for_tests();
        let runtime = NativeRuntime::test(Some(shell.clone()), workdir.path().to_path_buf());

        let err = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: foreground_command(shell.kind, "IGNORED", "env.txt"),
                    workdir: "missing".to_string(),
                    ..Default::default()
                },
                HashMap::new(),
            )
            .await
            .unwrap_err();

        assert_eq!(
            err.downcast_ref::<std::io::Error>().unwrap().kind(),
            ErrorKind::NotFound
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_reports_background_output_via_hook() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workdir = TestTempDir::new("anda-native-background").await;
        let shell = shell_for_tests();
        let (sender, receiver) = oneshot::channel();
        let hook: Arc<dyn ExecutorHook> = Arc::new(TestHook::new(sender));
        let runtime =
            NativeRuntime::test(Some(shell.clone()), workdir.path().to_path_buf()).with_hook(hook);
        let input = ExecArgs {
            command: background_command(shell.kind),
            workdir: String::new(),
            background: true,
            ..Default::default()
        };

        let output = runtime
            .execute(ctx.base, input.clone(), HashMap::new())
            .await
            .unwrap();

        assert!(output.process_id.is_some());
        assert_eq!(output.exit_status, None);
        assert_eq!(output.stdout, None);
        assert_eq!(output.stderr, None);

        let (hook_input, hook_output) = tokio::time::timeout(Duration::from_secs(5), receiver)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(hook_input.command, input.command);
        assert_eq!(hook_input.background, input.background);
        assert_eq!(hook_output.process_id, output.process_id);
        assert_eq!(hook_output.stdout.as_deref().map(str::trim), Some("bg-out"));
        assert_eq!(hook_output.stderr.as_deref().map(str::trim), Some("bg-err"));
    }
}
