use anda_core::{BoxError, StateFeatures, ToolOutput};
use async_trait::async_trait;
use ic_auth_types::Xid;
use std::{
    borrow::Cow,
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
};

use super::{ExecArgs, ExecOutput, Executor, ShellToolHook, join_current_dir};
use crate::{context::BaseCtx, hook::ToolHook};

/// Native runtime — full access, runs on Mac/Linux/Docker/Raspberry Pi
pub struct NativeRuntime {
    work_dir: PathBuf,
    temp_dir: PathBuf,
}

impl NativeRuntime {
    pub fn new(work_dir: PathBuf) -> Self {
        Self {
            work_dir,
            temp_dir: std::env::temp_dir(),
        }
    }

    fn build_shell_command(&self, command: &str, workspace_dir: &Path) -> tokio::process::Command {
        #[cfg(not(target_os = "windows"))]
        {
            let mut process = tokio::process::Command::new("sh");
            process.arg("-c").arg(command).current_dir(workspace_dir);
            process
        }

        #[cfg(target_os = "windows")]
        {
            let mut process = tokio::process::Command::new("cmd.exe");
            process.arg("/C").arg(command).current_dir(workspace_dir);
            process
        }
    }

    #[cfg(test)]
    fn test(work_dir: PathBuf) -> Self {
        Self {
            work_dir,
            temp_dir: std::env::temp_dir(),
        }
    }
}

#[async_trait]
impl Executor for NativeRuntime {
    fn name(&self) -> &str {
        "native"
    }

    fn work_dir(&self) -> &PathBuf {
        &self.work_dir
    }

    fn temp_dir(&self) -> &PathBuf {
        &self.temp_dir
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
        let hook = ctx.get_state::<ShellToolHook>();
        let work_dir = ctx
            .meta()
            .get_extra_as::<String>("work_dir")
            .map(PathBuf::from)
            .map(Cow::Owned)
            .unwrap_or_else(|| Cow::Borrowed(&self.work_dir));

        let work_dir = join_current_dir(&work_dir, &input.work_dir);
        let work_dir_str = work_dir.to_string_lossy().to_string();

        let mut cmd = self.build_shell_command(&input.command, &work_dir);
        cmd.env_clear();
        cmd.envs(envs);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        let child = cmd.spawn()?;
        let pid = child.id();
        if !input.background {
            let temp_dir = self.temp_dir();
            match child.wait_with_output().await {
                Ok(output) => {
                    let mut exec_output =
                        ExecOutput::from_output(pid, Some(output), temp_dir).await;
                    exec_output.work_dir = Some(work_dir_str);
                    return Ok(exec_output);
                }
                Err(err) => {
                    let exec_output = ExecOutput {
                        work_dir: Some(work_dir_str),
                        process_id: pid,
                        stderr: Some(format!("Failed to execute background process: {err}")),
                        ..Default::default()
                    };
                    return Ok(exec_output);
                }
            }
        }

        let task_id = format!("{}:{}", self.name(), Xid::new());
        let temp_dir = self.temp_dir();
        let exec_output = ExecOutput::from_output(pid, None, temp_dir).await;
        if let Some(hook) = &hook {
            hook.on_background_start(&ctx, &task_id, &input).await;
        }

        {
            let temp_dir = temp_dir.clone();
            tokio::spawn(async move {
                match child.wait_with_output().await {
                    Ok(output) => {
                        let mut exec_output =
                            ExecOutput::from_output(pid, Some(output), &temp_dir).await;
                        exec_output.work_dir = Some(work_dir_str);
                        if let Some(hook) = &hook {
                            hook.on_background_end(ctx, task_id, ToolOutput::new(exec_output))
                                .await;
                        }
                    }
                    Err(err) => {
                        let exec_output = ExecOutput {
                            work_dir: Some(work_dir_str),
                            process_id: pid,
                            stderr: Some(format!("Failed to execute background process: {err}")),
                            ..Default::default()
                        };
                        if let Some(hook) = &hook {
                            hook.on_background_end(ctx, task_id, ToolOutput::new(exec_output))
                                .await;
                        }
                    }
                }
            });
        }

        Ok(exec_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use std::{
        io::ErrorKind,
        sync::{Arc, Mutex},
        time::Duration,
    };
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
            _ctx: BaseCtx,
            task_id: String,
            output: ToolOutput<ExecOutput>,
        ) {
            if let Some(sender) = self.sender.lock().unwrap().take() {
                let _ = sender.send((task_id, output));
            }
        }
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

    fn background_command(runtime: &NativeRuntime) -> String {
        match runtime.shell() {
            "cmd.exe" => {
                "ping 127.0.0.1 -n 2 > nul & <nul set /p =bg-out & echo bg-err 1>&2".to_string()
            }
            _ => "sleep 0.2; printf '%s' 'bg-out'; printf '%s' 'bg-err' >&2".to_string(),
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
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_runs_foreground_command_with_envs_and_work_dir() {
        let ctx = EngineBuilder::new().mock_ctx();
        let work_dir = TestTempDir::new("anda-native-foreground").await;
        let nested_dir = work_dir.create_dir("nested").await;
        let runtime = NativeRuntime::test(work_dir.path().to_path_buf());
        let env_name = "ANDA_NATIVE_TEST_VALUE";
        let output_file = "env.txt";
        let mut envs = HashMap::new();
        envs.insert(env_name.to_string(), "secret-value".to_string());

        let output = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: foreground_command(&runtime, env_name, output_file),
                    work_dir: "nested".to_string(),
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
        let work_dir = TestTempDir::new("anda-native-missing-workdir").await;
        let runtime = NativeRuntime::test(work_dir.path().to_path_buf());

        let err = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: foreground_command(&runtime, "IGNORED", "env.txt"),
                    work_dir: "missing".to_string(),
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
        let work_dir = TestTempDir::new("anda-native-background").await;
        let (sender, receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(TestHook::new(sender)));
        ctx.base.set_state(hook);
        let runtime = NativeRuntime::test(work_dir.path().to_path_buf());
        let input = ExecArgs {
            command: background_command(&runtime),
            work_dir: String::new(),
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

        assert!(task_id.contains("native"));
        assert_eq!(hook_output.process_id, output.process_id);
        assert_eq!(hook_output.stdout.as_deref().map(str::trim), Some("bg-out"));
        assert_eq!(hook_output.stderr.as_deref().map(str::trim), Some("bg-err"));
    }
}
