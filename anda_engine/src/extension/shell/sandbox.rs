use anda_core::{BoxError, ToolOutput};
use async_trait::async_trait;
use boxlite::{
    BoxCommand, BoxOptions, BoxliteOptions, BoxliteRuntime, ExecResult, Execution, LiteBox,
    RootfsSpec,
    runtime::options::{ImageRegistry, VolumeSpec},
};
use futures_util::stream::{Stream, StreamExt};
use ic_auth_types::Xid;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    pin::Pin,
    process::{ExitStatus, Output},
    sync::Arc,
    time::Duration,
};

use super::{ExecArgs, ExecOutput, Executor, SHELL_TIMEOUT_SECS, ShellToolHook, join_current_dir};
use crate::{context::BaseCtx, hook::ToolHook};

type OutputStream = Pin<Box<dyn Stream<Item = String> + Send>>;

/// Sandbox runtime — restricted access, runs in a sandboxed environment
pub struct SandboxRuntime {
    workspace: PathBuf,
    temp_dir: PathBuf,
    runner: Arc<dyn SandboxCommandRunner>,
}

impl SandboxRuntime {
    /// Creates a new sandbox runtime instance with the specified directory for sandbox files.
    pub async fn new(home_dir: PathBuf) -> Result<Self, BoxError> {
        let runtime = BoxliteRuntime::new(BoxliteOptions {
            home_dir: home_dir.clone(),
            image_registries: vec![
                ImageRegistry::https("ghcr.io/ldclabs").with_search(true),
                ImageRegistry::https("docker.io").with_search(true),
            ],
        })?;

        tokio::fs::create_dir_all(&home_dir).await?;
        let options = BoxOptions {
            rootfs: RootfsSpec::Image("alpine:latest".into()),
            working_dir: Some("/home".to_string()),
            volumes: vec![VolumeSpec {
                host_path: home_dir.to_string_lossy().to_string(),
                guest_path: "/home".to_string(),
                read_only: false,
            }],
            ..Default::default()
        };

        let litebox = runtime.create(options, None).await?;
        Ok(Self {
            runner: Arc::new(BoxliteRunner { litebox }),
            workspace: "/home".into(),
            temp_dir: "/tmp".into(),
        })
    }

    #[cfg(test)]
    fn test(runner: Arc<dyn SandboxCommandRunner>, workspace: PathBuf, temp_dir: PathBuf) -> Self {
        Self {
            workspace,
            temp_dir,
            runner,
        }
    }

    fn build_command(
        &self,
        workspace: &Path,
        input: &ExecArgs,
        envs: HashMap<String, String>,
    ) -> SandboxCommandSpec {
        let mut envs: Vec<_> = envs.into_iter().collect();
        envs.sort();

        SandboxCommandSpec {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), input.command.clone()],
            working_dir: join_current_dir(workspace, &input.workspace)
                .to_string_lossy()
                .to_string(),
            envs,
            timeout: (!input.background).then_some(Duration::from_secs(SHELL_TIMEOUT_SECS)),
        }
    }
}

#[async_trait]
impl Executor for SandboxRuntime {
    fn name(&self) -> &str {
        "sandbox"
    }

    fn workspace(&self) -> &PathBuf {
        &self.workspace
    }

    fn os(&self) -> &str {
        "Alpine Linux"
    }

    fn shell(&self) -> &str {
        "sh"
    }

    fn temp_dir(&self) -> &PathBuf {
        &self.temp_dir
    }

    async fn execute(
        &self,
        ctx: BaseCtx,
        input: ExecArgs,
        envs: HashMap<String, String>,
    ) -> Result<ExecOutput, BoxError> {
        let hook = ctx.get_state::<ShellToolHook>();
        let cmd = self.build_command(&self.workspace, &input, envs);
        let workspace_str = cmd.working_dir.clone();
        let child = self.runner.exec(cmd).await?;
        if !input.background {
            let temp_dir = self.temp_dir();
            match wait_with_output(child).await {
                Ok(output) => {
                    let mut exec_output =
                        ExecOutput::from_output(None, Some(output), temp_dir).await;
                    exec_output.workspace = Some(workspace_str);

                    return Ok(exec_output);
                }
                Err(err) => {
                    let exec_output = ExecOutput {
                        workspace: Some(workspace_str),
                        stderr: Some(format!("Failed to execute background process: {err}")),
                        ..Default::default()
                    };

                    return Ok(exec_output);
                }
            }
        }

        let task_id = format!("{}:{}", self.name(), Xid::new());
        let temp_dir = self.temp_dir();
        let exec_output = ExecOutput::from_output(None, None, temp_dir).await;
        if let Some(hook) = &hook {
            hook.on_background_start(&ctx, &task_id, &input).await;
        }
        {
            let temp_dir = temp_dir.clone();
            tokio::spawn(async move {
                match wait_with_output(child).await {
                    Ok(output) => {
                        let mut exec_output =
                            ExecOutput::from_output(None, Some(output), &temp_dir).await;
                        exec_output.workspace = Some(workspace_str);
                        if let Some(hook) = &hook {
                            hook.on_background_end(ctx, task_id, ToolOutput::new(exec_output))
                                .await;
                        }
                    }
                    Err(err) => {
                        let exec_output = ExecOutput {
                            workspace: Some(workspace_str),
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct SandboxCommandSpec {
    program: String,
    args: Vec<String>,
    working_dir: String,
    envs: Vec<(String, String)>,
    timeout: Option<Duration>,
}

impl SandboxCommandSpec {
    fn into_box_command(self) -> BoxCommand {
        let mut cmd = BoxCommand::new(self.program)
            .args(self.args)
            .working_dir(self.working_dir);
        for (key, value) in self.envs {
            cmd = cmd.env(key, value);
        }
        if let Some(timeout) = self.timeout {
            cmd = cmd.timeout(timeout);
        }
        cmd
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SandboxExecStatus {
    exit_code: i32,
    error_message: Option<String>,
}

impl From<ExecResult> for SandboxExecStatus {
    fn from(value: ExecResult) -> Self {
        Self {
            exit_code: value.exit_code,
            error_message: value.error_message,
        }
    }
}

#[async_trait]
trait SandboxExecutionHandle: Send {
    fn stdout(&mut self) -> Option<OutputStream>;
    fn stderr(&mut self) -> Option<OutputStream>;
    async fn wait(&mut self) -> Result<SandboxExecStatus, BoxError>;
}

#[async_trait]
trait SandboxCommandRunner: Send + Sync {
    async fn exec(
        &self,
        command: SandboxCommandSpec,
    ) -> Result<Box<dyn SandboxExecutionHandle>, BoxError>;
}

struct BoxliteRunner {
    litebox: LiteBox,
}

struct BoxliteExecutionHandle {
    child: Execution,
}

#[async_trait]
impl SandboxCommandRunner for BoxliteRunner {
    async fn exec(
        &self,
        command: SandboxCommandSpec,
    ) -> Result<Box<dyn SandboxExecutionHandle>, BoxError> {
        let child = self.litebox.exec(command.into_box_command()).await?;
        Ok(Box::new(BoxliteExecutionHandle { child }))
    }
}

#[async_trait]
impl SandboxExecutionHandle for BoxliteExecutionHandle {
    fn stdout(&mut self) -> Option<OutputStream> {
        self.child
            .stdout()
            .map(|stream| Box::pin(stream) as OutputStream)
    }

    fn stderr(&mut self) -> Option<OutputStream> {
        self.child
            .stderr()
            .map(|stream| Box::pin(stream) as OutputStream)
    }

    async fn wait(&mut self) -> Result<SandboxExecStatus, BoxError> {
        Ok(self.child.wait().await?.into())
    }
}

async fn wait_with_output(mut child: Box<dyn SandboxExecutionHandle>) -> Result<Output, BoxError> {
    async fn read_to_end(stream: Option<OutputStream>) -> Vec<u8> {
        let mut output = String::new();
        if let Some(mut stream) = stream {
            while let Some(chunk) = stream.next().await {
                output.push_str(&chunk);
            }
        }

        output.into_bytes()
    }

    let stdout_fut = read_to_end(child.stdout());
    let stderr_fut = read_to_end(child.stderr());
    let (status, stdout, mut stderr) = tokio::join!(child.wait(), stdout_fut, stderr_fut);
    let status = status?;

    if let Some(error_message) = status.error_message {
        if !stderr.is_empty() && !stderr.ends_with(b"\n") {
            stderr.push(b'\n');
        }
        stderr.extend_from_slice(error_message.as_bytes());
    }

    Ok(Output {
        status: exit_status_from_code(status.exit_code),
        stdout,
        stderr,
    })
}

#[cfg(unix)]
fn exit_status_from_code(code: i32) -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;

    if code >= 0 {
        ExitStatus::from_raw(code << 8)
    } else {
        ExitStatus::from_raw(-code)
    }
}

#[cfg(windows)]
fn exit_status_from_code(code: i32) -> ExitStatus {
    use std::os::windows::process::ExitStatusExt;

    ExitStatus::from_raw(code.try_into().unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use futures_util::stream;
    use std::path::Path;
    use std::{collections::VecDeque, sync::Mutex};
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

    struct PanicRunner;

    #[async_trait]
    impl SandboxCommandRunner for PanicRunner {
        async fn exec(
            &self,
            _command: SandboxCommandSpec,
        ) -> Result<Box<dyn SandboxExecutionHandle>, BoxError> {
            panic!("runner should not be called")
        }
    }

    struct FakeExecution {
        stdout_chunks: Option<Vec<String>>,
        stderr_chunks: Option<Vec<String>>,
        wait_result: Option<Result<SandboxExecStatus, String>>,
    }

    impl FakeExecution {
        fn success(stdout: &[&str], stderr: &[&str], exit_code: i32) -> Self {
            Self {
                stdout_chunks: Some(stdout.iter().map(|chunk| (*chunk).to_string()).collect()),
                stderr_chunks: Some(stderr.iter().map(|chunk| (*chunk).to_string()).collect()),
                wait_result: Some(Ok(SandboxExecStatus {
                    exit_code,
                    error_message: None,
                })),
            }
        }

        fn success_with_error_message(
            stdout: &[&str],
            stderr: &[&str],
            exit_code: i32,
            error_message: &str,
        ) -> Self {
            Self {
                stdout_chunks: Some(stdout.iter().map(|chunk| (*chunk).to_string()).collect()),
                stderr_chunks: Some(stderr.iter().map(|chunk| (*chunk).to_string()).collect()),
                wait_result: Some(Ok(SandboxExecStatus {
                    exit_code,
                    error_message: Some(error_message.to_string()),
                })),
            }
        }

        fn wait_error(message: &str) -> Self {
            Self {
                stdout_chunks: Some(Vec::new()),
                stderr_chunks: Some(Vec::new()),
                wait_result: Some(Err(message.to_string())),
            }
        }
    }

    #[async_trait]
    impl SandboxExecutionHandle for FakeExecution {
        fn stdout(&mut self) -> Option<OutputStream> {
            self.stdout_chunks
                .take()
                .map(|chunks| Box::pin(stream::iter(chunks)) as OutputStream)
        }

        fn stderr(&mut self) -> Option<OutputStream> {
            self.stderr_chunks
                .take()
                .map(|chunks| Box::pin(stream::iter(chunks)) as OutputStream)
        }

        async fn wait(&mut self) -> Result<SandboxExecStatus, BoxError> {
            match self
                .wait_result
                .take()
                .expect("wait should only be called once")
            {
                Ok(status) => Ok(status),
                Err(message) => Err(message.into()),
            }
        }
    }

    enum RunnerOutcome {
        Execution(FakeExecution),
        Error(String),
    }

    struct TestRunner {
        commands: Mutex<Vec<SandboxCommandSpec>>,
        outcomes: Mutex<VecDeque<RunnerOutcome>>,
    }

    impl TestRunner {
        fn new(outcomes: Vec<RunnerOutcome>) -> Self {
            Self {
                commands: Mutex::new(Vec::new()),
                outcomes: Mutex::new(outcomes.into()),
            }
        }

        fn recorded_commands(&self) -> Vec<SandboxCommandSpec> {
            self.commands.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl SandboxCommandRunner for TestRunner {
        async fn exec(
            &self,
            command: SandboxCommandSpec,
        ) -> Result<Box<dyn SandboxExecutionHandle>, BoxError> {
            self.commands.lock().unwrap().push(command);
            let outcome = self
                .outcomes
                .lock()
                .unwrap()
                .pop_front()
                .expect("missing runner outcome");
            match outcome {
                RunnerOutcome::Execution(execution) => Ok(Box::new(execution)),
                RunnerOutcome::Error(message) => Err(message.into()),
            }
        }
    }

    type BackgroundEndSender = oneshot::Sender<(String, ToolOutput<ExecOutput>)>;

    struct TestHook {
        sender: Mutex<Option<BackgroundEndSender>>,
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

    #[test]
    fn metadata_accessors_return_expected_values() {
        let runtime = SandboxRuntime::test(
            Arc::new(PanicRunner),
            PathBuf::from("/home"),
            PathBuf::from("/tmp"),
        );

        assert_eq!(runtime.name(), "sandbox");
        assert_eq!(runtime.os(), "Alpine Linux");
        assert_eq!(runtime.shell(), "sh");
        assert_eq!(runtime.workspace(), &PathBuf::from("/home"));
        assert_eq!(runtime.temp_dir(), &PathBuf::from("/tmp"));
    }

    #[test]
    fn build_command_sets_shell_env_workspace_and_timeout() {
        let runtime = SandboxRuntime::test(
            Arc::new(PanicRunner),
            PathBuf::from("/home"),
            PathBuf::from("/tmp"),
        );
        let mut envs = HashMap::new();
        envs.insert("Z_VALUE".to_string(), "2".to_string());
        envs.insert("A_VALUE".to_string(), "1".to_string());

        let command = runtime.build_command(
            &PathBuf::from("/home"),
            &ExecArgs {
                command: "echo hello".to_string(),
                workspace: "nested".to_string(),
                ..Default::default()
            },
            envs,
        );

        assert_eq!(command.program, "sh");
        assert_eq!(command.args, vec!["-c", "echo hello"]);
        assert_eq!(command.working_dir, "/home/nested");
        assert_eq!(
            command.envs,
            vec![
                ("A_VALUE".to_string(), "1".to_string()),
                ("Z_VALUE".to_string(), "2".to_string()),
            ]
        );
        assert_eq!(
            command.timeout,
            Some(Duration::from_secs(SHELL_TIMEOUT_SECS))
        );
    }

    #[test]
    fn build_command_skips_timeout_for_background_execution() {
        let runtime = SandboxRuntime::test(
            Arc::new(PanicRunner),
            PathBuf::from("/home"),
            PathBuf::from("/tmp"),
        );

        let command = runtime.build_command(
            &PathBuf::from("/home"),
            &ExecArgs {
                command: "echo hello".to_string(),
                workspace: String::new(),
                background: true,
                ..Default::default()
            },
            HashMap::new(),
        );

        assert_eq!(command.timeout, None);
        assert_eq!(command.working_dir, "/home/");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_runs_foreground_command() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-sandbox-foreground-workdir").await;
        let tempdir = TestTempDir::new("anda-sandbox-foreground-temp").await;
        workspace.create_dir("nested").await;
        let runner = Arc::new(TestRunner::new(vec![RunnerOutcome::Execution(
            FakeExecution::success(&["done"], &["warn"], 0),
        )]));
        let runtime = SandboxRuntime::test(
            runner.clone(),
            workspace.path().to_path_buf(),
            tempdir.path().to_path_buf(),
        );
        let mut envs = HashMap::new();
        envs.insert("ANDA_SANDBOX".to_string(), "secret".to_string());

        let output = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: "echo hello".to_string(),
                    workspace: "nested".to_string(),
                    ..Default::default()
                },
                envs,
            )
            .await
            .unwrap();

        let expected_status = exit_status_from_code(0).to_string();
        assert_eq!(output.process_id, None);
        assert_eq!(output.raw_output_path, None);
        assert_eq!(
            output.exit_status.as_deref(),
            Some(expected_status.as_str())
        );
        assert_eq!(output.stdout.as_deref(), Some("done"));
        assert_eq!(output.stderr.as_deref(), Some("warn"));

        let recorded = runner.recorded_commands();
        assert_eq!(recorded.len(), 1);
        assert_eq!(
            recorded[0].working_dir,
            workspace.path().join("nested").to_string_lossy()
        );
        assert_eq!(
            recorded[0].envs,
            vec![("ANDA_SANDBOX".to_string(), "secret".to_string())]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_propagates_runner_errors() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-sandbox-runner-error-workdir").await;
        let temp_dir = TestTempDir::new("anda-sandbox-runner-error-temp").await;
        let runtime = SandboxRuntime::test(
            Arc::new(TestRunner::new(vec![RunnerOutcome::Error(
                "sandbox unavailable".to_string(),
            )])),
            workspace.path().to_path_buf(),
            temp_dir.path().to_path_buf(),
        );

        let err = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: "echo hello".to_string(),
                    ..Default::default()
                },
                HashMap::new(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.to_string(), "sandbox unavailable");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_reports_background_output_via_hook() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-sandbox-background-workdir").await;
        let temp_dir = TestTempDir::new("anda-sandbox-background-temp").await;
        let runner = Arc::new(TestRunner::new(vec![RunnerOutcome::Execution(
            FakeExecution::success(&["bg-out"], &["bg-err"], 0),
        )]));
        let (sender, receiver) = oneshot::channel();
        let hook = ShellToolHook::new(Arc::new(TestHook::new(sender)));
        ctx.base.set_state(hook);

        let runtime = SandboxRuntime::test(
            runner,
            workspace.path().to_path_buf(),
            temp_dir.path().to_path_buf(),
        );
        let input = ExecArgs {
            command: "echo background".to_string(),
            workspace: String::new(),
            background: true,
            ..Default::default()
        };

        let output = runtime
            .execute(ctx.base, input.clone(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(output.process_id, None);
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
        let expected_status = exit_status_from_code(0).to_string();

        assert!(task_id.contains("sandbox"));
        assert_eq!(hook_output.process_id, None);
        assert_eq!(
            hook_output.exit_status.as_deref(),
            Some(expected_status.as_str())
        );
        assert_eq!(hook_output.stdout.as_deref(), Some("bg-out"));
        assert_eq!(hook_output.stderr.as_deref(), Some("bg-err"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_returns_exec_output_when_wait_fails() {
        let ctx = EngineBuilder::new().mock_ctx();
        let workspace = TestTempDir::new("anda-sandbox-wait-error-workdir").await;
        let temp_dir = TestTempDir::new("anda-sandbox-wait-error-temp").await;
        let runtime = SandboxRuntime::test(
            Arc::new(TestRunner::new(vec![RunnerOutcome::Execution(
                FakeExecution::wait_error("broken pipe"),
            )])),
            workspace.path().to_path_buf(),
            temp_dir.path().to_path_buf(),
        );

        let output = runtime
            .execute(
                ctx.base,
                ExecArgs {
                    command: "echo hello".to_string(),
                    ..Default::default()
                },
                HashMap::new(),
            )
            .await
            .unwrap();

        assert_eq!(output.process_id, None);
        assert_eq!(output.exit_status, None);
        assert_eq!(
            output.stderr.as_deref(),
            Some("Failed to execute background process: broken pipe")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wait_with_output_appends_error_message_to_stderr() {
        let output = wait_with_output(Box::new(FakeExecution::success_with_error_message(
            &["hello", " world"],
            &["warn"],
            17,
            "sandbox crash report",
        )))
        .await
        .unwrap();

        assert_eq!(output.stdout, b"hello world");
        assert_eq!(output.stderr, b"warn\nsandbox crash report");
        assert_eq!(output.status, exit_status_from_code(17));
    }

    #[cfg(unix)]
    #[test]
    fn exit_status_from_negative_code_preserves_signal() {
        use std::os::unix::process::ExitStatusExt;

        assert_eq!(exit_status_from_code(-9).signal(), Some(9));
    }
}
