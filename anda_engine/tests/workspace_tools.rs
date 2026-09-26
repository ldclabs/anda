//! Public API regressions for coding tool bundles, patches and shell sessions.

#[cfg(unix)]
use anda_core::StateFeatures;
use anda_core::Tool;
use anda_engine::{
    context::BaseCtx,
    engine::EngineBuilder,
    extension::{
        fs::{
            ApplyPatchArgs, ApplyPatchTool, EditFileArgs, EditFileTool, FileVersion, ReadFileArgs,
            ReadFileTool, WriteFileArgs, WriteFileTool,
        },
        shell::{NativeRuntime, ShellSessionScope},
    },
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(unix)]
use anda_engine::extension::shell::{
    CommandArgs, CommandOutput, CommandState, Executor, SessionAction, SessionArgs, SessionLimits,
};
#[cfg(unix)]
use std::time::Duration;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "anda-workspace-regression-{:032x}",
            rand::random::<u128>()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(std::fs::canonicalize(path).unwrap())
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn context() -> BaseCtx {
    let ctx = EngineBuilder::new().mock_ctx().base;
    ctx.set_state(ShellSessionScope::new());
    ctx
}

#[tokio::test]
async fn concurrent_file_edits_preserve_both_changes() {
    let dir = Directory::new();
    let path = dir.path().join("file.txt");
    let a = EditFileTool::new(dir.0.clone());
    let b = EditFileTool::new(dir.0.clone());
    for _ in 0..20 {
        tokio::fs::write(&path, "alpha=old\nbeta=old\n")
            .await
            .unwrap();
        let (a, b) = tokio::join!(
            a.call(
                context(),
                EditFileArgs {
                    path: "file.txt".into(),
                    old_string: "alpha=old".into(),
                    new_string: "alpha=new".into(),
                    ..Default::default()
                },
                vec![]
            ),
            b.call(
                context(),
                EditFileArgs {
                    path: "file.txt".into(),
                    old_string: "beta=old".into(),
                    new_string: "beta=new".into(),
                    ..Default::default()
                },
                vec![]
            )
        );
        a.unwrap();
        b.unwrap();
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "alpha=new\nbeta=new\n"
        );
    }
}

#[tokio::test]
async fn oversized_write_leaves_old_file_unchanged() {
    let dir = Directory::new();
    tokio::fs::write(dir.path().join("file.txt"), "original")
        .await
        .unwrap();
    let tool = WriteFileTool::new(dir.0.clone());
    let result = tool
        .call(
            context(),
            WriteFileArgs {
                path: "file.txt".into(),
                content: "x".repeat(10 * 1024 * 1024 + 1),
                ..Default::default()
            },
            vec![],
        )
        .await;
    assert!(result.is_err());
    assert_eq!(
        tokio::fs::read_to_string(dir.path().join("file.txt"))
            .await
            .unwrap(),
        "original"
    );
}

#[tokio::test]
async fn patch_prevalidation_dry_run_versions_and_crlf() {
    let dir = Directory::new();
    tokio::fs::write(dir.path().join("old.txt"), "alpha\r\nbeta\r\n")
        .await
        .unwrap();
    let tool = ApplyPatchTool::new(dir.0.clone());
    let patch = "*** Begin Patch\n*** Update File: old.txt\n*** Move to: nested/new.txt\n@@\n alpha\n-beta\n+gamma\n*** Add File: added.txt\n+created\n*** End Patch";
    let preview = tool
        .call(
            context(),
            ApplyPatchArgs {
                patch: patch.into(),
                dry_run: true,
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap()
        .output;
    assert!(!dir.path().join("added.txt").exists());
    assert!(preview.changes.iter().all(|change| !change.applied));
    assert!(preview.diff.contains("+gamma"));
    let stale = tool
        .call(
            context(),
            ApplyPatchArgs {
                patch: patch.into(),
                expected_versions: vec![FileVersion {
                    path: "old.txt".into(),
                    sha256: "stale".into(),
                }],
                ..Default::default()
            },
            vec![],
        )
        .await;
    assert!(stale.unwrap_err().to_string().contains("Stale"));
    let result = tool
        .call(
            context(),
            ApplyPatchArgs {
                patch: patch.into(),
                expected_versions: preview
                    .changes
                    .iter()
                    .map(|change| FileVersion {
                        path: change.path.clone(),
                        sha256: change.original_sha256.clone(),
                    })
                    .collect(),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap();
    assert_eq!(result.is_error, None);
    assert!(result.output.changes.iter().all(|change| change.applied));
    assert!(!dir.path().join("old.txt").exists());
    assert_eq!(
        tokio::fs::read(dir.path().join("nested/new.txt"))
            .await
            .unwrap(),
        b"alpha\r\ngamma\r\n"
    );
}

#[tokio::test]
async fn invalid_patch_does_not_apply_earlier_changes() {
    let dir = Directory::new();
    tokio::fs::write(dir.path().join("ambiguous.txt"), "same\nsame\n")
        .await
        .unwrap();
    let tool = ApplyPatchTool::new(dir.0.clone());
    for patch in [
        "*** Begin Patch\n*** Add File: new.txt\n+first\n*** Update File: ambiguous.txt\n@@\n-same\n+changed\n*** End Patch",
        "*** Begin Patch\n*** Add File: new.txt\n+first\n*** Add File: ../escape.txt\n+outside\n*** End Patch",
        "*** Begin Patch\n*** Add File: ambiguous.txt\n+overwrite\n*** End Patch",
        "*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch",
    ] {
        assert!(
            tool.call(
                context(),
                ApplyPatchArgs {
                    patch: patch.into(),
                    ..Default::default()
                },
                vec![]
            )
            .await
            .is_err()
        );
        assert!(!dir.path().join("new.txt").exists());
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("ambiguous.txt"))
                .await
                .unwrap(),
            "same\nsame\n"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn patch_rejects_symlinks_and_hardlinks_for_every_mutation() {
    let dir = Directory::new();
    let outside = Directory::new();
    std::fs::write(outside.path().join("victim"), "secret\n").unwrap();
    std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
    std::fs::hard_link(outside.path().join("victim"), dir.path().join("hard")).unwrap();
    let tool = ApplyPatchTool::new(dir.0.clone());
    for body in [
        "*** Delete File: link/victim",
        "*** Add File: link/new\n+bad",
        "*** Update File: hard\n@@\n-secret\n+changed",
        "*** Delete File: hard",
    ] {
        assert!(
            tool.call(
                context(),
                ApplyPatchArgs {
                    patch: format!("*** Begin Patch\n{body}\n*** End Patch"),
                    ..Default::default()
                },
                vec![]
            )
            .await
            .is_err()
        );
    }
    assert_eq!(
        std::fs::read_to_string(outside.path().join("victim")).unwrap(),
        "secret\n"
    );
}

#[cfg(unix)]
async fn finish(
    runtime: &NativeRuntime,
    ctx: &BaseCtx,
    mut output: CommandOutput,
) -> CommandOutput {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while output.state == CommandState::Running {
        assert!(
            tokio::time::Instant::now() < deadline,
            "process failed to finish"
        );
        output = runtime
            .interact_session(
                ctx.clone(),
                SessionArgs {
                    task_id: Some(output.task_id),
                    yield_time_ms: Some(50),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .command
            .unwrap();
    }
    output
}

#[cfg(unix)]
#[tokio::test]
async fn sessions_poll_input_scopes_and_close_stdin() {
    let dir = Directory::new();
    let runtime = NativeRuntime::new(dir.0.clone()).session_limits(SessionLimits {
        allow_stdin: true,
        ..Default::default()
    });
    let ctx = context();
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "read value; printf 'result:%s' \"$value\"".into(),
                stdin: true,
                background: true,
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.state, CommandState::Running);
    let denied = runtime
        .interact_session(
            context(),
            SessionArgs {
                task_id: Some(output.task_id.clone()),
                ..Default::default()
            },
        )
        .await;
    assert!(denied.unwrap_err().to_string().contains("inaccessible"));
    let first = runtime
        .interact_session(
            ctx.clone(),
            SessionArgs {
                task_id: Some(output.task_id.clone()),
                action: SessionAction::Write,
                input: Some("hello\n".into()),
                yield_time_ms: Some(1000),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .command
        .unwrap();
    let text = first.output.stdout.clone().unwrap_or_default();
    let final_output = finish(&runtime, &ctx, first).await;
    assert_eq!(final_output.exit_code, Some(0));
    assert!(
        format!("{}{}", text, final_output.output.stdout.unwrap_or_default())
            .contains("result:hello")
    );
    let replay = runtime
        .interact_session(
            ctx.clone(),
            SessionArgs {
                task_id: Some(output.task_id),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .command
        .unwrap();
    assert_eq!(replay.output.stdout.as_deref(), Some(""));
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "cat >/dev/null".into(),
                stdin: true,
                background: true,
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    let closed = runtime
        .interact_session(
            ctx.clone(),
            SessionArgs {
                task_id: Some(output.task_id),
                action: SessionAction::CloseStdin,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .command
        .unwrap();
    assert_eq!(finish(&runtime, &ctx, closed).await.exit_code, Some(0));
}

#[cfg(unix)]
#[tokio::test]
async fn shell_exit_terminates_remaining_session_descendants() {
    // Cover descendants that close the pipes as well as those that keep them open.
    for redirect in [" >/dev/null 2>&1", ""] {
        let dir = Directory::new();
        let ctx = context();
        let runtime = NativeRuntime::new(dir.0.clone());
        let output = runtime
            .execute_session(
                ctx.clone(),
                CommandArgs {
                    command: format!("(sleep 0.5; printf escaped > escaped){redirect} &"),
                    ..Default::default()
                },
                Default::default(),
            )
            .await
            .unwrap();
        let output = finish(&runtime, &ctx, output).await;
        assert_eq!(output.state, CommandState::Exited);
        assert_eq!(output.exit_code, Some(0));
        assert!(output.log_complete);
        runtime
            .interact_session(
                ctx,
                SessionArgs {
                    task_id: Some(output.task_id),
                    action: SessionAction::Stop,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        drop(runtime);
        tokio::time::sleep(Duration::from_millis(650)).await;
        assert!(
            !dir.path().join("escaped").exists(),
            "descendant survived: {redirect:?}"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn runtime_deadline_session_cap_and_stop_are_enforced() {
    let dir = Directory::new();
    let runtime = NativeRuntime::new(dir.0.clone()).session_limits(SessionLimits {
        max_sessions: 1,
        max_runtime: Duration::from_millis(150),
        ..Default::default()
    });
    let ctx = context();
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "sleep 10".into(),
                background: true,
                timeout_ms: Some(30_000),
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    assert!(
        runtime
            .execute_session(
                ctx.clone(),
                CommandArgs {
                    command: "sleep 10".into(),
                    ..Default::default()
                },
                Default::default()
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("limit")
    );
    assert_eq!(
        finish(&runtime, &ctx, output).await.state,
        CommandState::TimedOut
    );
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "sleep 10".into(),
                background: true,
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    let stopped = runtime
        .interact_session(
            ctx.clone(),
            SessionArgs {
                task_id: Some(output.task_id),
                action: SessionAction::Stop,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .command
        .unwrap();
    assert_eq!(
        finish(&runtime, &ctx, stopped).await.state,
        CommandState::Cancelled
    );
}

#[cfg(unix)]
#[tokio::test]
async fn cancellation_retention_and_admission_do_not_leak_processes() {
    let dir = Directory::new();
    let runtime = NativeRuntime::new(dir.0.clone()).session_limits(SessionLimits {
        max_sessions: 1,
        retention: Duration::from_secs(10),
        ..Default::default()
    });
    let ctx = context();
    let running = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "sleep 10".into(),
                background: true,
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    assert!(
        runtime
            .execute_session(
                ctx.clone(),
                CommandArgs {
                    command: "touch forbidden".into(),
                    ..Default::default()
                },
                Default::default()
            )
            .await
            .is_err()
    );
    assert!(!dir.path().join("forbidden").exists());
    let mut other_agent = ctx.clone();
    other_agent.agent = "different-agent".into();
    assert!(
        runtime
            .interact_session(
                other_agent,
                SessionArgs {
                    task_id: Some(running.task_id.clone()),
                    action: SessionAction::Stop,
                    ..Default::default()
                }
            )
            .await
            .is_err()
    );
    ctx.cancellation_token().cancel();
    // A fresh request can query the same host-created scope after cancellation.
    let next = context();
    next.set_state(ctx.get_state::<ShellSessionScope>().unwrap());
    let finished = finish(&runtime, &next, running).await;
    assert_eq!(finished.state, CommandState::Cancelled);
    tokio::time::sleep(Duration::from_millis(20)).await;
    let retained = runtime
        .interact_session(
            next,
            SessionArgs {
                task_id: Some(finished.task_id),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(retained.command.unwrap().state, CommandState::Cancelled);
}

#[cfg(unix)]
#[tokio::test]
async fn changing_runtime_policy_invalidates_old_sessions() {
    let dir = Directory::new();
    let ctx = context();
    let runtime = NativeRuntime::new(dir.0.clone()).session_limits(SessionLimits {
        allow_stdin: true,
        ..Default::default()
    });
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "cat >/dev/null".into(),
                stdin: true,
                background: true,
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    let runtime = runtime.session_limits(SessionLimits::default());
    let result = runtime
        .interact_session(
            ctx,
            SessionArgs {
                task_id: Some(output.task_id),
                action: SessionAction::Write,
                input: Some("old authority\n".into()),
                ..Default::default()
            },
        )
        .await;
    assert!(result.unwrap_err().to_string().contains("inaccessible"));
}

#[cfg(target_os = "linux")]
#[tokio::test]
#[ignore = "requires bubblewrap and user namespaces; run explicitly for OS isolation validation"]
async fn process_sandbox_preserves_workspace_and_read_grants_under_tmp() {
    use anda_engine::extension::shell::SandboxPolicy;
    let root = PathBuf::from(format!(
        "/tmp/anda-sandbox-regression-{:032x}",
        rand::random::<u128>()
    ));
    std::fs::create_dir(&root).unwrap();
    let root = Directory(root.canonicalize().unwrap());
    let workspace = root.path().join("workspace");
    let readable = root.path().join("readable");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&readable).unwrap();
    std::fs::write(readable.join("input"), "granted").unwrap();
    let policy = SandboxPolicy::workspace(&workspace)
        .unwrap()
        .allow_read(&readable)
        .unwrap();
    let runtime = NativeRuntime::new(workspace.clone())
        .with_sandbox(policy)
        .unwrap();
    let ctx = context();
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "cat \"$INPUT\" > output && printf ':written' >> output".into(),
                ..Default::default()
            },
            [("INPUT".into(), readable.join("input").display().to_string())].into(),
        )
        .await
        .unwrap();
    let output = finish(&runtime, &ctx, output).await;
    assert_eq!(output.exit_code, Some(0), "{output:?}");
    assert_eq!(
        std::fs::read(workspace.join("output")).unwrap(),
        b"granted:written"
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires a host allowing sandbox-exec; run explicitly for OS isolation validation"]
async fn process_sandbox_denies_outside_reads_writes_and_network() {
    use anda_engine::extension::shell::SandboxPolicy;
    let dir = Directory::new();
    let outside = Directory::new();
    std::fs::write(outside.path().join("secret"), "sensitive").unwrap();
    // curl's system TLS library reads this configuration even for plain HTTP.
    // Both network policies receive the same explicit, read-only host grant.
    let policy = SandboxPolicy::workspace(dir.path())
        .unwrap()
        .allow_read("/private/etc/ssl/openssl.cnf")
        .unwrap();
    let runtime = NativeRuntime::new(dir.0.clone())
        .with_sandbox(policy.clone())
        .unwrap();
    let mut command = std::process::Command::new("/bin/sh");
    command.args(["-c", "printf ok > allowed; test -f allowed || exit 10; if cat \"$OUTSIDE/secret\"; then exit 11; fi; if printf bad > \"$OUTSIDE/new\"; then exit 12; fi; if /usr/bin/curl --noproxy '*' --max-time 1 http://127.0.0.1:$PORT/; then exit 13; fi; printf passed"]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        if let Ok((mut socket, _)) = listener.accept().await {
            use tokio::io::AsyncWriteExt;
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await;
        }
    });
    let result = runtime
        .execute_command(
            context(),
            "shell",
            command,
            [
                ("OUTSIDE".into(), outside.path().display().to_string()),
                ("PORT".into(), port.to_string()),
            ]
            .into(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(result.stdout.as_deref(), Some("passed"), "{result:?}");
    assert!(!outside.path().join("new").exists());
    // Prove that curl itself works under the same filesystem policy, so a
    // loader/configuration failure cannot masquerade as a denied connection.
    let permitted = NativeRuntime::new(dir.0.clone())
        .with_sandbox(policy.network(anda_engine::extension::shell::SandboxNetwork::Allow))
        .unwrap();
    let mut command = std::process::Command::new("/usr/bin/curl");
    command.args([
        "--noproxy",
        "*",
        "--max-time",
        "2",
        &format!("http://127.0.0.1:{port}/"),
    ]);
    let allowed = permitted
        .execute_command(context(), "shell", command, Default::default(), None)
        .await
        .unwrap();
    server.abort();
    assert_eq!(allowed.stdout.as_deref(), Some("ok"), "{allowed:?}");
    let terminal = runtime.session_limits(SessionLimits {
        allow_pty: true,
        ..Default::default()
    });
    let terminal_ctx = context();
    let output = terminal
        .execute_session(
            terminal_ctx.clone(),
            CommandArgs {
                command: "test -t 0 && printf terminal".into(),
                tty: true,
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    let text = output.output.stdout.clone().unwrap_or_default();
    let output = finish(&terminal, &terminal_ctx, output).await;
    assert_eq!(output.exit_code, Some(0), "{output:?}");
    assert!(format!("{text}{}", output.output.stdout.unwrap_or_default()).contains("terminal"));
}

#[cfg(unix)]
#[tokio::test]
async fn output_budget_logs_cwd_and_spawn_failures() {
    let dir = Directory::new();
    std::fs::create_dir(dir.path().join("nested")).unwrap();
    let runtime = NativeRuntime::new(dir.0.clone()).session_limits(SessionLimits {
        max_log_bytes: 128,
        ..Default::default()
    });
    let ctx = context();
    assert!(
        runtime
            .execute_session(
                ctx.clone(),
                CommandArgs {
                    command: "pwd".into(),
                    cwd: Some("..".into()),
                    ..Default::default()
                },
                Default::default()
            )
            .await
            .is_err()
    );
    let output = runtime.execute_session(ctx.clone(), CommandArgs { command: "printf START; i=0; while [ $i -lt 1000 ]; do printf x; i=$((i+1)); done; printf FAILED".into(), cwd: Some("nested".into()), max_output_bytes: Some(128), ..Default::default() }, Default::default()).await.unwrap();
    assert!(
        output.output.stdout.as_ref().unwrap().len()
            + output.output.stderr.as_ref().map_or(0, String::len)
            <= 128
    );
    let final_output = finish(&runtime, &ctx, output).await;
    let log = runtime
        .interact_session(
            ctx.clone(),
            SessionArgs {
                action: SessionAction::ReadLog,
                task_id: Some(final_output.task_id),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .log
        .unwrap();
    assert!(!log.complete);
    assert!(log.text.contains("START"));
    let missing = dir.path().join("missing");
    let result = runtime
        .execute_command(
            ctx,
            "shell",
            std::process::Command::new(missing),
            Default::default(),
            None,
        )
        .await;
    assert!(result.unwrap_err().to_string().contains("Failed to spawn"));
}

#[cfg(unix)]
#[tokio::test]
async fn pty_is_opt_in_and_reports_terminal_output() {
    let dir = Directory::new();
    let ctx = context();
    let args = CommandArgs {
        command: "test -t 0 && printf terminal".into(),
        tty: true,
        ..Default::default()
    };
    let disabled = NativeRuntime::new(dir.0.clone());
    assert!(
        disabled
            .execute_session(ctx.clone(), args.clone(), Default::default())
            .await
            .is_err()
    );
    let runtime = NativeRuntime::new(dir.0.clone()).session_limits(SessionLimits {
        allow_pty: true,
        ..Default::default()
    });
    let output = runtime
        .execute_session(ctx.clone(), args, Default::default())
        .await
        .unwrap();
    let text = output.output.stdout.clone().unwrap_or_default();
    let output = finish(&runtime, &ctx, output).await;
    assert_eq!(output.exit_code, Some(0));
    assert!(format!("{text}{}", output.output.stdout.unwrap_or_default()).contains("terminal"));
}

#[tokio::test]
async fn patch_deletes_binary_files() {
    let dir = Directory::new();
    std::fs::write(
        dir.path().join("logo.png"),
        [0x89, b'P', b'N', b'G', 0, 0xff],
    )
    .unwrap();
    let tool = ApplyPatchTool::new(dir.0.clone());
    let result = tool
        .call(
            context(),
            ApplyPatchArgs {
                patch: "*** Begin Patch\n*** Delete File: logo.png\n*** End Patch".into(),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap();
    assert!(result.output.changes[0].applied);
    assert!(result.output.diff.contains("Binary file deleted"));
    assert!(!dir.path().join("logo.png").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn foreground_wait_completes_commands_and_hooks_only_see_background_ones() {
    use anda_core::ToolOutput;
    use anda_engine::{
        extension::shell::{ExecArgs, ExecOutput, ShellToolHook},
        hook::{BackgroundHandle, ToolHook},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Events {
        start: AtomicUsize,
        end: AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ToolHook<ExecArgs, ExecOutput> for Events {
        async fn on_background_start(&self, _: &BaseCtx, _: BackgroundHandle, _: &ExecArgs) {
            self.start.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_background_end(&self, _: &BaseCtx, _: String, _: ToolOutput<ExecOutput>) {
            self.end.fetch_add(1, Ordering::SeqCst);
        }
    }

    let dir = Directory::new();
    let runtime = NativeRuntime::new(dir.0.clone());
    let ctx = context();
    let events = Arc::new(Events::default());
    ctx.set_state(ShellToolHook::new(events.clone()));
    // Early output must not end the foreground wait before the process exits.
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "printf a; sleep 0.2; printf b".into(),
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.state, CommandState::Exited);
    assert_eq!(output.output.stdout.as_deref(), Some("ab"));

    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: "sleep 10".into(),
                yield_time_ms: Some(50),
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.state, CommandState::Running);
    let stopped = runtime
        .interact_session(
            ctx.clone(),
            SessionArgs {
                task_id: Some(output.task_id),
                action: SessionAction::Stop,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .command
        .unwrap();
    finish(&runtime, &ctx, stopped).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while events.end.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(events.start.load(Ordering::SeqCst), 1);
    assert_eq!(events.end.load(Ordering::SeqCst), 1);
}

#[cfg(unix)]
#[tokio::test]
async fn read_log_keeps_multibyte_characters_whole() {
    let dir = Directory::new();
    let runtime = NativeRuntime::new(dir.0.clone());
    let ctx = context();
    let text = "汉字".repeat(10);
    let output = runtime
        .execute_session(
            ctx.clone(),
            CommandArgs {
                command: format!("printf '{text}'"),
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap();
    let output = finish(&runtime, &ctx, output).await;
    let (mut offset, mut log_text) = (0, String::new());
    loop {
        let log = runtime
            .interact_session(
                ctx.clone(),
                SessionArgs {
                    task_id: Some(output.task_id.clone()),
                    action: SessionAction::ReadLog,
                    max_output_bytes: Some(128),
                    offset,
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .log
            .unwrap();
        log_text.push_str(&log.text);
        offset = log.next_offset;
        if log.eof {
            break;
        }
    }
    assert!(!log_text.contains('\u{fffd}'), "{log_text:?}");
    assert!(log_text.contains(&text));
}

#[tokio::test]
async fn coding_bundle_registers_no_redundant_file_read_tools() {
    use anda_engine::extension::workspace::{coding_tools, readonly_file_tools};
    let dir = Directory::new();
    let tools = coding_tools(Arc::new(NativeRuntime::new(dir.0.clone())), vec![]).unwrap();
    let builder = EngineBuilder::new().register_tools(tools).unwrap();
    let ctx = builder.mock_ctx();
    use anda_core::AgentContext;
    let names = ctx
        .tool_definitions(None)
        .into_iter()
        .map(|definition| definition.name)
        .collect::<Vec<_>>();
    for name in ["shell", "shell_session", "apply_patch"] {
        assert!(names.iter().any(|actual| actual == name));
    }
    for name in ["read_file", "search_file", "write_file", "edit_file"] {
        assert!(!names.iter().any(|actual| actual == name));
    }
    let ctx = EngineBuilder::new()
        .register_tools(readonly_file_tools(dir.0.clone()).unwrap())
        .unwrap()
        .mock_ctx();
    assert!(
        !ctx.tool_definitions(None)
            .iter()
            .any(|definition| definition.name == "shell")
    );
    // Existing filesystem APIs remain independently usable.
    let tool = ReadFileTool::new(dir.0.clone());
    assert!(
        tool.call(
            context(),
            ReadFileArgs {
                path: "missing".into(),
                ..Default::default()
            },
            vec![]
        )
        .await
        .is_err()
    );
}
