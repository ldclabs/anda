use super::*;
use std::time::Duration;

fn context() -> AgentCtx {
    EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(HistoryCompleter)))
        .mock_ctx()
}
fn scope(ctx: &AgentCtx) -> SubAgentScope {
    ctx.base.get_state().unwrap()
}
fn worker() -> SubAgent {
    SubAgent {
        name: "worker".into(),
        ..Default::default()
    }
}
async fn invoke(
    agent: &SubAgent,
    ctx: &AgentCtx,
    prompt: &str,
    alias: &str,
) -> Result<AgentOutput, BoxError> {
    agent
        .run(
            ctx.clone(),
            serde_json::to_string(&SubAgentArgs {
                prompt: prompt.into(),
                session: alias.into(),
                ..Default::default()
            })?,
            vec![],
        )
        .await
}
async fn completed(scope: &SubAgentScope, id: &str, after: u64) -> SubAgentEvents {
    let events = scope
        .wait(
            after,
            &[id.into()],
            WaitMode::All,
            Duration::from_secs(3),
            anda_core::CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(!events.timed_out, "missing completion for {id}");
    assert!(
        events
            .events
            .iter()
            .any(|e| e.kind == SubAgentEventKind::TurnCompleted),
        "{:?}",
        events.events
    );
    events
}
fn session(agent: &SubAgent, ctx: &AgentCtx, alias: &str) -> Arc<SubSession> {
    agent
        .subsessions
        .get_session_in_scope(ctx.caller(), &scope(ctx), alias)
        .unwrap()
}

#[tokio::test]
async fn roots_isolate_aliases_and_turn_completion_keeps_session_alive() {
    let a = context();
    let b = context();
    let worker = worker();
    invoke(&worker, &a, "root a", "same").await.unwrap();
    invoke(&worker, &b, "root b", "same").await.unwrap();
    let sa = session(&worker, &a, "same");
    let sb = session(&worker, &b, "same");
    assert_ne!(sa.execution.id, sb.execution.id);
    assert!(
        worker
            .subsessions
            .get_session_for(a.caller(), "same")
            .is_none()
    );
    let ea = completed(&scope(&a), &sa.execution.id, 0).await;
    completed(&scope(&b), &sb.execution.id, 0).await;
    assert!(!sa.sender.is_closed());
    assert_eq!(sa.work_turn.load(Ordering::SeqCst), 1);
    assert!(
        ea.events
            .iter()
            .all(|e| e.execution.root_id == scope(&a).id())
    );
    invoke(&worker, &a, "follow up", "same").await.unwrap();
    completed(&scope(&a), &sa.execution.id, ea.cursor).await;
    assert_eq!(sa.work_turn.load(Ordering::SeqCst), 2);
    sa.close();
    sb.close();
}

#[tokio::test]
async fn queue_only_waits_for_followup_and_overflow_is_explicit() {
    let ctx = context();
    let limits = SubAgentLimits {
        max_pending_messages: 3,
        ..Default::default()
    };
    ctx.base.set_state(SubAgentScope::new(limits));
    let worker = worker();
    invoke(&worker, &ctx, "initial", "job").await.unwrap();
    let session = session(&worker, &ctx, "job");
    let done = completed(&scope(&ctx), &session.execution.id, 0).await;
    invoke(&worker, &ctx, "/message useful background", "job")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(session.work_turn.load(Ordering::SeqCst), 1);
    assert_eq!(session.mailbox_len(), 1);
    let message = |id: &str| SubAgentMessage {
        sender: "host".into(),
        id: id.into(),
        content: "data".into(),
        resources: vec![],
    };
    session
        .send(message("two"), MessageDelivery::QueueOnly)
        .unwrap();
    assert!(
        session
            .send(message("three"), MessageDelivery::QueueOnly)
            .unwrap_err()
            .to_string()
            .contains("full")
    );
    // A slot is reserved for the explicit follow-up, so queued notifications cannot prevent waking.
    invoke(&worker, &ctx, "process the background", "job")
        .await
        .unwrap();
    let next = completed(&scope(&ctx), &session.execution.id, done.cursor).await;
    let summary = next.events.last().unwrap().summary.as_ref().unwrap();
    assert!(summary.contains("useful background"));
    assert!(summary.contains("not user authorization"));
    session.close();
}

#[tokio::test]
async fn waits_have_no_lost_wakeup_and_report_retention_gaps_and_cancellation() {
    let scope = SubAgentScope::new(SubAgentLimits {
        max_events: 2,
        ..Default::default()
    });
    let identity = scope.identity(None);
    let event = |kind| SubAgentEvent {
        sequence: 0,
        execution: identity.clone(),
        agent: "a".into(),
        session: "s".into(),
        turn: 1,
        kind,
        summary: None,
    };
    scope.publish(event(SubAgentEventKind::Started));
    let task = {
        let scope = scope.clone();
        let id = identity.id.clone();
        tokio::spawn(async move {
            scope
                .wait(
                    1,
                    &[id],
                    WaitMode::All,
                    Duration::from_secs(1),
                    anda_core::CancellationToken::new(),
                )
                .await
        })
    };
    scope.publish(event(SubAgentEventKind::TurnCompleted));
    assert!(!task.await.unwrap().unwrap().timed_out);
    scope.publish(event(SubAgentEventKind::Closed));
    assert!(scope.events(0, &[]).lagged);
    let token = anda_core::CancellationToken::new();
    token.cancel();
    assert!(
        scope
            .wait(3, &[], WaitMode::Any, Duration::from_secs(1), token)
            .await
            .is_err()
    );
    assert!(
        scope
            .wait(
                3,
                &[],
                WaitMode::Any,
                Duration::ZERO,
                anda_core::CancellationToken::new()
            )
            .await
            .unwrap()
            .timed_out
    );
}

#[tokio::test]
async fn terminal_status_survives_session_cleanup() {
    let ctx = context();
    let worker = worker();
    invoke(&worker, &ctx, "initial", "job").await.unwrap();
    let session = session(&worker, &ctx, "job");
    let done = completed(&scope(&ctx), &session.execution.id, 0).await;
    invoke(&worker, &ctx, "/cancel operator cancelled", "job")
        .await
        .unwrap();
    let events = scope(&ctx)
        .wait(
            done.cursor,
            std::slice::from_ref(&session.execution.id),
            WaitMode::All,
            Duration::from_secs(1),
            ctx.cancellation_token(),
        )
        .await
        .unwrap();
    assert!(
        events
            .events
            .iter()
            .any(|e| e.kind == SubAgentEventKind::Failed)
    );
    tokio::task::yield_now().await;
    let status: Json = serde_json::from_str(
        &invoke(&worker, &ctx, "/status", "job")
            .await
            .unwrap()
            .content,
    )
    .unwrap();
    assert_eq!(status["active"], false);
    assert_eq!(status["failed_reason"], "operator cancelled");
    assert_eq!(status["execution"]["id"], session.execution.id);
}

#[tokio::test]
async fn atomic_admission_and_raii_release_share_root_budget() {
    let scope = SubAgentScope::new(SubAgentLimits {
        max_parallel_requests: 1,
        max_sessions: 1,
        max_requests: Some(2),
        max_tokens: Some(10),
        ..Default::default()
    });
    let permit = scope.admit_request().await.unwrap();
    // A busy scope queues the request; abandoning the wait admits nothing.
    assert!(
        tokio::time::timeout(Duration::from_millis(20), scope.admit_request())
            .await
            .is_err()
    );
    assert_eq!(scope.admitted_requests(), 1);
    drop(permit);
    let permit = scope.admit_request().await.unwrap();
    drop(permit);
    assert!(scope.admit_request().await.is_err());
    let resident = scope.reserve_session().unwrap();
    assert!(scope.reserve_session().is_err());
    drop(resident);
    assert!(scope.reserve_session().is_ok());
    let scope = SubAgentScope::new(SubAgentLimits {
        max_tokens: Some(5),
        ..Default::default()
    });
    scope.record_usage(&Usage {
        input_tokens: 3,
        output_tokens: 2,
        ..Default::default()
    });
    assert!(scope.admit_request().await.is_err());
    assert!(scope.bind_caller(Principal::anonymous()).is_ok());
    assert!(scope.bind_caller(Principal::management_canister()).is_err());
}

#[tokio::test]
async fn request_budget_counts_real_inference_and_compaction_once() {
    let ctx = context();
    let scope = SubAgentScope::new(SubAgentLimits {
        max_requests: Some(2),
        ..Default::default()
    });
    ctx.base.set_state(scope.clone());
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                prompt: "task".into(),
                ..Default::default()
            },
            vec![],
        )
        .unbound();
    runner.next().await.unwrap();
    // Reported nested usage is display accounting, never a second charge to the scope.
    runner.accumulate(&Usage {
        input_tokens: 1000,
        ..Default::default()
    });
    let (mut runner, _) = runner.handoff(None).await.unwrap();
    assert_eq!(scope.admitted_requests(), 2);
    assert_eq!(scope.usage().input_tokens, 4);
    runner.follow_up("more".to_string());
    assert!(
        runner
            .next()
            .await
            .unwrap_err()
            .to_string()
            .contains("budget")
    );
}

#[tokio::test]
async fn initialization_failure_releases_residency_and_alias() {
    let ctx = context();
    let scope = SubAgentScope::new(SubAgentLimits {
        max_sessions: 1,
        ..Default::default()
    });
    ctx.base.set_state(scope.clone());
    #[derive(Default)]
    struct Reject;
    #[async_trait]
    impl AgentHook for Reject {
        async fn after_agent_run(
            &self,
            _ctx: &AgentCtx,
            _out: AgentOutput,
        ) -> Result<AgentOutput, BoxError> {
            Err("reject acknowledgement".into())
        }
    }
    let worker = worker();
    let rejected = ctx.child("worker", "worker").unwrap();
    rejected.base.set_state(DynAgentHook::new(Arc::new(Reject)));
    assert!(invoke(&worker, &rejected, "task", "same").await.is_err());
    assert!(
        worker
            .subsessions
            .get_session_in_scope(ctx.caller(), &scope, "same")
            .is_none()
    );
    invoke(&worker, &ctx, "task", "same").await.unwrap();
    session(&worker, &ctx, "same").close();
}

#[tokio::test]
async fn idle_checkpoint_restores_identity_history_and_current_whitelist() {
    let ctx = context();
    let checkpoints =
        SubAgentCheckpoints::object_store(Arc::new(InMemory::new()), Path::from("checkpoints"));
    ctx.base.set_state(checkpoints.clone());
    let worker = worker();
    invoke(&worker, &ctx, "remember the first task", "job")
        .await
        .unwrap();
    let first = session(&worker, &ctx, "job");
    completed(&scope(&ctx), &first.execution.id, 0).await;
    let key = SubAgentCheckpoints::key(ctx.caller(), scope(&ctx).id(), "worker", "job");
    let saved = checkpoints.load(&key).await.unwrap().unwrap();
    assert!(
        saved
            .history
            .iter()
            .any(|m| m.text().is_some_and(|s| s.contains("first task")))
    );
    // Idle expiry, unlike cancellation, leaves the resumable checkpoint intact.
    first.active_at.store(1, Ordering::SeqCst);
    scope(&ctx)
        .wait(
            scope(&ctx).events(0, &[]).cursor,
            std::slice::from_ref(&first.execution.id),
            WaitMode::All,
            Duration::from_secs(3),
            ctx.cancellation_token(),
        )
        .await
        .unwrap();
    tokio::task::yield_now().await;
    let second_ctx = context();
    second_ctx.base.set_state(SubAgentScope::restore(
        scope(&ctx).id().into(),
        SubAgentLimits::default(),
    ));
    second_ctx.base.set_state(checkpoints.clone());
    // Recreate the definition: the checkpoint never restores old instructions/tools.
    let restored_worker = SubAgent {
        instructions: "new host instructions".into(),
        name: "worker".into(),
        ..Default::default()
    };
    invoke(&restored_worker, &second_ctx, "continue", "job")
        .await
        .unwrap();
    let second = session(&restored_worker, &second_ctx, "job");
    assert_eq!(second.execution.id, first.execution.id);
    completed(&scope(&second_ctx), &second.execution.id, 0).await;
    let saved = checkpoints.load(&key).await.unwrap().unwrap();
    assert!(
        saved
            .history
            .iter()
            .any(|m| m.text().is_some_and(|s| s.contains("first task")))
    );
    assert_eq!(saved.turn, 2);
    invoke(&restored_worker, &second_ctx, "/cancel done", "job")
        .await
        .unwrap();
}

#[tokio::test]
async fn nested_turn_result_preserves_artifacts_without_duplicate_session_end() {
    let ctx = context();
    let (parent, mut inbox) = test_session("parent");
    let child = ctx.child("child", "child").unwrap();
    child.base.set_state(scope(&ctx).identity(None));
    AgentHook::on_background_start(
        parent.as_ref(),
        &child,
        BackgroundHandle::new("child:job", anda_core::CancellationToken::new()),
        &CompletionRequest::default(),
    )
    .await;
    let output = AgentOutput {
        content: "done".into(),
        artifacts: vec![resource(7, &["text"])],
        usage: Usage {
            input_tokens: 4,
            requests: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    AgentHook::on_background_turn_end(
        parent.as_ref(),
        &child,
        "child:job".into(),
        1,
        output.clone(),
    )
    .await;
    let input = inbox.recv().await.unwrap();
    assert_eq!(input.resources.len(), 1);
    assert_eq!(input.resources[0]._id, 7);
    assert_eq!(input.usage.input_tokens, 4);
    assert_eq!(parent.detail()["background_tasks"][0]["idle"], true);
    assert!(parent.controls.get("child:job").is_some());
    AgentHook::on_background_end(parent.as_ref(), &child, "child:job".into(), output).await;
    assert!(inbox.try_recv().is_err());
    assert!(parent.controls.is_empty());
}

#[test]
fn handoff_keeps_pairs_and_rejects_partial_or_privileged_history() {
    let history = vec![
        Message {
            role: "user".into(),
            content: vec!["old".to_string().into()],
            ..Default::default()
        },
        Message {
            role: "assistant".into(),
            content: vec![ContentPart::ToolCall {
                name: "read".into(),
                args: json!({}),
                call_id: Some("a".into()),
            }],
            ..Default::default()
        },
        Message {
            role: "tool".into(),
            content: vec![ContentPart::ToolOutput {
                name: "read".into(),
                output: json!("result"),
                call_id: Some("a".into()),
                is_error: None,
                remote_id: None,
            }],
            ..Default::default()
        },
        Message {
            role: "user".into(),
            content: vec!["new".to_string().into()],
            ..Default::default()
        },
    ];
    assert_eq!(
        SubAgentHandoff::last_turns(&history, 1, 4096)
            .unwrap()
            .messages
            .len(),
        2
    );
    assert_eq!(
        SubAgentHandoff::last_turns(&history, 2, 4096)
            .unwrap()
            .messages
            .len(),
        5
    );
    assert!(SubAgentHandoff::messages(history[2..].to_vec(), 4096).is_err());
    assert!(SubAgentHandoff::messages(history[..2].to_vec(), 4096).is_err());
    assert!(SubAgentHandoff::summary("large".repeat(100), 32).is_err());
    assert!(
        SubAgentHandoff::messages(
            vec![Message {
                role: "system".into(),
                ..Default::default()
            }],
            4096
        )
        .is_err()
    );
}

#[tokio::test]
async fn cancelled_idle_checkpoint_cannot_resurrect_the_cancelled_task() {
    let ctx = context();
    let checkpoints =
        SubAgentCheckpoints::object_store(Arc::new(InMemory::new()), Path::from("saved"));
    ctx.base.set_state(checkpoints.clone());
    let worker = worker();
    invoke(&worker, &ctx, "task", "job").await.unwrap();
    let session = session(&worker, &ctx, "job");
    let done = completed(&scope(&ctx), &session.execution.id, 0).await;
    let key = SubAgentCheckpoints::key(ctx.caller(), scope(&ctx).id(), "worker", "job");
    assert!(checkpoints.load(&key).await.unwrap().is_some());
    invoke(&worker, &ctx, "/cancel do not resume", "job")
        .await
        .unwrap();
    scope(&ctx)
        .wait(
            done.cursor,
            std::slice::from_ref(&session.execution.id),
            WaitMode::All,
            Duration::from_secs(1),
            ctx.cancellation_token(),
        )
        .await
        .unwrap();
    assert!(checkpoints.load(&key).await.unwrap().is_none());
}

#[tokio::test]
async fn root_deadline_cancels_pending_requests_and_releases_permits() {
    #[derive(Debug)]
    struct Pending;
    impl CompletionFeaturesDyn for Pending {
        fn model_name(&self) -> String {
            "pending".into()
        }
        fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(std::future::pending())
        }
    }
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(Pending)))
        .mock_ctx();
    let scope = SubAgentScope::new(SubAgentLimits {
        deadline_ms: Some(unix_ms() + 30),
        ..Default::default()
    });
    ctx.base.set_state(scope.clone());
    let mut runner = ctx.completion_iter(
        CompletionRequest {
            prompt: "task".into(),
            ..Default::default()
        },
        vec![],
    );
    let output = tokio::time::timeout(Duration::from_secs(1), runner.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        output.failed_reason.as_deref(),
        Some("subagent root deadline exceeded")
    );
    assert_eq!(scope.admitted_requests(), 1);
    assert!(runner.is_done());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_admission_waits_without_exceeding_capacity() {
    let scope = SubAgentScope::new(SubAgentLimits {
        max_parallel_requests: 2,
        ..Default::default()
    });
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tasks = (0..16)
        .map(|_| {
            let (scope, active, peak) = (scope.clone(), active.clone(), peak.clone());
            tokio::spawn(async move {
                let _permit = scope.admit_request().await.unwrap();
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
                active.fetch_sub(1, Ordering::SeqCst);
            })
        })
        .collect::<Vec<_>>();
    for task in tasks {
        task.await.unwrap();
    }
    assert!(peak.load(Ordering::SeqCst) <= 2);
    assert_eq!(scope.admitted_requests(), 16);
}

#[tokio::test]
async fn stale_generation_callback_cannot_remove_reused_child_handle() {
    let ctx = context();
    let (parent, mut inbox) = test_session("parent");
    let old = ctx.child("worker", "worker").unwrap();
    old.base.set_state(scope(&ctx).identity(None));
    let new = ctx.child("worker", "worker").unwrap();
    new.base.set_state(scope(&ctx).identity(None));
    for child in [&old, &new] {
        AgentHook::on_background_start(
            parent.as_ref(),
            child,
            BackgroundHandle::new("worker:job", anda_core::CancellationToken::new()),
            &CompletionRequest::default(),
        )
        .await;
    }
    AgentHook::on_background_end(
        parent.as_ref(),
        &old,
        "worker:job".into(),
        AgentOutput {
            content: "stale".into(),
            ..Default::default()
        },
    )
    .await;
    assert!(parent.controls.get("worker:job").is_some());
    assert!(inbox.try_recv().is_err());
    AgentHook::on_background_end(
        parent.as_ref(),
        &new,
        "worker:job".into(),
        AgentOutput {
            content: "current".into(),
            ..Default::default()
        },
    )
    .await;
    assert!(recv_subagent_prompt(&mut inbox).await.contains("current"));
}

#[tokio::test]
async fn model_controls_validate_arguments_and_stale_wait_cursors() {
    let ctx = context();
    let worker = worker();
    assert!(
        invoke(&worker, &ctx, "/message hello", "missing")
            .await
            .is_err()
    );
    assert!(worker.subsessions.active_session_ids().is_empty());
    invoke(&worker, &ctx, "task", "job").await.unwrap();
    let session = session(&worker, &ctx, "job");
    completed(&scope(&ctx), &session.execution.id, 0).await;
    assert!(
        invoke(&worker, &ctx, "/wait invalid 1", "job")
            .await
            .is_err()
    );
    let events: Json = serde_json::from_str(
        &invoke(&worker, &ctx, "/wait 0 0", "job")
            .await
            .unwrap()
            .content,
    )
    .unwrap();
    assert_eq!(events["timed_out"], false);
    assert!(!events["events"].as_array().unwrap().is_empty());
    let events: Json = serde_json::from_str(
        &invoke(&worker, &ctx, "/wait 9999999 0", "job")
            .await
            .unwrap()
            .content,
    )
    .unwrap();
    assert_eq!(events["lagged"], true);
    session.close();
}

#[tokio::test]
async fn oversized_input_does_not_claim_a_session() {
    let ctx = context();
    let worker = worker();
    ctx.base.set_state(SubAgentScope::new(SubAgentLimits {
        max_message_bytes: 32,
        ..Default::default()
    }));
    assert!(invoke(&worker, &ctx, &"x".repeat(33), "job").await.is_err());
    assert!(worker.subsessions.active_session_ids().is_empty());
}

#[tokio::test]
async fn empty_ping_keeps_session_alive_without_fabricating_a_completed_turn() {
    let ctx = context();
    let worker = worker();
    invoke(&worker, &ctx, "task", "job").await.unwrap();
    let session = session(&worker, &ctx, "job");
    let done = completed(&scope(&ctx), &session.execution.id, 0).await;
    invoke(&worker, &ctx, "", "job").await.unwrap();
    let events = scope(&ctx)
        .wait(
            done.cursor,
            std::slice::from_ref(&session.execution.id),
            WaitMode::Any,
            Duration::from_millis(20),
            ctx.cancellation_token(),
        )
        .await
        .unwrap();
    assert!(events.timed_out);
    assert_eq!(session.work_turn.load(Ordering::SeqCst), 1);
    session.close();
}

#[test]
fn closing_runtime_keeps_alias_reserved_until_its_final_callbacks_finish() {
    let sessions = SubSessions::default();
    let (old, old_rx) = test_session("job");
    *old.leased.lock() = true;
    sessions.try_insert_session(old.clone());
    drop(old_rx);
    assert!(sessions.active_session_ids().is_empty());
    let (new, _new_rx) = test_session("job");
    assert!(Arc::ptr_eq(
        &sessions.try_insert_session(new.clone()).unwrap(),
        &old
    ));
    *old.leased.lock() = false;
    sessions.remove_session_if(&old);
    assert!(sessions.try_insert_session(new).is_none());
}

#[tokio::test]
async fn oversized_background_results_are_delivered_without_cancelling_the_session() {
    let ctx = context();
    let (parent, mut inbox) = test_session("parent");
    ToolBackgroundHook::on_background_start(
        parent.as_ref(),
        &ctx.base,
        BackgroundHandle::new("shell:big", anda_core::CancellationToken::new()),
        Json::Null,
    )
    .await;
    // Background results are produced by the session's own tools, not by the caller, so the
    // per-input byte limit must not turn a large result into a session cancellation.
    let output = "x".repeat(SubAgentLimits::default().max_message_bytes * 2);
    ToolBackgroundHook::on_background_end(
        parent.as_ref(),
        &ctx.base,
        "shell:big".into(),
        ToolOutput::new(json!(output)),
    )
    .await;
    let input = inbox.try_recv().unwrap();
    assert!(
        matches!(input.command, PromptCommand::Plain { ref prompt } if prompt.contains(&output))
    );
    assert!(!parent.has_control());
}

#[tokio::test]
async fn nested_turn_result_reaches_the_parent_once() {
    let ctx = context();
    let (parent, mut inbox) = test_session("parent");
    ctx.base.set_state(DynAgentHook::new(parent.clone()));
    let worker = worker();
    invoke(&worker, &ctx, "task", "job").await.unwrap();
    let child = session(&worker, &ctx, "job");
    let mut prompts: Vec<String> = Vec::new();
    while !prompts
        .iter()
        .any(|prompt| prompt.contains("turn completed"))
    {
        let input = tokio::time::timeout(Duration::from_secs(3), inbox.recv())
            .await
            .unwrap()
            .unwrap();
        if let PromptCommand::Plain { prompt } = input.command {
            prompts.push(prompt);
        }
    }
    assert_eq!(
        prompts
            .iter()
            .filter(|prompt| prompt.contains("done: task"))
            .count(),
        1,
        "{prompts:?}"
    );
    child.close();
}
