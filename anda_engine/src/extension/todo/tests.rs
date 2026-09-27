use super::*;
use crate::engine::EngineBuilder;

fn input(id: &str, content: Option<&str>, status: Option<&str>) -> TodoItemInput {
    TodoItemInput {
        id: id.to_string(),
        content: content.map(ToString::to_string),
        status: status.map(ToString::to_string),
    }
}

fn mock_ctx() -> BaseCtx {
    EngineBuilder::new().mock_ctx().base
}

#[test]
fn set_dedupes_normalizes_and_preserves_order() {
    let mut store = TodoStore::default();

    let items = store
        .set(vec![
            input("1", Some("draft plan"), Some(TODO_STATUS_PENDING)),
            input("1", Some("final plan"), Some(TODO_STATUS_COMPLETED)),
        ])
        .unwrap();

    assert_eq!(
        items,
        vec![TodoItem {
            id: "1".to_string(),
            content: "final plan".to_string(),
            status: TODO_STATUS_COMPLETED.to_string(),
        }]
    );
}

#[test]
fn update_patches_existing_items_and_preserves_order() {
    let mut store = TodoStore::default();
    store
        .set(vec![
            input("1", Some("draft"), Some(TODO_STATUS_PENDING)),
            input("2", Some("implement"), Some(TODO_STATUS_PENDING)),
        ])
        .unwrap();

    let items = store
        .update(vec![
            input(
                "2",
                Some("implement todo tool"),
                Some(TODO_STATUS_IN_PROGRESS),
            ),
            input("3", Some("write tests"), Some(TODO_STATUS_PENDING)),
            input(
                "3",
                Some("write tests thoroughly"),
                Some(TODO_STATUS_COMPLETED),
            ),
        ])
        .unwrap();

    assert_eq!(
        items,
        vec![
            TodoItem {
                id: "1".to_string(),
                content: "draft".to_string(),
                status: TODO_STATUS_PENDING.to_string(),
            },
            TodoItem {
                id: "2".to_string(),
                content: "implement todo tool".to_string(),
                status: TODO_STATUS_IN_PROGRESS.to_string(),
            },
            TodoItem {
                id: "3".to_string(),
                content: "write tests thoroughly".to_string(),
                status: TODO_STATUS_COMPLETED.to_string(),
            },
        ]
    );
}

#[test]
fn injection_format_only_includes_active_items() {
    let mut store = TodoStore::default();
    store
        .set(vec![
            input("1", Some("plan"), Some(TODO_STATUS_PENDING)),
            input("2", Some("build"), Some(TODO_STATUS_IN_PROGRESS)),
            input("3", Some("done"), Some(TODO_STATUS_COMPLETED)),
            input("4", Some("skip"), Some(TODO_STATUS_CANCELLED)),
        ])
        .unwrap();

    let injected = store.format_for_injection().unwrap();
    assert!(injected.contains(TODO_ACTIVE_LIST_PREFIX));
    assert!(injected.contains("- [ ] 1. plan (pending)"));
    assert!(injected.contains("- [>] 2. build (in_progress)"));
    assert!(!injected.contains("done"));
    assert!(!injected.contains("skip"));
}

#[tokio::test]
async fn tool_call_persists_session_state() {
    let ctx = mock_ctx();
    let tool = TodoTool::new();

    let first = tool
        .call(
            ctx.clone(),
            TodoArgs {
                op: Some(TODO_OP_SET.to_string()),
                items: Some(vec![input("1", Some("plan"), Some(TODO_STATUS_PENDING))]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(first.output.summary.total, 1);
    assert!(first.output.items.is_empty());
    assert!(todo_session(&ctx).has_items());

    let second = tool
        .call(ctx.clone(), TodoArgs::default(), Vec::new())
        .await
        .unwrap();
    assert_eq!(
        second.output.items,
        vec![TodoItem {
            id: "1".to_string(),
            content: "plan".to_string(),
            status: TODO_STATUS_PENDING.to_string(),
        }]
    );

    let third = tool
        .call(
            ctx.clone(),
            TodoArgs {
                op: Some(TODO_OP_UPDATE.to_string()),
                items: Some(vec![TodoItemInput {
                    id: "1".to_string(),
                    content: Some("plan carefully".to_string()),
                    status: None,
                }]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(third.output.summary.pending, 1);
    assert!(third.output.items.is_empty());

    let fourth = tool
        .call(
            ctx.clone(),
            TodoArgs {
                op: Some(TODO_OP_READ.to_string()),
                items: None,
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(fourth.output.items[0].content, "plan carefully");
}

#[tokio::test]
async fn session_state_persists_across_per_call_child_contexts() {
    // Mirror how the completion runner drives tools: a long-lived parent
    // context seeds the session, and each tool invocation runs on a fresh
    // `child_base` context that snapshot-copies parent state. Without the
    // seeded session each child would create its own empty store and writes
    // would be lost between calls.
    let parent = EngineBuilder::new().mock_ctx();
    parent.base.set_state(TodoSession::new());
    let tool = TodoTool::new();

    let write_ctx = parent.child_base("todo").unwrap();
    let first = tool
        .call(
            write_ctx,
            TodoArgs {
                op: Some(TODO_OP_SET.to_string()),
                items: Some(vec![input("1", Some("plan"), Some(TODO_STATUS_PENDING))]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(first.output.summary.total, 1);

    // A distinct child context (as produced by a later tool call) must
    // observe the write performed by the previous one.
    let read_ctx = parent.child_base("todo").unwrap();
    let second = tool
        .call(
            read_ctx,
            TodoArgs {
                op: Some(TODO_OP_READ.to_string()),
                items: None,
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        second.output.items,
        vec![TodoItem {
            id: "1".to_string(),
            content: "plan".to_string(),
            status: TODO_STATUS_PENDING.to_string(),
        }]
    );
}

#[test]
fn invalid_batches_leave_the_entire_list_unchanged() {
    let mut store = TodoStore::default();
    store
        .set(vec![input("old", Some("keep"), Some("completed"))])
        .unwrap();
    let before = store.snapshot();
    for bad in [
        input("", Some("bad"), None),
        input("bad", Some(""), None),
        input("bad", Some("new"), Some("done")),
        input("new", None, None),
    ] {
        assert!(
            store
                .update(vec![input("old", Some("changed"), None), bad.clone()])
                .is_err()
        );
        assert_eq!(store.snapshot(), before);
        assert!(store.set(vec![bad]).is_err());
        assert_eq!(store.snapshot(), before);
    }
    assert!(
        store
            .set(vec![
                input("old", None, None),
                input("old", Some("replacement"), None)
            ])
            .is_err()
    );
    assert_eq!(store.snapshot(), before);
    // Even a shadowed duplicate must validate.
    assert!(
        store
            .update(vec![
                input("old", None, Some("done")),
                input("old", None, Some("pending"))
            ])
            .is_err()
    );
    assert_eq!(store.snapshot(), before);
    store
        .update(vec![input(
            " old ",
            Some(" changed "),
            Some(" IN_PROGRESS "),
        )])
        .unwrap();
    assert_eq!(
        store.snapshot(),
        vec![TodoItem {
            id: "old".into(),
            content: "changed".into(),
            status: "in_progress".into()
        }]
    );
}

#[tokio::test]
async fn tool_rejects_missing_items_unknown_ops_and_invalid_statuses() {
    let ctx = mock_ctx();
    let session = todo_session(&ctx);
    let initial = session
        .set(vec![input("a", Some("keep"), Some("completed"))])
        .unwrap();
    let tool = TodoTool::new();
    for args in [
        serde_json::json!({"op":"set"}),
        serde_json::json!({"op":"set", "items":null}),
        serde_json::json!({"op":"update"}),
        serde_json::json!({"op":"typo"}),
        serde_json::json!({"op":"read", "items":[]}),
        serde_json::json!({"op":"update", "items":[{"id":"a","status":"done"}]}),
    ] {
        let result = tool.call_raw(ctx.clone(), args, vec![]).await.unwrap();
        assert_eq!(result.is_error, Some(true));
        assert!(result.output.get("error").is_some());
        assert_eq!(session.snapshot(), initial);
    }
    assert!(
        tool.call_raw(ctx.clone(), serde_json::json!({"operation":"set"}), vec![])
            .await
            .is_err()
    );
    assert!(
        tool.call_raw(
            ctx.clone(),
            serde_json::json!({"op":"set","items":[{"id":"a","content":"x","statuz":"pending"}]}),
            vec![]
        )
        .await
        .is_err()
    );
    let result = tool
        .call_raw(
            ctx.clone(),
            serde_json::json!({"op":"set","items":[]}),
            vec![],
        )
        .await
        .unwrap();
    assert_ne!(result.is_error, Some(true));
    assert!(session.snapshot().is_empty());
}

#[test]
fn task_limits_and_injection_are_bounded_for_unicode() {
    let mut store = TodoStore::default();
    store
        .set(
            (0..40)
                .map(|i| input(&i.to_string(), Some(&"中🙂".repeat(40)), None))
                .collect(),
        )
        .unwrap();
    let before = store.snapshot();
    let rendered = store.format_for_injection().unwrap();
    assert!(rendered.len() <= INJECTION_BYTES);
    assert!(rendered.contains("more active tasks omitted"));
    assert!(
        store
            .set(
                (0..257)
                    .map(|i| input(&i.to_string(), Some("x"), None))
                    .collect()
            )
            .is_err()
    );
    assert!(
        store
            .update(vec![input("0", Some(&"x".repeat(4097)), None)])
            .is_err()
    );
    assert_eq!(store.snapshot(), before);
    store
        .set(vec![input("a", Some("all done"), Some("completed"))])
        .unwrap();
    assert_eq!(store.format_for_injection(), None);
}

#[tokio::test]
async fn existing_hook_observes_explanation_and_committed_session() {
    use crate::hook::ToolHook;
    use async_trait::async_trait;
    type Observations = Arc<parking_lot::Mutex<Vec<(TodoOutput, Vec<TodoItem>)>>>;
    struct Observer(Observations);
    #[async_trait]
    impl ToolHook<TodoArgs, TodoOutput> for Observer {
        async fn after_tool_call(
            &self,
            ctx: &BaseCtx,
            output: ToolOutput<TodoOutput>,
        ) -> Result<ToolOutput<TodoOutput>, BoxError> {
            self.0
                .lock()
                .push((output.output.clone(), todo_session(ctx).snapshot()));
            Ok(output)
        }
    }
    let ctx = mock_ctx();
    let observed = Arc::new(parking_lot::Mutex::new(Vec::new()));
    ctx.set_state(TodoToolHook::new(Arc::new(Observer(observed.clone()))));
    let output = TodoTool::new()
        .call(
            ctx,
            TodoArgs {
                op: Some("set".into()),
                items: Some(vec![input("a", Some("implement"), None)]),
                explanation: Some("Split implementation from validation".into()),
            },
            vec![],
        )
        .await
        .unwrap();
    assert!(output.output.items.is_empty());
    let observed = observed.lock();
    assert_eq!(observed.len(), 1);
    assert_eq!(
        observed[0].0.explanation.as_deref(),
        Some("Split implementation from validation")
    );
    assert_eq!(observed[0].1[0].content, "implement");
}
