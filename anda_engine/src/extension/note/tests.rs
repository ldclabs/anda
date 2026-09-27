use super::*;
use crate::{context::AgentCtx, engine::EngineBuilder, hook::ToolHook};
use async_trait::async_trait;
use std::sync::Arc;

fn agent_ctx(name: &str) -> AgentCtx {
    EngineBuilder::new()
        .mock_ctx()
        .child(name, name)
        .expect("create child agent ctx")
}

fn note_ctx(name: &str) -> BaseCtx {
    agent_ctx(name)
        .child_base(NoteTool::NAME)
        .expect("create note tool ctx")
}

fn input(id: &str, content: Option<&str>) -> NoteItemInput {
    NoteItemInput {
        id: id.to_string(),
        content: content.map(ToString::to_string),
    }
}

fn item(id: &str, content: &str) -> NoteItem {
    NoteItem {
        id: id.to_string(),
        content: content.to_string(),
    }
}

struct MutatingHook;

#[async_trait]
impl ToolHook<NoteArgs, NoteOutput> for MutatingHook {
    async fn before_tool_call(
        &self,
        _ctx: &BaseCtx,
        mut args: NoteArgs,
    ) -> Result<NoteArgs, BoxError> {
        args.op = Some(NOTE_OP_UPSERT.to_string());
        args.items = Some(vec![input("hook", Some("hook inserted note"))]);
        Ok(args)
    }

    async fn after_tool_call(
        &self,
        _ctx: &BaseCtx,
        mut output: ToolOutput<NoteOutput>,
    ) -> Result<ToolOutput<NoteOutput>, BoxError> {
        output.output.summary.limit = Some(1);
        Ok(output)
    }
}

#[test]
fn store_set_upsert_delete_by_stable_id() {
    let mut store = NoteStore::default();

    assert!(
        store
            .set(
                vec![
                    input("release", Some("remember release checklist")),
                    input("release", Some("remember launch checklist")),
                ],
                NOTE_CHAR_LIMIT,
            )
            .unwrap()
    );
    assert_eq!(
        store.items,
        vec![item("release", "remember launch checklist")]
    );

    assert!(
        store
            .upsert(
                vec![
                    input("release", Some("remember stable release tags")),
                    input("review", Some("prefer focused review notes")),
                ],
                NOTE_CHAR_LIMIT,
            )
            .unwrap()
    );
    assert_eq!(
        store.items,
        vec![
            item("release", "remember stable release tags"),
            item("review", "prefer focused review notes"),
        ]
    );

    assert!(store.delete(vec![input("release", None)]).unwrap());
    assert_eq!(
        store.items,
        vec![item("review", "prefer focused review notes")]
    );
    assert!(!store.delete(vec![input("missing", None)]).unwrap());
}

#[test]
fn store_reports_validation_and_limit_errors() {
    let mut store = NoteStore::default();

    assert_eq!(
        store
            .set(vec![input(" ", Some("content"))], NOTE_CHAR_LIMIT)
            .unwrap_err(),
        "items[0].id cannot be empty"
    );
    assert_eq!(
        store
            .upsert(vec![input("alpha", None)], NOTE_CHAR_LIMIT)
            .unwrap_err(),
        "items[0].content is required"
    );
    assert_eq!(
        store
            .upsert(vec![input("alpha", Some(" "))], NOTE_CHAR_LIMIT)
            .unwrap_err(),
        "items[0].content cannot be empty"
    );
    assert_eq!(
        store.delete(vec![input("", None)]).unwrap_err(),
        "items[0].id cannot be empty"
    );

    let oversized = "x".repeat(33);
    assert!(
        store
            .set(vec![input("a", Some(&oversized))], 32)
            .unwrap_err()
            .contains("32")
    );
}

#[tokio::test]
async fn tool_reads_empty_store_before_first_write() {
    let tool = NoteTool::new();
    let output = tool
        .call(note_ctx("writer"), NoteArgs::default(), Vec::new())
        .await
        .unwrap();

    assert!(output.output.success);
    assert!(output.output.items.is_empty());
    assert_eq!(output.output.summary.total, 0);
}

#[tokio::test]
async fn tool_persists_items_and_write_outputs_are_compact() {
    let tool = NoteTool::new();
    let ctx = note_ctx("writer");

    let first = tool
        .call(
            ctx.clone(),
            NoteArgs {
                op: Some(NOTE_OP_UPSERT.to_string()),
                items: Some(vec![input("release", Some("remember to tag releases"))]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert!(first.output.success);
    assert_eq!(first.output.summary.total, 1);
    assert!(first.output.items.is_empty());

    let second = tool
        .call(ctx.clone(), NoteArgs::default(), Vec::new())
        .await
        .unwrap();
    assert_eq!(
        second.output.items,
        vec![item("release", "remember to tag releases")]
    );

    let third = tool
        .call(
            ctx,
            NoteArgs {
                op: Some(NOTE_OP_UPSERT.to_string()),
                items: Some(vec![input(
                    "release",
                    Some("remember to tag stable releases"),
                )]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert!(third.output.success);
    assert!(third.output.items.is_empty());
    assert_eq!(third.output.summary.total, 1);
}

#[tokio::test]
async fn tool_storage_is_isolated_between_agents() {
    let tool = NoteTool::new();

    let writer = tool
        .call(
            note_ctx("writer"),
            NoteArgs {
                op: Some(NOTE_OP_UPSERT.to_string()),
                items: Some(vec![input("owner", Some("writer only note"))]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert!(writer.output.success);
    assert_eq!(writer.output.summary.total, 1);

    let reviewer = tool
        .call(note_ctx("reviewer"), NoteArgs::default(), Vec::new())
        .await
        .unwrap();
    assert!(reviewer.output.success);
    assert!(reviewer.output.items.is_empty());
}

#[tokio::test]
async fn tool_reports_validation_errors_and_unknown_ops_without_persisting() {
    let tool = NoteTool::default()
        .with_char_limit(32)
        .with_description("custom note description".to_string());
    assert_eq!(tool.description(), "custom note description");
    let ctx = note_ctx("validation");

    let missing_items = tool
        .call(
            ctx.clone(),
            NoteArgs {
                op: Some(NOTE_OP_UPSERT.to_string()),
                items: None,
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_items.output.error.as_deref(),
        Some("items are required for upsert")
    );
    // A domain failure keeps its typed output but is flagged for hooks,
    // providers, and telemetry.
    assert!(!missing_items.output.success);
    assert_eq!(missing_items.is_error, Some(true));
    assert_eq!(missing_items.output.summary.limit, Some(32));

    let missing_content = tool
        .call(
            ctx.clone(),
            NoteArgs {
                op: Some(NOTE_OP_SET.to_string()),
                items: Some(vec![input("entry", None)]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_content.output.error.as_deref(),
        Some("items[0].content is required")
    );
    assert_eq!(missing_content.is_error, Some(true));

    let unknown = tool
        .call(
            ctx.clone(),
            NoteArgs {
                op: Some("archive".to_string()),
                items: None,
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert!(
        unknown
            .output
            .error
            .as_deref()
            .is_some_and(|error| error.contains("Unknown op"))
    );
    assert_eq!(unknown.is_error, Some(true));

    let read = tool
        .call(ctx, NoteArgs::default(), Vec::new())
        .await
        .unwrap();
    assert!(read.output.items.is_empty());
    // A successful read is never flagged.
    assert!(read.output.success);
    assert_eq!(read.is_error, None);
}

#[tokio::test]
async fn tool_hooks_and_load_notes_use_agent_scoped_store() {
    let engine_ctx = EngineBuilder::new().mock_ctx();
    let agent = engine_ctx
        .child("hooked", "hooked")
        .expect("create child agent ctx");
    let ctx = agent.child_base(NoteTool::NAME).unwrap();
    ctx.set_state(NoteToolHook::new(Arc::new(MutatingHook)));

    let tool = NoteTool::new();
    let output = tool
        .call(
            ctx,
            NoteArgs {
                op: Some(NOTE_OP_READ.to_string()),
                items: None,
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
    assert!(output.output.success);
    assert_eq!(output.output.summary.limit, Some(1));

    let loaded = load_notes(&agent).await.unwrap();
    assert!(loaded.success);
    assert_eq!(loaded.items, vec![item("hook", "hook inserted note")]);
    assert_eq!(loaded.summary.limit, None);
}

#[tokio::test]
async fn load_notes_from_legacy_reads_old_store_without_touching_v2() {
    let agent = agent_ctx("legacy");
    let mut ctx = agent.child_base(NoteTool::NAME).unwrap();
    ctx.path = "t:note".into();
    let legacy = LegacyNoteStore {
        notes: vec![
            "remember old release process".to_string(),
            "prefer concise persisted notes".to_string(),
        ],
    };
    ctx.store_put(
        &NoteTool::legacy_store_path(&ctx.agent),
        PutMode::Overwrite,
        to_canonical_vec(&legacy).unwrap().into(),
    )
    .await
    .unwrap();

    let loaded = load_notes_from_legacy(&agent).await.unwrap();
    assert!(loaded.success);
    assert_eq!(
        loaded.items,
        vec![
            item("legacy_1", "remember old release process"),
            item("legacy_2", "prefer concise persisted notes"),
        ]
    );
    assert_eq!(loaded.summary.total, 2);
    assert_eq!(loaded.summary.limit, None);

    let current = load_notes(&agent).await.unwrap();
    assert!(current.items.is_empty());
}

#[tokio::test]
async fn bounded_pages_reconstruct_unicode_and_escaped_content() {
    let ctx = note_ctx("reader");
    let tool = NoteTool::new().with_response_bytes(2048);
    let contents = ["中文🙂\"\\\n".repeat(1100) + "end", "next note".to_string()];
    tool.call(
        ctx.clone(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![
                input("large", Some(&contents[0])),
                input("small", Some(&contents[1])),
            ]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    let mut args = NoteArgs::default();
    let mut recovered = std::collections::BTreeMap::<String, String>::new();
    let mut pages = 0;
    loop {
        let output = tool
            .call(ctx.clone(), args.clone(), vec![])
            .await
            .unwrap()
            .output;
        assert!(output.success, "{:?}", output.error);
        assert!(serde_json::to_vec(&output).unwrap().len() <= 2048);
        assert!(!output.items.is_empty());
        for (i, item) in output.items.iter().enumerate() {
            let value = recovered.entry(item.id.clone()).or_default();
            assert_eq!(
                value.chars().count(),
                if i == 0 { output.offset_chars } else { 0 }
            );
            value.push_str(&item.content);
        }
        pages += 1;
        assert!(pages < 100);
        assert_eq!(output.truncated, output.next_cursor.is_some());
        args.cursor = output.next_cursor;
        if args.cursor.is_none() {
            break;
        }
    }
    assert!(pages > 1);
    assert_eq!(recovered.get("large"), Some(&contents[0]));
    assert_eq!(recovered.get("small"), Some(&contents[1]));
}

#[tokio::test]
async fn listing_search_and_exact_reads_preserve_order_and_locations() {
    let ctx = note_ctx("searcher");
    let tool = NoteTool::new().with_response_bytes(2048);
    let content = "开头".repeat(100) + "needle" + &"末尾".repeat(100);
    tool.call(
        ctx.clone(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![
                input("first", Some("plain")),
                input("second", Some(&content)),
                input("third", Some("needle here")),
            ]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    let mut args = NoteArgs {
        op: Some("list".into()),
        limit: Some(1),
        ..Default::default()
    };
    let mut ids = Vec::new();
    loop {
        let output = tool
            .call(ctx.clone(), args.clone(), vec![])
            .await
            .unwrap()
            .output;
        assert!(output.success);
        assert!(output.items.is_empty());
        assert!(serde_json::to_vec(&output).unwrap().len() <= 2048);
        ids.extend(output.entries.iter().map(|entry| entry.id.clone()));
        args.cursor = output.next_cursor;
        if args.cursor.is_none() {
            break;
        }
    }
    assert_eq!(ids, ["first", "second", "third"]);
    let searched = tool
        .call(
            ctx.clone(),
            NoteArgs {
                op: Some("search".into()),
                query: Some("needle".into()),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap()
        .output;
    assert_eq!(searched.entries.len(), 2);
    assert_eq!(searched.entries[0].match_offset_chars, Some(200));
    assert!(searched.entries[0].excerpt.contains("needle"));
    assert_eq!(searched.entries[0].chars, content.chars().count());
    let read = tool
        .call(
            ctx,
            NoteArgs {
                ids: Some(vec!["third".into()]),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap()
        .output;
    assert_eq!(read.items, vec![item("third", "needle here")]);
}

#[tokio::test]
async fn cursors_reject_changes_of_query_scope_and_store() {
    let parent = agent_ctx("one");
    let ctx = parent.child_base("note").unwrap();
    let other = parent
        .child("two", "two")
        .unwrap()
        .child_base("note")
        .unwrap();
    let tool = NoteTool::new();
    let items = vec![input("a", Some("first")), input("b", Some("second"))];
    for ctx in [&ctx, &other] {
        tool.call(
            ctx.clone(),
            NoteArgs {
                op: Some("set".into()),
                items: Some(items.clone()),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap();
    }
    let args = NoteArgs {
        op: Some("list".into()),
        limit: Some(1),
        ..Default::default()
    };
    let cursor = tool
        .call(ctx.clone(), args.clone(), vec![])
        .await
        .unwrap()
        .output
        .next_cursor;
    assert!(cursor.is_some());
    let continued = NoteArgs { cursor, ..args };
    for (ctx, args) in [
        (other, continued.clone()),
        (
            ctx.clone(),
            NoteArgs {
                op: Some("read".into()),
                ..continued.clone()
            },
        ),
        (
            ctx.clone(),
            NoteArgs {
                ids: Some(vec!["a".into()]),
                ..continued.clone()
            },
        ),
        (
            ctx.clone(),
            NoteArgs {
                cursor: Some("garbage".into()),
                ..continued.clone()
            },
        ),
    ] {
        assert_eq!(
            tool.call(ctx, args, vec![]).await.unwrap().is_error,
            Some(true)
        );
    }
    tool.call(
        ctx.clone(),
        NoteArgs {
            op: Some("upsert".into()),
            items: Some(vec![input("a", Some("edited"))]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    assert_eq!(
        tool.call(ctx, continued, vec![]).await.unwrap().is_error,
        Some(true)
    );
}

#[tokio::test]
async fn invalid_queries_and_unknown_fields_do_not_mutate_notes() {
    let ctx = note_ctx("validation");
    let tool = NoteTool::new();
    tool.call(
        ctx.clone(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![input("a", Some("keep"))]),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    for args in [
        NoteArgs {
            op: Some("search".into()),
            query: Some(" ".into()),
            ..Default::default()
        },
        NoteArgs {
            limit: Some(0),
            ..Default::default()
        },
        NoteArgs {
            ids: Some(vec!["missing".into()]),
            ..Default::default()
        },
        NoteArgs {
            query: Some("keep".into()),
            ..Default::default()
        },
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![]),
            ids: Some(vec![]),
            ..Default::default()
        },
    ] {
        assert_eq!(
            tool.call(ctx.clone(), args, vec![]).await.unwrap().is_error,
            Some(true)
        );
    }
    assert!(
        tool.call_raw(ctx.clone(), serde_json::json!({"operation":"set"}), vec![])
            .await
            .is_err()
    );
    let result = tool
        .call(ctx.clone(), NoteArgs::default(), vec![])
        .await
        .unwrap();
    assert_eq!(result.output.items, vec![item("a", "keep")]);
    let result = tool
        .call(
            ctx,
            NoteArgs {
                op: Some("search".into()),
                query: Some("absent".into()),
                ..Default::default()
            },
            vec![],
        )
        .await
        .unwrap();
    assert!(result.output.success);
    assert!(result.output.entries.is_empty());
    assert!(!result.output.truncated);
}

#[tokio::test]
async fn summaries_are_opt_in_bounded_and_load_errors_are_preserved() {
    let agent = agent_ctx("summary");
    let ctx = agent.child_base("note").unwrap();
    let tool = NoteTool::new();
    tool.call(
        ctx.clone(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(
                (0..10)
                    .map(|i| input(&i.to_string(), Some(&"中🙂\n\"".repeat(100))))
                    .collect(),
            ),
            ..Default::default()
        },
        vec![],
    )
    .await
    .unwrap();
    let text = load_note_summary(
        &agent,
        &NoteContextConfig {
            max_bytes: 512,
            ids: None,
        },
    )
    .await
    .unwrap()
    .unwrap();
    assert!(text.len() <= 512);
    assert!(text.contains("more notes omitted"));
    let text = load_note_summary(
        &agent,
        &NoteContextConfig {
            ids: Some(vec!["1".into()]),
            ..Default::default()
        },
    )
    .await
    .unwrap()
    .unwrap();
    assert!(text.contains("\"id\":\"1\""));
    assert!(!text.contains("\"id\":\"0\""));
    ctx.store_put(
        &NoteTool::store_path(&ctx.agent),
        PutMode::Overwrite,
        b"invalid cbor".to_vec().into(),
    )
    .await
    .unwrap();
    assert!(try_load_notes(&agent).await.is_err());
    assert!(
        load_note_summary(&agent, &NoteContextConfig::default())
            .await
            .is_err()
    );
    assert!(load_notes(&agent).await.is_none());
}

#[tokio::test]
async fn cancellation_while_waiting_for_note_lock_never_writes() {
    let ctx = note_ctx("cancel");
    let tool = NoteTool::new();
    let lock = tool.update_lock(&ctx);
    let guard = lock.lock().await;
    let token = ctx.cancellation_token();
    let call = tool.call(
        ctx.clone(),
        NoteArgs {
            op: Some("set".into()),
            items: Some(vec![input("a", Some("never"))]),
            ..Default::default()
        },
        vec![],
    );
    let cancel = async {
        tokio::task::yield_now().await;
        token.cancel();
    };
    let (result, ()) = tokio::join!(call, cancel);
    assert!(result.is_err());
    drop(guard);
    assert!(NoteTool::load_store(&ctx).await.unwrap().items.is_empty());
}
