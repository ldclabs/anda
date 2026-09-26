//! Catalog, package access and lifecycle regressions.

use super::*;
use crate::engine::EngineBuilder;
use anda_core::{StateFeatures, validate_function_name};
use std::{collections::BTreeSet, fs};

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("anda-skill-catalog-{:016x}", rand::random::<u64>()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn put(&self, folder: &str, name: &str, body: &str) -> PathBuf {
        let path = self.0.join(folder).join("SKILL.md");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            format!(
                "---\nname: {name}\ndescription: Test {name}.\nexecution: subagent\n---\n{body}\n"
            ),
        )
        .unwrap();
        path
    }
    fn metadata(&self, folder: &str, content: &str) {
        let path = self.0.join(folder).join("agents/openai.yaml");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }
    fn manager(&self) -> Arc<SkillManager> {
        Arc::new(SkillManager::new(self.0.clone()))
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn ctx() -> BaseCtx {
    EngineBuilder::new().mock_ctx().base
}
async fn read(
    manager: &Arc<SkillManager>,
    skill: &str,
    resource: &str,
    cursor: Option<String>,
) -> Result<SkillsReadOutput, BoxError> {
    SkillsReadTool::new(manager.clone())
        .call(
            ctx(),
            SkillsReadArgs {
                skill: skill.into(),
                resource: Some(resource.into()),
                cursor,
            },
            vec![],
        )
        .await
        .map(|out| out.output)
}
async fn list(
    manager: &Arc<SkillManager>,
    query: Option<&str>,
    cursor: Option<String>,
) -> Result<SkillsListOutput, BoxError> {
    SkillsListTool::new(manager.clone())
        .call(
            ctx(),
            SkillsListArgs {
                query: query.map(str::to_owned),
                cursor,
            },
            vec![],
        )
        .await
        .map(|out| out.output)
}

#[test]
fn names_and_text_lengths_follow_the_documented_contract() {
    for count in 58..=64 {
        let name = "a".repeat(count);
        for mode in ["inline", "subagent"] {
            let text = format!(
                "---\nname: {name}\ndescription: {}\ncompatibility: {}\nexecution: {mode}\n---\nbody",
                "中".repeat(1024),
                "中".repeat(500)
            );
            let skill = parse_skill_md(PathBuf::from("/tmp/skill"), &text).unwrap();
            validate_function_name(&skill.agent_name).unwrap();
            if count > 58 {
                assert!(skill.agent_name.len() + crate::context::SUB_AGENT_PREFIX.len() <= 64);
            }
            assert_eq!(skill.frontmatter.name, name);
            assert_eq!(
                parse_skill_md(skill.base_dir.clone(), &format_skill_md(&skill).unwrap())
                    .unwrap()
                    .agent_name,
                skill.agent_name
            );
        }
    }
    assert_ne!(
        normalise_skill_agent_name(&"a".repeat(63)),
        normalise_skill_agent_name(&"a".repeat(64))
    );
    for description in [" ".repeat(10), "中".repeat(1025)] {
        assert!(
            parse_skill_md(
                PathBuf::new(),
                &format!("---\nname: test\ndescription: '{description}'\n---")
            )
            .is_err()
        );
    }
    assert!(parse_skill_md(PathBuf::new(), "---oops\nname: x\ndescription: y\n---").is_err());
    assert!(parse_skill_md(PathBuf::new(), "---\nname: x\ndescription: y\n---oops").is_err());
}

#[tokio::test]
async fn load_rejects_large_files_and_publishes_diagnostics() {
    let root = Root::new();
    root.put(
        "large",
        "large",
        &"x".repeat(discovery::MAX_SKILL_FILE_BYTES as usize),
    );
    let manager = root.manager();
    let report = manager.reload().await.unwrap();
    assert_eq!(report.loaded, 0);
    assert_eq!(report.rejected, 1);
    assert!(report.diagnostics[0].message.contains("exceeds maximum"));
    assert!(!manager.contains_lowercase("skill_large"));
    assert!(read(&manager, "large", "SKILL.md", None).await.is_err());
    assert!(load_skills_from_dir(&root.0).await.unwrap().is_empty());
}

#[cfg(unix)]
#[tokio::test]
async fn loading_and_resources_reject_links_nonregular_files_and_escapes() {
    use std::os::unix::fs::symlink;
    let root = Root::new();
    let outside = Root::new();
    let original = outside.put("source", "linked", "outside instructions");
    fs::create_dir_all(root.0.join("linked")).unwrap();
    fs::hard_link(&original, root.0.join("linked/SKILL.md")).unwrap();
    fs::create_dir_all(root.0.join("symlink")).unwrap();
    symlink(&original, root.0.join("symlink/SKILL.md")).unwrap();
    root.put("good", "good", "body");
    symlink(&outside.0, root.0.join("good/escape")).unwrap();
    fs::hard_link(&original, root.0.join("good/hard.md")).unwrap();
    let fifo = std::ffi::CString::new(root.0.join("good/pipe").to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    let manager = root.manager();
    manager.load().await.unwrap();
    assert_eq!(manager.catalog().report.loaded, 1);
    for resource in [
        "../source/SKILL.md",
        "/etc/passwd",
        "escape/source/SKILL.md",
        "hard.md",
        "pipe",
        "references/./x",
        "a\\b",
        "C:/x",
        "a//b",
        "SKILL.md/",
    ] {
        assert!(
            read(&manager, "good", resource, None).await.is_err(),
            "{resource}"
        );
    }
    assert!(read(&manager, "linked", "SKILL.md", None).await.is_err());
    // Replacing a loaded package directory with a symlink cannot authorize the target.
    fs::rename(root.0.join("good"), outside.0.join("saved")).unwrap();
    symlink(outside.0.join("source"), root.0.join("good")).unwrap();
    assert!(read(&manager, "good", "SKILL.md", None).await.is_err());
}

#[tokio::test]
async fn names_conflicts_filters_and_session_identity_stay_consistent() {
    let root = Root::new();
    let other = Root::new();
    let first = root.put("one", "same", "first");
    other.put("one", "same", "fallback");
    let manager = Arc::new(SkillManager::new_with_dirs(
        root.0.clone(),
        vec![other.0.clone()],
    ));
    manager.load().await.unwrap();
    let old = manager.get_lowercase("skill_same").unwrap();
    let id = manager
        .catalog()
        .skills
        .iter()
        .find(|s| s.active)
        .unwrap()
        .id
        .clone();
    root.put("two", "same", "ambiguous");
    manager.invalidate();
    assert!(
        read(&manager, "same", "SKILL.md", None)
            .await
            .unwrap_err()
            .to_string()
            .contains("multiple skills")
    );
    assert!(!manager.contains_lowercase("skill_same"));
    assert!(manager.list().is_empty());
    assert!(
        read(&manager, &id, "SKILL.md", None)
            .await
            .unwrap()
            .callable
            .is_none()
    );
    let two = root.0.join("two");
    manager.set_skill_filter(Some(Arc::new(move |skill| skill.base_dir != two)));
    manager.load().await.unwrap();
    assert!(manager.contains_lowercase("skill_same"));
    fs::remove_file(first).unwrap();
    let output = read(&manager, "same", "SKILL.md", None).await.unwrap();
    assert!(output.content.contains("fallback"));
    let current = manager.get_lowercase("skill_same").unwrap();
    assert!(!Arc::ptr_eq(&old.subsessions, &current.subsessions));
}

#[tokio::test]
async fn snapshots_survive_edits_but_do_not_keep_deleted_or_renamed_callables_alive() {
    let root = Root::new();
    root.put("folder", "old", "body");
    let manager = root.manager();
    manager.load().await.unwrap();
    let snapshot = manager.catalog();
    let generation = snapshot.report.generation;
    let original_id = snapshot.skills[0].id.clone();
    assert_eq!(manager.reload().await.unwrap().generation, generation);
    let old = manager.get_lowercase("skill_old").unwrap();
    root.put("folder", "old", "changed");
    read(&manager, "old", "SKILL.md", None).await.unwrap();
    assert!(Arc::ptr_eq(
        &old.subsessions,
        &manager.get_lowercase("skill_old").unwrap().subsessions
    ));
    root.put("folder", "new", "renamed");
    read(&manager, "new", "SKILL.md", None).await.unwrap();
    assert!(!manager.contains_lowercase("skill_old"));
    assert!(manager.contains_lowercase("skill_new"));
    assert_eq!(manager.catalog().skills[0].id, original_id);
    assert_eq!(snapshot.skills[0].name, "old");
    fs::remove_dir_all(&root.0).unwrap();
    manager.load().await.unwrap();
    assert!(manager.catalog().skills.is_empty());
    assert!(manager.subagents().is_empty());
    assert!(
        read(&manager, &original_id, "SKILL.md", None)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn root_reordering_and_aliases_preserve_stable_ids() {
    let root = Root::new();
    let other = Root::new();
    root.put("same", "same", "first");
    other.put("same", "same", "second");
    let first = SkillManager::new_with_dirs(root.0.clone(), vec![other.0.clone()]);
    let second = SkillManager::new_with_dirs(other.0.clone(), vec![root.0.clone()]);
    first.load().await.unwrap();
    second.load().await.unwrap();
    let ids = |manager: &SkillManager| {
        manager
            .catalog()
            .skills
            .iter()
            .map(|s| s.id.clone())
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(ids(&first), ids(&second));
    let nested = SkillManager::new_with_dirs(root.0.clone(), vec![root.0.join("same")]);
    nested.load().await.unwrap();
    assert_eq!(nested.catalog().skills.len(), 1);
}

#[tokio::test]
async fn explicit_policy_and_preflight_never_grant_or_install_capabilities() {
    let root = Root::new();
    root.put("deploy", "deploy", "deploy instructions");
    root.metadata("deploy", "interface:\n  short_description: Deploy on request.\npolicy:\n  allow_implicit_invocation: false\ndependencies:\n  tools:\n    - type: tool\n      value: shell\n    - type: mcp\n      value: releases\n");
    let manager = root.manager();
    manager.load().await.unwrap();
    assert!(!manager.description().contains("deploy"));
    assert!(manager.definitions(None).is_empty());
    assert!(list(&manager, None, None).await.unwrap().skills.is_empty());
    assert_eq!(
        list(&manager, Some("deploy"), None)
            .await
            .unwrap()
            .skills
            .len(),
        1
    );
    assert_eq!(manager.definitions(Some(&["skill_deploy".into()])).len(), 1);
    assert!(
        read(&manager, "deploy", "SKILL.md", None)
            .await
            .unwrap()
            .callable
            .is_some()
    );
    let snapshot = manager.catalog();
    let skill = &snapshot.skills[0];
    let preflight = skill.preflight(&BTreeSet::from(["shell".into()]), &BTreeSet::new());
    assert!(!preflight.ready);
    assert_eq!(preflight.missing[0].value, "releases");
    assert!(
        skill
            .preflight(
                &BTreeSet::from(["shell".into()]),
                &BTreeSet::from(["releases".into()])
            )
            .ready
    );
    let id = skill.id.clone();
    manager.set_skill_filter(Some(Arc::new(|_| false)));
    assert!(read(&manager, &id, "SKILL.md", None).await.is_err());
    assert!(
        list(&manager, Some("deploy"), None)
            .await
            .unwrap()
            .skills
            .is_empty()
    );
}

#[tokio::test]
async fn malformed_sidecar_does_not_enable_an_explicit_only_skill() {
    let root = Root::new();
    root.put("worker", "worker", "body");
    root.metadata("worker", "policy:\n  allow_implicit_invocation: false\n");
    let manager = root.manager();
    manager.load().await.unwrap();
    root.metadata("worker", "policy: [broken");
    assert!(read(&manager, "worker", "SKILL.md", None).await.is_err());
    assert!(manager.subagents().is_empty());
    assert_eq!(manager.catalog().report.rejected, 1);
}

#[tokio::test]
async fn read_pages_bound_json_bytes_and_reject_stale_or_cross_resource_cursors() {
    let root = Root::new();
    let body = "中\"\\\n".repeat(2000);
    let path = root.put("worker", "worker", &body);
    fs::write(root.0.join("worker/reference.md"), &body).unwrap();
    let manager = Arc::new(SkillManager::new(root.0.clone()).with_limits(SkillLimits {
        response_bytes: 2048,
        ..Default::default()
    }));
    let first = read(&manager, "worker", "SKILL.md", None).await.unwrap();
    let cursor = first.next_cursor.clone().unwrap();
    assert!(
        manager
            .call(
                ctx(),
                SkillArgs {
                    name: "worker".into()
                },
                vec![]
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("skills_read")
    );
    let mut joined = String::new();
    let mut continuation = None;
    loop {
        let page = read(&manager, "worker", "SKILL.md", continuation)
            .await
            .unwrap();
        assert!(serde_json::to_vec(&page).unwrap().len() <= 2048);
        joined.push_str(&page.content);
        continuation = page.next_cursor;
        if continuation.is_none() {
            break;
        }
    }
    assert_eq!(joined, fs::read_to_string(&path).unwrap());
    assert!(
        read(&manager, "worker", "reference.md", Some(cursor.clone()))
            .await
            .is_err()
    );
    // Change bytes even when the original body has no literal word "body".
    root.put("worker", "worker", &(body + "changed"));
    assert!(
        read(&manager, "worker", "SKILL.md", Some(cursor))
            .await
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert!(
        read(&manager, "worker", "SKILL.md", Some("bad".into()))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn listing_is_bounded_and_membership_changes_invalidate_cursors() {
    let root = Root::new();
    for index in 0..43 {
        root.put(&format!("s{index}"), &format!("s{index}"), "body");
    }
    let manager = Arc::new(SkillManager::new(root.0.clone()).with_limits(SkillLimits {
        catalog_bytes: 256,
        response_bytes: 2048,
        ..Default::default()
    }));
    manager.load().await.unwrap();
    assert!(manager.skills_catalog().len() <= 256);
    assert!(manager.description().contains("omitted"));
    let first = list(&manager, None, None).await.unwrap();
    let stale = first.next_cursor.clone().unwrap();
    let mut ids = BTreeSet::new();
    let mut continuation = None;
    loop {
        let page = list(&manager, None, continuation).await.unwrap();
        assert!(page.skills.len() <= 20);
        assert!(serde_json::to_vec(&page).unwrap().len() <= 2048);
        for skill in page.skills {
            assert!(ids.insert(skill.id));
        }
        continuation = page.next_cursor;
        if continuation.is_none() {
            break;
        }
    }
    assert_eq!(ids.len(), 43);
    root.put("added", "added", "new");
    manager.invalidate();
    assert!(
        list(&manager, None, Some(stale))
            .await
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert_eq!(
        list(&manager, Some("added"), None)
            .await
            .unwrap()
            .skills
            .len(),
        1
    );
}

#[tokio::test]
async fn traversal_and_total_content_limits_are_reported() {
    let root = Root::new();
    root.put("one", "one", "body");
    root.put("deep/nested/skill", "deep", "body");
    for limits in [
        SkillLimits {
            max_depth: 1,
            ..Default::default()
        },
        SkillLimits {
            max_entries: 1,
            ..Default::default()
        },
        SkillLimits {
            max_skills: 1,
            ..Default::default()
        },
        SkillLimits {
            max_total_bytes: 1,
            ..Default::default()
        },
    ] {
        let manager = SkillManager::new(root.0.clone()).with_limits(limits);
        let report = manager.reload().await.unwrap();
        assert!(report.truncated);
        assert!(!report.diagnostics.is_empty());
    }
}

#[tokio::test]
async fn cancelled_reads_do_not_load_or_publish_a_catalog() {
    let root = Root::new();
    root.put("worker", "worker", "body");
    let manager = root.manager();
    let ctx = ctx();
    ctx.cancellation_token().cancel();
    let error = SkillsReadTool::new(manager.clone())
        .call(
            ctx,
            SkillsReadArgs {
                skill: "worker".into(),
                resource: None,
                cursor: None,
            },
            vec![],
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    assert_eq!(manager.catalog().report.generation, 0);
}

#[tokio::test]
async fn skill_bundle_integrates_with_discovery_merging_and_compaction() {
    use crate::context::{DiscoveredTools, ToolsSearch, ToolsSelect};
    use anda_core::{CompletionRequest, Json};
    let root = Root::new();
    root.put("worker", "worker", "body");
    let manager = root.manager();
    manager.load().await.unwrap();
    let ctx = EngineBuilder::new()
        .register_tools(manager.tools().unwrap())
        .unwrap()
        .mock_ctx();
    let search = ToolsSearch::new()
        .run(
            ctx.clone(),
            r#"{"query":"skills_read","limit":5}"#.into(),
            vec![],
        )
        .await
        .unwrap();
    let searched: Json = serde_json::from_str(&search.content).unwrap();
    assert!(
        searched["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "skills_read")
    );
    let selected = ToolsSelect::new()
        .run(
            ctx.clone(),
            r#"{"tools":[],"query":"","group":"skills","limit":5}"#.into(),
            vec![],
        )
        .await
        .unwrap();
    let output: Json = serde_json::from_str(&selected.content).unwrap();
    assert_eq!(output["tools"].as_array().unwrap().len(), 3);
    assert_eq!(output["groups"][0]["id"], "skills");
    for tool in output["tools"].as_array().unwrap() {
        assert_eq!(tool["parameters"]["additionalProperties"], false);
    }
    let mut discovered = DiscoveredTools::default();
    discovered.observe_output("tools_select", &output);
    discovered.observe_output("tools_select", &output);
    assert_eq!(discovered.merge_policy(), Some(true));
    let mut request = CompletionRequest::default();
    discovered.merge_into_request(&mut request);
    assert_eq!(request.tools.len(), 3);
    let mut compacted = output.clone();
    discovered.compact_output_for_request("tools_select", &mut compacted, &request.tools);
    assert!(
        compacted["tools"]
            .as_array()
            .unwrap()
            .iter()
            .all(|tool| tool.get("parameters").is_none())
    );
    let missing = ToolsSelect::new()
        .run(
            ctx,
            r#"{"tools":["missing_skill_tool"],"query":"","group":"","limit":5}"#.into(),
            vec![],
        )
        .await
        .unwrap();
    let missing: Json = serde_json::from_str(&missing.content).unwrap();
    assert!(missing["tools"].as_array().unwrap().is_empty());
    assert!(SkillsReadTool::new(manager).call_raw(super::catalog_tests::ctx(), serde_json::json!({"skill":"worker","resource":"SKILL.md","cursor":null,"command":"shell"}), vec![]).await.is_err());
}

#[tokio::test]
async fn resource_hook_observes_resolved_identity_without_other_conversation_data() {
    use crate::hook::ToolHook;
    use parking_lot::Mutex;
    struct Recorder(Arc<Mutex<Vec<(String, String)>>>);
    #[async_trait::async_trait]
    impl ToolHook<SkillsReadArgs, SkillsReadOutput> for Recorder {
        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            output: ToolOutput<SkillsReadOutput>,
        ) -> Result<ToolOutput<SkillsReadOutput>, BoxError> {
            self.0.lock().push((
                output.output.skill_id.clone(),
                output.output.resource.clone(),
            ));
            Ok(output)
        }
    }
    let root = Root::new();
    root.put("worker", "worker", "body");
    let manager = root.manager();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = ctx();
    ctx.set_state(SkillsReadHook::new(Arc::new(Recorder(seen.clone()))));
    let out = SkillsReadTool::new(manager)
        .call(
            ctx,
            SkillsReadArgs {
                skill: "worker".into(),
                resource: None,
                cursor: None,
            },
            vec![],
        )
        .await
        .unwrap();
    assert_eq!(*seen.lock(), vec![(out.output.skill_id, "SKILL.md".into())]);
}

#[tokio::test]
async fn coalesced_invalidation_preserves_generation_for_identical_contents() {
    let root = Root::new();
    root.put("worker", "worker", "body");
    let manager = root.manager();
    manager.load().await.unwrap();
    let generation = manager.catalog().report.generation;
    manager.invalidate();
    let (a, b) = tokio::join!(list(&manager, None, None), list(&manager, None, None));
    assert_eq!(a.unwrap().generation, generation);
    assert_eq!(b.unwrap().generation, generation);
    assert_eq!(
        manager.registry.read().epoch,
        manager.epoch.load(Ordering::Acquire)
    );
}

#[tokio::test]
async fn cancellation_while_waiting_for_refresh_publishes_nothing() {
    let root = Root::new();
    root.put("worker", "worker", "body");
    let manager = root.manager();
    let guard = manager.refresh.lock().await;
    let ctx = ctx();
    let cancellation = ctx.cancellation_token();
    let tool = SkillsListTool::new(manager.clone());
    let call = tool.call(ctx, SkillsListArgs::default(), vec![]);
    let cancel = async {
        tokio::task::yield_now().await;
        cancellation.cancel();
    };
    let (result, ()) = tokio::join!(call, cancel);
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert_eq!(manager.catalog().report.generation, 0);
    drop(guard);
    assert_eq!(manager.reload().await.unwrap().loaded, 1);
}

#[tokio::test]
async fn diagnostics_and_escape_heavy_listing_metadata_stay_bounded() {
    let root = Root::new();
    for index in 0..130 {
        let path = root.put(&format!("bad{index}"), &format!("bad{index}"), "body");
        fs::write(path, "invalid").unwrap();
    }
    let path = root.put("valid", "valid", "body");
    let description = "\"\\".repeat(250);
    let yaml = serde_json::to_string(&description).unwrap();
    fs::write(
        path,
        format!("---\nname: valid\ndescription: {yaml}\n---\nbody"),
    )
    .unwrap();
    let manager = Arc::new(SkillManager::new(root.0.clone()).with_limits(SkillLimits {
        response_bytes: 1024,
        ..Default::default()
    }));
    let report = manager.reload().await.unwrap();
    assert_eq!(report.diagnostics.len(), 128);
    assert!(report.truncated);
    assert_eq!(report.rejected, 130);
    let page = list(&manager, Some("valid"), None).await.unwrap();
    assert_eq!(page.skills.len(), 1);
    assert!(serde_json::to_vec(&page).unwrap().len() <= 1024);
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn non_utf8_directory_names_keep_ids_usable_and_diagnostics_serializable() {
    use std::os::unix::ffi::OsStrExt;
    let root = Root::new();
    let good = root.0.join(std::ffi::OsStr::from_bytes(b"good-\xff"));
    let bad = root.0.join(std::ffi::OsStr::from_bytes(b"bad-\xff"));
    fs::create_dir_all(&good).unwrap();
    fs::create_dir_all(&bad).unwrap();
    fs::write(
        good.join("SKILL.md"),
        "---\nname: valid\ndescription: Valid.\n---\nbody",
    )
    .unwrap();
    fs::write(bad.join("SKILL.md"), "invalid").unwrap();
    let manager = root.manager();
    let report = manager.reload().await.unwrap();
    assert_eq!((report.loaded, report.rejected), (1, 1));
    let snapshot = manager.catalog();
    serde_json::to_vec(&snapshot).unwrap();
    assert!(
        read(&manager, &snapshot.skills[0].id, "SKILL.md", None)
            .await
            .unwrap()
            .content
            .contains("body")
    );
}

#[cfg(unix)]
#[test]
fn native_diagnostic_paths_serialize_even_when_they_are_not_utf8() {
    use std::os::unix::ffi::OsStrExt;
    let mut report = SkillLoadReport::default();
    report.note(
        "invalid",
        PathBuf::from(std::ffi::OsStr::from_bytes(b"bad-\xff")),
        "Invalid skill",
    );
    let value = serde_json::to_value(report).unwrap();
    assert!(
        value["diagnostics"][0]["path"]
            .as_str()
            .unwrap()
            .starts_with("bad-")
    );
}
