use super::*;
use crate::{context::BaseCtx, engine::EngineBuilder, subagent::SubAgentSet};
use serde_json::json;
use std::sync::Arc;

fn mock_ctx() -> BaseCtx {
    EngineBuilder::new().mock_ctx().base
}

/// Builds a `SKILL.md`; `frontmatter` holds extra raw YAML lines such as
/// `"execution: subagent"` or `"allowed-tools: shell fetch"`.
fn skill_md(name: &str, description: &str, body: &str, frontmatter: &[&str]) -> String {
    let mut content = format!("---\nname: {name}\ndescription: {description}\n");
    for line in frontmatter {
        content.push_str(line);
        content.push('\n');
    }
    content.push_str("---\n\n");
    content.push_str(body);
    if !body.ends_with('\n') {
        content.push('\n');
    }
    content
}

// -- Tool definition --

#[test]
fn skill_manager_tool_definition_schema() {
    let mgr = SkillManager::new(PathBuf::from("/tmp/skills"));
    let def = mgr.definition();
    assert_eq!(def.name, "skills_manager");
    assert!(def.description.contains("Agent Skills specification"));
    assert_eq!(def.parameters["additionalProperties"], json!(false));
    assert_eq!(def.parameters["required"], json!(["name"]));
    assert!(def.parameters["properties"].get("action").is_none());
}

// -- integration: load and read --

#[tokio::test]
async fn load_and_read_from_temp_dir() {
    let tmp = std::env::temp_dir().join(format!("anda-skills-test-{:016x}", rand::random::<u64>()));
    tokio::fs::create_dir_all(tmp.join("alpha")).await.unwrap();
    tokio::fs::create_dir_all(tmp.join("beta-skill"))
        .await
        .unwrap();

    tokio::fs::write(
        tmp.join("alpha/SKILL.md"),
        "\
---
name: alpha
description: Alpha skill for testing.
---

Alpha instructions.
",
    )
    .await
    .unwrap();

    tokio::fs::write(
        tmp.join("beta-skill/SKILL.md"),
        "\
---
name: beta-skill
description: Beta skill for testing.
license: MIT
execution: subagent
allowed-tools: shell fetch
---

Beta instructions.
",
    )
    .await
    .unwrap();

    let mgr = SkillManager::new(tmp.clone());
    mgr.load().await.unwrap();

    // Alpha declares no execution mode, so it stays inline: loaded and readable, never
    // callable.
    assert!(mgr.list().contains_key("skill_alpha"));
    assert!(!mgr.contains_lowercase("skill_alpha"));
    assert!(mgr.get_lowercase("skill_alpha").is_none());

    assert!(mgr.contains_lowercase("skill_beta_skill"));
    assert!(!mgr.contains_lowercase("skill_gamma"));

    // `allowed-tools` is an upper bound: beta gets exactly what it asked for, and none of the
    // manager defaults are unioned in.
    let beta = mgr.get_lowercase("skill_beta_skill").unwrap();
    assert_eq!(beta.tools, vec!["shell", "fetch"]);
    assert!(beta.instructions.contains("Beta instructions."));

    let beta_skill = mgr.get_skill("skill_beta_skill").unwrap();
    assert_eq!(beta_skill.frontmatter.license.as_deref(), Some("MIT"));

    let beta_content = mgr
        .call_raw(mock_ctx(), json!({ "name": "beta-skill" }), Vec::new())
        .await
        .unwrap();
    assert_eq!(beta_content.output["name"], json!("beta-skill"));
    assert_eq!(beta_content.output["execution"], json!("subagent"));
    assert_eq!(
        beta_content.output["callable"],
        json!("SA_skill_beta_skill")
    );
    assert_eq!(beta_content.output["path"], json!("beta-skill/SKILL.md"));
    assert_eq!(
        beta_content.output["base_dir"],
        json!(tmp.join("beta-skill").display().to_string())
    );
    assert!(
        beta_content.output["content"]
            .as_str()
            .unwrap()
            .contains("Beta instructions.")
    );

    // An inline skill reports no callable; the agent follows the returned content itself.
    let alpha_content = mgr
        .call_raw(mock_ctx(), json!({ "name": "alpha" }), Vec::new())
        .await
        .unwrap();
    assert_eq!(alpha_content.output["execution"], json!("inline"));
    assert!(alpha_content.output.get("callable").is_none());

    // Gamma delegates without declaring tools, so it inherits the manager defaults.
    tokio::fs::create_dir_all(tmp.join("gamma")).await.unwrap();
    tokio::fs::write(
        tmp.join("gamma/SKILL.md"),
        skill_md(
            "gamma",
            "Gamma skill for testing.",
            "Gamma instructions.",
            &["execution: subagent"],
        ),
    )
    .await
    .unwrap();

    mgr.load().await.unwrap();

    assert!(mgr.contains_lowercase("skill_gamma"));
    assert!(tmp.join("gamma/SKILL.md").exists());
    assert_eq!(
        mgr.get_lowercase("skill_gamma").unwrap().tools,
        DEFAULT_SKILL_TOOLS
    );

    // Verify on-disk content is valid SKILL.md.
    let on_disk = tokio::fs::read_to_string(tmp.join("gamma/SKILL.md"))
        .await
        .unwrap();
    let reparsed = parse_skill_md(tmp.to_path_buf(), &on_disk).unwrap();
    assert_eq!(reparsed.frontmatter.name, "gamma");

    // Definitions cover the two subagent skills only; alpha never reaches the model's tool
    // list.
    let defs = mgr.definitions(None);
    assert_eq!(defs.len(), 2);
    assert!(!defs.iter().any(|def| def.name == "skill_alpha"));

    let defs_filtered = mgr.definitions(Some(&["skill_gamma".to_string()]));
    assert_eq!(defs_filtered.len(), 1);
    assert_eq!(defs_filtered[0].name, "skill_gamma");

    assert!(
        mgr.definitions(Some(&["skill_alpha".to_string()]))
            .is_empty()
    );

    // Clean up.
    let _ = tokio::fs::remove_dir_all(&tmp).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reading_a_skill_refreshes_the_materialized_subagent() {
    let root = std::env::temp_dir().join(format!(
        "anda-skills-read-refresh-{:016x}",
        rand::random::<u64>()
    ));
    let skill_dir = root.join("alpha");
    tokio::fs::create_dir_all(&skill_dir).await.unwrap();
    tokio::fs::write(
        skill_dir.join("SKILL.md"),
        skill_md(
            "alpha",
            "Alpha skill before refresh.",
            "Original instructions.",
            &["execution: subagent"],
        ),
    )
    .await
    .unwrap();

    let mgr = SkillManager::new(root.clone());

    // A direct read loads a newly created skill without requiring a separate full reload.
    mgr.call_raw(mock_ctx(), json!({"name": "alpha"}), Vec::new())
        .await
        .unwrap();
    let before = mgr
        .get_lowercase("skill_alpha")
        .expect("the directly read skill must be callable");
    assert!(before.instructions.contains("Original instructions."));

    tokio::fs::write(
        skill_dir.join("SKILL.md"),
        skill_md(
            "alpha",
            "Alpha skill after refresh.",
            "Updated instructions.",
            &["execution: subagent"],
        ),
    )
    .await
    .unwrap();
    mgr.call_raw(mock_ctx(), json!({"name": "alpha"}), Vec::new())
        .await
        .unwrap();

    let after = mgr
        .get_lowercase("skill_alpha")
        .expect("the refreshed skill must remain callable");
    assert_eq!(after.description, "Alpha skill after refresh.");
    assert!(after.instructions.contains("Updated instructions."));
    assert!(
        Arc::ptr_eq(&before.subsessions, &after.subsessions),
        "refreshing instructions must not disconnect live sessions"
    );
    assert_eq!(
        mgr.definitions(Some(&["skill_alpha".to_string()]))[0].description,
        after.definition().description
    );

    // Switching the skill back to inline on disk retires the callable.
    tokio::fs::write(
        skill_dir.join("SKILL.md"),
        skill_md(
            "alpha",
            "Alpha skill, now inline.",
            "Inline instructions.",
            &[],
        ),
    )
    .await
    .unwrap();
    let output = mgr
        .call_raw(mock_ctx(), json!({"name": "alpha"}), Vec::new())
        .await
        .unwrap();
    assert_eq!(output.output["execution"], json!("inline"));
    assert!(mgr.get_lowercase("skill_alpha").is_none());
    assert!(mgr.list().contains_key("skill_alpha"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn load_and_read_platform_encoded_skill_file_when_available() {
    let Some(encoding) =
        anda_core::platform_text_encoding().filter(|encoding| encoding.name() != "UTF-8")
    else {
        return;
    };
    let Some(marker) = [
        "中文",
        "café",
        "日本語",
        "한국어",
        "тест",
        "γειά",
        "שלום",
        "مرحبا",
    ]
    .into_iter()
    .find(|candidate| {
        let (bytes, _, had_errors) = encoding.encode(candidate);
        !had_errors && std::str::from_utf8(&bytes).is_err()
    }) else {
        return;
    };

    let tmp =
        std::env::temp_dir().join(format!("anda-skills-legacy-{:016x}", rand::random::<u64>()));
    tokio::fs::create_dir_all(tmp.join("legacy-skill"))
        .await
        .unwrap();
    let body = format!("Legacy encoded skill marker: {marker}");
    let content = skill_md(
        "legacy-skill",
        "Legacy encoded skill for testing.",
        &body,
        &["execution: subagent"],
    );
    let (encoded, _, had_errors) = encoding.encode(&content);
    assert!(!had_errors);
    assert!(std::str::from_utf8(encoded.as_ref()).is_err());
    tokio::fs::write(tmp.join("legacy-skill/SKILL.md"), encoded.as_ref())
        .await
        .unwrap();

    let mgr = SkillManager::new(tmp.clone());
    mgr.load().await.unwrap();

    assert!(mgr.contains_lowercase("skill_legacy_skill"));
    let agent = mgr.get_lowercase("skill_legacy_skill").unwrap();
    assert!(agent.instructions.contains(&body));

    let output = mgr
        .call_raw(mock_ctx(), json!({ "name": "legacy-skill" }), Vec::new())
        .await
        .unwrap();
    assert_eq!(output.output["name"], json!("legacy-skill"));
    assert!(output.output["content"].as_str().unwrap().contains(&body));

    let _ = tokio::fs::remove_dir_all(&tmp).await;
}

#[tokio::test]
async fn load_and_read_from_multiple_dirs() {
    let root =
        std::env::temp_dir().join(format!("anda-skills-multi-{:016x}", rand::random::<u64>()));
    let default_dir = root.join("default");
    let extra_dir = root.join("extra");

    tokio::fs::create_dir_all(default_dir.join("alpha"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(extra_dir.join("beta"))
        .await
        .unwrap();

    tokio::fs::write(
        default_dir.join("alpha/SKILL.md"),
        skill_md(
            "alpha",
            "Alpha skill from default directory.",
            "Alpha instructions.",
            &[],
        ),
    )
    .await
    .unwrap();

    tokio::fs::write(
        extra_dir.join("beta/SKILL.md"),
        skill_md(
            "beta",
            "Beta skill from extra directory.",
            "Beta instructions.",
            &["execution: subagent"],
        ),
    )
    .await
    .unwrap();

    let mgr = SkillManager::new_with_dirs(
        default_dir.clone(),
        vec![extra_dir.clone(), default_dir.clone()],
    );
    let expected_dirs = vec![default_dir.clone(), extra_dir.clone()];
    assert_eq!(mgr.default_skills_dir(), default_dir.as_path());
    assert_eq!(mgr.skills_dirs(), expected_dirs.as_slice());

    mgr.load().await.unwrap();

    assert!(mgr.list().contains_key("skill_alpha"));
    assert!(mgr.contains_lowercase("skill_beta"));

    let beta_content = mgr
        .call_raw(mock_ctx(), json!({ "name": "beta" }), Vec::new())
        .await
        .unwrap();
    assert_eq!(beta_content.output["name"], json!("beta"));
    assert_eq!(beta_content.output["callable"], json!("SA_skill_beta"));
    assert_eq!(beta_content.output["path"], json!("beta/SKILL.md"));
    assert!(
        beta_content.output["content"]
            .as_str()
            .unwrap()
            .contains("Beta instructions.")
    );

    // Creation workflows should keep using the original default directory.
    assert!(mgr.default_skills_dir().ends_with("default"));
    let description = mgr.description();
    assert!(description.contains(&format!(
        "Skill directories: {}, {}.",
        default_dir.display(),
        extra_dir.display()
    )));
    assert!(description.contains(&format!(
        "Default skill creation directory: {}.",
        default_dir.display()
    )));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test(flavor = "current_thread")]
async fn manager_custom_options_lists_subagents_and_selects_resource_paths() {
    let root = std::env::temp_dir().join(format!(
        "anda-skills-manager-{:016x}",
        rand::random::<u64>()
    ));
    tokio::fs::create_dir_all(root.join("alpha")).await.unwrap();
    tokio::fs::write(
        root.join("alpha/SKILL.md"),
        skill_md(
            "alpha",
            "Alpha skill for manager coverage.",
            "Alpha body.",
            &[
                "execution: subagent",
                "allowed-tools: shell todo shell custom_tool",
            ],
        ),
    )
    .await
    .unwrap();
    tokio::fs::create_dir_all(root.join("inline-one"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("inline-one/SKILL.md"),
        skill_md(
            "inline-one",
            "Inline skill for manager coverage.",
            "Inline body.",
            &[],
        ),
    )
    .await
    .unwrap();
    tokio::fs::create_dir_all(root.join("locked-down"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("locked-down/SKILL.md"),
        skill_md(
            "locked-down",
            "Delegates but asks for no tools.",
            "Locked body.",
            &["execution: subagent", "allowed-tools: []"],
        ),
    )
    .await
    .unwrap();

    let mgr = Arc::new(
        SkillManager::new(root.clone())
            .with_description("custom skill reader".to_string())
            .with_default_skill_tools(vec!["read_file".to_string(), "todo".to_string()]),
    );
    assert_eq!(mgr.description(), "custom skill reader");
    assert_eq!(mgr.list().len(), 0);

    mgr.load().await.unwrap();
    assert_eq!(mgr.list().len(), 3);

    // Inline skills are in no tool list, so the description carries the resident catalog:
    // without it the model would have to guess that `inline-one` exists.
    let description = mgr.description();
    assert!(
        description.starts_with("custom skill reader"),
        "{description}"
    );
    assert!(
        description.contains("- inline-one [inline]: Inline skill for manager coverage."),
        "{description}"
    );
    assert!(
        description.contains("- alpha [subagent]: Alpha skill for manager coverage."),
        "{description}"
    );

    // Only the skills that opted in are materialized, and their declared tools replace the
    // configured defaults rather than merging with them.
    let subagents = mgr.subagents();
    assert_eq!(subagents.len(), 2);
    assert_eq!(subagents[0].name, "skill_alpha");
    assert_eq!(subagents[0].tools, vec!["shell", "todo", "custom_tool"]);
    // An empty `allowed-tools` is a declaration of none, not an omission, so the manager's
    // defaults do not fill it in.
    assert_eq!(subagents[1].name, "skill_locked_down");
    assert!(subagents[1].tools.is_empty());

    let any = mgr.clone().into_any();
    assert!(any.downcast_ref::<SkillManager>().is_some());

    let mut resources = vec![Resource {
        _id: 1,
        name: "text".to_string(),
        tags: vec!["text".to_string()],
        ..Default::default()
    }];
    assert!(SubAgentSet::select_resources(mgr.as_ref(), "missing", &mut resources).is_empty());
    // Inline skills are not callables, so they never claim resources.
    assert!(
        SubAgentSet::select_resources(mgr.as_ref(), "skill_inline_one", &mut resources).is_empty()
    );
    assert_eq!(resources.len(), 1);
    // A subagent skill that declares no `resource-tags` takes what the caller offers.
    let selected = SubAgentSet::select_resources(mgr.as_ref(), "skill_alpha", &mut resources);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].name, "text");
    assert!(resources.is_empty());
    assert!(SubAgentSet::select_resources(mgr.as_ref(), "skill_alpha", &mut resources).is_empty());

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test(flavor = "current_thread")]
async fn manager_finds_frontmatter_names_and_reports_duplicates_or_bad_files() {
    let root =
        std::env::temp_dir().join(format!("anda-skills-find-{:016x}", rand::random::<u64>()));
    let default_dir = root.join("default");
    let extra_dir = root.join("extra");
    tokio::fs::create_dir_all(default_dir.join("folder-name"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(extra_dir.join("duplicate-one"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(extra_dir.join("duplicate-two"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(extra_dir.join("bad"))
        .await
        .unwrap();

    tokio::fs::write(
        default_dir.join("folder-name/SKILL.md"),
        skill_md(
            "frontmatter-name",
            "Looked up by parsed frontmatter.",
            "Frontmatter body.",
            &["execution: subagent"],
        ),
    )
    .await
    .unwrap();
    tokio::fs::write(
        extra_dir.join("duplicate-one/SKILL.md"),
        skill_md("dupe", "Duplicate one.", "One.", &[]),
    )
    .await
    .unwrap();
    tokio::fs::write(
        extra_dir.join("duplicate-two/SKILL.md"),
        skill_md("dupe", "Duplicate two.", "Two.", &[]),
    )
    .await
    .unwrap();
    tokio::fs::write(extra_dir.join("bad/SKILL.md"), "not frontmatter")
        .await
        .unwrap();

    let mgr = SkillManager::new_with_dirs(default_dir.clone(), vec![extra_dir.clone()]);
    mgr.load().await.unwrap();
    assert!(mgr.contains_lowercase("skill_frontmatter_name"));

    let read = mgr
        .call_raw(mock_ctx(), json!({"name": "frontmatter-name"}), Vec::new())
        .await
        .unwrap();
    assert_eq!(read.output["callable"], json!("SA_skill_frontmatter_name"));
    assert_eq!(read.output["path"], json!("folder-name/SKILL.md"));

    let duplicate = mgr
        .call_raw(mock_ctx(), json!({"name": "dupe"}), Vec::new())
        .await
        .unwrap_err();
    assert!(duplicate.to_string().contains("multiple skills named"));

    let missing = mgr
        .call_raw(mock_ctx(), json!({"name": "missing"}), Vec::new())
        .await
        .unwrap_err();
    assert!(missing.to_string().contains("skill \"missing\" not found"));

    let invalid = mgr
        .call_raw(mock_ctx(), json!({"name": "Bad"}), Vec::new())
        .await
        .unwrap_err();
    assert!(invalid.to_string().contains("invalid character"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test(flavor = "current_thread")]
async fn manager_read_rejects_unsafe_large_non_utf8_or_mismatched_skill_files() {
    let root =
        std::env::temp_dir().join(format!("anda-skills-errors-{:016x}", rand::random::<u64>()));
    tokio::fs::create_dir_all(root.join("mismatch"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("mismatch/SKILL.md"),
        skill_md(
            "other-name",
            "Mismatched frontmatter name.",
            "Mismatch body.",
            &[],
        ),
    )
    .await
    .unwrap();

    let mgr = SkillManager::new(root.clone());
    let mismatch = mgr
        .call_raw(mock_ctx(), json!({"name": "mismatch"}), Vec::new())
        .await
        .unwrap_err();
    assert!(
        mismatch
            .to_string()
            .contains("must match requested skill name")
    );

    tokio::fs::create_dir_all(root.join("binary"))
        .await
        .unwrap();
    tokio::fs::write(root.join("binary/SKILL.md"), vec![0x81, 0x00])
        .await
        .unwrap();
    let binary = mgr
        .call_raw(mock_ctx(), json!({"name": "binary"}), Vec::new())
        .await
        .unwrap_err();
    assert!(
        binary
            .to_string()
            .contains("Only UTF-8 or supported text-encoded skill files")
    );

    tokio::fs::create_dir_all(root.join("large")).await.unwrap();
    tokio::fs::write(
        root.join("large/SKILL.md"),
        vec![b'a'; MAX_SKILL_FILE_BYTES as usize + 1],
    )
    .await
    .unwrap();
    let large = mgr
        .call_raw(mock_ctx(), json!({"name": "large"}), Vec::new())
        .await
        .unwrap_err();
    assert!(large.to_string().contains("exceeds maximum"));

    let missing_dirs = SkillManager::new(root.join("missing-default"));
    missing_dirs.load().await.unwrap();
    assert!(missing_dirs.list().is_empty());

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn load_uses_frontmatter_name_when_dir_differs() {
    let tmp = std::env::temp_dir().join(format!(
        "anda-skills-mismatch-{:016x}",
        rand::random::<u64>()
    ));
    tokio::fs::create_dir_all(tmp.join("wrong-dir"))
        .await
        .unwrap();

    tokio::fs::write(
        tmp.join("wrong-dir/SKILL.md"),
        "\
---
name: correct-name
description: Name does not match directory.
---

Body.
",
    )
    .await
    .unwrap();

    let mgr = SkillManager::new(tmp.clone());
    mgr.load().await.unwrap();

    assert!(mgr.list().contains_key("skill_correct_name"));

    let _ = tokio::fs::remove_dir_all(&tmp).await;
}

#[tokio::test(flavor = "current_thread")]
async fn tool_requires_name() {
    let tmp = std::env::temp_dir().join(format!(
        "anda-skills-requires-name-{:016x}",
        rand::random::<u64>()
    ));
    let mgr = SkillManager::new(tmp.clone());

    let err = mgr
        .call_raw(mock_ctx(), json!({}), Vec::new())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("missing field `name`"));
}

#[tokio::test(flavor = "current_thread")]
async fn tool_rejects_mutation_fields() {
    let tmp = std::env::temp_dir().join(format!(
        "anda-skills-rejects-action-{:016x}",
        rand::random::<u64>()
    ));
    let mgr = SkillManager::new(tmp.clone());

    let err = mgr
        .call_raw(
            mock_ctx(),
            json!({
                "action": "create",
                "name": "golf"
            }),
            Vec::new(),
        )
        .await
        .unwrap_err();

    assert!(err.to_string().contains("unknown field `action`"));
}

#[tokio::test(flavor = "current_thread")]
async fn sub_agents_manager_register_skills_manager() {
    let tmp = std::env::temp_dir().join(format!("anda-skills-val-{:016x}", rand::random::<u64>()));
    let tool = SkillManager::new(tmp.clone());
    let engine = EngineBuilder::new().empty().await.unwrap();
    assert!(engine.sub_agents_manager().insert(Arc::new(tool)).is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn reads_refresh_changed_files_and_retire_renamed_callables() {
    let root = std::env::temp_dir().join(format!(
        "anda-skills-parse-cache-{:016x}",
        rand::random::<u64>()
    ));
    write_subagent_skill(&root, "worker", "worker", "First instructions.").await;
    let mgr = SkillManager::new(root.clone());
    mgr.load().await.unwrap();

    let read = mgr
        .call_raw(mock_ctx(), json!({ "name": "worker" }), Vec::new())
        .await
        .unwrap();
    assert!(
        read.output["content"]
            .as_str()
            .unwrap()
            .contains("First instructions.")
    );
    assert_eq!(mgr.catalog().skills.len(), 1);

    // A name miss refreshes membership and removes the old callable for this path.
    write_subagent_skill(&root, "worker", "renamed-worker", "Second, longer text.").await;
    let read = mgr
        .call_raw(mock_ctx(), json!({ "name": "renamed-worker" }), Vec::new())
        .await
        .unwrap();
    assert!(
        read.output["content"]
            .as_str()
            .unwrap()
            .contains("Second, longer text.")
    );
    let err = mgr
        .call_raw(mock_ctx(), json!({ "name": "worker" }), Vec::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("renamed-worker"), "{err}");

    let _ = tokio::fs::remove_dir_all(&root).await;
}

/// Writes `<root>/<dir>/SKILL.md` for a subagent skill named `name`.
async fn write_subagent_skill(root: &Path, dir: &str, name: &str, body: &str) {
    let skill_dir = root.join(dir);
    tokio::fs::create_dir_all(&skill_dir).await.unwrap();
    tokio::fs::write(
        skill_dir.join("SKILL.md"),
        skill_md(
            name,
            &format!("{name} skill for filter testing."),
            body,
            &["execution: subagent"],
        ),
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn a_rejected_skill_is_invisible_everywhere() {
    let root =
        std::env::temp_dir().join(format!("anda-skills-filter-{:016x}", rand::random::<u64>()));
    write_subagent_skill(&root, "kept", "kept", "Kept instructions.").await;
    write_subagent_skill(&root, "hidden", "hidden", "Hidden instructions.").await;

    let mgr = SkillManager::new(root.clone());
    mgr.set_skill_filter(Some(Arc::new(|skill: &Skill| {
        skill.frontmatter.name != "hidden"
    })));
    mgr.load().await.unwrap();

    assert!(mgr.list().contains_key("skill_kept"));
    assert!(mgr.contains_lowercase("skill_kept"));

    // Not loaded, not callable, absent from the resident catalog, and — because
    // `find_skill_dir` would otherwise turn it up on disk — not readable by name either.
    assert!(!mgr.list().contains_key("skill_hidden"));
    assert!(!mgr.contains_lowercase("skill_hidden"));
    assert!(mgr.get_skill("skill_hidden").is_none());
    assert!(!mgr.description().contains("hidden"));
    let err = mgr
        .call_raw(mock_ctx(), json!({ "name": "hidden" }), Vec::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"), "{err}");
    // A rejected read must not sneak the skill back in through `upsert_skill`.
    assert!(!mgr.contains_lowercase("skill_hidden"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test(flavor = "current_thread")]
async fn rejecting_the_winning_copy_promotes_the_next_directory() {
    let root = std::env::temp_dir().join(format!(
        "anda-skills-filter-shadow-{:016x}",
        rand::random::<u64>()
    ));
    let personal = root.join("personal");
    let bundled = root.join("bundled");
    write_subagent_skill(&personal, "dup", "dup", "Personal instructions.").await;
    write_subagent_skill(&bundled, "dup", "dup", "Bundled instructions.").await;

    let mgr = SkillManager::new_with_dirs(personal.clone(), vec![bundled.clone()]);
    mgr.load().await.unwrap();
    assert!(
        mgr.get_lowercase("skill_dup")
            .unwrap()
            .instructions
            .contains("Personal instructions.")
    );

    // The filter runs before the duplicate check, so rejecting the higher-priority copy hands
    // the name to the next directory instead of dropping the skill entirely.
    let personal_dir = personal.join("dup");
    mgr.set_skill_filter(Some(Arc::new(move |skill: &Skill| {
        skill.base_dir != personal_dir
    })));
    mgr.load().await.unwrap();
    assert!(
        mgr.get_lowercase("skill_dup")
            .unwrap()
            .instructions
            .contains("Bundled instructions.")
    );

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reading_a_shadowed_name_resolves_by_directory_priority() {
    let root = std::env::temp_dir().join(format!(
        "anda-skills-read-shadow-{:016x}",
        rand::random::<u64>()
    ));
    let personal = root.join("personal");
    let bundled = root.join("bundled");
    write_subagent_skill(&personal, "dup", "dup", "Personal instructions.").await;
    write_subagent_skill(&bundled, "dup", "dup", "Bundled instructions.").await;

    let mgr = SkillManager::new_with_dirs(personal.clone(), vec![bundled.clone()]);
    mgr.load().await.unwrap();

    // The same name in two roots is what shadowing *is*; `load` already resolves it by
    // priority, so reading it by name must not report it as an unresolvable duplicate.
    let read = mgr
        .call_raw(mock_ctx(), json!({ "name": "dup" }), Vec::new())
        .await
        .unwrap();
    assert!(
        read.output["content"]
            .as_str()
            .unwrap()
            .contains("Personal instructions.")
    );

    // Two copies inside one root stay ambiguous: there is no priority to break the tie.
    write_subagent_skill(&personal, "dup-alias", "dup", "Second personal copy.").await;
    mgr.invalidate();
    let err = mgr
        .call_raw(mock_ctx(), json!({ "name": "dup" }), Vec::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("multiple skills named"), "{err}");

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test(flavor = "current_thread")]
async fn installing_a_filter_drops_what_it_rejects_without_a_reload() {
    let root = std::env::temp_dir().join(format!(
        "anda-skills-filter-prune-{:016x}",
        rand::random::<u64>()
    ));
    write_subagent_skill(&root, "alpha", "alpha", "Alpha instructions.").await;

    let mgr = SkillManager::new(root.clone());
    mgr.load().await.unwrap();
    assert!(mgr.contains_lowercase("skill_alpha"));

    mgr.set_skill_filter(Some(Arc::new(|_: &Skill| false)));
    assert!(mgr.list().is_empty());
    assert!(!mgr.contains_lowercase("skill_alpha"));

    // Clearing the filter does not resurrect anything on its own; the next load re-reads disk.
    mgr.set_skill_filter(None);
    assert!(mgr.list().is_empty());
    mgr.load().await.unwrap();
    assert!(mgr.contains_lowercase("skill_alpha"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}
