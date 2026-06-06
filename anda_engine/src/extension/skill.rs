//! Skills manager extension.
//!
//! This module provides:
//! - Loading skills from directory trees of `SKILL.md` files into [`SubAgent`] instances.
//! - Reading loaded skill files via the [`SkillManager`] tool.
//!
//! Each `SKILL.md` follows the [Agent Skills specification](https://agentskills.io):
//! YAML frontmatter (`---` delimiters) with `name`, `description`, and optional
//! `license`, `compatibility`, `metadata`, `allowed-tools` fields. The Markdown body
//! becomes the agent's `instructions`.
//!
//! Skill names use kebab-case on disk (e.g. `my-skill`); they are normalised to
//! snake_case (`my_skill`) when loaded as [`SubAgent`] instances.

use anda_core::{
    Agent, BoxError, FunctionDefinition, Resource, Tool, ToolOutput, select_resources,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    any::Any,
    collections::BTreeMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    context::BaseCtx,
    extension::fs::{ensure_file_size_within_limit, ensure_regular_file},
    subagent::{SubAgent, SubAgentSet},
};

mod types;
pub use types::*;

// ---------------------------------------------------------------------------
// SkillManager
// ---------------------------------------------------------------------------

const MAX_SKILL_FILE_BYTES: u64 = 512 * 1024;

/// Arguments for reading a skill via the [`SkillManager`] tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SkillArgs {
    /// Skill name in kebab-case (e.g. `pdf-processing`). 1-64 chars,
    /// lowercase alphanumeric and hyphens only.
    pub name: String,
}

/// Content returned by the [`SkillManager`] tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SkillContentOutput {
    /// Skill name from the SKILL.md frontmatter.
    pub name: String,
    /// Normalised subagent name exposed by this skill.
    pub agent_name: String,
    /// Skill description from the SKILL.md frontmatter.
    pub description: String,
    /// Path to SKILL.md, relative to a configured skills directory when possible.
    pub path: String,
    /// Full SKILL.md content including YAML frontmatter and Markdown body.
    pub content: String,
}

/// Manages skills loaded from `SKILL.md` files on disk.
///
/// [`SkillManager`] implements [`Tool<BaseCtx>`] so that LLMs can inspect skill
/// files at runtime. Skills loaded here are exposed as [`SubAgent`] instances
/// that the engine can invoke.
pub struct SkillManager {
    /// Directory used by skill creation workflows. Loading also includes this directory.
    default_skills_dir: PathBuf,
    /// Directories scanned for skills, with `default_skills_dir` first.
    skills_dirs: Vec<PathBuf>,
    skills: RwLock<BTreeMap<String, Skill>>,
    description: String,
    default_skill_tools: Vec<String>,
}

static DEFAULT_SKILL_TOOLS: &[&str] = &[
    "shell",
    "read_file",
    "search_file",
    "write_file",
    "edit_file",
    "todo",
    "tools_select",
];

fn build_skills_dirs(
    default_skills_dir: PathBuf,
    additional_skills_dirs: Vec<PathBuf>,
) -> Vec<PathBuf> {
    let mut skills_dirs = vec![default_skills_dir];
    for dir in additional_skills_dirs {
        if !skills_dirs.iter().any(|existing| existing == &dir) {
            skills_dirs.push(dir);
        }
    }
    skills_dirs
}

fn format_path_list(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_description(default_skills_dir: &Path, skills_dirs: &[PathBuf]) -> String {
    format!(
        "Load reusable skills following the Agent Skills specification and read \
        a skill's SKILL.md content by name. Agent Skills are folders of instructions, \
        scripts, and resources that agents can follow directly or invoke as subagents. \
        Skill directories: {}. Default skill creation directory: {}",
        format_path_list(skills_dirs),
        default_skills_dir.display()
    )
}

impl SkillManager {
    /// Tool name used for registration.
    pub const NAME: &'static str = "skills_manager";

    /// Create a new, empty manager rooted at `skills_dir`.
    pub fn new(skills_dir: PathBuf) -> Self {
        Self::new_with_dirs(skills_dir, Vec::new())
    }

    /// Create a new, empty manager that loads from the default directory and
    /// additional skill directories.
    ///
    /// New skills should still be created under `default_skills_dir`; additional
    /// directories are read-only load roots from the manager's perspective.
    pub fn new_with_dirs(
        default_skills_dir: PathBuf,
        additional_skills_dirs: Vec<PathBuf>,
    ) -> Self {
        let skills_dirs = build_skills_dirs(default_skills_dir.clone(), additional_skills_dirs);
        Self {
            skills: RwLock::new(BTreeMap::new()),
            description: build_description(&default_skills_dir, &skills_dirs),
            default_skills_dir,
            skills_dirs,
            default_skill_tools: DEFAULT_SKILL_TOOLS.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Directory where new skills should be created.
    pub fn default_skills_dir(&self) -> &Path {
        &self.default_skills_dir
    }

    /// Directories scanned when loading or reading skills.
    pub fn skills_dirs(&self) -> &[PathBuf] {
        &self.skills_dirs
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    pub fn with_default_skill_tools(mut self, tools: Vec<String>) -> Self {
        self.default_skill_tools = tools;
        self
    }

    fn with_default_tools(&self, agent: SubAgent) -> SubAgent {
        let mut tools = self.default_skill_tools.clone();
        for tool in agent.tools {
            if !tools.contains(&tool) {
                tools.push(tool);
            }
        }

        SubAgent { tools, ..agent }
    }

    async fn read_text_file(&self, path: &Path, max_size: u64) -> Result<String, BoxError> {
        let meta = tokio::fs::symlink_metadata(path).await.map_err(|err| {
            format!(
                "Failed to inspect file metadata (path: {}): {err}",
                path.display()
            )
        })?;
        ensure_regular_file(&meta, path, "Reading multiply-linked files is not allowed")?;
        ensure_file_size_within_limit(&meta, path, max_size)?;

        let data = tokio::fs::read(path)
            .await
            .map_err(|err| format!("Failed to read file (path: {}): {err}", path.display()))?;
        types::decode_skill_md_bytes(data).map_err(|_| {
            format!(
                "Only UTF-8 or supported text-encoded skill files are supported by skills_manager (path: {})",
                path.display()
            )
            .into()
        })
    }

    async fn find_skill_dir(&self, name: &str) -> Result<Option<PathBuf>, BoxError> {
        validate_skill_name(name)?;

        let mut matches = Vec::new();

        {
            let skills = self.skills.read();
            for skill in skills.values() {
                let dir_name_matches = skill.base_dir.file_name() == Some(OsStr::new(name));
                if (skill.frontmatter.name == name || dir_name_matches)
                    && !matches.iter().any(|path| path == &skill.base_dir)
                {
                    matches.push(skill.base_dir.clone());
                }
            }
        }

        for skills_dir in &self.skills_dirs {
            if skills_dir.is_dir() {
                for path in find_skill_files(skills_dir).await? {
                    let Some(base_dir) = path.parent() else {
                        continue;
                    };
                    let base_dir = base_dir.to_path_buf();
                    let dir_name_matches = base_dir.file_name() == Some(OsStr::new(name));
                    let frontmatter_name_matches = if dir_name_matches {
                        true
                    } else if let Ok(content) =
                        self.read_text_file(&path, MAX_SKILL_FILE_BYTES).await
                    {
                        parse_skill_md(base_dir.clone(), &content)
                            .map(|skill| skill.frontmatter.name == name)
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    if frontmatter_name_matches
                        && !matches.iter().any(|candidate| candidate == &base_dir)
                    {
                        matches.push(base_dir);
                    }
                }
            }
        }

        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.pop()),
            _ => Err(format!(
                "multiple skills named {:?} exist under configured skills directories: {}",
                name,
                format_path_list(&self.skills_dirs)
            )
            .into()),
        }
    }

    fn display_path(&self, path: &Path) -> String {
        for skills_dir in &self.skills_dirs {
            if let Ok(stripped) = path.strip_prefix(skills_dir) {
                return stripped.display().to_string();
            }
        }

        if let Ok(canonical_path) = std::fs::canonicalize(path) {
            for skills_dir in &self.skills_dirs {
                if let Ok(root) = std::fs::canonicalize(skills_dir)
                    && let Ok(stripped) = canonical_path.strip_prefix(&root)
                {
                    return stripped.display().to_string();
                }
            }
        }

        path.display().to_string()
    }

    async fn read_skill_action(&self, args: SkillArgs) -> Result<SkillContentOutput, BoxError> {
        validate_skill_name(&args.name)?;
        let skill_dir = self
            .find_skill_dir(&args.name)
            .await?
            .ok_or_else(|| format!("skill {:?} not found", args.name))?;
        let target = skill_dir.join("SKILL.md");

        let content = self.read_text_file(&target, MAX_SKILL_FILE_BYTES).await?;
        let skill = parse_skill_md(skill_dir, &content)?;
        if skill.frontmatter.name != args.name {
            return Err(format!(
                "SKILL.md frontmatter name {:?} must match requested skill name {:?}",
                skill.frontmatter.name, args.name
            )
            .into());
        }
        self.skills
            .write()
            .entry(skill.agent_name.clone())
            .insert_entry(skill.clone());

        Ok(SkillContentOutput {
            name: skill.frontmatter.name,
            agent_name: skill.agent_name,
            description: skill.frontmatter.description,
            path: self.display_path(&target),
            content,
        })
    }

    /// Recursively load all `SKILL.md` files from the configured directories.
    pub async fn load(&self) -> Result<(), BoxError> {
        let mut skills = BTreeMap::new();
        let mut loaded_dirs = 0usize;

        for skills_dir in &self.skills_dirs {
            if !skills_dir.is_dir() {
                log::error!(
                    "skills directory {} does not exist, skipping load",
                    skills_dir.display()
                );
                continue;
            }

            loaded_dirs += 1;
            for (agent_name, skill) in load_skills_from_dir(skills_dir).await? {
                #[allow(clippy::map_entry)]
                if skills.contains_key(&agent_name) {
                    log::warn!(
                        "duplicate skill name {} at {}, skipping",
                        agent_name,
                        skill.base_dir.join("SKILL.md").display()
                    );
                } else {
                    skills.insert(agent_name, skill);
                }
            }
        }

        if loaded_dirs == 0 {
            return Ok(());
        }

        log::info!(
            "loaded {} skill(s) from {} configured skill directories: {}",
            skills.len(),
            loaded_dirs,
            format_path_list(&self.skills_dirs)
        );
        *self.skills.write() = skills;
        Ok(())
    }

    /// Retrieve the full [`Skill`] by its normalised name.
    pub fn get_skill(&self, lowercase_name: &str) -> Option<Skill> {
        self.skills.read().get(lowercase_name).cloned()
    }

    /// Return all loaded skills as [`SubAgent`]s with default tools included.
    pub fn subagents(&self) -> Vec<SubAgent> {
        self.skills
            .read()
            .values()
            .map(SubAgent::from)
            .map(|agent| self.with_default_tools(agent))
            .collect::<Vec<_>>()
    }

    /// Return all loaded skills.
    pub fn list(&self) -> BTreeMap<String, Skill> {
        self.skills.read().clone()
    }
}

impl SubAgentSet for SkillManager {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.skills.read().contains_key(lowercase_name)
    }

    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent> {
        self.skills
            .read()
            .get(lowercase_name)
            .map(SubAgent::from)
            .map(|agent| self.with_default_tools(agent))
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        match names {
            None => self
                .skills
                .read()
                .values()
                .map(SubAgent::from)
                .map(|agent| agent.definition())
                .collect(),
            Some(names) => {
                let skills = self.skills.read();
                names
                    .iter()
                    .filter_map(|name| {
                        skills
                            .get(&name.to_ascii_lowercase())
                            .map(SubAgent::from)
                            .map(|agent| agent.definition())
                    })
                    .collect()
            }
        }
    }

    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        self.skills
            .read()
            .get(&name.to_ascii_lowercase())
            .map(SubAgent::from)
            .map(|agent| {
                let supported_tags = agent.supported_resource_tags();
                select_resources(resources, &supported_tags)
            })
            .unwrap_or_default()
    }
}

impl Tool<BaseCtx> for SkillManager {
    type Args = SkillArgs;
    type Output = SkillContentOutput;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "description": "Read a reusable skill's SKILL.md file content by skill name. Create or update skills by editing files directly with shell or file tools, then reload the manager.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name in kebab-case (e.g. 'pdf-processing'). Returns the matching SKILL.md content so the agent can follow it directly."
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        Ok(ToolOutput::new(self.read_skill_action(args).await?))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::BaseCtx, engine::EngineBuilder, subagent::SubAgentSet};
    use std::sync::Arc;

    fn mock_ctx() -> BaseCtx {
        EngineBuilder::new().mock_ctx().base
    }

    fn skill_md(name: &str, description: &str, body: &str, allowed_tools: Option<&str>) -> String {
        let mut content = format!("---\nname: {name}\ndescription: {description}\n");
        if let Some(allowed_tools) = allowed_tools {
            content.push_str(&format!("allowed-tools: {allowed_tools}\n"));
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
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-test-{:016x}", rand::random::<u64>()));
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
allowed-tools: shell fetch
---

Beta instructions.
",
        )
        .await
        .unwrap();

        let mgr = SkillManager::new(tmp.clone());
        mgr.load().await.unwrap();

        assert!(mgr.contains_lowercase("skill_alpha"));
        assert!(mgr.contains_lowercase("skill_beta_skill"));
        assert!(!mgr.contains_lowercase("skill_gamma"));

        let alpha = mgr.get_lowercase("skill_alpha").unwrap();
        assert_eq!(alpha.description, "Alpha skill for testing.");
        assert_eq!(alpha.tools, DEFAULT_SKILL_TOOLS);

        let beta = mgr.get_lowercase("skill_beta_skill").unwrap();
        assert_eq!(
            beta.tools,
            vec![
                "shell",
                "read_file",
                "search_file",
                "write_file",
                "edit_file",
                "todo",
                "tools_select",
                "fetch"
            ]
        );
        assert!(beta.instructions.contains("Beta instructions."));

        let beta_skill = mgr.get_skill("skill_beta_skill").unwrap();
        assert_eq!(beta_skill.frontmatter.license.as_deref(), Some("MIT"));

        let beta_content = mgr
            .call_raw(mock_ctx(), json!({ "name": "beta-skill" }), Vec::new())
            .await
            .unwrap();
        assert_eq!(beta_content.output["name"], json!("beta-skill"));
        assert_eq!(beta_content.output["agent_name"], json!("skill_beta_skill"));
        assert_eq!(beta_content.output["path"], json!("beta-skill/SKILL.md"));
        assert!(
            beta_content.output["content"]
                .as_str()
                .unwrap()
                .contains("Beta instructions.")
        );

        tokio::fs::create_dir_all(tmp.join("gamma")).await.unwrap();
        tokio::fs::write(
            tmp.join("gamma/SKILL.md"),
            skill_md(
                "gamma",
                "Gamma skill for testing.",
                "Gamma instructions.",
                None,
            ),
        )
        .await
        .unwrap();

        mgr.load().await.unwrap();

        assert!(mgr.contains_lowercase("skill_gamma"));
        assert!(tmp.join("gamma/SKILL.md").exists());

        // Verify on-disk content is valid SKILL.md.
        let on_disk = tokio::fs::read_to_string(tmp.join("gamma/SKILL.md"))
            .await
            .unwrap();
        let reparsed = parse_skill_md(tmp.to_path_buf(), &on_disk).unwrap();
        assert_eq!(reparsed.frontmatter.name, "gamma");

        // Definitions.
        let defs = mgr.definitions(None);
        assert_eq!(defs.len(), 3);

        let defs_filtered = mgr.definitions(Some(&["skill_gamma".to_string()]));
        assert_eq!(defs_filtered.len(), 1);
        assert_eq!(defs_filtered[0].name, "skill_gamma");

        // Clean up.
        let _ = tokio::fs::remove_dir_all(&tmp).await;
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
            None,
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
                None,
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
                None,
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

        assert!(mgr.contains_lowercase("skill_alpha"));
        assert!(mgr.contains_lowercase("skill_beta"));

        let beta_content = mgr
            .call_raw(mock_ctx(), json!({ "name": "beta" }), Vec::new())
            .await
            .unwrap();
        assert_eq!(beta_content.output["name"], json!("beta"));
        assert_eq!(beta_content.output["agent_name"], json!("skill_beta"));
        assert_eq!(beta_content.output["path"], json!("beta/SKILL.md"));
        assert!(
            beta_content.output["content"]
                .as_str()
                .unwrap()
                .contains("Beta instructions.")
        );

        // Creation workflows should keep using the original default directory.
        assert!(mgr.default_skills_dir().ends_with("default"));

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
                Some("shell todo shell custom_tool"),
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
        assert_eq!(mgr.list().len(), 1);

        let subagents = mgr.subagents();
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].name, "skill_alpha");
        assert_eq!(
            subagents[0].tools,
            vec!["read_file", "todo", "shell", "custom_tool",]
        );

        let any = mgr.clone().into_any();
        assert!(any.downcast_ref::<SkillManager>().is_some());

        let mut resources = vec![Resource {
            _id: 1,
            name: "text".to_string(),
            tags: vec!["text".to_string()],
            ..Default::default()
        }];
        assert!(SubAgentSet::select_resources(mgr.as_ref(), "missing", &mut resources).is_empty());
        assert!(
            SubAgentSet::select_resources(mgr.as_ref(), "skill_alpha", &mut resources).is_empty()
        );
        assert_eq!(resources.len(), 1);
        resources.clear();
        assert!(
            SubAgentSet::select_resources(mgr.as_ref(), "skill_alpha", &mut resources).is_empty()
        );

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
                None,
            ),
        )
        .await
        .unwrap();
        tokio::fs::write(
            extra_dir.join("duplicate-one/SKILL.md"),
            skill_md("dupe", "Duplicate one.", "One.", None),
        )
        .await
        .unwrap();
        tokio::fs::write(
            extra_dir.join("duplicate-two/SKILL.md"),
            skill_md("dupe", "Duplicate two.", "Two.", None),
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
        assert_eq!(read.output["agent_name"], json!("skill_frontmatter_name"));
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
                None,
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

        assert!(mgr.contains_lowercase("skill_correct_name"));
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
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-val-{:016x}", rand::random::<u64>()));
        let tool = SkillManager::new(tmp.clone());
        let engine = EngineBuilder::new().empty().await.unwrap();
        assert!(engine.sub_agents_manager().insert(Arc::new(tool)).is_none());
    }
}
