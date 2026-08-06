//! Skills manager extension.
//!
//! This module provides:
//! - Loading skills from directory trees of `SKILL.md` files.
//! - Reading loaded skill files via the [`SkillManager`] tool.
//! - Materializing the subset of skills that opted into subagent execution as [`SubAgent`]s.
//!
//! Each `SKILL.md` follows the [Agent Skills specification](https://agentskills.io):
//! YAML frontmatter (`---` delimiters) with `name`, `description`, and optional
//! `license`, `compatibility`, `metadata`, `allowed-tools` fields. The Markdown body
//! becomes the skill's instructions.
//!
//! # Execution modes
//!
//! Skills default to [`SkillExecution::Inline`]: the calling agent reads SKILL.md through the
//! [`SkillManager`] tool and follows it in its own context, which is what progressive disclosure
//! means in the specification — the body reaches the agent that holds the conversation, the user,
//! and the turn's resources.
//!
//! A skill can opt into [`SkillExecution::Subagent`] with `execution: subagent` (or
//! `metadata.execution: subagent`) in its frontmatter. Those skills are additionally exposed as
//! isolated workers callable as `SA_<agent_name>`. Reserve it for procedures that are genuinely
//! independent of the conversation — long-running, parallelisable, or context-hungry work — since
//! a subagent receives only a self-contained prompt plus the resources matching its
//! `resource-tags`, and has no channel to the user.
//!
//! # Bundled files
//!
//! A skill's SKILL.md routinely points at scripts and references next to it. Reaching them goes
//! through the filesystem tools, which are sandboxed to their configured workspaces, so register
//! [`SkillManager::skills_dirs`] as filesystem workspaces (see [`crate::extension::fs`]) when
//! wiring the engine. Without that the agent is handed a `base_dir` it is not allowed to read.
//!
//! Skill names use kebab-case on disk (e.g. `my-skill`); they are normalised to
//! snake_case (`skill_my_skill`) for the subagent registry.

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
    context::{BaseCtx, SUB_AGENT_PREFIX},
    extension::fs::{ensure_file_size_within_limit, ensure_regular_file, normalize_relative_path},
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
    /// Skill description from the SKILL.md frontmatter.
    pub description: String,
    /// How this skill runs: `inline` (follow `content` yourself) or `subagent` (delegate).
    pub execution: SkillExecution,
    /// Callable name for `subagent` skills, absent for `inline` ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callable: Option<String>,
    /// Base directory of the skill, for resolving the bundled files `content` references.
    pub base_dir: String,
    /// Path to SKILL.md, relative to a configured skills directory when possible.
    pub path: String,
    /// Full SKILL.md content including YAML frontmatter and Markdown body.
    pub content: String,
}

/// Manages skills loaded from `SKILL.md` files on disk.
///
/// [`SkillManager`] implements [`Tool<BaseCtx>`] so that LLMs can read skill files at runtime and
/// follow them inline, and [`SubAgentSet`] so that the skills which declared
/// [`SkillExecution::Subagent`] are additionally callable as delegated workers.
pub struct SkillManager {
    /// Directory used by skill creation workflows. Loading also includes this directory.
    default_skills_dir: PathBuf,
    /// Directories scanned for skills, with `default_skills_dir` first.
    skills_dirs: Vec<PathBuf>,
    skills: RwLock<BTreeMap<String, Skill>>,
    /// Materialized [`SubAgent`] per skill, keyed by lowercase name.
    ///
    /// A `SubAgent` owns its live session registry in `Arc<SubSessions>`, so lookups must hand
    /// out clones of one stable instance. Rebuilding a `SubAgent` per lookup would give every
    /// caller a fresh empty registry, and a running session could never be found again.
    subagents: RwLock<BTreeMap<String, SubAgent>>,
    description: String,
    default_skill_tools: Vec<String>,
}

/// Tools granted to a subagent skill that declares no `allowed-tools`.
///
/// Includes [`SkillManager::NAME`] so skills compose: a delegated skill can read another skill's
/// SKILL.md and follow it inline, the same way its caller would.
static DEFAULT_SKILL_TOOLS: &[&str] = &[
    "shell",
    "read_file",
    "search_file",
    "write_file",
    "edit_file",
    "todo",
    "tools_select",
    SkillManager::NAME,
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
        "Read a skill's SKILL.md by name, following the Agent Skills specification. Agent Skills \
        are folders of instructions, scripts, and resources. Most skills run inline: this tool \
        returns the full SKILL.md and you follow it yourself in this conversation, reading the \
        bundled files it references from `base_dir` as you need them. A skill that declares \
        `execution: subagent` instead returns a `callable` name to delegate to, which runs it in \
        an isolated worker that cannot see this conversation or ask the user anything. \
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
            subagents: RwLock::new(BTreeMap::new()),
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

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Sets the tool names granted to skill subagents that declare no `allowed-tools`.
    pub fn with_default_skill_tools(mut self, tools: Vec<String>) -> Self {
        self.default_skill_tools = tools;
        self
    }

    /// Materializes a skill into its callable [`SubAgent`].
    ///
    /// `allowed-tools` is an upper bound, per the Agent Skills specification: a skill that
    /// declares it is granted exactly those tools and nothing else. Only a skill that declares no
    /// tools at all inherits [`Self::default_skill_tools`]. Unioning the defaults in would turn a
    /// restriction into an escalation — SKILL.md files are third-party content on disk, so a
    /// manifest asking for `read_file` must not come back holding `shell`.
    fn materialize(&self, skill: &Skill) -> SubAgent {
        let mut agent = SubAgent::from(skill);
        if agent.tools.is_empty() {
            agent.tools = self.default_skill_tools.clone();
        }
        agent
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
                return normalize_relative_path(stripped);
            }
        }

        if let Ok(canonical_path) = std::fs::canonicalize(path) {
            for skills_dir in &self.skills_dirs {
                if let Ok(root) = std::fs::canonicalize(skills_dir)
                    && let Ok(stripped) = canonical_path.strip_prefix(&root)
                {
                    return normalize_relative_path(stripped);
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
        let callable = skill
            .is_subagent()
            .then(|| format!("{SUB_AGENT_PREFIX}{}", skill.agent_name));
        let base_dir = skill.base_dir.display().to_string();
        self.upsert_skill(skill.clone());

        Ok(SkillContentOutput {
            name: skill.frontmatter.name,
            description: skill.frontmatter.description,
            execution: skill.execution,
            callable,
            base_dir,
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
        self.replace_skills(skills);
        Ok(())
    }

    /// Replaces the loaded skills and materialized subagents as one update, carrying over the
    /// live session registry of every skill that survived the reload.
    ///
    /// Only skills that opted into [`SkillExecution::Subagent`] become callables; inline skills
    /// are loaded and readable but never appear in the model's tool list.
    fn replace_skills(&self, skills: BTreeMap<String, Skill>) {
        let mut subagents = self.subagents.write();
        let rebuilt = skills
            .iter()
            .filter(|(_, skill)| skill.is_subagent())
            .map(|(name, skill)| {
                let mut agent = self.materialize(skill);
                if let Some(existing) = subagents.get(name) {
                    agent.subsessions = existing.subsessions.clone();
                }
                (name.clone(), agent)
            })
            .collect();
        *subagents = rebuilt;
        *self.skills.write() = skills;
    }

    /// Inserts or refreshes one skill after a direct `SKILL.md` read.
    ///
    /// Reading a skill makes a newly created subagent skill immediately callable without a full
    /// reload. Keep the stable materialized agent in sync with the parsed skill while preserving
    /// any live sessions already owned by that skill, and drop the callable when the skill on disk
    /// switched back to inline execution.
    fn upsert_skill(&self, skill: Skill) {
        let name = skill.agent_name.clone();
        let mut subagents = self.subagents.write();
        if skill.is_subagent() {
            let mut agent = self.materialize(&skill);
            if let Some(existing) = subagents.get(&name) {
                agent.subsessions = existing.subsessions.clone();
            }
            subagents.insert(name.clone(), agent);
        } else {
            subagents.remove(&name);
        }
        self.skills.write().insert(name, skill);
    }

    /// Retrieve the full [`Skill`] by its normalised name.
    pub fn get_skill(&self, lowercase_name: &str) -> Option<Skill> {
        self.skills.read().get(lowercase_name).cloned()
    }

    /// Return the loaded skills that opted into [`SkillExecution::Subagent`], materialized as
    /// [`SubAgent`]s. Inline skills are excluded: they are followed by the calling agent through
    /// the [`SkillManager`] tool rather than dispatched as callables.
    pub fn subagents(&self) -> Vec<SubAgent> {
        self.subagents.read().values().cloned().collect::<Vec<_>>()
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

    // Every lookup below goes through `subagents`, not `skills`: inline skills are followed by the
    // calling agent through the `skills_manager` tool and must never be dispatchable as callables.
    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.subagents.read().contains_key(lowercase_name)
    }

    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent> {
        // Clone the materialized instance so the returned agent shares the live session
        // registry; building a fresh `SubAgent` here would hand out an empty one.
        self.subagents.read().get(lowercase_name).cloned()
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        let subagents = self.subagents.read();
        match names {
            None => subagents.values().map(|agent| agent.definition()).collect(),
            Some(names) => names
                .iter()
                .filter_map(|name| {
                    subagents
                        .get(&name.to_ascii_lowercase())
                        .map(|agent| agent.definition())
                })
                .collect(),
        }
    }

    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        self.subagents
            .read()
            .get(&name.to_ascii_lowercase())
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
                        "description": "Skill name in kebab-case (e.g. 'pdf-processing'). Returns the matching SKILL.md content. When `execution` is `inline`, follow that content yourself; when it is `subagent`, call the returned `callable` with a self-contained prompt instead."
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

        let mgr = Arc::new(
            SkillManager::new(root.clone())
                .with_description("custom skill reader".to_string())
                .with_default_skill_tools(vec!["read_file".to_string(), "todo".to_string()]),
        );
        assert_eq!(mgr.description(), "custom skill reader");
        assert_eq!(mgr.list().len(), 0);

        mgr.load().await.unwrap();
        assert_eq!(mgr.list().len(), 2);

        // Only the skill that opted in is materialized, and its declared tools replace the
        // configured defaults rather than merging with them.
        let subagents = mgr.subagents();
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].name, "skill_alpha");
        assert_eq!(subagents[0].tools, vec!["shell", "todo", "custom_tool"]);

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
            SubAgentSet::select_resources(mgr.as_ref(), "skill_inline_one", &mut resources)
                .is_empty()
        );
        assert_eq!(resources.len(), 1);
        // A subagent skill that declares no `resource-tags` takes what the caller offers.
        let selected = SubAgentSet::select_resources(mgr.as_ref(), "skill_alpha", &mut resources);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "text");
        assert!(resources.is_empty());
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
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-val-{:016x}", rand::random::<u64>()));
        let tool = SkillManager::new(tmp.clone());
        let engine = EngineBuilder::new().empty().await.unwrap();
        assert!(engine.sub_agents_manager().insert(Arc::new(tool)).is_none());
    }
}
