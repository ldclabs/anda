//! Skills manager extension.
//!
//! This module provides:
//! - Loading skills from a directory tree of `SKILL.md` files into [`SubAgent`] instances.
//! - Creating or updating skill files on disk via the [`SkillManager`] tool.
//!
//! Each `SKILL.md` follows the [Agent Skills specification](https://agentskills.io):
//! YAML frontmatter (`---` delimiters) with `name`, `description`, and optional
//! `license`, `compatibility`, `metadata`, `allowed-tools` fields. The Markdown body
//! becomes the agent's `instructions`.
//!
//! Skill names use kebab-case on disk (e.g. `my-skill`); they are normalised to
//! snake_case (`my_skill`) when loaded as [`SubAgent`] instances.

use anda_core::{
    Agent, BoxError, FunctionDefinition, Resource, Tool, ToolOutput, validate_function_name,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

use crate::context::{BaseCtx, SubAgent};

mod types;
pub use types::*;

// ---------------------------------------------------------------------------
// SkillManager
// ---------------------------------------------------------------------------

/// Arguments for creating or updating a skill via the [`SkillManager`] tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SkillArgs {
    /// Skill name in kebab-case (e.g. `pdf-processing`). 1-64 chars,
    /// lowercase alphanumeric and hyphens only.
    pub name: String,
    /// What the skill does and when to use it (1-1024 characters).
    pub description: String,
    /// Durable Markdown instructions for the skill agent.
    pub instructions: String,
    /// Optional license name or reference.
    #[serde(default)]
    pub license: Option<String>,
    /// Optional environment requirements (1-500 characters).
    #[serde(default)]
    pub compatibility: Option<String>,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Space-delimited list of pre-approved tools. Defaults to `"shell"`.
    #[serde(default)]
    pub allowed_tools: Option<String>,
}

/// Manages skills loaded from `SKILL.md` files on disk.
///
/// [`SkillManager`] implements [`Tool<BaseCtx>`] so that LLMs can create or
/// update skill files at runtime. Skills loaded here are exposed as [`SubAgent`]
/// instances that the engine can invoke.
pub struct SkillManager {
    skills_dir: PathBuf,
    skills: RwLock<BTreeMap<String, Skill>>,
    description: String,
}

impl SkillManager {
    /// Tool name used for registration.
    pub const NAME: &'static str = "skills";

    /// Create a new, empty manager rooted at `skills_dir`.
    pub fn new(skills_dir: PathBuf) -> Self {
        Self {
            skills_dir,
            skills: RwLock::new(BTreeMap::new()),
            description:
                "Create or update a reusable skill following the Agent Skills specification. \
         Agent Skills are folders of instructions, scripts, and resources that agents \
         can discover and use to perform tasks more accurately and efficiently."
                    .to_string(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Recursively load all `SKILL.md` files from the configured directory.
    pub async fn load(&self) -> Result<(), BoxError> {
        if !self.skills_dir.is_dir() {
            log::error!(
                "skills directory {} does not exist, skipping load",
                self.skills_dir.display()
            );
            return Ok(());
        }

        let skills = load_skills_from_dir(&self.skills_dir).await?;
        log::info!(
            "loaded {} skill(s) from {}",
            skills.len(),
            self.skills_dir.display()
        );
        *self.skills.write() = skills;
        Ok(())
    }

    /// Check whether a skill with the given (lowercase) name exists.
    pub fn has(&self, lowercase_name: &str) -> bool {
        self.skills.read().contains_key(lowercase_name)
    }

    /// Retrieve a skill as a [`SubAgent`] by its normalised name.
    pub fn get(&self, lowercase_name: &str) -> Option<SubAgent> {
        self.skills.read().get(lowercase_name).map(SubAgent::from)
    }

    /// Retrieve the full [`Skill`] by its normalised name.
    pub fn get_skill(&self, lowercase_name: &str) -> Option<Skill> {
        self.skills.read().get(lowercase_name).cloned()
    }

    pub fn subagents(&self) -> Vec<SubAgent> {
        self.skills
            .read()
            .values()
            .map(SubAgent::from)
            .collect::<Vec<_>>()
    }

    /// Return all loaded skills.
    pub fn list(&self) -> BTreeMap<String, Skill> {
        self.skills.read().clone()
    }

    /// Return [`FunctionDefinition`]s for all or specified skills.
    pub fn definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        let names: Option<Vec<String>> =
            names.map(|names| names.iter().map(|n| n.to_ascii_lowercase()).collect());
        self.skills
            .read()
            .iter()
            .filter_map(|(name, skill): (&String, &Skill)| {
                let agent = SubAgent::from(skill);
                match &names {
                    Some(names) => {
                        if names.contains(name) {
                            Some(agent.definition())
                        } else {
                            None
                        }
                    }
                    None => Some(agent.definition()),
                }
            })
            .collect()
    }

    /// Create or update a skill on disk and in memory.
    pub async fn upsert(&self, args: SkillArgs) -> Result<Skill, BoxError> {
        // Normalise and validate the name.
        validate_skill_name(&args.name)?;
        let agent_name = normalise_skill_agent_name(&args.name);
        validate_function_name(&agent_name)?;

        if args.description.is_empty() {
            return Err("description must not be empty".into());
        }
        if args.description.len() > 1024 {
            return Err("description must not exceed 1024 characters".into());
        }
        if let Some(compat) = &args.compatibility
            && (compat.is_empty() || compat.len() > 500)
        {
            return Err("compatibility must be 1-500 characters".into());
        }

        let fm = SkillFrontmatter {
            name: args.name,
            description: args.description,
            license: args.license,
            compatibility: args.compatibility,
            metadata: args.metadata,
            allowed_tools: args.allowed_tools.clone(),
        };

        let tools = match &args.allowed_tools {
            Some(at) if !at.trim().is_empty() => {
                at.split_whitespace().map(|s| s.to_string()).collect()
            }
            _ => Vec::new(),
        };

        let base_dir = self.skills_dir.join(&fm.name);
        let skill = Skill {
            frontmatter: fm,
            instructions: args.instructions,
            agent_name: agent_name.clone(),
            tools,
            base_dir,
        };

        let content = format_skill_md(&skill)?;
        tokio::fs::create_dir_all(&skill.base_dir).await?;
        tokio::fs::write(skill.base_dir.join("SKILL.md"), content.as_bytes()).await?;

        self.skills
            .write()
            .insert(agent_name.clone(), skill.clone());
        Ok(skill)
    }
}

impl Tool<BaseCtx> for SkillManager {
    type Args = SkillArgs;
    type Output = Skill;

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
                "description": "Create or update a reusable skill stored as SKILL.md on disk.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name in kebab-case (e.g. 'pdf-processing'). 1-64 chars, lowercase alphanumeric and hyphens only. Must not start/end with a hyphen or contain consecutive hyphens.",
                        "pattern": "^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$",
                        "maxLength": 64
                    },
                    "description": {
                        "type": "string",
                        "description": "What the skill does and when to use it. Include specific keywords that help agents identify relevant tasks.",
                        "minLength": 1,
                        "maxLength": 1024
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Durable Markdown instructions for the skill agent. Include step-by-step workflows, examples, edge cases, and expected output style. Keep under 500 lines; reference separate files for detailed material.",
                        "minLength": 1
                    },
                    "license": {
                        "type": "string",
                        "description": "Optional license name or reference (e.g. 'Apache-2.0' or 'Proprietary. LICENSE.txt has complete terms')."
                    },
                    "compatibility": {
                        "type": "string",
                        "description": "Optional environment requirements (e.g. 'Requires Python 3.14+ and uv').",
                        "minLength": 1,
                        "maxLength": 500
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional arbitrary key-value metadata (e.g. author, version).",
                        "additionalProperties": { "type": "string" }
                    },
                    "allowed_tools": {
                        "type": "string",
                        "description": "Space-delimited list of pre-approved tool names the skill may use"
                    }
                },
                "required": ["name", "description", "instructions"],
                "additionalProperties": false
            }),
            strict: None,
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let skill = self.upsert(args).await?;
        Ok(ToolOutput::new(skill))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Tool definition --

    #[test]
    fn skill_manager_tool_definition_schema() {
        let mgr = SkillManager::new(PathBuf::from("/tmp/skills"));
        let def = mgr.definition();
        assert_eq!(def.name, "skills");
        assert!(def.description.contains("Agent Skills specification"));
        assert_eq!(def.parameters["additionalProperties"], json!(false));
        assert_eq!(def.parameters["properties"]["name"]["maxLength"], json!(64));
        assert_eq!(
            def.parameters["properties"]["description"]["maxLength"],
            json!(1024)
        );
    }

    // -- integration: load & upsert --

    #[tokio::test]
    async fn load_and_upsert_from_temp_dir() {
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-test-{:016x}", rand::random::<u64>()));
        // Create skills with directory names matching the skill name.
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

        assert!(mgr.has("skill_alpha"));
        assert!(mgr.has("skill_beta_skill"));
        assert!(!mgr.has("skill_gamma"));

        let alpha = mgr.get("skill_alpha").unwrap();
        assert_eq!(alpha.description, "Alpha skill for testing.");
        assert!(alpha.tools.is_empty());

        let beta = mgr.get("skill_beta_skill").unwrap();
        assert_eq!(beta.tools, vec!["shell", "fetch"]);
        assert!(beta.instructions.contains("Beta instructions."));

        let beta_skill = mgr.get_skill("skill_beta_skill").unwrap();
        assert_eq!(beta_skill.frontmatter.license.as_deref(), Some("MIT"));

        // Upsert a new skill.
        mgr.upsert(SkillArgs {
            name: "gamma".to_string(),
            description: "Gamma skill for testing.".to_string(),
            instructions: "Gamma instructions.".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();

        assert!(mgr.has("skill_gamma"));
        assert!(tmp.join("gamma/SKILL.md").exists());

        // Verify on-disk content is valid SKILL.md.
        let on_disk = tokio::fs::read_to_string(tmp.join("gamma/SKILL.md"))
            .await
            .unwrap();
        let reparsed = parse_skill_md(tmp.to_path_buf(), &on_disk).unwrap();
        assert_eq!(reparsed.frontmatter.name, "gamma");

        // Update existing skill.
        mgr.upsert(SkillArgs {
            name: "alpha".to_string(),
            description: "Alpha v2 for testing.".to_string(),
            instructions: "Updated.".to_string(),
            allowed_tools: Some("shell fetch".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();

        let alpha2 = mgr.get("skill_alpha").unwrap();
        assert_eq!(alpha2.description, "Alpha v2 for testing.");
        assert!(
            alpha2.instructions.contains(
            "You are a sub-agent specialised in \"alpha\" skill. Follow these instructions:\n\nUpdated.")
        );
        assert_eq!(alpha2.tools, vec!["shell", "fetch"]);

        // File should be at the original path.
        let on_disk = tokio::fs::read_to_string(tmp.join("alpha/SKILL.md"))
            .await
            .unwrap();
        assert!(on_disk.contains("Alpha v2 for testing."));

        // Definitions.
        let defs = mgr.definitions(None);
        assert_eq!(defs.len(), 3);

        let defs_filtered = mgr.definitions(Some(&["skill_gamma"]));
        assert_eq!(defs_filtered.len(), 1);
        assert_eq!(defs_filtered[0].name, "skill_gamma");

        // Clean up.
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    #[ignore = "skip"]
    async fn load_skips_mismatched_dir_name() {
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

        // Should be skipped because dir name != skill name.
        assert!(!mgr.has("skill_correct_name"));

        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[test]
    fn upsert_validates_description_length() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-val-{:016x}", rand::random::<u64>()));

        let mgr = SkillManager::new(tmp.clone());

        // Empty description.
        let err = rt.block_on(mgr.upsert(SkillArgs {
            name: "test".to_string(),
            description: "".to_string(),
            instructions: "body".to_string(),
            ..Default::default()
        }));
        assert!(err.is_err());

        // Too-long description.
        let err = rt.block_on(mgr.upsert(SkillArgs {
            name: "test".to_string(),
            description: "x".repeat(1025),
            instructions: "body".to_string(),
            ..Default::default()
        }));
        assert!(err.is_err());
    }
}
