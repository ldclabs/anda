//! Skills manager extension.
//!
//! This module provides:
//! - Loading skills from a directory tree of `SKILL.md` files into [`SubAgent`] instances.
//! - Managing skill files on disk via the [`SkillManager`] tool.
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
    validate_function_name,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    any::Any,
    collections::BTreeMap,
    ffi::OsStr,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use crate::{
    context::{BaseCtx, SubAgent, SubAgentSet},
    extension::fs::{
        atomic_write_file, ensure_file_size_within_limit, ensure_regular_file, resolve_write_path,
    },
};

mod types;
pub use types::*;

// ---------------------------------------------------------------------------
// SkillManager
// ---------------------------------------------------------------------------

const MAX_DESCRIPTION_LENGTH: usize = 1024;
const MAX_COMPATIBILITY_LENGTH: usize = 500;
const MAX_SKILL_CONTENT_CHARS: usize = 100_000;
const MAX_SKILL_FILE_BYTES: u64 = 512 * 1024;
static ALLOWED_SKILL_SUBDIRS: &[&str] = &["references", "templates", "scripts", "assets"];

/// Supported skill management actions.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkillAction {
    Create,
    Patch,
    Edit,
    Delete,
    WriteFile,
    RemoveFile,
}

/// Normalized result returned by the [`SkillManager`] tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SkillActionOutput {
    /// The action that was executed.
    pub action: String,
    /// Skill name targeted by the action.
    pub name: String,
    /// Human-readable status message.
    pub message: String,
    /// Path affected by the action, relative to the configured skills directory when possible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Number of replacements applied by `patch`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacements: Option<usize>,
    /// Number of matches found by `patch` before replacement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_matches: Option<usize>,
}

/// Arguments for managing skills via the [`SkillManager`] tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SkillArgs {
    /// Action to perform.
    pub action: SkillAction,
    /// Skill name in kebab-case (e.g. `pdf-processing`). 1-64 chars,
    /// lowercase alphanumeric and hyphens only.
    pub name: String,
    /// What the skill does and when to use it (1-1024 characters).
    #[serde(default)]
    pub description: String,
    /// Durable Markdown instructions for the skill agent.
    #[serde(default)]
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
    /// Full SKILL.md content including YAML frontmatter and Markdown body.
    #[serde(default)]
    pub content: Option<String>,
    /// Optional category directory used by `create`.
    #[serde(default)]
    pub category: Option<String>,
    /// Supporting file path used by `patch`, `write_file`, and `remove_file`.
    #[serde(default)]
    pub file_path: Option<String>,
    /// Supporting file content used by `write_file`.
    #[serde(default)]
    pub file_content: Option<String>,
    /// String to replace when `action = patch`.
    #[serde(default)]
    pub old_string: Option<String>,
    /// Replacement string when `action = patch`.
    #[serde(default)]
    pub new_string: Option<String>,
    /// Whether `patch` should replace all matches instead of requiring a unique one.
    #[serde(default)]
    pub replace_all: bool,
}

/// Manages skills loaded from `SKILL.md` files on disk.
///
/// [`SkillManager`] implements [`Tool<BaseCtx>`] so that LLMs can manage skill
/// files at runtime. Skills loaded here are exposed as [`SubAgent`]
/// instances that the engine can invoke.
pub struct SkillManager {
    skills_dir: PathBuf,
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
];

impl SkillManager {
    /// Tool name used for registration.
    pub const NAME: &'static str = "skills_manager";

    /// Create a new, empty manager rooted at `skills_dir`.
    pub fn new(skills_dir: PathBuf) -> Self {
        Self {
            skills_dir,
            skills: RwLock::new(BTreeMap::new()),
            description: "Manage reusable skills following the Agent Skills specification. \
         Agent Skills are folders of instructions, scripts, and resources that agents \
         can discover and use to perform tasks more accurately and efficiently."
                .to_string(),
            default_skill_tools: DEFAULT_SKILL_TOOLS.iter().map(|s| s.to_string()).collect(),
        }
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

    fn validate_skill_fields(
        description: &str,
        instructions: &str,
        compatibility: &Option<String>,
    ) -> Result<(), BoxError> {
        if description.trim().is_empty() {
            return Err("description must not be empty".into());
        }
        if description.len() > MAX_DESCRIPTION_LENGTH {
            return Err(
                format!("description must not exceed {MAX_DESCRIPTION_LENGTH} characters").into(),
            );
        }
        if instructions.trim().is_empty() {
            return Err("instructions must not be empty".into());
        }
        if let Some(compat) = compatibility
            && (compat.is_empty() || compat.len() > MAX_COMPATIBILITY_LENGTH)
        {
            return Err(
                format!("compatibility must be 1-{MAX_COMPATIBILITY_LENGTH} characters").into(),
            );
        }

        Ok(())
    }

    fn validate_skill_content_size(content: &str, label: &str) -> Result<(), BoxError> {
        if content.chars().count() > MAX_SKILL_CONTENT_CHARS {
            return Err(
                format!("{label} must not exceed {MAX_SKILL_CONTENT_CHARS} characters").into(),
            );
        }

        Ok(())
    }

    fn validate_supporting_file_content(content: &str, file_path: &str) -> Result<(), BoxError> {
        Self::validate_skill_content_size(content, file_path)?;

        let size = content.len() as u64;
        if size > MAX_SKILL_FILE_BYTES {
            return Err(format!(
                "{file_path} is {size} bytes which exceeds the {MAX_SKILL_FILE_BYTES} byte limit"
            )
            .into());
        }

        Ok(())
    }

    fn normalize_category(category: Option<&str>) -> Result<Option<String>, BoxError> {
        let Some(category) = category.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };

        if category.len() > 64 {
            return Err("category must not exceed 64 characters".into());
        }

        let mut chars = category.chars();
        let Some(first) = chars.next() else {
            return Ok(None);
        };
        if !matches!(first, 'a'..='z' | '0'..='9') {
            return Err("category must start with a lowercase letter or digit".into());
        }
        for ch in chars {
            if !matches!(ch, 'a'..='z' | '0'..='9' | '-' | '_' | '.') {
                return Err(
                    "category may only contain lowercase letters, digits, hyphens, underscores, and dots"
                        .into(),
                );
            }
        }

        Ok(Some(category.to_string()))
    }

    fn validate_supporting_file_path(file_path: &str) -> Result<PathBuf, BoxError> {
        if file_path.trim().is_empty() {
            return Err("file_path is required".into());
        }

        let path = PathBuf::from(file_path);
        if path.is_absolute() {
            return Err("file_path must be relative to the skill directory".into());
        }

        let components: Vec<_> = path.components().collect();
        if components.len() < 2 {
            return Err(
                "file_path must point to a file inside references/, templates/, scripts/, or assets/"
                    .into(),
            );
        }

        for component in &components {
            match component {
                Component::Normal(_) => {}
                Component::CurDir => return Err("file_path must not contain '.' segments".into()),
                Component::ParentDir => {
                    return Err("file_path must not contain '..' segments".into());
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err("file_path must be relative to the skill directory".into());
                }
            }
        }

        let first = match components.first() {
            Some(Component::Normal(first)) => first.to_string_lossy().into_owned(),
            _ => {
                return Err(
                    "file_path must start inside references/, templates/, scripts/, or assets/"
                        .into(),
                );
            }
        };
        if !ALLOWED_SKILL_SUBDIRS.contains(&first.as_str()) {
            return Err(format!(
                "file_path must start with one of {}",
                ALLOWED_SKILL_SUBDIRS.join(", ")
            )
            .into());
        }

        Ok(path)
    }

    fn build_skill_from_fields(
        &self,
        args: &SkillArgs,
        base_dir: PathBuf,
    ) -> Result<Skill, BoxError> {
        validate_skill_name(&args.name)?;
        let agent_name = normalise_skill_agent_name(&args.name);
        validate_function_name(&agent_name)?;
        Self::validate_skill_fields(&args.description, &args.instructions, &args.compatibility)?;

        let fm = SkillFrontmatter {
            name: args.name.clone(),
            description: args.description.clone(),
            license: args.license.clone(),
            compatibility: args.compatibility.clone(),
            metadata: args.metadata.clone(),
            allowed_tools: args.allowed_tools.clone(),
        };

        let tools = match &args.allowed_tools {
            Some(at) if !at.trim().is_empty() => {
                at.split_whitespace().map(|s| s.to_string()).collect()
            }
            _ => Vec::new(),
        };

        Ok(Skill {
            frontmatter: fm,
            instructions: args.instructions.clone(),
            agent_name,
            tools,
            base_dir,
        })
    }

    fn validate_skill_content(
        &self,
        name: &str,
        content: &str,
        base_dir: PathBuf,
    ) -> Result<Skill, BoxError> {
        Self::validate_skill_content_size(content, "SKILL.md")?;

        let skill = parse_skill_md(base_dir, content)?;
        if skill.frontmatter.name != name {
            return Err(format!(
                "SKILL.md frontmatter name {:?} must match target skill name {:?}",
                skill.frontmatter.name, name
            )
            .into());
        }
        if skill.instructions.trim().is_empty() {
            return Err("SKILL.md must have content after the frontmatter".into());
        }

        Ok(skill)
    }

    async fn write_text_file(&self, path: &Path, content: &str) -> Result<(), BoxError> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|err| format!("Failed to create parent directories: {err}"))?;
        }

        let existing_permissions = match tokio::fs::symlink_metadata(path).await {
            Ok(meta) => {
                ensure_regular_file(&meta, "Writing multiply-linked files is not allowed")?;
                Some(meta.permissions())
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
            Err(err) => return Err(format!("Failed to inspect file metadata: {err}").into()),
        };

        atomic_write_file(path, content.as_bytes(), existing_permissions.as_ref()).await
    }

    async fn read_text_file(&self, path: &Path, max_size: u64) -> Result<String, BoxError> {
        let meta = tokio::fs::symlink_metadata(path)
            .await
            .map_err(|err| format!("Failed to inspect file metadata: {err}"))?;
        ensure_regular_file(&meta, "Reading multiply-linked files is not allowed")?;
        ensure_file_size_within_limit(&meta, max_size)?;

        let data = tokio::fs::read(path)
            .await
            .map_err(|err| format!("Failed to read file: {err}"))?;
        String::from_utf8(data)
            .map_err(|_| "Only UTF-8 skill files are supported by skills_manager".into())
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

        if self.skills_dir.is_dir() {
            for path in find_skill_files(&self.skills_dir).await? {
                let Some(base_dir) = path.parent() else {
                    continue;
                };
                let base_dir = base_dir.to_path_buf();
                let dir_name_matches = base_dir.file_name() == Some(OsStr::new(name));
                let frontmatter_name_matches = if dir_name_matches {
                    true
                } else if let Ok(content) = tokio::fs::read_to_string(&path).await {
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

        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.pop()),
            _ => Err(format!(
                "multiple skills named {:?} exist under {}",
                name,
                self.skills_dir.display()
            )
            .into()),
        }
    }

    async fn reload_skill(&self, name: &str) -> Result<Skill, BoxError> {
        self.load().await?;

        let agent_name = normalise_skill_agent_name(name);
        self.get_skill(&agent_name).ok_or_else(|| {
            format!(
                "skill {:?} was written to disk but could not be reloaded",
                name
            )
            .into()
        })
    }

    async fn prune_empty_dirs(&self, mut dir: PathBuf, stop_at: &Path) -> Result<(), BoxError> {
        while dir != stop_at {
            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(entries) => entries,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
                Err(err) => {
                    return Err(
                        format!("Failed to inspect directory {}: {err}", dir.display()).into(),
                    );
                }
            };

            if entries
                .next_entry()
                .await
                .map_err(|err| format!("Failed to inspect directory {}: {err}", dir.display()))?
                .is_some()
            {
                break;
            }

            tokio::fs::remove_dir(&dir).await.map_err(|err| {
                format!("Failed to remove empty directory {}: {err}", dir.display())
            })?;

            let Some(parent) = dir.parent() else {
                break;
            };
            dir = parent.to_path_buf();
        }

        Ok(())
    }

    fn skill_md_relative_path(name: &str, category: Option<&str>) -> String {
        match category {
            Some(category) => format!("{category}/{name}/SKILL.md"),
            None => format!("{name}/SKILL.md"),
        }
    }

    fn display_path(&self, path: &Path) -> String {
        if let Ok(stripped) = path.strip_prefix(&self.skills_dir) {
            return stripped.display().to_string();
        }

        if let Ok(root) = std::fs::canonicalize(&self.skills_dir)
            && let Ok(stripped) = path.strip_prefix(&root)
        {
            return stripped.display().to_string();
        }

        path.display().to_string()
    }

    fn action_output(
        &self,
        action: &str,
        name: &str,
        message: String,
        path: Option<&Path>,
        replacements: Option<usize>,
        total_matches: Option<usize>,
    ) -> SkillActionOutput {
        SkillActionOutput {
            action: action.to_string(),
            name: name.to_string(),
            message,
            path: path.map(|path| self.display_path(path)),
            replacements,
            total_matches,
        }
    }

    async fn create_skill_action(&self, args: SkillArgs) -> Result<SkillActionOutput, BoxError> {
        validate_skill_name(&args.name)?;
        if self.find_skill_dir(&args.name).await?.is_some() {
            return Err(format!("skill {:?} already exists", args.name).into());
        }

        let category = Self::normalize_category(args.category.as_deref())?;
        tokio::fs::create_dir_all(&self.skills_dir)
            .await
            .map_err(|err| format!("Failed to create skills directory: {err}"))?;

        let relative_path = Self::skill_md_relative_path(&args.name, category.as_deref());
        let target = resolve_write_path(&self.skills_dir, &relative_path).await?;
        let base_dir = target
            .parent()
            .ok_or_else(|| "Failed to determine skill directory".to_string())?
            .to_path_buf();

        let content = match args.content.as_deref() {
            Some(content) => {
                self.validate_skill_content(&args.name, content, base_dir)?;
                content.to_string()
            }
            None => {
                let skill = self.build_skill_from_fields(&args, base_dir)?;
                let content = format_skill_md(&skill)?;
                Self::validate_skill_content_size(&content, "SKILL.md")?;
                content
            }
        };

        self.write_text_file(&target, &content).await?;
        self.reload_skill(&args.name).await?;

        Ok(self.action_output(
            "create",
            &args.name,
            format!("Skill '{}' created.", args.name),
            Some(&target),
            None,
            None,
        ))
    }

    async fn edit_skill_action(&self, args: SkillArgs) -> Result<SkillActionOutput, BoxError> {
        validate_skill_name(&args.name)?;
        let skill_dir = self
            .find_skill_dir(&args.name)
            .await?
            .ok_or_else(|| format!("skill {:?} not found", args.name))?;
        let target = skill_dir.join("SKILL.md");

        let content = match args.content.as_deref() {
            Some(content) => {
                self.validate_skill_content(&args.name, content, skill_dir.clone())?;
                content.to_string()
            }
            None => {
                let skill = self.build_skill_from_fields(&args, skill_dir.clone())?;
                let content = format_skill_md(&skill)?;
                Self::validate_skill_content_size(&content, "SKILL.md")?;
                content
            }
        };

        self.write_text_file(&target, &content).await?;
        self.reload_skill(&args.name).await?;

        Ok(self.action_output(
            "edit",
            &args.name,
            format!("Skill '{}' updated.", args.name),
            Some(&target),
            None,
            None,
        ))
    }

    async fn patch_skill_action(&self, args: SkillArgs) -> Result<SkillActionOutput, BoxError> {
        validate_skill_name(&args.name)?;
        let old_string = args
            .old_string
            .as_deref()
            .ok_or("old_string is required for patch")?;
        if old_string.is_empty() {
            return Err("old_string must not be empty".into());
        }
        let new_string = args
            .new_string
            .as_deref()
            .ok_or("new_string is required for patch")?;

        let skill_dir = self
            .find_skill_dir(&args.name)
            .await?
            .ok_or_else(|| format!("skill {:?} not found", args.name))?;

        let target = match args.file_path.as_deref() {
            Some(file_path) => {
                Self::validate_supporting_file_path(file_path)?;
                resolve_write_path(&skill_dir, file_path).await?
            }
            None => skill_dir.join("SKILL.md"),
        };

        let original = self.read_text_file(&target, MAX_SKILL_FILE_BYTES).await?;
        let total_matches = original.match_indices(old_string).count();
        if total_matches == 0 {
            return Err("old_string was not found in the target file".into());
        }
        if !args.replace_all && total_matches != 1 {
            return Err(format!(
                "old_string matched {total_matches} times; set replace_all=true or provide a more specific snippet"
            )
            .into());
        }

        let replacements = if args.replace_all { total_matches } else { 1 };
        let updated = if args.replace_all {
            original.replace(old_string, new_string)
        } else {
            original.replacen(old_string, new_string, 1)
        };

        if args.file_path.is_some() {
            let file_path = args.file_path.as_deref().unwrap_or_default();
            Self::validate_supporting_file_content(&updated, file_path)?;
        } else {
            self.validate_skill_content(&args.name, &updated, skill_dir.clone())?;
        }

        self.write_text_file(&target, &updated).await?;
        if args.file_path.is_none() {
            self.reload_skill(&args.name).await?;
        }

        let target_name = args.file_path.as_deref().unwrap_or("SKILL.md");
        Ok(self.action_output(
            "patch",
            &args.name,
            format!("Patched {target_name} in skill '{}'.", args.name),
            Some(&target),
            Some(replacements),
            Some(total_matches),
        ))
    }

    async fn delete_skill_action(&self, args: SkillArgs) -> Result<SkillActionOutput, BoxError> {
        validate_skill_name(&args.name)?;
        let skill_dir = self
            .find_skill_dir(&args.name)
            .await?
            .ok_or_else(|| format!("skill {:?} not found", args.name))?;

        tokio::fs::remove_dir_all(&skill_dir)
            .await
            .map_err(|err| format!("Failed to remove skill directory: {err}"))?;

        if let Some(parent) = skill_dir.parent() {
            self.prune_empty_dirs(parent.to_path_buf(), &self.skills_dir)
                .await?;
        }
        self.load().await?;

        Ok(self.action_output(
            "delete",
            &args.name,
            format!("Skill '{}' deleted.", args.name),
            Some(&skill_dir),
            None,
            None,
        ))
    }

    async fn write_skill_file_action(
        &self,
        args: SkillArgs,
    ) -> Result<SkillActionOutput, BoxError> {
        validate_skill_name(&args.name)?;
        let file_path = args
            .file_path
            .as_deref()
            .ok_or("file_path is required for write_file")?;
        Self::validate_supporting_file_path(file_path)?;
        let file_content = args
            .file_content
            .as_deref()
            .ok_or("file_content is required for write_file")?;
        Self::validate_supporting_file_content(file_content, file_path)?;

        let skill_dir = self
            .find_skill_dir(&args.name)
            .await?
            .ok_or_else(|| format!("skill {:?} not found", args.name))?;
        let target = resolve_write_path(&skill_dir, file_path).await?;

        self.write_text_file(&target, file_content).await?;

        Ok(self.action_output(
            "write_file",
            &args.name,
            format!("Wrote {file_path} in skill '{}'.", args.name),
            Some(&target),
            None,
            None,
        ))
    }

    async fn remove_skill_file_action(
        &self,
        args: SkillArgs,
    ) -> Result<SkillActionOutput, BoxError> {
        validate_skill_name(&args.name)?;
        let file_path = args
            .file_path
            .as_deref()
            .ok_or("file_path is required for remove_file")?;
        Self::validate_supporting_file_path(file_path)?;

        let skill_dir = self
            .find_skill_dir(&args.name)
            .await?
            .ok_or_else(|| format!("skill {:?} not found", args.name))?;
        let target = resolve_write_path(&skill_dir, file_path).await?;
        let meta = tokio::fs::symlink_metadata(&target)
            .await
            .map_err(|err| format!("Failed to inspect file metadata: {err}"))?;
        ensure_regular_file(&meta, "Removing multiply-linked files is not allowed")?;

        tokio::fs::remove_file(&target)
            .await
            .map_err(|err| format!("Failed to remove file: {err}"))?;

        if let Some(parent) = target.parent()
            && parent != skill_dir.as_path()
        {
            self.prune_empty_dirs(parent.to_path_buf(), &skill_dir)
                .await?;
        }

        Ok(self.action_output(
            "remove_file",
            &args.name,
            format!("Removed {file_path} from skill '{}'.", args.name),
            Some(&target),
            None,
            None,
        ))
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

    fn select_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        if let Some(name) = prefixed_name.strip_prefix("SA_") {
            self.skills
                .read()
                .get(&name.to_ascii_lowercase())
                .map(SubAgent::from)
                .map(|agent| {
                    let supported_tags = agent.supported_resource_tags();
                    select_resources(resources, &supported_tags)
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

impl Tool<BaseCtx> for SkillManager {
    type Args = SkillArgs;
    type Output = SkillActionOutput;

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
                "description": "Manage reusable skills stored as SKILL.md directories on disk. Supports create, patch, edit, delete, write_file, and remove_file.",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "patch", "edit", "delete", "write_file", "remove_file"],
                        "description": "Operation to perform."
                    },
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
                        "description": "Durable Markdown instructions for the skill agent. Used by create or edit when content is not provided.",
                        "minLength": 1
                    },
                    "content": {
                        "type": "string",
                        "description": "Full SKILL.md content including YAML frontmatter and Markdown body. Preferred for create and edit."
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
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Text to find when action is patch. Must be unique unless replace_all is true."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text when action is patch. Use an empty string to delete matched content."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "When action is patch, replace all occurrences instead of requiring a unique match. Defaults to false."
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category directory used by create, for example 'devops' or 'research_tools'."
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Supporting file path used by patch, write_file, or remove_file. Must stay under references/, templates/, scripts/, or assets/."
                    },
                    "file_content": {
                        "type": "string",
                        "description": "Supporting file content used by write_file."
                    }
                },
                "required": ["action", "name"],
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
        let output = match args.action {
            SkillAction::Create => self.create_skill_action(args).await?,
            SkillAction::Patch => self.patch_skill_action(args).await?,
            SkillAction::Edit => self.edit_skill_action(args).await?,
            SkillAction::Delete => self.delete_skill_action(args).await?,
            SkillAction::WriteFile => self.write_skill_file_action(args).await?,
            SkillAction::RemoveFile => self.remove_skill_file_action(args).await?,
        };

        Ok(ToolOutput::new(output))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::{BaseCtx, SubAgentSet},
        engine::EngineBuilder,
    };
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
        assert_eq!(def.parameters["required"], json!(["action", "name"]));
        assert_eq!(
            def.parameters["properties"]["action"]["enum"],
            json!([
                "create",
                "patch",
                "edit",
                "delete",
                "write_file",
                "remove_file"
            ])
        );
        assert_eq!(def.parameters["properties"]["name"]["maxLength"], json!(64));
        assert_eq!(
            def.parameters["properties"]["description"]["maxLength"],
            json!(1024)
        );
    }

    fn skill_args(action: SkillAction, name: &str) -> SkillArgs {
        SkillArgs {
            action,
            name: name.to_string(),
            description: String::new(),
            instructions: String::new(),
            license: None,
            compatibility: None,
            metadata: BTreeMap::new(),
            allowed_tools: None,
            content: None,
            category: None,
            file_path: None,
            file_content: None,
            old_string: None,
            new_string: None,
            replace_all: false,
        }
    }

    // -- integration: load, create, edit --

    #[tokio::test]
    async fn load_create_and_edit_from_temp_dir() {
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
                "fetch"
            ]
        );
        assert!(beta.instructions.contains("Beta instructions."));

        let beta_skill = mgr.get_skill("skill_beta_skill").unwrap();
        assert_eq!(beta_skill.frontmatter.license.as_deref(), Some("MIT"));

        // Create a new skill.
        mgr.create_skill_action(SkillArgs {
            description: "Gamma skill for testing.".to_string(),
            instructions: "Gamma instructions.".to_string(),
            ..skill_args(SkillAction::Create, "gamma")
        })
        .await
        .unwrap();

        assert!(mgr.contains_lowercase("skill_gamma"));
        assert!(tmp.join("gamma/SKILL.md").exists());

        // Verify on-disk content is valid SKILL.md.
        let on_disk = tokio::fs::read_to_string(tmp.join("gamma/SKILL.md"))
            .await
            .unwrap();
        let reparsed = parse_skill_md(tmp.to_path_buf(), &on_disk).unwrap();
        assert_eq!(reparsed.frontmatter.name, "gamma");

        // Edit existing skill.
        mgr.edit_skill_action(SkillArgs {
            description: "Alpha v2 for testing.".to_string(),
            instructions: "Updated.".to_string(),
            allowed_tools: Some("shell fetch".to_string()),
            ..skill_args(SkillAction::Edit, "alpha")
        })
        .await
        .unwrap();

        let alpha2 = mgr.get_lowercase("skill_alpha").unwrap();
        assert_eq!(alpha2.description, "Alpha v2 for testing.");
        assert!(
            alpha2.instructions.contains(
            "You are a sub-agent specialised in \"alpha\" skill. Follow these instructions:\n\nUpdated.")
        );
        assert_eq!(
            alpha2.tools,
            vec![
                "shell",
                "read_file",
                "search_file",
                "write_file",
                "edit_file",
                "fetch"
            ]
        );

        // File should be at the original path.
        let on_disk = tokio::fs::read_to_string(tmp.join("alpha/SKILL.md"))
            .await
            .unwrap();
        assert!(on_disk.contains("Alpha v2 for testing."));

        // Definitions.
        let defs = mgr.definitions(None);
        assert_eq!(defs.len(), 3);

        let defs_filtered = mgr.definitions(Some(&["skill_gamma".to_string()]));
        assert_eq!(defs_filtered.len(), 1);
        assert_eq!(defs_filtered[0].name, "skill_gamma");

        // Clean up.
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn skill_manager_actions_manage_skill_lifecycle() {
        let tmp = std::env::temp_dir().join(format!(
            "anda-skills-actions-{:016x}",
            rand::random::<u64>()
        ));
        let mgr = SkillManager::new(tmp.clone());

        let created = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "create",
                    "name": "delta",
                    "category": "ops",
                    "content": skill_md("delta", "Delta skill.", "Initial instructions.", Some("shell fetch"))
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(created.output["action"], json!("create"));
        assert_eq!(created.output["path"], json!("ops/delta/SKILL.md"));
        assert!(tmp.join("ops/delta/SKILL.md").exists());
        assert!(mgr.contains_lowercase("skill_delta"));

        let wrote = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "write_file",
                    "name": "delta",
                    "file_path": "references/guide.md",
                    "file_content": "before middle before"
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(wrote.output["action"], json!("write_file"));
        assert_eq!(
            tokio::fs::read_to_string(tmp.join("ops/delta/references/guide.md"))
                .await
                .unwrap(),
            "before middle before"
        );

        let patched_file = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "patch",
                    "name": "delta",
                    "file_path": "references/guide.md",
                    "old_string": "before",
                    "new_string": "after",
                    "replace_all": true
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(patched_file.output["replacements"], json!(2));
        assert_eq!(patched_file.output["total_matches"], json!(2));
        assert_eq!(
            tokio::fs::read_to_string(tmp.join("ops/delta/references/guide.md"))
                .await
                .unwrap(),
            "after middle after"
        );

        let patched_skill = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "patch",
                    "name": "delta",
                    "old_string": "Initial instructions.",
                    "new_string": "Updated instructions."
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(patched_skill.output["action"], json!("patch"));
        let delta = mgr.get_lowercase("skill_delta").unwrap();
        assert!(delta.instructions.contains("Updated instructions."));

        let edited = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "edit",
                    "name": "delta",
                    "content": skill_md("delta", "Delta edited.", "Edited body.", Some("shell fetch search_file"))
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(edited.output["action"], json!("edit"));
        let delta = mgr.get_lowercase("skill_delta").unwrap();
        assert_eq!(delta.description, "Delta edited.");
        assert_eq!(
            delta.tools,
            vec![
                "shell",
                "read_file",
                "search_file",
                "write_file",
                "edit_file",
                "fetch"
            ]
        );
        assert!(delta.instructions.contains("Edited body."));

        let removed = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "remove_file",
                    "name": "delta",
                    "file_path": "references/guide.md"
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(removed.output["action"], json!("remove_file"));
        assert!(!tmp.join("ops/delta/references/guide.md").exists());
        assert!(!tmp.join("ops/delta/references").exists());

        let deleted = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "delete",
                    "name": "delta"
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(deleted.output["action"], json!("delete"));
        assert!(!tmp.join("ops/delta").exists());
        assert!(!tmp.join("ops").exists());
        assert!(!mgr.contains_lowercase("skill_delta"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn patch_requires_unique_match_unless_replace_all() {
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-patch-{:016x}", rand::random::<u64>()));
        let mgr = SkillManager::new(tmp.clone());

        mgr.call_raw(
            mock_ctx(),
            json!({
                "action": "create",
                "name": "echo",
                "content": skill_md("echo", "Echo skill.", "repeat\nrepeat\n", None)
            }),
            Vec::new(),
        )
        .await
        .unwrap();

        let err = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "patch",
                    "name": "echo",
                    "old_string": "repeat",
                    "new_string": "done"
                }),
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("old_string matched 2 times; set replace_all=true")
        );

        let ok = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "patch",
                    "name": "echo",
                    "old_string": "repeat",
                    "new_string": "done",
                    "replace_all": true
                }),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(ok.output["replacements"], json!(2));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn write_file_rejects_invalid_supporting_paths() {
        let tmp = std::env::temp_dir().join(format!(
            "anda-skills-invalid-path-{:016x}",
            rand::random::<u64>()
        ));
        let mgr = SkillManager::new(tmp.clone());

        mgr.call_raw(
            mock_ctx(),
            json!({
                "action": "create",
                "name": "foxtrot",
                "content": skill_md("foxtrot", "Foxtrot skill.", "Body.", None)
            }),
            Vec::new(),
        )
        .await
        .unwrap();

        let err = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "action": "write_file",
                    "name": "foxtrot",
                    "file_path": "../escape.md",
                    "file_content": "x"
                }),
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must not contain '..' segments"));
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
        assert!(!mgr.contains_lowercase("skill_correct_name"));

        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[test]
    fn create_validates_description_length() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-val-{:016x}", rand::random::<u64>()));

        let mgr = SkillManager::new(tmp.clone());

        // Empty description.
        let err = rt.block_on(mgr.create_skill_action(SkillArgs {
            description: "".to_string(),
            instructions: "body".to_string(),
            ..skill_args(SkillAction::Create, "test")
        }));
        assert!(err.is_err());

        // Too-long description.
        let err = rt.block_on(mgr.create_skill_action(SkillArgs {
            description: "x".repeat(1025),
            instructions: "body".to_string(),
            ..skill_args(SkillAction::Create, "test")
        }));
        assert!(err.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tool_requires_action() {
        let tmp = std::env::temp_dir().join(format!(
            "anda-skills-requires-action-{:016x}",
            rand::random::<u64>()
        ));
        let mgr = SkillManager::new(tmp.clone());

        let err = mgr
            .call_raw(
                mock_ctx(),
                json!({
                    "name": "golf",
                    "content": skill_md("golf", "Golf skill.", "Body.", None)
                }),
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("missing field `action`"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sub_agents_manager_register_skills_manager() {
        let tmp =
            std::env::temp_dir().join(format!("anda-skills-val-{:016x}", rand::random::<u64>()));
        let tool = SkillManager::new(tmp.clone());
        let engine = EngineBuilder::new().empty();
        assert!(engine.sub_agents_manager().insert(Arc::new(tool)).is_none());
    }
}
