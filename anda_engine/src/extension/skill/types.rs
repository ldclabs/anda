//! Skill frontmatter and parsed skill types.
//!
//! These structs implement the local Agent Skills file format used by the skill
//! extension. They validate frontmatter, normalize names, and convert a skill
//! directory into a runnable subagent definition.

use anda_core::{BoxError, Json, validate_function_name};
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::{extension::fs::decode_file_text, subagent::SubAgent};

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// How a skill is executed once the agent decides to use it.
///
/// The Agent Skills specification is built on progressive disclosure: only the skill's name and
/// description stay resident, and the SKILL.md body is pulled into the *calling* agent's context
/// when it becomes relevant. [`SkillExecution::Inline`] implements that model and is the default.
///
/// [`SkillExecution::Subagent`] is an opt-in escape hatch for skills that are genuinely
/// independent units of work — long-running, parallelisable, or context-hungry procedures that do
/// not need the live conversation. Delegating trades context fidelity for isolation: a subagent
/// receives only a self-contained prompt plus the resources matching its tags, and it cannot ask
/// the user anything.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillExecution {
    /// Read SKILL.md into the calling agent's own context and follow it there.
    #[default]
    Inline,
    /// Expose the skill as an isolated subagent worker callable as `SA_<agent_name>`.
    Subagent,
}

impl SkillExecution {
    /// Returns the lowercase wire name of this mode.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Subagent => "subagent",
        }
    }

    /// Whether the skill should be materialized as a callable subagent.
    pub const fn is_subagent(&self) -> bool {
        matches!(self, Self::Subagent)
    }
}

impl fmt::Display for SkillExecution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SkillExecution {
    type Err = BoxError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "inline" => Ok(Self::Inline),
            "subagent" => Ok(Self::Subagent),
            other => Err(format!(
                "unknown skill execution mode {other:?}, expected \"inline\" or \"subagent\""
            )
            .into()),
        }
    }
}

// ---------------------------------------------------------------------------
// SKILL.md frontmatter — Agent Skills specification
// ---------------------------------------------------------------------------

/// YAML frontmatter of a `SKILL.md` file, following the Agent Skills spec.
///
/// See <https://agentskills.io/specification> for the full field reference.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SkillFrontmatter {
    /// Skill name: 1-64 lowercase alphanumeric + hyphens, no leading/trailing/
    /// consecutive hyphens. Must match parent directory name.
    pub name: String,

    /// What the skill does and when to use it (1-1024 characters).
    pub description: String,

    /// Optional license name or reference to a bundled license file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Optional environment requirements (1-500 characters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<String>,

    /// Arbitrary key-value metadata.
    ///
    /// `metadata.execution` is honoured as a spec-conformant alternative to the top-level
    /// `execution` field.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,

    /// Space-delimited list of tools the skill may use.
    ///
    /// This is an upper bound, not an addition: a skill that declares `allowed-tools` is granted
    /// exactly those tools when run as a subagent. Skills that declare nothing inherit the
    /// manager's default tool set.
    #[serde(
        default,
        alias = "allowed_tools",
        rename = "allowed-tools",
        deserialize_with = "deserialize_optional_token_list",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_tools: Option<String>,

    /// How this skill runs. Defaults to [`SkillExecution::Inline`] when absent.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_execution",
        skip_serializing_if = "Option::is_none"
    )]
    pub execution: Option<SkillExecution>,

    /// Space-delimited resource tags this skill consumes when run as a subagent.
    ///
    /// Only meaningful for [`SkillExecution::Subagent`]. `*` takes every resource offered by the
    /// caller, which is the default when the field is absent; a narrower list lets sibling
    /// callables in the same turn keep the resources they need.
    #[serde(
        default,
        alias = "resource_tags",
        rename = "resource-tags",
        deserialize_with = "deserialize_optional_token_list",
        skip_serializing_if = "Option::is_none"
    )]
    pub resource_tags: Option<String>,

    #[serde(flatten)]
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    /// Unrecognized frontmatter fields preserved for round-tripping and callers.
    pub extra: BTreeMap<String, Json>,
}

/// A fully-parsed skill: frontmatter + body + derived fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Parsed YAML frontmatter.
    pub frontmatter: SkillFrontmatter,
    /// Markdown body (the agent instructions).
    pub instructions: String,
    /// Normalised snake_case name derived from `frontmatter.name`.
    pub agent_name: String,
    /// Resolved execution mode (from `execution`, `metadata.execution`, or the default).
    pub execution: SkillExecution,
    /// Tools declared by `allowed-tools`, the complete allowlist for this skill. Empty when the
    /// skill declared an empty list, and also when it declared nothing — see
    /// [`Self::declares_tools`], which is what decides whether the manager's defaults apply.
    pub tools: Vec<String>,
    /// Resource tags declared by `resource-tags`. Empty when the skill declared an empty list, and
    /// also when it declared nothing — see [`Self::declares_resource_tags`].
    pub tags: Vec<String>,
    /// Skill directory path (parent of SKILL.md) for resolving relative resources.
    pub base_dir: PathBuf,
}

impl Skill {
    /// Whether this skill is exposed as a callable subagent.
    pub fn is_subagent(&self) -> bool {
        self.execution.is_subagent()
    }

    /// Whether the skill declared `allowed-tools` at all.
    ///
    /// Distinct from a non-empty [`Self::tools`]: `allowed-tools: []` is a skill asking for *no*
    /// tools, and must not fall through to the manager's default grant.
    pub fn declares_tools(&self) -> bool {
        self.frontmatter.allowed_tools.is_some()
    }

    /// Whether the skill declared `resource-tags` at all.
    ///
    /// Distinct from a non-empty [`Self::tags`]: `resource-tags: []` is a skill asking for no
    /// resources, and must not fall through to the `*` default.
    pub fn declares_resource_tags(&self) -> bool {
        self.frontmatter.resource_tags.is_some()
    }
}

// ---------------------------------------------------------------------------
// Name validation & normalisation
// ---------------------------------------------------------------------------

/// Validate a skill name per the Agent Skills specification.
///
/// Rules:
/// - 1-64 characters
/// - Lowercase alphanumeric (`a-z`, `0-9`) and hyphens (`-`) only
/// - Must not start or end with a hyphen
/// - Must not contain consecutive hyphens (`--`)
pub fn validate_skill_name(name: &str) -> Result<(), BoxError> {
    if name.is_empty() {
        return Err("skill name must not be empty".into());
    }
    if name.len() > 64 {
        return Err("skill name must not exceed 64 characters".into());
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err("skill name must not start or end with a hyphen".into());
    }
    if name.contains("--") {
        return Err("skill name must not contain consecutive hyphens".into());
    }
    for c in name.chars() {
        if !matches!(c, 'a'..='z' | '0'..='9' | '-') {
            return Err(format!("skill name contains invalid character: {c}").into());
        }
    }
    Ok(())
}

/// Normalise a kebab-case skill name to snake_case for the [`SubAgent`] registry.
pub fn normalise_skill_agent_name(name: &str) -> String {
    format!(
        "skill_{}",
        name.trim().to_ascii_lowercase().replace('-', "_")
    )
}

// ---------------------------------------------------------------------------
// SKILL.md parsing & formatting
// ---------------------------------------------------------------------------

/// Accepts either a space/comma-delimited string or a YAML list of strings, normalising both to
/// one space-delimited string. Used by `allowed-tools` and `resource-tags`.
///
/// An explicitly empty declaration (`[]` or `""`) stays `Some("")` rather than collapsing to
/// `None`: both fields widen the grant when they are absent — tools fall back to the manager's
/// defaults, resource tags to `*` — so a manifest that asks for nothing must not be read as a
/// manifest that asked for nothing *in particular*. Only a missing or null key means "undeclared".
fn deserialize_optional_token_list<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<Value>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.trim().to_string())),
        Value::Array(items) => {
            let mut tokens = Vec::new();
            for item in items {
                match item {
                    Value::String(s) => {
                        let s = s.trim();
                        if !s.is_empty() {
                            tokens.push(s.to_string());
                        }
                    }
                    other => {
                        return Err(de::Error::custom(format!(
                            "list entries must be strings, got {other}"
                        )));
                    }
                }
            }
            Ok(Some(tokens.join(" ")))
        }
        other => Err(de::Error::custom(format!(
            "expected a string or a list of strings, got {other}"
        ))),
    }
}

fn deserialize_optional_execution<'de, D>(
    deserializer: D,
) -> Result<Option<SkillExecution>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<String>::deserialize(deserializer)? else {
        return Ok(None);
    };

    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }

    SkillExecution::from_str(value)
        .map(Some)
        .map_err(de::Error::custom)
}

/// Splits a space/comma-delimited token list into its entries, dropping duplicates while
/// preserving the declared order.
fn split_tokens(value: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for token in value.split(|c: char| c.is_whitespace() || c == ',') {
        if !token.is_empty() && !tokens.iter().any(|seen| seen == token) {
            tokens.push(token.to_string());
        }
    }
    tokens
}

/// Resolves the execution mode, preferring the top-level `execution` field and falling back to
/// `metadata.execution` — the spec-sanctioned place for custom keys.
fn resolve_execution(fm: &SkillFrontmatter) -> Result<SkillExecution, BoxError> {
    if let Some(execution) = fm.execution {
        return Ok(execution);
    }

    match fm.metadata.get("execution") {
        None => Ok(SkillExecution::default()),
        Some(Json::String(value)) => SkillExecution::from_str(value),
        Some(other) => {
            Err(format!("SKILL.md metadata.execution must be a string, got {other}").into())
        }
    }
}

/// Split a `SKILL.md` into YAML frontmatter string and Markdown body.
fn split_frontmatter(content: &str) -> Result<(&str, &str), BoxError> {
    let content = content.trim();
    if !content.starts_with("---") {
        return Err("SKILL.md must start with YAML frontmatter (---)".into());
    }
    let after_open = &content[3..];
    let close_pos = after_open
        .find("\n---")
        .ok_or("missing closing --- in YAML frontmatter")?;

    let frontmatter = &after_open[..close_pos];
    let body_start = 3 + close_pos + 4; // skip opening "---", frontmatter, "\n---"
    let body = if body_start < content.len() {
        content[body_start..].trim()
    } else {
        ""
    };
    Ok((frontmatter, body))
}

fn parse_skill_frontmatter(yaml_str: &str) -> Result<SkillFrontmatter, BoxError> {
    match serde_saphyr::from_str(yaml_str) {
        Ok(fm) => Ok(fm),
        Err(strict_err) => {
            let relaxed = relax_common_frontmatter_scalars(yaml_str);
            if relaxed == yaml_str {
                return Err(format!("invalid SKILL.md frontmatter: {strict_err}").into());
            }

            serde_saphyr::from_str(&relaxed).map_err(|relaxed_err| {
                format!(
                    "invalid SKILL.md frontmatter: {strict_err}; relaxed parser also failed: {relaxed_err}"
                )
                .into()
            })
        }
    }
}

fn relax_common_frontmatter_scalars(yaml_str: &str) -> String {
    let mut out = String::with_capacity(yaml_str.len());
    for line in yaml_str.lines() {
        if let Some((key, value)) = split_relaxable_top_level_scalar(line) {
            out.push_str(key);
            out.push_str(": |-\n");
            out.push_str("  ");
            out.push_str(value.trim());
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    if !yaml_str.ends_with('\n') {
        out.pop();
    }
    out
}

fn split_relaxable_top_level_scalar(line: &str) -> Option<(&str, &str)> {
    if line.starts_with([' ', '\t']) {
        return None;
    }

    let (key, value) = line.split_once(':')?;
    if !matches!(
        key,
        "name" | "description" | "license" | "compatibility" | "allowed-tools" | "allowed_tools"
    ) {
        return None;
    }

    let value = value.trim_start();
    if value.is_empty() || value.starts_with(['"', '\'', '|', '>', '[', '{']) {
        return None;
    }

    Some((key, value))
}

/// Parse a `SKILL.md` file content into a [`Skill`].
pub fn parse_skill_md(base_dir: PathBuf, content: &str) -> Result<Skill, BoxError> {
    let (yaml_str, body) = split_frontmatter(content)?;

    let fm = parse_skill_frontmatter(yaml_str)?;

    // Validate required fields.
    validate_skill_name(&fm.name)?;

    if fm.description.is_empty() {
        return Err("SKILL.md frontmatter missing required field: description".into());
    }
    if fm.description.len() > 1024 {
        return Err("SKILL.md description must not exceed 1024 characters".into());
    }
    if let Some(compat) = &fm.compatibility
        && (compat.is_empty() || compat.len() > 500)
    {
        return Err("SKILL.md compatibility must be 1-500 characters".into());
    }

    // Derive agent name (snake_case).
    let agent_name = normalise_skill_agent_name(&fm.name);
    validate_function_name(&agent_name)?;

    let execution = resolve_execution(&fm)?;

    // An empty list means "not declared": the manager falls back to its default tool set.
    let tools = match &fm.allowed_tools {
        Some(at) => split_tokens(at),
        None => Vec::new(),
    };

    // An empty list means "not declared": the subagent accepts every offered resource.
    let tags = match &fm.resource_tags {
        Some(tags) => split_tokens(tags),
        None => Vec::new(),
    };

    Ok(Skill {
        frontmatter: fm,
        instructions: body.to_string(),
        agent_name,
        execution,
        tools,
        tags,
        base_dir,
    })
}

/// Convert a [`Skill`] into a [`SubAgent`].
///
/// Only meaningful for [`SkillExecution::Subagent`] skills; inline skills are never materialized
/// as callables. Tools are taken verbatim from the skill — [`crate::extension::skill::SkillManager`]
/// substitutes its defaults when the skill declares none.
///
/// A skill that declares no `resource-tags` gets `*`, so a delegated skill receives the resources
/// the caller is holding. Without it a skill subagent can never see the current turn's
/// attachments, since [`anda_core::select_resources`] returns nothing for an empty tag list. A
/// skill that declares an empty list means the opposite — take nothing — so it keeps its empty
/// tags.
impl From<&Skill> for SubAgent {
    fn from(skill: &Skill) -> Self {
        let tags = if skill.declares_resource_tags() {
            skill.tags.clone()
        } else {
            vec!["*".to_string()]
        };

        SubAgent {
            name: skill.agent_name.clone(),
            description: skill.frontmatter.description.clone(),
            instructions: format!(
                "Base directory for this skill: {}\n\nYou are a subagent specialised in {:?} skill. Follow these instructions:\n\n{}",
                skill.base_dir.display(),
                skill.frontmatter.name,
                skill.instructions,
            ),
            tools: skill.tools.clone(),
            tags,
            ..Default::default()
        }
    }
}

/// Format a [`Skill`] back into SKILL.md content.
pub fn format_skill_md(skill: &Skill) -> Result<String, BoxError> {
    let yaml = serde_saphyr::to_string(&skill.frontmatter)
        .map_err(|e| format!("failed to serialize SKILL.md frontmatter: {e}"))?;

    let mut buf = String::new();
    buf.push_str("---\n");
    buf.push_str(yaml.trim());
    buf.push_str("\n---\n\n");
    buf.push_str(&skill.instructions);
    if !skill.instructions.ends_with('\n') {
        buf.push('\n');
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Directory loading
// ---------------------------------------------------------------------------

/// Recursively find all `SKILL.md` files under `dir`.
pub async fn find_skill_files(dir: &Path) -> Result<Vec<PathBuf>, BoxError> {
    let mut result = Vec::new();
    let mut stack = vec![dir.to_path_buf()];

    while let Some(current) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&current).await?;
        while let Some(entry) = entries.next_entry().await? {
            let ft = entry.file_type().await?;
            let path = entry.path();
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() && entry.file_name() == "SKILL.md" {
                result.push(path);
            }
        }
    }

    Ok(result)
}

/// Load all skills from a directory tree.
///
/// Returns `skills` where `skills` maps normalised agent name -> [`Skill`].
pub async fn load_skills_from_dir(dir: &Path) -> Result<BTreeMap<String, Skill>, BoxError> {
    let files = find_skill_files(dir).await?;
    let mut skills = BTreeMap::new();

    for path in files {
        let Some(base_dir) = path.parent() else {
            continue;
        };
        match read_skill_md_text(&path).await {
            Ok(content) => match parse_skill_md(base_dir.to_path_buf(), &content) {
                Ok(skill) => {
                    if skills.contains_key(&skill.agent_name) {
                        log::warn!(
                            "duplicate skill name {} at {}, skipping",
                            skill.agent_name,
                            path.display()
                        );
                    } else {
                        skills.insert(skill.agent_name.clone(), skill);
                    }
                }
                Err(err) => log::error!("skipping {}: {err}", path.display()),
            },
            Err(err) => {
                log::error!("skipping {}: {err}", path.display());
            }
        }
    }

    Ok(skills)
}

async fn read_skill_md_text(path: &Path) -> Result<String, BoxError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|err| format!("Failed to read skill file {}: {err}", path.display()))?;
    decode_skill_md_bytes(bytes).map_err(|_| {
        format!(
            "Only UTF-8 or supported text-encoded skill files are supported (path: {})",
            path.display()
        )
        .into()
    })
}

pub(crate) fn decode_skill_md_bytes(bytes: Vec<u8>) -> Result<String, Vec<u8>> {
    decode_file_text(bytes).map(|decoded| decoded.text)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::{Agent, Resource, select_resources};

    // -- name validation --

    #[test]
    fn validate_skill_name_accepts_valid() {
        assert!(validate_skill_name("pdf-processing").is_ok());
        assert!(validate_skill_name("data-analysis").is_ok());
        assert!(validate_skill_name("code-review").is_ok());
        assert!(validate_skill_name("simple").is_ok());
        assert!(validate_skill_name("a1b2").is_ok());
    }

    #[test]
    fn validate_skill_name_rejects_invalid() {
        assert!(validate_skill_name("").is_err()); // empty
        assert!(validate_skill_name("-pdf").is_err()); // leading hyphen
        assert!(validate_skill_name("pdf-").is_err()); // trailing hyphen
        assert!(validate_skill_name("pdf--x").is_err()); // consecutive hyphens
        assert!(validate_skill_name("PDF").is_err()); // uppercase
        assert!(validate_skill_name("my_skill").is_err()); // underscore
        assert!(validate_skill_name(&"a".repeat(65)).is_err()); // too long
    }

    #[test]
    fn normalise_name_converts_kebab_to_snake() {
        assert_eq!(normalise_skill_agent_name("my-skill"), "skill_my_skill");
        assert_eq!(
            normalise_skill_agent_name("pdf-processing"),
            "skill_pdf_processing"
        );
        assert_eq!(normalise_skill_agent_name("simple"), "skill_simple");
    }

    // -- frontmatter parsing --

    #[test]
    fn parse_minimal_skill_md() {
        let md = "\
---
name: my-skill
description: A test skill.
---

# Instructions

Do something useful.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.frontmatter.name, "my-skill");
        assert_eq!(skill.agent_name, "skill_my_skill");
        assert_eq!(skill.frontmatter.description, "A test skill.");
        assert!(skill.tools.is_empty());
        assert_eq!(skill.base_dir, PathBuf::from("/test_dir"));
        assert!(skill.instructions.contains("# Instructions"));
        assert!(skill.instructions.contains("Do something useful."));
        assert!(skill.frontmatter.license.is_none());
        assert!(skill.frontmatter.compatibility.is_none());
        assert!(skill.frontmatter.metadata.is_empty());
        assert!(skill.frontmatter.allowed_tools.is_none());
    }

    #[test]
    fn parse_full_spec_skill_md() {
        let md = "\
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
license: Apache-2.0
compatibility: Requires Python 3.14+ and uv
metadata:
  author: example-org
  version: '1.0'
allowed-tools: shell google_web_search
---

# PDF Processing

Process PDFs here.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.frontmatter.name, "pdf-processing");
        assert_eq!(skill.agent_name, "skill_pdf_processing");
        assert_eq!(skill.frontmatter.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(
            skill.frontmatter.compatibility.as_deref(),
            Some("Requires Python 3.14+ and uv")
        );
        assert_eq!(
            skill
                .frontmatter
                .metadata
                .get("author")
                .and_then(|v| v.as_str()),
            Some("example-org")
        );
        assert_eq!(
            skill
                .frontmatter
                .metadata
                .get("version")
                .and_then(|v| v.as_str()),
            Some("1.0")
        );
        assert_eq!(
            skill.tools,
            vec!["shell".to_string(), "google_web_search".to_string()]
        );
        assert!(skill.instructions.contains("# PDF Processing"));

        let md = "\
---
name: gif-search
description: \"Search/download GIFs from Tenor via curl + jq.\"
version: 1.1.0
author: Hermes Agent
license: MIT
prerequisites:
  env_vars: [TENOR_API_KEY]
  commands: [curl, jq]
metadata:
  tags: [GIF, Media, Search, Tenor, API]
---

# GIF Search (Tenor API)

Search and download GIFs directly via the Tenor API using curl. No extra tools needed.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.frontmatter.name, "gif-search");
        assert_eq!(
            skill
                .frontmatter
                .metadata
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|tags| tags.len()),
            Some(5)
        );
        assert_eq!(
            skill
                .frontmatter
                .extra
                .get("author")
                .and_then(|v| v.as_str()),
            Some("Hermes Agent")
        );
        assert!(skill.frontmatter.extra.contains_key("version"));
        assert!(skill.frontmatter.extra.contains_key("prerequisites"));
    }

    #[test]
    fn parse_skill_md_relaxes_unquoted_description_with_colon() {
        let md = "\
---
name: baoyu-design
description: Create polished design artifacts. It drives a full design process: clarifying questions, design-context gathering, and production of HTML deliverables.
---

# Design

Follow the design process.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.frontmatter.name, "baoyu-design");
        assert!(
            skill
                .frontmatter
                .description
                .contains("process: clarifying questions")
        );
        assert!(skill.instructions.contains("# Design"));
    }

    #[test]
    fn parse_skill_md_accepts_allowed_tools_alias_and_list() {
        let md = "\
---
name: list-tools
description: A skill with tools as a YAML list.
allowed_tools:
  - shell
  - read_file
---

Use the listed tools.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(
            skill.frontmatter.allowed_tools.as_deref(),
            Some("shell read_file")
        );
        assert_eq!(
            skill.tools,
            vec!["shell".to_string(), "read_file".to_string()]
        );

        let md = "\
---
name: comma-tools
description: A skill with comma-separated tools.
allowed-tools: shell, read_file
---

Use the listed tools.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(
            skill.tools,
            vec!["shell".to_string(), "read_file".to_string()]
        );
    }

    #[test]
    fn an_empty_declaration_is_not_an_undeclared_one() {
        // Both fields widen the grant when absent, so asking for nothing must stay a request for
        // nothing rather than falling through to the defaults.
        let md = "\
---
name: locked-down
description: Wants no tools and no resources.
execution: subagent
allowed-tools: []
resource-tags: []
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert!(skill.tools.is_empty());
        assert!(skill.declares_tools());
        assert!(skill.tags.is_empty());
        assert!(skill.declares_resource_tags());
        assert!(SubAgent::from(&skill).supported_resource_tags().is_empty());

        let md = "\
---
name: undeclared
description: Declares neither field.
execution: subagent
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert!(!skill.declares_tools());
        assert!(!skill.declares_resource_tags());
        assert_eq!(
            SubAgent::from(&skill).supported_resource_tags(),
            vec!["*".to_string()]
        );
    }

    #[test]
    fn parse_skill_md_missing_name_errors() {
        let md = "\
---
description: No name here.
---

Body.
";
        assert!(parse_skill_md(PathBuf::from("/test_dir"), md).is_err());
    }

    #[test]
    fn parse_skill_md_missing_frontmatter_errors() {
        let md = "# Just markdown, no frontmatter.";
        assert!(parse_skill_md(PathBuf::from("/test_dir"), md).is_err());
    }

    #[test]
    fn parse_skill_md_invalid_name_errors() {
        let md = "\
---
name: PDF-Processing
description: Invalid name.
---

Body.
";
        assert!(parse_skill_md(PathBuf::from("/test_dir"), md).is_err());
    }

    #[test]
    fn parse_skill_md_description_too_long_errors() {
        let long_desc = "x".repeat(1025);
        let md = format!(
            "---\nname: test\ndescription: {}\n---\n\nBody.\n",
            long_desc
        );
        assert!(parse_skill_md(PathBuf::from("/test_dir"), &md).is_err());
    }

    // -- format roundtrip --

    #[test]
    fn format_and_reparse_roundtrip() {
        let skill = Skill {
            base_dir: PathBuf::from("/test_dir"),
            frontmatter: SkillFrontmatter {
                name: "my-skill".to_string(),
                description: "A test skill.".to_string(),
                license: Some("MIT".to_string()),
                metadata: BTreeMap::from([("author".to_string(), "test".into())]),
                allowed_tools: Some("shell fetch".to_string()),
                execution: Some(SkillExecution::Subagent),
                resource_tags: Some("text image".to_string()),
                ..Default::default()
            },
            instructions: "# Instructions\n\nDo something useful.".to_string(),
            agent_name: "skill_my_skill".to_string(),
            execution: SkillExecution::Subagent,
            tools: vec!["shell".to_string(), "fetch".to_string()],
            tags: vec!["text".to_string(), "image".to_string()],
        };

        let md = format_skill_md(&skill).unwrap();
        let parsed = parse_skill_md(PathBuf::from("/test_dir"), &md).unwrap();
        assert_eq!(parsed.frontmatter.name, skill.frontmatter.name);
        assert_eq!(
            parsed.frontmatter.description,
            skill.frontmatter.description
        );
        assert_eq!(parsed.frontmatter.license, skill.frontmatter.license);
        assert_eq!(
            parsed.frontmatter.metadata.get("author"),
            skill.frontmatter.metadata.get("author")
        );
        assert_eq!(parsed.tools, skill.tools);
        assert_eq!(parsed.tags, skill.tags);
        assert_eq!(parsed.execution, SkillExecution::Subagent);
        assert_eq!(parsed.instructions, skill.instructions);
    }

    // -- execution mode --

    #[test]
    fn skills_are_inline_unless_they_opt_in() {
        let md = "\
---
name: inline-by-default
description: No execution field.
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.execution, SkillExecution::Inline);
        assert!(!skill.is_subagent());

        let md = "\
---
name: opted-in
description: Declares subagent execution.
execution: subagent
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.execution, SkillExecution::Subagent);
        assert!(skill.is_subagent());

        // `metadata` is the spec-sanctioned place for custom keys, so it is honoured too.
        let md = "\
---
name: opted-in-via-metadata
description: Declares subagent execution in metadata.
metadata:
  execution: subagent
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert!(skill.is_subagent());

        // The top-level field wins over metadata.
        let md = "\
---
name: conflicting
description: Top-level field wins.
execution: inline
metadata:
  execution: subagent
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.execution, SkillExecution::Inline);
    }

    #[test]
    fn invalid_execution_mode_errors() {
        let md = "\
---
name: bad-mode
description: Unknown execution mode.
execution: background
---

Body.
";
        let err = parse_skill_md(PathBuf::from("/test_dir"), md)
            .unwrap_err()
            .to_string();
        assert!(err.contains("background"), "{err}");

        let md = "\
---
name: bad-metadata-mode
description: Non-string metadata execution.
metadata:
  execution: 42
---

Body.
";
        let err = parse_skill_md(PathBuf::from("/test_dir"), md)
            .unwrap_err()
            .to_string();
        assert!(err.contains("metadata.execution must be a string"), "{err}");
    }

    #[test]
    fn parse_skill_md_accepts_resource_tags_as_string_or_list() {
        let md = "\
---
name: tagged
description: Declares the resources it consumes.
execution: subagent
resource-tags: text, image
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.tags, vec!["text".to_string(), "image".to_string()]);

        let md = "\
---
name: tagged-list
description: Declares the resources it consumes as a list.
execution: subagent
resource_tags:
  - text
  - audio
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        assert_eq!(skill.tags, vec!["text".to_string(), "audio".to_string()]);
    }

    // -- SubAgent conversion --

    #[test]
    fn subagent_from_skill_has_correct_definition() {
        let md = "\
---
name: research
description: Research things thoroughly.
execution: subagent
allowed-tools: shell google_web_search
---

Research instructions here.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        let agent = SubAgent::from(&skill);
        assert_eq!(agent.name, "skill_research");
        assert_eq!(agent.description, "Research things thoroughly.");
        assert_eq!(agent.tools, vec!["shell", "google_web_search"]);
        assert_eq!(
            agent.instructions,
            "Base directory for this skill: /test_dir\n\nYou are a subagent specialised in \"research\" skill. Follow these instructions:\n\nResearch instructions here."
        );

        let def = agent.definition();
        assert_eq!(def.name, "skill_research");
    }

    #[test]
    fn subagent_from_skill_accepts_resources_by_default() {
        let md = "\
---
name: untagged
description: Declares no resource tags.
execution: subagent
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        let agent = SubAgent::from(&skill);
        // Without this an empty tag list makes `select_resources` return nothing, so a delegated
        // skill could never see the current turn's attachments.
        assert_eq!(agent.supported_resource_tags(), vec!["*".to_string()]);

        let mut resources = vec![
            Resource {
                _id: 1,
                name: "doc".to_string(),
                tags: vec!["text".to_string()],
                ..Default::default()
            },
            Resource {
                _id: 2,
                name: "pic".to_string(),
                tags: vec!["image".to_string()],
                ..Default::default()
            },
        ];
        let selected = select_resources(&mut resources, &agent.supported_resource_tags());
        assert_eq!(selected.len(), 2);
        assert!(resources.is_empty());

        let md = "\
---
name: narrowed
description: Declares the resource tags it consumes.
execution: subagent
resource-tags: image
---

Body.
";
        let skill = parse_skill_md(PathBuf::from("/test_dir"), md).unwrap();
        let agent = SubAgent::from(&skill);
        assert_eq!(agent.supported_resource_tags(), vec!["image".to_string()]);

        let mut resources = vec![
            Resource {
                _id: 1,
                name: "doc".to_string(),
                tags: vec!["text".to_string()],
                ..Default::default()
            },
            Resource {
                _id: 2,
                name: "pic".to_string(),
                tags: vec!["image".to_string()],
                ..Default::default()
            },
        ];
        let selected = select_resources(&mut resources, &agent.supported_resource_tags());
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "pic");
        // Sibling callables in the same turn keep what the skill did not claim.
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].name, "doc");
    }
}
