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
    path::{Path, PathBuf},
};

use crate::{extension::fs::decode_file_text, subagent::SubAgent};

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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,

    /// Space-delimited list of pre-approved tools the skill may use.
    #[serde(
        default,
        alias = "allowed_tools",
        rename = "allowed-tools",
        deserialize_with = "deserialize_optional_tools",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_tools: Option<String>,

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
    /// Resolved tools list (from `allowed-tools` or default).
    pub tools: Vec<String>,
    /// Skill directory path (parent of SKILL.md) for resolving relative resources.
    pub base_dir: PathBuf,
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

fn deserialize_optional_tools<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<Value>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::String(s) => {
            let s = s.trim().to_string();
            if s.is_empty() { Ok(None) } else { Ok(Some(s)) }
        }
        Value::Array(items) => {
            let mut tools = Vec::new();
            for item in items {
                match item {
                    Value::String(s) => {
                        let s = s.trim();
                        if !s.is_empty() {
                            tools.push(s.to_string());
                        }
                    }
                    other => {
                        return Err(de::Error::custom(format!(
                            "allowed-tools entries must be strings, got {other}"
                        )));
                    }
                }
            }
            if tools.is_empty() {
                Ok(None)
            } else {
                Ok(Some(tools.join(" ")))
            }
        }
        other => Err(de::Error::custom(format!(
            "allowed-tools must be a string or a list of strings, got {other}"
        ))),
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

    // Parse allowed-tools (space-delimited) or use defaults.
    let tools = match &fm.allowed_tools {
        Some(at) if !at.trim().is_empty() => at
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect(),
        _ => Vec::new(),
    };

    Ok(Skill {
        frontmatter: fm,
        instructions: body.to_string(),
        agent_name,
        tools,
        base_dir,
    })
}

/// Convert a [`Skill`] into a [`SubAgent`].
impl From<&Skill> for SubAgent {
    fn from(skill: &Skill) -> Self {
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
    use anda_core::Agent;

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
                ..Default::default()
            },
            instructions: "# Instructions\n\nDo something useful.".to_string(),
            agent_name: "skill_my_skill".to_string(),
            tools: vec!["shell".to_string(), "fetch".to_string()],
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
        assert_eq!(parsed.instructions, skill.instructions);
    }

    // -- SubAgent conversion --

    #[test]
    fn subagent_from_skill_has_correct_definition() {
        let md = "\
---
name: research
description: Research things thoroughly.
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
}
