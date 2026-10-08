//! Stable skill identities, host metadata and immutable catalog views.

use super::{Skill, SkillExecution};
use anda_core::BoxError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::Arc,
};

/// Bounds applied independently to discovery, resident metadata and tool responses.
#[derive(Debug, Clone)]
pub struct SkillLimits {
    /// Maximum descendant directory depth below each configured root.
    pub max_depth: usize,
    /// Maximum directories inspected per root.
    pub max_directories: usize,
    /// Maximum directory entries inspected per root.
    pub max_entries: usize,
    /// Maximum skill files retained across all roots.
    pub max_skills: usize,
    /// Maximum combined decoded SKILL.md and sidecar bytes retained per scan.
    pub max_total_bytes: usize,
    /// Maximum rendered catalog bytes appended to the manager description.
    pub catalog_bytes: usize,
    /// Maximum serialized JSON bytes returned by each skill tool.
    pub response_bytes: usize,
}

impl Default for SkillLimits {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_directories: 2_000,
            max_entries: 20_000,
            max_skills: 1_024,
            max_total_bytes: 32 * 1024 * 1024,
            catalog_bytes: 8_000,
            response_bytes: 32 * 1024,
        }
    }
}

/// A bounded diagnostic from discovery, parsing, filtering, or name resolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillDiagnostic {
    /// Stable category, such as `io`, `invalid`, `conflict`, or `limit`.
    pub kind: String,
    /// File or root associated with the diagnostic.
    #[serde(serialize_with = "serialize_path")]
    pub path: PathBuf,
    /// Bounded human-readable explanation; external metadata is untrusted.
    pub message: String,
}

/// Outcome of the last published scan or policy change.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillLoadReport {
    /// Catalog generation; unchanged reloads preserve this value.
    pub generation: u64,
    /// Number of admitted identities, including shadowed and ambiguous copies.
    pub loaded: usize,
    /// Number of files or parsed skills rejected during loading.
    pub rejected: usize,
    /// Whether traversal, content or diagnostic limits omitted information.
    pub truncated: bool,
    /// Up to 128 bounded diagnostics. Inspect `truncated` for omissions.
    pub diagnostics: Vec<SkillDiagnostic>,
}
impl SkillLoadReport {
    pub(super) fn note(&mut self, kind: &str, path: PathBuf, message: impl ToString) {
        if self.diagnostics.len() == 128 {
            self.truncated = true;
            return;
        }
        self.diagnostics.push(SkillDiagnostic {
            kind: kind.into(),
            path,
            message: truncate(&message.to_string(), 1_024).into(),
        });
    }
}

/// Additional metadata from `agents/openai.yaml`, or matching frontmatter metadata keys.
/// It never grants execution permissions or installs dependencies.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillMetadata {
    /// Optional compact presentation metadata. Asset paths are intentionally not consumed.
    #[serde(default)]
    pub interface: SkillInterface,
    /// Selection policy, independent of the host's enable/disable filter.
    #[serde(default)]
    pub policy: SkillPolicy,
    /// Required capabilities for a host-owned preflight.
    #[serde(default)]
    pub dependencies: SkillDependencies,
}

/// Text-only presentation metadata; it is not injected as privileged instructions.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillInterface {
    /// Optional display name for host UIs.
    pub display_name: Option<String>,
    /// Compact description used by the model catalog.
    pub short_description: Option<String>,
    /// Suggested prompt for host UIs; not executed automatically.
    pub default_prompt: Option<String>,
}

/// Controls automatic discovery, not authorization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillPolicy {
    /// False hides the skill from automatic catalogs; exact names and IDs still resolve.
    #[serde(default = "default_true")]
    pub allow_implicit_invocation: bool,
}
fn default_true() -> bool {
    true
}
impl Default for SkillPolicy {
    fn default() -> Self {
        Self {
            allow_implicit_invocation: true,
        }
    }
}

/// Requirements declared by a skill, separate from `allowed-tools`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillDependencies {
    /// Required local tools or MCP providers. Unknown kinds are reported as missing.
    #[serde(default)]
    pub tools: Vec<SkillDependency>,
}

/// A declarative capability requirement. Connection and installation details belong to the host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillDependency {
    /// `tool` for a callable name or `mcp` for a provider identity.
    #[serde(rename = "type")]
    pub kind: String,
    /// Exact host-defined capability name.
    pub value: String,
    /// Optional user-facing explanation.
    pub description: Option<String>,
}

/// Preflight result; this operation never connects, installs, or grants access.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillPreflight {
    /// Skill identity that was checked.
    pub skill_id: String,
    /// True when every declared requirement is available.
    pub ready: bool,
    /// Requirements absent from the supplied host capability inventories.
    pub missing: Vec<SkillDependency>,
}

/// Model- and host-readable metadata for a single admitted skill identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillSummary {
    /// Opaque path-based identity. Stable across frontmatter edits and root reordering.
    pub id: String,
    /// Opaque identity of the configured source directory.
    pub source_id: String,
    /// Frontmatter name, independent of the identity and callable name.
    pub name: String,
    /// Description of when to use the skill.
    pub description: String,
    /// Inline or delegated execution.
    pub execution: SkillExecution,
    /// Callable for the unambiguous winning subagent copy, otherwise absent.
    pub callable: Option<String>,
    /// Configured skill directory for resolving scripts in the host filesystem.
    #[serde(serialize_with = "serialize_path")]
    pub base_dir: PathBuf,
    /// True when this copy is the unambiguous winner for its name.
    pub active: bool,
    /// Presentation, selection policy and capability requirements.
    pub metadata: SkillMetadata,
}

/// An immutable catalog captured at a published generation. Old snapshots remain readable
/// as metadata, but never authorize future resource reads or callable dispatch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillCatalogSnapshot {
    /// Generation and diagnostics for this snapshot.
    pub report: SkillLoadReport,
    /// All admitted identities, in source priority and path order.
    pub skills: Vec<SkillSummary>,
}

#[derive(Clone)]
pub(super) struct Entry {
    pub id: String,
    pub source_id: String,
    pub rank: usize,
    pub root: PathBuf,
    pub canonical_root: PathBuf,
    pub skill: Arc<Skill>,
    pub metadata: SkillMetadata,
    pub fingerprint: String,
}

#[derive(Default)]
pub(super) struct Catalog {
    pub entries: BTreeMap<String, Entry>,
    pub winners: BTreeMap<String, String>,
    pub report: SkillLoadReport,
    pub fingerprint: String,
}
impl Catalog {
    pub fn summary(&self, entry: &Entry) -> SkillSummary {
        let skill = &entry.skill;
        let active = self.winners.get(&skill.agent_name) == Some(&entry.id);
        SkillSummary {
            id: entry.id.clone(),
            source_id: entry.source_id.clone(),
            name: skill.frontmatter.name.clone(),
            description: skill.frontmatter.description.clone(),
            execution: skill.execution,
            callable: (active && skill.is_subagent())
                .then(|| format!("{}{}", crate::context::SUB_AGENT_PREFIX, skill.agent_name)),
            base_dir: skill.base_dir.clone(),
            active,
            metadata: entry.metadata.clone(),
        }
    }
    pub fn snapshot(&self) -> SkillCatalogSnapshot {
        let mut entries = self.entries.values().collect::<Vec<_>>();
        entries.sort_by_key(|entry| (entry.rank, &entry.skill.base_dir));
        SkillCatalogSnapshot {
            report: self.report.clone(),
            skills: entries
                .into_iter()
                .map(|entry| self.summary(entry))
                .collect(),
        }
    }
    pub fn resolve(&self, name: &str) -> Result<Option<&Entry>, BoxError> {
        if let Some(entry) = self.entries.get(name) {
            return Ok(Some(entry));
        }
        let agent = super::normalise_skill_agent_name(name);
        if let Some(id) = self.winners.get(&agent) {
            return Ok(self.entries.get(id));
        }
        if self
            .entries
            .values()
            .any(|entry| entry.skill.frontmatter.name == name)
        {
            return Err(
                format!("multiple skills named {name:?}; use skills_list and a skill ID").into(),
            );
        }
        Ok(None)
    }
}

impl SkillSummary {
    /// Compare declared dependencies to host-approved callable and provider inventories.
    pub fn preflight(
        &self,
        tools: &BTreeSet<String>,
        providers: &BTreeSet<String>,
    ) -> SkillPreflight {
        let missing = self
            .metadata
            .dependencies
            .tools
            .iter()
            .filter(|dependency| !match dependency.kind.as_str() {
                "tool" => tools.contains(&dependency.value),
                "mcp" => providers.contains(&dependency.value),
                _ => false,
            })
            .cloned()
            .collect::<Vec<_>>();
        SkillPreflight {
            skill_id: self.id.clone(),
            ready: missing.is_empty(),
            missing,
        }
    }
}

pub(super) fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub(super) fn truncate(text: &str, bytes: usize) -> &str {
    let mut end = bytes.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn serialize_path<S: serde::Serializer>(
    path: &std::path::Path,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&path.to_string_lossy())
}
