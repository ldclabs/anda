//! Bounded discovery and package reads, exposed as ordinary local tools.

use super::catalog::{digest, truncate};
use super::discovery::{MAX_RESOURCE_BYTES, read_text, validate_resource};
use super::{SkillExecution, SkillManager};
use crate::{
    context::BaseCtx,
    extension::{hooked_call, tool_definition},
    hook::DynToolHook,
};
use anda_core::{BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Arguments for bounded skill discovery.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillsListArgs {
    /// Optional search of names/descriptions. Explicit-only skills require an exact name or ID.
    pub query: Option<String>,
    /// Continue an unchanged catalog/query using its previous next_cursor.
    pub cursor: Option<String>,
}

/// Compact catalog entry. Full policy and dependency metadata remains in the host snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillListing {
    /// Opaque identity accepted by skills_read.
    pub id: String,
    /// Opaque source identity for distinguishing copies.
    pub source_id: String,
    /// Exact frontmatter name.
    pub name: String,
    /// Bounded, single-line description; consult SKILL.md for full instructions.
    pub description: String,
    /// True when this is the unambiguous winning copy for its name.
    pub active: bool,
    /// Whether this skill requires an explicit selection.
    pub explicit_only: bool,
}

/// Bounded discovery result; omitted catalog descriptions can be recovered by reading the skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsListOutput {
    /// Catalog generation used for this page.
    pub generation: u64,
    /// At most 20 matching entries.
    pub skills: Vec<SkillListing>,
    /// Whether discovery omitted files because it reached a limit.
    pub scan_truncated: bool,
    /// Number of bounded diagnostics available to the host via SkillManager::catalog.
    pub diagnostic_count: usize,
    /// Continuation for the same catalog and query, or null at EOF.
    pub next_cursor: Option<String>,
}

/// Typed observation/customization hook for catalog discovery.
pub type SkillsListHook = DynToolHook<SkillsListArgs, SkillsListOutput>;

/// A pageable catalog tool sharing a SkillManager's registry.
pub struct SkillsListTool {
    manager: Arc<SkillManager>,
}
impl SkillsListTool {
    /// Stable function name.
    pub const NAME: &'static str = "skills_list";
    /// Share the supplied manager's roots, policies and catalog.
    pub fn new(manager: Arc<SkillManager>) -> Self {
        Self { manager }
    }

    async fn list(&self, args: SkillsListArgs) -> Result<SkillsListOutput, BoxError> {
        if args.query.as_ref().is_some_and(|query| query.len() > 1_024) {
            return Err("Skill query exceeds 1024 bytes".into());
        }
        self.manager.ensure_loaded().await?;
        let snapshot = self.manager.catalog();
        let query = args.query.unwrap_or_default().trim().to_string();
        let lowered = query.to_lowercase();
        let items = snapshot
            .skills
            .iter()
            .filter(|skill| {
                let exact = query == skill.name || query == skill.id;
                exact
                    || (skill.metadata.policy.allow_implicit_invocation
                        && (query.is_empty()
                            || skill.name.to_lowercase().contains(&lowered)
                            || skill.description.to_lowercase().contains(&lowered)))
            })
            .map(|skill| SkillListing {
                id: skill.id.clone(),
                source_id: skill.source_id.clone(),
                name: skill.name.clone(),
                description: truncate(
                    &skill
                        .metadata
                        .interface
                        .short_description
                        .as_deref()
                        .unwrap_or(&skill.description)
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" "),
                    512,
                )
                .to_string(),
                active: skill.active,
                explicit_only: !skill.metadata.policy.allow_implicit_invocation,
            })
            .collect::<Vec<_>>();
        let fingerprint = digest(&serde_json::to_vec(&(
            snapshot.report.generation,
            &query,
            &items,
        ))?);
        let start = cursor_offset(args.cursor.as_deref(), &fingerprint, items.len())?;
        let mut response = SkillsListOutput {
            generation: snapshot.report.generation,
            skills: vec![],
            scan_truncated: snapshot.report.truncated,
            diagnostic_count: snapshot.report.diagnostics.len(),
            next_cursor: None,
        };
        for (index, item) in items.iter().enumerate().skip(start).take(20) {
            response.skills.push(item.clone());
            response.next_cursor =
                (index + 1 < items.len()).then(|| cursor(&fingerprint, index + 1));
            // An unusually escape-heavy description must not make its identity undiscoverable.
            if response.skills.len() == 1 {
                while serde_json::to_vec(&response)?.len() > self.manager.limits.response_bytes {
                    let description = &mut response.skills[0].description;
                    if description.is_empty() {
                        break;
                    }
                    *description = truncate(description, description.len() / 2).to_string();
                }
            }
            if serde_json::to_vec(&response)?.len() > self.manager.limits.response_bytes {
                response.skills.pop();
                response.next_cursor = Some(cursor(&fingerprint, index));
                if response.skills.is_empty() {
                    return Err("Skill metadata exceeds response budget".into());
                }
                break;
            }
        }
        if serde_json::to_vec(&response)?.len() > self.manager.limits.response_bytes {
            return Err("Skill list exceeds response budget".into());
        }
        Ok(response)
    }
}
impl Tool<BaseCtx> for SkillsListTool {
    type Args = SkillsListArgs;
    type Output = SkillsListOutput;
    fn name(&self) -> String {
        Self::NAME.into()
    }
    fn description(&self) -> String {
        "List or search available skills with stable IDs. Follow next_cursor for more results. Use exact names or IDs for explicitly requested skills; use skills_read to read their instructions and bundled resources.".into()
    }
    fn definition(&self) -> FunctionDefinition {
        tool_definition::<Self::Args>(self.name(), self.description())
    }
    fn group(&self) -> Option<anda_core::ToolGroupInfo> {
        Some(skill_group())
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        hooked_call(&ctx, args, |args| async {
            let cancellation = ctx.cancellation_token();
            tokio::select! {
                _ = cancellation.cancelled() => Err("call was cancelled".into()),
                result = self.list(args) => result.map(ToolOutput::new),
            }
        })
        .await
    }
}

/// Read one page of a skill document or a package-contained text resource.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillsReadArgs {
    /// Exact skill name or stable ID from skills_list. IDs resolve name ambiguity.
    pub skill: String,
    /// Slash-separated path within this skill, e.g. references/guide.md; defaults to SKILL.md.
    pub resource: Option<String>,
    /// Continue an unchanged resource using the preceding next_cursor. Read instructions to EOF.
    pub cursor: Option<String>,
}

/// One complete UTF-8 page. A changed document, policy or package invalidates its cursor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsReadOutput {
    /// Resolved stable skill identity, including when the request used a name.
    pub skill_id: String,
    /// Frontmatter name.
    pub name: String,
    /// Execution mode of the skill.
    pub execution: SkillExecution,
    /// Callable only for the active, unambiguous subagent copy. Shadowed copies have none.
    pub callable: Option<String>,
    /// Skill directory in the host filesystem. Executing scripts still requires host permissions.
    pub base_dir: String,
    /// Package-relative resource read by this response.
    pub resource: String,
    /// Content fingerprint used by the continuation, not an authorization token.
    pub fingerprint: String,
    /// Current page of decoded text; follow next_cursor until EOF before acting on instructions.
    pub content: String,
    /// Next page, or null when the entire document has been read.
    pub next_cursor: Option<String>,
}
/// Typed hook for skill/resource usage, including resolved identity and content fingerprint.
pub type SkillsReadHook = DynToolHook<SkillsReadArgs, SkillsReadOutput>;

/// Package-scoped read tool that does not grant general filesystem or execution access.
pub struct SkillsReadTool {
    manager: Arc<SkillManager>,
}
impl SkillsReadTool {
    /// Stable function name.
    pub const NAME: &'static str = "skills_read";
    /// Share the supplied manager's roots, policies and catalog.
    pub fn new(manager: Arc<SkillManager>) -> Self {
        Self { manager }
    }
    async fn read(&self, args: SkillsReadArgs) -> Result<SkillsReadOutput, BoxError> {
        let resource = args.resource.unwrap_or_else(|| "SKILL.md".into());
        validate_resource(&resource)?;
        let (_summary, main_content, entry) = self.manager.read_selected(&args.skill).await?;
        let content = if resource == "SKILL.md" {
            main_content
        } else {
            let relative = entry
                .skill
                .base_dir
                .strip_prefix(&entry.root)?
                .join(&resource);
            read_text(&entry.canonical_root, &relative, MAX_RESOURCE_BYTES).await?
        };
        // Recheck admission and bind the response metadata to the same document revision.
        let summary = self.manager.current_summary(&entry)?;
        let fingerprint = digest(&serde_json::to_vec(&(
            &summary.id,
            &resource,
            &entry.fingerprint,
            &content,
        ))?);
        let start = cursor_offset(args.cursor.as_deref(), &fingerprint, content.len())?;
        if !content.is_char_boundary(start) {
            return Err("Invalid skill cursor character boundary".into());
        }
        let mut output = SkillsReadOutput {
            skill_id: summary.id,
            name: summary.name,
            execution: summary.execution,
            callable: summary.callable,
            base_dir: summary.base_dir.display().to_string(),
            resource,
            fingerprint,
            content: content[start..].into(),
            next_cursor: None,
        };
        if serde_json::to_vec(&output)?.len() <= self.manager.limits.response_bytes {
            return Ok(output);
        }
        let mut low = start;
        let mut high = content.len();
        let mut best = None;
        while low < high {
            let mut end = low + (high - low).div_ceil(2);
            while !content.is_char_boundary(end) {
                end += 1;
            }
            output.content = content[start..end].into();
            output.next_cursor = Some(cursor(&output.fingerprint, end));
            if serde_json::to_vec(&output)?.len() <= self.manager.limits.response_bytes {
                low = end;
                best = Some(output.clone());
            } else {
                high = end - 1;
                while !content.is_char_boundary(high) {
                    high -= 1;
                }
            }
        }
        best.ok_or_else(|| "Skill response budget leaves no room for content".into())
    }
}
impl Tool<BaseCtx> for SkillsReadTool {
    type Args = SkillsReadArgs;
    type Output = SkillsReadOutput;
    fn name(&self) -> String {
        Self::NAME.into()
    }
    fn description(&self) -> String {
        "Read a skill's SKILL.md or bundled text resource by exact name or ID. Resolve resource paths relative to this skill package. Follow every next_cursor until EOF before acting on instructions. A stale cursor requires restarting the read. Reading never grants shell or file-write permissions; non-active subagent copies have no callable.".into()
    }
    fn definition(&self) -> FunctionDefinition {
        tool_definition::<Self::Args>(self.name(), self.description())
    }
    fn group(&self) -> Option<anda_core::ToolGroupInfo> {
        Some(skill_group())
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        hooked_call(&ctx, args, |args| async {
            let cancellation = ctx.cancellation_token();
            tokio::select! {
                _ = cancellation.cancelled() => Err("call was cancelled".into()),
                result = self.read(args) => result.map(ToolOutput::new),
            }
        })
        .await
    }
}
fn cursor(fingerprint: &str, offset: usize) -> String {
    format!("{fingerprint}:{offset}")
}
fn cursor_offset(value: Option<&str>, fingerprint: &str, length: usize) -> Result<usize, BoxError> {
    let Some(value) = value else {
        return Ok(0);
    };
    if value.len() > 96 {
        return Err("Invalid skill cursor".into());
    }
    let (hash, offset) = value.split_once(':').ok_or("Invalid skill cursor")?;
    if hash != fingerprint {
        return Err("Skill cursor is stale; restart without cursor".into());
    }
    let offset = offset
        .parse::<usize>()
        .map_err(|_| "Invalid skill cursor offset")?;
    if offset > length {
        return Err("Invalid skill cursor offset".into());
    }
    Ok(offset)
}

pub(super) fn skill_group() -> anda_core::ToolGroupInfo {
    anda_core::ToolGroupInfo {
        id: "skills".into(), title: "Skills".into(),
        description: "Discover reusable skills and read their complete instructions and bundled text resources.".into(),
        instructions: Some("Use skills_list to discover identities; skills_manager reads small SKILL.md files by name; skills_read pages large documents and package resources. Follow next_cursor until EOF. Reading a skill never grants execution permissions.".into()),
    }
}
