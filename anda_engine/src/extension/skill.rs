//! File-backed skills with bounded discovery, immutable catalogs and package resource access.
//!
//! Register [`SkillManager::tools`] for `skills_manager`, `skills_list`, and `skills_read`.
//! Skills default to inline execution; only `execution: subagent` creates `SA_` callables.
//! Register the same manager as a [`SubAgentSet`] when delegated skills are wanted.
//! Bundled text resources are read through `skills_read` without granting general file or shell
//! access. Script execution remains subject to the host's workspace and shell permissions.
//!
//! [`SkillManager::load`] explicitly reloads membership. Hosts can call [`SkillManager::invalidate`]
//! after filesystem notifications; the next async read/list coalesces a refresh. Known reads
//! revalidate their file and metadata, while name misses also refresh. Synchronous catalogs and
//! callable lookups always use the last published snapshot, never perform filesystem I/O.

use crate::{
    context::BaseCtx,
    extension::{hooked_call, tool_definition},
    hook::DynToolHook,
    subagent::{SubAgent, SubAgentSet},
};
use anda_core::{
    Agent, BoxError, FunctionDefinition, Resource, Tool, ToolOutput, ToolSet, select_resources,
};
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

mod catalog;
mod discovery;
mod tools;
mod types;
pub use catalog::*;
use catalog::{Catalog, Entry, digest, truncate};
#[cfg(test)]
use discovery::MAX_SKILL_FILE_BYTES;
pub use tools::*;
pub use types::*;
/// Arguments for reading a skill via the [`SkillManager`] tool.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillArgs {
    /// Skill name in kebab-case (e.g. 'pdf-processing'). Returns the matching SKILL.md content. When `execution` is `inline`, follow that content yourself; when it is `subagent`, call the returned `callable` with a self-contained prompt instead.
    pub name: String,
}

/// Typed hook for skills-manager tool calls.
pub type SkillToolHook = DynToolHook<SkillArgs, SkillContentOutput>;

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

/// Host admission predicate. Rejected identities are absent from catalogs, reads and callables.
/// The callback executes under internal locks and must not call back into this manager.
pub type SkillFilter = Arc<dyn Fn(&Skill) -> bool + Send + Sync>;

#[derive(Default)]
struct Registry {
    catalog: Arc<Catalog>,
    subagents: BTreeMap<String, SubAgent>,
    epoch: u64,
}

/// Shared skill catalog and optional delegated workers. All published views share one registry
/// generation; a retained metadata snapshot never authorizes a later read.
pub struct SkillManager {
    default_skills_dir: PathBuf,
    skills_dirs: Vec<PathBuf>,
    registry: RwLock<Registry>,
    filter: RwLock<Option<SkillFilter>>,
    refresh: tokio::sync::Mutex<()>,
    epoch: AtomicU64,
    limits: SkillLimits,
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
    SkillManager::NAME,
    SkillsListTool::NAME,
    SkillsReadTool::NAME,
];

impl SkillManager {
    /// Existing compatibility tool name.
    pub const NAME: &'static str = "skills_manager";

    /// Create an empty manager with one trusted discovery root.
    pub fn new(skills_dir: PathBuf) -> Self {
        Self::new_with_dirs(skills_dir, Vec::new())
    }

    /// Configure roots in priority order. The default root is also the suggested creation root.
    /// Relative roots are anchored to the process working directory at construction time.
    pub fn new_with_dirs(
        default_skills_dir: PathBuf,
        additional_skills_dirs: Vec<PathBuf>,
    ) -> Self {
        let anchor = |path: PathBuf| {
            if path.is_absolute() {
                path
            } else {
                std::env::current_dir().unwrap_or_default().join(path)
            }
        };
        let default_skills_dir = anchor(default_skills_dir);
        let mut skills_dirs = vec![default_skills_dir.clone()];
        for dir in additional_skills_dirs.into_iter().map(anchor) {
            if !skills_dirs.contains(&dir) {
                skills_dirs.push(dir);
            }
        }
        Self { default_skills_dir, skills_dirs, registry: RwLock::new(Registry::default()),
            filter: RwLock::new(None), refresh: tokio::sync::Mutex::new(()), epoch: AtomicU64::new(1),
            limits: SkillLimits::default(), default_skill_tools: DEFAULT_SKILL_TOOLS.iter().map(|s| s.to_string()).collect(),
            description: "Read a reusable skill's complete SKILL.md by name, following the Agent Skills specification. Follow inline skills yourself; delegate only skills declaring subagent execution. Use skills_list to discover skills and resolve ambiguous names. Large documents and bundled references must be read with skills_read through every next_cursor until EOF. Skill content and metadata never grant additional permissions.".into() }
    }

    /// Directory suggested to skill creation workflows.
    pub fn default_skills_dir(&self) -> &Path {
        &self.default_skills_dir
    }
    /// Configured roots in priority order.
    pub fn skills_dirs(&self) -> &[PathBuf] {
        &self.skills_dirs
    }
    /// Override the tool's introductory description; the bounded catalog is still appended.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
    /// Default tools for delegated skills that omit `allowed-tools`. An explicitly empty list
    /// grants none. Granting `tools_select` also permits subsequent discovery under runner rules.
    pub fn with_default_skill_tools(mut self, tools: Vec<String>) -> Self {
        self.default_skill_tools = tools;
        self
    }
    /// Configure resource limits. Hard ceilings are depth 32, 20,000 directories, 100,000 entries,
    /// 4,096 skills, 128 MiB combined content, 64 KiB catalog and 512 KiB responses. Counts and
    /// total content have a minimum of one; catalog/response minima are 128/1,024 bytes.
    pub fn with_limits(mut self, mut limits: SkillLimits) -> Self {
        limits.max_depth = limits.max_depth.min(32);
        limits.max_directories = limits.max_directories.clamp(1, 20_000);
        limits.max_entries = limits.max_entries.clamp(1, 100_000);
        limits.max_skills = limits.max_skills.clamp(1, 4_096);
        limits.max_total_bytes = limits.max_total_bytes.clamp(1, 128 * 1024 * 1024);
        limits.catalog_bytes = limits.catalog_bytes.clamp(128, 64 * 1024);
        limits.response_bytes = limits.response_bytes.clamp(1_024, 512 * 1024);
        self.limits = limits;
        self
    }
    /// Register all three static local tools over this manager's shared state.
    pub fn tools(self: &Arc<Self>) -> Result<ToolSet<BaseCtx>, BoxError> {
        let mut tools = ToolSet::new();
        tools.add(self.clone())?;
        tools.add(Arc::new(SkillsListTool::new(self.clone())))?;
        tools.add(Arc::new(SkillsReadTool::new(self.clone())))?;
        Ok(tools)
    }
    /// Invalidate membership after a filesystem or host configuration change. No I/O is done here.
    pub fn invalidate(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Replace the host admission policy and immediately remove rejected identities. Newly
    /// admitted files are discovered on the next async access or explicit reload.
    pub fn set_skill_filter(&self, filter: Option<SkillFilter>) {
        let mut current = self.filter.write();
        *current = filter;
        let mut registry = self.registry.write();
        let entries = registry.catalog.entries.values().cloned().collect();
        let report = SkillLoadReport::default();
        self.publish(&mut registry, entries, report, current.as_ref());
        self.invalidate();
    }
    /// Last published immutable metadata and diagnostics, without filesystem I/O.
    pub fn catalog(&self) -> Arc<SkillCatalogSnapshot> {
        Arc::new(self.registry.read().catalog.snapshot())
    }
    /// Compatibility reload API. Inspect [`Self::catalog`] for per-file failures and truncation.
    pub async fn load(&self) -> Result<(), BoxError> {
        let report = self.reload().await?;
        for diagnostic in &report.diagnostics {
            log::warn!(
                "skill {} at {}: {}",
                diagnostic.kind,
                diagnostic.path.display(),
                diagnostic.message
            );
        }
        Ok(())
    }
    /// Rescan and publish all roots atomically. Missing roots become empty; invalid or unreadable
    /// files are omitted with diagnostics. No stale entries survive a successful publication.
    pub async fn reload(&self) -> Result<SkillLoadReport, BoxError> {
        let _guard = self.refresh.lock().await;
        self.scan_and_publish().await
    }
    async fn ensure_loaded(&self) -> Result<(), BoxError> {
        if self.registry.read().epoch == self.epoch.load(Ordering::Acquire) {
            return Ok(());
        }
        let _guard = self.refresh.lock().await;
        if self.registry.read().epoch != self.epoch.load(Ordering::Acquire) {
            self.scan_and_publish().await?;
        }
        Ok(())
    }
    async fn scan_and_publish(&self) -> Result<SkillLoadReport, BoxError> {
        let epoch = self.epoch.load(Ordering::Acquire);
        let mut entries = Vec::new();
        let mut report = SkillLoadReport::default();
        let mut bytes = 0usize;
        let mut visited = 0;
        'roots: for (rank, root) in self.skills_dirs.iter().enumerate() {
            let scan = discovery::scan_files(root, &self.limits).await;
            report.rejected += scan.report.rejected;
            report.truncated |= scan.report.truncated;
            for diagnostic in scan.report.diagnostics {
                report.note(&diagnostic.kind, diagnostic.path, diagnostic.message);
            }
            for path in scan.files {
                if visited == self.limits.max_skills {
                    report.truncated = true;
                    report.note("limit", root.clone(), "Global skill count limit reached");
                    break 'roots;
                }
                visited += 1;
                match discovery::read_entry(root, &path, rank).await {
                    Ok((entry, _, size)) => {
                        bytes = bytes.saturating_add(size);
                        if bytes > self.limits.max_total_bytes {
                            report.truncated = true;
                            report.note("limit", path, "Combined skill content limit reached");
                            break 'roots;
                        }
                        entries.push(entry);
                    }
                    Err(error) => {
                        report.rejected += 1;
                        report.note("invalid", path, error);
                    }
                }
            }
        }
        let filter = self.filter.read();
        let mut registry = self.registry.write();
        self.publish(&mut registry, entries, report, filter.as_ref());
        registry.epoch = epoch;
        Ok(registry.catalog.report.clone())
    }

    fn publish(
        &self,
        registry: &mut Registry,
        entries: Vec<Entry>,
        mut report: SkillLoadReport,
        filter: Option<&SkillFilter>,
    ) {
        let mut catalog = Catalog::default();
        for entry in entries {
            if filter.is_some_and(|filter| !filter(&entry.skill)) {
                report.rejected += 1;
                continue;
            }
            // A nested/repeated root cannot create a second identity for the same file.
            catalog.entries.entry(entry.id.clone()).or_insert(entry);
        }
        let mut by_name: BTreeMap<String, Vec<&Entry>> = BTreeMap::new();
        for entry in catalog.entries.values() {
            by_name
                .entry(entry.skill.agent_name.clone())
                .or_default()
                .push(entry);
        }
        for (name, mut copies) in by_name {
            copies.sort_by_key(|entry| (entry.rank, &entry.skill.base_dir));
            let first = copies[0];
            // Callable hash collisions must never route one frontmatter name to another.
            let collision = copies
                .iter()
                .any(|entry| entry.skill.frontmatter.name != first.skill.frontmatter.name);
            if collision || copies.get(1).is_some_and(|entry| entry.rank == first.rank) {
                report.note(
                    "conflict",
                    first.skill.base_dir.clone(),
                    format!(
                        "multiple skills named {:?}; select by ID",
                        first.skill.frontmatter.name
                    ),
                );
            } else {
                catalog.winners.insert(name, first.id.clone());
            }
        }
        report.loaded = catalog.entries.len();
        let identities = catalog
            .entries
            .values()
            .map(|entry| (&entry.id, &entry.fingerprint, entry.rank))
            .collect::<Vec<_>>();
        catalog.fingerprint =
            digest(&serde_json::to_vec(&(identities, &report)).expect("catalog data serializes"));
        report.generation = registry.catalog.report.generation
            + u64::from(catalog.fingerprint != registry.catalog.fingerprint);
        catalog.report = report;
        let mut agents = BTreeMap::new();
        for (name, id) in &catalog.winners {
            let entry = &catalog.entries[id];
            if !entry.skill.is_subagent() {
                continue;
            }
            let mut agent = SubAgent::from(entry.skill.as_ref());
            if !entry.skill.declares_tools() {
                agent.tools = self.default_skill_tools.clone();
            }
            if registry.catalog.winners.get(name) == Some(id)
                && let Some(old) = registry.subagents.get(name)
            {
                agent.subsessions = old.subsessions.clone();
            }
            agents.insert(name.clone(), agent);
        }
        registry.subagents = agents;
        registry.catalog = Arc::new(catalog);
    }

    async fn read_selected(
        &self,
        selector: &str,
    ) -> Result<(SkillSummary, String, Entry), BoxError> {
        if !selector.starts_with("skill://") {
            validate_skill_name(selector)?;
        }
        if selector.len() > 128 {
            return Err("Invalid skill identity".into());
        }
        self.ensure_loaded().await?;
        let mut last_error = None;
        for attempt in 0..2 {
            let entry = self.registry.read().catalog.resolve(selector)?.cloned();
            if let Some(entry) = entry {
                let path = entry.skill.base_dir.join("SKILL.md");
                match discovery::read_entry(&entry.root, &path, entry.rank).await {
                    Ok((fresh, content, _))
                        if fresh.id == entry.id && fresh.fingerprint == entry.fingerprint =>
                    {
                        let summary = self.current_summary(&fresh)?;
                        if !selector.starts_with("skill://") && !summary.active {
                            return Err(
                                "Skill name changed precedence during read; retry or select by ID"
                                    .into(),
                            );
                        }
                        return Ok((summary, content, fresh));
                    }
                    Ok(_) => (),
                    Err(error) => last_error = Some(error),
                }
            }
            if attempt == 0 {
                self.reload().await?;
            }
        }
        if let Some(error) = last_error {
            return Err(error);
        }
        let registry = self.registry.read();
        if let Some(entry) = registry.catalog.entries.values().find(|entry| {
            entry
                .skill
                .base_dir
                .file_name()
                .is_some_and(|name| name == selector)
        }) {
            return Err(format!(
                "SKILL.md frontmatter name {:?} must match requested skill name {selector:?}",
                entry.skill.frontmatter.name
            )
            .into());
        }
        if let Some(diagnostic) = registry
            .catalog
            .report
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic
                    .path
                    .parent()
                    .and_then(Path::file_name)
                    .is_some_and(|name| name == selector)
            })
        {
            return Err(diagnostic.message.clone().into());
        }
        Err(format!("skill {selector:?} not found; reload after changing skill files").into())
    }

    fn current_summary(&self, entry: &Entry) -> Result<SkillSummary, BoxError> {
        let filter = self.filter.read();
        if filter.as_ref().is_some_and(|filter| !filter(&entry.skill)) {
            return Err("Skill is no longer available".into());
        }
        let registry = self.registry.read();
        let current = registry
            .catalog
            .entries
            .get(&entry.id)
            .ok_or("Skill is no longer available")?;
        if current.fingerprint != entry.fingerprint {
            return Err("Skill changed during read; retry".into());
        }
        Ok(registry.catalog.summary(current))
    }

    async fn read_skill_action(&self, args: SkillArgs) -> Result<SkillContentOutput, BoxError> {
        // Preserve the existing name-only wire contract; IDs are accepted by skills_read.
        validate_skill_name(&args.name)?;
        let (summary, content, _) = self.read_selected(&args.name).await?;
        let target = summary.base_dir.join("SKILL.md");
        let path = self
            .skills_dirs
            .iter()
            .find_map(|root| target.strip_prefix(root).ok())
            .map(crate::extension::fs::normalize_relative_path)
            .unwrap_or_else(|| target.display().to_string());
        let output = SkillContentOutput {
            name: summary.name,
            description: summary.description,
            execution: summary.execution,
            callable: summary.callable,
            base_dir: summary.base_dir.display().to_string(),
            path,
            content,
        };
        if serde_json::to_vec(&output)?.len() > self.limits.response_bytes {
            return Err(format!("Skill exceeds inline response budget; read the complete document with skills_read using skill {:?} and follow next_cursor until EOF", summary.id).into());
        }
        Ok(output)
    }

    /// Retrieve the winning skill by its normalized callable name, without I/O.
    pub fn get_skill(&self, lowercase_name: &str) -> Option<Skill> {
        let registry = self.registry.read();
        let id = registry.catalog.winners.get(lowercase_name)?;
        Some(registry.catalog.entries.get(id)?.skill.as_ref().clone())
    }
    /// All unambiguous delegated workers, including explicit-only skills, for host inspection.
    pub fn subagents(&self) -> Vec<SubAgent> {
        self.registry.read().subagents.values().cloned().collect()
    }
    /// Compatibility view of unambiguous winning skills, keyed by normalized callable name.
    pub fn list(&self) -> BTreeMap<String, Skill> {
        let registry = self.registry.read();
        registry
            .catalog
            .winners
            .iter()
            .map(|(name, id)| {
                (
                    name.clone(),
                    registry.catalog.entries[id].skill.as_ref().clone(),
                )
            })
            .collect()
    }
    fn skills_catalog(&self) -> String {
        let registry = self.registry.read();
        let mut entries = registry
            .catalog
            .entries
            .values()
            .filter(|entry| entry.metadata.policy.allow_implicit_invocation)
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| (&entry.skill.frontmatter.name, entry.rank, &entry.id));
        if entries.is_empty() {
            return String::new();
        }
        let header = "\nLoaded skills (name, execution, description):";
        let lines = entries
            .into_iter()
            .map(|entry| {
                let description = entry
                    .metadata
                    .interface
                    .short_description
                    .as_deref()
                    .unwrap_or(&entry.skill.frontmatter.description)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                (
                    format!(
                        "\n- {} [{}]: ",
                        entry.skill.frontmatter.name, entry.skill.execution
                    ),
                    description,
                )
            })
            .collect::<Vec<_>>();
        let full_bytes = header.len()
            + lines
                .iter()
                .map(|(prefix, description)| prefix.len() + description.len())
                .sum::<usize>();
        if full_bytes <= self.limits.catalog_bytes {
            return format!(
                "{header}{}",
                lines
                    .into_iter()
                    .map(|(prefix, description)| prefix + &description)
                    .collect::<String>()
            );
        }
        // Allocate names first, then divide remaining bytes fairly between descriptions.
        let mut output = header.to_string();
        let mut names_bytes = 0;
        let mut count = 0;
        for (prefix, _) in &lines {
            if header.len() + names_bytes + prefix.len() + 80 > self.limits.catalog_bytes {
                break;
            }
            names_bytes += prefix.len();
            count += 1;
        }
        let description_budget = self
            .limits
            .catalog_bytes
            .saturating_sub(header.len() + names_bytes + 80)
            / count.max(1);
        for (prefix, description) in lines.iter().take(count) {
            output.push_str(prefix);
            output.push_str(truncate(description, description_budget));
        }
        let omitted = lines.len() - count;
        if omitted > 0 {
            output.push_str(&format!(
                "\n{omitted} additional skills omitted; use skills_list to discover them."
            ));
        } else {
            output
                .push_str("\nDescriptions shortened; use skills_list and skills_read for details.");
        }
        truncate(&output, self.limits.catalog_bytes).into()
    }
}

impl SubAgentSet for SkillManager {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
    fn contains_lowercase(&self, name: &str) -> bool {
        self.registry.read().subagents.contains_key(name)
    }
    fn get_lowercase(&self, name: &str) -> Option<SubAgent> {
        self.registry.read().subagents.get(name).cloned()
    }
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        let registry = self.registry.read();
        registry
            .subagents
            .iter()
            .filter(|(name, _)| match names {
                Some(names) => names
                    .iter()
                    .any(|requested| requested.eq_ignore_ascii_case(name)),
                None => registry
                    .catalog
                    .winners
                    .get(*name)
                    .and_then(|id| registry.catalog.entries.get(id))
                    .is_some_and(|entry| entry.metadata.policy.allow_implicit_invocation),
            })
            .map(|(_, agent)| agent.definition())
            .collect()
    }
    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        self.get_lowercase(&name.to_ascii_lowercase())
            .map(|agent| select_resources(resources, &agent.supported_resource_tags()))
            .unwrap_or_default()
    }
}
impl Tool<BaseCtx> for SkillManager {
    type Args = SkillArgs;
    type Output = SkillContentOutput;
    fn name(&self) -> String {
        Self::NAME.into()
    }
    fn description(&self) -> String {
        format!("{}{}", self.description, self.skills_catalog())
    }
    fn definition(&self) -> FunctionDefinition {
        let mut definition = tool_definition::<Self::Args>(self.name(), self.description());
        definition.parameters["description"] = "Read a reusable skill's SKILL.md file content by skill name. Create or update skills with separately authorized file tools, then reload or invalidate the manager. Use skills_read for paginated documents and package resources.".into();
        definition
    }
    fn group(&self) -> Option<anda_core::ToolGroupInfo> {
        Some(tools::skill_group())
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        use anda_core::StateFeatures;
        hooked_call(&ctx, args, |args| async {
            let cancellation = ctx.cancellation_token();
            tokio::select! {
                _ = cancellation.cancelled() => Err("call was cancelled".into()),
                result = self.read_skill_action(args) => result.map(ToolOutput::new),
            }
        })
        .await
    }
}

#[cfg(test)]
mod catalog_tests;
#[cfg(test)]
mod tests;
