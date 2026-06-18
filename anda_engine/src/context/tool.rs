//! Built-in callable discovery tools.
//!
//! The search and select helpers keep large tool schemas out of the prompt
//! until the model asks for them. They are registered as lightweight agents
//! whose outputs carry the matching [`FunctionDefinition`] values.

use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest,
    FunctionDefinition, Resource, ToolGroup,
};
use anda_db_tfs::{TokenizerChain, collect_tokens, jieba_tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

use crate::context::{
    AgentCtx, REMOTE_AGENT_PREFIX, REMOTE_TOOL_PREFIX, SUB_AGENT_PREFIX,
    strip_prefix_ignore_ascii_case,
};

/// Built-in name for keyword search across available callables.
pub const TOOLS_SEARCH_NAME: &str = "tools_search";
/// Built-in name for selecting callables by exact name.
pub const TOOLS_SELECT_NAME: &str = "tools_select";
/// Built-in name for listing capability groups (the group directory).
pub const TOOLS_GROUPS_NAME: &str = "tools_groups";

/// Arguments for [`ToolsSearch`].
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsSearchArgs {
    /// Search terms, or `*` to enumerate every available callable name.
    pub query: String,
    /// Maximum number of results to return. Defaults to `10`.
    #[serde(default)]
    pub limit: usize,
}

/// Output returned by callable discovery tools.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsOutput {
    /// Matching callable definitions returned as tool-output context.
    ///
    /// These definitions are not dynamically inserted into [`CompletionRequest::tools`].
    pub tools: Vec<FunctionDefinition>,
    /// Capability groups that the returned tools belong to.
    ///
    /// Each group bundles related tools from a single source (for example one
    /// MCP server) and carries the bundle's purpose and usage instructions plus
    /// the full list of sibling member names, so the model can understand how
    /// the tools combine and select additional members it still needs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<ToolGroup>,
    /// Total number of callables to the current model turn.
    #[serde(default)]
    pub total_tools: usize,
}

/// Searches the callable surface currently available to the model.
/// The "tools_search" tool has been registered as a built-in agent with label "flash".
pub struct ToolsSearch {
    tokenizer: TokenizerChain,
}

impl Default for ToolsSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolsSearch {
    /// Function name used when registering the search helper.
    pub const NAME: &'static str = TOOLS_SEARCH_NAME;

    /// Creates a search helper with the default tokenizer.
    pub fn new() -> Self {
        let tokenizer = jieba_tokenizer();
        Self { tokenizer }
    }

    /// Searches candidate definitions by name, description, and token overlap.
    pub fn search(&self, candidates: &[FunctionDefinition], args: &ToolsSearchArgs) -> ToolsOutput {
        let normalized_query = args.query.trim().to_lowercase();
        let tools: Vec<FunctionDefinition> = candidates.to_vec();

        let total_tools = tools.len();
        if normalized_query == "*" {
            return ToolsOutput {
                tools,
                total_tools,
                ..Default::default()
            };
        }

        let normalized_tokens: Vec<(String, usize)> =
            collect_tokens(&mut self.tokenizer.clone(), &normalized_query, None)
                .into_iter()
                .collect();

        let mut tools_name =
            rank_search_items(&tools, &normalized_query, &normalized_tokens, false);
        tools_name.truncate(if args.limit == 0 { 10 } else { args.limit });
        let tools = select_requested_definitions(tools, &tools_name);
        ToolsOutput {
            tools,
            total_tools,
            ..Default::default()
        }
    }
}

impl Agent<AgentCtx> for ToolsSearch {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Search callable tools and agents by keyword. Returns full callable schemas in this tool output; after a schema is returned, call that tool/agent directly instead of searching again. The output may also include `groups`: related tool bundles (for example one MCP server) with their purpose, usage instructions, and sibling member names to help you combine them."
            .to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms for callable tools/agents, or `*` to list every available callable name. Do not repeat the same search after the needed schema is returned.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Defaults to `10`."
                    }
                },
                "required": ["query", "limit"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let args: ToolsSearchArgs = match serde_json::from_str(&prompt) {
            Ok(v) => v,
            Err(e) => {
                return Ok(AgentOutput {
                    content: format!("Invalid input: {e}"),
                    ..Default::default()
                });
            }
        };

        let definitions = ctx.definitions(None).await;
        if args.query.trim().is_empty() || definitions.is_empty() {
            return Ok(AgentOutput {
                content: serde_json::to_string(&ToolsOutput {
                    tools: Vec::new(),
                    total_tools: definitions.len(),
                    ..Default::default()
                })?,
                ..Default::default()
            });
        }
        let mut rt = self.search(&definitions, &args);
        rt.groups = relevant_groups(ctx.tool_groups(), &rt.tools);
        Ok(AgentOutput {
            content: serde_json::to_string(&rt)?,
            ..Default::default()
        })
    }
}

/// Arguments for [`ToolsSelect`].
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsSelectArgs {
    /// Callable names whose schemas should be returned for direct calls.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Natural-language intent used to select tools when exact names are unknown.
    #[serde(default)]
    pub query: String,
    /// Capability group id to expand. When set, every member of that group is
    /// returned, on top of any names listed in `tools`. Discover group ids with
    /// `tools_groups`.
    #[serde(default)]
    pub group: String,
    /// Maximum number of resolved definitions to return. Defaults to `5`, and is capped at `16` to prevent overloading the next model turn.
    #[serde(default)]
    pub limit: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct ToolsSelectNamesOutput {
    #[serde(default)]
    tools: Vec<String>,
}

/// Returns a subset of callable definitions for direct use in the next reasoning turn.
/// The "tools_select" tool has been registered as a built-in agent with label "flash".
pub struct ToolsSelect {
    tokenizer: TokenizerChain,
}

impl Default for ToolsSelect {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_SELECTOR_LIMIT: usize = 16;
const MAX_SELECTOR_CANDIDATE_LIMIT: usize = 1000;

impl ToolsSelect {
    /// Function name used when registering the selection helper.
    pub const NAME: &'static str = TOOLS_SELECT_NAME;

    /// Creates a callable selection helper.
    pub fn new() -> Self {
        Self {
            tokenizer: jieba_tokenizer(),
        }
    }

    async fn select_requested_definitions_by_query(
        &self,
        ctx: &AgentCtx,
        definitions: Vec<FunctionDefinition>,
        args: &ToolsSelectArgs,
    ) -> Vec<FunctionDefinition> {
        let normalized_query = args.query.trim().to_lowercase();
        let limit = if args.limit > 0 {
            args.limit.min(MAX_SELECTOR_LIMIT)
        } else {
            5
        };

        let mut candidates = self.collect_query_candidates(
            &definitions,
            &normalized_query,
            MAX_SELECTOR_CANDIDATE_LIMIT,
        );

        if candidates.is_empty() {
            return Vec::new();
        }

        if normalized_query.len() <= 3 {
            candidates.truncate(limit);
            return select_requested_definitions(definitions, &candidates);
        }

        let mut requested =
            select_requested_names_with_model(ctx, &definitions, &normalized_query, limit).await;
        if requested.is_empty() {
            candidates.truncate(limit);
            requested = candidates;
        }

        select_requested_definitions(definitions, &requested)
    }

    fn collect_query_candidates(
        &self,
        items: &[FunctionDefinition],
        query: &str,
        candidate_limit: usize,
    ) -> Vec<String> {
        let mut rt = if !query.is_empty() && query != "*" {
            // (lowercase token, weight)
            let normalized_tokens: Vec<(String, usize)> =
                collect_tokens(&mut self.tokenizer.clone(), query, None)
                    .into_iter()
                    .collect();

            rank_search_items(items, query, &normalized_tokens, true)
        } else {
            items
                .iter()
                .take(candidate_limit)
                .map(|item| item.name.clone())
                .collect::<Vec<_>>()
        };

        rt.truncate(candidate_limit);
        rt
    }
}

impl Agent<AgentCtx> for ToolsSelect {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Select callable tools or agents and return full schemas in this tool output for direct tool calls. Use exact names via `tools`; use `query` only when exact names are unknown; use `group` to pull in every tool of a capability group at once (discover group ids with `tools_groups`). Do not call tools_select again for the same returned tools. The output may also include `groups`: related tool bundles (for example one MCP server) with their purpose, usage instructions, and sibling member names.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Exact callable names to select. After these schemas are returned in this tool output, call the selected tools/agents directly."
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural-language intent for selecting relevant callables when exact names are unknown. Prefer `tools` when exact names are known."
                    },
                    "group": {
                        "type": "string",
                        "description": "Capability group id to expand. Returns every member tool of that group. Use an empty string when not selecting by group."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of resolved callables to return for `query` selection. Defaults to `5`, and is capped at `16`."
                    }
                },
                "required": ["tools", "query", "group", "limit"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let args: ToolsSelectArgs = match serde_json::from_str(&prompt) {
            Ok(v) => v,
            Err(e) => {
                return Ok(AgentOutput {
                    content: format!("Invalid input: {e}"),
                    ..Default::default()
                });
            }
        };

        if args.tools.is_empty() && args.query.trim().is_empty() && args.group.trim().is_empty() {
            return Ok(AgentOutput {
                content: "Invalid input: one of `tools`, `query`, or `group` must be provided"
                    .to_string(),
                ..Default::default()
            });
        }

        let definitions = ctx.definitions(None).await;
        let total_tools = definitions.len();
        let all_groups = ctx.tool_groups();

        // Explicit names plus, when a group is named, every member of that group.
        let mut requested = args.tools.clone();
        if !args.group.trim().is_empty()
            && let Some(group) = all_groups
                .iter()
                .find(|group| group.id.eq_ignore_ascii_case(args.group.trim()))
        {
            requested.extend(group.members.iter().cloned());
        }

        let tool_definitions = if !requested.is_empty() {
            select_requested_definitions(definitions, &requested)
        } else {
            self.select_requested_definitions_by_query(&ctx, definitions, &args)
                .await
        };

        let groups = relevant_groups(all_groups, &tool_definitions);
        Ok(AgentOutput {
            content: serde_json::to_string(&ToolsOutput {
                tools: tool_definitions,
                groups,
                total_tools,
            })?,
            ..Default::default()
        })
    }
}

/// Arguments for [`ToolsGroups`].
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsGroupsArgs {
    /// Optional keyword filter. Empty or `*` lists every group.
    #[serde(default)]
    pub query: String,
}

/// One entry in the capability group directory returned by [`ToolsGroups`].
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolGroupSummary {
    /// Stable group id; pass it to `tools_select`'s `group` to expand the bundle.
    pub id: String,
    /// Human-facing group title.
    pub title: String,
    /// Concise summary of what this bundle of tools does.
    pub description: String,
    /// Number of member tools in the group.
    pub member_count: usize,
}

/// Output returned by [`ToolsGroups`].
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolGroupsOutput {
    /// The matching capability groups, without per-tool schemas.
    pub groups: Vec<ToolGroupSummary>,
    /// Total number of capability groups available this turn.
    #[serde(default)]
    pub total_groups: usize,
}

/// Lists the capability groups available to the model as a compact directory.
///
/// This is the top of the discovery funnel: the model sees which related tool
/// bundles exist (one MCP server, the filesystem tools, …) without paying for
/// every tool schema, then expands a chosen bundle with `tools_select`'s `group`
/// argument. Registered as a built-in agent with label "flash".
pub struct ToolsGroups;

impl Default for ToolsGroups {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolsGroups {
    /// Function name used when registering the group directory helper.
    pub const NAME: &'static str = TOOLS_GROUPS_NAME;

    /// Creates a group directory helper.
    pub fn new() -> Self {
        Self
    }
}

impl Agent<AgentCtx> for ToolsGroups {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "List the capability groups available this turn as a compact directory (no tool schemas). A group is a related bundle of tools from one source, such as a single MCP server or the filesystem tools. Use this first to see which bundles exist, then call `tools_select` with the `group` id to expand a bundle into its tool schemas.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional keyword to filter groups by id, title, or description. Use an empty string or `*` to list every group."
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let args: ToolsGroupsArgs = match serde_json::from_str(&prompt) {
            Ok(v) => v,
            Err(e) => {
                return Ok(AgentOutput {
                    content: format!("Invalid input: {e}"),
                    ..Default::default()
                });
            }
        };

        let groups = ctx.tool_groups();
        let total_groups = groups.len();
        let normalized = args.query.trim().to_lowercase();
        let summaries = groups
            .into_iter()
            .filter(|group| {
                normalized.is_empty()
                    || normalized == "*"
                    || group.id.to_lowercase().contains(&normalized)
                    || group.title.to_lowercase().contains(&normalized)
                    || group.description.to_lowercase().contains(&normalized)
            })
            .map(|group| ToolGroupSummary {
                id: group.id,
                title: group.title,
                description: group.description,
                member_count: group.members.len(),
            })
            .collect();

        Ok(AgentOutput {
            content: serde_json::to_string(&ToolGroupsOutput {
                groups: summaries,
                total_groups,
            })?,
            ..Default::default()
        })
    }
}

#[derive(Debug, Serialize)]
struct ToolItemRef<'a> {
    pub name: &'a str,
    pub description: &'a str,
}

impl<'a> From<&'a FunctionDefinition> for ToolItemRef<'a> {
    fn from(def: &'a FunctionDefinition) -> Self {
        Self {
            name: &def.name,
            description: &def.description,
        }
    }
}

async fn select_requested_names_with_model(
    ctx: &AgentCtx,
    candidates: &[FunctionDefinition],
    query: &str,
    limit: usize,
) -> Vec<String> {
    let tools = candidates.iter().map(ToolItemRef::from).collect::<Vec<_>>();
    let req = CompletionRequest {
        instructions: "You are selecting callable tools or agents for the next model turn. Choose only from the provided candidates. Prefer the smallest set that can plausibly help with the user intent. Return exact candidate names only. Never invent names. If no candidate is relevant, return an empty list.".to_string(),
        prompt: format!(
            "# User intent:\n{:?}\n\n---\n\n# Task:\n\nSelect up to {} callable names from the candidate list below. Return JSON only.\n\nCandidate callables:\n{}",
            query,
            limit,
            serde_json::to_string(&tools).unwrap_or_default()
        ),
        output_schema: Some(json!({
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["tools"],
            "additionalProperties": false
        })),
        temperature: Some(0.0),
        ..Default::default()
    };

    let output = match ctx.completion(req, Vec::new()).await {
        Ok(output) if output.failed_reason.is_none() => output,
        _ => return Vec::new(),
    };

    parse_selector_tool_names(output.content.trim())
}

fn parse_selector_tool_names(content: &str) -> Vec<String> {
    for candidate in json_candidates(content) {
        if let Ok(output) = serde_json::from_str::<ToolsSelectNamesOutput>(&candidate) {
            return output.tools;
        }
        if let Ok(output) = serde_json::from_str::<Vec<String>>(&candidate) {
            return output;
        }
    }

    Vec::new()
}

/// Extracts JSON candidates from the input string by applying various heuristics, such as stripping markdown code blocks and extracting JSON-like substrings. This increases the chances of successfully parsing the model output even when it is not perfectly formatted.
pub fn json_candidates(content: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    for candidate in [
        Some(content.to_string()),
        strip_markdown_code_block(content),
        extract_json_block(content, '{', '}'),
        extract_json_block(content, '[', ']'),
    ]
    .into_iter()
    .flatten()
    {
        if !candidate.is_empty() && seen.insert(candidate.clone()) {
            candidates.push(candidate);
        }
    }

    candidates
}

fn strip_markdown_code_block(content: &str) -> Option<String> {
    if !content.starts_with("```") || !content.ends_with("```") {
        return None;
    }

    let stripped = content
        .strip_prefix("```json")
        .or_else(|| content.strip_prefix("```JSON"))
        .or_else(|| content.strip_prefix("```"))?
        .strip_suffix("```")?
        .trim();

    Some(stripped.to_string())
}

fn extract_json_block(content: &str, open: char, close: char) -> Option<String> {
    let start = content.find(open)?;
    let end = content.rfind(close)?;
    if end < start {
        return None;
    }

    Some(content[start..=end].trim().to_string())
}

const NAME_EXACT_MATCH_BONUS: usize = 10_000;
const NAME_PREFIX_MATCH_BONUS: usize = 2_500;
const NAME_SUBSTRING_MATCH_BONUS: usize = 500;
const TOKEN_NAME_MATCH_WEIGHT: usize = 100;
const TOKEN_DESCRIPTION_MATCH_WEIGHT: usize = 10;

fn rank_search_items(
    items: &[FunctionDefinition],
    normalized_query: &str,
    normalized_tokens: &[(String, usize)],
    fallback: bool,
) -> Vec<String> {
    let mut candidates: Vec<(bool, usize, String)> = Vec::new();
    for item in items {
        // Returned names must stay the full callable names so that definition
        // lookups succeed; the routing prefix is only ignored for matching.
        let normalized_name = item.name.to_lowercase();
        let stripped_name = strip_prefix_ignore_ascii_case(&normalized_name, SUB_AGENT_PREFIX)
            .or_else(|| strip_prefix_ignore_ascii_case(&normalized_name, REMOTE_AGENT_PREFIX))
            .or_else(|| strip_prefix_ignore_ascii_case(&normalized_name, REMOTE_TOOL_PREFIX))
            .unwrap_or(&normalized_name);
        let normalized_description = item.description.to_lowercase();
        let mut score = 0usize;

        let exact_name_match =
            normalized_name == normalized_query || stripped_name == normalized_query;
        if exact_name_match {
            score += NAME_EXACT_MATCH_BONUS;
        } else if normalized_name.starts_with(normalized_query)
            || stripped_name.starts_with(normalized_query)
        {
            score += NAME_PREFIX_MATCH_BONUS;
        } else if normalized_name.contains(normalized_query) {
            score += NAME_SUBSTRING_MATCH_BONUS;
        }

        for (token, weight) in normalized_tokens {
            score +=
                weight * normalized_name.match_indices(token).count() * TOKEN_NAME_MATCH_WEIGHT;
            score += weight
                * normalized_description.match_indices(token).count()
                * TOKEN_DESCRIPTION_MATCH_WEIGHT;
        }

        if score > 0 {
            candidates.push((exact_name_match, score, normalized_name));
        } else if fallback {
            candidates.push((false, 0, normalized_name));
        }
    }

    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    candidates.into_iter().map(|(_, _, name)| name).collect()
}

/// Returns the capability groups that any of the `selected` definitions belong
/// to, so the discovery output can explain the bundle a tool came from and list
/// its sibling members. A group is included when at least one of its members is
/// among the selected definitions.
fn relevant_groups(groups: Vec<ToolGroup>, selected: &[FunctionDefinition]) -> Vec<ToolGroup> {
    if groups.is_empty() || selected.is_empty() {
        return Vec::new();
    }

    let selected_names: BTreeSet<String> = selected
        .iter()
        .map(|def| def.name.to_ascii_lowercase())
        .collect();

    groups
        .into_iter()
        .filter(|group| {
            group
                .members
                .iter()
                .any(|member| selected_names.contains(&member.to_ascii_lowercase()))
        })
        .collect()
}

fn select_requested_definitions(
    definitions: Vec<FunctionDefinition>,
    requested: &[String],
) -> Vec<FunctionDefinition> {
    let mut index = BTreeMap::new();
    for def in definitions {
        index
            .entry(def.name.to_ascii_lowercase())
            .or_insert_with(|| def);
    }

    let mut seen_requests = BTreeSet::new();
    let mut selected = Vec::new();

    for name in requested {
        let lookup = name.trim().to_ascii_lowercase();
        if lookup.is_empty() || !seen_requests.insert(lookup.clone()) {
            continue;
        }

        if let Some(def) = index.remove(&lookup) {
            selected.push(def);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use anda_core::{
        Agent, AgentOutput, BoxError, BoxFut, CompletionRequest, FunctionDefinition, Json,
        Resource, Tool, ToolGroup, ToolInput, ToolOutput, ToolProvider,
    };
    use candid::Principal;
    use serde::Deserialize;
    use serde_json::json;
    use std::sync::Arc;

    use super::*;
    use crate::{
        context::BaseCtx,
        engine::{Engine, EngineBuilder},
        model::{CompletionFeaturesDyn, Model, Models},
    };

    struct EchoTool;

    #[derive(Debug, Default, Deserialize)]
    struct EchoToolArgs {
        #[serde(default)]
        input: String,
    }

    impl Tool<BaseCtx> for EchoTool {
        type Args = EchoToolArgs;
        type Output = String;

        fn name(&self) -> String {
            "echo_tool".to_string()
        }

        fn description(&self) -> String {
            "Echoes the input back to the caller".to_string()
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: self.name(),
                description: self.description(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "input": { "type": "string" }
                    },
                    "required": ["input"],
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
            Ok(ToolOutput {
                output: args.input,
                ..Default::default()
            })
        }
    }

    struct HelpTool;

    #[derive(Debug, Default, Deserialize)]
    struct HelpToolArgs;

    impl Tool<BaseCtx> for HelpTool {
        type Args = HelpToolArgs;
        type Output = String;

        fn name(&self) -> String {
            "help_tool".to_string()
        }

        fn description(&self) -> String {
            "Useful when the user needs echo-oriented support".to_string()
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: self.name(),
                description: self.description(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }),
                strict: Some(true),
            }
        }

        async fn call(
            &self,
            _ctx: BaseCtx,
            _args: Self::Args,
            _resources: Vec<Resource>,
        ) -> Result<ToolOutput<Self::Output>, BoxError> {
            Ok(ToolOutput {
                output: "help".to_string(),
                ..Default::default()
            })
        }
    }

    struct EchoAgent;

    impl Agent<AgentCtx> for EchoAgent {
        fn name(&self) -> String {
            "echo_agent".to_string()
        }

        fn description(&self) -> String {
            "Handles echo tasks as an agent".to_string()
        }

        async fn run(
            &self,
            _ctx: AgentCtx,
            prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: prompt,
                ..Default::default()
            })
        }
    }

    /// A tool provider that exposes two tools bundled into one capability group.
    struct GroupedToolProvider;

    impl GroupedToolProvider {
        fn defs() -> Vec<FunctionDefinition> {
            ["grouped_read", "grouped_write"]
                .into_iter()
                .map(|name| FunctionDefinition {
                    name: name.to_string(),
                    description: format!("{name} from the test bundle"),
                    parameters: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    strict: Some(false),
                })
                .collect()
        }
    }

    impl ToolProvider<BaseCtx> for GroupedToolProvider {
        fn name(&self) -> String {
            "grouped".to_string()
        }

        fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
            match names {
                Some([]) => Vec::new(),
                Some(names) => Self::defs()
                    .into_iter()
                    .filter(|def| {
                        names
                            .iter()
                            .any(|name| name.eq_ignore_ascii_case(&def.name))
                    })
                    .collect(),
                None => Self::defs(),
            }
        }

        fn groups(&self) -> Vec<ToolGroup> {
            vec![ToolGroup {
                id: "grouped:bundle".to_string(),
                title: "Test bundle".to_string(),
                description: "A related bundle of tools".to_string(),
                instructions: Some("Read before write.".to_string()),
                members: vec!["grouped_read".to_string(), "grouped_write".to_string()],
            }]
        }

        fn call(
            &self,
            _ctx: BaseCtx,
            input: ToolInput<Json>,
        ) -> BoxFut<'_, Result<ToolOutput<Json>, BoxError>> {
            Box::pin(async move { Ok(ToolOutput::new(json!({ "called": input.name }))) })
        }
    }

    /// A provider that advertises a stale group member and a member shadowed by
    /// a static tool. Discovery should only expose the provider-backed member
    /// that is actually visible in the current callable set.
    struct OverlappingToolProvider;

    impl OverlappingToolProvider {
        fn defs() -> Vec<FunctionDefinition> {
            ["echo_tool", "provider_only"]
                .into_iter()
                .map(|name| FunctionDefinition {
                    name: name.to_string(),
                    description: format!("{name} from an overlapping provider"),
                    parameters: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    strict: Some(false),
                })
                .collect()
        }
    }

    impl ToolProvider<BaseCtx> for OverlappingToolProvider {
        fn name(&self) -> String {
            "overlap".to_string()
        }

        fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
            match names {
                Some([]) => Vec::new(),
                Some(names) => Self::defs()
                    .into_iter()
                    .filter(|def| {
                        names
                            .iter()
                            .any(|name| name.eq_ignore_ascii_case(&def.name))
                    })
                    .collect(),
                None => Self::defs(),
            }
        }

        fn groups(&self) -> Vec<ToolGroup> {
            vec![ToolGroup {
                id: "overlap:bundle".to_string(),
                title: "Overlap bundle".to_string(),
                description: "Contains visible, stale, and shadowed members".to_string(),
                members: vec![
                    "echo_tool".to_string(),
                    "provider_only".to_string(),
                    "missing_provider_member".to_string(),
                ],
                ..Default::default()
            }]
        }

        fn call(
            &self,
            _ctx: BaseCtx,
            input: ToolInput<Json>,
        ) -> BoxFut<'_, Result<ToolOutput<Json>, BoxError>> {
            Box::pin(async move { Ok(ToolOutput::new(json!({ "called": input.name }))) })
        }
    }

    struct SharedGroupProvider {
        provider_name: &'static str,
        tool_name: &'static str,
    }

    impl ToolProvider<BaseCtx> for SharedGroupProvider {
        fn name(&self) -> String {
            self.provider_name.to_string()
        }

        fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
            if names.is_some_and(|names| {
                !names
                    .iter()
                    .any(|name| name.eq_ignore_ascii_case(self.tool_name))
            }) {
                return Vec::new();
            }

            vec![FunctionDefinition {
                name: self.tool_name.to_string(),
                description: format!("{} from a duplicate group id provider", self.tool_name),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                strict: Some(false),
            }]
        }

        fn groups(&self) -> Vec<ToolGroup> {
            vec![ToolGroup {
                id: "shared:bundle".to_string(),
                title: "Shared bundle".to_string(),
                description: "A group id shared across providers".to_string(),
                members: vec![self.tool_name.to_string()],
                ..Default::default()
            }]
        }

        fn call(
            &self,
            _ctx: BaseCtx,
            input: ToolInput<Json>,
        ) -> BoxFut<'_, Result<ToolOutput<Json>, BoxError>> {
            Box::pin(async move { Ok(ToolOutput::new(json!({ "called": input.name }))) })
        }
    }

    #[derive(Clone, Debug)]
    struct SelectorCompleter {
        content: String,
    }

    impl CompletionFeaturesDyn for SelectorCompleter {
        fn model_name(&self) -> String {
            TOOLS_SELECT_NAME.to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            assert!(req.tools.is_empty());
            assert!(req.prompt.contains("Candidate callables:"));

            let content = self.content.clone();
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content,
                ..Default::default()
            })))
        }
    }

    async fn run_search(ctx: AgentCtx, args: ToolsSearchArgs) -> ToolsOutput {
        let output = ToolsSearch::new()
            .run(ctx, serde_json::to_string(&args).unwrap(), Vec::new())
            .await
            .unwrap();
        serde_json::from_str(&output.content).unwrap()
    }

    async fn run_select(ctx: AgentCtx, args: ToolsSelectArgs) -> ToolsOutput {
        let output = ToolsSelect::new()
            .run(ctx, serde_json::to_string(&args).unwrap(), Vec::new())
            .await
            .unwrap();
        serde_json::from_str(&output.content).unwrap()
    }

    async fn run_groups(ctx: AgentCtx, args: ToolsGroupsArgs) -> ToolGroupsOutput {
        let output = ToolsGroups::new()
            .run(ctx, serde_json::to_string(&args).unwrap(), Vec::new())
            .await
            .unwrap();
        serde_json::from_str(&output.content).unwrap()
    }

    async fn build_engine(builder: EngineBuilder) -> Engine {
        builder.build("echo_agent".to_string()).await.unwrap()
    }

    #[test]
    fn tools_search_and_tools_select_definitions_constrain_inputs() {
        let search_definition = ToolsSearch::new().definition();
        assert_eq!(
            search_definition.parameters["additionalProperties"],
            json!(false)
        );
        assert_eq!(
            search_definition.parameters["required"],
            json!(["query", "limit"])
        );

        let select_definition = ToolsSelect::new().definition();
        assert_eq!(
            select_definition.parameters["required"],
            json!(["tools", "query", "group", "limit"])
        );
        assert_eq!(
            select_definition.parameters["additionalProperties"],
            json!(false)
        );

        let groups_definition = ToolsGroups::new().definition();
        assert_eq!(groups_definition.parameters["required"], json!(["query"]));
        assert_eq!(
            groups_definition.parameters["additionalProperties"],
            json!(false)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_search_wildcard_returns_local_tools_and_agents() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let output = run_search(
            ctx,
            ToolsSearchArgs {
                query: "*".to_string(),
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "echo_tool",
                "echo_agent",
                "subagents_manager",
                "tools_groups",
                "tools_search",
                "tools_select"
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_search_prioritizes_name_matches_over_description_matches() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_tool(Arc::new(HelpTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let output = run_search(
            ctx,
            ToolsSearchArgs {
                query: "echo".to_string(),
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names.first().copied(), Some("echo_agent"));
        assert!(names.contains(&"help_tool"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_select_resolves_prefixed_agents_and_deduplicates_requests() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let output = run_select(
            ctx,
            ToolsSelectArgs {
                tools: vec![
                    "echo_agent".to_string(),
                    "echo_tool".to_string(),
                    "missing".to_string(),
                ],
                query: String::new(),
                group: String::new(),
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["echo_agent", "echo_tool"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_select_attaches_capability_group_for_provider_tools() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool_provider(Arc::new(GroupedToolProvider))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        // Selecting one bundle member surfaces the group so the model learns the
        // bundle's purpose, instructions, and the sibling it has not selected.
        let output = run_select(
            ctx,
            ToolsSelectArgs {
                tools: vec!["grouped_read".to_string()],
                query: String::new(),
                group: String::new(),
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["grouped_read"]);
        assert_eq!(output.groups.len(), 1);
        let group = &output.groups[0];
        assert_eq!(group.id, "grouped:bundle");
        assert_eq!(group.instructions.as_deref(), Some("Read before write."));
        assert_eq!(
            group.members,
            vec!["grouped_read".to_string(), "grouped_write".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_select_group_expands_all_members() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool_provider(Arc::new(GroupedToolProvider))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        // Naming the group expands every member's schema in one call.
        let output = run_select(
            ctx,
            ToolsSelectArgs {
                tools: Vec::new(),
                query: String::new(),
                group: "grouped:bundle".to_string(),
                limit: 0,
            },
        )
        .await;

        let mut names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["grouped_read", "grouped_write"]);
        assert_eq!(output.groups.len(), 1);
        assert_eq!(output.groups[0].id, "grouped:bundle");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tool_groups_hide_unavailable_provider_members() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_tool_provider(Arc::new(OverlappingToolProvider))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let directory = run_groups(
            ctx.clone(),
            ToolsGroupsArgs {
                query: "overlap".to_string(),
            },
        )
        .await;
        assert_eq!(directory.groups.len(), 1);
        assert_eq!(directory.groups[0].member_count, 1);

        let selected = run_select(
            ctx,
            ToolsSelectArgs {
                tools: Vec::new(),
                query: String::new(),
                group: "overlap:bundle".to_string(),
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = selected
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        assert_eq!(names, vec!["provider_only"]);
        assert_eq!(selected.groups.len(), 1);
        assert_eq!(
            selected.groups[0].members,
            vec!["provider_only".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn duplicate_tool_group_ids_merge_visible_members() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool_provider(Arc::new(SharedGroupProvider {
                    provider_name: "shared_a_provider",
                    tool_name: "shared_a_tool",
                }))
                .unwrap()
                .register_tool_provider(Arc::new(SharedGroupProvider {
                    provider_name: "shared_b_provider",
                    tool_name: "shared_b_tool",
                }))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let directory = run_groups(
            ctx.clone(),
            ToolsGroupsArgs {
                query: "shared".to_string(),
            },
        )
        .await;
        assert_eq!(directory.total_groups, 1);
        assert_eq!(directory.groups.len(), 1);
        assert_eq!(directory.groups[0].member_count, 2);

        let selected = run_select(
            ctx,
            ToolsSelectArgs {
                tools: Vec::new(),
                query: String::new(),
                group: "SHARED:BUNDLE".to_string(),
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = selected
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        assert_eq!(names, vec!["shared_a_tool", "shared_b_tool"]);
        assert_eq!(selected.groups.len(), 1);
        assert_eq!(
            selected.groups[0].members,
            vec!["shared_a_tool".to_string(), "shared_b_tool".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_groups_lists_capability_group_directory() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool_provider(Arc::new(GroupedToolProvider))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        // Wildcard lists the bundle as a compact entry without tool schemas.
        let output = run_groups(
            ctx.clone(),
            ToolsGroupsArgs {
                query: "*".to_string(),
            },
        )
        .await;
        assert_eq!(output.total_groups, 1);
        assert_eq!(output.groups.len(), 1);
        assert_eq!(output.groups[0].id, "grouped:bundle");
        assert_eq!(output.groups[0].member_count, 2);

        // A keyword that matches nothing filters the directory to empty, while
        // the total still reflects every available group.
        let filtered = run_groups(
            ctx,
            ToolsGroupsArgs {
                query: "no_such_group".to_string(),
            },
        )
        .await;
        assert!(filtered.groups.is_empty());
        assert_eq!(filtered.total_groups, 1);
    }

    #[test]
    fn relevant_groups_filters_by_membership_and_ignores_empty_inputs() {
        let groups = vec![
            ToolGroup {
                id: "a".to_string(),
                members: vec!["mcp_a_read".to_string(), "mcp_a_write".to_string()],
                ..Default::default()
            },
            ToolGroup {
                id: "b".to_string(),
                members: vec!["mcp_b_run".to_string()],
                ..Default::default()
            },
        ];
        let selected = vec![FunctionDefinition {
            // Membership matching is case-insensitive.
            name: "MCP_A_READ".to_string(),
            ..Default::default()
        }];

        let relevant = relevant_groups(groups.clone(), &selected);
        assert_eq!(relevant.len(), 1);
        assert_eq!(relevant[0].id, "a");

        assert!(relevant_groups(groups, &[]).is_empty());
        assert!(relevant_groups(Vec::new(), &selected).is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_select_prioritizes_explicit_tools_over_query_selection() {
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_tool(Arc::new(HelpTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let output = run_select(
            ctx,
            ToolsSelectArgs {
                tools: vec!["echo_tool".to_string()],
                query: "mirror my text".to_string(),
                group: String::new(),
                limit: 1,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["echo_tool"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_select_query_falls_back_to_ranked_candidates_on_invalid_selector_output() {
        let models = Arc::new(Models::default());
        models.set_model(Model::not_implemented());
        models.set(
            TOOLS_SELECT_NAME.to_string(),
            Model::with_completer(Arc::new(SelectorCompleter {
                content: "not json".to_string(),
            })),
        );

        let engine = build_engine(
            EngineBuilder::new()
                .with_models(models)
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_tool(Arc::new(HelpTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let output = run_select(
            ctx,
            ToolsSelectArgs {
                tools: Vec::new(),
                query: "echo".to_string(),
                group: String::new(),
                limit: 1,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["echo_agent"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_search_and_select_report_invalid_or_empty_inputs() {
        let _search = ToolsSearch::default();
        let _select = ToolsSelect::default();
        let engine = build_engine(
            EngineBuilder::new()
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let output = ToolsSearch::new()
            .run(ctx.clone(), "not json".to_string(), Vec::new())
            .await
            .unwrap();
        assert!(output.content.contains("Invalid input"));

        let output = run_search(
            ctx.clone(),
            ToolsSearchArgs {
                query: "   ".to_string(),
                limit: 4,
            },
        )
        .await;
        assert!(output.tools.is_empty());
        assert!(output.total_tools > 0);

        let output = ToolsSelect::new()
            .run(ctx.clone(), "not json".to_string(), Vec::new())
            .await
            .unwrap();
        assert!(output.content.contains("Invalid input"));

        let output = ToolsSelect::new()
            .run(
                ctx,
                serde_json::to_string(&ToolsSelectArgs::default()).unwrap(),
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(
            output
                .content
                .contains("one of `tools`, `query`, or `group`")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tools_select_query_covers_empty_candidates_short_queries_and_selector_success() {
        let models = Arc::new(Models::default());
        models.set_model(Model::not_implemented());
        models.set(
            TOOLS_SELECT_NAME.to_string(),
            Model::with_completer(Arc::new(SelectorCompleter {
                content: "```json\n{\"tools\":[\"help_tool\"]}\n```".to_string(),
            })),
        );
        let engine = build_engine(
            EngineBuilder::new()
                .with_models(models)
                .register_tool(Arc::new(EchoTool))
                .unwrap()
                .register_tool(Arc::new(HelpTool))
                .unwrap()
                .register_agent(Arc::new(EchoAgent), None)
                .unwrap(),
        )
        .await;
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                TOOLS_SELECT_NAME,
                Default::default(),
            )
            .unwrap();
        let selector = ToolsSelect::new();

        let selected = selector
            .select_requested_definitions_by_query(
                &ctx,
                Vec::new(),
                &ToolsSelectArgs {
                    query: "echo".to_string(),
                    limit: 0,
                    ..Default::default()
                },
            )
            .await;
        assert!(selected.is_empty());

        let definitions = vec![EchoTool.definition(), HelpTool.definition()];
        let selected = selector
            .select_requested_definitions_by_query(
                &ctx,
                definitions.clone(),
                &ToolsSelectArgs {
                    query: "he".to_string(),
                    limit: 1,
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(selected[0].name, "help_tool");

        let selected = selector
            .select_requested_definitions_by_query(
                &ctx,
                definitions,
                &ToolsSelectArgs {
                    query: "echo support".to_string(),
                    limit: 2,
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(selected[0].name, "help_tool");
    }

    #[test]
    fn selector_json_candidate_parsing_ranking_and_deduping_are_stable() {
        assert_eq!(
            parse_selector_tool_names("```JSON\n[\"echo_tool\", \"help_tool\"]\n```"),
            vec!["echo_tool", "help_tool"]
        );
        assert_eq!(
            parse_selector_tool_names("prefix {\"tools\":[\"echo_tool\"]} suffix"),
            vec!["echo_tool"]
        );
        assert!(parse_selector_tool_names("not json").is_empty());

        let candidates = json_candidates("prefix [\"echo_tool\"] suffix");
        assert!(candidates.contains(&"[\"echo_tool\"]".to_string()));

        let definitions = vec![
            FunctionDefinition {
                name: "alpha".to_string(),
                description: "first".to_string(),
                ..Default::default()
            },
            FunctionDefinition {
                name: "my_echo_tool".to_string(),
                description: "second".to_string(),
                ..Default::default()
            },
        ];
        assert_eq!(
            rank_search_items(&definitions, "alpha", &[], false)[0],
            "alpha"
        );
        assert_eq!(
            rank_search_items(&definitions, "echo", &[], false),
            vec!["my_echo_tool"]
        );

        let selected = select_requested_definitions(
            vec![EchoTool.definition()],
            &[
                String::new(),
                "echo_tool".to_string(),
                "echo_tool".to_string(),
                "missing".to_string(),
            ],
        );
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "echo_tool");
    }

    #[test]
    fn rank_search_items_keeps_prefixed_names_resolvable() {
        let definitions = vec![
            FunctionDefinition {
                name: "RT_remote_lookup".to_string(),
                description: "Remote lookup tool".to_string(),
                ..Default::default()
            },
            FunctionDefinition {
                name: "SA_researcher".to_string(),
                description: "Research subagent".to_string(),
                ..Default::default()
            },
            FunctionDefinition {
                name: "RA_remote_chat".to_string(),
                description: "Remote chat agent".to_string(),
                ..Default::default()
            },
        ];

        // The routing prefix is ignored for matching, and the ranked names stay
        // resolvable through select_requested_definitions.
        let ranked = rank_search_items(&definitions, "researcher", &[], false);
        assert_eq!(ranked, vec!["sa_researcher"]);
        let ranked = rank_search_items(&definitions, "remote_lookup", &[], false);
        assert_eq!(ranked.first().map(String::as_str), Some("rt_remote_lookup"));
        let ranked = rank_search_items(&definitions, "remote_chat", &[], false);
        assert_eq!(ranked.first().map(String::as_str), Some("ra_remote_chat"));

        let ranked = rank_search_items(&definitions, "remote_lookup", &[], false);
        let selected = select_requested_definitions(definitions, &ranked);
        assert_eq!(selected[0].name, "RT_remote_lookup");
    }
}
