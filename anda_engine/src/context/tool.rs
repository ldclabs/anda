use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest,
    FunctionDefinition, Resource,
};
use anda_db_tfs::{TokenizerChain, collect_tokens, jieba_tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

use crate::context::{AgentCtx, REMOTE_AGENT_PREFIX, REMOTE_TOOL_PREFIX, SUB_AGENT_PREFIX};

pub const TOOLS_SEARCH_NAME: &str = "tools_search";
pub const TOOLS_SELECT_NAME: &str = "tools_select";

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsSearchArgs {
    /// Search terms, or `*` to enumerate every available callable name.
    pub query: String,
    /// Maximum number of results to return. Defaults to `10`.
    #[serde(default)]
    pub limit: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsOutput {
    /// Matching callable definitions returned as tool-output context.
    ///
    /// These definitions are not dynamically inserted into [`CompletionRequest::tools`].
    pub tools: Vec<FunctionDefinition>,
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
    pub const NAME: &'static str = TOOLS_SEARCH_NAME;

    pub fn new() -> Self {
        let tokenizer = jieba_tokenizer();
        Self { tokenizer }
    }

    pub fn search(&self, candidates: &[FunctionDefinition], args: &ToolsSearchArgs) -> ToolsOutput {
        let normalized_query = args.query.trim().to_lowercase();
        let tools: Vec<FunctionDefinition> = candidates.to_vec();

        let total_tools = tools.len();
        if normalized_query == "*" {
            return ToolsOutput { tools, total_tools };
        }

        let normalized_tokens: Vec<(String, usize)> =
            collect_tokens(&mut self.tokenizer.clone(), &normalized_query, None)
                .into_iter()
                .collect();

        let mut tools_name =
            rank_search_items(&tools, &normalized_query, &normalized_tokens, false);
        tools_name.truncate(if args.limit == 0 { 10 } else { args.limit });
        let tools = select_requested_definitions(tools, &tools_name);
        ToolsOutput { tools, total_tools }
    }
}

impl Agent<AgentCtx> for ToolsSearch {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Search callable tools and agents by keyword. Returns full callable schemas in this tool output; after a schema is returned, call that tool/agent directly instead of searching again."
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
                })?,
                ..Default::default()
            });
        }
        let rt = self.search(&definitions, &args);
        Ok(AgentOutput {
            content: serde_json::to_string(&rt)?,
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolsSelectArgs {
    /// Callable names whose schemas should be returned for direct calls.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Natural-language intent used to select tools when exact names are unknown.
    #[serde(default)]
    pub query: String,
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
    pub const NAME: &'static str = TOOLS_SELECT_NAME;

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
        "Select callable tools or agents and return full schemas in this tool output for direct tool calls. Use exact names via `tools`; use `query` only when exact names are unknown. Do not call tools_select again for the same returned tools.".to_string()
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
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of resolved callables to return. Defaults to `5`, and is capped at `16`."
                    }
                },
                "required": ["tools", "query", "limit"],
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

        if args.tools.is_empty() && args.query.trim().is_empty() {
            return Ok(AgentOutput {
                content: "Invalid input: either `tools` or `query` must be provided".to_string(),
                ..Default::default()
            });
        }

        let definitions = ctx.definitions(None).await;
        let total_tools = definitions.len();
        let tool_definitions = if !args.tools.is_empty() {
            select_requested_definitions(definitions, &args.tools)
        } else {
            self.select_requested_definitions_by_query(&ctx, definitions, &args)
                .await
        };

        Ok(AgentOutput {
            content: serde_json::to_string(&ToolsOutput {
                tools: tool_definitions,
                total_tools,
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
        let normalized_name = item
            .name
            .strip_prefix(SUB_AGENT_PREFIX)
            .unwrap_or(&item.name)
            .strip_prefix(REMOTE_AGENT_PREFIX)
            .unwrap_or(&item.name)
            .strip_prefix(REMOTE_TOOL_PREFIX)
            .unwrap_or(&item.name)
            .to_lowercase();
        let normalized_description = item.description.to_lowercase();
        let mut score = 0usize;

        let exact_name_match = normalized_name == normalized_query;
        if exact_name_match {
            score += NAME_EXACT_MATCH_BONUS;
        } else if normalized_name.starts_with(normalized_query) {
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
        Agent, AgentOutput, BoxError, CompletionRequest, FunctionDefinition, Resource, Tool,
        ToolOutput,
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
            json!(["tools", "query", "limit"])
        );
        assert_eq!(
            select_definition.parameters["additionalProperties"],
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
                limit: 0,
            },
        )
        .await;

        let names: Vec<&str> = output.tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["echo_agent", "echo_tool"]);
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
        assert!(output.content.contains("either `tools` or `query`"));
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
}
