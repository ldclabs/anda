//! Agent Context Implementation
//!
//! This module provides the core implementation of the Agent context ([`AgentCtx`]) which serves as
//! the primary execution environment for agents in the Anda system. The context provides:
//!
//! - Access to AI models for completions and embeddings;
//! - Tool execution capabilities;
//! - Agent-to-agent communication;
//! - Cryptographic operations;
//! - Storage and caching facilities;
//! - Canister interaction capabilities;
//! - HTTP communication features.
//!
//! The [`AgentCtx`] implements multiple traits that provide different sets of functionality:
//! - [`AgentContext`]: Core agent operations and tool/agent management;
//! - [`CompletionFeatures`]: AI model completion capabilities;
//! - [`EmbeddingFeatures`]: Text embedding generation;
//! - [`StateFeatures`]: Context state management;
//! - [`KeysFeatures`]: Cryptographic key operations;
//! - [`StoreFeatures`]: Persistent storage operations;
//! - [`CacheFeatures`]: Caching mechanisms;
//! - [`CanisterCaller`]: Canister interaction capabilities;
//! - [`HttpFeatures`]: HTTPs communication features.
//!
//! The context is designed to be hierarchical, allowing creation of child contexts for specific
//! agents or tools while maintaining access to the core functionality.

use anda_core::{
    Agent, AgentContext, AgentInput, AgentOutput, AgentSet, BaseContext, BoxError, BoxPinFut,
    CacheExpiry, CacheFeatures, CacheStoreFeatures, CancellationToken, CanisterCaller,
    CompletionFeatures, CompletionRequest, ContentPart, FunctionDefinition, HttpFeatures, Json,
    KeysFeatures, Message, ModelEffort, ObjectMeta, Path, PutMode, PutResult, RequestMeta,
    Resource, StateFeatures, StoreFeatures, ToolCall, ToolGroup, ToolInput, ToolOutput,
    ToolProviderSet, ToolSet, Usage,
};
use bytes::Bytes;
use candid::{CandidType, Principal, utils::ArgumentEncoder};
use futures_util::Stream;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use super::{
    base::BaseCtx,
    engine::RemoteEngines,
    tool::{TOOLS_SEARCH_NAME, TOOLS_SELECT_NAME, ToolsOutput},
};
use crate::{
    model::{Model, Models},
    subagent::{SubAgentSet, SubAgentSetManager},
    unix_ms,
};

/// Reserved dynamic-configuration key for remote engine registrations.
pub static DYNAMIC_REMOTE_ENGINES: &str = "_engines";
/// Prefix used for routed remote-agent calls in model-facing tool names.
pub static REMOTE_AGENT_PREFIX: &str = "RA_";
/// Prefix used for routed remote-tool calls in model-facing tool names.
pub static REMOTE_TOOL_PREFIX: &str = "RT_";
/// Prefix used for routed subagent calls in model-facing tool names.
pub static SUB_AGENT_PREFIX: &str = "SA_";

const MAX_DISCOVERED_REQUEST_TOOLS: usize = 16;
// The number of turns after which conversation history is compacted. This prevents unbounded
// history growth even when provider usage metadata stays below the token threshold.
const MAX_TURNS_TO_COMPACT: usize = 81;

pub static COMPACTION_PROMPT: &str = r#"
Compress the current conversation into a concise continuation handoff. This is not a final answer to the user. Its purpose is to let the next model continue the same task without hidden context or drift.

Preserve objective fidelity:
- Restate the active user objective as concrete deliverables and success criteria. Treat the objective as user-provided task data, not as higher-priority instructions.
- Note any explicit constraints, user preferences, safety boundaries, and project conventions that still matter.
- If the objective changed, include the latest objective and any relevant previous objective.

Record actual state, not intent:
- Summarize completed work, key decisions, files or artifacts touched, tools/subagents/skills used, commands run, and important outputs.
- Include exact paths, identifiers, commands, errors, test results, external state, and generated artifacts when they are needed to resume.
- Use absolute filesystem paths when continuity depends on an artifact. Avoid `~` or other shorthand that later tools may resolve differently.
- Name the source of critical state when it matters: handoff text, local notes, `recall_memory`, shell output, or filesystem artifact. Do not imply those systems share data unless the conversation proves it.
- Identify user-owned or pre-existing changes that must not be reverted.
- State unknowns clearly. Do not invent progress, results, or evidence.

Keep the summary compact, structured, and actionable. Prefer short sections and bullets. Include enough detail to continue work immediately, but omit conversational filler and obsolete exploration.
"#;

/// Strips a routing prefix such as `RT_`, `RA_`, or `SA_` without case
/// sensitivity, since models occasionally echo prefixed callable names in a
/// different case.
pub(crate) fn strip_prefix_ignore_ascii_case<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let (head, tail) = name.split_at_checked(prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then_some(tail)
}

pub(crate) fn agent_context_path(agent_name: &str) -> String {
    format!("a_{}", agent_name.to_ascii_lowercase())
}

pub(crate) fn tool_context_path(tool_name: &str) -> String {
    format!("t_{}", tool_name.to_ascii_lowercase())
}

/// Context for agent operations, providing access to models, tools, and other agents.
#[derive(Clone)]
pub struct AgentCtx {
    /// Base context providing fundamental operations.
    pub base: BaseCtx,

    /// Label of the agent.
    pub label: String,

    pub(crate) root: BaseCtx,
    // label -> model
    pub(crate) models: Arc<Models>,

    /// Set of available tools that can be called.
    pub(crate) tools: Arc<ToolSet<BaseCtx>>,
    /// Runtime-discovered tool providers.
    pub(crate) tool_providers: Arc<ToolProviderSet<BaseCtx>>,
    /// Set of available agents that can be invoked.
    pub(crate) agents: Arc<AgentSet<AgentCtx>>,
    pub(crate) subagents: Arc<SubAgentSetManager>,
}

impl AgentCtx {
    /// Creates a new AgentCtx instance.
    ///
    /// # Arguments
    /// * `base` - Base context.
    /// * `model` - AI model instance.
    /// * `tools` - Set of available tools.
    /// * `agents` - Set of available agents.
    pub(crate) fn new(
        base: BaseCtx,
        models: Arc<Models>,
        tools: Arc<ToolSet<BaseCtx>>,
        tool_providers: Arc<ToolProviderSet<BaseCtx>>,
        agents: Arc<AgentSet<AgentCtx>>,
        subagents: Arc<SubAgentSetManager>,
    ) -> Self {
        Self {
            base: base.clone(),
            label: String::new(),
            root: base,
            models,
            tools,
            tool_providers,
            agents,
            subagents,
        }
    }

    /// Creates a child context for a specific agent.
    ///
    /// # Arguments
    /// * `agent_name` - Name of the agent to create context for.
    pub fn child(&self, agent_name: &str, agent_label: &str) -> Result<Self, BoxError> {
        Ok(Self {
            base: self
                .base
                .child(agent_name.to_string(), agent_context_path(agent_name))?,
            label: agent_label.to_string(),
            root: self.root.clone(),
            models: self.models.clone(),
            tools: self.tools.clone(),
            tool_providers: self.tool_providers.clone(),
            agents: self.agents.clone(),
            subagents: self.subagents.clone(),
        })
    }

    /// Creates a child base context for a specific tool.
    ///
    /// # Arguments
    /// * `tool_name` - Name of the tool to create context for.
    pub fn child_base(&self, tool_name: &str) -> Result<BaseCtx, BoxError> {
        self.base
            .child(self.base.agent.clone(), tool_context_path(tool_name))
    }

    /// Creates a child context with caller and meta information.
    ///
    /// # Arguments
    /// * `caller` - caller principal from request.
    /// * `agent_name` - Name of the agent to run.
    /// * `meta` - Metadata from request.
    pub(crate) fn child_with(
        &self,
        caller: Principal,
        agent_name: &str,
        agent_label: &str,
        meta: RequestMeta,
    ) -> Result<Self, BoxError> {
        Ok(Self {
            base: self.base.child_with(
                caller,
                agent_name.to_string(),
                agent_context_path(agent_name),
                meta,
            )?,
            label: agent_label.to_string(),
            root: self.root.clone(),
            models: self.models.clone(),
            tools: self.tools.clone(),
            tool_providers: self.tool_providers.clone(),
            agents: self.agents.clone(),
            subagents: self.subagents.clone(),
        })
    }

    /// Creates a child base context with caller and meta information.
    ///
    /// # Arguments
    /// * `caller` - caller principal from request.
    /// * `tool_name` - Name of the tool to call.
    /// * `meta` - Metadata from request.
    pub(crate) fn child_base_with(
        &self,
        caller: Principal,
        agent_name: &str,
        tool_name: &str,
        meta: RequestMeta,
    ) -> Result<BaseCtx, BoxError> {
        self.base.child_with(
            caller,
            agent_name.to_string(),
            tool_context_path(tool_name),
            meta,
        )
    }

    /// Clones the context with a new caller principal.
    pub fn with_caller(&self, caller: Principal) -> Self {
        Self {
            base: self.base.with_caller(caller),
            ..self.clone()
        }
    }

    pub(crate) fn has_tool_lowercase(&self, lowercase_name: &str) -> bool {
        self.tools.contains_lowercase(lowercase_name)
            || self.tool_providers.contains_lowercase(lowercase_name)
    }

    /// Returns the capability groups exposed to discovery.
    ///
    /// Groups bundle related callables — static tools that declare a group (for
    /// example the filesystem or memory tools), agents that declare a group (for
    /// example the media-understanding agents), and provider-backed tools (for
    /// example all tools from one MCP server) — so the discovery helpers can
    /// tell the model the callables are related and how to combine them.
    pub fn tool_groups(&self) -> Vec<ToolGroup> {
        let mut groups = BTreeMap::new();
        let mut visible_names = BTreeSet::new();

        let static_names: BTreeMap<String, String> = self
            .tools
            .definitions(None)
            .into_iter()
            .map(|definition| {
                (
                    definition.name.to_ascii_lowercase(),
                    definition.name.clone(),
                )
            })
            .collect();
        visible_names.extend(static_names.keys().cloned());
        for group in self.tools.groups() {
            merge_visible_group(&mut groups, group, &static_names);
        }

        let agent_names: BTreeMap<String, String> = self
            .agents
            .definitions(None)
            .into_iter()
            .filter_map(|definition| {
                let lowercase = definition.name.to_ascii_lowercase();
                visible_names
                    .insert(lowercase.clone())
                    .then_some((lowercase, definition.name))
            })
            .collect();
        for group in self.agents.groups() {
            merge_visible_group(&mut groups, group, &agent_names);
        }

        for provider in self.tool_providers.set.values() {
            let mut provider_names = BTreeMap::new();
            for definition in provider.definitions(None) {
                let lowercase = definition.name.to_ascii_lowercase();
                if visible_names.insert(lowercase.clone()) {
                    provider_names.insert(lowercase, definition.name);
                }
            }

            for group in provider.groups() {
                merge_visible_group(&mut groups, group, &provider_names);
            }
        }

        groups.into_values().collect()
    }

    /// Creates a completion runner for iterative processing of completion requests.
    pub fn completion_iter(
        self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> CompletionRunner {
        let label = req.model.as_deref().unwrap_or(&self.label);
        let model = self
            .models
            .resolve(label)
            .unwrap_or_else(Model::not_implemented);
        CompletionRunner {
            ctx: self,
            req,
            model,
            resources,
            chat_history: Vec::new(),
            tool_calls: Vec::new(),
            total_usage: Usage::default(),
            current_usage: Usage::default(),
            artifacts: Vec::new(),
            steering_message: Vec::new(),
            follow_up_message: VecDeque::new(),
            implicit_context: None,
            pending_tool_calls: Vec::new(),
            pending_tool_call_raw_history_start: None,
            tools_usage: HashMap::new(),
            last_output: None,
            discovered_tools: BTreeMap::new(),
            discovery_selection_counts: BTreeMap::new(),
            merge_discovered_tools: None,
            done: false,
            unbound: false,
            turns: 0,
        }
    }

    /// Creates a completion stream for processing of completion requests.
    pub fn completion_stream(
        self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> CompletionStream {
        CompletionStream::new(self.completion_iter(req, resources))
    }
}

impl CacheStoreFeatures for AgentCtx {}

impl AgentContext for AgentCtx {
    /// Retrieves definitions for available tools.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific tool names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools.
    fn tool_definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        let mut definitions = self.tools.definitions(names);
        let mut seen: BTreeSet<String> =
            BTreeSet::from_iter(definitions.iter().map(|d| d.name.to_ascii_lowercase()));
        for definition in self.tool_providers.definitions(names) {
            if seen.insert(definition.name.to_ascii_lowercase()) {
                definitions.push(definition);
            }
        }
        definitions
    }

    /// Retrieves definitions for available tools in the remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint;
    /// * `names` - Optional filter for specific tool names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools.
    async fn remote_tool_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[String]>,
    ) -> Result<Vec<FunctionDefinition>, BoxError> {
        if let Some(names) = names
            && names.is_empty()
        {
            return Ok(Vec::new());
        }

        let mut defs = self.base.remote.tool_definitions(endpoint, names);
        let mut seen: BTreeSet<String> =
            BTreeSet::from_iter(defs.iter().map(|d| d.name.to_ascii_lowercase()));
        if let Ok((engines, _)) = self
            .root
            .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
            .await
        {
            for def in engines.tool_definitions(endpoint, names) {
                if seen.insert(def.name.to_ascii_lowercase()) {
                    defs.push(def);
                }
            }
        }

        Ok(defs
            .into_iter()
            .map(|d| d.name_with_prefix(REMOTE_TOOL_PREFIX))
            .collect())
    }

    /// Extracts resources from the provided list based on the tool's supported tags.
    async fn select_tool_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if let Some(name) = strip_prefix_ignore_ascii_case(prefixed_name, REMOTE_TOOL_PREFIX) {
            let res = self.base.remote.select_tool_resources(name, resources);
            if !res.is_empty() {
                return res;
            }

            if let Ok((engines, _)) = self
                .root
                .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
                .await
            {
                return engines.select_tool_resources(name, resources);
            }
        }

        if self.tools.contains(prefixed_name) {
            return self.tools.select_resources(prefixed_name, resources);
        }

        self.tool_providers
            .select_resources(prefixed_name, resources)
    }

    /// Retrieves definitions for available agents.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific agent names;
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents.
    fn agent_definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        if let Some(names) = names
            && names.is_empty()
        {
            return Vec::new();
        }

        let mut defs = self.agents.definitions(names);
        defs.extend(
            self.subagents
                .definitions(names)
                .into_iter()
                .map(|d| d.name_with_prefix(SUB_AGENT_PREFIX)),
        );
        defs
    }

    /// Retrieves definitions for available agents in the remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint;
    /// * `names` - Optional filter for specific agent names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents.
    async fn remote_agent_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[String]>,
    ) -> Result<Vec<FunctionDefinition>, BoxError> {
        if let Some(names) = names
            && names.is_empty()
        {
            return Ok(Vec::new());
        }

        let mut defs = self.base.remote.agent_definitions(endpoint, names);
        if let Ok((engines, _)) = self
            .root
            .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
            .await
        {
            for def in engines.agent_definitions(endpoint, names) {
                if !defs.iter().any(|d| d.name == def.name) {
                    defs.push(def);
                }
            }
        }

        Ok(defs
            .into_iter()
            .map(|d| d.name_with_prefix(REMOTE_AGENT_PREFIX))
            .collect())
    }

    /// Extracts resources from the provided list based on the agent's supported tags.
    async fn select_agent_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if let Some(name) = strip_prefix_ignore_ascii_case(prefixed_name, REMOTE_AGENT_PREFIX) {
            let res = self.base.remote.select_agent_resources(name, resources);
            if !res.is_empty() {
                return res;
            }

            if let Ok((engines, _)) = self
                .root
                .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
                .await
            {
                return engines.select_agent_resources(name, resources);
            }
        }

        if let Some(prefix) = strip_prefix_ignore_ascii_case(prefixed_name, SUB_AGENT_PREFIX) {
            let res = self.subagents.select_resources(prefix, resources);
            if !res.is_empty() {
                return res;
            }
        }

        self.agents.select_resources(prefixed_name, resources)
    }

    /// Retrieves definitions for available tools and agents, including those from remote engines.
    async fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        if let Some(names) = names
            && names.is_empty()
        {
            return Vec::new();
        }

        let mut definitions = self.tool_definitions(names);
        definitions.extend(self.agent_definitions(names));
        if let Ok(remote) = self.remote_tool_definitions(None, names).await {
            definitions.extend(remote);
        }
        if let Ok(remote) = self.remote_agent_definitions(None, names).await {
            definitions.extend(remote);
        }

        definitions
    }

    /// Executes a tool call with the given arguments
    ///
    /// # Arguments
    /// * `name` - Name of the tool to call
    /// * `args` - Arguments for the tool call as a JSON string
    ///
    /// # Returns
    /// Tuple containing the result string and a boolean indicating if further processing is needed
    async fn tool_call(
        &self,
        mut input: ToolInput<Json>,
    ) -> Result<(ToolOutput<Json>, Option<Principal>), BoxError> {
        if let Some(name) = strip_prefix_ignore_ascii_case(&input.name, REMOTE_TOOL_PREFIX) {
            // find registered remote tool and call it
            if let Some((id, endpoint, tool_name)) = self.base.remote.get_tool_endpoint(name) {
                input.name = tool_name;
                input.meta = Some(self.base.self_meta(id));
                return self
                    .base
                    .remote_tool_call(&endpoint, input)
                    .await
                    .map(|output| (output, Some(id)));
            }

            // find dynamic remote tool and call it
            if let Ok((engines, _)) = self
                .root
                .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
                .await
                && let Some((id, endpoint, tool_name)) = engines.get_tool_endpoint(name)
            {
                input.name = tool_name;
                input.meta = Some(self.base.self_meta(id));
                return self
                    .base
                    .remote_tool_call(&endpoint, input)
                    .await
                    .map(|output| (output, Some(id)));
            }
        }

        let ctx = self.child_base(&input.name)?;
        if let Some(tool) = self.tools.get(&input.name) {
            return tool
                .call(ctx, input.args, input.resources)
                .await
                .map(|output| (output, None));
        }

        self.tool_providers
            .call(ctx, input)
            .await
            .map(|output| (output, None))
    }

    /// Runs a local agent.
    ///
    /// # Arguments
    /// * `args` - Tool input arguments, [`AgentInput`].
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    fn agent_run(
        self,
        mut input: AgentInput,
    ) -> impl Future<Output = Result<(AgentOutput, Option<Principal>), BoxError>> + Send {
        let ctx = self;
        Box::pin(async move {
            // Prefixed names route to remote engines or subagents first; on a
            // miss they fall through to the local agent lookup so that local
            // agents whose names happen to start with a routing prefix stay
            // reachable.
            if let Some(name) = strip_prefix_ignore_ascii_case(&input.name, REMOTE_AGENT_PREFIX) {
                if let Some((id, endpoint, agent_name)) = ctx.base.remote.get_agent_endpoint(name) {
                    input.name = agent_name;
                    input.meta = Some(ctx.base.self_meta(id));
                    return ctx
                        .remote_agent_run(&endpoint, input)
                        .await
                        .map(|output| (output, Some(id)));
                }

                if let Ok((engines, _)) = ctx
                    .root
                    .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
                    .await
                    && let Some((id, endpoint, agent_name)) = engines.get_agent_endpoint(name)
                {
                    input.name = agent_name;
                    input.meta = Some(ctx.base.self_meta(id));
                    return ctx
                        .remote_agent_run(&endpoint, input)
                        .await
                        .map(|output| (output, Some(id)));
                }
            }

            if let Some(name) = strip_prefix_ignore_ascii_case(&input.name, SUB_AGENT_PREFIX) {
                let name = name.to_ascii_lowercase();
                if let Some(agent) = ctx.subagents.get_lowercase(&name) {
                    let child = ctx.child(&name, &name)?;
                    return agent
                        .run(child, input.prompt, input.resources)
                        .await
                        .map(|output| (output, None));
                }
            }

            let name = input.name.to_ascii_lowercase();
            if let Some(agent) = ctx.agents.get(&name) {
                let child = ctx.child(&name, agent.label())?;
                agent
                    .run(child, input.prompt, input.resources)
                    .await
                    .map(|output| (output, None))
            } else {
                Err(format!("agent {} not found", name).into())
            }
        })
    }

    /// Runs a remote agent via HTTP RPC.
    ///
    /// # Arguments
    /// * `endpoint` - Remote endpoint URL;
    /// * `args` - Tool input arguments, [`AgentInput`]. The `meta` field will be set to the current agent's metadata.
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    async fn remote_agent_run(
        &self,
        endpoint: &str,
        mut args: AgentInput,
    ) -> Result<AgentOutput, BoxError> {
        let target = self
            .base
            .remote
            .get_id_by_endpoint(endpoint)
            .ok_or_else(|| format!("remote engine endpoint {} not found", endpoint))?;
        let meta = self.base.self_meta(target);
        args.meta = Some(meta);
        let output: AgentOutput = self
            .https_signed_rpc(endpoint, "agent_run", &(&args,))
            .await?;

        Ok(output)
    }
}

impl CompletionFeatures for AgentCtx {
    fn model_name(&self) -> String {
        self.models
            .get_model()
            .unwrap_or_else(Model::not_implemented)
            .model_name()
    }

    /// Executes a completion request with automatic tool call handling.
    ///
    /// This method handles the completion request in a loop, automatically executing
    /// any tool calls that are returned by the model and feeding their results back
    /// into the model until no more tool calls need to be processed.
    ///
    /// # Arguments
    /// * `req` - [`CompletionRequest`] containing the input parameters;
    /// * `resources` - Optional list of resources to use for tool calls.
    ///
    /// # Returns
    /// [`AgentOutput`] containing the final completion result.
    ///
    /// # Process Flow
    /// 1. Makes initial completion request to the model;
    /// 2. If tool calls are returned:
    ///    - Executes each tool call;
    ///    - Adds tool results to the chat history;
    ///    - Repeats the completion with updated history;
    /// 3. Returns final result when no more tool calls need processing.
    fn completion(
        &self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send {
        let ctx = self.clone();
        Box::pin(async move {
            let mut runner = ctx.completion_iter(req, resources);
            let mut last: Option<AgentOutput> = None;

            while let Some(step) = runner.next().await? {
                if step.failed_reason.is_some() {
                    return Ok(step);
                }
                last = Some(step);
            }

            last.ok_or_else(|| "completion runner returned no output".into())
        })
    }
}

impl BaseContext for AgentCtx {
    /// Executes a remote tool call via HTTP RPC.
    ///
    /// # Arguments
    /// * `endpoint` - Remote endpoint URL;
    /// * `args` - Tool input arguments, [`ToolInput`].
    ///
    /// # Returns
    /// [`ToolOutput`] containing the final result.
    async fn remote_tool_call(
        &self,
        endpoint: &str,
        args: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        self.base.remote_tool_call(endpoint, args).await
    }
}

impl StateFeatures for AgentCtx {
    fn engine_id(&self) -> &Principal {
        &self.base.id
    }

    fn engine_name(&self) -> &str {
        &self.base.name
    }

    fn caller(&self) -> &Principal {
        &self.base.caller
    }

    fn meta(&self) -> &RequestMeta {
        &self.base.meta
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.base.cancellation_token.clone()
    }

    fn time_elapsed(&self) -> Duration {
        self.base.time_elapsed()
    }
}

impl KeysFeatures for AgentCtx {
    /// Derives a 256-bit AES-GCM key from the given derivation path.
    async fn a256gcm_key(&self, derivation_path: Vec<Vec<u8>>) -> Result<[u8; 32], BoxError> {
        self.base.a256gcm_key(derivation_path).await
    }

    /// Signs a message using Ed25519 signature scheme from the given derivation path.
    async fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .ed25519_sign_message(derivation_path, message)
            .await
    }

    /// Verifies an Ed25519 signature from the given derivation path.
    async fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.base
            .ed25519_verify(derivation_path, message, signature)
            .await
    }

    /// Gets the public key for Ed25519 from the given derivation path.
    async fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 32], BoxError> {
        self.base.ed25519_public_key(derivation_path).await
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path.
    async fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .secp256k1_sign_message_bip340(derivation_path, message)
            .await
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path.
    async fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.base
            .secp256k1_verify_bip340(derivation_path, message, signature)
            .await
    }

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path.
    /// The message will be hashed with SHA-256 before signing.
    async fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .secp256k1_sign_message_ecdsa(derivation_path, message)
            .await
    }

    /// Signs a message hash using Secp256k1 ECDSA signature from the given derivation path.
    async fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .secp256k1_sign_digest_ecdsa(derivation_path, message_hash)
            .await
    }

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path.
    async fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.base
            .secp256k1_verify_ecdsa(derivation_path, message_hash, signature)
            .await
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path.
    async fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 33], BoxError> {
        self.base.secp256k1_public_key(derivation_path).await
    }
}

impl StoreFeatures for AgentCtx {
    /// Retrieves data from storage at the specified path.
    async fn store_get(&self, path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        self.base.store_get(path).await
    }

    /// Lists objects in storage with optional prefix and offset filters.
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results;
    /// * `offset` - Optional path to start listing from (exclude).
    async fn store_list(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> Result<Vec<ObjectMeta>, BoxError> {
        self.base.store_list(prefix, offset).await
    }

    /// Stores data at the specified path with a given write mode.
    ///
    /// # Arguments
    /// * `path` - Target storage path;
    /// * `mode` - Write mode (Create, Overwrite, etc.);
    /// * `value` - Data to store as bytes.
    async fn store_put(
        &self,
        path: &Path,
        mode: PutMode,
        value: bytes::Bytes,
    ) -> Result<PutResult, BoxError> {
        self.base.store_put(path, mode, value).await
    }

    /// Renames a storage object if the target path doesn't exist.
    ///
    /// # Arguments
    /// * `from` - Source path;
    /// * `to` - Destination path.
    async fn store_rename_if_not_exists(&self, from: &Path, to: &Path) -> Result<(), BoxError> {
        self.base.store_rename_if_not_exists(from, to).await
    }

    /// Deletes data at the specified path.
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete.
    async fn store_delete(&self, path: &Path) -> Result<(), BoxError> {
        self.base.store_delete(path).await
    }
}

impl CacheFeatures for AgentCtx {
    /// Checks if a key exists in the cache.
    fn cache_contains(&self, key: &str) -> bool {
        self.base.cache_contains(key)
    }

    /// Gets a cached value by key, returns error if not found or deserialization fails.
    async fn cache_get<T>(&self, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.base.cache_get(key).await
    }

    /// Gets a cached value or initializes it if missing.
    ///
    /// If key doesn't exist, calls init function to create value and cache it.
    async fn cache_get_with<T, F>(&self, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        self.base.cache_get_with(key, init).await
    }

    /// Sets a value in cache with optional expiration policy.
    async fn cache_set<T>(&self, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        self.base.cache_set(key, val).await
    }

    /// Sets a value in cache if key doesn't exist, returns true if set.
    async fn cache_set_if_not_exists<T>(&self, key: &str, val: (T, Option<CacheExpiry>)) -> bool
    where
        T: Sized + Serialize + Send,
    {
        self.base.cache_set_if_not_exists(key, val).await
    }

    /// Deletes a cached value by key, returns true if key existed.
    async fn cache_delete(&self, key: &str) -> bool {
        self.base.cache_delete(key).await
    }

    /// Returns an iterator over all cached items with raw value.
    fn cache_raw_iter(
        &self,
    ) -> impl Iterator<Item = (Arc<String>, Arc<(Bytes, Option<CacheExpiry>)>)> {
        self.base.cache_raw_iter()
    }
}

impl CanisterCaller for AgentCtx {
    /// Performs a query call to a canister (read-only, no state changes).
    ///
    /// # Arguments
    /// * `canister` - Target canister principal;
    /// * `method` - Method name to call;
    /// * `args` - Input arguments encoded in Candid format.
    async fn canister_query<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        self.base.canister_query(canister, method, args).await
    }

    /// Performs an update call to a canister (may modify state).
    ///
    /// # Arguments
    /// * `canister` - Target canister principal;
    /// * `method` - Method name to call;
    /// * `args` - Input arguments encoded in Candid format.
    async fn canister_update<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        self.base.canister_update(canister, method, args).await
    }
}

impl HttpFeatures for AgentCtx {
    /// Makes an HTTPs request.
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`;
    /// * `method` - HTTP method (GET, POST, etc.);
    /// * `headers` - Optional HTTP headers;
    /// * `body` - Optional request body (default empty).
    async fn https_call(
        &self,
        url: &str,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>,
    ) -> Result<reqwest::Response, BoxError> {
        self.base.https_call(url, method, headers, body).await
    }

    /// Makes a signed HTTPs request with message authentication.
    ///
    /// # Arguments
    /// * `url` - Target URL;
    /// * `method` - HTTP method (GET, POST, etc.);
    /// * `message_digest` - 32-byte message digest for signing;
    /// * `headers` - Optional HTTP headers;
    /// * `body` - Optional request body (default empty).
    async fn https_signed_call(
        &self,
        url: &str,
        method: http::Method,
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>,
    ) -> Result<reqwest::Response, BoxError> {
        self.base
            .https_signed_call(url, method, message_digest, headers, body)
            .await
    }

    /// Makes a signed CBOR-encoded RPC call.
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to;
    /// * `method` - RPC method name to call;
    /// * `args` - Arguments to serialize as CBOR and send with the request.
    async fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        args: impl Serialize + Send,
    ) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.base.https_signed_rpc(endpoint, method, args).await
    }
}

/// A iteration style executor for completion.
pub struct CompletionRunner {
    ctx: AgentCtx,
    req: CompletionRequest,
    model: Model,
    resources: Vec<Resource>,
    chat_history: Vec<Message>,
    tool_calls: Vec<ToolCall>,
    total_usage: Usage,
    current_usage: Usage,
    artifacts: Vec<Resource>,
    steering_message: Vec<ContentPart>,
    follow_up_message: VecDeque<ContentPart>,
    implicit_context: Option<Message>,
    pending_tool_calls: Vec<ToolCall>,
    pending_tool_call_raw_history_start: Option<usize>,
    tools_usage: HashMap<String, Usage>,
    last_output: Option<AgentOutput>,
    discovered_tools: BTreeMap<String, FunctionDefinition>,
    discovery_selection_counts: BTreeMap<String, usize>,
    merge_discovered_tools: Option<bool>,
    done: bool,
    unbound: bool,
    turns: usize,
}

impl CompletionRunner {
    /// Enables unbound mode for the completion runner.
    pub fn unbound(self) -> Self {
        Self {
            unbound: true,
            ..self
        }
    }

    /// Reserves the chat history for the completion runner.
    pub fn reserve_chat_history(self, chat_history: Vec<Message>) -> Self {
        Self {
            chat_history,
            ..self
        }
    }

    /// Appends messages to the chat history.
    pub fn append_chat_history(&mut self, messages: Vec<Message>) {
        self.chat_history.extend(messages);
    }

    /// Returns mutable access to the accumulated chat history messages.
    ///
    /// This allows callers to update already recorded messages in place while
    /// preserving message order. Use [`Self::append_chat_history`] to add new
    /// messages.
    pub fn chat_history_mut(&mut self) -> &mut [Message] {
        &mut self.chat_history
    }

    /// Returns whether the completion has finished.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Returns whether the completion is idle, meaning it has no pending tasks.
    pub fn is_idle(&self) -> bool {
        !self.has_request_input()
            && self.steering_message.is_empty()
            && self.follow_up_message.is_empty()
            && self.pending_tool_calls.is_empty()
    }

    /// Returns whether there are no pending tool calls.
    pub fn no_pending_tool_calls(&self) -> bool {
        self.pending_tool_calls.is_empty()
    }

    /// Returns the number of turns executed.
    pub fn turns(&self) -> usize {
        self.turns
    }

    /// Returns the agent context driving this completion.
    pub fn ctx(&self) -> &AgentCtx {
        &self.ctx
    }

    /// Get the original completion request.
    pub fn req(&self) -> &CompletionRequest {
        &self.req
    }

    /// Get the model used for this completion.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Returns the chat history of the completion so far.
    pub fn chat_history(&self) -> &Vec<Message> {
        &self.chat_history
    }

    /// Get the total usage accumulated so far, including all intermediate steps.
    pub fn total_usage(&self) -> &Usage {
        &self.total_usage
    }

    /// Get the usage from the most recent turn.
    pub fn current_usage(&self) -> &Usage {
        &self.current_usage
    }

    /// Returns the accumulated usage of the tools so far.
    pub fn tools_usage(&self) -> &HashMap<String, Usage> {
        &self.tools_usage
    }

    /// Returns the most recent non-final output, when one is available.
    pub fn last_output(&self) -> Option<&AgentOutput> {
        self.last_output.as_ref()
    }

    /// Returns the discovered-tool merge policy.
    ///
    /// `Some(true)` forces discovered tool definitions into later requests, `Some(false)` keeps
    /// them only in discovery-tool output context, and `None` lets the runner probe whether the
    /// current model needs request-side merging.
    pub fn merge_discovered_tools(&self) -> Option<bool> {
        self.merge_discovered_tools
    }

    /// Enables or disables unbound mode.
    ///
    /// In unbound mode, reaching an idle boundary does not finalize the runner. The step still
    /// returns its latest [`AgentOutput`], and later calls to [`Self::next`] return `Ok(None)`
    /// while the runner is idle until new input is queued via [`Self::follow_up`] or
    /// [`Self::steer`]. Terminal failures still finalize the runner.
    ///
    /// This mode is primarily useful when driving [`CompletionRunner`] directly. A
    /// [`CompletionStream`] still terminates permanently after it yields `None`, per the [`Stream`]
    /// contract.
    pub fn set_unbound(&mut self, unbound: bool) {
        self.unbound = unbound;
    }

    /// Sets the discovered-tool merge policy.
    ///
    /// `Some(true)` adds definitions returned by discovery tools such as `tools_search` and
    /// `tools_select` to later request tool lists, and compacts discovery output kept in
    /// conversation context. `Some(false)` keeps discovered schemas only in discovery-tool output
    /// context. `None` enables the repeated-selection probe so the runner can discover whether the
    /// current model needs request-side merging.
    pub fn set_merge_discovered_tools(&mut self, merge_discovered_tools: Option<bool>) {
        self.merge_discovered_tools = merge_discovered_tools;
    }

    /// Queue a steering message to interrupt the agent mid-run.
    /// Delivered after current tool execution, skips remaining tools.
    /// No effect if the completion has finished.
    pub fn steer(&mut self, message: impl Into<ContentPart>) {
        if self.done {
            return;
        }
        self.steering_message.push(message.into());
    }

    /// Queue a steering message with multiple content parts to interrupt the agent mid-run.
    pub fn steer_content(&mut self, content: Vec<ContentPart>) {
        if self.done {
            return;
        }
        self.steering_message.extend(content);
    }

    /// Returns the iter over the queued steering message content parts.
    pub fn steering_message_iter(&'_ self) -> core::slice::Iter<'_, ContentPart> {
        self.steering_message.iter()
    }

    /// Queue a follow-up message for the next safe user turn.
    /// Delivered with the current pending tool-call results when they finish, or at the next idle
    /// boundary when no tools are pending. Steering still takes priority.
    /// No effect if the completion has finished.
    pub fn follow_up(&mut self, message: impl Into<ContentPart>) {
        if self.done {
            return;
        }
        self.follow_up_message.push_back(message.into());
    }

    /// Queue a follow-up message with multiple content parts for the next safe user turn.
    pub fn follow_up_content(&mut self, content: Vec<ContentPart>) {
        if self.done {
            return;
        }
        self.follow_up_message.extend(content);
    }

    /// Returns the iter over the queued follow-up message content parts.
    pub fn follow_up_message_iter(&'_ self) -> std::collections::vec_deque::Iter<'_, ContentPart> {
        self.follow_up_message.iter()
    }

    /// Drops the current in-flight request after a transport-level model failure.
    ///
    /// This keeps accumulated chat history, usage, artifacts, and queued follow-up messages, but
    /// removes request content that was already sent to the failed completion. Long-lived callers
    /// can use this before processing newly queued input, so stale tool results are not resent.
    /// Pending, unexecuted tool calls are closed in the visible history before their raw provider
    /// history is pruned.
    pub fn discard_in_flight_request(&mut self) {
        self.discard_in_flight_request_with_interrupted_tool_outputs("tool call discarded", None);
    }

    /// Prunes completed tool interactions from the accumulated provider raw history.
    ///
    /// `req.raw_history` holds provider-native message JSON (OpenAI, Anthropic, and Gemini
    /// shapes all differ), so pruning rewrites the raw values in place rather than
    /// round-tripping through [`Message`], which would drop or corrupt provider-specific
    /// formats. Tool-call requests, their results, and items left without meaningful
    /// content are removed; visible text and reasoning stay untouched.
    ///
    /// Long-lived callers (for example subagent sessions) can invoke this at an idle
    /// boundary to reclaim context-window budget from tool payloads the model has already
    /// consumed. The call is a no-op unless the runner is idle, so an in-flight tool round
    /// is never left with an orphaned call or result.
    pub fn prune_req_raw_history(&mut self) {
        if !self.is_idle() || self.req.raw_history.is_empty() {
            return;
        }
        Self::prune_tool_interactions_from_raw_history(&mut self.req.raw_history);
    }

    fn discard_in_flight_request_with_interrupted_tool_outputs(
        &mut self,
        error: &str,
        reason: Option<&str>,
    ) {
        if !self.pending_tool_calls.is_empty() {
            self.append_interrupted_tool_outputs(error, reason);
        }
        self.req.prompt.clear();
        self.req.content.clear();
        self.req.documents.clear();
        self.req.role = None;
        self.req.tool_choice_required = false;
        self.req.output_schema = None;
        self.pending_tool_calls.clear();
        self.discard_pending_tool_call_raw_history();
    }

    /// Stops the current task while keeping the runner reusable for later input.
    ///
    /// This drops in-flight request state, pending tools, queued steering, and queued follow-up
    /// text, but it does not mark the runner done. The returned output is recorded as the latest
    /// idle-state output and includes accumulated usage/history so observers can account for work
    /// already performed before the stop.
    pub fn stop_current_task(&mut self, mut output: AgentOutput) -> AgentOutput {
        self.discard_in_flight_request_with_interrupted_tool_outputs(
            "tool call stopped",
            Some(&output.content),
        );
        self.req.chat_history.clear();
        self.steering_message.clear();
        self.follow_up_message.clear();
        self.implicit_context = None;

        output.chat_history = self.chat_history.clone();
        output.tool_calls = self.tool_calls.clone();
        output.artifacts = self.artifacts.clone();
        output.usage = self.total_usage.clone();
        output.tools_usage = self.tools_usage.clone();

        self.last_output = Some(output.clone());
        output
    }

    fn append_interrupted_tool_outputs(&mut self, error: &str, reason: Option<&str>) {
        let calls = Self::unanswered_tool_calls(&self.chat_history);
        if calls.is_empty() {
            return;
        }

        let reason = reason.map(str::trim).filter(|reason| !reason.is_empty());
        let output = match reason {
            Some(reason) => json!({"error": error, "reason": reason}),
            None => json!({"error": error}),
        };
        let content = calls
            .into_iter()
            .map(|call| ContentPart::ToolOutput {
                name: call.name,
                output: output.clone(),
                is_error: Some(true),
                call_id: call.call_id,
                remote_id: None,
            })
            .collect::<Vec<_>>();

        self.chat_history.push(Message {
            role: "tool".to_string(),
            content,
            ..Default::default()
        });
    }

    /// Set an implicit context message that is automatically included in the next request.
    pub fn implicit_context(&mut self, message: Message) {
        self.implicit_context = Some(message);
    }

    /// Selects the model label to use for subsequent completion turns.
    pub fn set_model(&mut self, model: Option<String>) {
        self.req.model = model;
    }

    /// Selects the reasoning/thinking effort to use for subsequent completion turns.
    pub fn set_effort(&mut self, effort: Option<ModelEffort>) {
        self.req.effort = effort;
    }

    /// Selects the tool definitions to use for subsequent completion turns.
    pub fn set_tools(&mut self, tools: Vec<FunctionDefinition>) {
        self.req.tools = tools;
        self.discovered_tools.clear();
        self.discovery_selection_counts.clear();
    }

    /// Accumulate usage from an intermediate step into the runner's total usage.
    pub fn accumulate(&mut self, other: &Usage) {
        self.total_usage.accumulate(other);
    }

    /// Accumulate tool usage from an intermediate step into the runner's total tools usage.
    pub fn accumulate_tools_usage(&mut self, other: &HashMap<String, Usage>) {
        for (tool, usage) in other.iter() {
            self.tools_usage
                .entry(tool.clone())
                .or_default()
                .accumulate(usage);
        }
    }

    /// Returns whether the current runner should be compacted based on usage and turn count.
    pub fn needs_compaction_with<F>(&self, pending_tokens: F) -> bool
    where
        F: FnOnce() -> u64,
    {
        if self.turns >= MAX_TURNS_TO_COMPACT {
            return true;
        }
        let context_window = self.model.context_window as u64;
        let threshold = if context_window == 0 {
            100_000
        } else {
            context_window.saturating_mul(8).saturating_div(10).max(1)
        };

        self.current_usage
            .input_tokens
            .saturating_add(pending_tokens())
            >= threshold
    }

    fn add_discovered_tools_from_output(&mut self, tool_name: &str, output: &Json) {
        if self.merge_discovered_tools == Some(false)
            || (!tool_name.eq_ignore_ascii_case(TOOLS_SELECT_NAME)
                && !tool_name.eq_ignore_ascii_case(TOOLS_SEARCH_NAME))
        {
            return;
        }

        let Ok(tools_output) = ToolsOutput::deserialize(output) else {
            return;
        };

        let count_selection = tool_name.eq_ignore_ascii_case(TOOLS_SELECT_NAME)
            && self.merge_discovered_tools.is_none();
        let mut added = 0;
        for definition in tools_output.tools {
            if definition.name.trim().is_empty() {
                continue;
            }

            let key = definition.name.to_ascii_lowercase();
            if count_selection {
                let count = self
                    .discovery_selection_counts
                    .entry(key.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
                if *count >= 2 {
                    self.merge_discovered_tools = Some(true);
                }
            }
            self.discovered_tools.entry(key).or_insert(definition);
            added += 1;
            if added >= MAX_DISCOVERED_REQUEST_TOOLS {
                break;
            }
        }
    }

    fn merge_discovered_tools_into_request(&self, req: &mut CompletionRequest) {
        if self.merge_discovered_tools != Some(true) || self.discovered_tools.is_empty() {
            return;
        }

        let mut seen: BTreeSet<String> = req
            .tools
            .iter()
            .map(|tool| tool.name.to_ascii_lowercase())
            .collect();
        for (name, definition) in &self.discovered_tools {
            if seen.insert(name.clone()) {
                req.tools.push(definition.clone());
            }
        }
    }

    /// Compacts a discovery tool output in place once discovered definitions
    /// are merged into the request tools, so full schemas are not duplicated
    /// in the conversation context. Non-discovery outputs are left untouched.
    fn compact_discovery_tool_output_for_context(&self, tool_name: &str, output: &mut Json) {
        if self.merge_discovered_tools != Some(true) {
            return;
        }

        let keep_description = if tool_name.eq_ignore_ascii_case(TOOLS_SEARCH_NAME) {
            true
        } else if tool_name.eq_ignore_ascii_case(TOOLS_SELECT_NAME) {
            false
        } else {
            return;
        };

        let Ok(tools_output) = ToolsOutput::deserialize(&*output) else {
            return;
        };

        let tools = tools_output
            .tools
            .into_iter()
            .map(|definition| {
                if keep_description {
                    json!({
                        "name": definition.name,
                        "description": definition.description,
                    })
                } else {
                    json!({
                        "name": definition.name,
                    })
                }
            })
            .collect::<Vec<_>>();

        *output = json!({
            "tools": tools,
            "total_tools": tools_output.total_tools,
        });
    }

    // Drains all queued steering messages into a single user turn. When steering exists, queued
    // follow-up messages are prepended so the next round sees one combined instruction.
    fn drain_steering_message(&mut self) -> Option<Vec<ContentPart>> {
        if self.steering_message.is_empty() {
            None
        } else {
            // Follow-up messages are placed before steering messages to preserve the deferred user
            // intent when an operator also injects steering.
            let mut msgs: Vec<ContentPart> = self.follow_up_message.drain(..).collect();
            msgs.append(&mut self.steering_message);
            Some(msgs)
        }
    }

    fn drain_queued_message(&mut self) -> Option<Vec<ContentPart>> {
        let mut msgs: Vec<ContentPart> = self.follow_up_message.drain(..).collect();
        msgs.append(&mut self.steering_message);
        if msgs.is_empty() { None } else { Some(msgs) }
    }

    fn drain_follow_up_message(&mut self) -> Option<Vec<ContentPart>> {
        let msgs: Vec<ContentPart> = self.follow_up_message.drain(..).collect();
        if msgs.is_empty() { None } else { Some(msgs) }
    }

    fn set_next_user_content(&mut self, content: Vec<ContentPart>) {
        self.req.content = content;
        self.req.role = Some("user".to_string());
    }

    fn has_request_input(&self) -> bool {
        !self.req.prompt.is_empty()
            || !self.req.content.is_empty()
            || !self.req.documents.is_empty()
    }

    fn unanswered_tool_calls(messages: &[Message]) -> Vec<ToolCall> {
        let mut pending: Vec<ToolCall> = Vec::new();

        for message in messages {
            for part in &message.content {
                match part {
                    ContentPart::ToolCall {
                        name,
                        args,
                        call_id,
                    } => pending.push(ToolCall {
                        name: name.clone(),
                        args: args.clone(),
                        call_id: call_id.clone(),
                        result: None,
                        remote_id: None,
                    }),
                    ContentPart::ToolOutput { name, call_id, .. } => {
                        if let Some(pos) =
                            Self::matching_tool_call_position(&pending, name, call_id.as_deref())
                        {
                            pending.remove(pos);
                        }
                    }
                    _ => {}
                }
            }
        }

        pending
    }

    fn matching_tool_call_position(
        pending: &[ToolCall],
        output_name: &str,
        output_call_id: Option<&str>,
    ) -> Option<usize> {
        pending.iter().position(|tool| {
            Self::tool_call_matches_output(
                &tool.name,
                tool.call_id.as_deref(),
                output_name,
                output_call_id,
            )
        })
    }

    fn tool_call_matches_output(
        pending_name: &str,
        pending_call_id: Option<&str>,
        output_name: &str,
        output_call_id: Option<&str>,
    ) -> bool {
        match (pending_call_id, output_call_id) {
            (Some(pending), Some(output)) => pending == output,
            (None, None) => pending_name == output_name,
            _ => false,
        }
    }

    fn discard_pending_tool_call_raw_history(&mut self) {
        if let Some(start) = self.pending_tool_call_raw_history_start.take() {
            Self::prune_unanswered_tool_calls_from_raw_history(&mut self.req.raw_history, start);
        }
    }

    fn prune_unanswered_tool_calls_from_raw_history(raw_history: &mut Vec<Json>, start: usize) {
        if start >= raw_history.len() {
            return;
        }

        let retained: Vec<Json> = raw_history
            .drain(start..)
            .filter_map(Self::prune_unanswered_tool_calls_from_raw_item)
            .collect();
        raw_history.extend(retained);
    }

    fn prune_unanswered_tool_calls_from_raw_item(mut value: Json) -> Option<Json> {
        if Self::is_tool_call_raw_item(&value) {
            return None;
        }

        Self::prune_nested_tool_calls(&mut value);
        if Self::raw_history_item_has_context(&value) {
            Some(value)
        } else {
            None
        }
    }

    fn prune_nested_tool_calls(value: &mut Json) {
        match value {
            Json::Array(items) => {
                let retained: Vec<Json> = items
                    .drain(..)
                    .filter_map(Self::prune_unanswered_tool_calls_from_raw_item)
                    .collect();
                *items = retained;
            }
            Json::Object(map) => {
                // OpenAI Chat Completions keeps tool calls as fields on an assistant message.
                // Remove the unanswered calls, but keep any text/reasoning fields on the same
                // message. Provider raw history may also wrap output items under arbitrary fields,
                // so recurse through every remaining value instead of only common content arrays.
                map.remove("tool_calls");
                map.remove("function_call");
                map.remove("functionCall");

                for value in map.values_mut() {
                    Self::prune_nested_tool_calls(value);
                }
            }
            _ => {}
        }
    }

    // Removes completed tool interactions (calls and results) from provider raw history.
    // Walks backwards so an OpenAI Responses `reasoning` item can see whether the item it
    // must immediately precede survived; the API rejects a reasoning item whose required
    // following item is missing, so orphaned reasoning follows its pruned sibling out.
    fn prune_tool_interactions_from_raw_history(raw_history: &mut Vec<Json>) {
        let mut retained: Vec<Json> = Vec::with_capacity(raw_history.len());
        let mut next_kept = true;
        for value in raw_history.drain(..).rev() {
            if !next_kept && value.get("type").and_then(|v| v.as_str()) == Some("reasoning") {
                continue;
            }
            match Self::prune_tool_interactions_from_raw_item(value) {
                Some(value) => {
                    retained.push(value);
                    next_kept = true;
                }
                None => next_kept = false,
            }
        }
        retained.reverse();
        *raw_history = retained;
    }

    fn prune_tool_interactions_from_raw_item(mut value: Json) -> Option<Json> {
        if Self::is_tool_call_raw_item(&value) || Self::is_tool_output_raw_item(&value) {
            return None;
        }

        Self::prune_nested_tool_interactions(&mut value);
        if Self::raw_history_item_has_context(&value) {
            Some(value)
        } else {
            None
        }
    }

    // Like `prune_nested_tool_calls`, but also drops tool results, so a wrapper message
    // whose content held only a tool interaction loses its remaining context and is
    // pruned by `raw_history_item_has_context`.
    fn prune_nested_tool_interactions(value: &mut Json) {
        match value {
            Json::Array(items) => {
                let retained: Vec<Json> = items
                    .drain(..)
                    .filter_map(Self::prune_tool_interactions_from_raw_item)
                    .collect();
                *items = retained;
            }
            Json::Object(map) => {
                map.remove("tool_calls");
                map.remove("function_call");
                map.remove("functionCall");

                for value in map.values_mut() {
                    Self::prune_nested_tool_interactions(value);
                }
            }
            _ => {}
        }
    }

    fn is_tool_call_raw_item(value: &Json) -> bool {
        let Some(map) = value.as_object() else {
            return false;
        };

        if matches!(
            map.get("type").and_then(|v| v.as_str()),
            Some(
                "function_call"
                    | "custom_tool_call"
                    | "computer_call"
                    | "tool_search_call"
                    | "local_shell_call"
                    | "shell_call"
                    | "apply_patch_call"
                    | "mcp_approval_request"
                    | "tool_call"
                    | "tool_use"
                    | "server_tool_use"
                    | "ToolCall"
                    | "toolCall"
            )
        ) {
            return true;
        }

        // Gemini function-call parts do not use a `type` field. Treat only the part itself as a
        // tool call; wrapper objects that also carry text or metadata should be pruned recursively
        // so their non-tool context survives.
        map.contains_key("functionCall")
            && map.keys().all(|key| {
                matches!(
                    key.as_str(),
                    "functionCall" | "thought" | "thoughtSignature"
                )
            })
    }

    fn is_tool_output_raw_item(value: &Json) -> bool {
        let Some(map) = value.as_object() else {
            return false;
        };

        // OpenAI Chat Completions carries tool results as whole messages with the "tool" role.
        if map.get("role").and_then(|v| v.as_str()) == Some("tool") {
            return true;
        }

        if matches!(
            map.get("type").and_then(|v| v.as_str()),
            Some(
                "function_call_output"
                    | "custom_tool_call_output"
                    | "computer_call_output"
                    | "local_shell_call_output"
                    | "shell_call_output"
                    | "apply_patch_call_output"
                    | "mcp_approval_response"
                    | "tool_result"
                    | "tool_output"
                    | "ToolOutput"
                    | "toolOutput"
            )
        ) {
            return true;
        }

        // Gemini function-response parts do not use a `type` field; mirror the
        // `is_tool_call_raw_item` handling for `functionCall` parts.
        map.contains_key("functionResponse")
            && map.keys().all(|key| {
                matches!(
                    key.as_str(),
                    "functionResponse" | "thought" | "thoughtSignature"
                )
            })
    }

    fn raw_history_item_has_context(value: &Json) -> bool {
        match value {
            Json::Null => false,
            Json::Bool(_) | Json::Number(_) => true,
            Json::String(text) => !text.is_empty(),
            Json::Array(items) => items.iter().any(Self::raw_history_item_has_context),
            Json::Object(map) => map.iter().any(|(key, value)| {
                !matches!(
                    key.as_str(),
                    "role" | "type" | "name" | "id" | "status" | "phase" | "timestamp"
                ) && Self::raw_history_item_has_context(value)
            }),
        }
    }

    fn stream_placeholder(&self) -> Self {
        Self {
            ctx: self.ctx.clone(),
            req: CompletionRequest::default(),
            model: self.model.clone(),
            resources: Vec::new(),
            chat_history: Vec::new(),
            tool_calls: Vec::new(),
            total_usage: Usage::default(),
            current_usage: Usage::default(),
            artifacts: Vec::new(),
            steering_message: Vec::new(),
            follow_up_message: VecDeque::new(),
            implicit_context: None,
            pending_tool_calls: Vec::new(),
            pending_tool_call_raw_history_start: None,
            tools_usage: HashMap::new(),
            last_output: None,
            discovered_tools: BTreeMap::new(),
            discovery_selection_counts: BTreeMap::new(),
            merge_discovered_tools: None,
            done: true,
            unbound: self.unbound,
            turns: self.turns,
        }
    }

    /// Execute the next step.
    /// - Calls the model completion.
    /// - Automatically handles tool/agent calls and writes the results back to the conversation history.
    /// - If there are more steps, it constructs the next request and returns the current intermediate result.
    /// - If completed or failed, it returns the final result; the next call will return `Ok(None)`.
    /// - In unbound mode, an idle boundary returns the latest step output without finalizing. A
    ///   later call returns `Ok(None)` until new input is queued.
    ///
    pub async fn next(&mut self) -> Result<Option<AgentOutput>, BoxError> {
        if self.done {
            return Ok(None);
        }

        let token = self.ctx.base.cancellation_token();
        tokio::select! {
            _ = token.cancelled() => {
                let output = AgentOutput {
                    failed_reason: Some("operation cancelled".to_string()),
                    ..Default::default()
                };
                Ok(Some(self.final_output(output)))
            }
            res = self.inner_next() => res
        }
    }

    /// Summarizes the current conversation into a single handoff message and swaps in a fresh
    /// runner seeded with that summary, discarding the bloated history. Pending tool calls are
    /// executed first unless queued steering interrupts them; interrupted calls are closed before
    /// compaction so provider tool-call requirements are not stranded. Queued follow-up or
    /// steering input is preserved and delivered after the handoff so user intent is not folded
    /// into the compaction prompt.
    pub async fn handoff(
        &mut self,
        compaction_prompt: Option<String>,
    ) -> Result<(Self, AgentOutput), BoxError> {
        let unbound = self.unbound;
        let prompt = compaction_prompt.unwrap_or_else(|| COMPACTION_PROMPT.to_string());

        if !self.pending_tool_calls.is_empty() {
            if self.steering_message.is_empty() {
                if self.has_request_input() {
                    return Err(
                        "cannot compact while pending tool calls and request input are both queued"
                            .into(),
                    );
                }

                self.execute_pending_tool_calls_into_request().await?;
                self.commit_tool_outputs_to_history();
            } else {
                self.discard_in_flight_request_with_interrupted_tool_outputs(
                    "tool call interrupted by steering",
                    None,
                );
            }
        }

        let merge_discovered_tools = self.merge_discovered_tools;
        let discovered_tools = self.discovered_tools.clone();
        let discovery_selection_counts = self.discovery_selection_counts.clone();
        // Captured before clearing tools so the replacement runner restores the base toolset.
        let handoff_req = self.req.clone();
        let queued_follow_up = std::mem::take(&mut self.follow_up_message);
        let queued_steering = std::mem::take(&mut self.steering_message);

        self.steer(prompt);
        // Drop tools so the summarization turn cannot spawn more tool calls.
        self.set_tools(Vec::new());
        let output = self.finalize(None).await?;
        if let Some(failed_reason) = output.failed_reason.clone() {
            return Err(failed_reason.into());
        }

        let summary = output.content.trim().to_string();
        if summary.is_empty() {
            return Err("context compaction produced an empty summary".into());
        }

        // The summary seeds the next conversation as its first message. It lives in `chat_history`
        // for the first request and migrates into the runner's raw history on later turns.
        let compaction_msg = Message {
            role: "assistant".into(),
            content: vec![summary.into()],
            timestamp: Some(unix_ms()),
            ..Default::default()
        };

        let req = CompletionRequest {
            instructions: handoff_req.instructions,
            role: handoff_req.role,
            chat_history: vec![compaction_msg.clone()],
            tools: handoff_req.tools,
            model: handoff_req.model,
            effort: handoff_req.effort,
            ..Default::default()
        };
        let mut runner = self
            .ctx
            .clone()
            .completion_iter(req, Vec::new())
            // Seed the reported chat history too, so the summary survives into the final output.
            .reserve_chat_history(vec![compaction_msg]);
        runner.set_unbound(unbound);
        runner.set_merge_discovered_tools(merge_discovered_tools);
        runner.discovered_tools = discovered_tools;
        runner.discovery_selection_counts = discovery_selection_counts;
        runner.follow_up_message = queued_follow_up;
        runner.steering_message = queued_steering;
        Ok((runner, output))
    }

    /// Finalize the completion with an optional prompt.
    ///
    /// Queued messages, plus the optional prompt, are processed through the normal runner flow.
    /// If the runner is already idle, finalization returns the latest intermediate output with
    /// accumulated usage, tool calls, artifacts, and chat history attached.
    pub async fn finalize(&mut self, prompt: Option<String>) -> Result<AgentOutput, BoxError> {
        if self.done {
            return Err("completion already finalized".into());
        }

        self.unbound = false;

        if let Some(prompt) = prompt {
            self.follow_up_message.push_back(prompt.into());
        }

        if !self.has_request_input() && self.pending_tool_calls.is_empty() {
            if let Some(content) = self.drain_queued_message() {
                self.set_next_user_content(content);
            } else {
                return Ok(self.final_idle_output());
            }
        }

        let mut last: Option<AgentOutput> = None;
        while let Some(step) = self.next().await? {
            if step.failed_reason.is_some() {
                return Ok(step);
            }
            last = Some(step);
        }

        last.ok_or_else(|| "completion runner returned no output".into())
    }

    async fn execute_pending_tool_calls_into_request(&mut self) -> Result<bool, BoxError> {
        let tool_calls = std::mem::take(&mut self.pending_tool_calls);
        if tool_calls.is_empty() {
            return Ok(false);
        }

        let mut tool_call_futs: Vec<BoxPinFut<ToolCall>> = Vec::new();
        for mut tool in tool_calls.into_iter() {
            let tool_name = tool.name.to_ascii_lowercase();
            if self.ctx.has_tool_lowercase(&tool_name)
                || strip_prefix_ignore_ascii_case(&tool_name, REMOTE_TOOL_PREFIX).is_some()
            {
                let ctx = self.ctx.clone();
                let input = ToolInput {
                    name: tool.name.clone(),
                    args: tool.args.clone(),
                    resources: self
                        .ctx
                        .select_tool_resources(&tool.name, &mut self.resources)
                        .await,
                    meta: None,
                };
                tool_call_futs.push(Box::pin(async move {
                    match ctx.tool_call(input).await {
                        Ok((res, remote_id)) => {
                            tool.remote_id = remote_id;
                            tool.result = Some(res);
                        }
                        Err(err) => {
                            // The tool call failed, but we must not abort the whole conversation:
                            // surface the error so the LLM can try to correct it and continue.
                            tool.result = Some(ToolOutput {
                                output: json!({ "error": format!(
                                    "tool call failed: {}",
                                    err
                                )}),
                                is_error: Some(true),
                                ..Default::default()
                            });
                        }
                    }
                    tool
                }));
            } else if self.ctx.agents.contains_lowercase(&tool_name)
                || self.ctx.subagents.contains_lowercase(&tool_name)
                || strip_prefix_ignore_ascii_case(&tool_name, SUB_AGENT_PREFIX).is_some()
                || strip_prefix_ignore_ascii_case(&tool_name, REMOTE_AGENT_PREFIX).is_some()
            {
                // Subagents consume structured controls such as `session`, `model`, and
                // `effort`, so preserve their full argument object. Plain string args and
                // normal agents keep the historical prompt behavior.
                let prompt = if let Some(args) = tool.args.as_str() {
                    args.to_string()
                } else if let Some(args) = tool.args.as_object()
                    && args.len() == 1
                    && let Some(prompt) = args.get("prompt").and_then(|v| v.as_str())
                {
                    prompt.to_string()
                } else {
                    serde_json::to_string(&tool.args).unwrap_or_else(|_| tool.args.to_string())
                };

                let ctx = self.ctx.clone();
                let input = AgentInput {
                    name: tool.name.clone(),
                    prompt,
                    resources: self
                        .ctx
                        .select_agent_resources(&tool.name, &mut self.resources)
                        .await,
                    ..Default::default()
                };
                tool_call_futs.push(Box::pin(async move {
                    match ctx.agent_run(input).await {
                        Ok((res, remote_id)) => {
                            tool.remote_id = remote_id;
                            tool.result = Some(res.into_tool_output());
                        }
                        Err(err) => {
                            // The agent run failed, but we must not abort the whole conversation:
                            // surface the error so the LLM can try to correct it and continue.
                            tool.result = Some(ToolOutput {
                                output: json!({ "error": format!(
                                    "agent run failed: {}",
                                    err
                                )}),
                                is_error: Some(true),
                                ..Default::default()
                            });
                        }
                    }
                    tool
                }));
            } else {
                tool_call_futs.push(Box::pin(async move {
                    tool.result = Some(ToolOutput {
                        output: json!({ "error": format!(
                            "tool call failed: {} not found",
                            tool.name
                        )}),
                        is_error: Some(true),
                        ..Default::default()
                    });
                    tool
                }));
            }
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_calls_continue: Vec<ContentPart> = Vec::new();
        if !tool_call_futs.is_empty() {
            let results = futures::future::join_all(tool_call_futs).await;

            for mut tool in results {
                if let Some(res) = &mut tool.result {
                    let mut usage = res.usage.clone();
                    // usage.requests originally counts internal calls; reset it to 1 so it
                    // represents this single model-triggered invocation, keeping per-tool call
                    // counts meaningful.
                    usage.requests = 1;
                    self.tools_usage
                        .entry(tool.name.to_ascii_lowercase())
                        .and_modify(|u| u.accumulate(&usage))
                        .or_insert(usage);
                    self.accumulate_tools_usage(&res.tools_usage);
                    self.accumulate(&res.usage);
                    self.add_discovered_tools_from_output(&tool.name, &res.output);
                    self.compact_discovery_tool_output_for_context(&tool.name, &mut res.output);

                    // We can not ignore some tool calls.
                    // GPT-5: An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'.
                    tool_calls_continue.push(ContentPart::ToolOutput {
                        name: tool.name.clone(),
                        output: res.output.clone(),
                        is_error: res.is_error,
                        call_id: tool.call_id.clone(),
                        remote_id: tool.remote_id,
                    });

                    self.artifacts.append(&mut res.artifacts);
                    tool_calls.push(tool);
                }
            }
        }

        // Accumulate this round's tool calls.
        self.tool_calls.append(&mut tool_calls);
        self.req.role = Some("tool".to_string());
        if !tool_calls_continue.is_empty() {
            self.req.content.append(&mut tool_calls_continue);
        }

        Ok(true)
    }

    fn commit_tool_outputs_to_history(&mut self) {
        if self.req.role.as_deref() != Some("tool") || self.req.content.is_empty() {
            return;
        }

        let msg = Message {
            role: "tool".to_string(),
            content: std::mem::take(&mut self.req.content),
            ..Default::default()
        };

        self.req.chat_history.push(msg.clone());
        self.chat_history.push(msg);
        self.req.role = None;
    }

    async fn inner_next(&mut self) -> Result<Option<AgentOutput>, BoxError> {
        let mut pending_tool_calls = false;
        if !self.pending_tool_calls.is_empty()
            && let Some(content) = self.drain_steering_message()
        {
            self.discard_in_flight_request_with_interrupted_tool_outputs(
                "tool call interrupted by steering",
                None,
            );
            self.req.content = content;
            self.req.role = Some("user".to_string());
        } else if !self.has_request_input() {
            // Automatically execute the pending tool/agent calls.
            if self.execute_pending_tool_calls_into_request().await? {
                pending_tool_calls = true;
                let follow_up_content: Vec<ContentPart> =
                    self.drain_follow_up_message().unwrap_or_default();

                if !follow_up_content.is_empty() {
                    if self.req.content.is_empty() {
                        self.set_next_user_content(follow_up_content);
                    } else {
                        self.commit_tool_outputs_to_history();
                        self.set_next_user_content(follow_up_content);
                    }
                }
            } else if let Some(content) = self.drain_queued_message() {
                self.set_next_user_content(content);
            } else {
                return Ok(None);
            }
        }

        self.turns += 1;
        let mut req = self.req.clone();
        if !pending_tool_calls && let Some(implicit_context) = self.implicit_context.take() {
            req.chat_history.push(implicit_context);
        }
        self.merge_discovered_tools_into_request(&mut req);

        let label = req.model.as_ref().unwrap_or(&self.ctx.label);
        if let Some(model) = self.ctx.models.resolve(label) {
            self.model = model;
        }

        let mut output = self.model.completion(req).await?;
        output.model = Some(self.model.model_name());

        self.current_usage = output.usage.clone();
        self.accumulate(&output.usage);

        if output.failed_reason.is_some() {
            return Ok(Some(self.final_output(output)));
        }

        // Clear one-shot constraints before preparing the next turn.
        self.req.tool_choice_required = false;
        self.req.output_schema = None;
        self.req.chat_history.clear();
        self.req.documents.clear();
        self.req.content.clear();
        self.req.prompt.clear();
        self.req.role = None;
        // Accumulate all raw history, including the original request history.
        let raw_history_start = self.req.raw_history.len();
        self.req.raw_history.append(&mut output.raw_history);
        self.pending_tool_call_raw_history_start = None;
        // Accumulate all generated chat history, excluding the original request history.
        self.chat_history.append(&mut output.chat_history);

        if let Some(content) = self.drain_steering_message() {
            if !output.tool_calls.is_empty() {
                self.append_interrupted_tool_outputs("tool call interrupted by steering", None);
                // Drop unanswered raw tool-call requests so the redirected round does not inherit
                // an unfinished tool-call requirement.
                Self::prune_unanswered_tool_calls_from_raw_history(
                    &mut self.req.raw_history,
                    raw_history_start,
                );
            }
            // Clear pending tool calls since the operator's steering should take priority and interrupt the current flow, even if there are still pending tool calls.
            self.pending_tool_calls.clear();
            self.set_next_user_content(content);
            return Ok(Some(self.intermediate_output(output)));
        }

        self.pending_tool_calls.extend(output.tool_calls.clone());
        if !self.pending_tool_calls.is_empty() {
            self.pending_tool_call_raw_history_start = Some(raw_history_start);
            // run tool calls in next turn
            return Ok(Some(self.intermediate_output(output)));
        }

        if let Some(content) = self.drain_queued_message() {
            self.set_next_user_content(content);
            return Ok(Some(self.intermediate_output(output)));
        }

        if self.unbound {
            return Ok(Some(self.intermediate_output(output)));
        }

        Ok(Some(self.final_output(output)))
    }

    fn intermediate_output(&mut self, mut output: AgentOutput) -> AgentOutput {
        output.usage = self.total_usage.clone();
        output.tools_usage = self.tools_usage.clone();
        output.chat_history = self.chat_history.clone();
        self.last_output = Some(output.clone());
        output
    }

    fn final_idle_output(&mut self) -> AgentOutput {
        self.done = true;
        let mut output = self.last_output.take().unwrap_or_default();
        output.chat_history = std::mem::take(&mut self.chat_history);
        output.tool_calls = std::mem::take(&mut self.tool_calls);
        output.artifacts = std::mem::take(&mut self.artifacts);
        output.usage = std::mem::take(&mut self.total_usage);
        output.tools_usage = std::mem::take(&mut self.tools_usage);

        output
    }

    fn final_output(&mut self, mut output: AgentOutput) -> AgentOutput {
        self.done = true;
        self.last_output = None;
        self.chat_history.append(&mut output.chat_history);
        output.chat_history = std::mem::take(&mut self.chat_history);
        output.tool_calls = std::mem::take(&mut self.tool_calls);
        output.artifacts = std::mem::take(&mut self.artifacts);
        output.usage = std::mem::take(&mut self.total_usage);
        output.tools_usage = std::mem::take(&mut self.tools_usage);

        output
    }
}

/// Stream wrapper for [`CompletionRunner`].
///
/// Note that a stream is terminal after yielding `None`. If you need resumable idle behavior via
/// `set_unbound(true)`, drive [`CompletionRunner::next`] directly instead of using this stream.
///
/// While a step is in flight, the `runner` field holds an inert placeholder, so queue mid-run
/// messages through [`CompletionStream::steer`] and [`CompletionStream::follow_up`] instead of
/// calling the runner directly; they are delivered to the live runner when the step completes.
pub struct CompletionStream {
    /// Runner owned by the stream between poll steps.
    pub runner: CompletionRunner,
    pending: Option<PendingCompletion>,
    queued_steering: Vec<ContentPart>,
    queued_follow_up: Vec<ContentPart>,
}

type PendingCompletion = BoxPinFut<(CompletionRunner, Result<Option<AgentOutput>, BoxError>)>;

impl CompletionStream {
    pub(crate) fn new(runner: CompletionRunner) -> Self {
        Self {
            runner,
            pending: None,
            queued_steering: Vec::new(),
            queued_follow_up: Vec::new(),
        }
    }

    /// Queue a steering message, buffering it while a step is in flight.
    pub fn steer(&mut self, message: impl Into<ContentPart>) {
        if self.pending.is_none() {
            self.runner.steer(message);
        } else {
            self.queued_steering.push(message.into());
        }
    }

    /// Queue a follow-up message, buffering it while a step is in flight.
    pub fn follow_up(&mut self, message: impl Into<ContentPart>) {
        if self.pending.is_none() {
            self.runner.follow_up(message);
        } else {
            self.queued_follow_up.push(message.into());
        }
    }

    fn restore_runner(&mut self, runner: CompletionRunner) {
        self.runner = runner;
        self.pending = None;
        if !self.queued_follow_up.is_empty() {
            self.runner
                .follow_up_content(std::mem::take(&mut self.queued_follow_up));
        }
        if !self.queued_steering.is_empty() {
            self.runner
                .steer_content(std::mem::take(&mut self.queued_steering));
        }
    }
}

impl Stream for CompletionStream {
    type Item = Result<AgentOutput, BoxError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.pending.is_none() {
            let placeholder = this.runner.stream_placeholder();
            let mut runner = std::mem::replace(&mut this.runner, placeholder);
            this.pending = Some(Box::pin(async move {
                let res = runner.next().await;
                (runner, res)
            }));
        }

        let pending = this
            .pending
            .as_mut()
            .expect("completion stream pending future must be initialized");
        match pending.as_mut().poll(cx) {
            Poll::Ready((runner, res)) => {
                this.restore_runner(runner);
                match res {
                    Ok(Some(output)) => Poll::Ready(Some(Ok(output))),
                    Ok(None) => Poll::Ready(None),
                    Err(e) => Poll::Ready(Some(Err(e))),
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn merge_visible_group(
    groups: &mut BTreeMap<String, ToolGroup>,
    mut group: ToolGroup,
    visible_names: &BTreeMap<String, String>,
) {
    let id = group.id.trim();
    if id.is_empty() || visible_names.is_empty() {
        return;
    }

    let mut seen_members = BTreeSet::new();
    let mut members = group
        .members
        .into_iter()
        .filter_map(|member| {
            let lowercase = member.trim().to_ascii_lowercase();
            if lowercase.is_empty() || !seen_members.insert(lowercase.clone()) {
                return None;
            }
            visible_names.get(&lowercase).cloned()
        })
        .collect::<Vec<_>>();
    if members.is_empty() {
        return;
    }
    members.sort_by_key(|name| name.to_ascii_lowercase());

    let key = id.to_ascii_lowercase();
    match groups.get_mut(&key) {
        Some(existing) => {
            let mut existing_members = existing
                .members
                .iter()
                .map(|member| member.to_ascii_lowercase())
                .collect::<BTreeSet<_>>();
            for member in members {
                if existing_members.insert(member.to_ascii_lowercase()) {
                    existing.members.push(member);
                }
            }
            existing
                .members
                .sort_by_key(|name| name.to_ascii_lowercase());
        }
        None => {
            group.id = id.to_string();
            group.members = members;
            groups.insert(key, group);
        }
    }
}

#[cfg(test)]
mod tests {
    use anda_core::{
        Agent, AgentContext as _, AgentInput, AgentOutput, BaseContext as _, BoxError,
        CacheFeatures as _, CacheStoreFeatures as _, CancellationToken, CanisterCaller as _,
        CompletionFeatures as _, CompletionRequest, ContentPart, Function, FunctionDefinition,
        HttpFeatures as _, Json, KeysFeatures as _, Message, ModelEffort, Path, PutMode, Resource,
        StateFeatures as _, StoreFeatures as _, Tool, ToolCall, ToolInput, ToolOutput, Usage,
    };
    use bytes::Bytes;
    use candid::Principal;
    use cbor2::from_slice;
    use futures_util::StreamExt;
    use ic_cose_types::to_cbor_bytes;
    use serde::Deserialize;
    use serde_json::json;
    use std::{
        collections::{BTreeMap, HashMap},
        sync::{Arc, Mutex},
    };

    use super::{
        AgentCtx, CompletionRunner, DYNAMIC_REMOTE_ENGINES, REMOTE_AGENT_PREFIX,
        REMOTE_TOOL_PREFIX, SUB_AGENT_PREFIX,
    };
    use crate::context::base::BaseCtx;
    use crate::context::engine::{AgentInfo, EngineCard, RemoteEngines};
    use crate::{
        engine::EngineBuilder,
        model::{CompletionFeaturesDyn, Model},
        subagent::{SubAgent, SubAgentManager},
    };

    #[test]
    fn json_in_cbor_works() {
        let json = json!({
            "level": "info",
            "message": "Hello, world!",
            "timestamp": "2021-09-01T12:00:00Z",
            "data": {
                "key": "value",
                "number": 42,
                "flag": true
            }
        });
        let data = to_cbor_bytes(&json);
        let val: serde_json::Value = from_slice(&data[..]).unwrap();
        assert_eq!(json, val);
    }

    // ── Helper completers ──

    #[test]
    fn agent_child_context_switches_agent_namespace_and_preserves_state() {
        let ctx = EngineBuilder::new().mock_ctx();
        ctx.base.set_state("parent-state".to_string());

        let child = ctx.child("worker_agent", "Worker").unwrap();
        assert_eq!(ctx.base.agent, "Mocker");
        assert_eq!(child.base.agent, "worker_agent");
        assert_eq!(child.label, "Worker");
        assert_eq!(child.base.path.as_ref(), "a_worker_agent");
        assert_eq!(
            child.base.get_state::<String>().as_deref(),
            Some("parent-state")
        );

        let tool_ctx = child.child_base("note").unwrap();
        assert_eq!(tool_ctx.agent, "worker_agent");
        assert_eq!(tool_ctx.path.as_ref(), "t_note");
        assert_eq!(
            tool_ctx.get_state::<String>().as_deref(),
            Some("parent-state")
        );
    }

    #[derive(Clone, Debug)]
    struct AlwaysFailCompleter;

    impl CompletionFeaturesDyn for AlwaysFailCompleter {
        fn model_name(&self) -> String {
            "always_fail".to_string()
        }

        fn completion(
            &self,
            _req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Ok(AgentOutput {
                failed_reason: Some("primary failed".to_string()),
                ..Default::default()
            })))
        }
    }

    /// Completer that echoes prompt as content, no tool calls.
    #[derive(Clone, Debug)]
    struct EchoCompleter;

    impl CompletionFeaturesDyn for EchoCompleter {
        fn model_name(&self) -> String {
            "echo".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            let content = if req.prompt.is_empty() {
                req.content
                    .iter()
                    .map(|part| match part {
                        anda_core::ContentPart::Text { text }
                        | anda_core::ContentPart::Reasoning { text } => text.clone(),
                        _ => serde_json::to_string(part).unwrap_or_default(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                req.prompt.clone()
            };

            Box::pin(futures::future::ready(Ok(AgentOutput {
                content,
                usage: Usage {
                    input_tokens: 5,
                    output_tokens: 10,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    /// Completer that returns tool calls on the first call, then echoes on subsequent calls.
    #[derive(Clone, Debug)]
    struct ToolCallCompleter {
        tool_calls: Vec<ToolCall>,
    }

    impl CompletionFeaturesDyn for ToolCallCompleter {
        fn model_name(&self) -> String {
            "tool_call".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            // If the request role is "tool", it means we already executed tools,
            // so respond with final content.
            let role = req.role.as_deref().unwrap_or("");
            if role == "tool" {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    content: "tool_result_processed".to_string(),
                    usage: Usage {
                        input_tokens: 3,
                        output_tokens: 6,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            let tool_calls = self.tool_calls.clone();
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content: String::new(),
                tool_calls,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct ToolChainUntilFollowUpCompleter {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for ToolChainUntilFollowUpCompleter {
        fn model_name(&self) -> String {
            "tool_chain_until_follow_up".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            let request_index = {
                let mut requests = self.requests.lock().unwrap();
                requests.push(req.clone());
                requests.len()
            };
            let saw_follow_up = req.content.iter().any(|part| {
                matches!(
                    part,
                    ContentPart::Text { text }
                        if text == "follow up while tool chain is pending"
                )
            });

            if saw_follow_up {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    content: "follow_up_seen_with_tool_result".to_string(),
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "chain"}),
                    call_id: Some(format!("chain_call_{request_index}")),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct DiscoveryCompleter {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for DiscoveryCompleter {
        fn model_name(&self) -> String {
            "discovery".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            self.requests.lock().unwrap().push(req.clone());

            if req.role.as_deref() == Some("tool")
                && req.content.iter().any(|part| {
                    matches!(
                        part,
                        ContentPart::ToolOutput { name, .. } if name == "echo_tool"
                    )
                })
            {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    content: "echo tool used after discovery".to_string(),
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            if req.role.as_deref() == Some("tool")
                && req
                    .tools
                    .iter()
                    .any(|tool| tool.name.as_str() == "echo_tool")
            {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    tool_calls: vec![ToolCall {
                        name: "echo_tool".to_string(),
                        args: json!({"input": "after-select"}),
                        call_id: Some("call_echo_after_select".into()),
                        result: None,
                        remote_id: None,
                    }],
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            if req.role.as_deref() == Some("tool") {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    tool_calls: vec![ToolCall {
                        name: "tools_select".to_string(),
                        args: json!({
                            "tools": ["echo_tool"],
                            "query": "",
                            "limit": 0
                        }),
                        call_id: Some("select_echo_tool_again".into()),
                        result: None,
                        remote_id: None,
                    }],
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "tools_select".to_string(),
                    args: json!({
                        "tools": ["echo_tool"],
                        "query": "",
                        "limit": 0
                    }),
                    call_id: Some("select_echo_tool".into()),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct DiscoveryCompactionCompleter {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for DiscoveryCompactionCompleter {
        fn model_name(&self) -> String {
            "discovery_compaction".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            self.requests.lock().unwrap().push(req.clone());

            let prompt = if req.prompt.is_empty() {
                req.content
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                            text.clone()
                        }
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                req.prompt.clone()
            };

            if prompt.trim() == super::COMPACTION_PROMPT.trim() {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    content: "compacted handoff".to_string(),
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            if req.role.as_deref() != Some("tool")
                && req
                    .tools
                    .iter()
                    .any(|tool| tool.name.as_str() == "echo_tool")
            {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    tool_calls: vec![ToolCall {
                        name: "echo_tool".to_string(),
                        args: json!({"input": "after-handoff"}),
                        call_id: Some("call_echo_after_handoff".into()),
                        result: None,
                        remote_id: None,
                    }],
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "tools_select".to_string(),
                    args: json!({
                        "tools": ["echo_tool"],
                        "query": "",
                        "limit": 0
                    }),
                    call_id: Some("select_echo_tool".into()),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    /// Completer that returns agent calls.
    #[derive(Clone, Debug)]
    struct AgentCallCompleter {
        agent_name: String,
    }

    impl CompletionFeaturesDyn for AgentCallCompleter {
        fn model_name(&self) -> String {
            "agent_call".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            let role = req.role.as_deref().unwrap_or("");
            if role == "tool" {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    content: "agent_result_processed".to_string(),
                    usage: Usage {
                        input_tokens: 2,
                        output_tokens: 4,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            let agent_name = self.agent_name.clone();
            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: agent_name,
                    args: json!({"prompt": "subagent task"}),
                    call_id: Some("agent_call_1".into()),
                    result: None,
                    remote_id: None,
                }],
                usage: Usage {
                    input_tokens: 8,
                    output_tokens: 16,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    /// Completer that returns an Err (not failed_reason, but actual error).
    #[derive(Clone, Debug)]
    struct ErrorCompleter;

    impl CompletionFeaturesDyn for ErrorCompleter {
        fn model_name(&self) -> String {
            "error".to_string()
        }

        fn completion(
            &self,
            _req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Err("model error".into())))
        }
    }

    #[derive(Clone, Debug)]
    struct ToolResultErrorCompleter;

    impl CompletionFeaturesDyn for ToolResultErrorCompleter {
        fn model_name(&self) -> String {
            "tool_result_error".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            if req.role.as_deref() == Some("tool") {
                return Box::pin(futures::future::ready(Err("model error".into())));
            }

            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "hello"}),
                    call_id: Some("call_1".into()),
                    result: None,
                    remote_id: None,
                }],
                raw_history: vec![json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo_tool",
                            "arguments": "{\"input\":\"hello\"}"
                        }
                    }]
                })],
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct ToolCallHistoryCompleter {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for ToolCallHistoryCompleter {
        fn model_name(&self) -> String {
            "tool_call_history".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            self.requests.lock().unwrap().push(req.clone());
            let text = req
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                        Some(text.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if req.prompt == "start tool" || text.contains("start tool") {
                let call = ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "stop"}),
                    call_id: Some("call_stop_test".to_string()),
                    result: None,
                    remote_id: None,
                };
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    tool_calls: vec![call.clone()],
                    chat_history: vec![
                        Message {
                            role: "user".to_string(),
                            content: vec![ContentPart::Text {
                                text: "start tool".to_string(),
                            }],
                            ..Default::default()
                        },
                        Message {
                            role: "assistant".to_string(),
                            content: vec![
                                ContentPart::Text {
                                    text: "planning before tool".to_string(),
                                },
                                ContentPart::ToolCall {
                                    name: call.name,
                                    args: call.args,
                                    call_id: call.call_id,
                                },
                            ],
                            ..Default::default()
                        },
                    ],
                    raw_history: vec![
                        json!({
                            "role": "assistant",
                            "content": "planning before tool",
                            "tool_calls": [{
                                "id": "call_stop_test",
                                "type": "function",
                                "function": {
                                    "name": "echo_tool",
                                    "arguments": "{\"input\":\"stop\"}"
                                }
                            }]
                        }),
                        json!({
                            "type": "function_call",
                            "call_id": "call_stop_test",
                            "name": "echo_tool",
                            "arguments": "{\"input\":\"stop\"}"
                        }),
                    ],
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "continued".to_string(),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    /// Completer that waits forever (for cancellation tests).
    #[derive(Clone, Debug)]
    struct SlowCompleter;

    impl CompletionFeaturesDyn for SlowCompleter {
        fn model_name(&self) -> String {
            "slow".to_string()
        }

        fn completion(
            &self,
            _req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(async {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                Ok(AgentOutput::default())
            })
        }
    }

    /// Completer that completes after a short delay, exercising stream pending state.
    #[derive(Clone, Debug)]
    struct DelayedEchoCompleter;

    impl CompletionFeaturesDyn for DelayedEchoCompleter {
        fn model_name(&self) -> String {
            "delayed_echo".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                Ok(AgentOutput {
                    content: req.prompt,
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })
            })
        }
    }

    #[derive(Clone, Debug)]
    struct RawHistoryToolCallCompleter {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for RawHistoryToolCallCompleter {
        fn model_name(&self) -> String {
            "raw_history_tool_call".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            self.requests.lock().unwrap().push(req.clone());

            if req.role.as_deref() == Some("user") {
                return Box::pin(futures::future::ready(Ok(AgentOutput {
                    content: "steered".to_string(),
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                })));
            }

            Box::pin(futures::future::ready(Ok(AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "raw history"}),
                    call_id: Some("raw_call".into()),
                    result: None,
                    remote_id: None,
                }],
                raw_history: vec![
                    json!({
                        "role": "assistant",
                        "content": "planning tool call",
                        "tool_calls": [{
                            "id": "raw_call",
                            "type": "function",
                            "function": {
                                "name": "echo_tool",
                                "arguments": "{\"input\":\"raw history\"}"
                            }
                        }],
                        "reasoning": "keep this reasoning"
                    }),
                    json!({"type": "function_call", "call_id": "raw_call"}),
                ],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    #[derive(Clone, Debug)]
    struct RecordingCompleter {
        name: String,
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl CompletionFeaturesDyn for RecordingCompleter {
        fn model_name(&self) -> String {
            self.name.clone()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            self.requests.lock().unwrap().push(req.clone());
            let content = if req.prompt.is_empty() {
                req.content
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                            text.clone()
                        }
                        _ => serde_json::to_string(part).unwrap_or_default(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                req.prompt
            };

            Box::pin(futures::future::ready(Ok(AgentOutput {
                content,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })))
        }
    }

    // ── Helper tool ──

    struct EchoTool;

    #[derive(Debug, Deserialize)]
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
            "Echoes input back".to_string()
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: "echo_tool".to_string(),
                description: "Echoes input back".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
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
        ) -> Result<ToolOutput<String>, BoxError> {
            Ok(ToolOutput {
                output: format!("echoed:{}", args.input),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })
        }
    }

    /// A tool that always fails.
    struct FailTool;

    #[derive(Debug, Deserialize)]
    struct FailToolArgs {}

    impl Tool<BaseCtx> for FailTool {
        type Args = FailToolArgs;
        type Output = String;

        fn name(&self) -> String {
            "fail_tool".to_string()
        }

        fn description(&self) -> String {
            "Always fails".to_string()
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: "fail_tool".to_string(),
                description: "Always fails".to_string(),
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
        ) -> Result<ToolOutput<String>, BoxError> {
            Err("tool execution failed".into())
        }
    }

    // ── Helper agent ──

    struct EchoAgent;

    impl Agent<AgentCtx> for EchoAgent {
        fn name(&self) -> String {
            "echo_agent".to_string()
        }

        fn description(&self) -> String {
            "Echoes prompt back".to_string()
        }

        async fn run(
            &self,
            _ctx: AgentCtx,
            prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: format!("agent_echoed:{}", prompt),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cached_tokens: 0,
                    requests: 1,
                },
                ..Default::default()
            })
        }
    }

    struct FailAgent;

    impl Agent<AgentCtx> for FailAgent {
        fn name(&self) -> String {
            "fail_agent".to_string()
        }

        fn description(&self) -> String {
            "Always fails".to_string()
        }

        async fn run(
            &self,
            _ctx: AgentCtx,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Err("agent execution failed".into())
        }
    }

    fn function(name: &str, description: &str, tags: &[&str]) -> Function {
        Function {
            definition: FunctionDefinition {
                name: name.to_string(),
                description: description.to_string(),
                parameters: json!({"type": "object"}),
                ..Default::default()
            },
            supported_resource_tags: tags.iter().map(|tag| tag.to_string()).collect(),
        }
    }

    fn resource(id: u64, tags: &[&str]) -> Resource {
        Resource {
            _id: id,
            name: format!("resource-{id}"),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        }
    }

    fn dynamic_remote_engines() -> RemoteEngines {
        let mut engines = BTreeMap::new();
        engines.insert(
            "dyn".to_string(),
            EngineCard {
                id: Principal::self_authenticating([9; 32]),
                info: AgentInfo {
                    handle: "Dynamic".to_string(),
                    endpoint: "https://dynamic.example".to_string(),
                    ..Default::default()
                },
                agents: vec![function("chat", "Chat remotely", &["md"])],
                tools: vec![function("lookup", "Lookup remotely", &["text"])],
            },
        );
        RemoteEngines { engines }
    }

    // ── Tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn agent_context_definitions_dynamic_resources_and_missing_runs() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .register_agent(Arc::new(EchoAgent), None)
            .unwrap()
            .mock_ctx();

        let empty: Vec<String> = Vec::new();
        assert!(ctx.tool_definitions(Some(&empty)).is_empty());
        assert!(ctx.agent_definitions(Some(&empty)).is_empty());
        assert!(
            ctx.remote_tool_definitions(None, Some(&empty))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            ctx.remote_agent_definitions(None, Some(&empty))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(ctx.definitions(Some(&empty)).await.is_empty());

        ctx.root
            .cache_store_set(DYNAMIC_REMOTE_ENGINES, dynamic_remote_engines(), None)
            .await
            .unwrap();

        let definitions = ctx.definitions(None).await;
        assert!(definitions.iter().any(|d| d.name == "echo_tool"));
        assert!(definitions.iter().any(|d| d.name == "echo_agent"));
        assert!(
            definitions
                .iter()
                .any(|d| d.name == format!("{REMOTE_TOOL_PREFIX}dyn_lookup"))
        );
        assert!(
            definitions
                .iter()
                .any(|d| d.name == format!("{REMOTE_AGENT_PREFIX}dyn_chat"))
        );

        let mut resources = vec![resource(1, &["text"]), resource(2, &["md"])];
        let selected = ctx
            .select_tool_resources(&format!("{REMOTE_TOOL_PREFIX}dyn_lookup"), &mut resources)
            .await;
        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            resources
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![2]
        );

        let selected = ctx
            .select_agent_resources(&format!("{REMOTE_AGENT_PREFIX}dyn_chat"), &mut resources)
            .await;
        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert!(resources.is_empty());

        let tool_err = ctx
            .tool_call(ToolInput {
                name: format!("{REMOTE_TOOL_PREFIX}dyn_lookup"),
                args: json!({}),
                resources: Vec::new(),
                meta: None,
            })
            .await
            .unwrap_err();
        assert!(
            tool_err
                .to_string()
                .contains("remote engine endpoint https://dynamic.example not found")
        );

        let agent_err = ctx
            .clone()
            .agent_run(AgentInput {
                name: format!("{REMOTE_AGENT_PREFIX}dyn_chat"),
                prompt: "hello".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(
            agent_err
                .to_string()
                .contains("remote engine endpoint https://dynamic.example not found")
        );

        let agent_err = ctx
            .clone()
            .agent_run(AgentInput {
                name: format!("{REMOTE_AGENT_PREFIX}missing"),
                prompt: "hello".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(agent_err.to_string().contains("agent ra_missing not found"));

        let agent_err = ctx
            .clone()
            .agent_run(AgentInput {
                name: format!("{SUB_AGENT_PREFIX}missing"),
                prompt: "hello".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(agent_err.to_string().contains("agent sa_missing not found"));

        let agent_err = ctx
            .agent_run(AgentInput {
                name: "missing_agent".to_string(),
                prompt: "hello".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(
            agent_err
                .to_string()
                .contains("agent missing_agent not found")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn agent_context_trait_forwarders_cover_base_store_cache_keys_and_http() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        assert_eq!(ctx.model_name(), "echo");
        assert_eq!(ctx.engine_name(), "Mocker");
        assert_eq!(*ctx.engine_id(), Principal::anonymous());
        assert_eq!(*ctx.caller(), Principal::anonymous());
        assert!(ctx.meta().user.is_none());
        assert!(!ctx.cancellation_token().is_cancelled());
        assert!(ctx.time_elapsed() < std::time::Duration::from_secs(60));

        let caller = Principal::self_authenticating([4; 32]);
        let called_by = ctx.with_caller(caller);
        assert_eq!(*called_by.caller(), caller);

        let path = Path::from("agent_ctx_file");
        let renamed = Path::from("agent_ctx_file_renamed");
        ctx.store_put(&path, PutMode::Overwrite, Bytes::from_static(b"data"))
            .await
            .unwrap();
        let (stored, meta) = ctx.store_get(&path).await.unwrap();
        assert_eq!(stored, Bytes::from_static(b"data"));
        assert_eq!(meta.location, path);
        ctx.store_rename_if_not_exists(&path, &renamed)
            .await
            .unwrap();
        let listed = ctx.store_list(None, &Path::from("")).await.unwrap();
        assert!(listed.iter().any(|meta| meta.location == renamed));
        ctx.store_delete(&renamed).await.unwrap();
        assert!(ctx.store_get(&renamed).await.is_err());

        assert!(
            ctx.cache_get_with("missing_path", async {
                Ok::<_, BoxError>(("created".to_string(), None))
            })
            .await
            .unwrap_err()
            .to_string()
            .contains("cache path")
        );

        let cache_ctx = ctx.child("tools_search", "Tools Search").unwrap();
        assert!(!cache_ctx.cache_contains("number"));
        cache_ctx.cache_set("number", (42_u64, None)).await;
        assert_eq!(cache_ctx.cache_get::<u64>("number").await.unwrap(), 42);
        let initialized: String = cache_ctx
            .cache_get_with("initialized", async {
                Ok::<_, BoxError>(("created".to_string(), None))
            })
            .await
            .unwrap();
        assert_eq!(initialized, "created");
        let cache_keys = cache_ctx
            .cache_raw_iter()
            .map(|(key, _)| key.as_str().to_string())
            .collect::<Vec<_>>();
        assert!(cache_keys.contains(&"number".to_string()));
        assert!(cache_ctx.cache_delete("number").await);
        assert!(!cache_ctx.cache_contains("number"));

        assert!(ctx.a256gcm_key(Vec::new()).await.is_err());
        assert!(ctx.ed25519_sign_message(Vec::new(), b"msg").await.is_err());
        assert!(
            ctx.ed25519_verify(Vec::new(), b"msg", &[0; 64])
                .await
                .is_err()
        );
        assert!(ctx.ed25519_public_key(Vec::new()).await.is_err());
        assert!(
            ctx.secp256k1_sign_message_bip340(Vec::new(), b"msg")
                .await
                .is_err()
        );
        assert!(
            ctx.secp256k1_verify_bip340(Vec::new(), b"msg", &[0; 64])
                .await
                .is_err()
        );
        assert!(
            ctx.secp256k1_sign_message_ecdsa(Vec::new(), b"msg")
                .await
                .is_err()
        );
        assert!(
            ctx.secp256k1_sign_digest_ecdsa(Vec::new(), &[0; 32])
                .await
                .is_err()
        );
        assert!(
            ctx.secp256k1_verify_ecdsa(Vec::new(), &[0; 32], &[0; 64])
                .await
                .is_err()
        );
        assert!(ctx.secp256k1_public_key(Vec::new()).await.is_err());

        assert!(
            ctx.canister_query::<_, ()>(&Principal::anonymous(), "status", ())
                .await
                .is_err()
        );
        assert!(
            ctx.canister_update::<_, ()>(&Principal::anonymous(), "update", ())
                .await
                .is_err()
        );
        assert!(
            ctx.https_call("https://example.test", http::Method::GET, None, None)
                .await
                .is_err()
        );
        assert!(
            ctx.https_signed_call(
                "https://example.test",
                http::Method::POST,
                [0; 32],
                None,
                Some(Vec::new()),
            )
            .await
            .is_err()
        );
        let rpc: Result<Json, BoxError> = ctx
            .https_signed_rpc("https://example.test", "method", &())
            .await;
        assert!(rpc.is_err());

        let err = ctx
            .remote_tool_call(
                "https://missing.example",
                ToolInput {
                    name: "lookup".to_string(),
                    args: json!({}),
                    resources: Vec::new(),
                    meta: None,
                },
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("remote engine endpoint https://missing.example not found")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_accessors_mutators_and_implicit_context_are_observable() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingCompleter {
            name: "recording".to_string(),
            requests: requests.clone(),
        }))
        .with_labels(vec!["alt".to_string()]);
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let mut runner = ctx.completion_iter(CompletionRequest::default(), Vec::new());

        assert!(runner.is_idle());
        assert_eq!(runner.ctx().engine_name(), "Mocker");
        assert_eq!(runner.req().prompt, "");
        assert_eq!(runner.model().model_name(), "recording");
        assert_eq!(runner.total_usage().requests, 0);
        assert_eq!(runner.current_usage().requests, 0);
        assert!(runner.tools_usage().is_empty());
        assert!(runner.last_output().is_none());
        assert_eq!(runner.merge_discovered_tools(), None);

        runner.append_chat_history(vec![Message {
            role: "system".to_string(),
            content: vec![ContentPart::Text {
                text: "preloaded".to_string(),
            }],
            ..Default::default()
        }]);
        assert_eq!(runner.chat_history().len(), 1);
        runner.chat_history_mut()[0].content = vec![ContentPart::Text {
            text: "updated".to_string(),
        }];
        assert_eq!(runner.chat_history().len(), 1);
        assert_eq!(runner.chat_history()[0].text().as_deref(), Some("updated"));

        runner.follow_up_content(vec![ContentPart::Text {
            text: "follow".to_string(),
        }]);
        runner.steer_content(vec![ContentPart::Text {
            text: "steer".to_string(),
        }]);
        assert!(!runner.is_idle());
        runner.implicit_context(Message {
            role: "system".to_string(),
            content: vec![ContentPart::Text {
                text: "implicit".to_string(),
            }],
            ..Default::default()
        });
        runner.set_model(Some("alt".to_string()));
        runner.set_effort(Some(ModelEffort::Low));
        runner.set_merge_discovered_tools(Some(true));
        runner.set_tools(vec![FunctionDefinition {
            name: "forced_tool".to_string(),
            ..Default::default()
        }]);
        assert_eq!(runner.merge_discovered_tools(), Some(true));

        let output = runner.next().await.unwrap().unwrap();
        assert_eq!(output.content, "follow\n\nsteer");
        assert_eq!(output.model, Some("recording".to_string()));
        assert_eq!(runner.current_usage().requests, 1);
        assert_eq!(output.usage.requests, 1);
        assert_eq!(runner.total_usage().requests, 0);
        assert!(runner.last_output().is_none());

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model.as_deref(), Some("alt"));
        assert_eq!(requests[0].effort, Some(ModelEffort::Low));
        assert_eq!(requests[0].tools[0].name, "forced_tool");
        assert_eq!(requests[0].chat_history.len(), 1);
        assert_eq!(
            requests[0].chat_history[0].text().as_deref(),
            Some("implicit")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_records_missing_tool_and_failed_agent_as_tool_outputs() {
        let model = Model::with_completer(Arc::new(ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "missing_tool".to_string(),
                args: json!({}),
                call_id: Some("missing_tool_call".into()),
                result: None,
                remote_id: None,
            }],
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let req = CompletionRequest {
            prompt: "call missing".to_string(),
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());
        let step = runner.next().await.unwrap().unwrap();
        assert_eq!(step.tool_calls.len(), 1);
        assert!(step.tool_calls[0].result.is_none());
        let output = runner.next().await.unwrap().unwrap();
        let result = output.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(result.is_error, Some(true));
        assert!(
            result
                .output
                .to_string()
                .contains("tool call failed: missing_tool not found")
        );

        let model = Model::with_completer(Arc::new(AgentCallCompleter {
            agent_name: "fail_agent".to_string(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_agent(Arc::new(FailAgent), None)
            .unwrap()
            .mock_ctx();
        let req = CompletionRequest {
            prompt: "call failing agent".to_string(),
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.next().await.unwrap().unwrap();
        let output = runner.next().await.unwrap().unwrap();
        let result = output.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(result.is_error, Some(true));
        assert!(
            result
                .output
                .to_string()
                .contains("agent run failed: agent execution failed")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_injects_discovered_tool_schemas_after_repeated_selection() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(DiscoveryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let initial_tools = ctx.definitions(Some(&["tools_select".to_string()])).await;
        let req = CompletionRequest {
            prompt: "select echo tool".to_string(),
            tools: initial_tools,
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());

        let first = runner.next().await.unwrap().unwrap();
        assert_eq!(first.tool_calls[0].name, "tools_select");

        let second = runner.next().await.unwrap().unwrap();
        assert_eq!(second.tool_calls[0].name, "tools_select");

        let third = runner.next().await.unwrap().unwrap();
        assert_eq!(third.tool_calls[0].name, "echo_tool");
        assert_eq!(runner.merge_discovered_tools(), Some(true));

        let fourth = runner.next().await.unwrap().unwrap();
        assert_eq!(fourth.content, "echo tool used after discovery");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        let initial_tool_names = requests[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert!(initial_tool_names.contains(&"tools_select"));
        assert!(!initial_tool_names.contains(&"echo_tool"));

        let first_after_select_tool_names = requests[1]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert!(first_after_select_tool_names.contains(&"tools_select"));
        assert!(!first_after_select_tool_names.contains(&"echo_tool"));
        assert!(requests[1].content.iter().any(|part| matches!(
            part,
            ContentPart::ToolOutput { name, output, .. }
                if name == "tools_select" && output["tools"][0]["name"] == "echo_tool"
                    && output["tools"][0].get("parameters").is_some()
        )));

        let second_after_select_tool_names = requests[2]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert!(second_after_select_tool_names.contains(&"tools_select"));
        assert!(second_after_select_tool_names.contains(&"echo_tool"));
        assert!(requests[2].content.iter().any(|part| matches!(
            part,
            ContentPart::ToolOutput { name, output, .. }
                if name == "tools_select"
                    && output["tools"][0]["name"] == "echo_tool"
                    && output["tools"][0].get("parameters").is_none()
                    && output["tools"][0].get("description").is_none()
        )));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_can_merge_discovered_tool_schemas_without_repeated_selection_probe() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(DiscoveryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let initial_tools = ctx.definitions(Some(&["tools_select".to_string()])).await;
        let req = CompletionRequest {
            prompt: "select echo tool".to_string(),
            tools: initial_tools,
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.set_merge_discovered_tools(Some(true));

        let first = runner.next().await.unwrap().unwrap();
        assert_eq!(first.tool_calls[0].name, "tools_select");

        let second = runner.next().await.unwrap().unwrap();
        assert_eq!(second.tool_calls[0].name, "echo_tool");

        let third = runner.next().await.unwrap().unwrap();
        assert_eq!(third.content, "echo tool used after discovery");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        let after_first_select_tool_names = requests[1]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert!(after_first_select_tool_names.contains(&"tools_select"));
        assert!(after_first_select_tool_names.contains(&"echo_tool"));
        assert!(requests[1].content.iter().any(|part| matches!(
            part,
            ContentPart::ToolOutput { name, output, .. }
                if name == "tools_select"
                    && output["tools"][0]["name"] == "echo_tool"
                    && output["tools"][0].get("parameters").is_none()
                    && output["tools"][0].get("description").is_none()
        )));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_does_not_probe_when_discovered_tool_merge_is_disabled() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(DiscoveryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let initial_tools = ctx.definitions(Some(&["tools_select".to_string()])).await;
        let req = CompletionRequest {
            prompt: "select echo tool".to_string(),
            tools: initial_tools,
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.set_merge_discovered_tools(Some(false));

        let first = runner.next().await.unwrap().unwrap();
        assert_eq!(first.tool_calls[0].name, "tools_select");

        let second = runner.next().await.unwrap().unwrap();
        assert_eq!(second.tool_calls[0].name, "tools_select");

        let third = runner.next().await.unwrap().unwrap();
        assert_eq!(third.tool_calls[0].name, "tools_select");
        assert_eq!(runner.merge_discovered_tools(), Some(false));

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(
            requests[1]
                .tools
                .iter()
                .all(|tool| tool.name != "echo_tool")
        );
        assert!(
            requests[2]
                .tools
                .iter()
                .all(|tool| tool.name != "echo_tool")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_handoff_preserves_discovered_tool_state_after_pending_selection() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(DiscoveryCompactionCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let initial_tools = ctx.definitions(Some(&["tools_select".to_string()])).await;
        let req = CompletionRequest {
            prompt: "select echo tool".to_string(),
            tools: initial_tools,
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new()).unbound();

        let first = runner.next().await.unwrap().unwrap();
        assert_eq!(first.tool_calls[0].name, "tools_select");

        let second = runner.next().await.unwrap().unwrap();
        assert_eq!(second.tool_calls[0].name, "tools_select");
        assert_eq!(runner.merge_discovered_tools(), None);
        assert!(!runner.no_pending_tool_calls());

        runner.follow_up("continue after handoff".to_string());
        let (mut runner, output) = runner.handoff(None).await.unwrap();
        assert_eq!(output.content, "compacted handoff");
        assert_eq!(runner.merge_discovered_tools(), Some(true));

        let after_handoff = runner.next().await.unwrap().unwrap();
        assert_eq!(after_handoff.tool_calls[0].name, "echo_tool");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests[2].tools.is_empty());
        let after_handoff_tool_names = requests[3]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert!(after_handoff_tool_names.contains(&"tools_select"));
        assert!(after_handoff_tool_names.contains(&"echo_tool"));
        assert!(requests[3].content.iter().any(|part| matches!(
            part,
            ContentPart::Text { text } if text == "continue after handoff"
        )));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_compacts_discovery_outputs_after_schema_merge_is_enabled() {
        let ctx = EngineBuilder::new().mock_ctx();
        let mut runner = ctx.completion_iter(CompletionRequest::default(), Vec::new());
        runner.set_merge_discovered_tools(Some(true));
        let full_output = json!({
            "tools": [{
                "name": "echo_tool",
                "description": "Echoes input",
                "parameters": {"type": "object"},
                "strict": true
            }],
            "total_tools": 9
        });

        let mut search_output = full_output.clone();
        runner.compact_discovery_tool_output_for_context("tools_search", &mut search_output);
        assert_eq!(search_output["tools"][0]["name"], "echo_tool");
        assert_eq!(search_output["tools"][0]["description"], "Echoes input");
        assert!(search_output["tools"][0].get("parameters").is_none());
        assert!(search_output["tools"][0].get("strict").is_none());
        assert_eq!(search_output["total_tools"], 9);

        let mut select_output = full_output.clone();
        runner.compact_discovery_tool_output_for_context("tools_select", &mut select_output);
        assert_eq!(select_output["tools"][0]["name"], "echo_tool");
        assert!(select_output["tools"][0].get("description").is_none());
        assert!(select_output["tools"][0].get("parameters").is_none());
        assert!(select_output["tools"][0].get("strict").is_none());
        assert_eq!(select_output["total_tools"], 9);

        // Non-discovery tool outputs stay untouched.
        let mut other_output = full_output.clone();
        runner.compact_discovery_tool_output_for_context("echo_tool", &mut other_output);
        assert_eq!(other_output, full_output);

        // Without schema merge, discovery outputs also stay untouched.
        runner.set_merge_discovered_tools(Some(false));
        let mut unmerged_output = full_output.clone();
        runner.compact_discovery_tool_output_for_context("tools_search", &mut unmerged_output);
        assert_eq!(unmerged_output, full_output);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_accumulates_nested_tool_usage() {
        struct AccountingTool;

        #[derive(Debug, Deserialize)]
        struct AccountingArgs {}

        impl Tool<BaseCtx> for AccountingTool {
            type Args = AccountingArgs;
            type Output = String;

            fn name(&self) -> String {
                "accounting_tool".to_string()
            }

            fn description(&self) -> String {
                "Returns nested tool usage".to_string()
            }

            fn definition(&self) -> FunctionDefinition {
                FunctionDefinition {
                    name: "accounting_tool".to_string(),
                    description: "Returns nested tool usage".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: Some(true),
                }
            }

            async fn call(
                &self,
                _ctx: BaseCtx,
                _args: Self::Args,
                _resources: Vec<Resource>,
            ) -> Result<ToolOutput<String>, BoxError> {
                Ok(ToolOutput {
                    output: "accounted".to_string(),
                    usage: Usage {
                        input_tokens: 7,
                        output_tokens: 11,
                        cached_tokens: 3,
                        requests: 4,
                    },
                    tools_usage: HashMap::from([(
                        "nested_tool".to_string(),
                        Usage {
                            input_tokens: 2,
                            output_tokens: 3,
                            cached_tokens: 1,
                            requests: 2,
                        },
                    )]),
                    ..Default::default()
                })
            }
        }

        let model = Model::with_completer(Arc::new(ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "accounting_tool".to_string(),
                args: json!({}),
                call_id: Some("accounting_call".into()),
                result: None,
                remote_id: None,
            }],
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(AccountingTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "account".to_string(),
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.next().await.unwrap().unwrap();
        let output = runner.next().await.unwrap().unwrap();

        assert_eq!(output.tools_usage["accounting_tool"].requests, 1);
        assert_eq!(output.tools_usage["accounting_tool"].input_tokens, 7);
        assert_eq!(output.tools_usage["nested_tool"].requests, 2);
        assert_eq!(output.tools_usage["nested_tool"].cached_tokens, 1);
        assert!(output.usage.input_tokens >= 20);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_finalize_reports_already_finalized() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let req = CompletionRequest {
            prompt: "done".to_string(),
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.next().await.unwrap().unwrap();

        let err = runner.finalize(None).await.unwrap_err();
        assert!(err.to_string().contains("completion already finalized"));
    }

    #[test]
    fn runner_prunes_contextless_raw_history_items() {
        let mut raw_history = vec![
            json!(null),
            json!(""),
            json!({"role": "assistant", "id": "meta-only", "status": "ok"}),
            json!({"role": "assistant", "content": []}),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
                    {"text": "kept text"}
                ],
                "tool_calls": [{"id": "call_1"}],
                "function_call": {"name": "lookup"}
            }),
            json!({"type": "function_call", "call_id": "call_1"}),
            json!(42),
        ];

        CompletionRunner::prune_unanswered_tool_calls_from_raw_history(&mut raw_history, 0);

        assert_eq!(raw_history.len(), 2);
        assert_eq!(raw_history[0]["content"], json!([{"text": "kept text"}]));
        assert!(raw_history[0].get("tool_calls").is_none());
        assert!(raw_history[0].get("function_call").is_none());
        assert_eq!(raw_history[1], json!(42));
    }

    #[test]
    fn runner_prunes_deeply_nested_raw_tool_calls() {
        let sentinel = json!({"role": "user", "content": "prior"});
        let mut raw_history = vec![
            sentinel.clone(),
            json!({
                "role": "assistant",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "kept nested text"}
                        ]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_nested_function",
                        "name": "lookup",
                        "arguments": "{}"
                    },
                    {
                        "type": "local_shell_call",
                        "id": "lsh_1",
                        "action": {"type": "exec", "command": ["pwd"]},
                        "call_id": "call_nested_shell",
                        "status": "completed"
                    },
                    {
                        "type": "apply_patch_call",
                        "id": "ap_1",
                        "call_id": "call_nested_patch",
                        "operation": {"type": "update", "path": "src/lib.rs", "diff": "@@"},
                        "status": "completed"
                    }
                ],
                "metadata": {
                    "items": [
                        {
                            "type": "custom_tool_call",
                            "call_id": "call_nested_custom",
                            "input": "select 1",
                            "name": "sql"
                        },
                        {"note": "keep nested metadata"}
                    ]
                },
                "tool_calls": [{"id": "call_nested_chat"}],
                "function_call": {"name": "legacy"},
                "functionCall": {"name": "gemini"}
            }),
        ];

        CompletionRunner::prune_unanswered_tool_calls_from_raw_history(&mut raw_history, 1);

        assert_eq!(raw_history.len(), 2);
        assert_eq!(raw_history[0], sentinel);
        let pruned = serde_json::to_string(&raw_history[1]).unwrap();
        assert!(pruned.contains("kept nested text"));
        assert!(pruned.contains("keep nested metadata"));
        assert!(!pruned.contains("call_nested"));
        assert!(raw_history[1].get("tool_calls").is_none());
        assert!(raw_history[1].get("function_call").is_none());
        assert!(raw_history[1].get("functionCall").is_none());
    }

    #[test]
    fn runner_prunes_completed_tool_interactions_from_raw_history() {
        let mut raw_history = vec![
            // OpenAI Chat Completions shapes.
            json!({"role": "user", "content": "question"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "chat tool result"}),
            // OpenAI Responses shapes: orphaned reasoning must follow its pruned call out.
            json!({"type": "reasoning", "id": "rs_orphan", "encrypted_content": "opaque"}),
            json!({"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_2", "output": "ok"}),
            // Anthropic shapes.
            json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "kept thinking", "signature": "sig"},
                    {"type": "text", "text": "anthropic text"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
                ]
            }),
            json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "anthropic result"}
                ]
            }),
            // Gemini shapes.
            json!({
                "role": "model",
                "parts": [
                    {"text": "gemini text"},
                    {"functionCall": {"name": "lookup", "args": {}}}
                ]
            }),
            json!({
                "role": "user",
                "parts": [{"functionResponse": {"name": "lookup", "response": {"ok": true}}}]
            }),
            // Reasoning followed by a surviving item stays.
            json!({"type": "reasoning", "id": "rs_kept", "encrypted_content": "opaque"}),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "final answer"}]
            }),
        ];

        CompletionRunner::prune_tool_interactions_from_raw_history(&mut raw_history);

        let pruned = serde_json::to_string(&raw_history).unwrap();
        assert_eq!(raw_history.len(), 5);
        assert_eq!(raw_history[0]["content"], json!("question"));
        assert!(pruned.contains("kept thinking"));
        assert!(pruned.contains("anthropic text"));
        assert!(pruned.contains("gemini text"));
        assert!(pruned.contains("rs_kept"));
        assert!(pruned.contains("final answer"));
        assert!(!pruned.contains("call_1"));
        assert!(!pruned.contains("call_2"));
        assert!(!pruned.contains("rs_orphan"));
        assert!(!pruned.contains("toolu_1"));
        assert!(!pruned.contains("chat tool result"));
        assert!(!pruned.contains("anthropic result"));
        assert!(!pruned.contains("functionCall"));
        assert!(!pruned.contains("functionResponse"));
    }

    // ── CompletionRunner basic tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_basic_completion_no_tool_calls() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "hello world".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        assert!(!runner.is_done());
        assert_eq!(runner.turns(), 0);

        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(runner.turns(), 1);
        assert_eq!(output.content, "hello world");
        assert!(output.failed_reason.is_none());
        assert_eq!(output.model, Some("echo".to_string()));
        assert_eq!(output.usage.input_tokens, 5);
        assert_eq!(output.usage.output_tokens, 10);
        assert_eq!(output.usage.requests, 1);

        // Subsequent call returns None.
        let output = runner.next().await.unwrap();
        assert!(output.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_executes_document_only_request() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            ..Default::default()
        }
        .context("doc_1".to_string(), "context without prompt".to_string());

        let mut runner = ctx.completion_iter(req, Vec::new());

        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(runner.turns(), 1);
        assert_eq!(output.model, Some("echo".to_string()));
        assert_eq!(output.usage.requests, 1);
    }

    #[test]
    fn runner_prunes_only_unanswered_raw_tool_call_items() {
        let sentinel = json!({"role": "user", "content": "prior"});
        let mut raw_history = vec![
            sentinel.clone(),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "anthropic text"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
                ]
            }),
            json!({
                "role": "model",
                "parts": [
                    {"text": "gemini text"},
                    {"functionCall": {"name": "lookup", "args": {}}}
                ]
            }),
            json!({"type": "function_call", "call_id": "call_1"}),
            json!({"type": "custom_tool_call", "call_id": "call_2"}),
        ];

        CompletionRunner::prune_unanswered_tool_calls_from_raw_history(&mut raw_history, 1);

        assert_eq!(raw_history.len(), 3);
        assert_eq!(raw_history[0], sentinel);
        assert_eq!(raw_history[1]["content"].as_array().unwrap().len(), 1);
        assert_eq!(raw_history[1]["content"][0]["text"], "anthropic text");
        assert_eq!(raw_history[2]["parts"].as_array().unwrap().len(), 1);
        assert_eq!(raw_history[2]["parts"][0]["text"], "gemini text");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_is_done_returns_none_immediately() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "test".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        // Complete the runner.
        runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());

        // Further calls return None.
        assert!(runner.next().await.unwrap().is_none());
        assert!(runner.next().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_unbound_returns_none_only_after_becoming_idle() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.set_unbound(true);

        let step1 = runner.next().await.unwrap().unwrap();
        assert_eq!(step1.content, "initial");
        assert!(!runner.is_done());
        assert_eq!(runner.turns(), 1);

        let idle = runner.next().await.unwrap();
        assert!(idle.is_none());
        assert!(!runner.is_done());
        assert_eq!(runner.turns(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_unbound_can_resume_after_idle_with_follow_up() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.set_unbound(true);

        let step1 = runner.next().await.unwrap().unwrap();
        assert_eq!(step1.content, "initial");
        assert!(!runner.is_done());

        assert!(runner.next().await.unwrap().is_none());

        runner.follow_up("resume".to_string());

        let step2 = runner.next().await.unwrap().unwrap();
        assert_eq!(step2.content, "resume");
        assert_eq!(step2.usage.input_tokens, 10);
        assert_eq!(step2.usage.output_tokens, 20);
        assert!(!runner.is_done());

        assert!(runner.next().await.unwrap().is_none());
        assert!(!runner.is_done());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_unbound_still_finishes_on_failed_reason() {
        let model = Model::with_completer(Arc::new(AlwaysFailCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.set_unbound(true);

        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(output.failed_reason.as_deref(), Some("primary failed"));
        assert!(runner.next().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_finishes_on_primary_failure() {
        let primary = Model::with_completer(Arc::new(AlwaysFailCompleter));

        let ctx = EngineBuilder::new().with_model(primary).mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert!(output.failed_reason.is_some());
        assert_eq!(output.failed_reason.unwrap(), "primary failed");
    }

    // ── Model error propagation ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_model_error_propagates() {
        let model = Model::with_completer(Arc::new(ErrorCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let result = runner.next().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model error"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_discards_in_flight_tool_result_request_after_model_error() {
        let model = Model::with_completer(Arc::new(ToolResultErrorCompleter));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "call tool".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );

        let step = runner.next().await.unwrap().unwrap();
        assert_eq!(step.tool_calls[0].name, "echo_tool");
        assert!(runner.req.raw_history[0].get("tool_calls").is_some());

        let err = runner.next().await.unwrap_err();
        assert!(err.to_string().contains("model error"));
        assert_eq!(runner.req.role.as_deref(), Some("tool"));
        assert!(!runner.req.content.is_empty());

        runner.discard_in_flight_request();

        assert!(runner.req.content.is_empty());
        assert!(runner.req.prompt.is_empty());
        assert!(runner.req.role.is_none());
        assert!(runner.req.raw_history.is_empty());
        assert!(runner.pending_tool_calls.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_stop_current_task_closes_pending_tool_call_history() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(ToolCallHistoryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "start tool".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );

        let step = runner.next().await.unwrap().unwrap();
        assert_eq!(
            step.tool_calls[0].call_id.as_deref(),
            Some("call_stop_test")
        );
        assert!(!runner.no_pending_tool_calls());
        assert!(!CompletionRunner::unanswered_tool_calls(runner.chat_history()).is_empty());

        let stopped = runner.stop_current_task(AgentOutput {
            content: "operator stopped the task".to_string(),
            ..Default::default()
        });

        assert!(runner.no_pending_tool_calls());
        assert!(CompletionRunner::unanswered_tool_calls(runner.chat_history()).is_empty());
        assert!(
            stopped.chat_history.iter().any(|message| {
                message.content.iter().any(|part| {
                    matches!(
                        part,
                        ContentPart::ToolOutput {
                            call_id,
                            is_error: Some(true),
                            ..
                        } if call_id.as_deref() == Some("call_stop_test")
                    )
                })
            }),
            "stopped output should answer the pending tool call"
        );
        assert!(
            stopped.chat_history.iter().any(|message| {
                message.content.iter().any(|part| {
                    matches!(
                        part,
                        ContentPart::Text { text } if text == "planning before tool"
                    )
                })
            }),
            "stopped output should preserve assistant text that shared the tool-call message"
        );

        runner.follow_up("continue after stop".to_string());
        let continued = runner.next().await.unwrap().unwrap();
        assert_eq!(continued.content, "continued");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].raw_history.len(), 1);
        assert_eq!(
            requests[1].raw_history[0]["content"],
            "planning before tool"
        );
        assert!(requests[1].raw_history[0].get("tool_calls").is_none());
        assert!(
            !serde_json::to_string(&requests[1].raw_history)
                .unwrap()
                .contains("call_stop_test"),
            "follow-up request should not resend the interrupted raw tool call"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_discard_in_flight_request_closes_pending_tool_call_history() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(ToolCallHistoryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "start tool".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );

        let step = runner.next().await.unwrap().unwrap();
        assert_eq!(
            step.tool_calls[0].call_id.as_deref(),
            Some("call_stop_test")
        );

        runner.discard_in_flight_request();

        assert!(runner.no_pending_tool_calls());
        assert!(CompletionRunner::unanswered_tool_calls(runner.chat_history()).is_empty());
        assert!(runner.chat_history().iter().any(|message| {
            message.content.iter().any(|part| {
                matches!(
                    part,
                    ContentPart::ToolOutput {
                        call_id,
                        is_error: Some(true),
                        output,
                        ..
                    } if call_id.as_deref() == Some("call_stop_test")
                        && output.get("error").and_then(Json::as_str)
                            == Some("tool call discarded")
                )
            })
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_queued_steering_closes_skipped_tool_call_history() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(ToolCallHistoryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "start tool".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );
        runner.steer("redirect".to_string());

        let step1 = runner.next().await.unwrap().unwrap();
        assert_eq!(step1.tool_calls.len(), 1);
        assert!(CompletionRunner::unanswered_tool_calls(&step1.chat_history).is_empty());
        assert!(step1.chat_history.iter().any(|message| {
            message.content.iter().any(|part| {
                matches!(
                    part,
                    ContentPart::ToolOutput {
                        call_id,
                        is_error: Some(true),
                        output,
                        ..
                    } if call_id.as_deref() == Some("call_stop_test")
                        && output.get("error").and_then(Json::as_str)
                            == Some("tool call interrupted by steering")
                )
            })
        }));

        let step2 = runner.next().await.unwrap().unwrap();
        assert_eq!(step2.content, "continued");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(
            !serde_json::to_string(&requests[1].raw_history)
                .unwrap()
                .contains("call_stop_test"),
            "steered request should not resend the interrupted raw tool call"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_late_steering_closes_pending_tool_call_history() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(ToolCallHistoryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "start tool".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );

        let step1 = runner.next().await.unwrap().unwrap();
        assert_eq!(step1.tool_calls.len(), 1);
        assert!(!CompletionRunner::unanswered_tool_calls(runner.chat_history()).is_empty());

        runner.steer("redirect".to_string());
        let step2 = runner.next().await.unwrap().unwrap();
        assert_eq!(step2.content, "continued");
        assert!(CompletionRunner::unanswered_tool_calls(&step2.chat_history).is_empty());
        assert!(step2.chat_history.iter().any(|message| {
            message.content.iter().any(|part| {
                matches!(
                    part,
                    ContentPart::ToolOutput {
                        call_id,
                        is_error: Some(true),
                        output,
                        ..
                    } if call_id.as_deref() == Some("call_stop_test")
                        && output.get("error").and_then(Json::as_str)
                            == Some("tool call interrupted by steering")
                )
            })
        }));

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(
            !serde_json::to_string(&requests[1].raw_history)
                .unwrap()
                .contains("call_stop_test"),
            "late-steered request should not resend the interrupted raw tool call"
        );
    }

    // ── Tool call tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_executes_tool_calls() {
        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "hello"}),
                call_id: Some("call_1".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "call tool".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        // Step 1: model returns tool calls, runner executes them and returns intermediate.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.usage.input_tokens, 10);
        assert_eq!(step1.usage.output_tokens, 20);

        // Step 2: model processes tool results and returns final.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "tool_result_processed");
        // Usage accumulated from both steps.
        assert_eq!(step2.usage.input_tokens, 13); // 10 + 3
        assert_eq!(step2.usage.output_tokens, 26); // 20 + 6
        // tool_calls accumulated.
        assert_eq!(step2.tool_calls.len(), 1);
        assert_eq!(step2.tool_calls[0].name, "echo_tool");
        assert!(step2.tool_calls[0].result.is_some());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_appends_follow_up_after_pending_tool_calls_finish() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(ToolChainUntilFollowUpCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "start tool chain".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.tool_calls.len(), 1);

        runner.follow_up("follow up while tool chain is pending".to_string());

        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "follow_up_seen_with_tool_result");
        assert_eq!(runner.turns(), 2);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].chat_history.len(), 1);
        assert_eq!(requests[1].chat_history[0].role, "tool");
        assert_eq!(requests[1].chat_history[0].content.len(), 1);
        assert!(matches!(
            &requests[1].chat_history[0].content[0],
            ContentPart::ToolOutput { name, .. } if name == "echo_tool"
        ));
        assert_eq!(requests[1].role.as_deref(), Some("user"));
        assert_eq!(requests[1].content.len(), 1);
        assert!(matches!(
            &requests[1].content[0],
            ContentPart::Text { text } if text == "follow up while tool chain is pending"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_tool_call_failure_dont_produces_failed_reason() {
        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "fail_tool".to_string(),
                args: json!({}),
                call_id: Some("call_fail".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(FailTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "call fail".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let output = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert!(output.failed_reason.is_none());
    }

    // ── Agent call tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_executes_agent_calls() {
        let completer = AgentCallCompleter {
            agent_name: "echo_agent".to_string(),
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_agent(Arc::new(EchoAgent), None)
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "call agent".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        // Step 1: agent call returns intermediate result.
        let _step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        // Step 2: final result.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "agent_result_processed");
        assert_eq!(step2.tool_calls.len(), 1);
        assert_eq!(step2.tool_calls[0].name, "echo_agent");
        // Agent call result should be stored.
        let result = step2.tool_calls[0].result.as_ref().unwrap();
        assert!(
            result
                .output
                .as_str()
                .unwrap()
                .contains("agent_echoed:subagent task")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_agent_call_with_arbitrary_args() {
        // Agent call args no longer require a "prompt" field.
        // When missing, the whole args JSON should be used as the prompt.
        #[derive(Clone, Debug)]
        struct BadArgsCompleter;

        impl CompletionFeaturesDyn for BadArgsCompleter {
            fn model_name(&self) -> String {
                "bad_args".to_string()
            }

            fn completion(
                &self,
                req: CompletionRequest,
            ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
                let role = req.role.as_deref().unwrap_or("");
                if role == "tool" {
                    return Box::pin(futures::future::ready(Ok(AgentOutput {
                        content: "agent_result_processed".to_string(),
                        ..Default::default()
                    })));
                }

                Box::pin(futures::future::ready(Ok(AgentOutput {
                    tool_calls: vec![ToolCall {
                        name: "echo_agent".to_string(),
                        args: json!({"invalid_field": 42}),
                        call_id: Some("bad_call".into()),
                        result: None,
                        remote_id: None,
                    }],
                    ..Default::default()
                })))
            }
        }

        let model = Model::with_completer(Arc::new(BadArgsCompleter));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_agent(Arc::new(EchoAgent), None)
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "bad args".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let _step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert!(output.failed_reason.is_none());
        assert_eq!(output.content, "agent_result_processed");
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "echo_agent");

        // The whole args object should be forwarded as prompt JSON.
        let result = output.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(
            result.output.as_str().unwrap(),
            "agent_echoed:{\"invalid_field\":42}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_preserves_structured_subagent_session_args() {
        let model = Model::with_completer(Arc::new(ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: format!("{SUB_AGENT_PREFIX}echo_helper"),
                args: json!({
                    "prompt": "session task",
                    "session": "AsyncJob",
                    "model": "",
                    "effort": null,
                }),
                call_id: Some("subagent_session_call".into()),
                result: None,
                remote_id: None,
            }],
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let manager: Arc<SubAgentManager> = ctx.subagents.get().unwrap();
        manager
            .upsert_temporary(SubAgent {
                name: "echo_helper".to_string(),
                description: "Echoes input.".to_string(),
                instructions: "Echo the prompt.".to_string(),
                ..Default::default()
            })
            .unwrap();

        let req = CompletionRequest {
            prompt: "call subagent".to_string(),
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());

        let _step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        let output = runner.next().await.unwrap().unwrap();
        assert_eq!(output.tool_calls.len(), 1);
        let result = output.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(result.output["session"], json!("asyncjob"));
        assert!(
            result.output["content"]
                .as_str()
                .unwrap()
                .contains("session mode")
        );
    }

    // ── Steering message tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_steering_message_before_first_step() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.steer("redirect to this".to_string());

        // Step 1: model completes "initial", but steering intercepts.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial"); // Original completion before steering.

        // Step 2: processes the steering prompt.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "redirect to this");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_steering_skips_pending_tool_calls() {
        // If model returns tool_calls and steering is set, tool_calls should be skipped.
        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "test"}),
                call_id: Some("skipped_call".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "call tool".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.steer("abort and redirect".to_string());

        // Step 1: steering intercepts — tool calls are NOT executed.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        // The tool calls in step1 are the raw model output (not executed).
        assert!(!step1.tool_calls.is_empty());

        // Step 2: ToolCallCompleter sees role != "tool", returns tool_calls again,
        // but no steering now so tools execute.
        let _step2 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        // Step 3: model processes tool results and returns final.
        let step3 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step3.content, "tool_result_processed");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_steering_preserves_prior_raw_history_when_skipping_current_tool_call() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RawHistoryToolCallCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let sentinel = json!({"role": "user", "content": "original raw history"});

        let req = CompletionRequest {
            prompt: "call tool".to_string(),
            raw_history: vec![sentinel.clone()],
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.steer("redirect".to_string());

        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.tool_calls.len(), 1);

        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "steered");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].raw_history.len(), 2);
        assert_eq!(requests[1].raw_history[0], sentinel);
        assert_eq!(requests[1].raw_history[1]["content"], "planning tool call");
        assert_eq!(
            requests[1].raw_history[1]["reasoning"],
            "keep this reasoning"
        );
        assert!(requests[1].raw_history[1].get("tool_calls").is_none());
        assert!(
            !requests[1]
                .raw_history
                .iter()
                .any(|item| item["type"] == "function_call")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_late_steering_preserves_prior_raw_history_when_pending_tool_call_exists() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RawHistoryToolCallCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();
        let sentinel = json!({"role": "user", "content": "original raw history"});

        let req = CompletionRequest {
            prompt: "call tool".to_string(),
            raw_history: vec![sentinel.clone()],
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.tool_calls.len(), 1);

        runner.steer("redirect".to_string());

        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "steered");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].raw_history.len(), 2);
        assert_eq!(requests[1].raw_history[0], sentinel);
        assert_eq!(requests[1].raw_history[1]["content"], "planning tool call");
        assert_eq!(
            requests[1].raw_history[1]["reasoning"],
            "keep this reasoning"
        );
        assert!(requests[1].raw_history[1].get("tool_calls").is_none());
        assert!(
            !requests[1]
                .raw_history
                .iter()
                .any(|item| item["type"] == "function_call")
        );
    }

    // ── Follow-up message tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_follow_up_message_after_completion() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.follow_up("follow up question".to_string());

        // Step 1: initial completion, follow_up makes it continue.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes follow-up prompt.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "follow up question");
        // Usage accumulated.
        assert_eq!(step2.usage.input_tokens, 10); // 5 + 5
        assert_eq!(step2.usage.output_tokens, 20); // 10 + 10
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_steering_takes_priority_over_follow_up() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.steer("steering".to_string());
        runner.follow_up("follow_up".to_string());

        // Step 1: drain_steering_message() drains both follow_up and steering together
        // (follow_up placed before steering per drain_steering_message logic).
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes the combined prompt "follow_up\n\nsteering".
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "follow_up\n\nsteering");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_multiple_steering_messages_combined() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.steer("first steer".to_string());
        runner.steer("second steer".to_string());

        // Step 1: initial completion, steering intercepts.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes both steering messages combined.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "first steer\n\nsecond steer");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_multiple_follow_up_messages_combined() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.follow_up("first follow".to_string());
        runner.follow_up("second follow".to_string());

        // Step 1: initial completion, first follow_up is delivered.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes all queued follow-up messages as one user turn.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "first follow\n\nsecond follow");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_multiple_steering_and_follow_up_combined() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.steer("steer 1".to_string());
        runner.follow_up("follow 1".to_string());
        runner.steer("steer 2".to_string());
        runner.follow_up("follow 2".to_string());

        // Step 1: drain_steering_message drains all follow_up first (in order),
        // then all steering (in order): follow_1, follow_2, steer_1, steer_2.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes combined prompt.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "follow 1\n\nfollow 2\n\nsteer 1\n\nsteer 2");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_steering_empty_drains_follow_up_only() {
        // When steering_message is empty but follow_up has messages,
        // drain_steering_message returns None and queued follow-up messages are drained.
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.follow_up("follow only".to_string());

        // Step 1: drain_steering_message returns None (steering empty),
        // so queued follow-up messages are drained after the initial response.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes follow_up.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "follow only");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_finalize_idle_unbound_returns_latest_output() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new()).unbound();

        let step1 = runner.next().await.unwrap().unwrap();
        assert_eq!(step1.content, "initial");
        assert!(!runner.is_done());

        assert!(runner.next().await.unwrap().is_none());

        let output = runner.finalize(None).await.unwrap();
        assert!(runner.is_done());
        assert_eq!(output.content, "initial");
        assert_eq!(output.usage.input_tokens, 5);
        assert_eq!(output.usage.output_tokens, 10);
        assert!(runner.next().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_finalize_processes_queued_and_new_prompt() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new()).unbound();

        let step1 = runner.next().await.unwrap().unwrap();
        assert_eq!(step1.content, "initial");
        assert!(runner.next().await.unwrap().is_none());

        runner.follow_up("queued follow-up".to_string());

        let output = runner
            .finalize(Some("final prompt".to_string()))
            .await
            .unwrap();
        assert!(runner.is_done());
        assert_eq!(output.content, "queued follow-up\n\nfinal prompt");
        assert_eq!(output.usage.input_tokens, 10);
        assert_eq!(output.usage.output_tokens, 20);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_handoff_preserves_queued_input_for_new_runner() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingCompleter {
            name: "recording".to_string(),
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "initial".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new()).unbound();
        let step = runner.next().await.unwrap().unwrap();
        assert_eq!(step.content, "initial");

        runner.follow_up("queued follow-up".to_string());
        runner.steer("queued steering".to_string());
        let (mut runner, output) = runner.handoff(None).await.unwrap();
        assert_eq!(output.content.trim(), super::COMPACTION_PROMPT.trim());

        let continued = runner.next().await.unwrap().unwrap();
        assert_eq!(continued.content, "queued follow-up\n\nqueued steering");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        let compaction_text = requests[1]
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } | ContentPart::Reasoning { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        assert_eq!(compaction_text.trim(), super::COMPACTION_PROMPT.trim());
        assert!(!compaction_text.contains("queued follow-up"));
        assert!(!compaction_text.contains("queued steering"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_handoff_steering_interrupts_pending_tool_calls() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(ToolCallHistoryCompleter {
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let mut runner = ctx
            .completion_iter(
                CompletionRequest {
                    prompt: "start tool".to_string(),
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound();

        let step = runner.next().await.unwrap().unwrap();
        assert_eq!(
            step.tool_calls[0].call_id.as_deref(),
            Some("call_stop_test")
        );
        assert!(!runner.no_pending_tool_calls());

        runner.steer("redirect before tool".to_string());
        let (mut runner, output) = runner.handoff(None).await.unwrap();
        assert_eq!(output.content, "continued");
        assert!(output.tool_calls.is_empty());
        assert!(!output.tools_usage.contains_key("echo_tool"));
        assert!(CompletionRunner::unanswered_tool_calls(&output.chat_history).is_empty());
        assert!(output.chat_history.iter().any(|message| {
            message.content.iter().any(|part| {
                matches!(
                    part,
                    ContentPart::ToolOutput {
                        call_id,
                        is_error: Some(true),
                        output,
                        ..
                    } if call_id.as_deref() == Some("call_stop_test")
                        && output.get("error").and_then(Json::as_str)
                            == Some("tool call interrupted by steering")
                )
            })
        }));

        let continued = runner.next().await.unwrap().unwrap();
        assert_eq!(continued.content, "continued");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        let compaction_raw = serde_json::to_string(&requests[1].raw_history).unwrap();
        assert!(!compaction_raw.contains("call_stop_test"));
        assert_eq!(
            requests[1].raw_history[0]["content"],
            "planning before tool"
        );
        assert!(requests[1].content.iter().any(|part| matches!(
            part,
            ContentPart::Text { text } if text.trim() == super::COMPACTION_PROMPT.trim()
        )));
        assert!(requests[2].content.iter().any(|part| matches!(
            part,
            ContentPart::Text { text } if text == "redirect before tool"
        )));
    }

    // ── Cancellation tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_cancellation_returns_cancelled_output() {
        // Use SlowCompleter so tokio::select picks the cancellation branch.
        let model = Model::with_completer(Arc::new(SlowCompleter));
        let cancel_token = CancellationToken::new();

        let ctx = EngineBuilder::new()
            .with_model(model)
            .with_cancellation_token(cancel_token.clone())
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        // Cancel before first step.
        cancel_token.cancel();

        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert!(output.failed_reason.is_some());
        assert_eq!(output.failed_reason.unwrap(), "operation cancelled");

        // After cancellation, returns None.
        assert!(runner.next().await.unwrap().is_none());
    }

    // ── Usage accumulation tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_usage_accumulates_across_steps() {
        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "test"}),
                call_id: Some("call_1".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "test".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        // Step 1: intermediate (usage = step1_model + tool)
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert!(step1.usage.requests >= 1);

        // Step 2: final (usage = total accumulated)
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        // Should have accumulated usage from both model calls + tool call.
        assert!(step2.usage.requests >= 2);
        assert!(step2.usage.input_tokens > 0);
        assert!(step2.usage.output_tokens > 0);
    }

    // ── CompletionStream tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn stream_basic_completion() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "stream test".to_string(),
            ..Default::default()
        };

        let mut stream = ctx.completion_stream(req, Vec::new());

        let item = stream.next().await;
        assert!(item.is_some());
        let output = item.unwrap().unwrap();
        assert_eq!(output.content, "stream test");

        // Stream should end.
        let item = stream.next().await;
        assert!(item.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_keeps_pending_future_across_polls() {
        let model = Model::with_completer(Arc::new(DelayedEchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "delayed stream".to_string(),
            ..Default::default()
        };

        let mut stream = ctx.completion_stream(req, Vec::new());
        let output = tokio::time::timeout(std::time::Duration::from_millis(200), stream.next())
            .await
            .expect("stream should not restart a pending completion forever")
            .unwrap()
            .unwrap();

        assert_eq!(output.content, "delayed stream");
        assert_eq!(output.model, Some("delayed_echo".to_string()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_multi_step_with_tool_calls() {
        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({"input": "via_stream"}),
                call_id: Some("stream_call".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "stream with tools".to_string(),
            ..Default::default()
        };

        let stream = ctx.completion_stream(req, Vec::new());
        let results: Vec<_> = stream.collect().await;

        // Should have 2 items: intermediate tool call result + final.
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());

        let final_output = results.last().unwrap().as_ref().unwrap();
        assert_eq!(final_output.content, "tool_result_processed");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_error_propagation() {
        let model = Model::with_completer(Arc::new(ErrorCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "error stream".to_string(),
            ..Default::default()
        };

        let mut stream = ctx.completion_stream(req, Vec::new());

        let item = stream.next().await;
        assert!(item.is_some());
        assert!(item.unwrap().is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_buffers_steering_while_step_in_flight() {
        /// Completer whose first call blocks on a gate, then emits a tool call.
        /// A steered user turn is echoed back with a `steered:` marker.
        #[derive(Clone)]
        struct GatedToolCallCompleter {
            gate: Arc<tokio::sync::Notify>,
        }

        impl CompletionFeaturesDyn for GatedToolCallCompleter {
            fn model_name(&self) -> String {
                "gated_tool_call".to_string()
            }

            fn completion(
                &self,
                req: CompletionRequest,
            ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
                let gate = self.gate.clone();
                Box::pin(async move {
                    if req.role.as_deref() == Some("user") {
                        let text = req
                            .content
                            .iter()
                            .filter_map(|part| match part {
                                ContentPart::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        return Ok(AgentOutput {
                            content: format!("steered:{text}"),
                            usage: Usage {
                                requests: 1,
                                ..Default::default()
                            },
                            ..Default::default()
                        });
                    }
                    if req.role.as_deref() == Some("tool") {
                        return Ok(AgentOutput {
                            content: "tool_result_processed".to_string(),
                            ..Default::default()
                        });
                    }

                    gate.notified().await;
                    Ok(AgentOutput {
                        tool_calls: vec![ToolCall {
                            name: "echo_tool".to_string(),
                            args: json!({"input": "x"}),
                            call_id: Some("gated_call".into()),
                            result: None,
                            remote_id: None,
                        }],
                        usage: Usage {
                            requests: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                })
            }
        }

        let gate = Arc::new(tokio::sync::Notify::new());
        let model = Model::with_completer(Arc::new(GatedToolCallCompleter { gate: gate.clone() }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let mut stream = ctx.completion_stream(
            CompletionRequest {
                prompt: "start".to_string(),
                ..Default::default()
            },
            Vec::new(),
        );

        // Start the first step; the gated completer keeps it in flight.
        let waker = futures::task::noop_waker();
        let mut poll_cx = std::task::Context::from_waker(&waker);
        assert!(stream.poll_next_unpin(&mut poll_cx).is_pending());

        // Steering while the step is in flight must be buffered, not dropped.
        stream.steer("redirect".to_string());
        gate.notify_one();

        let step1 = stream.next().await.unwrap().unwrap();
        assert_eq!(step1.tool_calls.len(), 1);

        // The buffered steering interrupts the pending tool call and becomes
        // the next user turn.
        let step2 = stream.next().await.unwrap().unwrap();
        assert_eq!(step2.content, "steered:redirect");
        assert!(stream.next().await.is_none());
    }

    // ── Step counter tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_step_counter_increments() {
        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                args: json!({}),
                call_id: Some("step_call".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "steps".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        assert_eq!(runner.turns(), 0);

        runner.next().await.unwrap(); // turn 1
        assert_eq!(runner.turns(), 1);

        runner.next().await.unwrap(); // turn 2 (final)
        assert_eq!(runner.turns(), 2);
    }

    // ── Chat history accumulation tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_chat_history_accumulated_in_final() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.follow_up("follow up".to_string());

        let step1 = runner.next().await.unwrap().unwrap();
        // Intermediate output includes current chat history.
        let step1_history_len = step1.chat_history.len();

        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        // Final output should have more chat history (accumulated from both steps).
        assert!(step2.chat_history.len() >= step1_history_len);
    }

    // ── Artifacts accumulation tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_artifacts_accumulated_from_tool_calls() {
        /// A tool that returns artifacts.
        struct ArtifactTool;

        #[derive(Debug, Deserialize)]
        struct ArtifactArgs {}

        impl Tool<BaseCtx> for ArtifactTool {
            type Args = ArtifactArgs;
            type Output = String;

            fn name(&self) -> String {
                "artifact_tool".to_string()
            }

            fn description(&self) -> String {
                "Returns artifacts".to_string()
            }

            fn definition(&self) -> FunctionDefinition {
                FunctionDefinition {
                    name: "artifact_tool".to_string(),
                    description: "Returns artifacts".to_string(),
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
            ) -> Result<ToolOutput<String>, BoxError> {
                Ok(ToolOutput {
                    output: "done".to_string(),
                    artifacts: vec![Resource {
                        tags: vec!["test_artifact".to_string()],
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            }
        }

        let completer = ToolCallCompleter {
            tool_calls: vec![ToolCall {
                name: "artifact_tool".to_string(),
                args: json!({}),
                call_id: Some("art_call".into()),
                result: None,
                remote_id: None,
            }],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(ArtifactTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "artifacts".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.next().await.unwrap(); // step 1: tool execution
        let final_out = runner.next().await.unwrap().unwrap(); // step 2: final

        assert!(runner.is_done());
        assert_eq!(final_out.artifacts.len(), 1);
        assert_eq!(final_out.artifacts[0].tags, vec!["test_artifact"]);
    }

    // ── Model name in output ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_sets_model_name_in_output() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let req = CompletionRequest {
            prompt: "check model".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let output = runner.next().await.unwrap().unwrap();
        assert_eq!(output.model, Some("echo".to_string()));
    }

    // ── Multiple tool calls in parallel ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_multiple_tool_calls_in_parallel() {
        let completer = ToolCallCompleter {
            tool_calls: vec![
                ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "first"}),
                    call_id: Some("call_a".into()),
                    result: None,
                    remote_id: None,
                },
                ToolCall {
                    name: "echo_tool".to_string(),
                    args: json!({"input": "second"}),
                    call_id: Some("call_b".into()),
                    result: None,
                    remote_id: None,
                },
            ],
        };

        let model = Model::with_completer(Arc::new(completer));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "multi tools".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        runner.next().await.unwrap(); // step 1: both tools execute in parallel
        let final_out = runner.next().await.unwrap().unwrap();

        assert!(runner.is_done());
        assert_eq!(final_out.tool_calls.len(), 2);
        // Both tool results present.
        for tc in &final_out.tool_calls {
            assert!(tc.result.is_some());
        }
    }
}
