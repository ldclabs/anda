//! Agent Context Implementation
//!
//! This module provides the core implementation of the Agent context ([`AgentCtx`]) which serves as
//! the primary execution environment for agents in the Anda system. The context provides:
//!
//! - Access to AI models for completions;
//! - Tool execution capabilities;
//! - Agent-to-agent communication;
//! - Cryptographic operations;
//! - Storage and caching facilities;
//! - HTTP communication features.
//!
//! The [`AgentCtx`] implements multiple traits that provide different sets of functionality:
//! - [`AgentContext`]: Core agent operations and tool/agent management;
//! - [`CompletionFeatures`]: AI model completion capabilities;
//! - [`StateFeatures`]: Context state management;
//! - [`KeysFeatures`]: Cryptographic key operations;
//! - [`StoreFeatures`]: Persistent storage operations;
//! - [`CacheFeatures`]: Caching mechanisms;
//! - [`HttpFeatures`]: HTTPs communication features.
//!
//! The context is designed to be hierarchical, allowing creation of child contexts for specific
//! agents or tools while maintaining access to the core functionality.

use anda_core::{
    Agent, AgentContext, AgentInput, AgentOutput, AgentSet, BaseContext, BoxError, CacheExpiry,
    CacheFeatures, CacheStoreFeatures, CancellationToken, CompletionFeatures, CompletionRequest,
    FunctionDefinition, HttpFeatures, Json, KeysFeatures, ObjectMeta, Path, PutMode, PutResult,
    RequestMeta, Resource, StateFeatures, StoreFeatures, ToolGroup, ToolInput, ToolOutput,
    ToolProviderSet, ToolSet,
};
use bytes::Bytes;
use candid::Principal;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    sync::Arc,
    time::Duration,
};

use super::{
    base::BaseCtx,
    engine::RemoteEngines,
    runner::{CompletionRunner, CompletionStream},
};
use crate::{
    extension::todo::TodoSession,
    model::{Model, Models},
    subagent::{SubAgentSet, SubAgentSetManager},
};

/// Reserved dynamic-configuration key for remote engine registrations.
pub static DYNAMIC_REMOTE_ENGINES: &str = "_engines";
/// Prefix used for routed remote-agent calls in model-facing tool names.
pub static REMOTE_AGENT_PREFIX: &str = "RA_";
/// Prefix used for routed remote-tool calls in model-facing tool names.
pub static REMOTE_TOOL_PREFIX: &str = "RT_";
/// Prefix used for routed subagent calls in model-facing tool names.
pub static SUB_AGENT_PREFIX: &str = "SA_";

/// Strips a routing prefix such as `RT_`, `RA_`, or `SA_` without case
/// sensitivity, since models occasionally echo prefixed callable names in a
/// different case.
pub(crate) fn strip_prefix_ignore_ascii_case<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let (head, tail) = name.split_at_checked(prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then_some(tail)
}

/// Strips whichever routing prefix `name` carries, if any.
///
/// Routing prefixes are added when definitions are produced and stripped when calls are
/// dispatched, so any check that compares a model-emitted name against a registered name must
/// normalize through here first.
pub(crate) fn strip_routing_prefix(name: &str) -> Option<&str> {
    [SUB_AGENT_PREFIX, REMOTE_TOOL_PREFIX, REMOTE_AGENT_PREFIX]
        .into_iter()
        .find_map(|prefix| strip_prefix_ignore_ascii_case(name, prefix))
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
        // Seed a session-scoped todo store on the long-lived runner context so
        // per-tool-call child contexts (which snapshot-copy parent state) share
        // one list across the whole conversation. Seed only when absent so a
        // nested runner inherits the parent's list instead of resetting it.
        if self.base.get_state::<TodoSession>().is_none() {
            self.base.set_state(TodoSession::new());
        }
        CompletionRunner::new(self, req, model, resources)
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

        // Deduplicate across every source, not just within each one. Tools, agents,
        // subagents, and remote engines are independent registries, so nothing prevents one
        // name from being registered in two of them (e.g. a tool and an agent both named
        // `search`, or an MCP tool colliding with an agent). Emitting the name twice makes
        // providers reject the entire request with a duplicate-function-name error, taking
        // down every completion in the engine.
        let mut definitions = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut extend_unique = |source: Vec<FunctionDefinition>, definitions: &mut Vec<_>| {
            for definition in source {
                if seen.insert(definition.name.to_ascii_lowercase()) {
                    definitions.push(definition);
                }
            }
        };

        extend_unique(self.tool_definitions(names), &mut definitions);
        extend_unique(self.agent_definitions(names), &mut definitions);
        if let Ok(remote) = self.remote_tool_definitions(None, names).await {
            extend_unique(remote, &mut definitions);
        }
        if let Ok(remote) = self.remote_agent_definitions(None, names).await {
            extend_unique(remote, &mut definitions);
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
        AgentContext as _, AgentInput, BaseContext as _, BoxError, CacheFeatures as _,
        CacheStoreFeatures as _, CompletionFeatures as _, HttpFeatures as _, Json,
        KeysFeatures as _, Path, PutMode, StateFeatures as _, StoreFeatures as _, ToolInput,
    };
    use bytes::Bytes;
    use candid::Principal;
    use cbor2::from_slice;
    use ic_cose_types::to_cbor_bytes;
    use serde_json::json;
    use std::sync::Arc;

    use super::{
        DYNAMIC_REMOTE_ENGINES, REMOTE_AGENT_PREFIX, REMOTE_TOOL_PREFIX, SUB_AGENT_PREFIX,
    };
    use crate::context::test_fixtures::*;
    use crate::{engine::EngineBuilder, model::Model};

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

        // Tools sharing one `DynToolHook` slot (same argument/output types) are
        // distinguishable only through the public context path.
        assert_eq!(
            child.child_base("execute_kip").unwrap().path().as_ref(),
            "t_execute_kip"
        );
        assert_eq!(
            child.child_base("memory_readonly").unwrap().path().as_ref(),
            "t_memory_readonly"
        );
    }

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

        // The root ctx now has a registered cache namespace (the engine
        // builders register `Path::default()`), so root-level cache access
        // succeeds instead of failing with "cache path not found".
        let created: String = ctx
            .cache_get_with("root_key", async {
                Ok::<_, BoxError>(("created".to_string(), None))
            })
            .await
            .unwrap();
        assert_eq!(created, "created");

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
}
