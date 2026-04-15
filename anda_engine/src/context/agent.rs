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
    KeysFeatures, Message, ObjectMeta, Path, PutMode, PutResult, RequestMeta, Resource,
    StateFeatures, StoreFeatures, ToolCall, ToolInput, ToolOutput, ToolSet, Usage,
};
use bytes::Bytes;
use candid::{CandidType, Principal, utils::ArgumentEncoder};
use futures_util::Stream;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use super::{
    base::BaseCtx,
    engine::RemoteEngines,
    subagent::{SubAgentSet, SubAgentSetManager},
    tool::{ToolsSelectOutput, is_tools_select_name},
};
use crate::model::{Model, Models};

pub static DYNAMIC_REMOTE_ENGINES: &str = "_engines";

/// Context for agent operations, providing access to models, tools, and other agents.
#[derive(Clone)]
pub struct AgentCtx {
    /// Base context providing fundamental operations.
    pub base: BaseCtx,

    /// Label of the agent.
    pub label: String,

    pub(crate) root: BaseCtx,
    pub(crate) model: Model,
    // label -> model
    pub(crate) models: Arc<Models>,

    /// Set of available tools that can be called.
    pub(crate) tools: Arc<ToolSet<BaseCtx>>,
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
        agents: Arc<AgentSet<AgentCtx>>,
        subagents: Arc<SubAgentSetManager>,
    ) -> Self {
        Self {
            base: base.clone(),
            label: String::new(),
            root: base,
            model: models.get_model().unwrap_or_else(Model::not_implemented),
            models,
            tools,
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
                .child(format!("A:{}", agent_name.to_ascii_lowercase()))?,
            label: agent_label.to_string(),
            root: self.root.clone(),
            model: self.model.clone(),
            models: self.models.clone(),
            tools: self.tools.clone(),
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
            .child(format!("T:{}", tool_name.to_ascii_lowercase()))
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
                format!("A:{}", agent_name.to_ascii_lowercase()),
                meta,
            )?,
            label: agent_label.to_string(),
            root: self.root.clone(),
            model: self.model.clone(),
            models: self.models.clone(),
            tools: self.tools.clone(),
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
            format!("T:{}", tool_name.to_ascii_lowercase()),
            meta,
        )
    }

    /// Creates a completion runner for iterative processing of completion requests.
    pub fn completion_iter(
        self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> CompletionRunner {
        CompletionRunner {
            ctx: self,
            req,
            resources,
            chat_history: Vec::new(),
            tool_calls: Vec::new(),
            usage: Usage::default(),
            artifacts: Vec::new(),
            steering_message: None,
            follow_up_message: None,
            done: false,
            step: 0,
            pruned: 0,
        }
    }

    /// Creates a completion stream for processing of completion requests.
    pub fn completion_stream(
        self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> CompletionStream {
        CompletionStream {
            runner: self.completion_iter(req, resources),
        }
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
        self.tools.definitions(names)
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
        if let Ok((engines, _)) = self
            .root
            .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
            .await
        {
            let defs2 = engines.tool_definitions(endpoint, names);
            for def in defs2 {
                if !defs.iter().any(|d| d.name == def.name) {
                    defs.push(def);
                }
            }

            Ok(defs)
        } else {
            Ok(defs)
        }
    }

    /// Extracts resources from the provided list based on the tool's supported tags.
    async fn select_tool_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if prefixed_name.starts_with("RT_") {
            let res = self
                .base
                .remote
                .select_tool_resources(prefixed_name, resources);
            if !res.is_empty() {
                return res;
            }

            if let Ok((engines, _)) = self
                .root
                .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
                .await
            {
                return engines.select_tool_resources(prefixed_name, resources);
            }
        }

        self.tools.select_resources(prefixed_name, resources)
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
                .map(|d| d.name_with_prefix("SA_")),
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
            let defs2 = engines.agent_definitions(endpoint, names);
            for def in defs2 {
                if !defs.iter().any(|d| d.name == def.name) {
                    defs.push(def);
                }
            }

            Ok(defs)
        } else {
            Ok(defs)
        }
    }

    /// Extracts resources from the provided list based on the agent's supported tags.
    async fn select_agent_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if prefixed_name.starts_with("RA_") {
            let res = self
                .base
                .remote
                .select_agent_resources(prefixed_name, resources);
            if !res.is_empty() {
                return res;
            }

            if let Ok((engines, _)) = self
                .root
                .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
                .await
            {
                return engines.select_agent_resources(prefixed_name, resources);
            }
        }

        if prefixed_name.starts_with("SA_") {
            return self.subagents.select_resources(prefixed_name, resources);
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
        if input.name.starts_with("RT_") {
            // find registered remote tool and call it
            if let Some((id, endpoint, tool_name)) = self.base.remote.get_tool_endpoint(&input.name)
            {
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
                && let Some((id, endpoint, tool_name)) = engines.get_tool_endpoint(&input.name)
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
        let tool = self
            .tools
            .get(&input.name)
            .ok_or_else(|| format!("tool {} not found", &input.name))?;
        tool.call(ctx, input.args, input.resources)
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
            if input.name.starts_with("RA_") {
                if let Some((id, endpoint, agent_name)) =
                    ctx.base.remote.get_agent_endpoint(&input.name)
                {
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
                    && let Some((id, endpoint, agent_name)) =
                        engines.get_agent_endpoint(&input.name)
                {
                    input.name = agent_name;
                    input.meta = Some(ctx.base.self_meta(id));
                    return ctx
                        .remote_agent_run(&endpoint, input)
                        .await
                        .map(|output| (output, Some(id)));
                }

                return Err(format!("agent {} not found", input.name).into());
            }

            if input.name.starts_with("SA_") {
                let name = input.name[3..].to_ascii_lowercase();
                if let Some(agent) = ctx.subagents.get_lowercase(&name) {
                    let child = ctx.child(&name, &name)?;
                    return agent
                        .run(child, input.prompt, input.resources)
                        .await
                        .map(|output| (output, None));
                } else {
                    return Err(format!("agent {} not found", input.name).into());
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
                Err(format!("agent {} not found", input.name).into())
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
        self.model.model_name()
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
    resources: Vec<Resource>,
    chat_history: Vec<Message>,
    tool_calls: Vec<ToolCall>,
    usage: Usage,
    artifacts: Vec<Resource>,
    steering_message: Option<String>,
    follow_up_message: Option<String>,
    done: bool,
    step: usize,
    pruned: usize,
}

impl CompletionRunner {
    /// Returns whether the completion has finished.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Returns the number of steps executed.
    pub fn steps(&self) -> usize {
        self.step
    }

    /// Queue a steering message to interrupt the agent mid-run.
    /// Delivered after current tool execution, skips remaining tools.
    /// No effect if the completion has finished.
    pub fn steer(&mut self, message: String) {
        self.steering_message = Some(message);
    }

    /// Queue a follow-up message to be processed after the agent finishes.
    /// Delivered only when agent has no more tool calls or steering messages.
    /// No effect if the completion has finished.
    pub fn follow_up(&mut self, message: String) {
        self.follow_up_message = Some(message);
    }

    /// Prune the raw history of the completion request to reduce tokens usage.
    /// Only prunes messages that have tool calls (intermediate steps), and keeps the final response intact.
    /// # Arguments
    /// * `un_pruned_len` - The minimum length of raw history to keep un-pruned. Only when the current un-pruned length reaches or exceeds this threshold, pruning will be performed.
    /// * `prune_len` - The maximum number of messages to prune in this call. This is a soft limit, actual pruned messages may be less if it is greater than un-pruned messages.
    ///
    /// Returns the number of pruned messages.
    pub fn prune_raw_history_if(&mut self, un_pruned_len: usize, prune_len: usize) -> usize {
        let raw_history_len = self.req.raw_history.len().saturating_sub(self.pruned);
        if raw_history_len < un_pruned_len {
            return 0;
        }
        let pruned_len = prune_len.min(raw_history_len);
        for i in self.pruned..(self.pruned + pruned_len) {
            if let Ok(mut msg) = serde_json::from_value::<Message>(self.req.raw_history[i].clone())
                && msg.prune_content() > 0
                && let Ok(raw) = serde_json::to_value(&msg)
            {
                self.req.raw_history[i] = raw;
            }
        }
        self.pruned += pruned_len;
        pruned_len
    }

    /// Execute the next step.
    /// - Calls the model completion.
    /// - Automatically handles tool/agent calls and writes the results back to the conversation history.
    /// - If there are more steps, it constructs the next request and returns the current intermediate result.
    /// - If completed or failed, it returns the final result; the next call will return Ok(None).
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

    async fn inner_next(&mut self) -> Result<Option<AgentOutput>, BoxError> {
        self.step += 1;

        let req = self.req.clone();
        let label = req.model.as_ref().unwrap_or(&self.ctx.label);

        let model = self
            .ctx
            .models
            .get_model_by(label)
            .unwrap_or_else(|| self.ctx.model.clone());

        let mut output = model.completion(req.clone()).await?;
        output.model = Some(model.model_name());

        self.usage.accumulate(&output.usage);

        // If the primary model returns a failed result (failed_reason exists),
        // and a fallback model is configured, switch to the fallback model and retry.
        // After switching, subsequent steps will keep using the fallback model.
        if output.failed_reason.is_some()
            && let Some(fallback) = self.ctx.models.fallback_model()
        {
            let primary_reason = output
                .failed_reason
                .clone()
                .unwrap_or_else(|| "unknown error".to_string());

            let mut output2 = fallback.completion(req).await?;
            output2.model = Some(fallback.model_name());
            self.usage.accumulate(&output2.usage);

            if let Some(fallback_reason) = output2.failed_reason {
                output2.failed_reason = Some(format!(
                    "primary model failed: {}; fallback model failed: {}",
                    primary_reason, fallback_reason
                ));
                return Ok(Some(self.final_output(output2)));
            }

            output = output2;
        }

        // 关闭下一轮模型输出中的工具调用强制需求（如果有的话）
        self.req.tool_choice_required = false;
        self.req.output_schema = None;
        // 累计所有原始对话历史（包含初始的 req.raw_history 和 req.chat_history）
        self.req.raw_history.append(&mut output.raw_history);
        // 累计所有对话历史（不包含初始的 req.chat_history）
        self.chat_history.append(&mut output.chat_history);

        if let Some(steering) = self.steering_message.take() {
            // 准备下一轮转向请求
            if !output.tool_calls.is_empty() {
                // 去掉模型最后的 tool_calls 输出，避免对后续轮次造成干扰
                self.req.raw_history.pop();
            }

            self.req.chat_history.clear();
            self.req.documents.clear();
            self.req.content.clear();
            self.req.prompt = steering;
            self.req.role = Some("user".to_string());

            // 返回本轮的中间结果（带当前累计 usage；不强制覆盖 artifacts/tool_calls，
            // 让调用方查看模型本轮原始输出；最终轮会附带汇总）
            output.usage = self.usage.clone();
            // 本次 output 也包含当前所有对话
            output.chat_history = self.chat_history.clone();

            return Ok(Some(output));
        }

        // 自动执行工具/代理调用
        let mut tool_call_futs: Vec<BoxPinFut<(Option<ToolCall>, Option<String>)>> = Vec::new();
        let tool_calls = std::mem::take(&mut output.tool_calls);
        for mut tool in tool_calls.into_iter() {
            if self.ctx.cancellation_token().is_cancelled() {
                return Err("operation cancelled".into());
            }

            let tool_name = tool.name.to_ascii_lowercase();
            if self.ctx.tools.contains_lowercase(&tool_name) || tool_name.starts_with("rt_") {
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
                            (Some(tool), None)
                        }
                        Err(err) => {
                            // 工具调用失败了，但我们不一定要因此终止整个对话流程，可以让 LLM 尝试纠正错误并继续对话
                            {
                                tool.result = Some(ToolOutput {
                                    output: Json::String(format!("tool call error: {}", err)),
                                    ..Default::default()
                                });
                                (Some(tool), None)
                            }
                        }
                    }
                }));
            } else if self.ctx.agents.contains_lowercase(&tool_name)
                || self.ctx.subagents.contains_lowercase(&tool_name)
                || tool_name.starts_with("sa_")
                || tool_name.starts_with("ra_")
            {
                // 代理调用的 prompt 可能在 args 中的 "prompt" 字段，也可能直接在 args 的 JSON 中（如果 args 是一个字符串的话），也可能整个 args 就是 prompt（如果没有 "prompt" 字段的话）
                let prompt = if let Some(args) = tool.args.as_str() {
                    args.to_string()
                } else if let Some(args) = tool.args.get("prompt")
                    && let Some(prompt) = args.as_str()
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
                            tool.result = Some(ToolOutput {
                                output: if (res.content.starts_with("{")
                                    || res.content.starts_with("["))
                                    && let Ok(val) = serde_json::from_str(&res.content)
                                {
                                    val
                                } else {
                                    res.content.into()
                                },
                                artifacts: res.artifacts,
                                usage: res.usage,
                            });
                            if let Some(reason) = res.failed_reason {
                                (Some(tool), Some(reason))
                            } else {
                                (Some(tool), None)
                            }
                        }
                        Err(err) => (None, Some(err.to_string())),
                    }
                }));
            } else {
                tool_call_futs.push(Box::pin(async move {
                    tool.result = Some(ToolOutput {
                        output: Json::String(format!("tool call error: {} not found", tool.name)),
                        ..Default::default()
                    });
                    (Some(tool), None)
                }));
            }
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_calls_continue: Vec<ContentPart> = Vec::new();
        let mut tool_call_errors: Vec<String> = Vec::new();
        if !tool_call_futs.is_empty() {
            let results = futures::future::join_all(tool_call_futs).await;

            let mut selected_tools: Vec<FunctionDefinition> = Vec::new();
            for (tool, err) in results {
                if let Some(mut tool) = tool
                    && let Some(res) = &mut tool.result
                {
                    self.usage.accumulate(&res.usage);
                    if is_tools_select_name(&tool.name) {
                        // 从模型输出或工具调用结果中获取实际被选中的工具定义，传递给下一轮模型调用
                        if let Ok(selected) =
                            serde_json::from_value::<ToolsSelectOutput>(res.output.clone())
                        {
                            // 仅把选中的工具名称反馈给模型
                            res.output = json!(
                                selected
                                    .tools
                                    .iter()
                                    .map(|t| t.name.clone())
                                    .collect::<Vec<String>>()
                            );
                            selected_tools.extend(selected.tools);
                        }
                    }
                    // We can not ignore some tool calls.
                    // GPT-5: An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'.
                    tool_calls_continue.push(ContentPart::ToolOutput {
                        name: tool.name.clone(),
                        output: res.output.clone(),
                        call_id: tool.call_id.clone(),
                        remote_id: tool.remote_id,
                    });

                    self.artifacts.append(&mut res.artifacts);
                    tool_calls.push(tool);
                }
                if let Some(err) = err {
                    tool_call_errors.push(err);
                }
            }

            for tool in selected_tools {
                if !self.req.tools.iter().any(|t| t.name == tool.name) {
                    self.req.tools.push(tool);
                }
            }
        }

        // 累计当前轮的 tool_calls
        self.tool_calls.append(&mut tool_calls);

        if !tool_call_errors.is_empty() {
            output.failed_reason = Some(tool_call_errors.join("; "));
            return Ok(Some(self.final_output(output)));
        }

        // 如果有需要继续调用工具
        if !tool_calls_continue.is_empty() {
            // 准备下一轮请求
            self.req.chat_history.clear();
            self.req.documents.clear();
            self.req.content.clear();
            self.req.prompt.clear();
            // 追加到下一轮请求
            self.req.role = Some("tool".to_string());
            self.req.content.append(&mut tool_calls_continue);

            // 返回本轮的中间结果（带当前累计 usage；不强制覆盖 artifacts/tool_calls，
            // 让调用方查看模型本轮原始输出；最终轮会附带汇总）
            // // output.tool_calls = self.tool_calls_result.clone();
            // // output.artifacts = self.artifacts.clone();
            output.usage = self.usage.clone();
            // 本次 output 也包含当前所有对话
            output.chat_history = self.chat_history.clone();

            return Ok(Some(output));
        }

        // 如果有 steering_message / follow_up_message 则继续下一轮对话
        if let Some(prompt) = self
            .steering_message
            .take()
            .or_else(|| self.follow_up_message.take())
        {
            // 准备下一轮 steering 请求
            self.req.chat_history.clear();
            self.req.documents.clear();
            self.req.content.clear();
            self.req.prompt = prompt;
            self.req.role = Some("user".to_string());

            // 返回本轮的中间结果（带当前累计 usage；不强制覆盖 artifacts/tool_calls，
            // 让调用方查看模型本轮原始输出；最终轮会附带汇总）
            output.usage = self.usage.clone();
            // 本次 output 也包含当前所有对话
            output.chat_history = self.chat_history.clone();

            return Ok(Some(output));
        }

        Ok(Some(self.final_output(output)))
    }

    fn final_output(&mut self, mut output: AgentOutput) -> AgentOutput {
        self.done = true;
        self.chat_history.append(&mut output.chat_history);
        output.chat_history = std::mem::take(&mut self.chat_history);
        output.tool_calls = std::mem::take(&mut self.tool_calls);
        output.artifacts = std::mem::take(&mut self.artifacts);
        output.usage = std::mem::take(&mut self.usage);

        output
    }
}

pub struct CompletionStream {
    pub runner: CompletionRunner,
}

impl Stream for CompletionStream {
    type Item = Result<AgentOutput, BoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let future = self.runner.next();
        tokio::pin!(future);

        match future.poll(cx) {
            Poll::Ready(Ok(Some(output))) => Poll::Ready(Some(Ok(output))),
            Poll::Ready(Ok(None)) => Poll::Ready(None),
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use anda_core::{
        Agent, AgentContext, AgentOutput, BoxError, CancellationToken, CompletionFeatures,
        CompletionRequest, FunctionDefinition, Resource, Tool, ToolCall, ToolOutput, Usage,
    };
    use candid::Principal;
    use ciborium::from_reader;
    use futures_util::StreamExt;
    use ic_cose_types::to_cbor_bytes;
    use serde::Deserialize;
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::AgentCtx;
    use crate::context::{TOOLS_SELECT_NAME, base::BaseCtx};
    use crate::{
        engine::EngineBuilder,
        model::{CompletionFeaturesDyn, Model, Models},
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
        let val: serde_json::Value = from_reader(&data[..]).unwrap();
        assert_eq!(json, val);
    }

    // ── Helper completers ──

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

    #[derive(Clone, Debug)]
    struct AlwaysOkCompleter;

    impl CompletionFeaturesDyn for AlwaysOkCompleter {
        fn model_name(&self) -> String {
            "always_ok".to_string()
        }

        fn completion(
            &self,
            _req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content: "from_fallback".to_string(),
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
            Box::pin(futures::future::ready(Ok(AgentOutput {
                content: req.prompt.clone(),
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
                    args: json!({"prompt": "sub-agent task"}),
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

    /// Completer that first selects an agent, then invokes the selected agent.
    #[derive(Clone)]
    struct ToolsSelectFlowCompleter {
        calls: Arc<AtomicUsize>,
    }

    impl ToolsSelectFlowCompleter {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl CompletionFeaturesDyn for ToolsSelectFlowCompleter {
        fn model_name(&self) -> String {
            "tools_select_flow".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            let stage = self.calls.fetch_add(1, Ordering::SeqCst);

            Box::pin(futures::future::ready(match stage {
                0 => Ok(AgentOutput {
                    tool_calls: vec![ToolCall {
                        name: "tools_select".to_string(),
                        args: json!({"tools": ["echo_agent"]}),
                        call_id: Some("select_call".into()),
                        result: None,
                        remote_id: None,
                    }],
                    usage: Usage {
                        input_tokens: 2,
                        output_tokens: 4,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                }),
                1 => {
                    let tool_names: Vec<String> =
                        req.tools.iter().map(|tool| tool.name.clone()).collect();
                    assert_eq!(tool_names, vec!["echo_agent".to_string()]);

                    Ok(AgentOutput {
                        tool_calls: vec![ToolCall {
                            name: "echo_agent".to_string(),
                            args: json!({"prompt": "selected agent task"}),
                            call_id: Some("agent_call".into()),
                            result: None,
                            remote_id: None,
                        }],
                        usage: Usage {
                            input_tokens: 3,
                            output_tokens: 6,
                            cached_tokens: 0,
                            requests: 1,
                        },
                        ..Default::default()
                    })
                }
                2 => Ok(AgentOutput {
                    content: "tools_select_loaded".to_string(),
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 2,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                }),
                _ => panic!("unexpected completion stage {stage}"),
            }))
        }
    }

    /// Completer that first asks tools_select to resolve an agent by query, then invokes it.
    #[derive(Clone)]
    struct ToolsSelectQueryFlowCompleter {
        calls: Arc<AtomicUsize>,
    }

    impl ToolsSelectQueryFlowCompleter {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl CompletionFeaturesDyn for ToolsSelectQueryFlowCompleter {
        fn model_name(&self) -> String {
            "tools_select_query_flow".to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            let stage = self.calls.fetch_add(1, Ordering::SeqCst);

            Box::pin(futures::future::ready(match stage {
                0 => {
                    assert!(req.tools.iter().any(|t| t.name == TOOLS_SELECT_NAME));
                    Ok(AgentOutput {
                        tool_calls: vec![ToolCall {
                            name: TOOLS_SELECT_NAME.to_string(),
                            args: json!({"query": "mirror my text", "limit": 1}),
                            call_id: Some("select_call".into()),
                            result: None,
                            remote_id: None,
                        }],
                        usage: Usage {
                            input_tokens: 2,
                            output_tokens: 4,
                            cached_tokens: 0,
                            requests: 1,
                        },
                        ..Default::default()
                    })
                }
                1 => {
                    let tool_names: Vec<String> =
                        req.tools.iter().map(|tool| tool.name.clone()).collect();
                    assert_eq!(tool_names, vec!["tools_select", "echo_agent"]);

                    Ok(AgentOutput {
                        tool_calls: vec![ToolCall {
                            name: "echo_agent".to_string(),
                            args: json!({"prompt": "selected agent task"}),
                            call_id: Some("agent_call".into()),
                            result: None,
                            remote_id: None,
                        }],
                        usage: Usage {
                            input_tokens: 3,
                            output_tokens: 6,
                            cached_tokens: 0,
                            requests: 1,
                        },
                        ..Default::default()
                    })
                }
                2 => Ok(AgentOutput {
                    content: "tools_select_loaded".to_string(),
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 2,
                        cached_tokens: 0,
                        requests: 1,
                    },
                    ..Default::default()
                }),
                _ => panic!("unexpected completion stage {stage}"),
            }))
        }
    }

    #[derive(Clone)]
    struct ToolSelectorCompleter {
        calls: Arc<AtomicUsize>,
    }

    impl ToolSelectorCompleter {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl CompletionFeaturesDyn for ToolSelectorCompleter {
        fn model_name(&self) -> String {
            TOOLS_SELECT_NAME.to_string()
        }

        fn completion(
            &self,
            req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);

            assert!(req.tools.is_empty());
            assert!(req.prompt.contains("mirror my text"));
            assert!(req.prompt.contains("echo_agent"));

            Box::pin(futures::future::ready(Ok(AgentOutput {
                content: r#"{"tools":["echo_agent"]}"#.to_string(),
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

    /// Completer where both primary and fallback fail.
    #[derive(Clone, Debug)]
    struct AlwaysFailCompleter2;

    impl CompletionFeaturesDyn for AlwaysFailCompleter2 {
        fn model_name(&self) -> String {
            "always_fail_2".to_string()
        }

        fn completion(
            &self,
            _req: CompletionRequest,
        ) -> anda_core::BoxPinFut<Result<AgentOutput, BoxError>> {
            Box::pin(futures::future::ready(Ok(AgentOutput {
                failed_reason: Some("fallback also failed".to_string()),
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
                    }
                }),
                strict: None,
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
                artifacts: Vec::new(),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_tokens: 0,
                    requests: 1,
                },
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
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
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

    // ── Tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn completion_falls_back_on_failed_reason() {
        let primary = Model::with_completer(Arc::new(AlwaysFailCompleter));
        let fallback = Model::with_completer(Arc::new(AlwaysOkCompleter));

        let ctx = EngineBuilder::new()
            .with_model(primary)
            .with_fallback_model(fallback)
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let out = ctx.completion(req, Vec::<Resource>::new()).await.unwrap();
        assert!(out.failed_reason.is_none());
        assert_eq!(out.content, "from_fallback");
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
        assert_eq!(runner.steps(), 0);

        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(runner.steps(), 1);
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

    // ── Fallback model tests ──

    #[tokio::test(flavor = "current_thread")]
    async fn runner_fallback_on_primary_failure() {
        let primary = Model::with_completer(Arc::new(AlwaysFailCompleter));
        let fallback = Model::with_completer(Arc::new(AlwaysOkCompleter));

        let ctx = EngineBuilder::new()
            .with_model(primary)
            .with_fallback_model(fallback)
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(output.content, "from_fallback");
        assert!(output.failed_reason.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_both_primary_and_fallback_fail() {
        let primary = Model::with_completer(Arc::new(AlwaysFailCompleter));
        let fallback = Model::with_completer(Arc::new(AlwaysFailCompleter2));

        let ctx = EngineBuilder::new()
            .with_model(primary)
            .with_fallback_model(fallback)
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "hello".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        let output = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert!(output.failed_reason.is_some());
        let reason = output.failed_reason.unwrap();
        assert!(reason.contains("primary model failed"));
        assert!(reason.contains("fallback model failed"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_no_fallback_on_primary_failure() {
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

    #[tokio::test(flavor = "current_thread")]
    async fn runner_fallback_used_only_once() {
        // After fallback succeeds on step 1, step 2 should use the fallback model.
        let primary = Model::with_completer(Arc::new(AlwaysFailCompleter));
        let fallback = Model::with_completer(Arc::new(EchoCompleter));

        let ctx = EngineBuilder::new()
            .with_model(primary)
            .with_fallback_model(fallback)
            .mock_ctx();

        let req = CompletionRequest {
            prompt: "test".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());
        // Set follow-up so the runner won't finish after step 1.
        runner.follow_up("follow-up prompt".to_string());

        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "test"); // from EchoCompleter (fallback)

        // Step 2 uses fallback model (now primary) and processes follow-up.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step2.content, "follow-up prompt"); // echo from fallback model
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
                .contains("agent_echoed:sub-agent task")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_tools_select_loads_selected_agents_for_the_next_turn() {
        let completer = ToolsSelectFlowCompleter::new();
        let call_counter = completer.calls.clone();

        let model = Model::with_completer(Arc::new(completer));
        let engine = EngineBuilder::new()
            .with_model(model)
            .register_agent(Arc::new(EchoAgent), None)
            .unwrap()
            .build("echo_agent".to_string())
            .await
            .unwrap();
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let req = CompletionRequest {
            prompt: "select agent then call it".to_string(),
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        let final_out = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(final_out.content, "tools_select_loaded");
        assert_eq!(call_counter.load(Ordering::SeqCst), 3);

        let tool_names: Vec<&str> = final_out
            .tool_calls
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        assert_eq!(tool_names, vec!["tools_select", "echo_agent"]);
        assert!(
            final_out
                .tool_calls
                .iter()
                .all(|tool| tool.result.is_some())
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runner_tools_select_query_loads_selected_agents_for_the_next_turn() {
        let completer = ToolsSelectQueryFlowCompleter::new();
        let call_counter = completer.calls.clone();
        let selector = ToolSelectorCompleter::new();
        let selector_calls = selector.calls.clone();

        let models = Arc::new(Models::default());
        models.set_model(Model::with_completer(Arc::new(completer)));
        models.set_model_by(
            "flash".to_string(),
            Model::with_completer(Arc::new(selector)),
        );

        let engine = EngineBuilder::new()
            .set_models(models)
            .register_agent(Arc::new(EchoAgent), None)
            .unwrap()
            .build("echo_agent".to_string())
            .await
            .unwrap();
        let ctx = engine
            .ctx_with(
                Principal::anonymous(),
                "echo_agent",
                "echo_agent",
                Default::default(),
            )
            .unwrap();

        let req = CompletionRequest {
            prompt: "select agent by intent then call it".to_string(),
            tools: ctx
                .definitions(Some(&[TOOLS_SELECT_NAME.to_string()]))
                .await,
            ..Default::default()
        };

        let mut runner = ctx.completion_iter(req, Vec::new());

        let _rt = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        let _rt = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());

        let final_out = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(final_out.content, "tools_select_loaded");
        assert_eq!(call_counter.load(Ordering::SeqCst), 3);
        assert_eq!(selector_calls.load(Ordering::SeqCst), 1);

        let tool_names: Vec<&str> = final_out
            .tool_calls
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        assert_eq!(tool_names, vec![TOOLS_SELECT_NAME, "echo_agent"]);
        assert!(
            final_out
                .tool_calls
                .iter()
                .all(|tool| tool.result.is_some())
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

        // Step 1: steering intercepts.
        let step1 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step1.content, "initial");

        // Step 2: processes the steering prompt.
        let step2 = runner.next().await.unwrap().unwrap();
        assert!(!runner.is_done());
        assert_eq!(step2.content, "steering");

        // Step 3: processes the follow-up prompt.
        let step3 = runner.next().await.unwrap().unwrap();
        assert!(runner.is_done());
        assert_eq!(step3.content, "follow_up");
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
        assert_eq!(runner.steps(), 0);

        runner.next().await.unwrap(); // step 1
        assert_eq!(runner.steps(), 1);

        runner.next().await.unwrap(); // step 2 (final)
        assert_eq!(runner.steps(), 2);
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
                    parameters: json!({"type": "object", "properties": {}}),
                    strict: None,
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
                    usage: Usage::default(),
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

    fn raw_message(content: Vec<anda_core::ContentPart>) -> serde_json::Value {
        serde_json::to_value(anda_core::Message {
            role: "assistant".to_string(),
            content,
            name: None,
            user: None,
            timestamp: None,
        })
        .unwrap()
    }

    fn decode_message(raw: &serde_json::Value) -> anda_core::Message {
        serde_json::from_value(raw.clone()).unwrap()
    }

    fn pruned_placeholder(count: usize) -> anda_core::ContentPart {
        anda_core::ContentPart::Text {
            text: format!(
                "[{} items (tool calls or files) pruned due to limits]",
                count
            ),
        }
    }

    #[test]
    fn runner_prune_raw_history_if_noop_below_threshold() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let raw = raw_message(vec![
            anda_core::ContentPart::Text {
                text: "keep".to_string(),
            },
            anda_core::ContentPart::ToolCall {
                name: "echo_tool".to_string(),
                args: json!({ "input": "hello" }),
                call_id: Some("call_1".to_string()),
            },
        ]);

        let req = CompletionRequest {
            raw_history: vec![raw.clone()],
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());

        assert_eq!(runner.prune_raw_history_if(2, 10), 0);
        assert_eq!(runner.pruned, 0);
        assert_eq!(runner.req.raw_history, vec![raw]);
    }

    #[test]
    fn runner_prune_raw_history_if_prunes_prefix_and_continues_from_cursor() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let first = raw_message(vec![
            anda_core::ContentPart::Text {
                text: "keep".to_string(),
            },
            anda_core::ContentPart::ToolCall {
                name: "echo_tool".to_string(),
                args: json!({ "input": "hello" }),
                call_id: Some("call_1".to_string()),
            },
            anda_core::ContentPart::FileData {
                file_uri: "file:///tmp/a.txt".to_string(),
                mime_type: None,
            },
        ]);
        let second = raw_message(vec![anda_core::ContentPart::Text {
            text: "text only".to_string(),
        }]);
        let third = raw_message(vec![anda_core::ContentPart::ToolOutput {
            name: "echo_tool".to_string(),
            output: json!("done"),
            call_id: Some("call_2".to_string()),
            remote_id: None,
        }]);

        let req = CompletionRequest {
            raw_history: vec![first, second.clone(), third.clone()],
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());

        assert_eq!(runner.prune_raw_history_if(2, 2), 2);
        assert_eq!(runner.pruned, 2);

        let first_after = decode_message(&runner.req.raw_history[0]);
        assert_eq!(
            first_after.content,
            vec![
                anda_core::ContentPart::Text {
                    text: "keep".to_string(),
                },
                pruned_placeholder(2),
            ]
        );
        assert_eq!(
            decode_message(&runner.req.raw_history[1]),
            decode_message(&second)
        );
        assert_eq!(runner.req.raw_history[2], third);

        assert_eq!(runner.prune_raw_history_if(0, 2), 1);
        assert_eq!(runner.pruned, 3);
        assert_eq!(decode_message(&runner.req.raw_history[0]), first_after);
        assert_eq!(
            decode_message(&runner.req.raw_history[2]).content,
            vec![pruned_placeholder(1)]
        );
    }

    #[test]
    fn runner_prune_raw_history_if_skips_invalid_raw_entries_and_advances_cursor() {
        let model = Model::with_completer(Arc::new(EchoCompleter));
        let ctx = EngineBuilder::new().with_model(model).mock_ctx();

        let invalid = json!({ "provider": "opaque raw message" });
        let valid = raw_message(vec![anda_core::ContentPart::ToolCall {
            name: "echo_tool".to_string(),
            args: json!({ "input": "hello" }),
            call_id: Some("call_1".to_string()),
        }]);

        let req = CompletionRequest {
            raw_history: vec![invalid.clone(), valid.clone()],
            ..Default::default()
        };
        let mut runner = ctx.completion_iter(req, Vec::new());

        assert_eq!(runner.prune_raw_history_if(1, 1), 1);
        assert_eq!(runner.pruned, 1);
        assert_eq!(runner.req.raw_history[0], invalid);
        assert_eq!(runner.req.raw_history[1], valid);

        assert_eq!(runner.prune_raw_history_if(0, 1), 1);
        assert_eq!(runner.pruned, 2);
        assert_eq!(
            decode_message(&runner.req.raw_history[1]).content,
            vec![pruned_placeholder(1)]
        );
    }
}
