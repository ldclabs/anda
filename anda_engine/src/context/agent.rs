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
    AgentArgs, AgentContext, AgentInput, AgentOutput, AgentSet, BaseContext, BoxError, CacheExpiry,
    CacheFeatures, CacheStoreFeatures, CancellationToken, CanisterCaller, CompletionFeatures,
    CompletionRequest, ContentPart, Embedding, EmbeddingFeatures, FunctionDefinition, HttpFeatures,
    Json, KeysFeatures, Message, ObjectMeta, Path, PutMode, PutResult, RequestMeta, Resource,
    StateFeatures, StoreFeatures, ToolCall, ToolInput, ToolOutput, ToolSet, Usage,
};
use bytes::Bytes;
use candid::{CandidType, Principal, utils::ArgumentEncoder};
use futures_util::Stream;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use super::{base::BaseCtx, engine::RemoteEngines};
use crate::model::Model;

pub static DYNAMIC_REMOTE_ENGINES: &str = "_engines";

/// Context for agent operations, providing access to models, tools, and other agents.
#[derive(Clone)]
pub struct AgentCtx {
    /// Base context providing fundamental operations.
    pub base: BaseCtx,

    /// Label of the agent.
    pub(crate) label: String,
    /// AI model used for completions and embeddings.
    pub(crate) model: Model,
    // label -> model
    pub(crate) models: Arc<BTreeMap<String, Model>>,
    /// Set of available tools that can be called.
    pub(crate) tools: Arc<ToolSet<BaseCtx>>,
    /// Set of available agents that can be invoked.
    pub(crate) agents: Arc<AgentSet<AgentCtx>>,
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
        model: Model,
        models: Arc<BTreeMap<String, Model>>,
        tools: Arc<ToolSet<BaseCtx>>,
        agents: Arc<AgentSet<AgentCtx>>,
    ) -> Self {
        Self {
            base,
            label: String::new(),
            model,
            models,
            tools,
            agents,
        }
    }

    /// Creates a child context for a specific agent.
    ///
    /// # Arguments
    /// * `agent_name` - Name of the agent to create context for.
    pub(crate) fn child(&self, agent_name: &str, agent_label: &str) -> Result<Self, BoxError> {
        Ok(Self {
            base: self.base.child(format!("A:{}", agent_name))?,
            label: agent_label.to_string(),
            model: self.model.clone(),
            models: self.models.clone(),
            tools: self.tools.clone(),
            agents: self.agents.clone(),
        })
    }

    /// Creates a child base context for a specific tool.
    ///
    /// # Arguments
    /// * `tool_name` - Name of the tool to create context for.
    pub(crate) fn child_base(&self, tool_name: &str) -> Result<BaseCtx, BoxError> {
        self.base.child(format!("T:{}", tool_name))
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
            base: self
                .base
                .child_with(caller, format!("A:{}", agent_name), meta)?,
            label: agent_label.to_string(),
            model: self.model.clone(),
            models: self.models.clone(),
            tools: self.tools.clone(),
            agents: self.agents.clone(),
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
        tool_name: &str,
        meta: RequestMeta,
    ) -> Result<BaseCtx, BoxError> {
        self.base
            .child_with(caller, format!("T:{}", tool_name), meta)
    }

    /// Creates a completion runner for iterative processing of completion requests.
    pub fn completion_iter(
        &self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> CompletionRunner {
        let model = self
            .models
            .get(&self.label)
            .cloned()
            .unwrap_or_else(|| self.model.clone());

        CompletionRunner {
            ctx: self.clone(),
            model,
            req,
            resources,
            chat_history: Vec::new(),
            tool_calls: Vec::new(),
            usage: Usage::default(),
            artifacts: Vec::new(),
            done: false,
            step: 0,
        }
    }

    /// Creates a completion stream for processing of completion requests.
    pub fn completion_stream(
        &self,
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
    fn tool_definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
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
        names: Option<&[&str]>,
    ) -> Result<Vec<FunctionDefinition>, BoxError> {
        let mut defs = self.base.remote.tool_definitions(endpoint, names);
        if let Ok((engines, _)) = self
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
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if !name.starts_with("RT_") {
            return self.tools.select_resources(name, resources);
        }

        let res = self.base.remote.select_tool_resources(name, resources);
        if !res.is_empty() {
            return res;
        }

        if let Ok((engines, _)) = self
            .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
            .await
        {
            return engines.select_tool_resources(name, resources);
        }

        Vec::new()
    }

    /// Retrieves definitions for available agents.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific agent names;
    /// * `with_prefix` - Flag to add the prefix `LA_` to agent names to distinguish from tools.
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents.
    fn agent_definitions(
        &self,
        names: Option<&[&str]>,
        with_prefix: bool,
    ) -> Vec<FunctionDefinition> {
        let res = self.agents.definitions(names);
        if with_prefix {
            res.into_iter()
                .map(|mut d| {
                    d.name = format!("LA_{}", d.name);
                    d
                })
                .collect()
        } else {
            res
        }
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
        names: Option<&[&str]>,
    ) -> Result<Vec<FunctionDefinition>, BoxError> {
        let mut defs = self.base.remote.agent_definitions(endpoint, names);
        if let Ok((engines, _)) = self
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
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if !name.starts_with("RA_") {
            let name = name.strip_prefix("LA_").unwrap_or(name);
            return self
                .agents
                .select_resources(&name.to_ascii_lowercase(), resources);
        }

        let res = self.base.remote.select_agent_resources(name, resources);
        if !res.is_empty() {
            return res;
        }

        if let Ok((engines, _)) = self
            .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
            .await
        {
            return engines.select_agent_resources(name, resources);
        }

        Vec::new()
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
        if input.name.starts_with("rt_") {
            input.name.replace_range(..3, "RT_");
        }

        if !input.name.starts_with("RT_") {
            let ctx = self.child_base(&input.name)?;
            let tool = self
                .tools
                .get(&input.name)
                .ok_or_else(|| format!("tool {} not found", &input.name))?;
            return tool
                .call(ctx, input.args, input.resources)
                .await
                .map(|output| (output, None));
        }

        // find registered remote tool and call it
        if let Some((id, endpoint, tool_name)) = self.base.remote.get_tool_endpoint(&input.name) {
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

        Err(format!("tool {} not found", &input.name).into())
    }

    /// Runs a local agent.
    ///
    /// # Arguments
    /// * `args` - Tool input arguments, [`AgentInput`].
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    async fn agent_run(
        &self,
        mut input: AgentInput,
    ) -> Result<(AgentOutput, Option<Principal>), BoxError> {
        if input.name.starts_with("la_") {
            input.name.replace_range(..3, "LA_");
        }
        if input.name.starts_with("ra_") {
            input.name.replace_range(..3, "RA_");
        }

        if !input.name.starts_with("RA_") {
            let name = input.name.strip_prefix("LA_").unwrap_or(&input.name);
            let name = name.to_ascii_lowercase();
            let agent = self
                .agents
                .get(&name)
                .ok_or_else(|| format!("agent {} not found", name))?;
            let ctx = self.child(&name, agent.label())?;
            return agent
                .run(ctx, input.prompt, input.resources)
                .await
                .map(|output| (output, None));
        }

        // find registered remote agent and run it
        if let Some((id, endpoint, agent_name)) = self.base.remote.get_agent_endpoint(&input.name) {
            input.name = agent_name;
            input.meta = Some(self.base.self_meta(id));
            return self
                .remote_agent_run(&endpoint, input)
                .await
                .map(|output| (output, Some(id)));
        }

        // find dynamic remote agent and run it
        if let Ok((engines, _)) = self
            .cache_store_get::<RemoteEngines>(DYNAMIC_REMOTE_ENGINES)
            .await
            && let Some((id, endpoint, agent_name)) = engines.get_agent_endpoint(&input.name)
        {
            input.name = agent_name;
            input.meta = Some(self.base.self_meta(id));
            return self
                .remote_agent_run(&endpoint, input)
                .await
                .map(|output| (output, Some(id)));
        }

        Err(format!("agent {} not found", input.name).into())
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
    async fn completion(
        &self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let mut runner = self.completion_iter(req, resources);
        let mut last: Option<AgentOutput> = None;

        while let Some(step) = runner.next().await? {
            // 出错即返回（保持原行为）
            if step.failed_reason.is_some() {
                return Ok(step);
            }
            last = Some(step);
        }

        last.ok_or_else(|| "completion runner returned no output".into())
    }
}

impl EmbeddingFeatures for AgentCtx {
    /// Gets the number of dimensions for the embedding model.
    fn ndims(&self) -> usize {
        self.model.ndims()
    }

    /// Generates embeddings for a collection of texts.
    ///
    /// # Arguments
    /// * `texts` - Collection of text strings to embed.
    ///
    /// # Returns
    /// Vector of embeddings, one for each input text.
    async fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<(Vec<Embedding>, Usage), BoxError> {
        self.model.embed(texts).await
    }

    /// Generates an embedding for a single query text.
    ///
    /// # Arguments
    /// * `text` - Input text to embed.
    ///
    /// # Returns
    /// Embedding vector for the input text.
    async fn embed_query(&self, text: &str) -> Result<(Embedding, Usage), BoxError> {
        self.model.embed_query(text).await
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
    model: Model,
    req: CompletionRequest,
    resources: Vec<Resource>,
    chat_history: Vec<Message>,
    tool_calls: Vec<ToolCall>,
    usage: Usage,
    artifacts: Vec<Resource>,
    done: bool,
    step: usize,
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
        let mut output = self.model.completion(self.req.clone()).await?;
        self.usage.accumulate(&output.usage);
        // 累计所有原始对话历史（包含初始的 req.raw_history 和 req.chat_history）
        self.req.raw_history.append(&mut output.raw_history);
        // 累计所有对话历史（不包含初始的 req.chat_history）
        self.chat_history.append(&mut output.chat_history);

        // 自动执行工具/代理调用
        let mut tool_calls_continue: Vec<ContentPart> = Vec::new();
        for tool in output.tool_calls.iter_mut() {
            if self.ctx.cancellation_token().is_cancelled() {
                return Err("operation cancelled".into());
            }

            if self.ctx.tools.contains(&tool.name)
                || tool.name.starts_with("RT_")
                || tool.name.starts_with("rt_")
            {
                match self
                    .ctx
                    .tool_call(ToolInput {
                        name: tool.name.clone(),
                        args: tool.args.clone(),
                        resources: self
                            .ctx
                            .select_tool_resources(&tool.name, &mut self.resources)
                            .await,
                        meta: None,
                    })
                    .await
                {
                    Ok((mut res, remote_id)) => {
                        self.usage.accumulate(&res.usage);

                        // We can not ignore some tool calls.
                        // GPT-5: An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'.
                        tool_calls_continue.push(ContentPart::ToolOutput {
                            name: tool.name.clone(),
                            output: res.output.clone(),
                            call_id: tool.call_id.clone(),
                            remote_id,
                        });

                        self.artifacts.append(&mut res.artifacts);
                        tool.remote_id = remote_id;
                        tool.result = Some(res);
                    }
                    Err(err) => {
                        output.failed_reason = Some(err.to_string());
                        return Ok(Some(self.final_output(output)));
                    }
                }
            } else if self.ctx.agents.contains(&tool.name)
                || tool.name.starts_with("LA_")
                || tool.name.starts_with("la_")
                || tool.name.starts_with("RA_")
                || tool.name.starts_with("ra_")
            {
                // 代理调用
                let args: AgentArgs = match serde_json::from_value(tool.args.clone()) {
                    Ok(args) => args,
                    Err(err) => {
                        output.failed_reason = Some(format!(
                            "failed to parse agent args {:?}: {}",
                            tool.args, err
                        ));
                        return Ok(Some(self.final_output(output)));
                    }
                };
                match self
                    .ctx
                    .agent_run(AgentInput {
                        name: tool.name.clone(),
                        prompt: args.prompt,
                        resources: self
                            .ctx
                            .agents
                            .select_resources(&tool.name, &mut self.resources),
                        ..Default::default()
                    })
                    .await
                {
                    Ok((mut res, remote_id)) => {
                        self.usage.accumulate(&res.usage);
                        if res.failed_reason.is_some() {
                            output.failed_reason = res.failed_reason;
                            return Ok(Some(self.final_output(output)));
                        }

                        // TODO: remote agent id
                        tool_calls_continue.push(ContentPart::ToolOutput {
                            name: tool.name.clone(),
                            output: res.content.clone().into(),
                            call_id: tool.call_id.clone(),
                            remote_id,
                        });

                        self.artifacts.append(&mut res.artifacts);
                        tool.result = Some(ToolOutput {
                            output: res.content.clone().into(),
                            artifacts: vec![],
                            usage: res.usage,
                        });
                    }
                    Err(err) => {
                        output.failed_reason = Some(err.to_string());
                        return Ok(Some(self.final_output(output)));
                    }
                }
            }
            // 未知工具名，忽略
        }

        // 累计当前轮的 tool_calls
        self.tool_calls.append(&mut output.tool_calls);

        // 若无需继续，返回最终结果并结束
        if tool_calls_continue.is_empty() {
            return Ok(Some(self.final_output(output)));
        }

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

        Ok(Some(output))
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
    runner: CompletionRunner,
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
    use ciborium::from_reader;
    use ic_cose_types::to_cbor_bytes;
    use serde_json::json;

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
}
