//! Tool traits and registries.
//!
//! Tools are reusable capabilities that agents can call through the runtime.
//! This module provides:
//! - [`Tool`] for strongly typed tool implementations.
//! - [`DynTool`] for runtime dispatch through trait objects.
//! - [`ToolSet`] for name-based registration and lookup.
//!
//! Tools define their own JSON function schema through [`FunctionDefinition`]
//! and receive typed arguments after the runtime validates and deserializes a
//! raw JSON call.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    future::Future,
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    BoxError, BoxFut, BoxPinFut, Function, Json, Resource, ToolInput, ToolOutput,
    context::BaseContext, model::FunctionDefinition, select_resources, validate_function_name,
};

/// Strongly typed interface for an agent tool.
///
/// # Type Parameters
/// - `C`: Runtime context implementing [`BaseContext`].
pub trait Tool<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// The arguments type of the tool.
    type Args: DeserializeOwned + Send;

    /// The output type of the tool.
    type Output: Serialize;

    /// Returns the unique tool name.
    ///
    /// # Rules
    /// - Must not be empty;
    /// - Must not exceed 64 bytes;
    /// - Must start with a lowercase letter;
    /// - Can only contain: lowercase letters (a-z), digits (0-9), underscores (_), and hyphens (-);
    /// - Unique within the engine.
    fn name(&self) -> String;

    /// Returns a concise description of the tool's capability.
    fn description(&self) -> String;

    /// Returns the function definition, including the JSON parameter schema.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The schema definition of the tool's parameters and metadata.
    fn definition(&self) -> FunctionDefinition;

    /// Returns the capability group this tool belongs to, if any.
    ///
    /// Tools that form a coherent bundle (for example the filesystem workspace
    /// tools) return the same [`ToolGroupInfo`] so the registry can present them
    /// as one group in discovery. The default implementation returns `None`.
    fn group(&self) -> Option<ToolGroupInfo> {
        None
    }

    /// Returns resource tags this tool can consume.
    ///
    /// The default implementation returns an empty list, meaning no resources
    /// are selected for this tool. Return `vec!["*".into()]` to accept all
    /// attached resources.
    ///
    /// # Returns
    /// Resource tags supported by this tool.
    fn supported_resource_tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Removes and returns resources matching this tool's supported tags.
    fn select_resources(&self, resources: &mut Vec<Resource>) -> Vec<Resource> {
        let supported_tags = self.supported_resource_tags();
        select_resources(resources, &supported_tags)
    }

    /// Initializes the tool with the given context.
    ///
    /// Runtimes call this once while building the engine.
    fn init(&self, _ctx: C) -> impl Future<Output = Result<(), BoxError>> + Send {
        std::future::ready(Ok(()))
    }

    /// Executes the tool with typed arguments and selected resources.
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing [`BaseContext`].
    /// - `args`: struct arguments for the tool.
    /// - `resources`: Additional resources selected for this tool.
    ///
    /// # Returns
    /// A future resolving to [`ToolOutput<Self::Output>`].
    fn call(
        &self,
        ctx: C,
        args: Self::Args,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<ToolOutput<Self::Output>, BoxError>> + Send;

    /// Executes the tool from raw JSON arguments and returns JSON output.
    fn call_raw(
        &self,
        ctx: C,
        args: Json,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<ToolOutput<Json>, BoxError>> + Send {
        async move {
            let args: Self::Args = serde_json::from_value(args)
                .map_err(|err| format!("tool {}, invalid args: {}", self.name(), err))?;
            let mut result = self
                .call(ctx, args, resources)
                .await
                .map_err(|err| format!("tool {}, call failed: {}", self.name(), err))?;
            let output = serde_json::to_value(&result.output)?;
            if result.usage.requests == 0 {
                result.usage.requests = 1;
            }

            Ok(ToolOutput {
                output,
                is_error: result.is_error,
                artifacts: result.artifacts,
                usage: result.usage,
                tools_usage: result.tools_usage,
            })
        }
    }
}

/// Object-safe wrapper around [`Tool`] for runtime dispatch.
///
/// Runtime registries store tools through this trait so callers can select and
/// execute tools by name without knowing their concrete Rust types.
pub trait DynTool<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// Returns this tool as [`Any`] for type inspection.
    fn as_any(&self) -> &(dyn Any + Send + Sync);

    /// Converts the shared tool into [`Any`] for downcasting.
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the unique tool name.
    fn name(&self) -> String;

    /// Returns the function definition exposed to model providers.
    fn definition(&self) -> FunctionDefinition;

    /// Returns the capability group this tool belongs to, if any.
    fn group(&self) -> Option<ToolGroupInfo> {
        None
    }

    /// Returns resource tags this tool can consume.
    fn supported_resource_tags(&self) -> Vec<String>;

    /// Initializes the tool through object-safe dispatch.
    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>>;

    /// Executes the tool through object-safe dispatch with raw JSON arguments.
    fn call(
        &self,
        ctx: C,
        args: Json,
        resources: Vec<Resource>,
    ) -> BoxPinFut<Result<ToolOutput<Json>, BoxError>>;
}

/// Group membership a single [`Tool`] declares for itself.
///
/// A static tool uses this to say "I belong to bundle X" without knowing the
/// other members. The registry ([`ToolSet`]) collects every tool that declares
/// the same `id` and assembles the full [`ToolGroup`], so the member list always
/// reflects the tools actually registered (no stale or missing entries).
///
/// Share one constructor across a bundle's tools to keep the metadata identical;
/// when ids collide, the first-registered tool's metadata wins.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolGroupInfo {
    /// Stable group id, unique across the engine (for example `fs_workspace`).
    pub id: String,
    /// Human-facing group title.
    pub title: String,
    /// Concise summary of what this bundle of tools does.
    pub description: String,
    /// Optional usage instructions describing how the member tools work
    /// together. Reference for the model, never a runtime directive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// A related set of callables surfaced together from one source.
///
/// A group tells the model that a bundle of tools share an origin (for example
/// a single MCP server, or the built-in filesystem tools) and are meant to be
/// combined to complete related work. Groups are a *discovery-layer* concept
/// only: they are never sent to model providers as part of the function-calling
/// schema. They are returned by the built-in discovery helpers (`tools_search` /
/// `tools_select`) so the model can understand a bundle's purpose and pull in
/// sibling tools as needed.
///
/// `instructions`, `title`, and `description` may originate from untrusted
/// remote metadata. They are surfaced as plain data the model reads, never as
/// system instructions, so they cannot escalate into runtime directives.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolGroup {
    /// Stable group id, unique across providers (for example `mcp:filesystem`).
    pub id: String,
    /// Human-facing group title.
    pub title: String,
    /// Concise summary of what this bundle of tools does.
    pub description: String,
    /// Optional usage instructions describing how the member tools work
    /// together. Untrusted remote metadata; treat as reference, not directives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Model-facing names of the tools that belong to this group.
    pub members: Vec<String>,
}

impl ToolGroup {
    /// Builds a group from a per-tool [`ToolGroupInfo`] and resolved members.
    pub fn from_info(info: ToolGroupInfo, members: Vec<String>) -> Self {
        Self {
            id: info.id,
            title: info.title,
            description: info.description,
            instructions: info.instructions,
            members,
        }
    }
}

/// Dynamic source of callable tools.
///
/// Providers are useful for integrations whose tool set is discovered at
/// runtime, such as remote MCP servers. A provider exposes a synchronous
/// snapshot for model-facing discovery and async methods for refresh and call
/// execution.
pub trait ToolProvider<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// Returns the provider registry name.
    ///
    /// This name is for engine configuration and diagnostics, not a
    /// model-facing tool name.
    fn name(&self) -> String;

    /// Returns the current function definitions from this provider.
    ///
    /// Definition names must satisfy [`validate_function_name`] (lowercase ASCII
    /// letters, digits, `_`, and `-`, starting with a lowercase letter). The
    /// registry lowercases names defensively, but providers should return
    /// already-legal local names so dispatch and discovery stay consistent.
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Returns the capability groups exposed by this provider.
    ///
    /// Each group bundles related tools (for example all tools from one MCP
    /// server) so the discovery layer can tell the model the tools are related
    /// and how to combine them. The default implementation returns no groups.
    fn groups(&self) -> Vec<ToolGroup> {
        Vec::new()
    }

    /// Returns whether this provider can currently dispatch the lowercase name.
    ///
    /// The default implementation allocates and materializes a definition
    /// snapshot on every call. Providers on hot dispatch paths should override it
    /// with a direct lookup (as the MCP provider does).
    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.definitions(Some(&[lowercase_name.to_string()]))
            .iter()
            .any(|definition| definition.name.eq_ignore_ascii_case(lowercase_name))
    }

    /// Returns resource tags this provider's named tool can consume.
    fn supported_resource_tags(&self, _name: &str) -> Vec<String> {
        Vec::new()
    }

    /// Removes and returns resources matching the named tool.
    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        let supported_tags = self.supported_resource_tags(name);
        select_resources(resources, &supported_tags)
    }

    /// Initializes the provider and refreshes any runtime discovery cache.
    fn init(&self, _ctx: C) -> BoxFut<'_, Result<(), BoxError>> {
        Box::pin(async { Ok(()) })
    }

    /// Refreshes the provider's discovery cache.
    fn refresh(&self) -> BoxFut<'_, Result<(), BoxError>> {
        Box::pin(async { Ok(()) })
    }

    /// Executes a provider-backed tool by model-facing name.
    fn call(
        &self,
        ctx: C,
        input: ToolInput<Json>,
    ) -> BoxFut<'_, Result<ToolOutput<Json>, BoxError>>;
}

impl<C> dyn DynTool<C>
where
    C: BaseContext + Send + Sync + 'static,
{
    /// Returns the inner concrete tool type when it matches `T`.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: Tool<C> + 'static,
    {
        self.as_any().downcast_ref::<T>()
    }

    /// Returns the inner concrete tool when it matches `T`.
    pub fn downcast<T>(self: Arc<Self>) -> Result<Arc<T>, Arc<Self>>
    where
        T: Tool<C> + 'static,
    {
        match self.clone().into_any().downcast::<T>() {
            Ok(tool) => Ok(tool),
            Err(_) => Err(self),
        }
    }
}

/// Adapter that exposes a concrete [`Tool`] through [`DynTool`].
struct ToolWrapper<T, C>(Arc<T>, PhantomData<C>)
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static;

impl<T, C> DynTool<C> for ToolWrapper<T, C>
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static,
{
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self.0.as_ref()
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    fn name(&self) -> String {
        self.0.name()
    }

    fn definition(&self) -> FunctionDefinition {
        self.0.definition()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        self.0.group()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.0.supported_resource_tags()
    }

    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>> {
        let tool = self.0.clone();
        Box::pin(async move { tool.init(ctx).await })
    }

    fn call(
        &self,
        ctx: C,
        args: Json,
        resources: Vec<Resource>,
    ) -> BoxPinFut<Result<ToolOutput<Json>, BoxError>> {
        let tool = self.0.clone();
        Box::pin(async move { tool.call_raw(ctx, args, resources).await })
    }
}

/// Name-based registry for tools.
///
/// # Type Parameters
/// - `C`: The context type that implements [`BaseContext`].
#[derive(Default)]
pub struct ToolSet<C: BaseContext> {
    /// Registered tools keyed by their lowercase function names.
    ///
    /// # Invariant
    /// Keys must be lowercase names satisfying [`validate_function_name`] and
    /// must equal the tool's own lowercased name. [`ToolSet::add`] enforces this;
    /// code that mutates this map directly is responsible for upholding it, since
    /// lookup and dispatch assume lowercase keys.
    pub set: BTreeMap<String, Arc<dyn DynTool<C>>>,
}

/// Registry for runtime-discovered tool providers.
#[derive(Default)]
pub struct ToolProviderSet<C: BaseContext> {
    /// Registered providers keyed by provider name.
    ///
    /// # Invariant
    /// Keys must be lowercase names satisfying [`validate_function_name`].
    /// [`ToolProviderSet::add`] enforces this; direct mutation must uphold it.
    pub set: BTreeMap<String, Arc<dyn ToolProvider<C>>>,
}

impl<C> ToolProviderSet<C>
where
    C: BaseContext + Clone + Send + Sync + 'static,
{
    /// Creates an empty provider set.
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Returns whether a provider with the given name exists.
    pub fn contains_provider(&self, name: &str) -> bool {
        self.set.contains_key(&name.to_ascii_lowercase())
    }

    /// Registers a new dynamic tool provider.
    pub fn add<T>(&mut self, provider: Arc<T>) -> Result<(), BoxError>
    where
        T: ToolProvider<C> + Send + Sync + 'static,
    {
        let name = provider.name().to_ascii_lowercase();
        validate_function_name(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("tool provider {} already exists", name).into());
        }

        self.set.insert(name, provider);
        Ok(())
    }

    /// Returns whether any provider can currently dispatch the given name.
    pub fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.set
            .values()
            .any(|provider| provider.contains_lowercase(lowercase_name))
    }

    /// Returns dynamic function definitions for all providers or selected names.
    ///
    /// Definition names are normalized to lowercase so downstream lookups
    /// (dispatch, `supported_resource_tags`) stay consistent even if a provider
    /// returns a mixed-case name, and so duplicate names across providers are
    /// deduplicated by their canonical lowercase form.
    pub fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        match names {
            Some([]) => Vec::new(),
            _ => {
                let mut definitions = BTreeMap::new();
                for provider in self.set.values() {
                    for mut definition in provider.definitions(names) {
                        definition.name.make_ascii_lowercase();
                        definitions
                            .entry(definition.name.clone())
                            .or_insert(definition);
                    }
                }
                definitions.into_values().collect()
            }
        }
    }

    /// Returns the capability groups exposed by every registered provider.
    pub fn groups(&self) -> Vec<ToolGroup> {
        self.set
            .values()
            .flat_map(|provider| provider.groups())
            .collect()
    }

    /// Returns function metadata for all provider-backed tools or selected names.
    pub fn functions(&self, names: Option<&[String]>) -> Vec<Function> {
        self.definitions(names)
            .into_iter()
            .map(|definition| {
                let supported_resource_tags = self
                    .set
                    .values()
                    .find(|provider| provider.contains_lowercase(&definition.name))
                    .map(|provider| provider.supported_resource_tags(&definition.name))
                    .unwrap_or_default();
                Function {
                    definition,
                    supported_resource_tags,
                }
            })
            .collect()
    }

    /// Removes and returns resources supported by the named provider tool.
    pub fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        let lowercase_name = name.to_ascii_lowercase();
        self.set
            .values()
            .find(|provider| provider.contains_lowercase(&lowercase_name))
            .map(|provider| provider.select_resources(&lowercase_name, resources))
            .unwrap_or_default()
    }

    /// Initializes all providers.
    pub async fn init_all(&self, ctx: C) -> Result<(), BoxError> {
        for provider in self.set.values() {
            provider.init(ctx.clone()).await?;
        }
        Ok(())
    }

    /// Refreshes all providers.
    pub async fn refresh_all(&self) -> Result<(), BoxError> {
        for provider in self.set.values() {
            provider.refresh().await?;
        }
        Ok(())
    }

    /// Executes a dynamic provider-backed tool.
    pub async fn call(
        &self,
        ctx: C,
        mut input: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        input.name.make_ascii_lowercase();
        let provider = self
            .set
            .values()
            .find(|provider| provider.contains_lowercase(&input.name))
            .ok_or_else(|| format!("tool {} not found", input.name))?;
        provider.call(ctx, input).await
    }
}

impl<C> ToolSet<C>
where
    C: BaseContext + Send + Sync + 'static,
{
    /// Creates an empty tool set.
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Returns whether a tool with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(&name.to_ascii_lowercase())
    }

    /// Returns whether a tool with the given lowercase name exists.
    pub fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.set.contains_key(lowercase_name)
    }

    /// Returns the names of all registered tools.
    pub fn names(&self) -> Vec<String> {
        self.set.keys().cloned().collect()
    }

    /// Returns the capability groups assembled from registered tools.
    ///
    /// Tools that declare the same [`ToolGroupInfo::id`] are collected into one
    /// [`ToolGroup`] whose `members` are exactly the registered tool names in
    /// that group, sorted for determinism. Group metadata is taken from the
    /// first tool (by lowercase name order) that declares the id.
    pub fn groups(&self) -> Vec<ToolGroup> {
        let mut grouped: BTreeMap<String, (ToolGroupInfo, Vec<String>)> = BTreeMap::new();
        for (name, tool) in &self.set {
            if let Some(info) = tool.group() {
                grouped
                    .entry(info.id.clone())
                    .or_insert_with(|| (info, Vec::new()))
                    .1
                    .push(name.clone());
            }
        }

        grouped
            .into_values()
            .map(|(info, mut members)| {
                members.sort();
                ToolGroup::from_info(info, members)
            })
            .collect()
    }

    /// Returns the function definition for a specific tool.
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|tool| tool.definition())
    }

    /// Returns function definitions for all tools or the selected names.
    ///
    /// # Arguments
    /// - `names`: Optional slice of tool names to filter by.
    ///
    /// # Returns
    /// A vector of tool definitions.
    pub fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        match names {
            None => self.set.values().map(|tool| tool.definition()).collect(),
            Some(names) => {
                // Deduplicate by lowercase name so repeated requested names do
                // not emit duplicate schemas (some providers reject those).
                let mut seen = BTreeSet::new();
                names
                    .iter()
                    .filter_map(|name| {
                        let key = name.to_ascii_lowercase();
                        self.set
                            .get(&key)
                            .and_then(|tool| seen.insert(key).then(|| tool.definition()))
                    })
                    .collect()
            }
        }
    }

    /// Returns function metadata for all tools or the selected names.
    ///
    /// # Arguments
    /// - `names`: Optional slice of tool names to filter by.
    ///
    /// # Returns
    /// A vector of tool function metadata.
    pub fn functions(&self, names: Option<&[String]>) -> Vec<Function> {
        match names {
            None => self
                .set
                .values()
                .map(|tool| Function {
                    definition: tool.definition(),
                    supported_resource_tags: tool.supported_resource_tags(),
                })
                .collect(),
            Some(names) => {
                // Deduplicate by lowercase name (see `definitions`).
                let mut seen = BTreeSet::new();
                names
                    .iter()
                    .filter_map(|name| {
                        let key = name.to_ascii_lowercase();
                        self.set.get(&key).and_then(|tool| {
                            seen.insert(key).then(|| Function {
                                definition: tool.definition(),
                                supported_resource_tags: tool.supported_resource_tags(),
                            })
                        })
                    })
                    .collect()
            }
        }
    }

    /// Removes and returns resources supported by the named tool.
    pub fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        self.set
            .get(&name.to_ascii_lowercase())
            .map(|tool| {
                let supported_tags = tool.supported_resource_tags();
                select_resources(resources, &supported_tags)
            })
            .unwrap_or_default()
    }

    /// Registers a new tool.
    ///
    /// # Arguments
    /// - `tool`: The tool to register.
    pub fn add<T>(&mut self, tool: Arc<T>) -> Result<(), BoxError>
    where
        T: Tool<C> + Send + Sync + 'static,
    {
        let name = tool.name().to_ascii_lowercase();
        validate_function_name(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("tool {} already exists", name).into());
        }

        let tool_dyn = ToolWrapper(tool, PhantomData);
        self.set.insert(name, Arc::new(tool_dyn));
        Ok(())
    }

    /// Returns a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn DynTool<C>>> {
        self.set.get(&name.to_ascii_lowercase()).cloned()
    }

    /// Returns a tool by lowercase name.
    pub fn get_lowercase(&self, lowercase_name: &str) -> Option<Arc<dyn DynTool<C>>> {
        self.set.get(lowercase_name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candid::Principal;
    use serde_json::json;
    use std::{sync::Arc, time::Duration};

    use crate::{
        BaseContext, CacheExpiry, CacheFeatures, CancellationToken, HttpFeatures, KeysFeatures,
        ObjectMeta, Path, PutMode, PutResult, RequestMeta, StateFeatures, StoreFeatures, ToolInput,
    };

    #[derive(Clone)]
    struct TestContext {
        engine_id: Principal,
        caller: Principal,
        meta: RequestMeta,
        cancellation_token: CancellationToken,
    }

    impl Default for TestContext {
        fn default() -> Self {
            Self {
                engine_id: Principal::management_canister(),
                caller: Principal::anonymous(),
                meta: RequestMeta::default(),
                cancellation_token: CancellationToken::new(),
            }
        }
    }

    impl StateFeatures for TestContext {
        fn engine_id(&self) -> &Principal {
            &self.engine_id
        }

        fn engine_name(&self) -> &str {
            "test-engine"
        }

        fn caller(&self) -> &Principal {
            &self.caller
        }

        fn meta(&self) -> &RequestMeta {
            &self.meta
        }

        fn cancellation_token(&self) -> CancellationToken {
            self.cancellation_token.clone()
        }

        fn time_elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }

    impl KeysFeatures for TestContext {
        async fn a256gcm_key(&self, _derivation_path: Vec<Vec<u8>>) -> Result<[u8; 32], BoxError> {
            Ok([0; 32])
        }

        async fn ed25519_sign_message(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
        ) -> Result<[u8; 64], BoxError> {
            Ok([0; 64])
        }

        async fn ed25519_verify(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
            _signature: &[u8],
        ) -> Result<(), BoxError> {
            Ok(())
        }

        async fn ed25519_public_key(
            &self,
            _derivation_path: Vec<Vec<u8>>,
        ) -> Result<[u8; 32], BoxError> {
            Ok([0; 32])
        }

        async fn secp256k1_sign_message_bip340(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
        ) -> Result<[u8; 64], BoxError> {
            Ok([0; 64])
        }

        async fn secp256k1_verify_bip340(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
            _signature: &[u8],
        ) -> Result<(), BoxError> {
            Ok(())
        }

        async fn secp256k1_sign_message_ecdsa(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
        ) -> Result<[u8; 64], BoxError> {
            Ok([0; 64])
        }

        async fn secp256k1_sign_digest_ecdsa(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message_hash: &[u8],
        ) -> Result<[u8; 64], BoxError> {
            Ok([0; 64])
        }

        async fn secp256k1_verify_ecdsa(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message_hash: &[u8],
            _signature: &[u8],
        ) -> Result<(), BoxError> {
            Ok(())
        }

        async fn secp256k1_public_key(
            &self,
            _derivation_path: Vec<Vec<u8>>,
        ) -> Result<[u8; 33], BoxError> {
            Ok([0; 33])
        }
    }

    impl StoreFeatures for TestContext {
        async fn store_get(&self, _path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
            Err("not implemented".into())
        }

        async fn store_list(
            &self,
            _prefix: Option<&Path>,
            _offset: &Path,
        ) -> Result<Vec<ObjectMeta>, BoxError> {
            Ok(Vec::new())
        }

        async fn store_put(
            &self,
            _path: &Path,
            _mode: PutMode,
            _value: bytes::Bytes,
        ) -> Result<PutResult, BoxError> {
            Err("not implemented".into())
        }

        async fn store_rename_if_not_exists(
            &self,
            _from: &Path,
            _to: &Path,
        ) -> Result<(), BoxError> {
            Err("not implemented".into())
        }

        async fn store_delete(&self, _path: &Path) -> Result<(), BoxError> {
            Ok(())
        }
    }

    impl CacheFeatures for TestContext {
        fn cache_contains(&self, _key: &str) -> bool {
            false
        }

        async fn cache_get<T>(&self, _key: &str) -> Result<T, BoxError>
        where
            T: DeserializeOwned,
        {
            Err("not implemented".into())
        }

        async fn cache_get_with<T, F>(&self, _key: &str, _init: F) -> Result<T, BoxError>
        where
            T: Sized + DeserializeOwned + Serialize + Send,
            F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
        {
            Err("not implemented".into())
        }

        async fn cache_set<T>(&self, _key: &str, _val: (T, Option<CacheExpiry>))
        where
            T: Sized + Serialize + Send,
        {
        }

        async fn cache_set_if_not_exists<T>(
            &self,
            _key: &str,
            _val: (T, Option<CacheExpiry>),
        ) -> bool
        where
            T: Sized + Serialize + Send,
        {
            false
        }

        async fn cache_delete(&self, _key: &str) -> bool {
            false
        }

        fn cache_raw_iter(
            &self,
        ) -> impl Iterator<Item = (Arc<String>, Arc<(bytes::Bytes, Option<CacheExpiry>)>)> {
            std::iter::empty()
        }
    }

    impl HttpFeatures for TestContext {
        async fn https_call(
            &self,
            _url: &str,
            _method: http::Method,
            _headers: Option<http::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            Err("not implemented".into())
        }

        async fn https_signed_call(
            &self,
            _url: &str,
            _method: http::Method,
            _message_digest: [u8; 32],
            _headers: Option<http::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            Err("not implemented".into())
        }

        async fn https_signed_rpc<T>(
            &self,
            _endpoint: &str,
            _method: &str,
            _args: impl Serialize + Send,
        ) -> Result<T, BoxError>
        where
            T: DeserializeOwned,
        {
            Err("not implemented".into())
        }
    }

    impl BaseContext for TestContext {
        async fn remote_tool_call(
            &self,
            _endpoint: &str,
            _args: ToolInput<Json>,
        ) -> Result<ToolOutput<Json>, BoxError> {
            Err("not implemented".into())
        }
    }

    struct ExampleTool {
        id: usize,
    }

    struct OtherTool;

    #[derive(serde::Deserialize)]
    struct EchoArgs {
        value: String,
        fail: bool,
    }

    struct TaggedTool;

    struct InvalidTool;

    fn resource(id: u64, tags: &[&str]) -> Resource {
        Resource {
            _id: id,
            name: format!("resource-{id}"),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        }
    }

    impl Tool<TestContext> for ExampleTool {
        type Args = ();
        type Output = String;

        fn name(&self) -> String {
            "example_tool".to_string()
        }

        fn description(&self) -> String {
            "Example tool used for downcast tests".to_string()
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
            _ctx: TestContext,
            _args: Self::Args,
            _resources: Vec<Resource>,
        ) -> Result<ToolOutput<Self::Output>, BoxError> {
            Ok(ToolOutput::new(self.id.to_string()))
        }
    }

    impl Tool<TestContext> for OtherTool {
        type Args = ();
        type Output = String;

        fn name(&self) -> String {
            "other_tool".to_string()
        }

        fn description(&self) -> String {
            "Other tool used for downcast tests".to_string()
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
            _ctx: TestContext,
            _args: Self::Args,
            _resources: Vec<Resource>,
        ) -> Result<ToolOutput<Self::Output>, BoxError> {
            Ok(ToolOutput::new("other".to_string()))
        }
    }

    impl Tool<TestContext> for TaggedTool {
        type Args = EchoArgs;
        type Output = Json;

        fn name(&self) -> String {
            "tagged_tool".to_string()
        }

        fn description(&self) -> String {
            "Tool that consumes text and code resources".to_string()
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: self.name(),
                description: self.description(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "fail": {"type": "boolean"}
                    },
                    "required": ["value", "fail"],
                    "additionalProperties": false
                }),
                strict: Some(true),
            }
        }

        fn supported_resource_tags(&self) -> Vec<String> {
            vec!["text".to_string(), "code".to_string()]
        }

        async fn call(
            &self,
            _ctx: TestContext,
            args: Self::Args,
            resources: Vec<Resource>,
        ) -> Result<ToolOutput<Self::Output>, BoxError> {
            if args.fail {
                return Err("forced failure".into());
            }

            let mut output = ToolOutput::new(json!({
                "value": args.value,
                "resources": resources.len(),
            }));
            output.is_error = Some(false);
            Ok(output)
        }
    }

    impl Tool<TestContext> for InvalidTool {
        type Args = ();
        type Output = String;

        fn name(&self) -> String {
            "bad.tool".to_string()
        }

        fn description(&self) -> String {
            "Invalid function name".to_string()
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: self.name(),
                description: self.description(),
                parameters: json!({"type": "object"}),
                strict: Some(true),
            }
        }

        async fn call(
            &self,
            _ctx: TestContext,
            _args: Self::Args,
            _resources: Vec<Resource>,
        ) -> Result<ToolOutput<Self::Output>, BoxError> {
            Ok(ToolOutput::new(String::new()))
        }
    }

    #[test]
    fn dyn_tool_downcast_ref_returns_inner_tool() {
        let tool = Arc::new(ExampleTool { id: 7 });
        let mut tool_set = ToolSet::<TestContext>::new();
        tool_set.add(tool).unwrap();

        let dyn_tool = tool_set.get("example_tool").unwrap();
        let concrete = dyn_tool.downcast_ref::<ExampleTool>().unwrap();

        assert_eq!(concrete.id, 7);
        assert!(dyn_tool.downcast_ref::<OtherTool>().is_none());
    }

    #[test]
    fn dyn_tool_downcast_returns_original_arc() {
        let tool = Arc::new(ExampleTool { id: 9 });
        let mut tool_set = ToolSet::<TestContext>::new();
        tool_set.add(tool.clone()).unwrap();

        let dyn_tool = tool_set.get("example_tool").unwrap();
        let concrete = match dyn_tool.downcast::<ExampleTool>() {
            Ok(tool) => tool,
            Err(_) => panic!("expected downcast to ExampleTool to succeed"),
        };

        assert_eq!(concrete.id, 9);
        assert!(Arc::ptr_eq(&concrete, &tool));
    }

    #[test]
    fn dyn_tool_downcast_mismatch_returns_original_arc() {
        let tool = Arc::new(ExampleTool { id: 11 });
        let mut tool_set = ToolSet::<TestContext>::new();
        tool_set.add(tool).unwrap();

        let dyn_tool = tool_set.get("example_tool").unwrap();
        let original = dyn_tool.clone();
        let err = match dyn_tool.downcast::<OtherTool>() {
            Ok(_) => panic!("expected downcast to OtherTool to fail"),
            Err(err) => err,
        };

        assert!(Arc::ptr_eq(&err, &original));
        assert_eq!(err.name(), "example_tool");
    }

    #[test]
    fn fixture_tools_cover_direct_methods() {
        futures::executor::block_on(async {
            let other = OtherTool;
            assert_eq!(other.name(), "other_tool");
            assert_eq!(other.description(), "Other tool used for downcast tests");
            let definition = other.definition();
            assert_eq!(definition.name, "other_tool");
            assert_eq!(definition.description, "Other tool used for downcast tests");
            assert_eq!(definition.parameters["type"], "object");
            let output = other
                .call(TestContext::default(), (), Vec::new())
                .await
                .unwrap();
            assert_eq!(output.output, "other");

            let invalid = InvalidTool;
            assert_eq!(invalid.name(), "bad.tool");
            assert_eq!(invalid.description(), "Invalid function name");
            let definition = invalid.definition();
            assert_eq!(definition.name, "bad.tool");
            assert_eq!(definition.description, "Invalid function name");
            assert_eq!(definition.parameters["type"], "object");
            let output = invalid
                .call(TestContext::default(), (), Vec::new())
                .await
                .unwrap();
            assert!(output.output.is_empty());
        });
    }

    #[test]
    fn tool_default_methods_call_raw_and_dyn_wrapper_forward_calls() {
        futures::executor::block_on(async {
            let tool = Arc::new(ExampleTool { id: 42 });
            let mut resources = vec![resource(1, &["text"])];

            assert!(tool.supported_resource_tags().is_empty());
            assert!(tool.select_resources(&mut resources).is_empty());
            assert_eq!(resources.len(), 1);
            tool.init(TestContext::default()).await.unwrap();

            let raw = tool
                .call_raw(TestContext::default(), Json::Null, Vec::new())
                .await
                .unwrap();
            assert_eq!(raw.output, json!("42"));
            assert_eq!(raw.usage.requests, 1);

            let invalid = tool
                .call_raw(TestContext::default(), json!({"bad": true}), Vec::new())
                .await
                .unwrap_err();
            assert!(invalid.to_string().contains("invalid args"));

            let mut tool_set = ToolSet::<TestContext>::new();
            tool_set.add(tool).unwrap();
            let dyn_tool = tool_set.get("EXAMPLE_TOOL").unwrap();

            assert_eq!(dyn_tool.name(), "example_tool");
            assert_eq!(dyn_tool.definition().name, "example_tool");
            assert!(dyn_tool.supported_resource_tags().is_empty());
            dyn_tool.init(TestContext::default()).await.unwrap();

            let output = dyn_tool
                .call(TestContext::default(), Json::Null, Vec::new())
                .await
                .unwrap();
            assert_eq!(output.output, json!("42"));
            assert_eq!(output.usage.requests, 1);
        });
    }

    #[test]
    fn tool_set_registry_filters_resources_and_reports_errors() {
        futures::executor::block_on(async {
            let mut tool_set = ToolSet::<TestContext>::new();
            tool_set.add(Arc::new(ExampleTool { id: 1 })).unwrap();
            tool_set.add(Arc::new(TaggedTool)).unwrap();

            assert!(tool_set.contains("EXAMPLE_TOOL"));
            assert!(tool_set.contains_lowercase("tagged_tool"));
            assert!(!tool_set.contains("missing_tool"));
            assert_eq!(
                tool_set.names(),
                vec!["example_tool".to_string(), "tagged_tool".to_string()]
            );

            let definition = tool_set.definition("TAGGED_TOOL").unwrap();
            assert_eq!(definition.name, "tagged_tool");
            assert!(tool_set.definition("missing_tool").is_none());

            let selected_names = vec!["TAGGED_TOOL".to_string(), "missing_tool".to_string()];
            let selected_definitions = tool_set.definitions(Some(&selected_names));
            assert_eq!(selected_definitions.len(), 1);
            assert_eq!(selected_definitions[0].name, "tagged_tool");
            assert_eq!(tool_set.definitions(None).len(), 2);

            // Repeated (case-insensitive) requested names are deduplicated.
            let duplicate_names = vec![
                "tagged_tool".to_string(),
                "TAGGED_TOOL".to_string(),
                "tagged_tool".to_string(),
            ];
            assert_eq!(tool_set.definitions(Some(&duplicate_names)).len(), 1);
            assert_eq!(tool_set.functions(Some(&duplicate_names)).len(), 1);

            let selected_functions = tool_set.functions(Some(&selected_names));
            assert_eq!(selected_functions.len(), 1);
            assert_eq!(
                selected_functions[0].supported_resource_tags,
                vec!["text".to_string(), "code".to_string()]
            );
            assert_eq!(tool_set.functions(None).len(), 2);

            let mut resources = vec![
                resource(1, &["image"]),
                resource(2, &["text"]),
                resource(3, &["code", "text"]),
                resource(4, &["audio"]),
            ];
            let selected = tool_set.select_resources("TAGGED_TOOL", &mut resources);
            assert_eq!(
                selected
                    .iter()
                    .map(|resource| resource._id)
                    .collect::<Vec<_>>(),
                vec![2, 3]
            );
            assert_eq!(
                resources
                    .iter()
                    .map(|resource| resource._id)
                    .collect::<Vec<_>>(),
                vec![1, 4]
            );
            assert!(
                tool_set
                    .select_resources("missing_tool", &mut resources)
                    .is_empty()
            );

            let dyn_tool = tool_set.get_lowercase("tagged_tool").unwrap();
            let output = dyn_tool
                .call(
                    TestContext::default(),
                    json!({"value": "ok", "fail": false}),
                    vec![resource(9, &["text"])],
                )
                .await
                .unwrap();
            assert_eq!(output.output["value"], "ok");
            assert_eq!(output.output["resources"], 1);
            assert_eq!(output.is_error, Some(false));
            assert_eq!(output.usage.requests, 1);
            assert!(tool_set.get("missing_tool").is_none());
            assert!(tool_set.get_lowercase("missing_tool").is_none());

            let failed = dyn_tool
                .call(
                    TestContext::default(),
                    json!({"value": "bad", "fail": true}),
                    Vec::new(),
                )
                .await
                .unwrap_err();
            assert!(failed.to_string().contains("call failed"));

            let duplicate = tool_set.add(Arc::new(ExampleTool { id: 2 })).unwrap_err();
            assert!(duplicate.to_string().contains("already exists"));

            let invalid = tool_set.add(Arc::new(InvalidTool)).unwrap_err();
            assert!(invalid.to_string().contains("invalid character"));
        });
    }

    #[test]
    fn test_tool_context_mock_features_cover_default_paths() {
        futures::executor::block_on(async {
            let ctx = TestContext::default();
            assert_eq!(*ctx.engine_id(), Principal::management_canister());
            assert_eq!(ctx.engine_name(), "test-engine");
            assert_eq!(*ctx.caller(), Principal::anonymous());
            assert!(ctx.meta().user.is_none());
            assert!(!ctx.cancellation_token().is_cancelled());
            assert_eq!(ctx.time_elapsed(), Duration::ZERO);

            assert_eq!(ctx.a256gcm_key(Vec::new()).await.unwrap(), [0; 32]);
            assert_eq!(
                ctx.ed25519_sign_message(Vec::new(), b"message")
                    .await
                    .unwrap(),
                [0; 64]
            );
            ctx.ed25519_verify(Vec::new(), b"message", &[0; 64])
                .await
                .unwrap();
            assert_eq!(ctx.ed25519_public_key(Vec::new()).await.unwrap(), [0; 32]);
            assert_eq!(
                ctx.secp256k1_sign_message_bip340(Vec::new(), b"message")
                    .await
                    .unwrap(),
                [0; 64]
            );
            ctx.secp256k1_verify_bip340(Vec::new(), b"message", &[0; 64])
                .await
                .unwrap();
            assert_eq!(
                ctx.secp256k1_sign_message_ecdsa(Vec::new(), b"message")
                    .await
                    .unwrap(),
                [0; 64]
            );
            assert_eq!(
                ctx.secp256k1_sign_digest_ecdsa(Vec::new(), &[0; 32])
                    .await
                    .unwrap(),
                [0; 64]
            );
            ctx.secp256k1_verify_ecdsa(Vec::new(), &[0; 32], &[0; 64])
                .await
                .unwrap();
            assert_eq!(ctx.secp256k1_public_key(Vec::new()).await.unwrap(), [0; 33]);

            assert!(ctx.store_get(&Path::from("missing")).await.is_err());
            assert!(
                ctx.store_list(None, &Path::default())
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                ctx.store_put(&Path::from("file"), PutMode::Overwrite, bytes::Bytes::new())
                    .await
                    .is_err()
            );
            assert!(
                ctx.store_rename_if_not_exists(&Path::from("a"), &Path::from("b"))
                    .await
                    .is_err()
            );
            ctx.store_delete(&Path::from("file")).await.unwrap();

            assert!(!ctx.cache_contains("key"));
            assert!(ctx.cache_get::<String>("key").await.is_err());
            assert!(
                ctx.cache_get_with("key", async { Ok(("value".to_string(), None)) })
                    .await
                    .is_err()
            );
            ctx.cache_set("key", ("value".to_string(), None)).await;
            assert!(
                !ctx.cache_set_if_not_exists("key", ("value".to_string(), None))
                    .await
            );
            assert!(!ctx.cache_delete("key").await);
            assert_eq!(ctx.cache_raw_iter().count(), 0);

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
                    None,
                )
                .await
                .is_err()
            );
            let rpc: Result<String, BoxError> = ctx
                .https_signed_rpc("https://example.test", "method", &())
                .await;
            assert!(rpc.is_err());

            assert!(
                ctx.remote_tool_call(
                    "https://example.test",
                    ToolInput::new("tool".to_string(), Json::Null),
                )
                .await
                .is_err()
            );
        });
    }

    struct GroupedTool {
        name: &'static str,
        group: &'static str,
    }

    impl Tool<TestContext> for GroupedTool {
        type Args = ();
        type Output = String;

        fn name(&self) -> String {
            self.name.to_string()
        }

        fn description(&self) -> String {
            "Grouped tool fixture".to_string()
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

        fn group(&self) -> Option<ToolGroupInfo> {
            Some(ToolGroupInfo {
                id: self.group.to_string(),
                title: format!("{} title", self.group),
                description: format!("{} description", self.group),
                instructions: Some(format!("{} instructions", self.group)),
            })
        }

        async fn call(
            &self,
            _ctx: TestContext,
            _args: Self::Args,
            _resources: Vec<Resource>,
        ) -> Result<ToolOutput<Self::Output>, BoxError> {
            Ok(ToolOutput::new(String::new()))
        }
    }

    #[test]
    fn tool_set_groups_aggregate_members_by_id() {
        let mut tool_set = ToolSet::<TestContext>::new();
        tool_set
            .add(Arc::new(GroupedTool {
                name: "fs_write",
                group: "fs",
            }))
            .unwrap();
        tool_set
            .add(Arc::new(GroupedTool {
                name: "fs_read",
                group: "fs",
            }))
            .unwrap();
        tool_set
            .add(Arc::new(GroupedTool {
                name: "mem_get",
                group: "memory",
            }))
            .unwrap();
        // A tool with no group declaration is excluded from every group.
        tool_set.add(Arc::new(ExampleTool { id: 1 })).unwrap();

        let groups = tool_set.groups();
        assert_eq!(groups.len(), 2);

        let fs = groups.iter().find(|group| group.id == "fs").unwrap();
        // Members reflect the registered tools, sorted for determinism.
        assert_eq!(
            fs.members,
            vec!["fs_read".to_string(), "fs_write".to_string()]
        );
        assert_eq!(fs.title, "fs title");
        assert_eq!(fs.instructions.as_deref(), Some("fs instructions"));

        let memory = groups.iter().find(|group| group.id == "memory").unwrap();
        assert_eq!(memory.members, vec!["mem_get".to_string()]);
    }
}
