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

use serde::{Serialize, de::DeserializeOwned};
use std::{any::Any, collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{
    BoxError, BoxPinFut, Function, Json, Resource, ToolOutput, context::BaseContext,
    model::FunctionDefinition, select_resources, validate_function_name,
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
    /// - Must not exceed 64 characters;
    /// - Must start with a lowercase letter;
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_);
    /// - Unique within the engine.
    fn name(&self) -> String;

    /// Returns a concise description of the tool's capability.
    fn description(&self) -> String;

    /// Returns the function definition, including the JSON parameter schema.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The schema definition of the tool's parameters and metadata.
    fn definition(&self) -> FunctionDefinition;

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
        futures::future::ready(Ok(()))
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
    fn as_any(&self) -> &(dyn Any + Send + Sync);

    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    fn name(&self) -> String;

    fn definition(&self) -> FunctionDefinition;

    fn supported_resource_tags(&self) -> Vec<String>;

    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>>;

    fn call(
        &self,
        ctx: C,
        args: Json,
        resources: Vec<Resource>,
    ) -> BoxPinFut<Result<ToolOutput<Json>, BoxError>>;
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
    pub set: BTreeMap<String, Arc<dyn DynTool<C>>>,
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
            Some(names) => names
                .iter()
                .filter_map(|name| {
                    self.set
                        .get(&name.to_ascii_lowercase())
                        .map(|tool| tool.definition())
                })
                .collect(),
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
            Some(names) => names
                .iter()
                .filter_map(|name| {
                    self.set
                        .get(&name.to_ascii_lowercase())
                        .map(|tool| Function {
                            definition: tool.definition(),
                            supported_resource_tags: tool.supported_resource_tags(),
                        })
                })
                .collect(),
        }
    }

    /// Removes and returns resources supported by the named tool.
    pub fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
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
    use candid::{CandidType, Principal, utils::ArgumentEncoder};
    use serde_json::json;
    use std::{sync::Arc, time::Duration};

    use crate::{
        CacheExpiry, CacheFeatures, CancellationToken, HttpFeatures, KeysFeatures, ObjectMeta,
        Path, PutMode, PutResult, RequestMeta, StateFeatures, StoreFeatures, ToolInput,
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

    impl crate::CanisterCaller for TestContext {
        async fn canister_query<In, Out>(
            &self,
            _canister: &Principal,
            _method: &str,
            _args: In,
        ) -> Result<Out, BoxError>
        where
            In: ArgumentEncoder + Send,
            Out: CandidType + for<'a> candid::Deserialize<'a>,
        {
            Err("not implemented".into())
        }

        async fn canister_update<In, Out>(
            &self,
            _canister: &Principal,
            _method: &str,
            _args: In,
        ) -> Result<Out, BoxError>
        where
            In: ArgumentEncoder + Send,
            Out: CandidType + for<'a> candid::Deserialize<'a>,
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
}
