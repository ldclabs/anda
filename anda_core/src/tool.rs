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
        BaseContext, CacheExpiry, CacheFeatures, CancellationToken, CanisterCaller, HttpFeatures,
        KeysFeatures, ObjectMeta, Path, PutMode, PutResult, RequestMeta, StateFeatures,
        StoreFeatures, ToolInput,
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
            "bad-tool".to_string()
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
            assert_eq!(invalid.name(), "bad-tool");
            assert_eq!(invalid.description(), "Invalid function name");
            let definition = invalid.definition();
            assert_eq!(definition.name, "bad-tool");
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

            let query: Result<String, BoxError> = ctx
                .canister_query(&Principal::anonymous(), "query", ())
                .await;
            assert!(query.is_err());
            let update: Result<String, BoxError> = ctx
                .canister_update(&Principal::anonymous(), "update", ())
                .await;
            assert!(update.is_err());

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
}
