//! Agent traits and registries.
//!
//! This module defines how custom AI agents are described, registered, and
//! invoked by an Anda runtime. It provides:
//! - [`Agent`] for strongly typed agent implementations.
//! - [`DynAgent`] for runtime dispatch through trait objects.
//! - [`AgentSet`] for name-based registration and lookup.
//!
//! Agents may declare tool dependencies and supported resource tags. The
//! runtime uses those declarations to validate engine configuration and route
//! resource attachments to the components that can consume them.
//!
//! See the `anda_engine` extension modules for concrete agent implementations.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{any::Any, collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{
    BoxError, BoxPinFut, Function,
    context::AgentContext,
    model::{AgentOutput, FunctionDefinition, Resource},
    select_resources, validate_function_name,
};

/// Default JSON arguments for an agent exposed as a callable function.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentArgs {
    /// Self-contained task prompt for the agent.
    pub prompt: String,
}

/// Strongly typed interface for an AI agent.
///
/// # Type Parameters
/// - `C`: Runtime context implementing [`AgentContext`].
pub trait Agent<C>: Send + Sync
where
    C: AgentContext + Send + Sync,
{
    /// Returns the unique agent name.
    ///
    /// Names are registered case-insensitively and stored in lowercase.
    ///
    /// # Rules
    /// - Must not be empty;
    /// - Must not exceed 64 characters;
    /// - Must start with a lowercase letter;
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_);
    /// - Unique within the engine in lowercase.
    fn name(&self) -> String;

    /// Returns a concise description of the agent's capability.
    fn description(&self) -> String;

    /// Returns the function definition used for LLM/tool-call integration.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The structured definition of the agent's capabilities.
    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name().to_ascii_lowercase(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "description": "Run this agent on a focused task. Provide a self-contained prompt with the goal, relevant context, constraints, and expected output.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task for this agent. Include the objective, relevant context, constraints, preferred workflow or deliverable, and any success criteria needed to complete the work.",
                        "minLength": 1
                    },
                },
                "required": ["prompt"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    /// Returns resource tags this agent can consume.
    ///
    /// The default implementation returns an empty list, meaning no resources
    /// are selected for this agent. Return `vec!["*".into()]` to accept all
    /// attached resources.
    ///
    /// # Returns
    /// Resource tags supported by this agent.
    fn supported_resource_tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Removes and returns resources matching this agent's supported tags.
    fn select_resources(&self, resources: &mut Vec<Resource>) -> Vec<Resource> {
        let supported_tags = self.supported_resource_tags();
        select_resources(resources, &supported_tags)
    }

    /// Initializes the agent with the given context.
    ///
    /// Runtimes call this once while building the engine.
    fn init(&self, _ctx: C) -> impl Future<Output = Result<(), BoxError>> + Send {
        futures::future::ready(Ok(()))
    }

    /// Returns tool names required by this agent.
    ///
    /// Runtimes use this list to validate that required tools are registered.
    fn tool_dependencies(&self) -> Vec<String> {
        Vec::new()
    }

    /// Executes the agent with the given context and inputs.
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing [`AgentContext`].
    /// - `prompt`: The input prompt or message for the agent.
    /// - `resources`: Additional resources selected for this agent. Ignore resources that are not useful.
    ///
    /// # Returns
    /// A future resolving to [`AgentOutput`].
    fn run(
        &self,
        ctx: C,
        prompt: String,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Object-safe wrapper around [`Agent`] for runtime dispatch.
///
/// Runtime registries store agents through this trait so callers can select and
/// execute agents by name without knowing their concrete Rust types.
pub trait DynAgent<C>: Send + Sync
where
    C: AgentContext + Send + Sync,
{
    /// Returns this agent as [`Any`] for type inspection.
    fn as_any(&self) -> &(dyn Any + Send + Sync);

    /// Converts the shared agent into [`Any`] for downcasting.
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the registry label used for name-based lookup.
    fn label(&self) -> &str;

    /// Returns the unique agent name.
    fn name(&self) -> String;

    /// Returns the function definition exposed to model providers.
    fn definition(&self) -> FunctionDefinition;

    /// Returns tool names required by this agent.
    fn tool_dependencies(&self) -> Vec<String>;

    /// Returns resource tags this agent can consume.
    fn supported_resource_tags(&self) -> Vec<String>;

    /// Initializes the agent through object-safe dispatch.
    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>>;

    /// Executes the agent through object-safe dispatch.
    fn run(
        &self,
        ctx: C,
        prompt: String,
        resources: Vec<Resource>,
    ) -> BoxPinFut<Result<AgentOutput, BoxError>>;
}

impl<C> dyn DynAgent<C>
where
    C: AgentContext + Send + Sync + 'static,
{
    /// Returns the inner concrete agent type when it matches `T`.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: Agent<C> + 'static,
    {
        self.as_any().downcast_ref::<T>()
    }

    /// Returns the inner concrete agent when it matches `T`.
    pub fn downcast<T>(self: Arc<Self>) -> Result<Arc<T>, Arc<Self>>
    where
        T: Agent<C> + 'static,
    {
        match self.clone().into_any().downcast::<T>() {
            Ok(agent) => Ok(agent),
            Err(_) => Err(self),
        }
    }
}

/// Adapter that exposes a concrete [`Agent`] through [`DynAgent`].
struct AgentWrapper<T, C>
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static,
{
    inner: Arc<T>,
    label: String,
    _phantom: PhantomData<C>,
}

impl<T, C> DynAgent<C> for AgentWrapper<T, C>
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static,
{
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self.inner.as_ref()
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self.inner.clone()
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    fn definition(&self) -> FunctionDefinition {
        self.inner.definition()
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.inner.tool_dependencies()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.inner.supported_resource_tags()
    }

    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>> {
        let agent = self.inner.clone();
        Box::pin(async move { agent.init(ctx).await })
    }

    fn run(
        &self,
        ctx: C,
        prompt: String,
        resources: Vec<Resource>,
    ) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let agent = self.inner.clone();
        Box::pin(async move { agent.run(ctx, prompt, resources).await })
    }
}

/// Name-based registry for agents.
///
/// # Type Parameters
/// - `C`: The context type that implements [`AgentContext`].
#[derive(Default)]
pub struct AgentSet<C: AgentContext> {
    /// Registered agents keyed by their lowercase function names.
    pub set: BTreeMap<String, Arc<dyn DynAgent<C>>>,
}

impl<C> AgentSet<C>
where
    C: AgentContext + Send + Sync + 'static,
{
    /// Creates a new empty AgentSet.
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Returns whether an agent with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(&name.to_ascii_lowercase())
    }

    /// Returns whether an agent with the given lowercase name exists.
    pub fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.set.contains_key(lowercase_name)
    }

    /// Returns the names of all agents in the set.
    pub fn names(&self) -> Vec<String> {
        self.set.keys().cloned().collect()
    }

    /// Returns the function definition for a specific agent.
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|agent| agent.definition())
    }

    /// Returns function definitions for all agents or the selected names.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// A vector of agent definitions.
    pub fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        match names {
            None => self.set.values().map(|agent| agent.definition()).collect(),
            Some(names) => names
                .iter()
                .filter_map(|name| {
                    self.set
                        .get(&name.to_ascii_lowercase())
                        .map(|agent| agent.definition())
                })
                .collect(),
        }
    }

    /// Returns function metadata for all agents or the selected names.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// A vector of agent function metadata.
    pub fn functions(&self, names: Option<&[String]>) -> Vec<Function> {
        match names {
            None => self
                .set
                .values()
                .map(|agent| Function {
                    definition: agent.definition(),
                    supported_resource_tags: agent.supported_resource_tags(),
                })
                .collect(),
            Some(names) => names
                .iter()
                .filter_map(|name| {
                    self.set
                        .get(&name.to_ascii_lowercase())
                        .map(|agent| Function {
                            definition: agent.definition(),
                            supported_resource_tags: agent.supported_resource_tags(),
                        })
                })
                .collect(),
        }
    }

    /// Removes and returns resources supported by the named agent.
    pub fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        self.set
            .get(&name.to_ascii_lowercase())
            .map(|agent| {
                let supported_tags = agent.supported_resource_tags();
                select_resources(resources, &supported_tags)
            })
            .unwrap_or_default()
    }

    /// Registers a new agent.
    ///
    /// # Arguments
    /// - `agent`: The agent to register.
    pub fn add<T>(&mut self, agent: Arc<T>, label: Option<String>) -> Result<(), BoxError>
    where
        T: Agent<C> + Send + Sync + 'static,
    {
        let name = agent.name().to_ascii_lowercase();
        validate_function_name(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("agent {} already exists", name).into());
        }

        let agent_dyn = AgentWrapper {
            inner: agent,
            label: label.unwrap_or_else(|| name.clone()),
            _phantom: PhantomData,
        };
        self.set.insert(name, Arc::new(agent_dyn));
        Ok(())
    }

    /// Returns an agent by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn DynAgent<C>>> {
        self.set.get(&name.to_ascii_lowercase()).cloned()
    }

    /// Returns an agent by lowercase name.
    pub fn get_lowercase(&self, lowercase_name: &str) -> Option<Arc<dyn DynAgent<C>>> {
        self.set.get(lowercase_name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candid::{CandidType, Principal, utils::ArgumentEncoder};
    use std::time::Duration;

    use crate::{
        AgentInput, BaseContext, CacheExpiry, CacheFeatures, CancellationToken, CanisterCaller,
        CompletionFeatures, CompletionRequest, HttpFeatures, Json, KeysFeatures, ObjectMeta, Path,
        PutMode, PutResult, RequestMeta, StateFeatures, StoreFeatures, ToolInput, ToolOutput,
    };

    #[derive(Clone)]
    struct TestAgentContext {
        engine_id: Principal,
        caller: Principal,
        meta: RequestMeta,
        cancellation_token: CancellationToken,
    }

    impl Default for TestAgentContext {
        fn default() -> Self {
            Self {
                engine_id: Principal::management_canister(),
                caller: Principal::anonymous(),
                meta: RequestMeta::default(),
                cancellation_token: CancellationToken::new(),
            }
        }
    }

    impl StateFeatures for TestAgentContext {
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

    impl KeysFeatures for TestAgentContext {
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

    impl StoreFeatures for TestAgentContext {
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

    impl CacheFeatures for TestAgentContext {
        fn cache_contains(&self, _key: &str) -> bool {
            false
        }

        async fn cache_get<T>(&self, _key: &str) -> Result<T, BoxError>
        where
            T: serde::de::DeserializeOwned,
        {
            Err("not implemented".into())
        }

        async fn cache_get_with<T, F>(&self, _key: &str, _init: F) -> Result<T, BoxError>
        where
            T: Sized + serde::de::DeserializeOwned + Serialize + Send,
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

    impl HttpFeatures for TestAgentContext {
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
            T: serde::de::DeserializeOwned,
        {
            Err("not implemented".into())
        }
    }

    impl crate::CanisterCaller for TestAgentContext {
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

    impl crate::BaseContext for TestAgentContext {
        async fn remote_tool_call(
            &self,
            _endpoint: &str,
            _args: ToolInput<Json>,
        ) -> Result<ToolOutput<Json>, BoxError> {
            Err("not implemented".into())
        }
    }

    impl CompletionFeatures for TestAgentContext {
        async fn completion(
            &self,
            _req: CompletionRequest,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput::default())
        }

        fn model_name(&self) -> String {
            "test-model".to_string()
        }
    }

    impl AgentContext for TestAgentContext {
        fn tool_definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
            Vec::new()
        }

        async fn remote_tool_definitions(
            &self,
            _endpoint: Option<&str>,
            _names: Option<&[String]>,
        ) -> Result<Vec<FunctionDefinition>, BoxError> {
            Ok(Vec::new())
        }

        async fn select_tool_resources(
            &self,
            _name: &str,
            _resources: &mut Vec<Resource>,
        ) -> Vec<Resource> {
            Vec::new()
        }

        fn agent_definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
            Vec::new()
        }

        async fn remote_agent_definitions(
            &self,
            _endpoint: Option<&str>,
            _names: Option<&[String]>,
        ) -> Result<Vec<FunctionDefinition>, BoxError> {
            Ok(Vec::new())
        }

        async fn select_agent_resources(
            &self,
            _name: &str,
            _resources: &mut Vec<Resource>,
        ) -> Vec<Resource> {
            Vec::new()
        }

        async fn definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
            Vec::new()
        }

        async fn tool_call(
            &self,
            _args: ToolInput<Json>,
        ) -> Result<(ToolOutput<Json>, Option<Principal>), BoxError> {
            Ok((ToolOutput::new(Json::Null), None))
        }

        async fn agent_run(
            self,
            _args: AgentInput,
        ) -> Result<(AgentOutput, Option<Principal>), BoxError> {
            Ok((AgentOutput::default(), None))
        }

        async fn remote_agent_run(
            &self,
            _endpoint: &str,
            _args: AgentInput,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput::default())
        }
    }

    struct ExampleAgent {
        id: usize,
    }

    struct OtherAgent;

    struct TaggedAgent;

    struct InvalidAgent;

    fn resource(id: u64, tags: &[&str]) -> Resource {
        Resource {
            _id: id,
            name: format!("resource-{id}"),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        }
    }

    impl Agent<TestAgentContext> for ExampleAgent {
        fn name(&self) -> String {
            "example_agent".to_string()
        }

        fn description(&self) -> String {
            "Example agent used for downcast tests".to_string()
        }

        async fn run(
            &self,
            _ctx: TestAgentContext,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: self.id.to_string(),
                ..AgentOutput::default()
            })
        }
    }

    impl Agent<TestAgentContext> for OtherAgent {
        fn name(&self) -> String {
            "other_agent".to_string()
        }

        fn description(&self) -> String {
            "Other agent used for downcast tests".to_string()
        }

        async fn run(
            &self,
            _ctx: TestAgentContext,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: "other".to_string(),
                ..AgentOutput::default()
            })
        }
    }

    impl Agent<TestAgentContext> for TaggedAgent {
        fn name(&self) -> String {
            "tagged_agent".to_string()
        }

        fn description(&self) -> String {
            "Agent that consumes text and code resources".to_string()
        }

        fn supported_resource_tags(&self) -> Vec<String> {
            vec!["text".to_string(), "code".to_string()]
        }

        fn tool_dependencies(&self) -> Vec<String> {
            vec!["lookup".to_string(), "summarize".to_string()]
        }

        async fn run(
            &self,
            _ctx: TestAgentContext,
            prompt: String,
            resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: format!("{prompt}:{}", resources.len()),
                ..AgentOutput::default()
            })
        }
    }

    impl Agent<TestAgentContext> for InvalidAgent {
        fn name(&self) -> String {
            "bad-agent".to_string()
        }

        fn description(&self) -> String {
            "Invalid function name".to_string()
        }

        async fn run(
            &self,
            _ctx: TestAgentContext,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput::default())
        }
    }

    #[test]
    fn dyn_agent_downcast_ref_returns_inner_agent() {
        let agent = Arc::new(ExampleAgent { id: 7 });
        let mut agent_set = AgentSet::<TestAgentContext>::new();
        agent_set
            .add(agent, Some("test-label".to_string()))
            .unwrap();

        let dyn_agent = agent_set.get("example_agent").unwrap();
        let concrete = dyn_agent.downcast_ref::<ExampleAgent>().unwrap();

        assert_eq!(concrete.id, 7);
        assert!(dyn_agent.downcast_ref::<OtherAgent>().is_none());
    }

    #[test]
    fn dyn_agent_downcast_returns_original_arc() {
        let agent = Arc::new(ExampleAgent { id: 9 });
        let mut agent_set = AgentSet::<TestAgentContext>::new();
        agent_set
            .add(agent.clone(), Some("test-label".to_string()))
            .unwrap();

        let dyn_agent = agent_set.get("example_agent").unwrap();
        let concrete = dyn_agent
            .downcast::<ExampleAgent>()
            .ok()
            .expect("expected downcast to ExampleAgent to succeed");

        assert_eq!(concrete.id, 9);
        assert!(Arc::ptr_eq(&concrete, &agent));
    }

    #[test]
    fn dyn_agent_downcast_mismatch_returns_original_arc() {
        let agent = Arc::new(ExampleAgent { id: 11 });
        let mut agent_set = AgentSet::<TestAgentContext>::new();
        agent_set
            .add(agent, Some("test-label".to_string()))
            .unwrap();

        let dyn_agent = agent_set.get("example_agent").unwrap();
        let original = dyn_agent.clone();
        let err = dyn_agent
            .downcast::<OtherAgent>()
            .err()
            .expect("expected downcast to OtherAgent to fail");

        assert!(Arc::ptr_eq(&err, &original));
        assert_eq!(err.name(), "example_agent");
        assert_eq!(err.label(), "test-label");
    }

    #[test]
    fn agent_default_methods_and_dyn_wrapper_forward_calls() {
        futures::executor::block_on(async {
            let agent = Arc::new(ExampleAgent { id: 42 });
            let mut resources = vec![resource(1, &["text"])];

            let definition = agent.definition();
            assert_eq!(definition.name, "example_agent");
            assert_eq!(definition.description, agent.description());
            assert_eq!(definition.strict, Some(true));
            assert_eq!(definition.parameters["type"], "object");
            assert_eq!(
                definition.parameters["required"].as_array().unwrap()[0],
                "prompt"
            );
            assert!(agent.supported_resource_tags().is_empty());
            assert!(agent.select_resources(&mut resources).is_empty());
            assert_eq!(resources.len(), 1);
            agent.init(TestAgentContext::default()).await.unwrap();
            assert!(agent.tool_dependencies().is_empty());

            let mut agent_set = AgentSet::<TestAgentContext>::new();
            agent_set
                .add(agent, Some("example label".to_string()))
                .unwrap();
            let dyn_agent = agent_set.get("EXAMPLE_AGENT").unwrap();

            assert_eq!(dyn_agent.label(), "example label");
            assert_eq!(dyn_agent.name(), "example_agent");
            assert_eq!(dyn_agent.definition().name, "example_agent");
            assert!(dyn_agent.tool_dependencies().is_empty());
            assert!(dyn_agent.supported_resource_tags().is_empty());
            dyn_agent.init(TestAgentContext::default()).await.unwrap();

            let output = dyn_agent
                .run(
                    TestAgentContext::default(),
                    "ignored".to_string(),
                    Vec::new(),
                )
                .await
                .unwrap();
            assert_eq!(output.content, "42");
        });
    }

    #[test]
    fn fixture_agents_cover_direct_trait_methods() {
        futures::executor::block_on(async {
            let other = OtherAgent;
            assert_eq!(other.name(), "other_agent");
            assert_eq!(other.description(), "Other agent used for downcast tests");
            assert_eq!(
                other
                    .run(
                        TestAgentContext::default(),
                        "prompt".to_string(),
                        Vec::new()
                    )
                    .await
                    .unwrap()
                    .content,
                "other"
            );

            let tagged = TaggedAgent;
            assert_eq!(
                tagged.tool_dependencies(),
                vec!["lookup".to_string(), "summarize".to_string()]
            );

            let invalid = InvalidAgent;
            assert_eq!(invalid.name(), "bad-agent");
            assert_eq!(invalid.description(), "Invalid function name");
            assert!(
                invalid
                    .run(
                        TestAgentContext::default(),
                        "prompt".to_string(),
                        Vec::new()
                    )
                    .await
                    .unwrap()
                    .content
                    .is_empty()
            );
        });
    }

    #[test]
    fn agent_set_registry_filters_resources_and_reports_errors() {
        futures::executor::block_on(async {
            let mut agent_set = AgentSet::<TestAgentContext>::new();
            agent_set
                .add(Arc::new(ExampleAgent { id: 1 }), None)
                .unwrap();
            agent_set
                .add(Arc::new(TaggedAgent), Some("tagged label".to_string()))
                .unwrap();

            assert!(agent_set.contains("EXAMPLE_AGENT"));
            assert!(agent_set.contains_lowercase("tagged_agent"));
            assert!(!agent_set.contains("missing_agent"));
            assert_eq!(
                agent_set.names(),
                vec!["example_agent".to_string(), "tagged_agent".to_string()]
            );

            let definition = agent_set.definition("TAGGED_AGENT").unwrap();
            assert_eq!(definition.name, "tagged_agent");
            assert!(agent_set.definition("missing_agent").is_none());

            let selected_names = vec!["TAGGED_AGENT".to_string(), "missing_agent".to_string()];
            let selected_definitions = agent_set.definitions(Some(&selected_names));
            assert_eq!(selected_definitions.len(), 1);
            assert_eq!(selected_definitions[0].name, "tagged_agent");
            assert_eq!(agent_set.definitions(None).len(), 2);

            let selected_functions = agent_set.functions(Some(&selected_names));
            assert_eq!(selected_functions.len(), 1);
            assert_eq!(
                selected_functions[0].supported_resource_tags,
                vec!["text".to_string(), "code".to_string()]
            );
            assert_eq!(agent_set.functions(None).len(), 2);

            let mut empty = Vec::new();
            assert!(
                agent_set
                    .select_resources("tagged_agent", &mut empty)
                    .is_empty()
            );
            let mut resources = vec![
                resource(1, &["image"]),
                resource(2, &["text"]),
                resource(3, &["code", "text"]),
                resource(4, &["audio"]),
            ];
            let selected = agent_set.select_resources("TAGGED_AGENT", &mut resources);
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
                agent_set
                    .select_resources("missing_agent", &mut resources)
                    .is_empty()
            );

            let dyn_agent = agent_set.get_lowercase("tagged_agent").unwrap();
            assert_eq!(dyn_agent.label(), "tagged label");
            let output = dyn_agent
                .run(
                    TestAgentContext::default(),
                    "prompt".to_string(),
                    vec![resource(9, &["text"])],
                )
                .await
                .unwrap();
            assert_eq!(output.content, "prompt:1");
            assert!(agent_set.get("missing_agent").is_none());
            assert!(agent_set.get_lowercase("missing_agent").is_none());

            let duplicate = agent_set
                .add(Arc::new(ExampleAgent { id: 2 }), None)
                .unwrap_err();
            assert!(duplicate.to_string().contains("already exists"));

            let invalid = agent_set.add(Arc::new(InvalidAgent), None).unwrap_err();
            assert!(invalid.to_string().contains("invalid character"));
        });
    }

    #[test]
    fn test_agent_context_mock_features_cover_default_paths() {
        futures::executor::block_on(async {
            let ctx = TestAgentContext::default();
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
            assert_eq!(
                ctx.completion(CompletionRequest::default(), Vec::new())
                    .await
                    .unwrap()
                    .content,
                ""
            );
            assert_eq!(ctx.model_name(), "test-model");
            assert!(ctx.tool_definitions(None).is_empty());
            assert!(
                ctx.remote_tool_definitions(None, None)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                ctx.select_tool_resources("tool", &mut Vec::new())
                    .await
                    .is_empty()
            );
            assert!(ctx.agent_definitions(None).is_empty());
            assert!(
                ctx.remote_agent_definitions(None, None)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                ctx.select_agent_resources("agent", &mut Vec::new())
                    .await
                    .is_empty()
            );
            assert!(ctx.definitions(None).await.is_empty());
            assert!(
                ctx.tool_call(ToolInput::new("tool".to_string(), Json::Null))
                    .await
                    .unwrap()
                    .0
                    .output
                    .is_null()
            );
            assert!(
                ctx.clone()
                    .agent_run(AgentInput {
                        name: "agent".to_string(),
                        prompt: "prompt".to_string(),
                        ..Default::default()
                    })
                    .await
                    .unwrap()
                    .0
                    .content
                    .is_empty()
            );
            assert!(
                ctx.remote_agent_run(
                    "https://example.test",
                    AgentInput {
                        name: "agent".to_string(),
                        prompt: "prompt".to_string(),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
                .content
                .is_empty()
            );
        });
    }
}
