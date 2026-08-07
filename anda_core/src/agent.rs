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
    BoxError, BoxPinFut, Function, ToolGroup, ToolGroupInfo,
    context::AgentContext,
    model::{AgentOutput, FunctionDefinition, Resource},
    registry::{collect_groups, select_by_names},
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
    /// - Must not exceed 64 bytes;
    /// - Must start with a lowercase letter;
    /// - Can only contain: lowercase letters (a-z), digits (0-9), underscores (_), and hyphens (-);
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

    /// Returns the capability group this agent belongs to, if any.
    ///
    /// Agents that form a coherent bundle (for example the media-understanding
    /// agents) return the same [`ToolGroupInfo`] so the registry can present
    /// them as one group in discovery, alongside tool groups. The default
    /// implementation returns `None`.
    fn group(&self) -> Option<ToolGroupInfo> {
        None
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
        std::future::ready(Ok(()))
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

    /// Returns the agent's context label.
    ///
    /// The label seeds the child-context path and the default model label when
    /// the runtime executes this agent (see `Engine::ctx_with`). It defaults to
    /// the lowercase agent name and is not used for name-based lookup, which
    /// always keys on [`DynAgent::name`].
    fn label(&self) -> &str;

    /// Returns the unique agent name.
    fn name(&self) -> String;

    /// Returns the function definition exposed to model providers.
    fn definition(&self) -> FunctionDefinition;

    /// Returns tool names required by this agent.
    fn tool_dependencies(&self) -> Vec<String>;

    /// Returns the capability group this agent belongs to, if any.
    fn group(&self) -> Option<ToolGroupInfo>;

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

    fn group(&self) -> Option<ToolGroupInfo> {
        self.inner.group()
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
    ///
    /// Keys are lowercase names satisfying [`validate_function_name`] and equal
    /// each agent's own lowercased name; [`AgentSet::add_dyn`] is the only
    /// insert path, so lookup and dispatch can assume lowercase keys.
    set: BTreeMap<String, Arc<dyn DynAgent<C>>>,
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

    /// Returns the capability groups declared by the registered agents.
    ///
    /// Agents that declare the same [`ToolGroupInfo::id`] are collected into one
    /// [`ToolGroup`] whose `members` are exactly the registered agent names in
    /// that group, sorted for determinism. Group metadata is taken from the
    /// first agent (by lowercase name order) that declares the id.
    pub fn groups(&self) -> Vec<ToolGroup> {
        collect_groups(self.set.iter().map(|(name, agent)| (name, agent.group())))
    }

    /// Returns the function definition for a specific agent.
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|agent| agent.definition())
    }

    /// Returns function definitions for all agents or the selected names.
    ///
    /// Requested names are matched case-insensitively and deduplicated.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// A vector of agent definitions.
    pub fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        select_by_names(&self.set, names, |agent| agent.definition())
    }

    /// Returns function metadata for all agents or the selected names.
    ///
    /// Requested names are matched case-insensitively and deduplicated.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// A vector of agent function metadata.
    pub fn functions(&self, names: Option<&[String]>) -> Vec<Function> {
        select_by_names(&self.set, names, |agent| Function {
            definition: agent.definition(),
            supported_resource_tags: agent.supported_resource_tags(),
        })
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
    /// - `label`: Optional context label; defaults to the lowercase agent name.
    pub fn add<T>(&mut self, agent: Arc<T>, label: Option<String>) -> Result<(), BoxError>
    where
        T: Agent<C> + Send + Sync + 'static,
    {
        let label = label.unwrap_or_else(|| agent.name().to_ascii_lowercase());
        self.add_dyn(Arc::new(AgentWrapper {
            inner: agent,
            label,
            _phantom: PhantomData,
        }))
    }

    /// Registers a type-erased agent, e.g. one drained from another set.
    ///
    /// The registry key is the agent's lowercase name; it must satisfy
    /// [`validate_function_name`] and be unique within the set.
    pub fn add_dyn(&mut self, agent: Arc<dyn DynAgent<C>>) -> Result<(), BoxError> {
        let name = agent.name().to_ascii_lowercase();
        validate_function_name(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("agent {} already exists", name).into());
        }

        self.set.insert(name, agent);
        Ok(())
    }

    /// Iterates registered agents as `(lowercase_name, agent)` pairs in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<dyn DynAgent<C>>)> {
        self.set.iter().map(|(name, agent)| (name.as_str(), agent))
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

impl<C> IntoIterator for AgentSet<C>
where
    C: AgentContext + Send + Sync + 'static,
{
    type Item = Arc<dyn DynAgent<C>>;
    type IntoIter = std::collections::btree_map::IntoValues<String, Arc<dyn DynAgent<C>>>;

    /// Consumes the set, yielding agents in lowercase-name order.
    fn into_iter(self) -> Self::IntoIter {
        self.set.into_values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockContext, resource};

    struct ExampleAgent {
        id: usize,
    }

    struct OtherAgent;

    struct TaggedAgent;

    struct InvalidAgent;

    impl Agent<MockContext> for ExampleAgent {
        fn name(&self) -> String {
            "example_agent".to_string()
        }

        fn description(&self) -> String {
            "Example agent used for downcast tests".to_string()
        }

        fn group(&self) -> Option<ToolGroupInfo> {
            Some(ToolGroupInfo {
                id: "example_bundle".to_string(),
                title: "Example bundle".to_string(),
                description: "Agents used together in tests".to_string(),
                instructions: Some("Combine these agents.".to_string()),
            })
        }

        async fn run(
            &self,
            _ctx: MockContext,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: self.id.to_string(),
                ..AgentOutput::default()
            })
        }
    }

    impl Agent<MockContext> for OtherAgent {
        fn name(&self) -> String {
            "other_agent".to_string()
        }

        fn description(&self) -> String {
            "Other agent used for downcast tests".to_string()
        }

        fn group(&self) -> Option<ToolGroupInfo> {
            Some(ToolGroupInfo {
                id: "example_bundle".to_string(),
                title: "Example bundle".to_string(),
                description: "Agents used together in tests".to_string(),
                instructions: Some("Combine these agents.".to_string()),
            })
        }

        async fn run(
            &self,
            _ctx: MockContext,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: "other".to_string(),
                ..AgentOutput::default()
            })
        }
    }

    impl Agent<MockContext> for TaggedAgent {
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
            _ctx: MockContext,
            prompt: String,
            resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput {
                content: format!("{prompt}:{}", resources.len()),
                ..AgentOutput::default()
            })
        }
    }

    impl Agent<MockContext> for InvalidAgent {
        fn name(&self) -> String {
            "bad.agent".to_string()
        }

        fn description(&self) -> String {
            "Invalid function name".to_string()
        }

        async fn run(
            &self,
            _ctx: MockContext,
            _prompt: String,
            _resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            Ok(AgentOutput::default())
        }
    }

    #[test]
    fn dyn_agent_downcast_ref_returns_inner_agent() {
        let agent = Arc::new(ExampleAgent { id: 7 });
        let mut agent_set = AgentSet::<MockContext>::new();
        agent_set
            .add(agent, Some("test-label".to_string()))
            .unwrap();

        let dyn_agent = agent_set.get("example_agent").unwrap();
        let concrete = dyn_agent.downcast_ref::<ExampleAgent>().unwrap();

        assert_eq!(concrete.id, 7);
        assert!(dyn_agent.downcast_ref::<OtherAgent>().is_none());
    }

    #[test]
    fn agent_set_collects_declared_groups() {
        let mut agent_set = AgentSet::<MockContext>::new();
        agent_set
            .add(Arc::new(ExampleAgent { id: 1 }), None)
            .unwrap();
        agent_set.add(Arc::new(OtherAgent), None).unwrap();
        // TaggedAgent declares no group and must not appear.
        agent_set.add(Arc::new(TaggedAgent), None).unwrap();

        let groups = agent_set.groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, "example_bundle");
        // Both grouped agents land in the group, sorted by name.
        assert_eq!(
            groups[0].members,
            vec!["example_agent".to_string(), "other_agent".to_string()]
        );
        assert_eq!(
            groups[0].instructions.as_deref(),
            Some("Combine these agents.")
        );
    }

    #[test]
    fn dyn_agent_downcast_returns_original_arc() {
        let agent = Arc::new(ExampleAgent { id: 9 });
        let mut agent_set = AgentSet::<MockContext>::new();
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
        let mut agent_set = AgentSet::<MockContext>::new();
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
            agent.init(MockContext::default()).await.unwrap();
            assert!(agent.tool_dependencies().is_empty());

            let mut agent_set = AgentSet::<MockContext>::new();
            agent_set
                .add(agent, Some("example label".to_string()))
                .unwrap();
            let dyn_agent = agent_set.get("EXAMPLE_AGENT").unwrap();

            assert_eq!(dyn_agent.label(), "example label");
            assert_eq!(dyn_agent.name(), "example_agent");
            assert_eq!(dyn_agent.definition().name, "example_agent");
            assert!(dyn_agent.tool_dependencies().is_empty());
            assert!(dyn_agent.supported_resource_tags().is_empty());
            dyn_agent.init(MockContext::default()).await.unwrap();

            let output = dyn_agent
                .run(MockContext::default(), "ignored".to_string(), Vec::new())
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
                    .run(MockContext::default(), "prompt".to_string(), Vec::new())
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
            assert_eq!(invalid.name(), "bad.agent");
            assert_eq!(invalid.description(), "Invalid function name");
            assert!(
                invalid
                    .run(MockContext::default(), "prompt".to_string(), Vec::new())
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
            let mut agent_set = AgentSet::<MockContext>::new();
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
                    MockContext::default(),
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
}
