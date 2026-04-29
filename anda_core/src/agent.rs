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
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

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
    fn label(&self) -> &str;

    fn name(&self) -> String;

    fn definition(&self) -> FunctionDefinition;

    fn tool_dependencies(&self) -> Vec<String>;

    fn supported_resource_tags(&self) -> Vec<String>;

    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>>;

    fn run(
        &self,
        ctx: C,
        prompt: String,
        resources: Vec<Resource>,
    ) -> BoxPinFut<Result<AgentOutput, BoxError>>;
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
        if self.set.contains_key(&name) {
            return Err(format!("agent {} already exists", name).into());
        }

        validate_function_name(&name)?;
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
