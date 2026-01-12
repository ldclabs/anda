//! Module providing core agent functionality for AI systems.
//!
//! This module defines the core traits and structures for creating and managing AI agents. It provides:
//! - The [`Agent`] trait for defining custom agents with specific capabilities.
//! - Dynamic dispatch capabilities through [`AgentDyn`] trait.
//! - An [`AgentSet`] collection for managing multiple agents.
//!
//! # Key Features
//! - Type-safe agent definitions with clear interfaces.
//! - Asynchronous execution model.
//! - Dynamic dispatch support for runtime agent selection.
//! - Agent registration and management system.
//! - Tool dependency management.
//!
//! # Architecture Overview
//! The module follows a dual-trait pattern:
//! 1. [`Agent`] - Static trait for defining concrete agent implementations.
//! 2. [`AgentDyn`] - Dynamic trait for runtime polymorphism.
//!
//! The [`AgentSet`] acts as a registry and execution manager for agents, providing:
//! - Agent registration and lookup.
//! - Bulk definition retrieval.
//! - Execution routing.
//!
//! # Usage
//!
//! ## Reference Implementations
//! 1. [`Extractor`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/extractor.rs) -
//!    An agent for structured data extraction using LLMs
//! 2. [`DocumentSegmenter`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/segmenter.rs) -
//!    A document segmentation tool using LLMs
//! 3. [`CharacterAgent`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/character.rs) -
//!    A role-playing AI agent, also serving as the core agent for [`anda_bot`](https://github.com/ldclabs/anda/blob/main/agents/anda_bot/README.md)

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{
    BoxError, BoxPinFut, Function,
    context::AgentContext,
    model::{AgentOutput, FunctionDefinition, Resource},
    select_resources, validate_function_name,
};

/// Arguments for an AI agent.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentArgs {
    /// optimized prompt or message.
    pub prompt: String,
}

/// Core trait defining an AI agent's behavior.
///
/// # Type Parameters
/// - `C`: The context type that implements [`AgentContext`], must be thread-safe and have a static lifetime.
pub trait Agent<C>: Send + Sync
where
    C: AgentContext + Send + Sync,
{
    /// Returns the agent's name as a String
    /// The unique name of the agent, case-insensitive, must follow these rules in lowercase:
    ///
    /// # Rules
    /// - Must not be empty;
    /// - Must not exceed 64 characters;
    /// - Must start with a lowercase letter;
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_);
    /// - Unique within the engine in lowercase.
    fn name(&self) -> String;

    /// Returns the agent's capabilities description in a short string.
    fn description(&self) -> String;

    /// Returns the agent's function definition for API integration.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The structured definition of the agent's capabilities.
    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name().to_ascii_lowercase(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "optimized prompt or message."},
                },
                "required": ["prompt"],
            }),
            strict: None,
        }
    }

    /// It is used to select resources based on the provided tags.
    /// If the agent requires specific resources, it can filter them based on the tags.
    /// By default, it returns an empty list.
    ///
    /// # Returns
    /// - A list of resource tags from the tags provided that supported by the agent.
    fn supported_resource_tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Selects resources based on the agent's supported tags.
    fn select_resources(&self, resources: &mut Vec<Resource>) -> Vec<Resource> {
        let supported_tags = self.supported_resource_tags();
        select_resources(resources, &supported_tags)
    }

    /// Initializes the tool with the given context.
    /// It will be called once when building the Anda engine.
    fn init(&self, _ctx: C) -> impl Future<Output = Result<(), BoxError>> + Send {
        futures::future::ready(Ok(()))
    }

    /// Returns a list of tool dependencies required by the agent.
    /// The tool dependencies are checked when building the engine.
    fn tool_dependencies(&self) -> Vec<String> {
        Vec::new()
    }

    /// Executes the agent's main logic with given context and inputs.
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing [`AgentContext`].
    /// - `prompt`: The input prompt or message for the agent.
    /// - `resources`: Optional additional resources, If resources don’t meet the agent’s expectations, ignore them.
    ///
    /// # Returns
    /// - A future resolving to Result<[`AgentOutput`], BoxError>.
    fn run(
        &self,
        ctx: C,
        prompt: String,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Dynamic dispatch version of Agent trait for runtime flexibility.
///
/// This trait allows for runtime polymorphism of agents, enabling dynamic agent selection
/// and execution without knowing the concrete type at compile time.
pub trait AgentDyn<C>: Send + Sync
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

/// Adapter for converting static Agent to dynamic dispatch.
struct AgentWrapper<T, C>
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static,
{
    inner: Arc<T>,
    label: String,
    _phantom: PhantomData<C>,
}

impl<T, C> AgentDyn<C> for AgentWrapper<T, C>
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

/// Collection of agents with lookup capabilities.
///
/// # Type Parameters
/// - `C`: The context type that implements [`AgentContext`].
#[derive(Default)]
pub struct AgentSet<C: AgentContext> {
    pub set: BTreeMap<String, Box<dyn AgentDyn<C>>>,
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

    /// Checks if an agent with given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(&name.to_ascii_lowercase())
    }

    /// Returns the names of all agents in the set.
    pub fn names(&self) -> Vec<String> {
        self.set.keys().cloned().collect()
    }

    /// Retrieves definition for a specific agent.
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|agent| agent.definition())
    }

    /// Returns definitions for all or specified agents.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// - Vec<[`FunctionDefinition`]>: Vector of agent definitions.
    pub fn definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        let names: Option<Vec<String>> =
            names.map(|names| names.iter().map(|n| n.to_ascii_lowercase()).collect());
        self.set
            .iter()
            .filter_map(|(name, agent)| match &names {
                Some(names) => {
                    if names.contains(name) {
                        Some(agent.definition())
                    } else {
                        None
                    }
                }
                None => Some(agent.definition()),
            })
            .collect()
    }

    /// Returns a list of functions for all or specified agents.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// - Vec<[`Function`]>: Vector of agent functions.
    pub fn functions(&self, names: Option<&[&str]>) -> Vec<Function> {
        let names: Option<Vec<String>> =
            names.map(|names| names.iter().map(|n| n.to_ascii_lowercase()).collect());
        self.set
            .iter()
            .filter_map(|(name, agent)| match &names {
                Some(names) => {
                    if names.contains(name) {
                        Some(Function {
                            definition: agent.definition(),
                            supported_resource_tags: agent.supported_resource_tags(),
                        })
                    } else {
                        None
                    }
                }
                None => Some(Function {
                    definition: agent.definition(),
                    supported_resource_tags: agent.supported_resource_tags(),
                }),
            })
            .collect()
    }

    /// Extracts resources from the provided list based on the agent's supported tags.
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

    /// Registers a new agent in the set.
    ///
    /// # Arguments
    /// - `agent`: The agent to register, must implement [`Agent`] trait.
    pub fn add<T>(&mut self, agent: T, label: Option<String>) -> Result<(), BoxError>
    where
        T: Agent<C> + Send + Sync + 'static,
    {
        let name = agent.name().to_ascii_lowercase();
        if self.set.contains_key(&name) {
            return Err(format!("agent {} already exists", name).into());
        }

        validate_function_name(&name)?;
        let agent_dyn = AgentWrapper {
            inner: Arc::new(agent),
            label: label.unwrap_or_else(|| name.clone()),
            _phantom: PhantomData,
        };
        self.set.insert(name, Box::new(agent_dyn));
        Ok(())
    }

    /// Retrieves an agent by name.
    pub fn get(&self, name: &str) -> Option<&dyn AgentDyn<C>> {
        self.set.get(&name.to_ascii_lowercase()).map(|v| &**v)
    }
}
