//! Module providing core tooling functionality for AI Agents.
//!
//! This module defines the core traits and structures for creating and managing tools
//! that can be used by AI Agents. It provides:
//! - The [`Tool`] trait for defining custom tools with typed arguments and outputs.
//! - Dynamic dispatch capabilities through [`ToolDyn`] trait.
//! - A [`ToolSet`] collection for managing multiple tools.
//!
//! # Key Features
//! - Type-safe tool definitions with schema validation.
//! - Asynchronous execution model.
//! - Dynamic dispatch support for runtime tool selection.
//! - Tool registration and management system.
//!
//! # Usage
//!
//! ## Reference Implementations
//! 1. [`GoogleSearchTool`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/google.rs) -
//!    A tool for performing web searches and retrieve results.
//! 2. [`SubmitTool`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/extractor.rs) -
//!    A tool for extracting structured data using LLMs.
//! 3. [`TransferTool`](https://github.com/ldclabs/anda/blob/main/tools/anda_icp/src/ledger/transfer.rs) -
//!    A tool for handling ICP blockchain transfers.
//! 4. [`BalanceOfTool`](https://github.com/ldclabs/anda/blob/main/tools/anda_icp/src/ledger/balance.rs) -
//!    A tool for querying ICP blockchain balances.
//!
//! These reference implementations share a common feature: they automatically generate the JSON Schema.
//! required for LLMs Function Calling.

use serde::{Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{
    BoxError, BoxPinFut, Function, Json, Resource, ToolOutput, context::BaseContext,
    model::FunctionDefinition, select_resources, validate_function_name,
};

/// Core trait for implementing tools that can be used by the AI Agent system.
///
/// # Type Parameters
/// - `C`: The context type that implements [`BaseContext`], must be thread-safe and have a static lifetime.
pub trait Tool<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// The arguments type of the tool.
    type Args: DeserializeOwned + Send;

    /// The output type of the tool.
    type Output: Serialize;

    /// Returns the tool's name.
    ///
    /// # Rules
    /// - Must not be empty;
    /// - Must not exceed 64 characters;
    /// - Must start with a lowercase letter;
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_);
    /// - Unique within the engine.
    fn name(&self) -> String;

    /// Returns the tool's capabilities description in a short string.
    fn description(&self) -> String;

    /// Provides the tool's definition including its parameters schema.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The schema definition of the tool's parameters and metadata.
    fn definition(&self) -> FunctionDefinition;

    /// It is used to select resources based on the provided tags.
    /// If the tool requires specific resources, it can filter them based on the tags.
    /// By default, it returns an empty list.
    ///
    /// # Returns
    /// - A list of resource tags from the tags provided that supported by the tool
    fn supported_resource_tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Selects resources based on the tool's supported tags.
    /// This method filters the provided resources based on the tags that the tool supports.
    fn select_resources(&self, resources: &mut Vec<Resource>) -> Vec<Resource> {
        let supported_tags = self.supported_resource_tags();
        select_resources(resources, &supported_tags)
    }

    /// Initializes the tool with the given context.
    /// It will be called once when building the Anda engine.
    fn init(&self, _ctx: C) -> impl Future<Output = Result<(), BoxError>> + Send {
        futures::future::ready(Ok(()))
    }

    /// Executes the tool with given context and arguments.
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing [`BaseContext`].
    /// - `args`: struct arguments for the tool.
    /// - `resources`: Optional additional resources, If resources don’t meet the tool’s expectations, return an error.
    ///
    /// # Returns
    /// - A future resolving to Result<[`ToolOutput<Output>`], BoxError>
    fn call(
        &self,
        ctx: C,
        args: Self::Args,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<ToolOutput<Self::Output>, BoxError>> + Send;

    /// Executes the tool with given context and arguments using raw JSON string
    /// Returns the output as a JSON object.
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
                artifacts: result.artifacts,
                usage: result.usage,
            })
        }
    }
}

/// Dynamic dispatch version of the Tool trait.
///
/// This trait allows for runtime polymorphism of tools, enabling different tool implementations.
/// to be stored and called through a common interface.
pub trait ToolDyn<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
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

/// Wrapper to convert static Tool implementation to dynamic dispatch.
struct ToolWrapper<T, C>(Arc<T>, PhantomData<C>)
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static;

impl<T, C> ToolDyn<C> for ToolWrapper<T, C>
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static,
{
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

/// Collection of tools that can be used by the AI Agent
///
/// # Type Parameters
/// - `C`: The context type that implements [`BaseContext`].
#[derive(Default)]
pub struct ToolSet<C: BaseContext> {
    pub set: BTreeMap<String, Box<dyn ToolDyn<C>>>,
}

impl<C> ToolSet<C>
where
    C: BaseContext + Send + Sync + 'static,
{
    /// Creates a new empty ToolSet
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Checks if a tool with the given name exists in the set
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(&name.to_ascii_lowercase())
    }

    /// Returns the names of all tools in the set
    pub fn names(&self) -> Vec<String> {
        self.set.keys().cloned().collect()
    }

    /// Retrieves definition for a specific agent.
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|tool| tool.definition())
    }

    /// Returns definitions for all or specified tools.
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
            .filter_map(|(name, tool)| match &names {
                Some(names) => {
                    if names.contains(name) {
                        Some(tool.definition())
                    } else {
                        None
                    }
                }
                None => Some(tool.definition()),
            })
            .collect()
    }

    /// Returns a list of functions for all or specified tools.
    ///
    /// # Arguments
    /// - `names`: Optional slice of tool names to filter by.
    ///
    /// # Returns
    /// - Vec<[`Function`]>: Vector of tool functions.
    pub fn functions(&self, names: Option<&[&str]>) -> Vec<Function> {
        let names: Option<Vec<String>> =
            names.map(|names| names.iter().map(|n| n.to_ascii_lowercase()).collect());
        self.set
            .iter()
            .filter_map(|(name, tool)| match &names {
                Some(names) => {
                    if names.contains(name) {
                        Some(Function {
                            definition: tool.definition(),
                            supported_resource_tags: tool.supported_resource_tags(),
                        })
                    } else {
                        None
                    }
                }
                None => Some(Function {
                    definition: tool.definition(),
                    supported_resource_tags: tool.supported_resource_tags(),
                }),
            })
            .collect()
    }

    /// Extracts resources from the provided list based on the tool's supported tags.
    pub fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|tool| {
                let supported_tags = tool.supported_resource_tags();
                select_resources(resources, &supported_tags)
            })
            .unwrap_or_default()
    }

    /// Adds a new tool to the set
    ///
    /// # Arguments
    /// - `tool`: The tool to add, must implement the [`Tool`] trait.
    pub fn add<T>(&mut self, tool: T) -> Result<(), BoxError>
    where
        T: Tool<C> + Send + Sync + 'static,
    {
        let name = tool.name().to_ascii_lowercase();
        validate_function_name(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("tool {} already exists", name).into());
        }

        let tool_dyn = ToolWrapper(Arc::new(tool), PhantomData);
        self.set.insert(name, Box::new(tool_dyn));
        Ok(())
    }

    /// Retrieves a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn ToolDyn<C>> {
        self.set.get(&name.to_ascii_lowercase()).map(|v| &**v)
    }
}
