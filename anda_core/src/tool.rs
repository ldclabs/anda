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
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

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
                artifacts: result.artifacts,
                usage: result.usage,
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
