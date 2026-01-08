//! A module for extracting structured data from unstructured text using Language Models (LLMs).
//!
//! This module provides high-level abstractions for:
//! - Defining structured data schemas using Rust types
//! - Extracting data from text using LLMs
//! - Validating and processing extracted data
//!
//! # Key Components
//!
//! ## [`SubmitTool`]
//! - Wraps a type `T` that defines the JSON schema for structured data
//! - Provides functionality to submit and validate data
//! - Implements the [`Tool`] trait for integration with the LLM system
//!
//! ## [`Extractor`]
//! - Main interface for extracting structured data from text
//! - Uses LLMs to process unstructured input
//! - Implements the [`Agent`] trait for integration with the agent system
//!
//! # Usage
//!
//! 1. Define your data structure with `#[derive(JsonSchema, Serialize, Deserialize)]`
//! 2. Create an `Extractor` instance with your type
//! 3. Use the `extract()` method to process text
//!
//! # Example
//!
//! ```rust,ignore
//! #[derive(JsonSchema, Serialize, Deserialize)]
//! struct ContactInfo {
//!     name: String,
//!     phone: String,
//! }
//!
//! let extractor = Extractor::<ContactInfo>::default();
//! let (data, _) = extractor.extract(&ctx, "John Doe, phone: 123-456-7890").await?;
//! ```
//!
//! # Notes
//! - The target structure must implement `serde::Deserialize`, `serde::Serialize`,
//!   and `schemars::JsonSchema` traits
//! - These traits can be easily derived using the `derive` macro

use anda_core::{
    Agent, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, FunctionDefinition,
    Resource, Tool, ToolOutput, root_schema_for,
};
use schemars::JsonSchema;
use serde_json::Value;
use std::marker::PhantomData;

pub use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::context::{AgentCtx, BaseCtx};

/// A tool for submitting structured data extracted from text
///
/// Wraps a type `T` that defines the JSON schema for the structured data
/// and provides functionality to submit and validate the data
#[derive(Debug, Clone)]
pub struct SubmitTool<T: JsonSchema + DeserializeOwned + Send + Sync> {
    name: String,
    schema: Value,

    _t: PhantomData<T>,
}

impl<T> Default for SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    /// Creates a new SubmitTool instance
    ///
    /// Automatically generates a JSON schema from the type `T` and
    /// uses the type's title (if available) as the tool name
    pub fn new() -> SubmitTool<T> {
        let schema = root_schema_for::<T>();
        let name = schema
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_ascii_lowercase();
        SubmitTool {
            name,
            schema: schema.to_value(),
            _t: PhantomData,
        }
    }

    /// Validates and deserializes the submitted arguments
    ///
    /// # Arguments
    /// * `args` - JSON string containing the structured data
    ///
    /// # Returns
    /// Deserialized instance of type `T` or an error if validation fails
    pub fn submit(&self, args: Value) -> Result<T, BoxError> {
        serde_json::from_value(args).map_err(|err| format!("invalid args: {}", err).into())
    }
}

impl<T> Tool<BaseCtx> for SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    type Args = T;
    type Output = T;

    fn name(&self) -> String {
        format!("submit_{}", self.name)
    }

    fn description(&self) -> String {
        "Submit the structured data you extracted from the provided text.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        Ok(ToolOutput::new(args))
    }
}

/// Extractor for structured data from text using LLMs
///
/// Provides functionality to extract structured data from unstructured text
/// using a language model and a defined schema
#[derive(Debug, Clone)]
pub struct Extractor<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync> {
    tool: SubmitTool<T>,
    instructions: String,
    max_tokens: Option<usize>,
}

impl<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync> Default for Extractor<T> {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync> Extractor<T> {
    /// Creates a new Extractor instance with default system prompt
    ///
    /// # Arguments
    /// * `max_tokens` - Optional maximum number of tokens for the completion
    /// * `system_prompt` - Optional custom system prompt
    pub fn new(max_tokens: Option<usize>, system_prompt: Option<String>) -> Self {
        let tool = SubmitTool::new();
        Self::new_with_tool(tool, max_tokens, system_prompt)
    }

    /// Creates a new Extractor instance with a custom SubmitTool
    ///
    /// # Arguments
    /// * `tool` - Pre-configured SubmitTool instance
    /// * `max_tokens` - Optional maximum number of tokens for the completion
    /// * `instructions` - Optional custom system instructions
    pub fn new_with_tool(
        tool: SubmitTool<T>,
        max_tokens: Option<usize>,
        instructions: Option<String>,
    ) -> Self {
        let tool_name = tool.name();
        Self {
            tool,
            max_tokens,
            instructions: instructions.unwrap_or_else(|| format!("\
            You are an AI assistant whose purpose is to\
            extract structured data from the provided text.\n\
            You will have access to a `{tool_name}` function that defines the structure of the data to extract from the provided text.\n\
            Use the `{tool_name}` function to submit the structured data.\n\
            Be sure to fill out every field and ALWAYS CALL THE `{tool_name}` function, event with default values!!!.")),
        }
    }

    /// Extracts structured data from the provided text
    ///
    /// # Arguments
    /// * `ctx` - Context implementing CompletionFeatures
    /// * `prompt` - Input text to extract data from
    ///
    /// # Returns
    /// Tuple containing the extracted data and the full agent output
    pub async fn extract(
        &self,
        ctx: &impl CompletionFeatures,
        prompt: String,
    ) -> Result<(T, AgentOutput), BoxError> {
        let req = CompletionRequest {
            instructions: self.instructions.clone(),
            prompt,
            tools: vec![self.tool.definition()],
            tool_choice_required: true,
            max_output_tokens: self.max_tokens,
            ..Default::default()
        };

        let mut res = ctx.completion(req, Vec::new()).await?;
        if let Some(failed) = res.failed_reason {
            return Err(failed.into());
        }

        if let Some(tool) = res.tool_calls.iter_mut().next() {
            let result = self.tool.submit(tool.args.clone())?;
            return Ok((result, res));
        }

        Err(format!("extract with {} failed, no tool_calls", self.tool.name()).into())
    }
}

impl<T> Agent<AgentCtx> for Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    fn name(&self) -> String {
        format!("{}_extractor", self.tool.name)
    }

    fn description(&self) -> String {
        "Extract structured data from text using LLMs.".to_string()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let (_, res) = self.extract(&ctx, prompt).await?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use anda_core::{AgentContext, AgentInput, ToolInput};
    use serde_json::json;

    use super::*;
    use crate::{engine::EngineBuilder, model::Model};

    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }

    #[test]
    fn test_definition() {
        let tool = SubmitTool::<TestStruct>::new();
        let definition = tool.definition();
        assert_eq!(definition.name, "submit_teststruct");
        let s = serde_json::to_string(&definition).unwrap();
        println!("{}", s);
        // {"name":"submit_teststruct","description":"Submit the structured data you extracted from the provided text.","parameters":{"properties":{"age":{"maximum":255,"minimum":0,"type":"integer"},"name":{"type":"string"}},"required":["name","age"],"title":"TestStruct","type":"object"},"strict":true}
        assert!(s.contains(r#""required":["name"]"#));
        assert!(!s.contains("$schema"));

        let agent = Extractor::<TestStruct>::default();
        let definition = agent.definition();
        assert_eq!(definition.name, "teststruct_extractor");
        let s = serde_json::to_string(&definition).unwrap();
        println!("{}", s);
        // {"name":"teststruct_extractor","description":"Extract structured data from text using LLMs.","parameters":{"properties":{"prompt":{"description":"optimized prompt or message.","type":"string"}},"required":["prompt"],"type":"object"}}
        assert!(s.contains(
            r#""parameters":{"properties":{"prompt":{"description":"optimized prompt or message.","type":"string"}},"required":["prompt"],"type":"object"}}"#
        ));
        assert!(!s.contains("$schema"));
    }

    #[tokio::test]
    async fn test_with_ctx() {
        let tool = SubmitTool::<TestStruct>::default();
        let agent = Extractor::<TestStruct>::default();
        let tool_name = tool.name();
        let agent_name = agent.name();

        let ctx = EngineBuilder::new()
            .with_model(Model::mock_implemented())
            .register_tool(tool)
            .unwrap()
            .register_agent(agent, None)
            .unwrap()
            .mock_ctx();

        let (res, _) = ctx
            .tool_call(ToolInput::new(
                tool_name.clone(),
                json!({"name":"Anda","age": 1}),
            ))
            .await
            .unwrap();
        assert_eq!(res.output, json!({"name":"Anda","age": 1}));

        let (res, _) = ctx
            .tool_call(ToolInput::new(tool_name.clone(), json!({"name": "Anda"})))
            .await
            .unwrap();
        assert_eq!(res.output, json!({"name": "Anda","age": null}));

        let res = ctx
            .tool_call(ToolInput::new(tool_name.clone(), json!({"name": 123})))
            .await;
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("invalid args"));

        let _res = ctx
            .agent_run(AgentInput::new(
                agent_name.to_string(),
                r#"{"name": "Anda"}"#.into(),
            ))
            .await
            .unwrap();
        // println!("test_with_ctx: {:?}", res);
        // assert_eq!(
        //     res.tool_calls.as_ref().unwrap()[0].result.unwrap().as_str(),
        //     Some(r#"{"name":"Anda","age":null}"#)
        // );

        let res = ctx
            .agent_run(AgentInput::new(
                agent_name.to_string(),
                r#"{"name": 123}"#.into(),
            ))
            .await;
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("invalid args"));
    }
}
