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
    Resource, Tool, ToolOutput, normalize_strict_schema, root_schema_for,
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
            schema: normalize_strict_schema(schema.to_value()),
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
#[allow(deprecated)]
mod tests {
    use super::*;
    use anda_core::{CompletionFeatures, ToolCall};
    use serde_json::json;

    use crate::engine::EngineBuilder;

    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq)]
    struct Contact {
        name: String,
        phone: String,
    }

    #[derive(Clone)]
    struct MockCompletion {
        output: AgentOutput,
    }

    impl CompletionFeatures for MockCompletion {
        async fn completion(
            &self,
            req: CompletionRequest,
            resources: Vec<Resource>,
        ) -> Result<AgentOutput, BoxError> {
            assert!(req.instructions.contains("extract structured data"));
            assert_eq!(req.prompt, "Ada, 555-0100");
            assert_eq!(req.tools.len(), 1);
            assert!(req.tool_choice_required);
            assert_eq!(req.max_output_tokens, Some(64));
            assert!(resources.is_empty());
            Ok(self.output.clone())
        }

        fn model_name(&self) -> String {
            "mock-model".to_string()
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn submit_tool_schema_submit_and_call_round_trip_typed_payload() {
        let tool = SubmitTool::<Contact>::new();
        assert_eq!(tool.name(), "submit_contact");
        assert_eq!(
            tool.description(),
            "Submit the structured data you extracted from the provided text."
        );
        let definition = tool.definition();
        assert_eq!(definition.name, "submit_contact");
        assert_eq!(definition.strict, Some(true));
        assert_eq!(definition.parameters["type"], "object");

        let contact = Contact {
            name: "Ada".to_string(),
            phone: "555-0100".to_string(),
        };
        assert_eq!(
            tool.submit(json!({"name": "Ada", "phone": "555-0100"}))
                .unwrap(),
            contact
        );
        assert!(tool.submit(json!({"name": "Ada"})).is_err());

        let ctx = EngineBuilder::new().mock_ctx().base;
        let output = tool.call(ctx, contact.clone(), Vec::new()).await.unwrap();
        assert_eq!(output.output, contact);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn extractor_extracts_first_tool_call_and_reports_failure_modes() {
        let extractor = Extractor::<Contact>::new(Some(64), None);
        assert_eq!(extractor.name(), "contact_extractor");
        assert_eq!(
            extractor.description(),
            "Extract structured data from text using LLMs."
        );

        let ctx = MockCompletion {
            output: AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "submit_contact".to_string(),
                    args: json!({"name": "Ada", "phone": "555-0100"}),
                    call_id: Some("call-1".to_string()),
                    result: None,
                    remote_id: None,
                }],
                ..Default::default()
            },
        };
        let (contact, output) = extractor
            .extract(&ctx, "Ada, 555-0100".to_string())
            .await
            .unwrap();
        assert_eq!(
            contact,
            Contact {
                name: "Ada".to_string(),
                phone: "555-0100".to_string()
            }
        );
        assert_eq!(output.tool_calls.len(), 1);

        let failed = MockCompletion {
            output: AgentOutput {
                failed_reason: Some("model failed".to_string()),
                ..Default::default()
            },
        };
        assert!(
            extractor
                .extract(&failed, "Ada, 555-0100".to_string())
                .await
                .unwrap_err()
                .to_string()
                .contains("model failed")
        );

        let missing_tool = MockCompletion {
            output: AgentOutput::default(),
        };
        assert!(
            extractor
                .extract(&missing_tool, "Ada, 555-0100".to_string())
                .await
                .unwrap_err()
                .to_string()
                .contains("no tool_calls")
        );

        let invalid_args = MockCompletion {
            output: AgentOutput {
                tool_calls: vec![ToolCall {
                    name: "submit_contact".to_string(),
                    args: json!({"name": "Ada"}),
                    call_id: None,
                    result: None,
                    remote_id: None,
                }],
                ..Default::default()
            },
        };
        assert!(
            extractor
                .extract(&invalid_args, "Ada, 555-0100".to_string())
                .await
                .unwrap_err()
                .to_string()
                .contains("invalid args")
        );
    }
}
