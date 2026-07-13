use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{
    AgentOutput, BoxError, ContentPart, Document, Documents, FunctionDefinition, Json, Message,
    Resource,
};

/// LLM completion capability exposed by an agent context.
pub trait CompletionFeatures: Sized {
    /// Generates a completion for the request and optional resources.
    fn completion(
        &self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;

    /// Returns the name of the model.
    fn model_name(&self) -> String;
}

/// Provider-agnostic reasoning/thinking effort requested for a completion.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelEffort {
    /// Smallest reasoning budget supported by the provider.
    Minimal,
    /// Low reasoning budget.
    Low,
    /// Medium reasoning budget.
    Medium,
    /// High reasoning budget.
    High,
    /// Maximum reasoning budget supported by the provider.
    Max,
}

impl ModelEffort {
    /// Returns the lowercase wire value for this effort level.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

impl fmt::Display for ModelEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provider-neutral completion request.
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// System instructions sent to the completion provider.
    pub instructions: String,

    /// Role used for `prompt` and `content`; defaults to `user` when omitted.
    pub role: Option<String>,

    /// The chat history to be sent to the completion model provider.
    pub chat_history: Vec<Message>,

    /// Provider-specific history used by model adapters. It is empty for most callers.
    pub raw_history: Vec<Json>,

    /// The documents to embed into the prompt.
    pub documents: Documents,

    /// Prompt sent to the completion provider using `role`.
    /// It can be empty.
    pub prompt: String,

    /// The content parts to be sent to the completion model provider.
    /// It can be empty.
    pub content: Vec<ContentPart>,

    /// The tools to be sent to the completion model provider.
    pub tools: Vec<FunctionDefinition>,

    /// Whether the tool choice is required.
    pub tool_choice_required: bool,

    /// Sampling temperature requested from the provider, usually in the `[0.0, 2.0]` range.
    pub temperature: Option<f64>,

    /// Upper bound for the number of tokens that can be generated for a response.
    pub max_output_tokens: Option<usize>,

    /// An object specifying the JSON format that the model must output.
    pub output_schema: Option<Json>,

    /// The stop sequence to be sent to the completion model provider.
    pub stop: Option<Vec<String>>,

    /// The name or label of the model to be used for the completion request.
    pub model: Option<String>,

    /// Optional reasoning/thinking effort for providers and models that support it.
    pub effort: Option<ModelEffort>,
}

impl CompletionRequest {
    /// Adds a document to the request.
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.docs.push(Document::from_text(&id, &text));
        self
    }

    /// Adds multiple documents to the request.
    pub fn append_documents(mut self, docs: Documents) -> Self {
        self.documents.docs.extend(docs.docs);
        self
    }

    /// Adds multiple tools to the request.
    pub fn append_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }
}
