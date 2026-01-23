use std::collections::BTreeMap;

use crate::{
    AgentOutput, BoxError, ContentPart, Document, Documents, FunctionDefinition, Json, Message,
    Resource,
};

/// Provides LLM completion capabilities for agents.
pub trait CompletionFeatures: Sized {
    /// Generates a completion based on the given request and optional resources.
    fn completion(
        &self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;

    /// Returns the name of the model.
    fn model_name(&self) -> String;
}

/// Represents a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// The system instructions to be sent to the completion model provider, as the "system" role.
    pub instructions: String,

    /// The name of role, defaulting to "user".
    pub role: Option<String>,

    /// The chat history to be sent to the completion model provider.
    pub chat_history: Vec<Message>,

    /// raw_history is the model specialized history used by anda_engine.
    /// It should be empty in most cases.
    pub raw_history: Vec<Json>,

    /// The documents to embed into the prompt.
    pub documents: Documents,

    /// The prompt to be sent to the completion model provider as role
    /// It can be empty.
    pub prompt: String,

    /// The content parts to be sent to the completion model provider.
    /// It can be empty.
    pub content: Vec<ContentPart>,

    /// The tools to be sent to the completion model provider.
    pub tools: Vec<FunctionDefinition>,

    /// Whether the tool choice is required.
    pub tool_choice_required: bool,

    /// The temperature to be sent to the completion model provider. [0.0, 2.0]
    pub temperature: Option<f64>,

    /// An upper bound for the number of tokens that can be generated for a response,
    pub max_output_tokens: Option<usize>,

    /// An object specifying the JSON format that the model must output.
    pub output_schema: Option<Json>,

    /// The stop sequence to be sent to the completion model provider.
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    /// Adds a document to the request.
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.docs.push(Document {
            content: text.into(),
            metadata: BTreeMap::from([("id".to_string(), id.into())]),
        });
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
