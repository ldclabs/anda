use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, ContentPart,
    FunctionDefinition, Json, Path, PutMode, Resource, StoreFeatures, Tool, ToolOutput,
    select_resources, validate_function_name,
};
use ciborium::from_reader;
use ic_auth_types::{Xid, deterministic_cbor_into_vec};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    sync::Arc,
};

use super::{AgentCtx, BaseCtx};
use crate::hook::{AgentHook, DynAgentHook};

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct SubAgent {
    pub name: String,
    pub description: String,
    pub instructions: String,

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default)]
    pub background: bool,

    pub output_schema: Option<Json>,
}

impl Agent<AgentCtx> for SubAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "description": "Run this subagent on a focused task. Provide a self-contained prompt with the goal, relevant context, constraints, and expected output.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task for this subagent. Include the objective, relevant context, constraints, preferred workflow or deliverable, and any success criteria needed to complete the work.",
                        "minLength": 1
                    },
                },
                "required": ["prompt"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.tools.clone()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.tags.clone()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let hook = ctx.base.get_state::<DynAgentHook>();

        let (prompt, resources) = if let Some(hook) = &hook {
            hook.before_agent_run(&ctx, prompt, resources).await?
        } else {
            (prompt, resources)
        };

        let req = CompletionRequest {
            instructions: self.instructions.clone(),
            prompt,
            content: resources.into_iter().map(ContentPart::from).collect(),
            tools: ctx.definitions(Some(&self.tools)).await,
            output_schema: self.output_schema.clone(),
            ..Default::default()
        };

        if self.background {
            let task_id = format!("{}:{}", self.name(), Xid::new());
            let rt = AgentOutput {
                content: format!(
                    "subagent is running in the background with task ID: {}",
                    task_id
                ),
                ..Default::default()
            };

            let rt = if let Some(hook) = &hook {
                hook.after_agent_run(&ctx, rt).await?
            } else {
                rt
            };

            if let Some(hook) = &hook {
                hook.on_background_start(&ctx, &task_id, &req).await;
            }

            tokio::spawn(async move {
                let mut rt = match ctx.completion(req, Vec::new()).await {
                    Ok(rt) => rt,
                    Err(err) => AgentOutput {
                        content: format!("subagent background task {} error: {}", task_id, err),
                        ..Default::default()
                    },
                };
                rt.content = format!(
                    "subagent background task {} completed with output:\n\n{}",
                    task_id, rt.content
                );

                if let Some(hook) = hook {
                    hook.on_background_end(ctx, task_id, rt).await;
                }
            });
            Ok(rt)
        } else {
            let rt = ctx.completion(req, Vec::new()).await?;
            if let Some(hook) = &hook {
                return hook.after_agent_run(&ctx, rt).await;
            }

            Ok(rt)
        }
    }
}

pub trait SubAgentSet: Send + Sync {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Checks if a subagent with the given lowercase name exists.
    fn contains_lowercase(&self, lowercase_name: &str) -> bool;

    /// Retrieves a subagent by lowercase name.
    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent>;

    /// Returns definitions for all or specified agents.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// - Vec<[`FunctionDefinition`]>: Vector of agent definitions. The name in each definition is prefixed with "SA_" to avoid conflicts and indicate it's a subagent.
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Selects and returns resources relevant to the specified subagent name from the provided list.
    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource>;
}

pub struct SubAgentManager {
    agents: RwLock<BTreeMap<String, SubAgent>>,
}

impl Default for SubAgentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentManager {
    pub const NAME: &'static str = "subagents_manager";

    pub fn new() -> Self {
        Self {
            agents: RwLock::new(BTreeMap::new()),
        }
    }

    pub async fn load(&self, ctx: BaseCtx) -> Result<(), BoxError> {
        let offset = Path::from("");
        if let Ok(agents) = ctx.store_list(None, &offset).await {
            for meta in agents {
                let (data, _) = ctx.store_get(&meta.location).await?;
                if let Ok(agent) = from_reader::<SubAgent, _>(&data[..]) {
                    self.agents
                        .write()
                        .insert(agent.name.to_ascii_lowercase(), agent);
                }
            }
        };

        Ok(())
    }

    /// Creates or updates a subagent. The name is normalised to lowercase and validated. If an agent with the same name exists, it will be overwritten.
    pub async fn upsert(&self, ctx: BaseCtx, agent: SubAgent) -> Result<(), BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;

        let data = deterministic_cbor_into_vec(&agent)?;
        self.agents.write().insert(name.clone(), agent);

        ctx.store_put(&Path::from(name), PutMode::Overwrite, data.into())
            .await?;
        Ok(())
    }
}

impl SubAgentSet for SubAgentManager {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.agents.read().contains_key(lowercase_name)
    }

    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent> {
        self.agents.read().get(lowercase_name).cloned()
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        match names {
            None => self
                .agents
                .read()
                .values()
                .map(|agent| agent.definition().name_with_prefix("SA_"))
                .collect(),
            Some(names) => {
                let agents = self.agents.read();
                names
                    .iter()
                    .filter_map(|name| {
                        agents
                            .get(&name.to_ascii_lowercase())
                            .map(|agent| agent.definition().name_with_prefix("SA_"))
                    })
                    .collect()
            }
        }
    }

    fn select_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        if let Some(name) = prefixed_name.strip_prefix("SA_") {
            self.agents
                .read()
                .get(&name.to_ascii_lowercase())
                .map(|agent| {
                    let supported_tags = agent.supported_resource_tags();
                    select_resources(resources, &supported_tags)
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

pub struct SubAgentSetManager {
    sets: RwLock<BTreeMap<TypeId, Arc<dyn SubAgentSet>>>,
}

impl Default for SubAgentSetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentSetManager {
    pub fn new() -> Self {
        Self {
            sets: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn insert<T: SubAgentSet + Sized + 'static>(&self, set: Arc<T>) -> Option<Arc<T>> {
        let type_id = TypeId::of::<T>();
        self.sets
            .write()
            .insert(type_id, set)
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
    }

    pub fn get<T: SubAgentSet + Sized + 'static>(&self) -> Option<Arc<T>> {
        let type_id = TypeId::of::<T>();
        self.sets
            .read()
            .get(&type_id)
            .and_then(|boxed| boxed.clone().into_any().downcast::<T>().ok())
    }
}

impl SubAgentSet for SubAgentSetManager {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn contains_lowercase(&self, lowercase_name: &str) -> bool {
        self.sets
            .read()
            .values()
            .any(|set| set.contains_lowercase(lowercase_name))
    }

    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent> {
        for set in self.sets.read().values() {
            if let Some(agent) = set.get_lowercase(lowercase_name) {
                return Some(agent);
            }
        }
        None
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        self.sets
            .read()
            .values()
            .flat_map(|set| set.definitions(names))
            .collect()
    }

    fn select_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        if prefixed_name.starts_with("SA_") {
            for set in self.sets.read().values() {
                let selected = set.select_resources(prefixed_name, resources);
                if !selected.is_empty() {
                    return selected;
                }
            }
        }
        Vec::new()
    }
}

impl Tool<BaseCtx> for SubAgentManager {
    type Args = SubAgent;
    type Output = Json;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Create or update a reusable subagent for a specific scenario. Use this when a task would benefit from a dedicated role, stable instructions, or a restricted toolset. The subagent becomes callable later by its name and can handle repeated, domain-specific, or multi-step work with its own instructions and optional tool whitelist.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "description": "Create or update a reusable subagent configuration.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique callable subagent name. Must be lowercase snake_case, start with a letter, contain only letters, digits, or underscores, and be no longer than 64 characters. Choose a short, task-oriented name such as 'research_assistant' or 'tweet_writer'.",
                        "pattern": "^[a-z][a-z0-9_]{0,63}$"
                    },
                    "description": {
                        "type": "string",
                        "description": "Short routing description shown to the model when deciding whether to call this subagent. State when it should be used and what outcome it produces.",
                        "minLength": 1
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Durable system-style instructions for the subagent. Define its role, scope, workflow, constraints, decision rules, and expected output style. Write reusable guidance, not a one-off task prompt.",
                        "minLength": 1
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional whitelist of tool names the subagent may use. Include only the minimum tools it needs. Leave empty or omit this field to create a no-tool subagent.",
                        "default": [],
                        "uniqueItems": true
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of resource tags that are relevant to this subagent. When the subagent is called, resources with matching tags will be prioritized for selection and passed to the subagent in order of relevance.",
                        "default": [],
                        "uniqueItems": true
                    },
                    "output_schema": {
                        "type": "object",
                        "description": "Optional JSON schema that the subagent's output must conform to. If provided, the model will be guided to produce output in the specified format, and outputs that don't match the schema may be rejected or cause an error. Use this to enforce structured output, such as a JSON object with specific fields, when the subagent's response needs to be machine-readable or follow a strict format."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Whether this subagent should be executed asynchronously in the background. If true, the agent will return immediately with a message containing a unique task ID, and the final output will be passed to the `on_background_end` hook method when the task is completed. Use this for long-running tasks that don't need to return output immediately, or when you want to handle the final output separately in the hook (e.g. by storing it in a database or sending a notification)."
                    }
                },
                "required": ["name", "description", "instructions"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn init(&self, ctx: BaseCtx) -> Result<(), BoxError> {
        self.load(ctx).await
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        self.upsert(ctx, args).await?;
        Ok(ToolOutput::new(json!({"result": "success"})))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use anda_core::RequestMeta;
    use serde_json::json;

    #[test]
    fn subagent_definition_guides_self_contained_prompts() {
        let agent = SubAgent {
            name: "research_assistant".to_string(),
            description: "Handles recurring research tasks with concise synthesis.".to_string(),
            instructions: "Research carefully and synthesize findings.".to_string(),
            tools: vec!["google_web_search".to_string()],
            ..Default::default()
        };

        let definition = agent.definition();

        assert_eq!(definition.name, "research_assistant");
        assert_eq!(
            definition.description,
            "Handles recurring research tasks with concise synthesis."
        );
        assert_eq!(
            definition.parameters["description"],
            json!(
                "Run this subagent on a focused task. Provide a self-contained prompt with the goal, relevant context, constraints, and expected output."
            )
        );
        assert_eq!(
            definition.parameters["properties"]["prompt"]["minLength"],
            json!(1)
        );
        assert_eq!(definition.parameters["additionalProperties"], json!(false));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subagents_tool_definition_guides_reusable_configs_and_normalizes_names() {
        let engine = EngineBuilder::new().empty().await.unwrap();
        let ctx = engine
            .base_ctx_with(
                engine.id(),
                "",
                SubAgentManager::NAME,
                RequestMeta::default(),
            )
            .unwrap();
        let tool: Arc<SubAgentManager> = engine.sub_agents_manager().get().unwrap();

        let definition = tool.definition();

        assert_eq!(definition.name, "subagents_manager");
        assert!(definition.description.contains("reusable subagent"));
        assert_eq!(
            definition.parameters["properties"]["name"]["pattern"],
            json!("^[a-z][a-z0-9_]{0,63}$")
        );
        assert_eq!(
            definition.parameters["properties"]["tools"]["default"],
            json!([])
        );
        assert_eq!(definition.parameters["additionalProperties"], json!(false));

        tool.upsert(
            ctx,
            SubAgent {
                name: "Research_Assistant".to_string(),
                description: "Handles recurring research tasks with concise synthesis.".to_string(),
                instructions: "Research carefully and synthesize findings.".to_string(),
                tools: vec!["google_web_search".to_string()],
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let agent = tool.get_lowercase("research_assistant").unwrap();
        assert_eq!(agent.name, "Research_Assistant");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn load_restores_all_persisted_subagents() {
        let engine = EngineBuilder::new().empty().await.unwrap();
        let ctx = engine
            .base_ctx_with(
                engine.id(),
                "",
                SubAgentManager::NAME,
                RequestMeta::default(),
            )
            .unwrap();
        let tool: Arc<SubAgentManager> = engine.sub_agents_manager().get().unwrap();

        let agents = vec![
            SubAgent {
                name: "Research_Assistant".to_string(),
                description: "Handles recurring research tasks with concise synthesis.".to_string(),
                instructions: "Research carefully and synthesize findings.".to_string(),
                tools: vec!["google_web_search".to_string()],
                tags: vec!["research".to_string()],
                ..Default::default()
            },
            SubAgent {
                name: "code_reviewer".to_string(),
                description: "Reviews code for correctness and risks.".to_string(),
                instructions: "Review code changes and summarize findings.".to_string(),
                tools: vec!["read_file".to_string(), "grep_search".to_string()],
                tags: vec!["code".to_string(), "review".to_string()],
                background: true,
                ..Default::default()
            },
            SubAgent {
                name: "writer_helper".to_string(),
                description: "Drafts concise written content.".to_string(),
                instructions: "Write clearly and keep the response concise.".to_string(),
                tags: vec!["writing".to_string()],
                output_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" }
                    },
                    "required": ["summary"],
                    "additionalProperties": false
                })),
                ..Default::default()
            },
        ];

        for agent in agents.clone() {
            tool.upsert(ctx.clone(), agent).await.unwrap();
        }

        let stored = ctx.store_list(None, &Path::from("")).await.unwrap();
        assert_eq!(stored.len(), agents.len());

        for meta in &stored {
            let (data, _) = ctx.store_get(&meta.location).await.unwrap();
            let loaded = from_reader::<SubAgent, _>(&data[..]).unwrap();
            assert!(agents.iter().any(|agent| agent.name == loaded.name));
        }

        let reloaded = SubAgentManager::new();
        reloaded.load(ctx).await.unwrap();

        assert_eq!(reloaded.definitions(None).len(), agents.len());

        for expected in agents {
            let loaded = reloaded
                .get_lowercase(&expected.name.to_ascii_lowercase())
                .unwrap();

            assert_eq!(loaded.name, expected.name);
            assert_eq!(loaded.description, expected.description);
            assert_eq!(loaded.instructions, expected.instructions);
            assert_eq!(loaded.tools, expected.tools);
            assert_eq!(loaded.tags, expected.tags);
            assert_eq!(loaded.background, expected.background);
            assert_eq!(loaded.output_schema, expected.output_schema);
        }
    }
}
