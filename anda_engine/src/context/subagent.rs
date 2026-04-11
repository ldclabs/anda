use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, ContentPart,
    FunctionDefinition, PutMode, Resource, StoreFeatures, Tool, ToolOutput, select_resources,
    validate_function_name,
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
use crate::hook::AgentHook;

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

    #[serde(skip)]
    pub hook: Option<Arc<dyn AgentHook>>,
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
                "description": "Run this sub-agent on a focused task. Provide a self-contained prompt with the goal, relevant context, constraints, and expected output.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task for this sub-agent. Include the objective, relevant context, constraints, preferred workflow or deliverable, and any success criteria needed to complete the work.",
                        "minLength": 1
                    },
                },
                "required": ["prompt"],
                "additionalProperties": false
            }),
            strict: None,
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
        let (prompt, resources) = if let Some(hook) = &self.hook {
            hook.before_agent_run(&ctx, prompt, resources).await?
        } else {
            (prompt, resources)
        };

        let req = CompletionRequest {
            instructions: self.instructions.clone(),
            prompt,
            content: resources.into_iter().map(ContentPart::from).collect(),
            tools: if self.tools.is_empty() {
                vec![]
            } else {
                ctx.definitions(Some(&self.tools)).await
            },
            ..Default::default()
        };

        if self.background {
            let task_id = format!("{}:{}", self.name(), Xid::new());
            let rt = AgentOutput {
                content: format!(
                    "sub-agent is running in the background with task ID: {}",
                    task_id
                ),
                ..Default::default()
            };

            let rt = if let Some(hook) = &self.hook {
                hook.after_agent_run(&ctx, rt).await?
            } else {
                rt
            };

            let hook = self.hook.clone();
            tokio::spawn(async move {
                let mut rt = match ctx.completion(req, Vec::new()).await {
                    Ok(rt) => rt,
                    Err(err) => AgentOutput {
                        content: format!("sub-agent background task {} error: {}", task_id, err),
                        ..Default::default()
                    },
                };
                rt.content = format!(
                    "sub-agent background task {} completed with output:\n\n{}",
                    task_id, rt.content
                );

                if let Some(hook) = hook {
                    hook.on_background_end(ctx, rt).await;
                }
            });
            Ok(rt)
        } else {
            let rt = ctx.completion(req, Vec::new()).await?;
            if let Some(hook) = &self.hook {
                return hook.after_agent_run(&ctx, rt).await;
            }

            Ok(rt)
        }
    }
}

pub trait SubAgentSet: Send + Sync {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Checks if a sub-agent with the given lowercase name exists.
    fn contains_lowercase(&self, lowercase_name: &str) -> bool;

    /// Retrieves a sub-agent by lowercase name.
    fn get_lowercase(&self, lowercase_name: &str) -> Option<SubAgent>;

    /// Returns definitions for all or specified agents.
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by.
    ///
    /// # Returns
    /// - Vec<[`FunctionDefinition`]>: Vector of agent definitions. The name in each definition is prefixed with "SA_" to avoid conflicts and indicate it's a sub-agent.
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Selects and returns resources relevant to the specified sub-agent name from the provided list.
    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource>;
}

pub struct SubAgentManager {
    root_ctx: BaseCtx,
    agents: RwLock<BTreeMap<String, SubAgent>>,
    hook: Option<Arc<dyn AgentHook>>,
}

impl SubAgentManager {
    pub const NAME: &'static str = "subagents_manager";

    pub fn new(root_ctx: BaseCtx, hook: Option<Arc<dyn AgentHook>>) -> Self {
        Self {
            root_ctx,
            agents: RwLock::new(BTreeMap::new()),
            hook,
        }
    }

    pub async fn load(&self) -> Result<(), BoxError> {
        if let Ok((data, _)) = self.root_ctx.store_get(&Self::NAME.into()).await {
            let agents: BTreeMap<String, SubAgent> = from_reader(&data[..])?;
            *self.agents.write() = agents;
        };

        Ok(())
    }

    /// Creates or updates a sub-agent. The name is normalised to lowercase and validated. If an agent with the same name exists, it will be overwritten.
    pub async fn upsert(&self, agent: SubAgent) -> Result<(), BoxError> {
        let mut agent = agent;
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;
        agent.name = name.clone();

        let data = {
            let mut agents = self.agents.write();
            agents.insert(name, agent);
            deterministic_cbor_into_vec(&*agents)?
        };
        self.root_ctx
            .store_put(
                &SubAgentManager::NAME.into(),
                PutMode::Overwrite,
                data.into(),
            )
            .await?;
        Ok(())
    }

    /// Appends new sub-agents without overwriting existing ones. Invalid or duplicate names are skipped.
    pub async fn try_append(&self, new_agents: Vec<SubAgent>) -> Result<(), BoxError> {
        let data = {
            let mut agents = self.agents.write();
            for mut agent in new_agents {
                agent.name = agent.name.to_ascii_lowercase();
                if validate_function_name(&agent.name).is_err() || agents.contains_key(&agent.name)
                {
                    continue;
                }
                agents.insert(agent.name.clone(), agent);
            }
            deterministic_cbor_into_vec(&*agents)?
        };

        self.root_ctx
            .store_put(
                &SubAgentManager::NAME.into(),
                PutMode::Overwrite,
                data.into(),
            )
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
        self.agents
            .read()
            .get(lowercase_name)
            .cloned()
            .map(|mut agent| {
                agent.hook = self.hook.clone();
                agent
            })
    }

    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition> {
        let names: Option<Vec<String>> =
            names.map(|names| names.iter().map(|n| n.to_ascii_lowercase()).collect());
        self.agents
            .read()
            .iter()
            .filter_map(|(name, agent)| match &names {
                Some(names) => {
                    if names.contains(name) {
                        Some(agent.definition().name_with_prefix("SA_"))
                    } else {
                        None
                    }
                }
                None => Some(agent.definition().name_with_prefix("SA_")),
            })
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
    type Output = String;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Create or update a reusable sub-agent for a specific scenario. Use this when a task would benefit from a dedicated role, stable instructions, or a restricted toolset. The sub-agent becomes callable later by its name and can handle repeated, domain-specific, or multi-step work with its own instructions and optional tool whitelist.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "description": "Create or update a reusable sub-agent configuration.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique callable sub-agent name. Must be lowercase snake_case, start with a letter, contain only letters, digits, or underscores, and be no longer than 64 characters. Choose a short, task-oriented name such as 'research_assistant' or 'tweet_writer'.",
                        "pattern": "^[a-z][a-z0-9_]{0,63}$"
                    },
                    "description": {
                        "type": "string",
                        "description": "Short routing description shown to the model when deciding whether to call this sub-agent. State when it should be used and what outcome it produces.",
                        "minLength": 1
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Durable system-style instructions for the sub-agent. Define its role, scope, workflow, constraints, decision rules, and expected output style. Write reusable guidance, not a one-off task prompt.",
                        "minLength": 1
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional whitelist of tool names the sub-agent may use. Include only the minimum tools it needs. Leave empty or omit this field to create a no-tool sub-agent.",
                        "default": [],
                        "uniqueItems": true
                    }
                },
                "required": ["name", "description", "instructions"],
                "additionalProperties": false
            }),
            strict: None,
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        self.upsert(args).await?;
        Ok(ToolOutput::new("Success".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
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
                "Run this sub-agent on a focused task. Provide a self-contained prompt with the goal, relevant context, constraints, and expected output."
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
        let engine = EngineBuilder::new().empty();
        let tool: Arc<SubAgentManager> = engine.sub_agents_manager().get().unwrap();

        let definition = tool.definition();

        assert_eq!(definition.name, "subagents_manager");
        assert!(definition.description.contains("reusable sub-agent"));
        assert_eq!(
            definition.parameters["properties"]["name"]["pattern"],
            json!("^[a-z][a-z0-9_]{0,63}$")
        );
        assert_eq!(
            definition.parameters["properties"]["tools"]["default"],
            json!([])
        );
        assert_eq!(definition.parameters["additionalProperties"], json!(false));

        tool.upsert(SubAgent {
            name: "Research_Assistant".to_string(),
            description: "Handles recurring research tasks with concise synthesis.".to_string(),
            instructions: "Research carefully and synthesize findings.".to_string(),
            tools: vec!["google_web_search".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

        let agent = tool.get_lowercase("research_assistant").unwrap();
        assert_eq!(agent.name, "research_assistant");
    }
}
