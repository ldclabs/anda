use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest,
    FunctionDefinition, PutMode, Resource, StoreFeatures, Tool, ToolOutput, validate_function_name,
};
use ciborium::from_reader;
use ic_auth_types::deterministic_cbor_into_vec;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, future::Future, sync::Arc};

use super::{AgentCtx, BaseCtx};
use crate::hook::Hook;

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct SubAgent {
    pub name: String,
    pub description: String,
    pub instructions: String,

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(skip)]
    pub hook: Option<Arc<dyn Hook>>,
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
            name: self.name().to_ascii_lowercase(),
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

    fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send {
        let agent = self.clone();
        Box::pin(async move {
            if let Some(hook) = &agent.hook {
                hook.on_agent_start(&ctx, &agent.name).await?;
            }

            let req = CompletionRequest {
                instructions: agent.instructions.clone(),
                prompt,
                tools: if agent.tools.is_empty() {
                    vec![]
                } else {
                    ctx.tool_definitions(Some(
                        &agent.tools.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
                    ))
                },
                ..Default::default()
            };

            let rt = ctx.completion(req, resources).await?;
            if let Some(hook) = &agent.hook {
                return hook.on_agent_end(&ctx, &agent.name, rt).await;
            }

            Ok(rt)
        })
    }
}

pub struct SubAgents {
    ctx: BaseCtx,
    agents: RwLock<BTreeMap<String, SubAgent>>,
    hook: Option<Arc<dyn Hook>>,
}

impl SubAgents {
    pub const NAME: &'static str = "subagents";

    pub fn new(ctx: BaseCtx, hook: Option<Arc<dyn Hook>>) -> Self {
        Self {
            ctx,
            agents: RwLock::new(BTreeMap::new()),
            hook,
        }
    }

    pub async fn load(&mut self) -> Result<(), BoxError> {
        if let Ok((data, _)) = self.ctx.store_get(&Self::NAME.into()).await {
            let agents: BTreeMap<String, SubAgent> = from_reader(&data[..])?;
            self.agents = RwLock::new(agents);
        };

        Ok(())
    }

    // Checks if a sub-agent with given name (should be lowercase) exists.
    pub fn has(&self, lowercase_name: &str) -> bool {
        self.agents.read().contains_key(lowercase_name)
    }

    pub fn get(&self, lowercase_name: &str) -> Option<SubAgent> {
        self.agents
            .read()
            .get(lowercase_name)
            .cloned()
            .map(|mut agent| {
                agent.hook = self.hook.clone();
                agent
            })
    }

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
        self.ctx
            .store_put(&SubAgents::NAME.into(), PutMode::Overwrite, data.into())
            .await?;
        Ok(())
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
        self.agents
            .read()
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
}

impl Tool<BaseCtx> for SubAgents {
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
            hook: None,
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
        let ctx = EngineBuilder::new().mock_ctx();
        let tool = SubAgents::new(ctx.base.clone(), None);

        let definition = tool.definition();

        assert_eq!(definition.name, "subagents");
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
            hook: None,
        })
        .await
        .unwrap();

        let agent = tool.get("research_assistant").unwrap();
        assert_eq!(agent.name, "research_assistant");
    }
}
