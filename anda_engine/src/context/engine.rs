use anda_core::{
    Agent, AgentContext, AgentInput, AgentOutput, BaseContext, BoxError, Function,
    FunctionDefinition, HttpFeatures, Json, Resource, Tool, ToolInput, ToolOutput,
    select_resources, validate_function_name,
};
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub use anda_cloud_cdk::AgentInfo;

use crate::context::{AgentCtx, BaseCtx};

/// Information about the engine, including agent and tool definitions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EngineCard {
    /// The principal ID of the engine.
    pub id: Principal,
    /// Information about the agent, including name, description, and supported protocols.
    pub info: AgentInfo,
    /// Definitions for agents in the engine.
    pub agents: Vec<Function>,
    /// Definitions for tools in the engine.
    pub tools: Vec<Function>,
}

/// Collection of remote engines.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteEngines {
    pub engines: BTreeMap<String, EngineCard>,
}

/// Arguments for registering a remote engine.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteEngineArgs {
    /// The endpoint of the remote engine.
    pub endpoint: String,
    /// List of agents to include in the engine. If empty, all agents are included.
    pub agents: Vec<String>,
    /// List of tools to include in the engine. If empty, all tools are included.
    pub tools: Vec<String>,
    /// Optional handle for the engine. If not provided, the engine handle is used.
    pub handle: Option<String>,
}

impl Default for RemoteEngines {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoteEngines {
    pub fn new() -> Self {
        Self {
            engines: BTreeMap::new(),
        }
    }

    /// Registers a remote engine with the given arguments.
    pub async fn register(
        &mut self,
        ctx: impl HttpFeatures,
        args: RemoteEngineArgs,
    ) -> Result<(), BoxError> {
        let mut engine: EngineCard = ctx
            .https_signed_rpc(&args.endpoint, "information", &(true,))
            .await?;
        let handle = args
            .handle
            .unwrap_or_else(|| engine.info.handle.to_ascii_lowercase());
        validate_function_name(&handle)
            .map_err(|err| format!("invalid engine handle {:?}: {}", &handle, err))?;

        if !args.agents.is_empty() {
            let agents: Vec<Function> = engine
                .agents
                .into_iter()
                .filter(|d| args.agents.contains(&d.definition.name))
                .collect();
            for agent in args.agents {
                if !agents.iter().any(|d| d.definition.name == agent) {
                    return Err(
                        format!("agent {:?} not found in engine {:?}", agent, handle).into(),
                    );
                }
            }

            engine.agents = agents;
        }

        if !args.tools.is_empty() {
            let tools: Vec<Function> = engine
                .tools
                .into_iter()
                .filter(|d| args.tools.is_empty() || args.tools.contains(&d.definition.name))
                .collect();
            for tool in args.tools {
                if !tools.iter().any(|d| d.definition.name == tool) {
                    return Err(format!("tool {:?} not found in engine {:?}", tool, handle).into());
                }
            }
            engine.tools = tools;
        }

        self.engines.insert(handle, engine);
        Ok(())
    }

    /// Retrieves a remote tool endpoint and name from a prefixed name.
    pub fn get_tool_endpoint(&self, name: &str) -> Option<(Principal, String, String)> {
        self.engines
            .iter()
            .filter_map(|(handle, engine)| {
                let tool_name = name.strip_prefix(&format!("{handle}_"))?;
                engine
                    .tools
                    .iter()
                    .any(|tool| tool.definition.name == tool_name)
                    .then_some((handle.len(), engine, tool_name))
            })
            .max_by_key(|(handle_len, _, _)| *handle_len)
            .map(|(_, engine, tool_name)| {
                (
                    engine.id,
                    engine.info.endpoint.clone(),
                    tool_name.to_string(),
                )
            })
    }

    /// Retrieves a remote agent endpoint and name from a prefixed name.
    pub fn get_agent_endpoint(&self, name: &str) -> Option<(Principal, String, String)> {
        self.engines
            .iter()
            .filter_map(|(handle, engine)| {
                let agent_name = name.strip_prefix(&format!("{handle}_"))?;
                engine
                    .agents
                    .iter()
                    .any(|agent| agent.definition.name == agent_name)
                    .then_some((handle.len(), engine, agent_name))
            })
            .max_by_key(|(handle_len, _, _)| *handle_len)
            .map(|(_, engine, agent_name)| {
                (
                    engine.id,
                    engine.info.endpoint.clone(),
                    agent_name.to_string(),
                )
            })
    }

    /// Retrieves a remote engine ID by endpoint.
    pub fn get_id_by_endpoint(&self, endpoint: &str) -> Option<Principal> {
        for (_, engine) in self.engines.iter() {
            if engine.info.endpoint == endpoint {
                return Some(engine.id);
            }
        }
        None
    }

    /// Retrieves a remote engine endpoint by ID.
    pub fn get_endpoint_by_id(&self, id: &Principal) -> Option<String> {
        for (_, engine) in self.engines.iter() {
            if &engine.id == id {
                return Some(engine.info.endpoint.clone());
            }
        }
        None
    }

    /// Retrieves definitions for available tools in the remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint
    /// * `names` - Optional filter for specific tool names
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools
    pub fn tool_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[String]>,
    ) -> Vec<FunctionDefinition> {
        if let Some(endpoint) = endpoint {
            for (handle, engine) in self.engines.iter() {
                if endpoint == engine.info.endpoint {
                    let prefix = format!("{handle}_");
                    return engine
                        .tools
                        .iter()
                        .filter_map(|d| {
                            if let Some(names) = names {
                                if names.contains(&d.definition.name) {
                                    Some(d.definition.clone().name_with_prefix(&prefix))
                                } else {
                                    None
                                }
                            } else {
                                Some(d.definition.clone().name_with_prefix(&prefix))
                            }
                        })
                        .collect();
                }
            }
        }

        let mut definitions =
            Vec::with_capacity(self.engines.values().map(|e| e.tools.len()).sum());

        for (handle, engine) in self.engines.iter() {
            let prefix = format!("{handle}_");
            definitions.extend(engine.tools.iter().filter_map(|d| {
                if let Some(names) = names {
                    if names.contains(&d.definition.name) {
                        Some(d.definition.clone().name_with_prefix(&prefix))
                    } else {
                        None
                    }
                } else {
                    Some(d.definition.clone().name_with_prefix(&prefix))
                }
            }));
        }

        definitions
    }

    /// Extracts resources from the provided list based on the tool's supported tags.
    pub fn select_tool_resources(
        &self,
        prefixed_name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        for (handle, engine) in self.engines.iter() {
            if let Some(name) = prefixed_name.strip_prefix(&format!("{handle}_")) {
                for tool in engine.tools.iter() {
                    if tool.definition.name.eq_ignore_ascii_case(name) {
                        return select_resources(resources, &tool.supported_resource_tags);
                    }
                }
            }
        }

        Vec::new()
    }

    /// Retrieves definitions for available agents in the remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint
    /// * `names` - Optional filter for specific agent names
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents
    pub fn agent_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[String]>,
    ) -> Vec<FunctionDefinition> {
        if let Some(endpoint) = endpoint {
            for (handle, engine) in self.engines.iter() {
                if endpoint == engine.info.endpoint {
                    let prefix = format!("{handle}_");
                    return engine
                        .agents
                        .iter()
                        .filter_map(|d| {
                            if let Some(names) = names {
                                if names.contains(&d.definition.name) {
                                    Some(d.definition.clone().name_with_prefix(&prefix))
                                } else {
                                    None
                                }
                            } else {
                                Some(d.definition.clone().name_with_prefix(&prefix))
                            }
                        })
                        .collect();
                }
            }
        }

        let mut definitions =
            Vec::with_capacity(self.engines.values().map(|e| e.agents.len()).sum());
        for (handle, engine) in self.engines.iter() {
            let prefix = format!("{handle}_");
            definitions.extend(engine.agents.iter().filter_map(|d| {
                if let Some(names) = names {
                    if names.contains(&d.definition.name) {
                        Some(d.definition.clone().name_with_prefix(&prefix))
                    } else {
                        None
                    }
                } else {
                    Some(d.definition.clone().name_with_prefix(&prefix))
                }
            }));
        }

        definitions
    }

    /// Extracts resources from the provided list based on the agent's supported tags.
    pub fn select_agent_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        for (handle, engine) in self.engines.iter() {
            if let Some(name) = name.strip_prefix(&format!("{handle}_")) {
                for agent in engine.agents.iter() {
                    if agent.definition.name.eq_ignore_ascii_case(name) {
                        return select_resources(resources, &agent.supported_resource_tags);
                    }
                }
            }
        }

        Vec::new()
    }
}

/// Wraps a remote tool as a local tool.
#[derive(Debug, Clone)]
pub struct RemoteTool {
    engine: Principal,
    endpoint: String,
    function: Function,
    name: String,
}

impl RemoteTool {
    pub fn new(
        engine: Principal,
        endpoint: String,
        function: Function,
        name: Option<String>,
    ) -> Result<Self, BoxError> {
        let name = if let Some(name) = name {
            validate_function_name(&name)?;
            name
        } else {
            function.definition.name.clone()
        };

        Ok(Self {
            engine,
            endpoint,
            function,
            name,
        })
    }
}

impl Tool<BaseCtx> for RemoteTool {
    type Args = Json;
    type Output = Json;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.function.definition.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        let mut definition = self.function.definition.clone();
        definition.name = self.name.clone();
        definition
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.function.supported_resource_tags.clone()
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        ctx.remote_tool_call(
            &self.endpoint,
            ToolInput {
                name: self.function.definition.name.clone(),
                args,
                resources,
                meta: Some(ctx.self_meta(self.engine)),
            },
        )
        .await
    }
}

/// Wraps a remote agent as a local agent.
#[derive(Debug, Clone)]
pub struct RemoteAgent {
    engine: Principal,
    endpoint: String,
    function: Function,
    name: String,
}

impl RemoteAgent {
    pub fn new(
        engine: Principal,
        endpoint: String,
        function: Function,
        name: Option<String>,
    ) -> Result<Self, BoxError> {
        let name = if let Some(name) = name {
            validate_function_name(&name.to_ascii_lowercase())?;
            name
        } else {
            function.definition.name.clone()
        };

        Ok(Self {
            engine,
            endpoint,
            function,
            name,
        })
    }
}

impl Agent<AgentCtx> for RemoteAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.function.definition.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        let mut definition = self.function.definition.clone();
        definition.name = self.name.to_ascii_lowercase();
        definition
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.function.supported_resource_tags.clone()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        ctx.remote_agent_run(
            &self.endpoint,
            AgentInput {
                name: self.function.definition.name.clone(),
                prompt,
                resources,
                meta: Some(ctx.base.self_meta(self.engine)),
                ..Default::default()
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function(name: &str) -> Function {
        Function {
            definition: FunctionDefinition {
                name: name.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn engine(endpoint: &str, tools: &[&str], agents: &[&str]) -> EngineCard {
        EngineCard {
            id: Principal::anonymous(),
            info: AgentInfo {
                endpoint: endpoint.to_string(),
                ..Default::default()
            },
            tools: tools.iter().map(|name| function(name)).collect(),
            agents: agents.iter().map(|name| function(name)).collect(),
        }
    }

    #[test]
    fn remote_tool_endpoint_prefers_registered_function_on_longest_handle() {
        let mut remote = RemoteEngines::new();
        remote.engines.insert(
            "alpha".to_string(),
            engine("https://alpha.example", &["beta_tool", "status"], &[]),
        );
        remote.engines.insert(
            "alpha_beta".to_string(),
            engine("https://alpha-beta.example", &["tool"], &[]),
        );

        let (_, endpoint, tool_name) = remote.get_tool_endpoint("alpha_beta_tool").unwrap();
        assert_eq!(endpoint, "https://alpha-beta.example");
        assert_eq!(tool_name, "tool");

        let (_, endpoint, tool_name) = remote.get_tool_endpoint("alpha_status").unwrap();
        assert_eq!(endpoint, "https://alpha.example");
        assert_eq!(tool_name, "status");
        assert!(remote.get_tool_endpoint("alpha_missing").is_none());
    }

    #[test]
    fn remote_agent_endpoint_prefers_registered_function_on_longest_handle() {
        let mut remote = RemoteEngines::new();
        remote.engines.insert(
            "alpha".to_string(),
            engine("https://alpha.example", &[], &["beta_agent", "chat"]),
        );
        remote.engines.insert(
            "alpha_beta".to_string(),
            engine("https://alpha-beta.example", &[], &["agent"]),
        );

        let (_, endpoint, agent_name) = remote.get_agent_endpoint("alpha_beta_agent").unwrap();
        assert_eq!(endpoint, "https://alpha-beta.example");
        assert_eq!(agent_name, "agent");

        let (_, endpoint, agent_name) = remote.get_agent_endpoint("alpha_chat").unwrap();
        assert_eq!(endpoint, "https://alpha.example");
        assert_eq!(agent_name, "chat");
        assert!(remote.get_agent_endpoint("alpha_missing").is_none());
    }
}
