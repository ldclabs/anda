//! Remote engine registration and local wrappers.
//!
//! Remote engines expose agents and tools over the signed HTTP RPC protocol.
//! This module stores the discovered engine cards, maps prefixed function names
//! back to remote endpoints, and wraps remote functions as local [`Agent`] and
//! [`Tool`] implementations.

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
    /// Registered engine cards keyed by their lowercase handle.
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
    /// Creates an empty remote-engine registry.
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
                .filter(|d| args.tools.contains(&d.definition.name))
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

    fn strip_handle_prefix<'a>(name: &'a str, handle: &str) -> Option<&'a str> {
        name.strip_prefix(handle)?.strip_prefix('_')
    }

    /// Clones `function`'s definition with the engine `prefix` applied, when it
    /// passes the optional `names` filter.
    fn filter_definition(
        function: &Function,
        names: Option<&[String]>,
        prefix: &str,
    ) -> Option<FunctionDefinition> {
        names
            .is_none_or(|names| names.contains(&function.definition.name))
            .then(|| function.definition.clone().name_with_prefix(prefix))
    }

    /// Retrieves a remote tool endpoint and name from a prefixed name.
    pub fn get_tool_endpoint(&self, name: &str) -> Option<(Principal, String, String)> {
        self.engines
            .iter()
            .filter_map(|(handle, engine)| {
                let tool_name = Self::strip_handle_prefix(name, handle)?;
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
                let agent_name = Self::strip_handle_prefix(name, handle)?;
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
        let mut definitions = Vec::new();
        for (handle, engine) in self.engines.iter() {
            if endpoint.is_some_and(|endpoint| endpoint != engine.info.endpoint) {
                continue;
            }
            let prefix = format!("{handle}_");
            definitions.extend(
                engine
                    .tools
                    .iter()
                    .filter_map(|d| Self::filter_definition(d, names, &prefix)),
            );
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
            if let Some(name) = Self::strip_handle_prefix(prefixed_name, handle) {
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
        let mut definitions = Vec::new();
        for (handle, engine) in self.engines.iter() {
            if endpoint.is_some_and(|endpoint| endpoint != engine.info.endpoint) {
                continue;
            }
            let prefix = format!("{handle}_");
            definitions.extend(
                engine
                    .agents
                    .iter()
                    .filter_map(|d| Self::filter_definition(d, names, &prefix)),
            );
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
            if let Some(name) = Self::strip_handle_prefix(name, handle) {
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
    /// Creates a local wrapper around a tool exported by a remote engine.
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
    /// Creates a local wrapper around an agent exported by a remote engine.
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
    use serde::de::DeserializeOwned;
    use serde_json::json;

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

    fn tagged_function(name: &str, description: &str, tags: &[&str]) -> Function {
        Function {
            definition: FunctionDefinition {
                name: name.to_string(),
                description: description.to_string(),
                ..Default::default()
            },
            supported_resource_tags: tags.iter().map(|tag| tag.to_string()).collect(),
        }
    }

    fn resource(id: u64, tags: &[&str]) -> Resource {
        Resource {
            _id: id,
            name: format!("resource-{id}"),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        }
    }

    #[derive(Clone)]
    struct MockHttp {
        card: EngineCard,
    }

    impl HttpFeatures for MockHttp {
        async fn https_call(
            &self,
            _url: &str,
            _method: http::Method,
            _headers: Option<http::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            Err("unused https_call".into())
        }

        async fn https_signed_call(
            &self,
            _url: &str,
            _method: http::Method,
            _message_digest: [u8; 32],
            _headers: Option<http::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            Err("unused https_signed_call".into())
        }

        async fn https_signed_rpc<T>(
            &self,
            _endpoint: &str,
            method: &str,
            _args: impl Serialize + Send,
        ) -> Result<T, BoxError>
        where
            T: DeserializeOwned,
        {
            assert_eq!(method, "information");
            serde_json::from_value(serde_json::to_value(&self.card)?).map_err(|err| err.into())
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

    #[tokio::test]
    async fn remote_engine_register_filters_definitions_ids_and_resources() {
        let card = EngineCard {
            id: Principal::management_canister(),
            info: AgentInfo {
                handle: "RemoteMain".to_string(),
                endpoint: "https://remote.example".to_string(),
                ..Default::default()
            },
            agents: vec![
                tagged_function("chat", "Chat remotely", &["text"]),
                tagged_function("draft", "Draft remotely", &["md"]),
            ],
            tools: vec![
                tagged_function("lookup", "Lookup remotely", &["text"]),
                tagged_function("render", "Render remotely", &["image"]),
            ],
        };
        let mut remote = RemoteEngines::new();
        remote
            .register(
                MockHttp { card },
                RemoteEngineArgs {
                    endpoint: "https://remote.example".to_string(),
                    agents: vec!["chat".to_string()],
                    tools: vec!["lookup".to_string()],
                    handle: Some("remote".to_string()),
                },
            )
            .await
            .unwrap();

        assert_eq!(
            remote.get_id_by_endpoint("https://remote.example"),
            Some(Principal::management_canister())
        );
        assert_eq!(
            remote.get_endpoint_by_id(&Principal::management_canister()),
            Some("https://remote.example".to_string())
        );
        assert!(
            remote
                .get_id_by_endpoint("https://missing.example")
                .is_none()
        );
        assert!(remote.get_endpoint_by_id(&Principal::anonymous()).is_none());

        assert_eq!(
            remote
                .tool_definitions(Some("https://remote.example"), None)
                .into_iter()
                .map(|definition| definition.name)
                .collect::<Vec<_>>(),
            vec!["remote_lookup"]
        );
        assert!(
            remote
                .tool_definitions(Some("https://missing.example"), None)
                .is_empty()
        );
        assert_eq!(
            remote
                .tool_definitions(None, Some(&["lookup".to_string()]))
                .into_iter()
                .map(|definition| definition.name)
                .collect::<Vec<_>>(),
            vec!["remote_lookup"]
        );
        assert!(
            remote
                .tool_definitions(None, Some(&["render".to_string()]))
                .is_empty()
        );

        assert_eq!(
            remote
                .agent_definitions(Some("https://remote.example"), None)
                .into_iter()
                .map(|definition| definition.name)
                .collect::<Vec<_>>(),
            vec!["remote_chat"]
        );
        assert!(
            remote
                .agent_definitions(Some("https://missing.example"), None)
                .is_empty()
        );
        assert_eq!(
            remote
                .agent_definitions(None, Some(&["chat".to_string()]))
                .into_iter()
                .map(|definition| definition.name)
                .collect::<Vec<_>>(),
            vec!["remote_chat"]
        );
        assert!(
            remote
                .agent_definitions(None, Some(&["draft".to_string()]))
                .is_empty()
        );

        let mut resources = vec![resource(1, &["text"]), resource(2, &["image"])];
        let selected = remote.select_tool_resources("remote_lookup", &mut resources);
        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(resources[0]._id, 2);
        assert!(
            remote
                .select_tool_resources("remote_missing", &mut resources)
                .is_empty()
        );

        let mut resources = vec![resource(3, &["text"]), resource(4, &["md"])];
        let selected = remote.select_agent_resources("remote_chat", &mut resources);
        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![3]
        );
        assert_eq!(resources[0]._id, 4);
        assert!(
            remote
                .select_agent_resources("remote_missing", &mut resources)
                .is_empty()
        );
    }

    #[tokio::test]
    async fn remote_engine_register_reports_invalid_filters_and_handles() {
        let card = EngineCard {
            id: Principal::anonymous(),
            info: AgentInfo {
                handle: "remote".to_string(),
                endpoint: "https://remote.example".to_string(),
                ..Default::default()
            },
            agents: vec![function("chat")],
            tools: vec![function("lookup")],
        };

        let mut remote = RemoteEngines::new();
        let err = remote
            .register(
                MockHttp { card: card.clone() },
                RemoteEngineArgs {
                    endpoint: "https://remote.example".to_string(),
                    agents: vec!["missing".to_string()],
                    tools: Vec::new(),
                    handle: Some("remote".to_string()),
                },
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("agent \"missing\" not found"));

        let mut remote = RemoteEngines::new();
        let err = remote
            .register(
                MockHttp { card: card.clone() },
                RemoteEngineArgs {
                    endpoint: "https://remote.example".to_string(),
                    agents: Vec::new(),
                    tools: vec!["missing".to_string()],
                    handle: Some("remote".to_string()),
                },
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("tool \"missing\" not found"));

        let mut remote = RemoteEngines::new();
        let err = remote
            .register(
                MockHttp { card },
                RemoteEngineArgs {
                    endpoint: "https://remote.example".to_string(),
                    agents: Vec::new(),
                    tools: Vec::new(),
                    handle: Some("Invalid Handle".to_string()),
                },
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid engine handle"));
    }

    #[test]
    fn remote_tool_and_agent_wrap_definitions_and_validate_names() {
        let tool_function = tagged_function("lookup", "Lookup docs", &["text"]);
        let default_tool = RemoteTool::new(
            Principal::anonymous(),
            "https://remote.example".to_string(),
            tool_function.clone(),
            None,
        )
        .unwrap();
        assert_eq!(default_tool.name(), "lookup");
        assert_eq!(default_tool.definition().name, "lookup");

        let tool = RemoteTool::new(
            Principal::anonymous(),
            "https://remote.example".to_string(),
            tool_function.clone(),
            Some("remote_lookup".to_string()),
        )
        .unwrap();
        assert_eq!(tool.name(), "remote_lookup");
        assert_eq!(tool.description(), "Lookup docs");
        assert_eq!(tool.definition().name, "remote_lookup");
        assert_eq!(tool.supported_resource_tags(), vec!["text"]);
        assert!(
            RemoteTool::new(
                Principal::anonymous(),
                "https://remote.example".to_string(),
                tool_function,
                Some("bad name".to_string()),
            )
            .is_err()
        );

        let agent_function = tagged_function("chat", "Chat remotely", &["md"]);
        let default_agent = RemoteAgent::new(
            Principal::anonymous(),
            "https://remote.example".to_string(),
            agent_function.clone(),
            None,
        )
        .unwrap();
        assert_eq!(default_agent.name(), "chat");
        assert_eq!(default_agent.definition().name, "chat");

        let agent = RemoteAgent::new(
            Principal::anonymous(),
            "https://remote.example".to_string(),
            agent_function.clone(),
            Some("RemoteChat".to_string()),
        )
        .unwrap();
        assert_eq!(agent.name(), "RemoteChat");
        assert_eq!(agent.description(), "Chat remotely");
        assert_eq!(agent.definition().name, "remotechat");
        assert_eq!(agent.supported_resource_tags(), vec!["md"]);
        assert!(
            RemoteAgent::new(
                Principal::anonymous(),
                "https://remote.example".to_string(),
                agent_function,
                Some("bad name".to_string()),
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn remote_wrappers_forward_calls_to_context_and_report_missing_endpoints() {
        let ctx = crate::engine::EngineBuilder::new().mock_ctx();

        let tool = RemoteTool::new(
            Principal::anonymous(),
            "https://remote.example".to_string(),
            tagged_function("lookup", "Lookup docs", &["text"]),
            Some("remote_lookup".to_string()),
        )
        .unwrap();
        let err = Tool::<BaseCtx>::call(
            &tool,
            ctx.base.clone(),
            json!({"query": "anda"}),
            vec![resource(7, &["text"])],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("remote engine endpoint"));

        let agent = RemoteAgent::new(
            Principal::anonymous(),
            "https://remote.example".to_string(),
            tagged_function("chat", "Chat remotely", &["md"]),
            Some("RemoteChat".to_string()),
        )
        .unwrap();
        let err =
            Agent::<AgentCtx>::run(&agent, ctx, "hello".to_string(), vec![resource(8, &["md"])])
                .await
                .unwrap_err();
        assert!(err.to_string().contains("remote engine endpoint"));
    }
}
