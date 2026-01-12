//! The Engine module provides the core functionality for managing and executing agents and tools.
//!
//! # Overview
//! The Engine is the central component that orchestrates agent execution, tool management,
//! and context handling. It provides:
//! - Agent management and execution
//! - Tool registration and invocation
//! - Context management with cancellation support
//! - Builder pattern for configuration
//!
//! # Key Components
//! - [`Engine`]: The main struct that provides execution capabilities
//! - [`EngineBuilder`]: Builder pattern for constructing Engine instances
//! - Context management through [`AgentCtx`] and [`BaseCtx`]
//!
//! # Usage
//! 1. Create an Engine using the builder pattern
//! 2. Register tools and agents
//! 3. Execute agents or call tools
//!
//! # Example
//! ```rust,ignore
//! let engine = Engine::builder()
//!     .with_name("MyEngine".to_string())
//!     .register_tool(my_tool)?
//!     .register_agent(my_agent, None)?
//!     .build("default_agent".to_string())?;
//!
//! let output = engine.agent_run(None, "Hello".to_string(), None, None, None).await?;
//! ```

use anda_cloud_cdk::{ChallengeEnvelope, ChallengeRequest, TEEInfo, TEEKind};
use anda_core::{
    Agent, AgentInput, AgentOutput, AgentSet, BoxError, Function, Json, Path, RequestMeta,
    Resource, Tool, ToolInput, ToolOutput, ToolSet, validate_function_name,
};
use candid::Principal;
use ic_tee_cdk::AttestationRequest;
use object_store::memory::InMemory;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use structured_logger::unix_ms;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

use crate::{
    context::{AgentCtx, BaseCtx, Web3Client, Web3SDK},
    hook::{Hook, Hooks},
    management::{BaseManagement, Management, SYSTEM_PATH, Visibility},
    model::Model,
    store::Store,
};

pub use crate::context::{AgentInfo, EngineCard, RemoteEngineArgs, RemoteEngines};

/// Engine is the core component that manages agents, tools, and execution context.
/// It provides methods to interact with agents, call tools, and manage execution.
#[derive(Clone)]
pub struct Engine {
    id: Principal,
    ctx: AgentCtx,
    info: AgentInfo,
    default_agent: String,
    export_agents: BTreeSet<String>,
    export_tools: BTreeSet<String>,
    hooks: Arc<Hooks>,
    management: Arc<dyn Management>,
}

impl Engine {
    /// Creates a new EngineBuilder instance for constructing an Engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// Returns the engine ID.
    pub fn id(&self) -> Principal {
        self.id
    }

    /// Returns the information about the engine.
    pub fn info(&self) -> &AgentInfo {
        &self.info
    }

    pub fn info_mut(&mut self) -> &mut AgentInfo {
        &mut self.info
    }

    /// Returns the name of the default agent.
    pub fn default_agent(&self) -> String {
        self.default_agent.clone()
    }

    /// Cancels all tasks in the engine by triggering the cancellation token.
    pub fn cancel(&self) {
        self.ctx.base.cancellation_token.cancel()
    }

    /// Returns `true` if the Engine is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.ctx.base.cancellation_token.is_cancelled()
    }

    /// Returns a [`Future`] that gets fulfilled when cancellation is requested.
    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.ctx.base.cancellation_token.cancelled()
    }

    /// Creates and returns a child cancellation token.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.ctx.base.cancellation_token.child_token()
    }

    /// Closes the engine, cancelling all tasks and waiting for cancellation to complete.
    pub async fn close(&self) -> Result<(), BoxError> {
        self.ctx.base.cancellation_token.cancel();
        self.cancelled().await;
        Ok(())
    }

    /// Creates a new [`AgentCtx`] with the specified agent name, user, and caller.
    /// Returns an error if the agent is not found or if the user name is invalid.
    pub fn ctx_with(
        &self,
        caller: Principal,
        agent_name: &str,
        agent_label: &str,
        meta: RequestMeta,
    ) -> Result<AgentCtx, BoxError> {
        let name = agent_name.to_ascii_lowercase();

        // manager can access any agent
        if (!self.export_agents.contains(&name) && !self.management.is_manager(&caller))
            || !self.ctx.agents.contains(&name)
        {
            return Err(format!("agent {} not found", name).into());
        }

        self.ctx.child_with(caller, &name, agent_label, meta)
    }

    /// Executes an agent with the specified parameters.
    /// If no agent name is provided, uses the default agent.
    /// Returns the agent's output or an error if the agent is not found.
    pub async fn agent_run(
        &self,
        caller: Principal,
        mut input: AgentInput,
    ) -> Result<AgentOutput, BoxError> {
        let meta = input.meta.unwrap_or_default();
        if meta.engine.is_some() && meta.engine != Some(self.id) {
            return Err(format!(
                "invalid engine ID, expected {}, got {}",
                self.id.to_text(),
                meta.engine.unwrap().to_text()
            )
            .into());
        }

        input.name = if input.name.is_empty() {
            self.default_agent.clone()
        } else {
            input.name.to_ascii_lowercase()
        };
        let agent = self
            .ctx
            .agents
            .get(&input.name)
            .ok_or_else(|| format!("agent {} not found", input.name))?;

        let visibility = self.management.check_visibility(&caller)?;
        if visibility == Visibility::Protected && !self.management.is_manager(&caller) {
            return Err("caller does not have permission".into());
        }

        let ctx = self.ctx_with(caller, &input.name, agent.label(), meta)?;
        self.hooks.on_agent_start(&ctx, &input.name).await?;

        let output = agent
            .run(ctx.clone(), input.prompt, input.resources)
            .await?;
        let mut output = self.hooks.on_agent_end(&ctx, &input.name, output).await?;
        output.raw_history.clear(); // clear raw history
        Ok(output)
    }

    /// Calls a tool by name with the specified arguments.
    /// Returns tuple containing the result string and a boolean indicating if further processing is needed.
    pub async fn tool_call(
        &self,
        caller: Principal,
        input: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        let meta = input.meta.unwrap_or_default();
        if meta.engine.is_some() && meta.engine != Some(self.id) {
            return Err(format!(
                "invalid engine ID, expected {}, got {}",
                self.id.to_text(),
                meta.engine.unwrap().to_text()
            )
            .into());
        }

        // manager can call any tool
        if !self.export_tools.contains(&input.name) && !self.management.is_manager(&caller) {
            return Err(format!("tool {} not found", &input.name).into());
        }

        let tool = self
            .ctx
            .tools
            .get(&input.name)
            .ok_or_else(|| format!("tool {} not found", &input.name))?;

        let visibility = self.management.check_visibility(&caller)?;
        if visibility == Visibility::Protected && !self.management.is_manager(&caller) {
            return Err("caller does not have permission".into());
        }

        let ctx = self.ctx.child_base_with(caller, &input.name, meta)?;
        self.hooks.on_tool_start(&ctx, &input.name).await?;

        let output = tool.call(ctx.clone(), input.args, input.resources).await?;
        let res = self.hooks.on_tool_end(&ctx, &input.name, output).await?;
        Ok(res)
    }

    /// Returns function definitions for the specified agents.
    /// If no names are provided, returns definitions for all agents.
    pub fn agents(&self, names: Option<&[&str]>) -> Vec<Function> {
        self.ctx.agents.functions(names)
    }

    /// Returns function definitions for the specified tools.
    /// If no names are provided, returns definitions for all tools.
    pub fn tools(&self, names: Option<&[&str]>) -> Vec<Function> {
        self.ctx.tools.functions(names)
    }

    pub async fn challenge(
        &self,
        request: ChallengeRequest,
    ) -> Result<ChallengeEnvelope, BoxError> {
        let now_ms = unix_ms();
        request.verify(now_ms, request.registry)?;
        let message_digest = request.digest();
        let res = match self.ctx.base.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                let authentication = cli.sign_envelope(message_digest).await?;
                let tee = cli
                    .sign_attestation(AttestationRequest {
                        public_key: Some(authentication.pubkey.clone()),
                        user_data: None,
                        nonce: Some(request.code.to_vec().into()),
                    })
                    .await?;
                let info = cli
                    .tee_info()
                    .ok_or_else(|| "TEE not available".to_string())?;
                ChallengeEnvelope {
                    request,
                    authentication,
                    tee: Some(TEEInfo {
                        id: info.id,
                        kind: TEEKind::try_from(tee.kind.as_str())?,
                        url: info.url,
                        attestation: Some(tee.attestation),
                    }),
                }
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                let authentication = cli.sign_envelope(message_digest).await?;
                ChallengeEnvelope {
                    request,
                    authentication,
                    tee: None,
                }
            }
        };
        Ok(res)
    }

    /// Returns information about the engine, including agent and tool definitions.
    pub fn information(&self) -> EngineCard {
        EngineCard {
            id: self.id,
            info: self.info.clone(),
            agents: self.agents(Some(
                self.export_agents
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )),
            tools: self.tools(Some(
                self.export_tools
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )),
        }
    }
}

/// Builder pattern implementation for constructing an Engine.
/// Allows for step-by-step configuration of the engine's components.
#[non_exhaustive]
pub struct EngineBuilder {
    info: AgentInfo,
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    remote: BTreeMap<String, RemoteEngineArgs>,
    model: Model,
    models: BTreeMap<String, Model>, // label -> Model
    store: Store,
    web3: Arc<Web3SDK>,
    hooks: Arc<Hooks>,
    cancellation_token: CancellationToken,
    export_agents: BTreeSet<String>,
    export_tools: BTreeSet<String>,
    management: Option<Arc<dyn Management>>,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    /// Creates a new EngineBuilder with default values.
    pub fn new() -> Self {
        let mstore = Arc::new(InMemory::new());
        EngineBuilder {
            info: AgentInfo {
                handle: "anda".to_string(),
                name: "Anda Engine".to_string(),
                description: "Anda Engine for managing agents and tools".to_string(),
                endpoint: "https://localhost:8443/default".to_string(),
                ..Default::default()
            },
            tools: ToolSet::new(),
            agents: AgentSet::new(),
            remote: BTreeMap::new(),
            model: Model::not_implemented(),
            models: BTreeMap::new(),
            store: Store::new(mstore),
            web3: Arc::new(Web3SDK::Web3(Web3Client::not_implemented())),
            hooks: Arc::new(Hooks::new()),
            cancellation_token: CancellationToken::new(),
            export_agents: BTreeSet::new(),
            export_tools: BTreeSet::new(),
            management: None,
        }
    }

    /// Sets the engine information.
    pub fn with_info(mut self, info: AgentInfo) -> Self {
        self.info = info;
        self
    }

    /// Sets the cancellation token.
    pub fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = cancellation_token;
        self
    }

    /// Sets the TEE (Trusted Execution Environment) client.
    pub fn with_web3_client(mut self, web3: Arc<Web3SDK>) -> Self {
        self.web3 = web3;
        self
    }

    /// Sets the model to be used by the engine.
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    /// Sets multiple models with labels for the engine.
    pub fn with_models(mut self, models: BTreeMap<String, Model>) -> Self {
        self.models = models;
        self
    }

    /// Sets the storage backend for the engine.
    pub fn with_store(mut self, store: Store) -> Self {
        self.store = store;
        self
    }

    /// Sets the management builder for the engine.
    pub fn with_management(mut self, management: Arc<dyn Management>) -> Self {
        self.management = Some(management);
        self
    }

    /// Registers a single tool with the engine.
    /// Returns an error if the tool cannot be added.
    pub fn register_tool<T>(mut self, tool: T) -> Result<Self, BoxError>
    where
        T: Tool<BaseCtx> + Send + Sync + 'static,
    {
        self.tools.add(tool)?;
        Ok(self)
    }

    /// Registers multiple tools to the engine.
    /// Returns an error if any tool already exists.
    pub fn register_tools(mut self, tools: ToolSet<BaseCtx>) -> Result<Self, BoxError> {
        for (name, tool) in tools.set {
            if self.tools.set.contains_key(&name) {
                return Err(format!("tool {} already exists", name).into());
            }
            self.tools.set.insert(name, tool);
        }

        Ok(self)
    }

    /// Registers a single agent with optional label to the engine.
    /// Verifies that all required tools are registered before adding the agent.
    /// Returns an error if any dependency is missing or if the agent cannot be added.
    pub fn register_agent<T>(mut self, agent: T, label: Option<String>) -> Result<Self, BoxError>
    where
        T: Agent<AgentCtx> + Send + Sync + 'static,
    {
        for tool in agent.tool_dependencies() {
            if !self.tools.contains(&tool) {
                return Err(format!("dependent tool {} not found", tool).into());
            }
        }

        self.agents.add(agent, label)?;
        Ok(self)
    }

    /// Registers multiple agents to the engine.
    /// Verifies that all required tools are registered for each agent.
    /// Returns an error if any agent already exists or if any dependency is missing.
    pub fn register_agents(mut self, agents: AgentSet<AgentCtx>) -> Result<Self, BoxError> {
        for (name, agent) in agents.set {
            if self.agents.set.contains_key(&name) {
                return Err(format!("agent {} already exists", name).into());
            }

            for tool in agent.tool_dependencies() {
                if !self.tools.contains(&tool) {
                    return Err(format!("dependent tool {} not found", tool).into());
                }
            }
            self.agents.set.insert(name, agent);
        }

        Ok(self)
    }

    /// Registers a remote engine with given endpoint, optional agents, tools, and alias name.
    pub fn register_remote_engine(mut self, engine: RemoteEngineArgs) -> Result<Self, BoxError> {
        if self.remote.contains_key(&engine.endpoint) {
            return Err(format!("remote engine {} already exists", engine.endpoint).into());
        }
        if let Some(handle) = &engine.handle {
            validate_function_name(handle)
                .map_err(|err| format!("invalid engine handle {}: {}", handle, err))?;
        }

        self.remote.insert(engine.endpoint.clone(), engine);
        Ok(self)
    }

    /// Exports agents by name.
    pub fn export_agents(mut self, agents: Vec<String>) -> Self {
        for mut agent in agents {
            agent.make_ascii_lowercase();
            self.export_agents.insert(agent);
        }
        self
    }

    /// Exports tools by name.
    pub fn export_tools(mut self, tools: Vec<String>) -> Self {
        for tool in tools {
            self.export_tools.insert(tool);
        }
        self
    }

    /// Sets the hooks for the engine.
    pub fn with_hooks(mut self, hooks: Arc<Hooks>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Creates an empty Engine instance.
    pub fn empty(self) -> Engine {
        let id = self.web3.as_ref().get_principal();
        let ctx = BaseCtx::new(
            id,
            self.info.name.clone(),
            self.cancellation_token,
            BTreeSet::new(),
            self.web3,
            self.store,
            Arc::new(RemoteEngines::new()),
        );

        let tools = Arc::new(ToolSet::new());
        let agents = Arc::new(AgentSet::new());
        let ctx = AgentCtx::new(ctx, self.model, Arc::new(self.models), tools, agents);

        Engine {
            id,
            ctx,
            info: self.info,
            default_agent: String::new(),
            export_agents: self.export_agents,
            export_tools: self.export_tools,
            hooks: self.hooks,
            management: self.management.unwrap_or_else(|| {
                Arc::new(BaseManagement {
                    controller: id,
                    managers: BTreeSet::new(),
                    visibility: Visibility::Private, // default visibility
                })
            }),
        }
    }

    /// Finalizes the builder and creates an Engine instance.
    /// Requires a default agent name to be specified.
    /// Returns an error if the default agent is not found.
    pub async fn build(mut self, default_agent: String) -> Result<Engine, BoxError> {
        let default_agent = default_agent.to_ascii_lowercase();
        if !self.agents.contains(&default_agent) {
            return Err(format!("default agent {} not found", default_agent).into());
        }

        self.export_agents.insert(default_agent.clone());

        self.info.validate()?;
        let id = self.web3.as_ref().get_principal();
        let mut names: BTreeSet<Path> = self
            .tools
            .set
            .keys()
            .map(|p| Path::from(format!("T:{}", p)))
            .chain(
                self.agents
                    .set
                    .keys()
                    .map(|p| Path::from(format!("A:{}", p))),
            )
            .collect();
        names.insert(Path::from(SYSTEM_PATH));

        let mut remote = RemoteEngines::new();
        for (_, engine) in self.remote {
            remote.register(self.web3.as_ref(), engine).await?;
        }

        let ctx = BaseCtx::new(
            id,
            self.info.name.clone(),
            self.cancellation_token,
            names,
            self.web3,
            self.store,
            Arc::new(remote),
        );

        let tools = Arc::new(self.tools);
        let agents = Arc::new(self.agents);
        let ctx = AgentCtx::new(
            ctx,
            self.model,
            Arc::new(self.models),
            tools.clone(),
            agents.clone(),
        );

        let meta = RequestMeta::default();
        for (name, tool) in &tools.set {
            let ct = ctx.child_base_with(id, name, meta.clone())?;
            tool.init(ct).await?;
        }

        for (name, agent) in &agents.set {
            let ct = ctx.child_with(id, name, agent.label(), meta.clone())?;
            agent.init(ct).await?;
        }

        Ok(Engine {
            id,
            ctx,
            info: self.info,
            default_agent,
            export_agents: self.export_agents,
            export_tools: self.export_tools,
            hooks: self.hooks,
            management: self.management.unwrap_or_else(|| {
                Arc::new(BaseManagement {
                    controller: id,
                    managers: BTreeSet::new(),
                    visibility: Visibility::Private, // default visibility
                })
            }),
        })
    }

    /// Creates a mock context for testing purposes.
    // #[cfg(test)]
    pub fn mock_ctx(self) -> AgentCtx {
        let mut names: BTreeSet<Path> = self
            .tools
            .set
            .keys()
            .chain(self.agents.set.keys())
            .map(|s| Path::from(s.as_str()))
            .collect();
        names.insert(Path::from(SYSTEM_PATH));
        let ctx = BaseCtx::new(
            Principal::anonymous(),
            "Mocker".to_string(),
            self.cancellation_token,
            names,
            self.web3,
            self.store,
            Arc::new(RemoteEngines::new()),
        );

        AgentCtx::new(
            ctx,
            self.model,
            Arc::new(self.models),
            Arc::new(self.tools),
            Arc::new(self.agents),
        )
    }
}

/// A simple echo agent that returns its own information as JSON.
pub struct EchoEngineInfo {
    info: AgentInfo,
    content: String,
}

impl EchoEngineInfo {
    pub fn new(info: AgentInfo) -> Self {
        let content = serde_json::to_string(&info).unwrap_or_default();
        Self { info, content }
    }
}

impl Agent<AgentCtx> for EchoEngineInfo {
    /// Returns the agent's name identifier
    fn name(&self) -> String {
        self.info.handle.clone()
    }

    /// Returns a description of the agent's purpose and capabilities.
    fn description(&self) -> String {
        self.info.description.clone()
    }

    async fn run(
        &self,
        _ctx: AgentCtx,
        _prompt: String,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        Ok(AgentOutput {
            content: self.content.clone(),
            ..Default::default()
        })
    }
}
