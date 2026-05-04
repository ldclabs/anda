//! Engine construction and top-level execution APIs.
//!
//! [`Engine`] is the runtime boundary for a group of agents, tools, model
//! providers, storage backends, hooks, and remote engines. It is responsible for:
//! - validating callers and request metadata;
//! - creating scoped [`AgentCtx`] and [`BaseCtx`] values;
//! - dispatching local agent runs and direct tool calls;
//! - exporting selected agents and tools to other engines;
//! - signing challenge responses when Web3 or TEE clients are configured.
//!
//! Engines are created with [`EngineBuilder`]. A built engine is private by
//! default; configure [`Management`] to expose it to additional callers.
//!
//! # Usage
//! 1. Start from [`Engine::builder`].
//! 2. Register models, tools, agents, hooks, storage, and remote engines.
//! 3. Select the default agent with [`EngineBuilder::build`].
//! 4. Execute agents with [`Engine::agent_run`] or direct tools with
//!    [`Engine::tool_call`].
//!
//! # Example
//! ```rust,ignore
//! use anda_core::AgentInput;
//! use anda_engine::{
//!     ANONYMOUS,
//!     engine::{AgentInfo, EchoEngineInfo, Engine},
//! };
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let echo_info = AgentInfo {
//!     handle: "echo".to_string(),
//!     name: "Echo Agent".to_string(),
//!     description: "Returns engine metadata as JSON.".to_string(),
//!     ..Default::default()
//! };
//!
//! let engine = Engine::builder()
//!     .register_agent(Arc::new(EchoEngineInfo::new(echo_info)), None)?
//!     .build("echo".to_string())
//!     .await?;
//!
//! let output = engine
//!     .agent_run(
//!         ANONYMOUS,
//!         AgentInput::new("echo".to_string(), "hello".to_string()),
//!     )
//!     .await?;
//! # Ok(())
//! # }
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
    collections::{BTreeSet, HashMap},
    sync::{Arc, OnceLock, Weak},
};
use structured_logger::unix_ms;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

use crate::{
    context::{
        AgentCtx, BaseCtx, SubAgentManager, SubAgentSetManager, ToolsSearch, ToolsSelect,
        Web3Client, Web3SDK,
    },
    hook::{Hook, Hooks},
    management::{BaseManagement, Management, SYSTEM_PATH, Visibility},
    model::{Model, Models},
    store::Store,
};

pub use crate::context::{AgentInfo, EngineCard, RemoteEngineArgs, RemoteEngines};

/// Top-level runtime for a configured set of Anda agents and tools.
///
/// An engine owns the shared runtime state used to create child contexts for
/// agent and tool execution. It is cheap to wrap in an [`Arc`] and share across
/// server handlers.
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
    /// Creates a new builder with default engine metadata, in-memory storage,
    /// built-in tool discovery agents, and private visibility.
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

    /// Returns mutable engine metadata.
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

    /// Returns a reference to the engine's models.
    pub fn models(&self) -> Arc<Models> {
        self.ctx.models.clone()
    }

    /// Cancels the engine and resolves once cancellation has been requested.
    pub async fn close(&self) -> Result<(), BoxError> {
        self.ctx.base.cancellation_token.cancel();
        self.cancelled().await;
        Ok(())
    }

    /// Creates an agent context for an exported agent or a manager-only agent.
    ///
    /// Non-manager callers can only access names listed by
    /// [`EngineBuilder::export_agents`] or the default agent selected by
    /// [`EngineBuilder::build`]. Managers can access registered agents that are
    /// not exported.
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

    /// Creates a tool context for an exported tool or a manager-only tool.
    ///
    /// The returned [`BaseCtx`] is scoped to the requested tool name and carries
    /// the caller, agent name, request metadata, cancellation token, cache,
    /// storage, and shared runtime state.
    pub fn base_ctx_with(
        &self,
        caller: Principal,
        agent_name: &str,
        tool_name: &str,
        meta: RequestMeta,
    ) -> Result<BaseCtx, BoxError> {
        let name = tool_name.to_ascii_lowercase();

        // manager can access any tool
        if (!self.export_tools.contains(&name) && !self.management.is_manager(&caller))
            || !self.ctx.tools.contains(&name)
        {
            return Err(format!("tool {} not found", name).into());
        }

        self.ctx.child_base_with(
            caller,
            agent_name.to_ascii_lowercase().as_str(),
            &name,
            meta,
        )
    }

    /// Executes an agent request.
    ///
    /// If `input.name` is empty, the engine's default agent is used. The method
    /// validates request metadata, enforces visibility rules, runs engine hooks,
    /// and clears provider-native raw history before returning the output to the
    /// caller.
    pub async fn agent_run(
        &self,
        caller: Principal,
        mut input: AgentInput,
    ) -> Result<AgentOutput, BoxError> {
        let meta = input.meta.unwrap_or_default();
        if meta.engine.is_some() && meta.engine != Some(self.id) {
            return Err(format!(
                "invalid engine ID, expected {}, got {:?}",
                self.id.to_text(),
                meta.engine
            )
            .into());
        }
        if let Some(user) = &meta.user {
            let u = user.trim();
            if u.is_empty() || u != user || u.len() > 96 {
                return Err(format!("invalid user name {:?}", user).into());
            }
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

    /// Calls a registered tool directly.
    ///
    /// Direct tool calls follow the same engine-id, user-name, visibility, and
    /// hook checks as agent execution. Non-manager callers can only call tools
    /// exported through [`EngineBuilder::export_tools`].
    pub async fn tool_call(
        &self,
        caller: Principal,
        input: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        let meta = input.meta.unwrap_or_default();
        if meta.engine.is_some() && meta.engine != Some(self.id) {
            return Err(format!(
                "invalid engine ID, expected {}, got {:?}",
                self.id.to_text(),
                meta.engine
            )
            .into());
        }
        if let Some(user) = &meta.user {
            let u = user.trim();
            if u.is_empty() || u != user || u.len() > 96 {
                return Err(format!("invalid user name {:?}", user).into());
            }
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

        let ctx = self
            .ctx
            .child_base_with(caller, &self.default_agent, &input.name, meta)?;
        self.hooks.on_tool_start(&ctx, &input.name).await?;

        let output = tool.call(ctx.clone(), input.args, input.resources).await?;
        let res = self.hooks.on_tool_end(&ctx, &input.name, output).await?;
        Ok(res)
    }

    /// Returns metadata for registered agents.
    ///
    /// When `names` is `Some`, only matching names are returned.
    pub fn agents(&self, names: Option<&[String]>) -> Vec<Function> {
        self.ctx.agents.functions(names)
    }

    /// Returns metadata for registered tools.
    ///
    /// When `names` is `Some`, only matching names are returned.
    pub fn tools(&self, names: Option<&[String]>) -> Vec<Function> {
        self.ctx.tools.functions(names)
    }

    /// Returns a reference to the subagents manager.
    pub fn sub_agents_manager(&self) -> Arc<SubAgentSetManager> {
        self.ctx.subagents.clone()
    }

    /// Signs a challenge request with the configured Web3 or TEE identity.
    ///
    /// TEE-backed engines include attestation data in the returned envelope.
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

    /// Returns the public engine card used by remote engines for discovery.
    ///
    /// Only exported agents and tools are included.
    pub fn information(&self) -> EngineCard {
        EngineCard {
            id: self.id,
            info: self.info.clone(),
            agents: self.agents(Some(
                self.export_agents
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .as_slice(),
            )),
            tools: self.tools(Some(
                self.export_tools
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .as_slice(),
            )),
        }
    }
}

/// Builder for assembling an [`Engine`].
///
/// The builder starts with in-memory object storage, a non-implemented Web3
/// client, no external model provider, the built-in `tools_search` and
/// `tools_select` discovery agents, and private management visibility.
#[non_exhaustive]
pub struct EngineBuilder {
    info: AgentInfo,
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    subagents: SubAgentSetManager,
    remote: HashMap<String, RemoteEngineArgs>,
    models: Arc<Models>,
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
        let subagent_manager = Arc::new(SubAgentManager::new());
        let subagents = SubAgentSetManager::new();
        subagents.insert(subagent_manager.clone());

        let mut tools = ToolSet::new();
        tools.add(subagent_manager).unwrap();

        let mut agents = AgentSet::new();
        agents
            .add(Arc::new(ToolsSearch::new()), Some("flash".to_string()))
            .unwrap();
        agents
            .add(Arc::new(ToolsSelect::new()), Some("flash".to_string()))
            .unwrap();
        EngineBuilder {
            info: AgentInfo {
                handle: "anda".to_string(),
                name: "Anda Engine".to_string(),
                description: "Anda Engine for managing agents and tools".to_string(),
                endpoint: "https://localhost:8443/default".to_string(),
                ..Default::default()
            },
            tools,
            agents,
            subagents,
            remote: HashMap::new(),
            models: Arc::new(Models::default()),
            store: Store::new(mstore),
            web3: Arc::new(Web3SDK::Web3(Web3Client::not_implemented())),
            hooks: Arc::new(Hooks::new()),
            cancellation_token: CancellationToken::new(),
            export_agents: BTreeSet::new(),
            export_tools: BTreeSet::new(),
            management: None,
        }
    }

    /// Sets the public engine metadata returned by [`Engine::information`].
    pub fn with_info(mut self, info: AgentInfo) -> Self {
        self.info = info;
        self
    }

    /// Sets the cancellation token.
    pub fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = cancellation_token;
        self
    }

    /// Sets the Web3 or TEE client used for identity, signing, and challenges.
    pub fn with_web3_client(mut self, web3: Arc<Web3SDK>) -> Self {
        self.web3 = web3;
        self
    }

    /// Sets the primary default model used when a request does not specify a label.
    pub fn with_model(self, model: Model) -> Self {
        self.models.set_model(model);
        self
    }

    /// Replaces the model registry used by the engine.
    pub fn with_models(mut self, models: Arc<Models>) -> Self {
        self.models = models;
        self
    }

    /// Sets a global fallback model.
    ///
    /// The fallback model will be used when the primary model returns an `AgentOutput`
    /// with `failed_reason`.
    pub fn with_fallback_model(self, model: Model) -> Self {
        self.models.set_fallback_model(model);
        self
    }

    /// Sets the storage backend for the engine.
    pub fn with_store(mut self, store: Store) -> Self {
        self.store = store;
        self
    }

    /// Sets the management policy used for caller authorization and visibility.
    pub fn with_management(mut self, management: Arc<dyn Management>) -> Self {
        self.management = Some(management);
        self
    }

    /// Registers a single tool with the engine.
    /// Returns an error if the tool cannot be added.
    pub fn register_tool<T>(mut self, tool: Arc<T>) -> Result<Self, BoxError>
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
    /// Recommended labels: "pro", "flash", "lite", "fallback"
    pub fn register_agent<T>(
        mut self,
        agent: Arc<T>,
        label: Option<String>,
    ) -> Result<Self, BoxError>
    where
        T: Agent<AgentCtx> + Send + Sync + 'static,
    {
        for tool in agent.tool_dependencies() {
            if !self.tools.contains(&tool) && !self.agents.contains(&tool) {
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
                if !self.tools.contains(&tool) && !self.agents.contains(&tool) {
                    return Err(format!("dependent tool {} not found", tool).into());
                }
            }
            self.agents.set.insert(name, agent);
        }

        Ok(self)
    }

    /// Registers a remote engine for cross-engine agent and tool calls.
    ///
    /// Remote metadata is fetched during [`EngineBuilder::build`]. Optional
    /// agent and tool filters in [`RemoteEngineArgs`] limit which remote
    /// functions are exposed through this engine.
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

    /// Exports agents by name for non-manager callers and remote discovery.
    pub fn export_agents(mut self, agents: Vec<String>) -> Self {
        for mut agent in agents {
            agent.make_ascii_lowercase();
            self.export_agents.insert(agent);
        }
        self
    }

    /// Exports tools by name for non-manager callers and remote discovery.
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

    /// Creates an engine without selecting a default agent.
    ///
    /// This is mainly useful for tests or management-only engines. Most
    /// production engines should use [`EngineBuilder::build`].
    pub async fn empty(self) -> Result<Engine, BoxError> {
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
        let ctx = BaseCtx::new(
            id,
            self.info.name.clone(),
            "".to_string(),
            self.cancellation_token,
            names,
            self.web3,
            self.store,
            Arc::new(RemoteEngines::new()),
        );

        let tools = Arc::new(self.tools);
        let agents = Arc::new(self.agents);

        let ctx = AgentCtx::new(
            ctx,
            self.models,
            tools.clone(),
            agents.clone(),
            Arc::new(self.subagents),
        );

        let meta = RequestMeta::default();
        for (name, tool) in &tools.set {
            let ct = ctx.child_base_with(id, "", name, meta.clone())?;
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
        })
    }

    /// Finalizes the builder and creates an engine with a default agent.
    ///
    /// The default agent is automatically exported. Registered tools and agents
    /// are initialized before the engine is returned.
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
            default_agent.clone(),
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
            self.models,
            tools.clone(),
            agents.clone(),
            Arc::new(self.subagents),
        );

        let meta = RequestMeta::default();
        for (name, tool) in &tools.set {
            let ct = ctx.child_base_with(id, &default_agent, name, meta.clone())?;
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

    /// Creates a mock agent context for tests and examples.
    // #[cfg(test)]
    pub fn mock_ctx(self) -> AgentCtx {
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
        let ctx = BaseCtx::new(
            Principal::anonymous(),
            "Mocker".to_string(),
            "Mocker".to_string(),
            self.cancellation_token,
            names,
            self.web3,
            self.store,
            Arc::new(RemoteEngines::new()),
        );

        AgentCtx::new(
            ctx,
            self.models,
            Arc::new(self.tools),
            Arc::new(self.agents),
            Arc::new(self.subagents),
        )
    }
}

/// Simple built-in agent that returns its configured [`AgentInfo`] as JSON.
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

/// EngineRef is a helper struct that allows for late binding of an Engine instance.
pub struct EngineRef {
    inner: OnceLock<Weak<Engine>>,
}

impl Default for EngineRef {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineRef {
    pub fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// Binds the reference to an engine using a weak pointer.
    pub fn bind(&self, engine: Weak<Engine>) {
        let _ = self.inner.set(engine);
    }

    /// Attempts to upgrade the weak pointer to a live engine instance.
    ///
    /// Returns `None` if no engine has been bound or the engine has been
    /// dropped.
    pub fn get(&self) -> Option<Arc<Engine>> {
        self.inner.get().and_then(Weak::upgrade)
    }
}
