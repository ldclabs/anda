use super::*;

/// Object-safe registry interface for groups of subagents.
pub trait SubAgentSet: Send + Sync {
    /// Converts the registry into [`Any`] for downcasting.
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
    /// - Vec<[`FunctionDefinition`]>: Vector of agent definitions.
    fn definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Selects and returns resources relevant to the specified subagent name from the provided list.
    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource>;
}

/// Tool and registry for creating, updating, loading, and running subagents.
pub struct SubAgentManager {
    agents: RwLock<BTreeMap<String, SubAgent>>,
    models: Vec<String>,
}

impl Default for SubAgentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentManager {
    /// Function name used when registering the manager tool.
    pub const NAME: &'static str = "subagents_manager";

    /// Creates an empty subagent manager.
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(BTreeMap::new()),
            models: Vec::new(),
        }
    }

    /// Sets the model labels allowed for managed subagents.
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        self.models = models;
        self
    }

    pub(super) fn store_prefix() -> Path {
        Path::from(SUBAGENT_STORE_PATH)
    }

    pub(super) fn store_path(name: &str) -> Path {
        Path::from(format!("{SUBAGENT_STORE_PATH}/{name}"))
    }

    /// Loads persisted subagents from engine storage.
    pub async fn load(&self, ctx: AgentCtx) -> Result<(), BoxError> {
        let offset = Path::from("");
        let prefix = Self::store_prefix();
        let agents = match ctx.root.store_list(Some(&prefix), &offset).await {
            Ok(agents) => agents,
            Err(err) => {
                log::warn!("failed to list persisted subagents: {err}");
                return Ok(());
            }
        };

        // One corrupted or unreadable entry must not prevent the other subagents from loading.
        for meta in agents {
            let data = match ctx.root.store_get(&meta.location).await {
                Ok((data, _)) => data,
                Err(err) => {
                    log::warn!("failed to read persisted subagent {}: {err}", meta.location);
                    continue;
                }
            };

            match from_slice::<SubAgent>(&data[..]) {
                Ok(mut agent) => {
                    let name = agent.name.to_ascii_lowercase();
                    self.preserve_runtime_state(&name, &mut agent);
                    self.agents.write().insert(name, agent);
                }
                Err(err) => {
                    log::warn!(
                        "failed to decode persisted subagent {}: {err}",
                        meta.location
                    );
                }
            }
        }

        Ok(())
    }

    fn preserve_runtime_state(&self, name: &str, agent: &mut SubAgent) {
        if let Some(existing) = self.agents.read().get(name) {
            agent.subsessions = existing.subsessions.clone();
        }
    }

    /// Lists registered subagents and their active sessions. When `name_filter` is non-empty, only
    /// the matching subagent is reported. Each session entry carries a live status snapshot
    /// (elapsed run time, idle time, token usage, turns, latest progress, and background tasks) so
    /// the parent can poll progress without waiting for hook callbacks.
    fn catalog(&self, name_filter: &str) -> Json {
        let name_filter = name_filter.trim().to_ascii_lowercase();
        let agents = self.agents.read().values().cloned().collect::<Vec<_>>();
        let subagents = agents
            .into_iter()
            .filter(|agent| {
                name_filter.is_empty() || agent.name.to_ascii_lowercase() == name_filter
            })
            .map(|agent| {
                let name = agent.name.to_ascii_lowercase();
                let callable = format!("SA_{name}");
                let has_output_schema = agent.output_schema.is_some();
                let sessions = agent.subsessions.session_details();
                let active_sessions = sessions
                    .iter()
                    .map(|detail| detail["session"].clone())
                    .collect::<Vec<_>>();
                let model = selected_model_label(&agent.model);
                json!({
                    "name": name,
                    "callable": callable,
                    "description": agent.description,
                    "tools": agent.tools,
                    "tags": agent.tags,
                    "has_output_schema": has_output_schema,
                    "model": model,
                    "effort": agent.effort,
                    "idle_timeout": agent.idle_timeout,
                    "active_sessions": active_sessions,
                    "sessions": sessions,
                })
            })
            .collect::<Vec<_>>();

        json!({
            "result": "listed",
            "count": subagents.len(),
            "subagents": subagents,
            "hint": "Use SA_<name> for delegated work. Use a stable session ID for long-running, parallel, asynchronous, or follow-up tasks. To poll a running session, call SA_<name> with `/status` and its session ID, or list with a `name` to inspect that worker's live sessions."
        })
    }

    /// Creates or updates a subagent. The name is normalised to lowercase and validated. If an agent with the same name exists, it will be overwritten.
    pub async fn upsert(&self, ctx: AgentCtx, mut agent: SubAgent) -> Result<(), BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;
        self.preserve_runtime_state(&name, &mut agent);

        let data = to_canonical_vec(&agent)?;
        self.agents.write().insert(name.clone(), agent);

        ctx.root
            .store_put(&Self::store_path(&name), PutMode::Overwrite, data.into())
            .await?;
        Ok(())
    }

    /// Creates or updates an in-memory subagent without writing it to the store.
    pub fn upsert_temporary(&self, mut agent: SubAgent) -> Result<String, BoxError> {
        let name = agent.name.to_ascii_lowercase();
        validate_function_name(&name)?;
        self.preserve_runtime_state(&name, &mut agent);

        self.agents.write().insert(name.clone(), agent);
        Ok(name)
    }

    fn description_text(&self) -> String {
        if self.models.is_empty() {
            "Scheduler control plane for reusable subagents. Use it to list available workers, create or update focused helpers with stable instructions and restricted toolsets, optionally run an initial delegated task, and optionally persist useful helpers for future sessions and restarts. Temporary subagents are callable immediately as `SA_<name>`.".to_string()
        } else {
            format!(
                "Scheduler control plane for reusable subagents with model-aware routing. Use it to list available workers, create or update focused helpers with stable instructions and restricted toolsets, optionally run an initial delegated task, and optionally persist useful helpers for future sessions and restarts. Temporary subagents are callable immediately as `SA_<name>`. This manager supports the following models for routing decisions: {}.",
                self.models.join(", ")
            )
        }
    }

    fn manager_definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: Self::NAME.to_string(),
            description: self.description_text(),
            parameters: json!({
                "type": "object",
                "description": "List the subagent registry, or create/update a subagent configuration, optionally run it immediately, and optionally persist it for future reuse.",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["upsert", "list", "status"],
                        "description": "Use `list` or `status` to inspect registered subagents and the live progress of their active sessions (elapsed run time, idle time, token usage, turns, latest progress, and background tasks); pass a `name` to inspect a single worker. Use `upsert` to create or update a worker and optionally run a delegated task.",
                        "default": "upsert"
                    },
                    "name": {
                        "type": "string",
                        "description": "For operation=upsert, the unique callable subagent name. Must start with a lowercase letter, contain only lowercase letters, digits, underscores, or hyphens, and be no longer than 64 characters. The subagent becomes callable as SA_<name>. For operation=list or status, optionally set it to a subagent name to report only that worker's sessions, or use an empty string to report all."
                    },
                    "description": {
                        "type": "string",
                        "description": "For operation=upsert, the routing description shown when models decide whether to call this subagent. State when it should be used and what outcome it produces. For operation=list, use an empty string."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "For operation=upsert, durable system-style instructions for the subagent. Define its role, scope, workflow, constraints, decision rules, and expected output style. Write reusable guidance, not a one-off task prompt. For operation=list, use an empty string."
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional whitelist of tool names the subagent may use. Include only the minimum tools it needs. Leave empty to create a no-tool subagent.",
                        "default": []
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of resource tags relevant to this subagent, such as 'image', 'text', or 'audio'. Resources with matching tags are processed when the subagent is called.",
                        "default": []
                    },
                    "output_schema": {
                        "type": ["string", "null"],
                        "description": "Optional JSON schema encoded as a JSON string that the subagent's output must conform to. Use null for unstructured text output.",
                        "default": null
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional default model label used to run this subagent. Leave empty to use the caller context model. For operation=list, use an empty string.",
                        "default": ""
                    },
                    "effort": {
                        "type": ["string", "null"],
                        "enum": ["minimal", "low", "medium", "high", "max", null],
                        "description": "Optional default reasoning/thinking effort used to run this subagent. Use null to leave the selected model's default effort unchanged. For operation=list, use null.",
                        "default": null
                    },
                    "idle_timeout": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional idle timeout in seconds for this subagent's sessions. A session with no running background task ends after this much inactivity. Use 0 to keep the engine default; larger values are capped at the background-task wait ceiling. Tune it up for sessions you will revisit after gaps, down to reclaim idle workers sooner. For operation=list, use 0.",
                        "default": 0
                    },
                    "task": {
                        "type": "string",
                        "description": "Optional immediate task handoff to run with the newly created or updated subagent. Include objective, context/resources, constraints, dependencies, expected deliverable, and success criteria. Leave empty to only create/update or when operation=list.",
                        "default": ""
                    },
                    "session": {
                        "type": "string",
                        "description": "Optional session ID for the immediate task. Leave empty for blocking one-shot mode. Provide a stable ID for non-blocking, parallel, asynchronous, or follow-up work with hook-delivered progress and final output.",
                        "default": ""
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Set true to save or update this subagent for future calls and restarts. Leave false to keep it temporary in the current engine process.",
                        "default": false
                    }
                },
                "required": ["operation", "name", "description", "instructions", "tools", "tags", "output_schema", "model", "effort", "idle_timeout", "task", "session", "persist"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn configure(
        &self,
        ctx: AgentCtx,
        args: SubAgentManagerArgs,
    ) -> Result<(String, SubAgent, Option<String>, String, bool), BoxError> {
        let (mut agent, task, session, persist) = args.into_subagent();
        let name = agent.name.to_ascii_lowercase();
        self.preserve_runtime_state(&name, &mut agent);

        if persist {
            self.upsert(ctx, agent.clone()).await?;
        } else {
            self.upsert_temporary(agent.clone())?;
        }

        Ok((name, agent, task, session, persist))
    }
}

impl Agent<AgentCtx> for SubAgentManager {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description_text()
    }

    fn definition(&self) -> FunctionDefinition {
        self.manager_definition()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        vec!["*".to_string()]
    }

    async fn init(&self, ctx: AgentCtx) -> Result<(), BoxError> {
        self.load(ctx).await
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let args = SubAgentManagerArgs::from_prompt(prompt)?;
        let operation = args.operation.trim().to_ascii_lowercase();
        if matches!(operation.as_str(), "list" | "status" | "catalog") {
            return Ok(AgentOutput {
                content: self.catalog(&args.name).to_string(),
                ..Default::default()
            });
        }

        if !matches!(operation.as_str(), "" | "upsert" | "create" | "update") {
            return Err(format!("unsupported subagent manager operation: {operation}").into());
        }

        let (name, agent, task, session, persist) = self.configure(ctx.clone(), args).await?;
        let callable = format!("SA_{name}");
        let subagent = json!({
            "result": if persist { "persisted" } else { "created" },
            "name": name,
            "callable": callable,
            "persisted": persist,
            "model": selected_model_label(&agent.model),
            "effort": agent.effort,
            "active_sessions": agent.subsessions.active_session_ids(),
            "hint": "Call the subagent by this callable name. Use a stable session ID for long-running, parallel, asynchronous, or follow-up tasks. If a temporary subagent proves useful, call subagents_manager again with persist=true to save it."
        });
        let Some(task) = task else {
            return Ok(AgentOutput {
                content: subagent.to_string(),
                ..Default::default()
            });
        };

        let prompt = serde_json::to_string(&SubAgentArgs {
            prompt: task,
            session,
            model: String::new(),
            effort: None,
        })?;

        match agent.run(ctx.child(&name, &name)?, prompt, resources).await {
            Ok(mut rt) => {
                rt.content = json!({
                    "subagent": subagent,
                    "output": rt.content,
                })
                .to_string();
                Ok(rt)
            }
            Err(err) => Ok(AgentOutput {
                content: subagent.to_string(),
                failed_reason: Some(format!("Subagent run error: {err}")),
                ..Default::default()
            }),
        }
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
                .map(|agent| agent.definition())
                .collect(),
            Some(names) => {
                let agents = self.agents.read();
                names
                    .iter()
                    .filter_map(|name| {
                        agents
                            .get(&name.to_ascii_lowercase())
                            .map(|agent| agent.definition())
                    })
                    .collect()
            }
        }
    }

    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        self.agents
            .read()
            .get(&name.to_ascii_lowercase())
            .map(|agent| {
                let supported_tags = agent.supported_resource_tags();
                select_resources(resources, &supported_tags)
            })
            .unwrap_or_default()
    }
}

/// Type-indexed collection of subagent registries.
pub struct SubAgentSetManager {
    sets: RwLock<BTreeMap<TypeId, Arc<dyn SubAgentSet>>>,
}

impl Default for SubAgentSetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentSetManager {
    /// Creates an empty collection of subagent registries.
    pub fn new() -> Self {
        Self {
            sets: RwLock::new(BTreeMap::new()),
        }
    }

    /// Inserts a typed subagent registry and returns the previous registry of the same type.
    pub fn insert<T: SubAgentSet + Sized + 'static>(&self, set: Arc<T>) -> Option<Arc<T>> {
        let type_id = TypeId::of::<T>();
        self.sets
            .write()
            .insert(type_id, set)
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
    }

    /// Returns a typed subagent registry when one has been inserted.
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

    fn select_resources(&self, name: &str, resources: &mut Vec<Resource>) -> Vec<Resource> {
        if resources.is_empty() {
            return Vec::new();
        }

        for set in self.sets.read().values() {
            let selected = set.select_resources(name, resources);
            if !selected.is_empty() {
                return selected;
            }
        }

        Vec::new()
    }
}
