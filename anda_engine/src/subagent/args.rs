use super::*;

/// Arguments used to run a subagent.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SubAgentArgs {
    /// Task prompt passed to the subagent.
    pub prompt: String,

    /// Optional session ID for non-blocking session mode.
    ///
    /// When this field is empty, the subagent runs normally and returns the final output only after
    /// completion. When a session ID is provided, the subagent runs in the background and returns
    /// immediately with the normalized session ID. Progress and final output are delivered through
    /// [`AgentHook::on_background_progress`] and [`AgentHook::on_background_end`].
    ///
    /// Session mode keeps the subagent conversation alive across calls with the same session ID.
    /// Follow-up prompts, tool results, and background task results are accumulated into that
    /// conversation, allowing the subagent to preserve state across invocations. Session IDs are
    /// case-insensitive and scoped to each subagent. A missing session is created automatically.
    ///
    /// Subagents that need asynchronous tools should use session mode so background tool results can
    /// be fed back into later steps. In normal mode, asynchronous tool results cannot be delivered
    /// back into the completed subagent run.
    #[serde(default)]
    pub session: String,

    /// Optional model label for this subagent run. When empty, the subagent default is used.
    #[serde(default)]
    pub model: String,

    /// Optional reasoning/thinking effort for this subagent run.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,
}

impl SubAgentArgs {
    /// Parses tool-call arguments from the routed prompt string.
    ///
    /// A bare string is a plain blocking prompt. A JSON object whose keys all belong to
    /// [`SubAgentArgs`] must deserialize successfully, so invalid structured arguments surface as
    /// an error instead of silently running with the raw JSON as the prompt. Any other JSON
    /// payload is treated as task data for the subagent.
    pub(super) fn from_prompt(prompt: String) -> Result<Self, BoxError> {
        if !prompt.trim_start().starts_with('{') {
            return Ok(Self {
                prompt,
                ..Default::default()
            });
        }

        match serde_json::from_str::<Json>(&prompt) {
            Ok(Json::Object(args))
                if args.keys().all(|key| {
                    matches!(key.as_str(), "prompt" | "session" | "model" | "effort")
                }) =>
            {
                serde_json::from_value(Json::Object(args))
                    .map_err(|err| format!("invalid subagent arguments: {err}").into())
            }
            _ => Ok(Self {
                prompt,
                ..Default::default()
            }),
        }
    }
}

/// Arguments accepted by the subagent manager tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubAgentManagerArgs {
    /// Operation to perform. Defaults to creating or updating a subagent.
    #[serde(default = "default_manager_operation")]
    pub operation: String,

    #[serde(default)]
    /// Subagent name to create, update, remove, or inspect.
    pub name: String,

    #[serde(default)]
    /// Short capability summary for the subagent.
    pub description: String,

    #[serde(default)]
    /// System instructions stored on the subagent.
    pub instructions: String,

    #[serde(default)]
    /// Tool names allowed for the subagent.
    pub tools: Vec<String>,

    #[serde(default)]
    /// Resource tags the subagent can consume.
    pub tags: Vec<String>,

    #[serde(default, deserialize_with = "deserialize_optional_json_schema")]
    /// Optional JSON schema for the subagent's final output.
    pub output_schema: Option<Json>,

    /// Optional default model label used to run this subagent.
    #[serde(default)]
    pub model: String,

    /// Optional default reasoning/thinking effort used to run this subagent.
    #[serde(default, deserialize_with = "deserialize_optional_model_effort")]
    pub effort: Option<ModelEffort>,

    /// Optional idle timeout, in seconds, for this subagent's sessions. `0` keeps the engine
    /// default. See [`SubAgent::idle_timeout`].
    #[serde(default)]
    pub idle_timeout: u64,

    /// Optional task to run immediately after creating or updating the subagent.
    #[serde(default)]
    pub task: String,

    /// Optional session ID passed to the subagent when `task` is provided.
    #[serde(default)]
    pub session: String,

    /// Persist the subagent to storage so it remains available after restart.
    #[serde(default)]
    pub persist: bool,
}

fn default_manager_operation() -> String {
    "upsert".to_string()
}

impl Default for SubAgentManagerArgs {
    fn default() -> Self {
        Self {
            operation: default_manager_operation(),
            name: String::new(),
            description: String::new(),
            instructions: String::new(),
            tools: Vec::new(),
            tags: Vec::new(),
            output_schema: None,
            model: String::new(),
            effort: None,
            idle_timeout: 0,
            task: String::new(),
            session: String::new(),
            persist: false,
        }
    }
}

fn deserialize_optional_json_schema<'de, D>(deserializer: D) -> Result<Option<Json>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<Json>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Json::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                Ok(None)
            } else {
                serde_json::from_str(value)
                    .map(Some)
                    .map_err(serde::de::Error::custom)
            }
        }
        Json::Null => Ok(None),
        value => Ok(Some(value)),
    }
}

pub(super) fn deserialize_optional_model_effort<'de, D>(
    deserializer: D,
) -> Result<Option<ModelEffort>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<Json>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Json::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                Ok(None)
            } else {
                serde_json::from_value(Json::String(value.to_ascii_lowercase()))
                    .map(Some)
                    .map_err(serde::de::Error::custom)
            }
        }
        Json::Null => Ok(None),
        value => serde_json::from_value(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

impl SubAgentManagerArgs {
    pub(super) fn from_prompt(prompt: String) -> Result<Self, BoxError> {
        serde_json::from_str::<Self>(&prompt)
            .map_err(|err| format!("subagent manager expects JSON arguments: {err}").into())
    }

    pub(super) fn into_subagent(self) -> (SubAgent, Option<String>, String, bool) {
        let task = self.task.trim().to_string();
        let task = if task.is_empty() { None } else { Some(task) };
        let session = self.session.trim().to_ascii_lowercase();
        let persist = self.persist;
        let agent = SubAgent {
            name: self.name.trim().to_string(),
            description: self.description,
            instructions: self.instructions,
            tools: self.tools,
            tags: self.tags,
            output_schema: self.output_schema,
            model: self.model.trim().to_string(),
            effort: self.effort,
            idle_timeout: self.idle_timeout,
            ..Default::default()
        };

        (agent, task, session, persist)
    }
}
