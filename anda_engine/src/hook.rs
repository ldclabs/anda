//! Hook system for observing and customizing runtime behavior.
//!
//! Hooks let applications wrap agent and tool execution without changing the
//! agent or tool implementations themselves. Engine-level hooks are registered
//! through [`EngineBuilder::with_hooks`](crate::engine::EngineBuilder::with_hooks),
//! while typed tool and agent hooks are stored on [`BaseCtx`] state and consumed
//! by specific extensions.

use anda_core::{
    AgentOutput, BoxError, CacheExpiry, CacheFeatures, CompletionRequest, Json, Resource,
    StateFeatures, ToolOutput,
};
use async_trait::async_trait;
use core::{fmt, str::FromStr};
use std::{sync::Arc, time::Duration};
use structured_logger::unix_ms;

use crate::context::{AgentCtx, BaseCtx};

/// Identifier split into a namespace prefix and local ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrefixedId {
    /// Namespace or hook prefix.
    pub prefix: String,
    /// Local identifier within the prefix.
    pub id: String,
}

impl fmt::Display for PrefixedId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.prefix, self.id)
    }
}

impl FromStr for PrefixedId {
    type Err = BoxError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid PrefixedId format: {}", s).into());
        }
        if parts[0].trim().is_empty() || parts[1].trim().is_empty() {
            return Err(format!("Prefix and ID cannot be empty: {}", s).into());
        }
        if parts[0].trim() != parts[0] || parts[1].trim() != parts[1] {
            return Err(format!(
                "Prefix and ID cannot have leading or trailing whitespace: {}",
                s
            )
            .into());
        }
        Ok(Self {
            prefix: parts[0].to_string(),
            id: parts[1].to_string(),
        })
    }
}

/// Engine-level hook for agent runs and direct tool calls.
///
/// Returning an error from a start hook aborts execution. End hooks may inspect
/// or replace the output before it is returned to the caller.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Called before an agent is executed.
    async fn on_agent_start(&self, _ctx: &AgentCtx, _agent: &str) -> Result<(), BoxError> {
        Ok(())
    }

    /// Called after an agent is executed.
    async fn on_agent_end(
        &self,
        _ctx: &AgentCtx,
        _agent: &str,
        output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        Ok(output)
    }

    /// Called before a tool is called.
    async fn on_tool_start(&self, _ctx: &BaseCtx, _tool: &str) -> Result<(), BoxError> {
        Ok(())
    }

    /// Called after a tool is called.
    async fn on_tool_end(
        &self,
        _ctx: &BaseCtx,
        _tool: &str,
        output: ToolOutput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        Ok(output)
    }
}

/// Typed hook for a specific tool's input and output types.
///
/// Tool hooks provide fine-grained interception for extension tools. They can
/// transform arguments before execution, transform results after execution, and
/// observe background task lifecycle events.
#[async_trait]
pub trait ToolHook<I, O>: Send + Sync
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// This method is called before a tool is called, allowing you to modify the input arguments.
    async fn before_tool_call(&self, _ctx: &BaseCtx, args: I) -> Result<I, BoxError> {
        Ok(args)
    }

    /// This method is called after a tool is called, allowing you to modify the output.
    async fn after_tool_call(
        &self,
        _ctx: &BaseCtx,
        output: ToolOutput<O>,
    ) -> Result<ToolOutput<O>, BoxError> {
        Ok(output)
    }

    /// Called when a tool starts a background task.
    async fn on_background_start(&self, _ctx: &BaseCtx, _task_id: &str, _args: &I) {}

    /// Called when a tool reports progress from a background task.
    async fn on_background_progress(
        &self,
        _ctx: &BaseCtx,
        _task_id: String,
        _output: ToolOutput<O>,
    ) {
    }

    /// Called with the final output from a background tool task.
    async fn on_background_end(&self, _ctx: &BaseCtx, _task_id: String, _output: ToolOutput<O>) {}
}

/// Type-erased background hook used when a tool's typed arguments are not available.
#[async_trait]
pub trait ToolBackgroundHook: Send + Sync {
    /// Called when a tool starts a background task.
    async fn on_background_start(&self, _ctx: &BaseCtx, _task_id: &str, _args: Json) {}

    /// Called when a tool reports progress from a background task.
    async fn on_background_progress(
        &self,
        _ctx: &BaseCtx,
        _task_id: String,
        _output: ToolOutput<Json>,
    ) {
    }

    /// Called with the final output from a background tool task.
    async fn on_background_end(&self, _ctx: &BaseCtx, _task_id: String, _output: ToolOutput<Json>) {
    }
}

/// Cloneable type-erased wrapper for a typed [`ToolHook`].
#[derive(Clone)]
pub struct DynToolHook<I, O> {
    inner: Arc<dyn ToolHook<I, O>>,
}

impl<I, O> DynToolHook<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Wraps a concrete hook implementation.
    pub fn new(inner: Arc<dyn ToolHook<I, O>>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<I, O> ToolHook<I, O> for DynToolHook<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn before_tool_call(&self, ctx: &BaseCtx, args: I) -> Result<I, BoxError> {
        self.inner.before_tool_call(ctx, args).await
    }

    async fn after_tool_call(
        &self,
        ctx: &BaseCtx,
        output: ToolOutput<O>,
    ) -> Result<ToolOutput<O>, BoxError> {
        self.inner.after_tool_call(ctx, output).await
    }

    async fn on_background_start(&self, ctx: &BaseCtx, task_id: &str, args: &I) {
        self.inner.on_background_start(ctx, task_id, args).await;
    }

    async fn on_background_progress(&self, ctx: &BaseCtx, task_id: String, _output: ToolOutput<O>) {
        self.inner
            .on_background_progress(ctx, task_id, _output)
            .await;
    }

    async fn on_background_end(&self, ctx: &BaseCtx, task_id: String, output: ToolOutput<O>) {
        self.inner.on_background_end(ctx, task_id, output).await;
    }
}

/// Type-erased wrapper for [`ToolBackgroundHook`].
#[derive(Clone)]
pub struct DynToolJsonHook {
    inner: Arc<dyn ToolBackgroundHook>,
}

impl DynToolJsonHook {
    /// Wraps a concrete hook implementation.
    pub fn new(inner: Arc<dyn ToolBackgroundHook>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl ToolBackgroundHook for DynToolJsonHook {
    async fn on_background_start(&self, ctx: &BaseCtx, task_id: &str, args: Json) {
        self.inner.on_background_start(ctx, task_id, args).await;
    }

    async fn on_background_progress(
        &self,
        ctx: &BaseCtx,
        task_id: String,
        _output: ToolOutput<Json>,
    ) {
        self.inner
            .on_background_progress(ctx, task_id, _output)
            .await;
    }

    async fn on_background_end(&self, ctx: &BaseCtx, task_id: String, output: ToolOutput<Json>) {
        self.inner.on_background_end(ctx, task_id, output).await;
    }
}

/// Typed hook for nested agent calls and agent-runner extensions.
#[async_trait]
pub trait AgentHook: Send + Sync {
    /// Called before an agent is executed, allowing you to modify the prompt and resources.
    async fn before_agent_run(
        &self,
        _ctx: &AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<(String, Vec<Resource>), BoxError> {
        Ok((prompt, resources))
    }

    /// Called after an agent is executed, allowing you to modify the output.
    ///
    /// For background execution, implementations can use
    /// [`AgentHook::on_background_end`] to observe the final result.
    async fn after_agent_run(
        &self,
        _ctx: &AgentCtx,
        output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        Ok(output)
    }

    /// Called when an agent starts in the background. The session ID can be used to correlate progress and final output hooks.
    async fn on_background_start(
        &self,
        _ctx: &AgentCtx,
        _session_id: &str,
        _req: &CompletionRequest,
    ) {
    }

    /// Called when an agent reports progress from a background task.
    async fn on_background_progress(
        &self,
        _ctx: &AgentCtx,
        _session_id: String,
        _progress: AgentOutput,
    ) {
    }

    /// Called with the final output from a background agent task.
    async fn on_background_end(&self, _ctx: &AgentCtx, _session_id: String, _output: AgentOutput) {}
}

/// Cloneable type-erased wrapper for an [`AgentHook`].
#[derive(Clone)]
pub struct DynAgentHook {
    inner: Arc<dyn AgentHook>,
}

impl DynAgentHook {
    /// Wraps a concrete agent hook implementation.
    pub fn new(inner: Arc<dyn AgentHook>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl AgentHook for DynAgentHook {
    async fn before_agent_run(
        &self,
        ctx: &AgentCtx,
        prompt: String,
        resources: Vec<Resource>,
    ) -> Result<(String, Vec<Resource>), BoxError> {
        self.inner.before_agent_run(ctx, prompt, resources).await
    }

    async fn after_agent_run(
        &self,
        ctx: &AgentCtx,
        output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        self.inner.after_agent_run(ctx, output).await
    }

    async fn on_background_start(&self, ctx: &AgentCtx, session_id: &str, req: &CompletionRequest) {
        self.inner.on_background_start(ctx, session_id, req).await;
    }

    async fn on_background_progress(
        &self,
        ctx: &AgentCtx,
        session_id: String,
        progress: AgentOutput,
    ) {
        self.inner
            .on_background_progress(ctx, session_id, progress)
            .await;
    }

    async fn on_background_end(&self, ctx: &AgentCtx, session_id: String, output: AgentOutput) {
        self.inner.on_background_end(ctx, session_id, output).await;
    }
}

/// Ordered collection of engine-level hooks.
///
/// Hooks run in insertion order. End hooks receive the output from the previous
/// hook, so each hook can transform the value before the next one observes it.
pub struct Hooks {
    hooks: Vec<Box<dyn Hook>>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self::new()
    }
}

impl Hooks {
    /// Creates an empty hook collection.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Adds a hook to the end of the collection.
    pub fn add(&mut self, hook: Box<dyn Hook>) {
        self.hooks.push(hook);
    }
}

#[async_trait]
impl Hook for Hooks {
    async fn on_agent_start(&self, ctx: &AgentCtx, agent: &str) -> Result<(), BoxError> {
        for hook in &self.hooks {
            hook.on_agent_start(ctx, agent).await?;
        }
        Ok(())
    }

    async fn on_agent_end(
        &self,
        ctx: &AgentCtx,
        agent: &str,
        mut output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        for hook in &self.hooks {
            output = hook.on_agent_end(ctx, agent, output).await?;
        }
        Ok(output)
    }

    async fn on_tool_start(&self, ctx: &BaseCtx, tool: &str) -> Result<(), BoxError> {
        for hook in &self.hooks {
            hook.on_tool_start(ctx, tool).await?;
        }
        Ok(())
    }

    async fn on_tool_end(
        &self,
        ctx: &BaseCtx,
        tool: &str,
        mut output: ToolOutput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        for hook in &self.hooks {
            output = hook.on_tool_end(ctx, tool, output).await?;
        }
        Ok(output)
    }
}

/// Hook that limits each caller to one active agent prompt at a time.
///
/// The hook stores a per-caller lease in the context cache when an agent starts
/// and removes it when the agent ends. The TTL prevents stale leases from
/// blocking a caller forever after an interrupted process.
pub struct SingleThreadHook {
    ttl: Duration,
}

impl SingleThreadHook {
    /// Creates a single-thread hook with the lease time-to-live.
    pub fn new(ttl: Duration) -> Self {
        Self { ttl }
    }
}

#[async_trait]
impl Hook for SingleThreadHook {
    async fn on_agent_start(&self, ctx: &AgentCtx, _agent: &str) -> Result<(), BoxError> {
        let caller = ctx.caller();
        let now_ms = unix_ms();
        let ok = ctx
            .cache_set_if_not_exists(
                caller.to_string().as_str(),
                (now_ms, Some(CacheExpiry::TTL(self.ttl))),
            )
            .await;
        if !ok {
            return Err("Only one prompt can run at a time.".into());
        }
        Ok(())
    }

    async fn on_agent_end(
        &self,
        ctx: &AgentCtx,
        _agent: &str,
        output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        let caller = ctx.caller();
        ctx.cache_delete(caller.to_string().as_str()).await;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::{AgentSet, CancellationToken, ToolProviderSet, ToolSet};
    use parking_lot::Mutex;

    use crate::{
        context::{RemoteEngines, Web3SDK},
        model::Models,
        store::{InMemory, Store},
        subagent::SubAgentSetManager,
    };

    fn base_ctx() -> BaseCtx {
        BaseCtx::new(
            candid::Principal::self_authenticating([9; 32]),
            "engine".to_string(),
            "agent".to_string(),
            CancellationToken::new(),
            std::collections::BTreeSet::from([anda_core::Path::default()]),
            Arc::new(Web3SDK::not_implemented()),
            Store::new(Arc::new(InMemory::new())),
            Arc::new(RemoteEngines::new()),
        )
    }

    fn agent_ctx() -> AgentCtx {
        AgentCtx::new(
            base_ctx(),
            Arc::new(Models::default()),
            Arc::new(ToolSet::new()),
            Arc::new(ToolProviderSet::new()),
            Arc::new(AgentSet::new()),
            Arc::new(SubAgentSetManager::new()),
        )
    }

    #[test]
    fn prefixed_id_display_and_parse_validate_shape() {
        let id: PrefixedId = "agent:run-1".parse().unwrap();
        assert_eq!(id.prefix, "agent");
        assert_eq!(id.id, "run-1");
        assert_eq!(id.to_string(), "agent:run-1");

        assert!("missing_separator".parse::<PrefixedId>().is_err());
        assert!(":empty_prefix".parse::<PrefixedId>().is_err());
        assert!("empty_id:".parse::<PrefixedId>().is_err());
        assert!(" agent:run".parse::<PrefixedId>().is_err());
        assert!("agent:run ".parse::<PrefixedId>().is_err());
    }

    struct AppendHook(&'static str);

    #[async_trait]
    impl Hook for AppendHook {
        async fn on_agent_start(&self, _ctx: &AgentCtx, agent: &str) -> Result<(), BoxError> {
            assert!(!agent.is_empty());
            Ok(())
        }

        async fn on_agent_end(
            &self,
            _ctx: &AgentCtx,
            _agent: &str,
            mut output: AgentOutput,
        ) -> Result<AgentOutput, BoxError> {
            output.content.push_str(self.0);
            Ok(output)
        }

        async fn on_tool_start(&self, _ctx: &BaseCtx, tool: &str) -> Result<(), BoxError> {
            assert!(!tool.is_empty());
            Ok(())
        }

        async fn on_tool_end(
            &self,
            _ctx: &BaseCtx,
            _tool: &str,
            mut output: ToolOutput<Json>,
        ) -> Result<ToolOutput<Json>, BoxError> {
            let current = output.output.as_str().unwrap_or_default().to_string();
            output.output = Json::String(format!("{current}{}", self.0));
            Ok(output)
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn hooks_apply_start_checks_and_end_transforms_in_order() {
        let agent_ctx = agent_ctx();
        let base_ctx = base_ctx();
        let mut hooks = Hooks::new();
        hooks.add(Box::new(AppendHook("-one")));
        hooks.add(Box::new(AppendHook("-two")));

        hooks.on_agent_start(&agent_ctx, "worker").await.unwrap();
        let output = hooks
            .on_agent_end(
                &agent_ctx,
                "worker",
                AgentOutput {
                    content: "done".to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(output.content, "done-one-two");

        hooks.on_tool_start(&base_ctx, "tool").await.unwrap();
        let output = hooks
            .on_tool_end(
                &base_ctx,
                "tool",
                ToolOutput::new(Json::String("ok".to_string())),
            )
            .await
            .unwrap();
        assert_eq!(output.output, Json::String("ok-one-two".to_string()));
    }

    struct PrefixToolHook {
        events: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl ToolHook<String, String> for PrefixToolHook {
        async fn before_tool_call(&self, _ctx: &BaseCtx, args: String) -> Result<String, BoxError> {
            Ok(format!("before:{args}"))
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut output: ToolOutput<String>,
        ) -> Result<ToolOutput<String>, BoxError> {
            output.output = format!("after:{}", output.output);
            Ok(output)
        }

        async fn on_background_start(&self, _ctx: &BaseCtx, task_id: &str, _args: &String) {
            self.events.lock().push(format!("start:{task_id}"));
        }

        async fn on_background_progress(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            _output: ToolOutput<String>,
        ) {
            self.events.lock().push(format!("progress:{task_id}"));
        }

        async fn on_background_end(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            _output: ToolOutput<String>,
        ) {
            self.events.lock().push(format!("end:{task_id}"));
        }
    }

    struct JsonBackgroundHook {
        events: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl ToolBackgroundHook for JsonBackgroundHook {
        async fn on_background_start(&self, _ctx: &BaseCtx, task_id: &str, _args: Json) {
            self.events.lock().push(format!("json-start:{task_id}"));
        }

        async fn on_background_progress(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            _output: ToolOutput<Json>,
        ) {
            self.events.lock().push(format!("json-progress:{task_id}"));
        }

        async fn on_background_end(
            &self,
            _ctx: &BaseCtx,
            task_id: String,
            _output: ToolOutput<Json>,
        ) {
            self.events.lock().push(format!("json-end:{task_id}"));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dynamic_tool_hooks_forward_typed_and_json_background_calls() {
        let ctx = base_ctx();
        let events = Arc::new(Mutex::new(Vec::new()));
        let hook = DynToolHook::new(Arc::new(PrefixToolHook {
            events: events.clone(),
        }));

        let args = hook
            .before_tool_call(&ctx, "input".to_string())
            .await
            .unwrap();
        assert_eq!(args, "before:input");
        let output = hook
            .after_tool_call(&ctx, ToolOutput::new("value".to_string()))
            .await
            .unwrap();
        assert_eq!(output.output, "after:value");
        hook.on_background_start(&ctx, "task", &"args".to_string())
            .await;
        hook.on_background_progress(&ctx, "task".to_string(), ToolOutput::new("p".to_string()))
            .await;
        hook.on_background_end(&ctx, "task".to_string(), ToolOutput::new("e".to_string()))
            .await;
        assert_eq!(
            events.lock().clone(),
            vec!["start:task", "progress:task", "end:task"]
        );

        let json_events = Arc::new(Mutex::new(Vec::new()));
        let json_hook = DynToolJsonHook::new(Arc::new(JsonBackgroundHook {
            events: json_events.clone(),
        }));
        json_hook
            .on_background_start(&ctx, "json-task", Json::Null)
            .await;
        json_hook
            .on_background_progress(&ctx, "json-task".to_string(), ToolOutput::new(Json::Null))
            .await;
        json_hook
            .on_background_end(&ctx, "json-task".to_string(), ToolOutput::new(Json::Null))
            .await;
        assert_eq!(
            json_events.lock().clone(),
            vec![
                "json-start:json-task",
                "json-progress:json-task",
                "json-end:json-task"
            ]
        );
    }

    struct PrefixAgentHook {
        events: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl AgentHook for PrefixAgentHook {
        async fn before_agent_run(
            &self,
            _ctx: &AgentCtx,
            prompt: String,
            mut resources: Vec<Resource>,
        ) -> Result<(String, Vec<Resource>), BoxError> {
            resources.push(Resource {
                name: "added".to_string(),
                tags: vec!["text".to_string()],
                ..Default::default()
            });
            Ok((format!("before:{prompt}"), resources))
        }

        async fn after_agent_run(
            &self,
            _ctx: &AgentCtx,
            mut output: AgentOutput,
        ) -> Result<AgentOutput, BoxError> {
            output.content = format!("after:{}", output.content);
            Ok(output)
        }

        async fn on_background_start(
            &self,
            _ctx: &AgentCtx,
            session_id: &str,
            _req: &CompletionRequest,
        ) {
            self.events.lock().push(format!("agent-start:{session_id}"));
        }

        async fn on_background_progress(
            &self,
            _ctx: &AgentCtx,
            session_id: String,
            _progress: AgentOutput,
        ) {
            self.events
                .lock()
                .push(format!("agent-progress:{session_id}"));
        }

        async fn on_background_end(
            &self,
            _ctx: &AgentCtx,
            session_id: String,
            _output: AgentOutput,
        ) {
            self.events.lock().push(format!("agent-end:{session_id}"));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dynamic_agent_hook_forwards_prompt_output_and_background_calls() {
        let ctx = agent_ctx();
        let events = Arc::new(Mutex::new(Vec::new()));
        let hook = DynAgentHook::new(Arc::new(PrefixAgentHook {
            events: events.clone(),
        }));
        let (prompt, resources) = hook
            .before_agent_run(&ctx, "input".to_string(), Vec::new())
            .await
            .unwrap();
        assert_eq!(prompt, "before:input");
        assert_eq!(resources.len(), 1);

        let output = hook
            .after_agent_run(
                &ctx,
                AgentOutput {
                    content: "value".to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(output.content, "after:value");
        hook.on_background_start(&ctx, "session", &CompletionRequest::default())
            .await;
        hook.on_background_progress(&ctx, "session".to_string(), AgentOutput::default())
            .await;
        hook.on_background_end(&ctx, "session".to_string(), AgentOutput::default())
            .await;
        assert_eq!(
            events.lock().clone(),
            vec![
                "agent-start:session",
                "agent-progress:session",
                "agent-end:session"
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn single_thread_hook_rejects_second_prompt_until_end_releases_lease() {
        let ctx = agent_ctx().with_caller(candid::Principal::self_authenticating([4; 32]));
        let hook = SingleThreadHook::new(Duration::from_secs(30));

        hook.on_agent_start(&ctx, "agent").await.unwrap();
        assert!(
            hook.on_agent_start(&ctx, "agent")
                .await
                .unwrap_err()
                .to_string()
                .contains("Only one prompt")
        );
        let output = hook
            .on_agent_end(
                &ctx,
                "agent",
                AgentOutput {
                    content: "done".to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(output.content, "done");
        hook.on_agent_start(&ctx, "agent").await.unwrap();
    }
}
