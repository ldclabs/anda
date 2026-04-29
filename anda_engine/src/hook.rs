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
use std::{sync::Arc, time::Duration};
use structured_logger::unix_ms;

use crate::context::{AgentCtx, BaseCtx};

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

    /// Called with the final output from a background tool task.
    async fn on_background_end(&self, _ctx: BaseCtx, _task_id: String, _output: ToolOutput<O>) {}
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

    async fn on_background_end(&self, ctx: BaseCtx, task_id: String, output: ToolOutput<O>) {
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

    /// Called when an agent starts a background task.
    async fn on_background_start(&self, _ctx: &AgentCtx, _task_id: &str, _req: &CompletionRequest) {
    }

    /// Called with the final output from a background agent task.
    async fn on_background_end(&self, _ctx: AgentCtx, _task_id: String, _output: AgentOutput) {}
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

    async fn on_background_start(&self, ctx: &AgentCtx, task_id: &str, req: &CompletionRequest) {
        self.inner.on_background_start(ctx, task_id, req).await;
    }

    async fn on_background_end(&self, ctx: AgentCtx, task_id: String, output: AgentOutput) {
        self.inner.on_background_end(ctx, task_id, output).await;
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
