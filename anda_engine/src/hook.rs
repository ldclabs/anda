//! Hook system for customizing engine behavior.

use anda_core::{
    AgentOutput, BoxError, CacheExpiry, CacheFeatures, Json, StateFeatures, ToolOutput,
};
use async_trait::async_trait;
use std::time::Duration;
use structured_logger::unix_ms;

use crate::context::{AgentCtx, BaseCtx};

/// Hook trait for customizing engine behavior.
/// Hooks can be used to intercept and modify agent and tool execution.
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

/// Hooks struct for managing multiple hooks.
pub struct Hooks {
    hooks: Vec<Box<dyn Hook>>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self::new()
    }
}

impl Hooks {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Adds a new hook to the list of hooks.
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

pub struct SingleThreadHook {
    ttl: Duration,
}

impl SingleThreadHook {
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
