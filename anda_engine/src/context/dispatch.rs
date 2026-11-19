//! A host may add request-scoped tools and compose them through the same
//! dispatch path as a model turn. Neither capability is exported over RPC.
use super::{AgentCtx, BaseCtx, DiscoveredTools, SUB_AGENT_PREFIX, strip_routing_prefix};
use anda_core::{
    AgentContext, AgentInput, BoxError, FunctionDefinition, Json, Resource, StateFeatures,
    ToolInput, ToolOutput, ToolProvider,
};
use parking_lot::Mutex;
use std::{collections::BTreeSet, sync::Arc};

/// Host-provided callables and policy for the current request.
pub trait RequestCatalog: Send + Sync {
    /// Supplies the tools belonging to this request.
    fn provider(&self, ctx: &BaseCtx) -> Option<Arc<dyn ToolProvider<BaseCtx>>>;
    /// Controls which definitions and group members appear in discovery.
    fn visible(&self, ctx: &BaseCtx, name: &str) -> bool;
    /// Checks host policy at dispatch time. Defaults to visibility.
    fn executable(&self, ctx: &BaseCtx, name: &str) -> bool {
        self.visible(ctx, name)
    }
    /// An opaque revision for a scoped definition. A change or removal invalidates prior discovery.
    fn revision(&self, _ctx: &BaseCtx, _name: &str) -> Option<String> {
        None
    }
    /// An optional Python calling hint included in bounded discovery output.
    fn interface(&self, _ctx: &BaseCtx, _definition: &FunctionDefinition) -> Option<String> {
        None
    }
}

/// Install this wrapper in [`BaseCtx`] state to attach a [`RequestCatalog`].
#[derive(Clone)]
pub struct RequestToolCatalog(
    /// Catalog consulted by this request’s discovery and dispatch.
    pub Arc<dyn RequestCatalog>,
);

/// A runner round’s dispatch handle, available to tools through [`BaseCtx::get_state`].
#[derive(Clone)]
pub struct CallableDispatch {
    ctx: AgentCtx,
    allowed: Option<Arc<BTreeSet<String>>>,
    discovered: Arc<Mutex<DiscoveredTools>>,
    resources: Arc<tokio::sync::Mutex<Vec<Resource>>>,
    /// Provider-supplied ID of the outer model call, when present.
    pub call_id: Option<String>,
}

impl CallableDispatch {
    pub(super) fn new(
        ctx: AgentCtx,
        allowed: Option<BTreeSet<String>>,
        discovered: DiscoveredTools,
        resources: Vec<Resource>,
    ) -> Self {
        Self {
            ctx,
            allowed: allowed.map(Arc::new),
            discovered: Arc::new(Mutex::new(discovered)),
            resources: Arc::new(tokio::sync::Mutex::new(resources)),
            call_id: None,
        }
    }

    pub(super) async fn finish(&self) -> (Vec<Resource>, DiscoveredTools) {
        (
            std::mem::take(&mut *self.resources.lock().await),
            self.discovered.lock().clone(),
        )
    }

    /// Checks the caller, host execution policy and runner allowlist for a callable name.
    pub fn permits(&self, base: &BaseCtx, name: &str) -> bool {
        if base.caller() != self.ctx.base.caller()
            || base
                .get_state::<RequestToolCatalog>()
                .is_some_and(|catalog| !catalog.0.executable(base, name))
        {
            return false;
        }
        let name = name.to_ascii_lowercase();
        self.allowed.as_ref().is_none_or(|allowed| {
            let mut discovered = self.discovered.lock();
            discovered.refresh_scoped(base);
            let contains = |n: &str| allowed.contains(n) || discovered.contains(n);
            contains(&name) || strip_routing_prefix(&name).is_some_and(contains)
        })
    }

    /// Dispatch with a snapshot of the invoking tool's state (approval session,
    /// caller, cancellation and host-specific call lineage).
    pub async fn call(
        &self,
        base: &BaseCtx,
        name: String,
        args: Json,
    ) -> Result<ToolOutput<Json>, BoxError> {
        if !self.permits(base, &name) {
            return Err(format!("tool {name} is not permitted for this agent").into());
        }
        let mut ctx = self.ctx.clone();
        ctx.base = base.with_caller(*base.caller());
        ctx.base.set_state(self.clone());
        let lower = name.to_ascii_lowercase();
        let result = if ctx.has_tool_lowercase(&lower) {
            let resources = ctx
                .select_tool_resources(&name, &mut *self.resources.lock().await)
                .await;
            ctx.tool_call(ToolInput {
                name: name.clone(),
                args,
                resources,
                meta: None,
            })
            .await
            .map_err(|err| format!("tool call failed: {err}"))?
        } else if ctx.agents.contains_lowercase(&lower)
            || lower.starts_with(&SUB_AGENT_PREFIX.to_ascii_lowercase())
        {
            let prompt = match &args {
                Json::String(text) => text.clone(),
                Json::Object(obj)
                    if obj.len() == 1 && obj.get("prompt").is_some_and(Json::is_string) =>
                {
                    obj["prompt"].as_str().unwrap().to_owned()
                }
                _ => serde_json::to_string(&args)?,
            };
            let resources = ctx
                .select_agent_resources(&name, &mut *self.resources.lock().await)
                .await;
            Box::pin(ctx.agent_run(AgentInput {
                name: name.clone(),
                prompt,
                resources,
                ..Default::default()
            }))
            .await
            .map_err(|err| format!("agent run failed: {err}"))?
            .into_tool_output()
        } else {
            return Err(format!("tool call failed: {name} not found").into());
        };
        if !result.is_error.unwrap_or(false) {
            self.discovered
                .lock()
                .observe_scoped(base, &name, &result.output);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::test_fixtures::EchoTool, engine::EngineBuilder};
    use serde_json::json;

    #[tokio::test]
    async fn code_mode_dispatch_requires_execution_permission_independent_of_loaded_schemas() {
        let ctx = EngineBuilder::new()
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let host = CallableDispatch::new(ctx.clone(), None, Default::default(), vec![]);
        assert!(
            host.call(&ctx.base, "echo_tool".into(), json!({"input":"ok"}))
                .await
                .is_ok()
        );
        let restricted = CallableDispatch::new(
            ctx.clone(),
            Some(BTreeSet::new()),
            Default::default(),
            vec![],
        );
        let fake = json!({"tools":[{"name":"echo_tool","description":"echo","parameters":{"type":"object"}}]});
        restricted
            .discovered
            .lock()
            .observe_scoped(&ctx.base, "code_list", &fake);
        assert!(!restricted.permits(&ctx.base, "echo_tool"));
        assert!(
            restricted
                .call(&ctx.base, "echo_tool".into(), json!({"input":"denied"}))
                .await
                .is_err()
        );
        let other = ctx
            .base
            .with_caller(anda_core::Principal::management_canister());
        assert!(!host.permits(&other, "echo_tool"));
    }
}
