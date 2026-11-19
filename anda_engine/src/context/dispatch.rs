//! A host may add request-scoped tools and compose them through the same
//! dispatch path as a model turn. Neither capability is exported over RPC.
use super::{AgentCtx, BaseCtx, DiscoveredTools, strip_routing_prefix, tool::observes_discovery};
use anda_core::{
    AgentContext, AgentInput, BoxError, BoxPinFut, CancellationToken, FunctionDefinition, Json,
    Resource, ToolInput, ToolOutput, ToolProvider,
};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Host-provided callables and policy for the current request.
///
/// Name-based callbacks receive lowercase callable names including any routing prefix.
pub trait RequestCatalog: Send + Sync {
    /// Supplies the tools belonging to this request.
    fn provider(&self, ctx: &BaseCtx) -> Option<Arc<dyn ToolProvider<BaseCtx>>>;
    /// Controls which definitions and group members appear in discovery.
    fn visible(&self, ctx: &BaseCtx, name: &str) -> bool;
    /// Checks host policy at dispatch time. Defaults to visibility.
    fn executable(&self, ctx: &BaseCtx, name: &str) -> bool {
        self.visible(ctx, name)
    }
    /// An optional revision for a scoped definition. Revision changes and removal invalidate prior discovery.
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

impl RequestToolCatalog {
    pub(super) fn is_visible(&self, ctx: &BaseCtx, name: &str) -> bool {
        self.0.visible(ctx, &name.to_ascii_lowercase())
    }

    pub(super) fn permits_execution(ctx: &BaseCtx, name: &str) -> bool {
        ctx.get_state::<Self>()
            .is_none_or(|catalog| catalog.0.executable(ctx, &name.to_ascii_lowercase()))
    }
}

pub(super) fn callable_allowed(
    allowed: Option<&BTreeSet<String>>,
    discovered: &DiscoveredTools,
    lowercase_name: &str,
) -> bool {
    allowed.is_none_or(|allowed| {
        let contains = |name: &str| allowed.contains(name) || discovered.contains(name);
        contains(lowercase_name) || strip_routing_prefix(lowercase_name).is_some_and(contains)
    })
}

/// A runner round’s dispatch handle, available to tools through [`BaseCtx::get_state`].
#[derive(Clone)]
pub struct CallableDispatch {
    ctx: AgentCtx,
    allowed: Option<Arc<BTreeSet<String>>>,
    discovered: Arc<Mutex<DiscoveredTools>>,
    resources: Arc<tokio::sync::Mutex<Vec<Resource>>>,
    pub(super) cancellation: CancellationToken,
    /// Provider-supplied ID of the outer model call, when present.
    pub call_id: Option<String>,
}

impl CallableDispatch {
    pub(super) fn new(
        ctx: AgentCtx,
        allowed: Option<BTreeSet<String>>,
        discovered: DiscoveredTools,
        resources: Arc<tokio::sync::Mutex<Vec<Resource>>>,
    ) -> Self {
        Self {
            cancellation: ctx.base.cancellation_token.child_token(),
            ctx,
            allowed: allowed.map(Arc::new),
            discovered: Arc::new(Mutex::new(discovered)),
            resources,
            call_id: None,
        }
    }

    pub(super) fn take_discovered(&self) -> DiscoveredTools {
        std::mem::take(&mut *self.discovered.lock())
    }

    /// Checks request identity, host execution policy and the runner allowlist.
    pub fn permits(&self, base: &BaseCtx, name: &str) -> bool {
        if self.cancellation.is_cancelled()
            || !self.ctx.base.shares_request(base)
            || !RequestToolCatalog::permits_execution(base, name)
        {
            return false;
        }
        let mut discovered = self.discovered.lock();
        discovered.refresh_scoped(base);
        callable_allowed(
            self.allowed.as_deref(),
            &discovered,
            &name.to_ascii_lowercase(),
        )
    }

    /// Dispatch with a snapshot of the invoking tool's state in the same request.
    /// The call follows cancellation of either the invoking context or the runner round.
    pub async fn call(
        &self,
        base: &BaseCtx,
        name: String,
        args: Json,
    ) -> Result<ToolOutput<Json>, BoxError> {
        self.prepare_call(base, name, args).await?.await
    }

    // Reserve all direct model calls' inputs before starting any of them.
    pub(super) async fn prepare_call(
        &self,
        base: &BaseCtx,
        name: String,
        args: Json,
    ) -> Result<BoxPinFut<Result<ToolOutput<Json>, BoxError>>, BoxError> {
        if self.cancellation.is_cancelled() || base.cancellation_token.is_cancelled() {
            return Err("tool call cancelled".into());
        }
        if !self.permits(base, &name) {
            return Err(format!("tool {name} is not permitted for this agent").into());
        }
        let mut ctx = self.ctx.clone();
        ctx.base = base.with_caller(base.caller);
        ctx.base.set_state(self.clone());
        let lower = name.to_ascii_lowercase();
        let is_tool = ctx.has_tool_lowercase(&lower);
        if !is_tool
            && !ctx.agents.contains_lowercase(&lower)
            && strip_routing_prefix(&lower).is_none()
        {
            return Err(format!("tool call failed: {name} not found").into());
        }
        let resources = {
            let mut pool = self.resources.lock().await;
            if is_tool {
                ctx.select_tool_resources(&name, &mut pool).await
            } else {
                ctx.select_agent_resources(&name, &mut pool).await
            }
        };
        let cancellation = base.cancellation_token.clone();
        let dispatch = self.clone();
        Ok(Box::pin(async move {
            let run = async {
                if !dispatch.permits(&ctx.base, &name) {
                    return Err(format!("tool {name} is not permitted for this agent").into());
                }
                // Capture revisions before discovery can await its selector model.
                let revisions = observes_discovery(&name).then(|| scoped_revisions(&ctx));
                let result = if is_tool {
                    ctx.tool_call(ToolInput {
                        name: name.clone(),
                        args,
                        resources,
                        meta: None,
                    })
                    .await
                    .map_err(|err| format!("tool call failed: {err}"))?
                } else {
                    let prompt = if let Some(prompt) = args.as_str() {
                        prompt.to_owned()
                    } else if let Some(object) = args.as_object()
                        && object.len() == 1
                        && let Some(prompt) = object.get("prompt").and_then(Json::as_str)
                    {
                        prompt.to_owned()
                    } else {
                        args.to_string()
                    };
                    ctx.agent_run(AgentInput {
                        name: name.clone(),
                        prompt,
                        resources,
                        ..Default::default()
                    })
                    .await
                    .map_err(|err| format!("agent run failed: {err}"))?
                    .into_tool_output()
                };
                if let Some(revisions) = revisions
                    && !result.is_error.unwrap_or(false)
                {
                    dispatch
                        .discovered
                        .lock()
                        .observe_scoped(&name, &result.output, &revisions);
                }
                Ok(result)
            };
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err("tool call cancelled".into()),
                _ = dispatch.cancellation.cancelled() => Err("tool call cancelled".into()),
                result = run => result,
            }
        }))
    }
}

fn scoped_revisions(ctx: &AgentCtx) -> BTreeMap<String, Option<String>> {
    let Some(catalog) = ctx.base.get_state::<RequestToolCatalog>() else {
        return BTreeMap::new();
    };
    ctx.scoped_tool_definitions(None)
        .into_iter()
        .map(|d| {
            let revision = catalog.0.revision(&ctx.base, &d.name);
            (d.name, revision)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::test_fixtures::{EchoAgent, EchoTool, RecordingCompleter, ToolCallCompleter},
        engine::EngineBuilder,
        model::{CompletionFeaturesDyn, Model},
    };
    use anda_core::{
        AgentOutput, BoxFut, CompletionRequest, StateFeatures, Tool, ToolCall, ToolGroup,
    };
    use parking_lot::RwLock;
    use serde_json::json;

    struct ScopedProvider {
        definitions: RwLock<Vec<FunctionDefinition>>,
    }

    impl ToolProvider<BaseCtx> for ScopedProvider {
        fn name(&self) -> String {
            "scoped".into()
        }

        fn definitions(&self, _: Option<&[String]>) -> Vec<FunctionDefinition> {
            self.definitions.read().clone()
        }

        fn groups(&self) -> Vec<ToolGroup> {
            vec![ToolGroup {
                id: "scope".into(),
                members: self
                    .definitions
                    .read()
                    .iter()
                    .map(|def| def.name.clone())
                    .collect(),
                ..Default::default()
            }]
        }

        fn supported_resource_tags(&self, name: &str) -> Vec<String> {
            if name == "scoped_read" {
                vec!["*".into()]
            } else {
                Vec::new()
            }
        }

        fn call(
            &self,
            ctx: BaseCtx,
            input: ToolInput<Json>,
        ) -> BoxFut<'_, Result<ToolOutput<Json>, BoxError>> {
            Box::pin(async move {
                if input.name == "scoped_wait" {
                    std::future::pending::<()>().await;
                }
                let call_id = ctx.get_state::<CallableDispatch>().and_then(|d| d.call_id);
                Ok(ToolOutput::new(json!({
                    "name": input.name,
                    "resources": input.resources.len(),
                    "call_id": call_id,
                })))
            })
        }
    }

    struct ScopedCatalog(Arc<ScopedProvider>);

    impl RequestCatalog for ScopedCatalog {
        fn provider(&self, _: &BaseCtx) -> Option<Arc<dyn ToolProvider<BaseCtx>>> {
            Some(self.0.clone())
        }

        fn visible(&self, _: &BaseCtx, name: &str) -> bool {
            !matches!(name, "scoped_hidden" | "echo_agent" | "note")
        }

        fn revision(&self, _: &BaseCtx, name: &str) -> Option<String> {
            self.0
                .definitions
                .read()
                .iter()
                .find(|def| def.name != "scoped_wait" && def.name.eq_ignore_ascii_case(name))
                .map(|def| def.description.clone())
        }
    }

    fn tool_definition(name: &str, description: &str) -> FunctionDefinition {
        FunctionDefinition {
            name: name.into(),
            description: description.into(),
            parameters: json!({"type":"object","properties":{}}),
            ..Default::default()
        }
    }

    fn install_catalog(ctx: &AgentCtx) -> Arc<ScopedProvider> {
        let provider = Arc::new(ScopedProvider {
            definitions: RwLock::new(vec![
                tool_definition("SCOPED_READ", "v1"),
                tool_definition("scoped_hidden", "hidden"),
                tool_definition("echo_tool", "shadowed"),
                tool_definition("scoped_wait", "wait"),
            ]),
        });
        ctx.base
            .set_state(RequestToolCatalog(Arc::new(ScopedCatalog(
                provider.clone(),
            ))));
        provider
    }

    #[tokio::test]
    async fn code_mode_dispatch_requires_execution_permission_independent_of_loaded_schemas() {
        let ctx = EngineBuilder::new()
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let host = CallableDispatch::new(ctx.clone(), None, Default::default(), Default::default());
        assert!(
            host.call(&ctx.base, "echo_tool".into(), json!({"input":"ok"}))
                .await
                .is_ok()
        );
        let restricted = CallableDispatch::new(
            ctx.clone(),
            Some(BTreeSet::new()),
            Default::default(),
            Default::default(),
        );
        let fake = json!({"tools":[{"name":"echo_tool","description":"echo","parameters":{"type":"object"}}]});
        restricted
            .discovered
            .lock()
            .observe_output("code_list", &fake);
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

        let other_request = EngineBuilder::new().mock_ctx();
        assert_eq!(ctx.base.caller(), other_request.base.caller());
        assert!(!host.permits(&other_request.base, "echo_tool"));
        assert!(host.permits(&ctx.base.cancellable_child(), "echo_tool"));
    }

    #[tokio::test]
    async fn scoped_catalog_applies_filters_and_execution_policy() {
        let ctx = EngineBuilder::new()
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .register_agent(Arc::new(EchoAgent), None)
            .unwrap()
            .mock_ctx();
        install_catalog(&ctx);

        let definitions = ctx
            .definitions(Some(&["SCOPED_READ".into(), "scoped_hidden".into()]))
            .await;
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "scoped_read");
        assert!(ctx.tool_definitions(Some(&[])).is_empty());
        assert!(
            ctx.tool_definitions(None)
                .iter()
                .all(|def| !def.name.eq_ignore_ascii_case("scoped_hidden"))
        );
        assert!(
            ctx.agent_definitions(None)
                .iter()
                .all(|def| !def.name.eq_ignore_ascii_case("echo_agent"))
        );
        let groups = ctx.tool_groups();
        let group = groups.iter().find(|group| group.id == "scope").unwrap();
        assert_eq!(group.members, ["scoped_read", "scoped_wait"]);

        let dispatch =
            CallableDispatch::new(ctx.clone(), None, Default::default(), Default::default());
        assert!(!dispatch.permits(&ctx.base, "SCOPED_HIDDEN"));
        assert!(
            ctx.tool_call(ToolInput::new("SCOPED_HIDDEN".into(), json!({})))
                .await
                .is_err()
        );
        assert!(
            ctx.clone()
                .agent_run(AgentInput {
                    name: "ECHO_AGENT".into(),
                    ..Default::default()
                })
                .await
                .is_err()
        );
        let output = dispatch
            .call(&ctx.base, "SCOPED_READ".into(), json!({}))
            .await
            .unwrap();
        assert_eq!(output.output["name"], "scoped_read");
        let child = ctx.base.cancellable_child();
        child.cancellation_token().cancel();
        assert_eq!(
            dispatch
                .call(&child, "scoped_read".into(), json!({}))
                .await
                .unwrap_err()
                .to_string(),
            "tool call cancelled"
        );
    }

    #[tokio::test]
    async fn runner_refreshes_only_live_scoped_offers() {
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingCompleter {
            name: "scoped_offers".into(),
            requests: requests.clone(),
        }));
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(Arc::new(EchoTool))
            .unwrap()
            .mock_ctx();
        let provider = install_catalog(&ctx);
        let mut runner = ctx
            .clone()
            .completion_iter(
                CompletionRequest {
                    prompt: "initial offer".into(),
                    tools: ctx
                        .definitions(Some(&["scoped_read".into(), "echo_tool".into()]))
                        .await,
                    ..Default::default()
                },
                Vec::new(),
            )
            .unbound();
        runner.next().await.unwrap().unwrap();
        provider.definitions.write()[0].description = "v2".into();
        runner.follow_up("updated catalog".to_string());
        runner.next().await.unwrap().unwrap();
        provider.definitions.write().remove(0);
        runner.follow_up("removed scoped tool".to_string());
        runner.next().await.unwrap().unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests[0].tools[1].description, "v1");
        assert_eq!(requests[1].tools[1].description, "v2");
        assert_eq!(requests[2].tools.len(), 1);
        for request in requests.iter() {
            assert_eq!(request.tools[0].name, "echo_tool");
            assert_eq!(request.tools[0].description, EchoTool.description());
        }
    }

    #[tokio::test]
    async fn a_hidden_note_tool_does_not_inject_its_index() {
        use crate::extension::note::{NoteArgs, NoteContextConfig, NoteItemInput, NoteTool};

        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let model = Model::with_completer(Arc::new(RecordingCompleter {
            name: "hidden_note".into(),
            requests: requests.clone(),
        }));
        let note = Arc::new(NoteTool::new());
        let ctx = EngineBuilder::new()
            .with_model(model)
            .register_tool(note.clone())
            .unwrap()
            .mock_ctx();
        note.call(
            ctx.child_base(NoteTool::NAME).unwrap(),
            NoteArgs {
                op: Some("set".into()),
                items: Some(vec![NoteItemInput {
                    id: "fact".into(),
                    content: Some("stored note".into()),
                }]),
                ..Default::default()
            },
            Vec::new(),
        )
        .await
        .unwrap();
        ctx.base.set_state(NoteContextConfig::default());
        install_catalog(&ctx);

        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "start".into(),
                tools: vec![note.definition()],
                ..Default::default()
            },
            Vec::new(),
        );
        runner.next().await.unwrap().unwrap();
        let requests = requests.lock().unwrap();
        assert!(requests[0].tools.is_empty());
        assert!(requests[0].chat_history.is_empty());
    }

    struct UpdatingSelector(Arc<ScopedProvider>);

    impl CompletionFeaturesDyn for UpdatingSelector {
        fn model_name(&self) -> String {
            "updating_selector".into()
        }

        fn completion(&self, _: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
            self.0.definitions.write()[0].description = "v2".into();
            Box::pin(async {
                Ok(AgentOutput {
                    content: json!({"tools":["scoped_read"]}).to_string(),
                    ..Default::default()
                })
            })
        }
    }

    #[tokio::test]
    async fn composed_discovery_is_invalidated_if_the_catalog_changes_during_selection() {
        let ctx = EngineBuilder::new().mock_ctx();
        let provider = install_catalog(&ctx);
        ctx.models
            .set_model(Model::with_completer(Arc::new(UpdatingSelector(provider))));
        let dispatch = CallableDispatch::new(
            ctx.clone(),
            Some(BTreeSet::from(["tools_select".into()])),
            Default::default(),
            Default::default(),
        );

        let selected = dispatch
            .call(
                &ctx.base,
                "tools_select".into(),
                json!({"query":"scoped_read"}),
            )
            .await
            .unwrap();
        assert_eq!(selected.output["tools"][0]["description"], "v1");
        assert!(!dispatch.permits(&ctx.base, "scoped_read"));
        let selected = dispatch
            .call(
                &ctx.base,
                "tools_select".into(),
                json!({"tools":["scoped_read"]}),
            )
            .await
            .unwrap();
        assert_eq!(selected.output["tools"][0]["description"], "v2");
        assert!(dispatch.permits(&ctx.base, "scoped_read"));
    }

    #[tokio::test]
    async fn discovery_forgets_removed_unversioned_tools() {
        let ctx = EngineBuilder::new().mock_ctx();
        let provider = install_catalog(&ctx);
        let dispatch = CallableDispatch::new(
            ctx.clone(),
            Some(BTreeSet::from(["tools_select".into()])),
            Default::default(),
            Default::default(),
        );
        dispatch
            .call(
                &ctx.base,
                "tools_select".into(),
                json!({"tools":["scoped_wait"]}),
            )
            .await
            .unwrap();
        assert!(dispatch.permits(&ctx.base, "scoped_wait"));
        provider
            .definitions
            .write()
            .retain(|def| def.name != "scoped_wait");
        assert!(!dispatch.permits(&ctx.base, "scoped_wait"));
    }

    struct ComposeTool;

    impl Tool<BaseCtx> for ComposeTool {
        type Args = Json;
        type Output = Json;

        fn name(&self) -> String {
            "compose".into()
        }
        fn description(&self) -> String {
            "Call another tool".into()
        }
        fn definition(&self) -> FunctionDefinition {
            tool_definition(&self.name(), &self.description())
        }

        async fn call(
            &self,
            ctx: BaseCtx,
            args: Json,
            _: Vec<Resource>,
        ) -> Result<ToolOutput<Json>, BoxError> {
            let dispatch = ctx.get_state::<CallableDispatch>().unwrap();
            dispatch.call(&ctx, "scoped_read".into(), args).await
        }
    }

    fn calls_model(names: &[&str]) -> Model {
        Model::with_completer(Arc::new(ToolCallCompleter {
            tool_calls: names
                .iter()
                .map(|name| ToolCall {
                    name: name.to_string(),
                    args: json!({}),
                    call_id: Some(name.to_string()),
                    ..Default::default()
                })
                .collect(),
        }))
    }

    #[tokio::test]
    async fn composed_calls_share_resources_after_direct_calls_reserve_theirs() {
        let ctx = EngineBuilder::new()
            .with_model(calls_model(&["compose", "scoped_read"]))
            .register_tool(Arc::new(ComposeTool))
            .unwrap()
            .mock_ctx();
        install_catalog(&ctx);
        let mut runner = ctx.completion_iter(
            CompletionRequest {
                prompt: "run both tools".into(),
                ..Default::default()
            },
            vec![Resource::default()],
        );
        runner.next().await.unwrap().unwrap();
        let output = runner.next().await.unwrap().unwrap();
        let composed = &output.tool_calls[0].result.as_ref().unwrap().output;
        let direct = &output.tool_calls[1].result.as_ref().unwrap().output;
        assert_eq!(composed["call_id"], "compose");
        assert_eq!(composed["resources"], 0);
        assert_eq!(direct["resources"], 1);
    }

    #[tokio::test]
    async fn runner_keeps_unused_resources_when_a_tool_round_is_interrupted() {
        let ctx = EngineBuilder::new()
            .with_model(calls_model(&["scoped_wait"]))
            .mock_ctx();
        install_catalog(&ctx);
        let mut runner = ctx.clone().completion_iter(
            CompletionRequest {
                prompt: "start work".into(),
                ..Default::default()
            },
            vec![Resource::default()],
        );
        runner.next().await.unwrap().unwrap();
        {
            let mut step = Box::pin(runner.next());
            assert!(futures::poll!(step.as_mut()).is_pending());
        }

        runner.discard_in_flight_request();
        ctx.models.set_model(calls_model(&["scoped_read"]));
        runner.follow_up("read the attachment".to_string());
        runner.next().await.unwrap().unwrap();
        let output = runner.next().await.unwrap().unwrap();
        assert_eq!(
            output.tool_calls[0].result.as_ref().unwrap().output["resources"],
            1
        );
    }
}
