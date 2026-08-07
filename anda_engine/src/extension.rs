//! Built-in tool and agent extensions.
//!
//! Extensions are optional building blocks that can be registered with an
//! [`EngineBuilder`](crate::engine::EngineBuilder) or used directly in tests.
//! They cover common runtime needs such as web fetching, workspace filesystem
//! access, shell execution, notes, skills, and todos.
//!
//! # Key Components
//! - [`tool_definition`] and [`hooked_call`]: the shared tool-call protocol
//!   every built-in tool goes through (derived schema, hook wiring,
//!   cancellation gate, error policy).
//! - [`fetch`]: HTTP fetching and resource loading.
//! - [`fs`]: workspace-scoped file read, write, search, and edit tools.
//! - [`mcp`]: dynamic tool provider for external MCP servers.
//! - [`note`]: lightweight per-agent note storage.
//! - [`shell`]: shell command execution via a pluggable [`shell::Executor`]
//!   runtime (the host [`shell::NativeRuntime`] by default).
//! - [`skill`]: file-backed skill loading and lifecycle management.
//! - [`mod@todo`]: session-scoped task tracking for long-running agents.
//!
//! # Tool protocol
//!
//! The built-in tools share one calling convention, owned by this module:
//!
//! - **Schema**: `definition()` derives the parameter schema from the typed
//!   arguments via [`tool_definition`] with `strict: Some(true)`. Field
//!   descriptions come from doc comments on the argument structs; enum values
//!   and defaults on plain `String` fields are declared with
//!   `#[schemars(extend(...))]`. Instance-dependent details (for example
//!   descriptions computed from runtime configuration) are patched onto the
//!   returned definition (see the shell tool).
//! - **Hooks**: [`hooked_call`] wraps every execution with the
//!   [`DynToolHook`] found on [`BaseCtx`] state:
//!   `before_tool_call` may rewrite the arguments, `after_tool_call` may
//!   rewrite the output.
//! - **Cancellation**: a call whose context token is already cancelled fails
//!   inside [`hooked_call`] before the hook or the tool body execute.
//!   Long-running loops should still poll
//!   [`StateFeatures::cancellation_token`] cooperatively (see the search
//!   tool).
//! - **Error policy**: a tool body returns `Err` only when no meaningful
//!   typed output exists (the runner then feeds `{"error": ...}` with
//!   `is_error` to the model). A domain failure that still has a useful typed
//!   output returns `Ok` with [`ToolOutput::is_error`] set to `Some(true)` so
//!   hooks and providers see both the structured payload and the failure
//!   signal (see the note and shell tools).
//!
//! Truncation signals deliberately stay per-tool: inline-content cuts
//! (`read_file.truncated`), aborted scans (`search_file.scan_truncated`),
//! in-band stream markers plus `raw_output_path` (`shell`), and hard size
//! errors (`fetch`, `skills_manager`) describe genuinely different
//! conditions, and folding them into one field would lose that meaning.

use anda_core::{BoxError, FunctionDefinition, StateFeatures, ToolOutput, gen_schema_for};
use schemars::JsonSchema;
use std::future::Future;

use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

pub mod fetch;
pub mod fs;
pub mod mcp;
pub mod note;
pub mod shell;
pub mod skill;
pub mod todo;

/// Builds a strict [`FunctionDefinition`] from a typed argument schema.
///
/// The parameter schema is derived from `A` via [`gen_schema_for`], so the
/// argument struct (doc comments, `#[schemars(...)]` attributes) is the single
/// source of truth for the wire schema. Tools whose schema has
/// instance-dependent parts mutate `parameters` on the returned definition.
pub fn tool_definition<A: JsonSchema>(name: String, description: String) -> FunctionDefinition {
    FunctionDefinition {
        name,
        description,
        parameters: gen_schema_for::<A>(),
        strict: Some(true),
    }
}

/// Runs a tool body under the shared tool-call protocol.
///
/// In order: fails fast when `ctx` is already cancelled, applies the
/// [`DynToolHook<I, O>`] found on `ctx` state (`before_tool_call` rewrites the
/// arguments), executes `run`, then gives `after_tool_call` the chance to
/// rewrite the output. See the [module docs](self) for the full protocol.
pub async fn hooked_call<I, O, F, Fut>(
    ctx: &BaseCtx,
    args: I,
    run: F,
) -> Result<ToolOutput<O>, BoxError>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: FnOnce(I) -> Fut,
    Fut: Future<Output = Result<ToolOutput<O>, BoxError>> + Send,
{
    if ctx.cancellation_token().is_cancelled() {
        return Err("call was cancelled".into());
    }

    let hook = ctx.get_state::<DynToolHook<I, O>>();
    let args = match &hook {
        Some(hook) => hook.before_tool_call(ctx, args).await?,
        None => args,
    };

    let output = run(args).await?;

    match &hook {
        Some(hook) => hook.after_tool_call(ctx, output).await,
        None => Ok(output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use crate::extension::{
        fetch::FetchWebResourcesTool,
        fs::{EditFileTool, ReadFileTool, SearchFileTool, WriteFileTool},
        note::NoteTool,
        shell::{ExecArgs, ExecOutput, Executor, ShellTool},
        skill::SkillManager,
        todo::TodoTool,
    };
    use anda_core::{Json, Tool};
    use async_trait::async_trait;
    use serde_json::json;
    use std::{collections::HashMap, path::PathBuf, sync::Arc};

    struct StubRuntime {
        workspace: PathBuf,
    }

    #[async_trait]
    impl Executor for StubRuntime {
        fn name(&self) -> &str {
            "stub"
        }

        fn workspace(&self) -> &PathBuf {
            &self.workspace
        }

        fn shell(&self) -> &str {
            "sh"
        }

        async fn execute(
            &self,
            _ctx: BaseCtx,
            _input: ExecArgs,
            _envs: HashMap<String, String>,
        ) -> Result<ExecOutput, BoxError> {
            unreachable!("definition-only test")
        }
    }

    fn extension_tool_definitions() -> Vec<FunctionDefinition> {
        let dir = std::env::temp_dir();
        vec![
            FetchWebResourcesTool::new().definition(),
            ReadFileTool::new(dir.clone()).definition(),
            EditFileTool::new(dir.clone()).definition(),
            WriteFileTool::new(dir.clone()).definition(),
            SearchFileTool::new(dir.clone()).definition(),
            TodoTool::new().definition(),
            NoteTool::new().definition(),
            ShellTool::new(
                Arc::new(StubRuntime {
                    workspace: dir.clone(),
                }),
                HashMap::new(),
                None,
            )
            .definition(),
            SkillManager::new(dir).definition(),
        ]
    }

    /// Rejects the schema-union keywords strict function-calling providers do
    /// not accept. Nullable fields must derive as flat type unions (e.g.
    /// `"type": ["string", "null"]`) instead.
    fn assert_union_free(tool: &str, value: &Json) {
        match value {
            Json::Object(map) => {
                for key in ["anyOf", "oneOf", "allOf"] {
                    assert!(
                        !map.contains_key(key),
                        "tool {tool}: derived schema contains {key}"
                    );
                }
                for child in map.values() {
                    assert_union_free(tool, child);
                }
            }
            Json::Array(items) => {
                for child in items {
                    assert_union_free(tool, child);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn extension_tool_definitions_are_strict_and_union_free() {
        let definitions = extension_tool_definitions();
        assert_eq!(definitions.len(), 9);

        for definition in definitions {
            let tool = &definition.name;
            assert_eq!(definition.strict, Some(true), "tool {tool}");
            assert_eq!(
                definition.parameters["type"],
                json!("object"),
                "tool {tool}"
            );
            assert_eq!(
                definition.parameters["additionalProperties"],
                json!(false),
                "tool {tool}"
            );

            let mut properties: Vec<&str> = definition.parameters["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("tool {tool}: missing properties object"))
                .keys()
                .map(String::as_str)
                .collect();
            let mut required: Vec<&str> = definition.parameters["required"]
                .as_array()
                .unwrap_or_else(|| panic!("tool {tool}: missing required array"))
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect();
            properties.sort_unstable();
            required.sort_unstable();
            assert_eq!(
                required, properties,
                "tool {tool}: strict schemas require every property"
            );

            assert_union_free(tool, &definition.parameters);
        }
    }

    #[test]
    fn nullable_enum_args_derive_flat_type_unions() {
        let todo = TodoTool::new().definition().parameters;
        assert_eq!(todo["properties"]["op"]["type"], json!(["string", "null"]));
        assert_eq!(
            todo["properties"]["op"]["enum"],
            json!(["read", "set", "update", null])
        );
        assert_eq!(todo["properties"]["op"]["default"], json!("read"));
        assert_eq!(
            todo["properties"]["items"]["type"],
            json!(["array", "null"])
        );
        assert_eq!(
            todo["properties"]["items"]["items"]["properties"]["status"]["enum"],
            json!(["pending", "in_progress", "completed", "cancelled", null])
        );

        let note = NoteTool::new().definition().parameters;
        assert_eq!(
            note["properties"]["op"]["enum"],
            json!(["read", "set", "upsert", "delete", null])
        );
        assert_eq!(
            note["properties"]["items"]["items"]["properties"]["content"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn instance_dependent_schema_patches_apply() {
        let shell = ShellTool::new(
            Arc::new(StubRuntime {
                workspace: std::env::temp_dir(),
            }),
            HashMap::new(),
            None,
        )
        .definition()
        .parameters;
        assert!(
            shell["properties"]["env_keys"]["description"]
                .as_str()
                .unwrap()
                .contains("environment variable")
        );
        assert!(
            shell["properties"]["background"]["description"]
                .as_str()
                .unwrap()
                .contains("background")
        );

        let skill = SkillManager::new(std::env::temp_dir())
            .definition()
            .parameters;
        assert!(
            skill["description"]
                .as_str()
                .unwrap()
                .contains("SKILL.md file content")
        );
    }

    struct RewritingHook;

    #[async_trait]
    impl ToolHook<String, String> for RewritingHook {
        async fn before_tool_call(&self, _ctx: &BaseCtx, args: String) -> Result<String, BoxError> {
            Ok(format!("{args}+before"))
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            output: ToolOutput<String>,
        ) -> Result<ToolOutput<String>, BoxError> {
            Ok(ToolOutput::new(format!("{}+after", output.output)))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn hooked_call_applies_the_hook_and_gates_on_cancellation() {
        let ctx = EngineBuilder::new().mock_ctx().base;

        // No hook registered: the body runs and its output passes through.
        let output = hooked_call(&ctx, "args".to_string(), |args| async move {
            Ok(ToolOutput::new(format!("{args}+run")))
        })
        .await
        .unwrap();
        assert_eq!(output.output, "args+run");

        // With a hook: arguments are rewritten before the body, output after.
        ctx.set_state(DynToolHook::new(
            Arc::new(RewritingHook) as Arc<dyn ToolHook<String, String>>
        ));
        let output = hooked_call(&ctx, "args".to_string(), |args| async move {
            Ok(ToolOutput::new(format!("{args}+run")))
        })
        .await
        .unwrap();
        assert_eq!(output.output, "args+before+run+after");

        // An already-cancelled context fails before the hook or the body run.
        ctx.cancellation_token().cancel();
        let err = hooked_call(&ctx, "args".to_string(), |_args: String| async move {
            unreachable!("the body must not run on a cancelled context")
                as Result<ToolOutput<String>, BoxError>
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cancelled"));
    }
}
