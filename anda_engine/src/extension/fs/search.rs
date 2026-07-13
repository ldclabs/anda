//! Glob search tool for paths inside configured workspaces.
//!
//! Relative patterns are evaluated against every configured workspace, while
//! absolute patterns must remain inside a workspace boundary.

use anda_core::{
    BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolGroupInfo, ToolOutput,
};
use glob::{MatchOptions, glob_with};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Component, Path, PathBuf};

use super::{
    ensure_path_in_workspace, ensure_path_in_workspace_namespace, format_workspaces,
    nearest_existing_ancestor, normalize_relative_path, normalize_workspaces,
    path_contains_parent_reference, resolve_workspace_path, tool_workspaces,
    workspace_access_error,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

const DEFAULT_LIMIT: usize = 1000;
/// Hard cap on collected matches across all workspaces. Scanning stops here so
/// pathological patterns (e.g. `**/*` over a huge tree) stay bounded.
const MAX_GLOB_MATCHES: usize = 10_000;

/// Arguments for filesystem glob operations.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SearchFileArgs {
    /// Relative or absolute glob pattern inside the workspace namespace.
    pub pattern: String,
    /// Maximum number of matches to return. Defaults to 1000.
    #[serde(default)]
    pub limit: usize,
}

/// Normalized result returned by a filesystem glob operation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SearchFileOutput {
    /// Matching paths relative to the workspace root.
    pub paths: Vec<String>,
    /// Total matches before applying `limit`.
    pub total_matches: usize,
    /// True when scanning stopped early at the internal match cap; `total_matches`
    /// is a lower bound. Narrow the pattern to see the rest.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub scan_truncated: bool,
}

/// Typed hook for search-file tool calls.
pub type SearchFileHook = DynToolHook<SearchFileArgs, SearchFileOutput>;

/// Tool implementation for glob-based workspace path search.
#[derive(Clone)]
pub struct SearchFileTool {
    workspaces: Vec<PathBuf>,
    description: String,
}

impl SearchFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "search_file";

    /// Create a new `SearchFileTool` with the default workspace directory.
    /// You can add workspace directories for each call by including `workspace` or `workspaces` in the tool call's context meta extra.
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_workspaces([workspace])
    }

    /// Create a new `SearchFileTool` with the default workspace directories.
    /// Context meta workspaces take precedence over these defaults at call time.
    pub fn with_workspaces<I>(workspaces: I) -> Self
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let workspaces = normalize_workspaces(workspaces);
        let description = format!(
            "Match filesystem paths with glob patterns in the workspace directories ({})",
            format_workspaces(&workspaces)
        );
        Self {
            workspaces,
            description,
        }
    }

    /// Overrides the function description exposed to the model.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

impl Tool<BaseCtx> for SearchFileTool {
    type Args = SearchFileArgs;
    type Output = SearchFileOutput;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn group(&self) -> Option<ToolGroupInfo> {
        Some(super::fs_tool_group_info())
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Relative or absolute glob pattern. Relative patterns are expanded in all configured workspace namespaces; absolute patterns must be inside one configured workspace."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default: 1000)"
                    }
                },
                "required": ["pattern", "limit"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let hook = ctx.get_state::<SearchFileHook>();

        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let workspaces = tool_workspaces(ctx.meta(), &self.workspaces);
        let mut paths = Vec::new();
        let mut errors = Vec::new();
        let mut searched_any_workspace = false;
        let mut scan_truncated = false;

        'workspaces: for workspace in &workspaces {
            let workspace_display = workspace.display().to_string();
            let (resolved_workspace, pattern, restrict_to_workspace_targets) =
                match resolve_glob_pattern(workspace, &args.pattern).await {
                    Ok(resolved) => resolved,
                    Err(err) => {
                        errors.push(format!("{}: {err}", workspace.display()));
                        continue;
                    }
                };
            searched_any_workspace = true;

            for entry in glob_with(&pattern, glob_match_options())
                .map_err(|err| {
                    format!(
                        "Invalid glob pattern (workspace: {}, requested_pattern: {}, expanded_pattern: {}): {err}",
                        workspace_display,
                        args.pattern,
                        pattern
                    )
                })?
            {
                // Unreadable directories or entries removed mid-scan must not fail the
                // whole search; skip them and keep matching.
                let Ok(path) = entry else {
                    continue;
                };

                // Every match is canonicalized and re-checked against the
                // workspace root. A workspace-internal directory symlink pointing
                // outside the workspace would otherwise let a plain pattern (no
                // `..`) enumerate external filenames, violating the workspace
                // scope invariant.
                let resolved_path = match tokio::fs::canonicalize(&path).await {
                    // A match that resolves outside the workspace (via a symlinked
                    // directory component) is skipped, not fatal.
                    Ok(resolved) => {
                        if ensure_path_in_workspace(&resolved_workspace, &resolved).is_err() {
                            continue;
                        }
                        Some(resolved)
                    }
                    // Dangling symlink or entry removed mid-scan. Parent-traversal
                    // patterns skip it; plain patterns list it as-is, since a
                    // dangling link cannot leak an external path.
                    Err(_) => {
                        if restrict_to_workspace_targets {
                            continue;
                        }
                        None
                    }
                };

                let relative = match relative_match_path(&path, workspace, &resolved_workspace) {
                    Some(relative) => relative,
                    None => {
                        let resolved = match resolved_path {
                            Some(resolved) => resolved,
                            None => match tokio::fs::canonicalize(&path).await {
                                Ok(resolved) => resolved,
                                Err(_) => continue,
                            },
                        };
                        match resolved.strip_prefix(&resolved_workspace) {
                            Ok(relative) => normalize_relative_path(relative),
                            Err(_) => continue,
                        }
                    }
                };

                if paths.len() >= MAX_GLOB_MATCHES {
                    scan_truncated = true;
                    break 'workspaces;
                }
                paths.push(relative);
            }
        }

        if !searched_any_workspace {
            return Err(workspace_access_error(
                "Glob pattern",
                "requested_pattern",
                &args.pattern,
                &workspaces,
                errors,
            ));
        }

        paths.sort();
        paths.dedup();

        let total_matches = paths.len();
        paths.truncate(if args.limit == 0 {
            DEFAULT_LIMIT
        } else {
            args.limit
        });

        let output = SearchFileOutput {
            paths,
            total_matches,
            scan_truncated,
        };

        if let Some(hook) = &hook {
            return hook.after_tool_call(&ctx, ToolOutput::new(output)).await;
        }

        Ok(ToolOutput::new(output))
    }
}

fn glob_match_options() -> MatchOptions {
    MatchOptions {
        case_sensitive: !cfg!(windows),
        require_literal_separator: true,
        require_literal_leading_dot: false,
    }
}

async fn resolve_glob_pattern(
    workspace: &Path,
    pattern: &str,
) -> Result<(PathBuf, String, bool), BoxError> {
    if pattern.trim().is_empty() {
        return Err(format!(
            "Glob pattern must not be empty (workspace: {})",
            workspace.display()
        )
        .into());
    }

    let resolved_workspace = resolve_workspace_path(workspace).await?;
    let requested_pattern = Path::new(pattern);
    let absolute_pattern = workspace.join(requested_pattern);
    let literal_prefix = literal_glob_prefix(&absolute_pattern);

    if !path_contains_parent_reference(requested_pattern) {
        ensure_path_in_workspace_namespace(workspace, &resolved_workspace, &literal_prefix)?;

        return Ok((
            resolved_workspace,
            absolute_pattern.to_string_lossy().into_owned(),
            false,
        ));
    }

    let (existing_ancestor, _) = nearest_existing_ancestor(&literal_prefix).await?;
    let resolved_ancestor = tokio::fs::canonicalize(&existing_ancestor)
        .await
        .map_err(|err| {
            format!(
                "Failed to resolve glob path (workspace: {}, requested_pattern: {}, literal_prefix: {}, existing_ancestor: {}): {err}",
                workspace.display(),
                pattern,
                literal_prefix.display(),
                existing_ancestor.display()
            )
        })?;

    ensure_path_in_workspace(&resolved_workspace, &resolved_ancestor)?;

    Ok((
        resolved_workspace,
        absolute_pattern.to_string_lossy().into_owned(),
        true,
    ))
}

fn literal_glob_prefix(pattern: &Path) -> PathBuf {
    let mut prefix = PathBuf::new();

    for component in pattern.components() {
        match component {
            Component::Prefix(value) => prefix.push(value.as_os_str()),
            Component::RootDir => prefix.push(component.as_os_str()),
            Component::CurDir | Component::ParentDir => prefix.push(component.as_os_str()),
            Component::Normal(value) => {
                if has_glob_metacharacters(value) {
                    break;
                }
                prefix.push(value);
            }
        }
    }

    if prefix.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        prefix
    }
}

fn has_glob_metacharacters(component: &std::ffi::OsStr) -> bool {
    let component = component.to_string_lossy();
    component.contains('*')
        || component.contains('?')
        || component.contains('[')
        || component.contains(']')
        || component.contains('{')
        || component.contains('}')
}

/// Strips the workspace prefix without touching the filesystem. Returns `None`
/// when only the canonicalized path can be related to the workspace.
fn relative_match_path(path: &Path, workspace: &Path, resolved_workspace: &Path) -> Option<String> {
    if let Ok(relative) = path.strip_prefix(workspace) {
        return Some(normalize_relative_path(relative));
    }

    path.strip_prefix(resolved_workspace)
        .ok()
        .map(normalize_relative_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use serde_json::json;
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
    };

    struct TestTempDir(PathBuf);

    impl TestTempDir {
        async fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("anda-fs-glob-test-{:016x}", rand::random::<u64>()));
            tokio::fs::create_dir_all(&path).await.unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn mock_ctx() -> BaseCtx {
        EngineBuilder::new().mock_ctx().base
    }

    fn mock_ctx_with_workspaces(workspace: &Path, workspaces: &[&Path]) -> BaseCtx {
        let mut ctx = mock_ctx();
        ctx.meta.extra.insert(
            "workspace".to_string(),
            json!(workspace.to_string_lossy().to_string()),
        );
        ctx.meta.extra.insert(
            "workspaces".to_string(),
            json!(
                workspaces
                    .iter()
                    .map(|workspace| workspace.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
            ),
        );
        ctx
    }

    fn glob_tool(workspace: &Path) -> SearchFileTool {
        SearchFileTool::new(workspace.to_path_buf())
    }

    struct RewritingSearchHook;

    #[async_trait::async_trait]
    impl ToolHook<SearchFileArgs, SearchFileOutput> for RewritingSearchHook {
        async fn before_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut args: SearchFileArgs,
        ) -> Result<SearchFileArgs, BoxError> {
            args.pattern = "src/*.rs".to_string();
            args.limit = 1;
            Ok(args)
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut output: ToolOutput<SearchFileOutput>,
        ) -> Result<ToolOutput<SearchFileOutput>, BoxError> {
            output.output.paths.push("hook-added.rs".to_string());
            Ok(output)
        }
    }

    #[tokio::test]
    async fn metadata_hooks_empty_patterns_and_parent_paths_are_covered() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(workspace.join("src"))
            .await
            .unwrap();
        tokio::fs::write(workspace.join("src/a.rs"), "a")
            .await
            .unwrap();
        tokio::fs::write(workspace.join("src/b.rs"), "b")
            .await
            .unwrap();

        let tool = glob_tool(&workspace).with_description("custom search".to_string());
        assert_eq!(tool.name(), SearchFileTool::NAME);
        assert_eq!(tool.description(), "custom search");
        let definition = tool.definition();
        assert_eq!(definition.name, SearchFileTool::NAME);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(
            definition.parameters["required"],
            json!(["pattern", "limit"])
        );

        let empty_err = tool
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "   ".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(
            empty_err
                .to_string()
                .contains("Glob pattern must not be empty")
        );

        let parent_result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "src/../src/*.rs".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            parent_result.output.paths,
            ["src/../src/a.rs", "src/../src/b.rs"]
        );
        assert_eq!(parent_result.output.total_matches, 2);

        let ctx = mock_ctx();
        ctx.set_state(SearchFileHook::new(Arc::new(RewritingSearchHook)));
        let hooked = glob_tool(&workspace)
            .call(
                ctx,
                SearchFileArgs {
                    pattern: "ignored/*.txt".to_string(),
                    limit: 100,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(hooked.output.total_matches, 2);
        assert_eq!(hooked.output.paths, ["src/a.rs", "hook-added.rs"]);
    }

    #[tokio::test]
    async fn searches_meta_extra_and_default_workspaces() {
        let temp_dir = TestTempDir::new().await;
        let runtime_workspace = temp_dir.path().join("runtime");
        let extra_workspace = temp_dir.path().join("extra");
        let home_workspace = temp_dir.path().join("home");
        tokio::fs::create_dir_all(runtime_workspace.join("src"))
            .await
            .unwrap();
        tokio::fs::create_dir_all(extra_workspace.join("src"))
            .await
            .unwrap();
        tokio::fs::create_dir_all(home_workspace.join("src"))
            .await
            .unwrap();
        tokio::fs::write(runtime_workspace.join("src/runtime.rs"), "runtime")
            .await
            .unwrap();
        tokio::fs::write(extra_workspace.join("src/extra.rs"), "extra")
            .await
            .unwrap();
        tokio::fs::write(home_workspace.join("src/home.rs"), "home")
            .await
            .unwrap();

        let result = glob_tool(&home_workspace)
            .call(
                mock_ctx_with_workspaces(&runtime_workspace, &[&extra_workspace]),
                SearchFileArgs {
                    pattern: "src/*.rs".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(
            result.output.paths,
            ["src/extra.rs", "src/home.rs", "src/runtime.rs"]
        );
        assert_eq!(result.output.total_matches, 3);
    }

    #[tokio::test]
    async fn matches_requested_paths() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(workspace.join("src"))
            .await
            .unwrap();
        tokio::fs::write(workspace.join("src/lib.rs"), "pub fn lib() {}")
            .await
            .unwrap();
        tokio::fs::write(workspace.join("src/main.rs"), "fn main() {}")
            .await
            .unwrap();
        tokio::fs::write(workspace.join("README.md"), "# anda")
            .await
            .unwrap();

        let result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "src/*.rs".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.paths, ["src/lib.rs", "src/main.rs"]);
        assert_eq!(result.output.total_matches, 2);
    }

    #[tokio::test]
    async fn applies_limit_after_sorting_matches() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(workspace.join("logs"))
            .await
            .unwrap();
        tokio::fs::write(workspace.join("logs/c.txt"), "c")
            .await
            .unwrap();
        tokio::fs::write(workspace.join("logs/a.txt"), "a")
            .await
            .unwrap();
        tokio::fs::write(workspace.join("logs/b.txt"), "b")
            .await
            .unwrap();

        let result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "logs/*.txt".to_string(),
                    limit: 2,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.paths, ["logs/a.txt", "logs/b.txt"]);
        assert_eq!(result.output.total_matches, 3);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn matches_files_from_a_symlinked_workspace_root() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let workspace_link = temp_dir.path().join("workspace-link");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("notes.txt"), "hello")
            .await
            .unwrap();
        symlink(&workspace, &workspace_link).unwrap();

        let result = glob_tool(&workspace_link)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "*.txt".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.paths, ["notes.txt"]);
        assert_eq!(result.output.total_matches, 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn skips_file_symlink_escaping_workspace() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("secret.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&external, "secret").await.unwrap();
        symlink(&external, workspace.join("secret-link.txt")).unwrap();

        let result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "*.txt".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        // The symlink resolves outside the workspace, so it must not be listed.
        assert!(result.output.paths.is_empty());
        assert_eq!(result.output.total_matches, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn skips_files_through_symbolic_linked_directory_escaping_workspace() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("external");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&external).await.unwrap();
        tokio::fs::write(external.join("secret.txt"), "secret")
            .await
            .unwrap();
        symlink(&external, workspace.join("escape")).unwrap();

        let result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "escape/*.txt".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        // A workspace-internal directory symlink pointing outside must not let
        // the search enumerate external filenames.
        assert!(result.output.paths.is_empty());
        assert_eq!(result.output.total_matches, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dangling_symlinks_do_not_fail_the_scan() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(workspace.join("sub"))
            .await
            .unwrap();
        tokio::fs::write(workspace.join("real.txt"), "ok")
            .await
            .unwrap();
        symlink(
            temp_dir.path().join("does-not-exist"),
            workspace.join("broken.txt"),
        )
        .unwrap();

        // Plain patterns list dangling symlinks instead of failing the whole search.
        let result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "*.txt".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.output.paths, ["broken.txt", "real.txt"]);
        assert!(!result.output.scan_truncated);

        // Parent-traversal patterns verify targets, so the dangling entry is skipped.
        let result = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "sub/../*.txt".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.output.paths, ["sub/../real.txt"]);
    }

    #[tokio::test]
    async fn rejects_absolute_pattern_outside_workspace() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("external");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&external).await.unwrap();
        tokio::fs::write(external.join("secret.txt"), "secret")
            .await
            .unwrap();

        let err = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: external.join("*.txt").to_string_lossy().into_owned(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("Access to paths outside the workspace is not allowed")
        );
    }

    #[tokio::test]
    async fn rejects_parent_dir_pattern_escape_outside_workspace() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("external");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&external).await.unwrap();
        tokio::fs::write(external.join("secret.txt"), "secret")
            .await
            .unwrap();

        let err = glob_tool(&workspace)
            .call(
                mock_ctx(),
                SearchFileArgs {
                    pattern: "../external/*.txt".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("Access to paths outside the workspace is not allowed")
        );
    }
}
