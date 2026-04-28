use anda_core::{BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput};
use glob::{MatchOptions, glob_with};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    borrow::Cow,
    path::{Component, Path, PathBuf},
};

use super::{
    ensure_path_in_workspace, ensure_path_in_workspace_namespace, nearest_existing_ancestor,
    normalize_relative_path, path_contains_parent_reference, resolve_workspace_path,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

const DEFAULT_LIMIT: usize = 1000;

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
}

pub type SearchFileHook = DynToolHook<SearchFileArgs, SearchFileOutput>;

#[derive(Clone)]
pub struct SearchFileTool {
    workspace: PathBuf,
    description: String,
}

impl SearchFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "search_file";

    /// Create a new `SearchFileTool` with the default working directory.
    /// You can override the working directory for each call by including a `workspace` field in the tool call's context meta extra.
    pub fn new(workspace: PathBuf) -> Self {
        let description =
            "Match filesystem paths with glob patterns in the workspace directory".to_string();
        Self {
            workspace,
            description,
        }
    }

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

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Relative or absolute glob pattern. Matches are restricted to paths reachable from the workspace namespace."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default: 1000)"
                    }
                },
                "required": ["pattern"]
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

        let workspace = ctx
            .meta()
            .get_extra_as::<String>("workspace")
            .map(PathBuf::from)
            .map(Cow::Owned)
            .unwrap_or_else(|| Cow::Borrowed(&self.workspace));

        let (resolved_workspace, pattern, restrict_to_workspace_targets) =
            resolve_glob_pattern(&workspace, &args.pattern).await?;

        let mut paths = Vec::new();
        for entry in glob_with(&pattern, glob_match_options())
            .map_err(|err| format!("Invalid glob pattern: {err}"))?
        {
            let path = entry.map_err(|err| format!("Failed to expand glob pattern: {err}"))?;
            let resolved_path = tokio::fs::canonicalize(&path)
                .await
                .map_err(|err| format!("Failed to resolve matched path: {err}"))?;
            if restrict_to_workspace_targets {
                ensure_path_in_workspace(&resolved_workspace, &resolved_path)?;
            }

            paths.push(relative_match_path(
                &path,
                &resolved_path,
                &workspace,
                &resolved_workspace,
            )?);
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
        return Err("Glob pattern must not be empty".into());
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
        .map_err(|err| format!("Failed to resolve glob path: {err}"))?;

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

fn relative_match_path(
    path: &Path,
    resolved_path: &Path,
    workspace: &Path,
    resolved_workspace: &Path,
) -> Result<String, BoxError> {
    if let Ok(relative) = path.strip_prefix(workspace) {
        return Ok(normalize_relative_path(relative));
    }

    if let Ok(relative) = path.strip_prefix(resolved_workspace) {
        return Ok(normalize_relative_path(relative));
    }

    let relative = resolved_path
        .strip_prefix(resolved_workspace)
        .map_err(|err| format!("Failed to normalize matched path: {err}"))?;
    Ok(normalize_relative_path(relative))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use std::path::{Path, PathBuf};

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

    fn glob_tool(workspace: &Path) -> SearchFileTool {
        SearchFileTool::new(workspace.to_path_buf())
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
    async fn matches_files_through_symbolic_link_target() {
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

        assert_eq!(result.output.paths, ["secret-link.txt"]);
        assert_eq!(result.output.total_matches, 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn matches_files_through_symbolic_linked_directory_target() {
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

        assert_eq!(result.output.paths, ["escape/secret.txt"]);
        assert_eq!(result.output.total_matches, 1);
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
