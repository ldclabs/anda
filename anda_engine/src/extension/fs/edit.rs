use anda_core::{BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;

use super::{
    FileTextEncodeError, MAX_FILE_SIZE_BYTES, atomic_write_file, decode_file_text,
    encode_file_text, ensure_file_size_within_limit, ensure_regular_file, format_workspaces,
    normalize_workspaces, resolve_write_path_in_workspaces, tool_workspaces,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

/// Arguments for filesystem edit operations.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EditFileArgs {
    /// Relative or absolute path to a text file inside the workspace.
    pub path: String,
    /// The old string to replace.
    pub old_string: String,
    /// The replacement string.
    pub new_string: String,
    /// Maximum number of replacements to apply. `0` means replace all matches.
    #[serde(default)]
    pub limit: usize,
}

/// Normalized result returned by a filesystem edit operation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EditFileOutput {
    /// Number of non-overlapping replacements applied to the file.
    pub replacements: usize,
    /// Number of matches of the old string found in the file.
    pub total_matches: usize,
    /// Number of bytes in the resulting text file.
    pub size: u64,
}

pub type EditFileHook = DynToolHook<EditFileArgs, EditFileOutput>;

#[derive(Clone)]
pub struct EditFileTool {
    workspaces: Vec<PathBuf>,
    description: String,
}

impl EditFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "edit_file";

    /// Create a new `EditFileTool` with the default workspace directory.
    /// You can add workspace directories for each call by including `workspace` or `workspaces` in the tool call's context meta extra.
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_workspaces([workspace])
    }

    /// Create a new `EditFileTool` with the default workspace directories.
    /// Context meta workspaces take precedence over these defaults at call time.
    pub fn with_workspaces<I>(workspaces: I) -> Self
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let workspaces = normalize_workspaces(workspaces);
        let description = format!(
            "Atomically edit text files in the workspace directories ({}) by replacing strings",
            format_workspaces(&workspaces)
        );
        Self {
            workspaces,
            description,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

impl Tool<BaseCtx> for EditFileTool {
    type Args = EditFileArgs;
    type Output = EditFileOutput;

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
                    "path": {
                        "type": "string",
                        "description": "Path to the text file. Relative paths resolve from the configured workspaces in priority order; absolute paths must be inside one configured workspace."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Old decoded text string to replace."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement decoded text string."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of replacements to apply (default: 0, replace all matches)."
                    }
                },
                "required": ["path", "old_string", "new_string", "limit"],
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
        let hook = ctx.get_state::<EditFileHook>();

        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let workspaces = tool_workspaces(ctx.meta(), &self.workspaces);
        let workspace_display = format_workspaces(&workspaces);

        if args.old_string.is_empty() {
            return Err(format!(
                "Old string must not be empty (workspace: {}, path: {})",
                workspace_display, args.path
            )
            .into());
        }

        let resolved = resolve_write_path_in_workspaces(&workspaces, &args.path).await?;
        let workspace_display = resolved.workspace.display().to_string();
        let resolved_path = resolved.path;
        let meta = match tokio::fs::metadata(&resolved_path).await {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(format!(
                    "Path does not point to an existing file (workspace: {}, requested_path: {}, resolved_path: {})",
                    workspace_display,
                    args.path,
                    resolved_path.display()
                )
                .into());
            }
            Err(err) => {
                return Err(format!(
                    "Failed to read file metadata (workspace: {}, requested_path: {}, resolved_path: {}): {err}",
                    workspace_display,
                    args.path,
                    resolved_path.display()
                )
                .into())
            }
        };

        ensure_regular_file(
            &meta,
            &resolved_path,
            "Editing multiply-linked files is not allowed",
        )?;
        ensure_file_size_within_limit(&meta, &resolved_path, MAX_FILE_SIZE_BYTES)?;

        let data = tokio::fs::read(&resolved_path).await.map_err(|err| {
            format!(
                "Failed to read file (workspace: {}, requested_path: {}, resolved_path: {}): {err}",
                workspace_display,
                args.path,
                resolved_path.display()
            )
        })?;
        let original_size = data.len() as u64;
        let decoded = decode_file_text(data).map_err(|_| {
            format!(
                "Editing binary or unsupported-encoding files is not supported (workspace: {}, requested_path: {}, resolved_path: {})",
                workspace_display,
                args.path,
                resolved_path.display()
            )
        })?;
        let encoding = decoded.encoding;
        let text = decoded.text;
        let total_matches = text.match_indices(&args.old_string).count();

        let replacements = if args.limit == 0 {
            total_matches
        } else {
            total_matches.min(args.limit)
        };

        let output = if total_matches == 0 || args.old_string == args.new_string {
            EditFileOutput {
                replacements,
                total_matches,
                size: original_size,
            }
        } else {
            let updated = if args.limit == 0 {
                text.replace(&args.old_string, &args.new_string)
            } else {
                text.replacen(&args.old_string, &args.new_string, args.limit)
            };
            let updated_bytes = encode_file_text(&updated, &encoding).map_err(|err| match err {
                FileTextEncodeError::UnsupportedEncoding => format!(
                    "Unsupported text encoding while editing file (workspace: {}, requested_path: {}, resolved_path: {}, encoding: {})",
                    workspace_display,
                    args.path,
                    resolved_path.display(),
                    encoding
                ),
                FileTextEncodeError::UnmappableCharacters => format!(
                    "Failed to encode edited file (workspace: {}, requested_path: {}, resolved_path: {}, encoding: {}): {err}",
                    workspace_display,
                    args.path,
                    resolved_path.display(),
                    encoding
                ),
            })?;
            let size = updated_bytes.len() as u64;
            atomic_write_file(&resolved_path, &updated_bytes, Some(&meta.permissions())).await?;
            EditFileOutput {
                replacements,
                total_matches,
                size,
            }
        };

        if let Some(hook) = &hook {
            return hook.after_tool_call(&ctx, ToolOutput::new(output)).await;
        }

        Ok(ToolOutput::new(output))
    }
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
                .join(format!("anda-fs-edit-test-{:016x}", rand::random::<u64>()));
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

    fn mock_ctx_with_workspace(workspace: &Path) -> BaseCtx {
        let mut ctx = mock_ctx();
        ctx.meta.extra.insert(
            "workspace".to_string(),
            json!(workspace.to_string_lossy().to_string()),
        );
        ctx
    }

    fn edit_tool(workspace: &Path) -> EditFileTool {
        EditFileTool::new(workspace.to_path_buf())
    }

    struct RewritingEditHook;

    #[async_trait::async_trait]
    impl ToolHook<EditFileArgs, EditFileOutput> for RewritingEditHook {
        async fn before_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut args: EditFileArgs,
        ) -> Result<EditFileArgs, BoxError> {
            args.path = "notes.txt".to_string();
            args.old_string = "alpha".to_string();
            args.new_string = "beta".to_string();
            args.limit = 1;
            Ok(args)
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut output: ToolOutput<EditFileOutput>,
        ) -> Result<ToolOutput<EditFileOutput>, BoxError> {
            output.output.replacements += 10;
            Ok(output)
        }
    }

    #[tokio::test]
    async fn metadata_hooks_missing_files_and_same_string_branch_are_covered() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "alpha alpha").await.unwrap();

        let tool = edit_tool(&workspace).with_description("custom edit".to_string());
        assert_eq!(tool.name(), EditFileTool::NAME);
        assert_eq!(tool.description(), "custom edit");
        let definition = tool.definition();
        assert_eq!(definition.name, EditFileTool::NAME);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(
            definition.parameters["required"],
            json!(["path", "old_string", "new_string", "limit"])
        );

        let unchanged = tool
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "alpha".to_string(),
                    new_string: "alpha".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(unchanged.output.replacements, 2);
        assert_eq!(unchanged.output.total_matches, 2);
        assert_eq!(
            tokio::fs::read_to_string(&target).await.unwrap(),
            "alpha alpha"
        );

        let missing = tool
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "missing.txt".to_string(),
                    old_string: "alpha".to_string(),
                    new_string: "beta".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("Path does not point to an existing file")
        );

        let ctx = mock_ctx();
        ctx.set_state(EditFileHook::new(Arc::new(RewritingEditHook)));
        let hooked = tool
            .call(
                ctx,
                EditFileArgs {
                    path: "ignored.txt".to_string(),
                    old_string: "ignored".to_string(),
                    new_string: "ignored".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(hooked.output.replacements, 11);
        assert_eq!(hooked.output.total_matches, 2);
        assert_eq!(
            tokio::fs::read_to_string(&target).await.unwrap(),
            "beta alpha"
        );
    }

    #[tokio::test]
    async fn edits_default_workspace_when_meta_workspace_has_no_match() {
        let temp_dir = TestTempDir::new().await;
        let runtime_workspace = temp_dir.path().join("runtime");
        let home_workspace = temp_dir.path().join("home");
        tokio::fs::create_dir_all(&runtime_workspace).await.unwrap();
        tokio::fs::create_dir_all(&home_workspace).await.unwrap();
        tokio::fs::write(home_workspace.join("notes.txt"), "alpha alpha")
            .await
            .unwrap();

        let result = edit_tool(&home_workspace)
            .call(
                mock_ctx_with_workspace(&runtime_workspace),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "alpha".to_string(),
                    new_string: "omega".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.replacements, 2);
        let written = tokio::fs::read_to_string(home_workspace.join("notes.txt"))
            .await
            .unwrap();
        assert_eq!(written, "omega omega");
    }

    #[tokio::test]
    async fn replaces_all_occurrences_in_utf8_file() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "alpha beta alpha\nalpha")
            .await
            .unwrap();

        let result = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "alpha".to_string(),
                    new_string: "omega".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.replacements, 3);
        assert_eq!(result.output.total_matches, 3);
        assert_eq!(result.output.size, 22);
        let written = tokio::fs::read_to_string(&target).await.unwrap();
        assert_eq!(written, "omega beta omega\nomega");
    }

    #[tokio::test]
    async fn defaults_limit_to_replace_all_when_missing_from_raw_args() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "alpha beta alpha\nalpha")
            .await
            .unwrap();

        let result = edit_tool(&workspace)
            .call_raw(
                mock_ctx(),
                serde_json::json!({
                    "path": "notes.txt",
                    "old_string": "alpha",
                    "new_string": "omega"
                }),
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["replacements"], 3);
        assert_eq!(result.output["total_matches"], 3);
        let written = tokio::fs::read_to_string(&target).await.unwrap();
        assert_eq!(written, "omega beta omega\nomega");
    }

    #[tokio::test]
    async fn respects_limit_when_replacing_utf8_file() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "alpha beta alpha\nalpha")
            .await
            .unwrap();

        let result = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "alpha".to_string(),
                    new_string: "x".to_string(),
                    limit: 1,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.replacements, 1);
        assert_eq!(result.output.total_matches, 3);
        assert_eq!(result.output.size, 18);
        let written = tokio::fs::read_to_string(&target).await.unwrap();
        assert_eq!(written, "x beta alpha\nalpha");
    }

    #[tokio::test]
    async fn rejects_empty_old_string() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("notes.txt"), "hello")
            .await
            .unwrap();

        let err = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: String::new(),
                    new_string: "world".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("Old string must not be empty"));
    }

    #[tokio::test]
    async fn returns_zero_matches_when_old_string_is_missing() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "hello").await.unwrap();

        let result = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "missing".to_string(),
                    new_string: "world".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.replacements, 0);
        assert_eq!(result.output.total_matches, 0);
        assert_eq!(result.output.size, 5);
        let written = tokio::fs::read_to_string(&target).await.unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn rejects_non_utf8_files() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("payload.bin"), [0x66, 0x6f, 0x80, 0x6f])
            .await
            .unwrap();

        let err = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "payload.bin".to_string(),
                    old_string: "foo".to_string(),
                    new_string: "bar".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("Editing binary or unsupported-encoding files is not supported")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn preserves_permissions_when_replacing_existing_file() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "before before").await.unwrap();
        tokio::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o640))
            .await
            .unwrap();

        let result = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "before".to_string(),
                    new_string: "after".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.replacements, 2);
        assert_eq!(result.output.total_matches, 2);

        let mode = tokio::fs::metadata(&target)
            .await
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn edits_files_from_a_symlinked_workspace_root() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let workspace_link = temp_dir.path().join("workspace-link");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("notes.txt"), "hello hello")
            .await
            .unwrap();
        symlink(&workspace, &workspace_link).unwrap();

        let result = edit_tool(&workspace_link)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "notes.txt".to_string(),
                    old_string: "hello".to_string(),
                    new_string: "world".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.replacements, 2);
        assert_eq!(result.output.total_matches, 2);
        let written = tokio::fs::read_to_string(workspace.join("notes.txt"))
            .await
            .unwrap();
        assert_eq!(written, "world world");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_editing_symbolic_link_target() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("real.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "before").await.unwrap();
        symlink(&target, workspace.join("alias.txt")).unwrap();

        let err = edit_tool(&workspace)
            .call(
                mock_ctx(),
                EditFileArgs {
                    path: "alias.txt".to_string(),
                    old_string: "before".to_string(),
                    new_string: "after".to_string(),
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("Writing to symbolic links is not allowed")
        );
    }
}
