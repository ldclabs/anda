use anda_core::{BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{borrow::Cow, path::PathBuf};

use super::{
    MAX_FILE_SIZE_BYTES, atomic_write_file, ensure_file_size_within_limit, ensure_regular_file,
    resolve_write_path,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

/// Arguments for filesystem edit operations.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EditFileArgs {
    /// Relative or absolute path to a UTF-8 file inside the workspace.
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
    /// Number of bytes in the resulting UTF-8 file.
    pub size: u64,
}

pub type EditFileHook = DynToolHook<EditFileArgs, EditFileOutput>;

#[derive(Clone)]
pub struct EditFileTool {
    work_dir: PathBuf,
    description: String,
}

impl EditFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "edit_file";

    /// Create a new `EditFileTool` with the default working directory.
    /// You can override the working directory for each call by including a `work_dir` field in the tool call's context meta extra.
    pub fn new(work_dir: PathBuf) -> Self {
        let description =
            "Atomically edit UTF-8 files in the workspace directory by replacing strings"
                .to_string();
        Self {
            work_dir,
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
                        "description": "Path to the UTF-8 file. Relative paths resolve from the workspace; paths outside the workspace are not allowed."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Old UTF-8 string to replace."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement UTF-8 string."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of replacements to apply (default: 0, replace all matches)."
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
            strict: None,
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

        if args.old_string.is_empty() {
            return Err("Old string must not be empty".into());
        }

        let work_dir = ctx
            .meta()
            .get_extra_as::<String>("work_dir")
            .map(PathBuf::from)
            .map(Cow::Owned)
            .unwrap_or_else(|| Cow::Borrowed(&self.work_dir));

        let resolved_path = resolve_write_path(&work_dir, &args.path).await?;
        let meta = match tokio::fs::metadata(&resolved_path).await {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err("Path does not point to an existing file".into());
            }
            Err(err) => return Err(format!("Failed to read file metadata: {err}").into()),
        };

        ensure_regular_file(&meta, "Editing multiply-linked files is not allowed")?;
        ensure_file_size_within_limit(&meta, MAX_FILE_SIZE_BYTES)?;

        let data = tokio::fs::read(&resolved_path)
            .await
            .map_err(|err| format!("Failed to read file: {err}"))?;
        let text = String::from_utf8(data)
            .map_err(|_| "Editing non-UTF-8 files is not supported".to_string())?;
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
                size: text.len() as u64,
            }
        } else {
            let updated = if args.limit == 0 {
                text.replace(&args.old_string, &args.new_string)
            } else {
                text.replacen(&args.old_string, &args.new_string, args.limit)
            };
            let size = updated.len() as u64;
            atomic_write_file(
                &resolved_path,
                updated.as_bytes(),
                Some(&meta.permissions()),
            )
            .await?;
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
    use std::path::{Path, PathBuf};

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

    fn edit_tool(work_dir: &Path) -> EditFileTool {
        EditFileTool::new(work_dir.to_path_buf())
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
                .contains("Editing non-UTF-8 files is not supported")
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
