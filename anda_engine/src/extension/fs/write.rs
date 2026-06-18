//! Atomic file write tool for configured workspaces.
//!
//! Supports UTF-8, base64, and selected legacy text encodings through the
//! shared filesystem encoding helpers.

use anda_core::{
    BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolGroupInfo, ToolOutput,
};
use ic_auth_types::ByteBufB64;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{path::PathBuf, str::FromStr};

use super::{
    BASE64_ENCODING, FileTextEncodeError, atomic_write_file, default_write_encoding,
    encode_file_text, ensure_regular_file, format_workspaces, normalize_workspaces,
    resolve_write_path_in_workspaces, tool_workspaces,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

/// Arguments for filesystem write operations.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WriteFileArgs {
    /// Relative or absolute path to a file inside the workspace.
    pub path: String,
    /// File content encoded as text or base64, depending on `encoding`.
    pub content: String,
    /// Content encoding. Supported values are `utf8`, `base64`, and text encodings such as `gbk`.
    #[serde(default = "default_write_encoding")]
    pub encoding: String,
}

impl Default for WriteFileArgs {
    fn default() -> Self {
        Self {
            path: String::new(),
            content: String::new(),
            encoding: default_write_encoding(),
        }
    }
}

/// Normalized result returned by a filesystem write operation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WriteFileOutput {
    /// Number of bytes written to the target file.
    pub size: u64,
}

/// Typed hook for write-file tool calls.
pub type WriteFileHook = DynToolHook<WriteFileArgs, WriteFileOutput>;

/// Tool implementation for atomic writes inside configured workspaces.
#[derive(Clone)]
pub struct WriteFileTool {
    workspaces: Vec<PathBuf>,
    description: String,
}

impl WriteFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "write_file";

    /// Create a new `WriteFileTool` with the default workspace directory.
    /// You can add workspace directories for each call by including `workspace` or `workspaces` in the tool call's context meta extra.
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_workspaces([workspace])
    }

    /// Create a new `WriteFileTool` with the default workspace directories.
    /// Context meta workspaces take precedence over these defaults at call time.
    pub fn with_workspaces<I>(workspaces: I) -> Self
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let workspaces = normalize_workspaces(workspaces);
        let description = format!(
            "Atomically write files to the filesystem in the workspace directories ({})",
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

impl Tool<BaseCtx> for WriteFileTool {
    type Args = WriteFileArgs;
    type Output = WriteFileOutput;

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
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Relative paths resolve from the configured workspaces in priority order; absolute paths must be inside one configured workspace."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file. If encoding is 'base64', this should be base64-encoded data; otherwise it is text encoded with the requested encoding."
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Encoding of the content. Can be 'utf8', 'base64', or a supported text encoding such as 'gbk'. Defaults to 'utf8'."
                    }
                },
                "required": ["path", "content", "encoding"],
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
        let hook = ctx.get_state::<WriteFileHook>();

        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let workspaces = tool_workspaces(ctx.meta(), &self.workspaces);
        let resolved = resolve_write_path_in_workspaces(&workspaces, &args.path).await?;
        let workspace_display = resolved.workspace.display().to_string();
        let resolved_path = resolved.path;

        let data = decode_content(
            args.content,
            &args.encoding,
            &args.path,
            &workspace_display,
            &resolved_path,
        )?;

        let existing_permissions = match tokio::fs::metadata(&resolved_path).await {
            Ok(meta) => {
                ensure_regular_file(
                    &meta,
                    &resolved_path,
                    "Writing multiply-linked files is not allowed",
                )?;

                Some(meta.permissions())
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                if let Some(parent) = resolved_path.parent() {
                    // Ensure parent directories exist for newly created files.
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|err| {
                            format!(
                                "Failed to create parent directories (workspace: {}, requested_path: {}, resolved_path: {}, parent_path: {}): {err}",
                                workspace_display,
                                args.path,
                                resolved_path.display(),
                                parent.display()
                            )
                        })?;
                }

                None
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

        let size = data.len() as u64;
        atomic_write_file(&resolved_path, &data, existing_permissions.as_ref()).await?;

        if let Some(hook) = &hook {
            return hook
                .after_tool_call(&ctx, ToolOutput::new(WriteFileOutput { size }))
                .await;
        }

        Ok(ToolOutput::new(WriteFileOutput { size }))
    }
}
/// Decodes content according to the requested encoding.
fn decode_content(
    content: String,
    encoding: &str,
    requested_path: &str,
    workspace: &str,
    resolved_path: &std::path::Path,
) -> Result<Vec<u8>, BoxError> {
    match encoding {
        BASE64_ENCODING => ByteBufB64::from_str(&content)
            .map(|decoded| decoded.0)
            .map_err(|err| {
                format!(
                    "Failed to decode base64 content (workspace: {}, requested_path: {}, resolved_path: {}, encoding: {}): {err}",
                    workspace,
                    requested_path,
                    resolved_path.display(),
                    encoding
                )
                .into()
            }),
        text_encoding => encode_file_text(&content, text_encoding).map_err(|err| match err {
            FileTextEncodeError::UnsupportedEncoding => format!(
                "Unsupported encoding {text_encoding:?}. Expected 'utf8', 'base64', or a supported text encoding such as 'gbk' (workspace: {}, requested_path: {}, resolved_path: {})",
                workspace,
                requested_path,
                resolved_path.display()
            )
            .into(),
            FileTextEncodeError::UnmappableCharacters => format!(
                "Failed to encode text content (workspace: {}, requested_path: {}, resolved_path: {}, encoding: {}): {err}",
                workspace,
                requested_path,
                resolved_path.display(),
                text_encoding
            )
            .into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::EngineBuilder,
        extension::fs::{UTF8_ENCODING, commit_atomic_replace, write_temp_file_for_atomic_replace},
    };
    use serde_json::json;
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
    };

    struct TestTempDir(PathBuf);

    impl TestTempDir {
        async fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("anda-fs-write-test-{:016x}", rand::random::<u64>()));
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

    fn write_tool(workspace: &Path) -> WriteFileTool {
        WriteFileTool::new(workspace.to_path_buf())
    }

    struct RewritingWriteHook;

    #[async_trait::async_trait]
    impl ToolHook<WriteFileArgs, WriteFileOutput> for RewritingWriteHook {
        async fn before_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut args: WriteFileArgs,
        ) -> Result<WriteFileArgs, BoxError> {
            args.path = "hook.txt".to_string();
            args.content = "hooked".to_string();
            args.encoding = UTF8_ENCODING.to_string();
            Ok(args)
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut output: ToolOutput<WriteFileOutput>,
        ) -> Result<ToolOutput<WriteFileOutput>, BoxError> {
            output.output.size += 10;
            Ok(output)
        }
    }

    #[tokio::test]
    async fn defaults_metadata_invalid_base64_and_hooks_are_covered() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let default_args = WriteFileArgs::default();
        assert_eq!(default_args.path, "");
        assert_eq!(default_args.content, "");
        assert_eq!(default_args.encoding, UTF8_ENCODING);

        let tool = write_tool(&workspace).with_description("custom write".to_string());
        assert_eq!(tool.name(), WriteFileTool::NAME);
        assert_eq!(tool.description(), "custom write");
        let definition = tool.definition();
        assert_eq!(definition.name, WriteFileTool::NAME);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(
            definition.parameters["required"],
            json!(["path", "content", "encoding"])
        );

        let err = tool
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "bad.bin".to_string(),
                    content: "not base64%%".to_string(),
                    encoding: BASE64_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Failed to decode base64 content"));

        let ctx = mock_ctx();
        ctx.set_state(WriteFileHook::new(Arc::new(RewritingWriteHook)));
        let hooked = tool
            .call(
                ctx,
                WriteFileArgs {
                    path: "ignored.txt".to_string(),
                    content: "ignored".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(hooked.output.size, 16);
        assert_eq!(
            tokio::fs::read_to_string(workspace.join("hook.txt"))
                .await
                .unwrap(),
            "hooked"
        );
    }

    #[tokio::test]
    async fn writes_existing_file_in_default_workspace_when_meta_workspace_has_no_match() {
        let temp_dir = TestTempDir::new().await;
        let runtime_workspace = temp_dir.path().join("runtime");
        let home_workspace = temp_dir.path().join("home");
        tokio::fs::create_dir_all(&runtime_workspace).await.unwrap();
        tokio::fs::create_dir_all(&home_workspace).await.unwrap();
        tokio::fs::write(home_workspace.join("notes.txt"), "before")
            .await
            .unwrap();

        let result = write_tool(&home_workspace)
            .call(
                mock_ctx_with_workspace(&runtime_workspace),
                WriteFileArgs {
                    path: "notes.txt".to_string(),
                    content: "after".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.size, 5);
        let written = tokio::fs::read_to_string(home_workspace.join("notes.txt"))
            .await
            .unwrap();
        assert_eq!(written, "after");
        assert!(matches!(
            tokio::fs::metadata(runtime_workspace.join("notes.txt")).await,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[tokio::test]
    async fn writes_new_relative_file_in_meta_workspace_first() {
        let temp_dir = TestTempDir::new().await;
        let runtime_workspace = temp_dir.path().join("runtime");
        let home_workspace = temp_dir.path().join("home");
        tokio::fs::create_dir_all(&runtime_workspace).await.unwrap();
        tokio::fs::create_dir_all(&home_workspace).await.unwrap();

        write_tool(&home_workspace)
            .call(
                mock_ctx_with_workspace(&runtime_workspace),
                WriteFileArgs {
                    path: "notes.txt".to_string(),
                    content: "runtime".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        let written = tokio::fs::read_to_string(runtime_workspace.join("notes.txt"))
            .await
            .unwrap();
        assert_eq!(written, "runtime");
        assert!(matches!(
            tokio::fs::metadata(home_workspace.join("notes.txt")).await,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[tokio::test]
    async fn creates_new_file_with_missing_parent_directories() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let result = write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "nested/dir/output.txt".to_string(),
                    content: "hello".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.size, 5);
        let written = tokio::fs::read_to_string(workspace.join("nested/dir/output.txt"))
            .await
            .unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn defaults_encoding_to_utf8_when_missing_from_raw_args() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        write_tool(&workspace)
            .call_raw(
                mock_ctx(),
                json!({
                    "path": "notes.txt",
                    "content": "hello"
                }),
                Vec::new(),
            )
            .await
            .unwrap();

        let written = tokio::fs::read_to_string(workspace.join("notes.txt"))
            .await
            .unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn writes_base64_encoded_content() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let binary = vec![0x00, 0x7f, 0x80, 0xff];
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let result = write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "payload.bin".to_string(),
                    content: ByteBufB64(binary.clone()).to_base64(),
                    encoding: BASE64_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.size, 4);
        let written = tokio::fs::read(workspace.join("payload.bin"))
            .await
            .unwrap();
        assert_eq!(written, binary);
    }

    #[tokio::test]
    async fn writes_legacy_text_encoding_content() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let result = write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "notes.txt".to_string(),
                    content: "中文.txt\n".to_string(),
                    encoding: "gbk".to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.size, 9);
        let written = tokio::fs::read(workspace.join("notes.txt")).await.unwrap();
        assert_eq!(
            written,
            vec![0xd6, 0xd0, 0xce, 0xc4, b'.', b't', b'x', b't', b'\n']
        );
    }

    #[tokio::test]
    async fn rejects_unsupported_encoding() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let err = write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "notes.txt".to_string(),
                    content: "hello".to_string(),
                    encoding: "hex".to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("Unsupported encoding"));
    }

    #[tokio::test]
    async fn staged_atomic_replace_keeps_previous_content_visible_until_commit() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "before").await.unwrap();

        let metadata = tokio::fs::metadata(&target).await.unwrap();
        let temp_path =
            write_temp_file_for_atomic_replace(&target, b"after", Some(&metadata.permissions()))
                .await
                .unwrap();

        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), "before");
        assert_eq!(
            tokio::fs::read_to_string(&temp_path).await.unwrap(),
            "after"
        );

        commit_atomic_replace(&temp_path, &target).await.unwrap();

        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), "after");
        assert!(matches!(
            tokio::fs::metadata(&temp_path).await,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn preserves_permissions_when_replacing_existing_file() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("notes.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "before").await.unwrap();
        tokio::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o640))
            .await
            .unwrap();

        write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "notes.txt".to_string(),
                    content: "after".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

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
    async fn writes_files_from_a_symlinked_workspace_root() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let workspace_link = temp_dir.path().join("workspace-link");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        symlink(&workspace, &workspace_link).unwrap();

        let result = write_tool(&workspace_link)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "notes.txt".to_string(),
                    content: "hello".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.size, 5);
        let written = tokio::fs::read_to_string(workspace.join("notes.txt"))
            .await
            .unwrap();
        assert_eq!(written, "hello");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_writing_to_symbolic_link_target() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let target = workspace.join("real.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&target, "before").await.unwrap();
        symlink(&target, workspace.join("alias.txt")).unwrap();

        let err = write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "alias.txt".to_string(),
                    content: "after".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
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

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlink_escape_outside_workspace_for_new_files() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("external");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&external).await.unwrap();
        symlink(&external, workspace.join("escape")).unwrap();

        let err = write_tool(&workspace)
            .call(
                mock_ctx(),
                WriteFileArgs {
                    path: "escape/secret.txt".to_string(),
                    content: "secret".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
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
