//! File read tool for configured workspaces.
//!
//! Text files are decoded with platform-aware fallbacks, while binary or
//! unsupported files are returned as base64. Large inline output is truncated
//! with paging metadata.

use anda_core::{
    BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolGroupInfo, ToolOutput,
};
use ic_auth_types::ByteBufB64;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;

use super::{
    BASE64_ENCODING, MAX_FILE_SIZE_BYTES, MAX_INLINE_CONTENT_BYTES, UTF8_ENCODING,
    decode_file_text, ensure_file_size_within_limit, ensure_regular_file, format_workspaces,
    normalize_workspaces, resolve_read_path_in_workspaces, tool_workspaces, truncate_inline_text,
};
use crate::{
    context::BaseCtx,
    hook::{DynToolHook, ToolHook},
};

/// Arguments for filesystem read operations.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ReadFileArgs {
    /// Relative or absolute path to a file inside the workspace.
    pub path: String,
    /// Zero-based line offset for text output.
    #[serde(default)]
    pub offset: usize,
    /// Maximum number of text lines to return. `0` means all remaining lines.
    #[serde(default)]
    pub limit: usize,
}

/// Normalized result returned by a filesystem read operation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ReadFileOutput {
    /// File content as decoded text or base64-encoded bytes for unsupported/binary files.
    pub content: String,
    /// The encoding of the file content.
    pub encoding: String,
    /// The size of the file in bytes.
    pub size: u64,
    /// The MIME type of the file content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// The number of lines in the file content, if the content is decoded text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_lines: Option<usize>,
    /// True when `content` was truncated to the inline output limit. Page through
    /// large text files with `offset` and `limit`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Typed hook for read-file tool calls.
pub type ReadFileHook = DynToolHook<ReadFileArgs, ReadFileOutput>;

/// Tool implementation for reading files inside configured workspaces.
#[derive(Clone)]
pub struct ReadFileTool {
    workspaces: Vec<PathBuf>,
    description: String,
}

impl ReadFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "read_file";

    /// Create a new `ReadFileTool` with the default workspace directory.
    /// You can add workspace directories for each call by including `workspace` or `workspaces` in the tool call's context meta extra.
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_workspaces([workspace])
    }

    /// Create a new `ReadFileTool` with the default workspace directories.
    /// Context meta workspaces take precedence over these defaults at call time.
    pub fn with_workspaces<I>(workspaces: I) -> Self
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let workspaces = normalize_workspaces(workspaces);
        let description = format!(
            "Read files from the filesystem in the workspace directories ({})",
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

impl Tool<BaseCtx> for ReadFileTool {
    type Args = ReadFileArgs;
    type Output = ReadFileOutput;

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
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based line offset for decoded text output (default: 0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of decoded text lines to return (default: 0, all remaining lines). Responses are capped at 256KiB and marked with `truncated: true` when cut; use offset and limit to page through large files."
                    }
                },
                "required": ["path", "offset", "limit"],
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
        let hook = ctx.get_state::<ReadFileHook>();

        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let workspaces = tool_workspaces(ctx.meta(), &self.workspaces);
        let resolved = resolve_read_path_in_workspaces(&workspaces, &args.path).await?;
        let workspace_display = resolved.workspace.display().to_string();
        let resolved_path = resolved.path;

        let meta = tokio::fs::metadata(&resolved_path)
            .await
            .map_err(|err| {
                format!(
                    "Failed to read file metadata (workspace: {}, requested_path: {}, resolved_path: {}): {err}",
                    workspace_display,
                    args.path,
                    resolved_path.display()
                )
            })?;

        ensure_regular_file(
            &meta,
            &resolved_path,
            "Reading multiply-linked file is not allowed",
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
        let mut output = ReadFileOutput {
            content: String::new(),
            encoding: UTF8_ENCODING.to_string(),
            size: meta.len(),
            ..Default::default()
        };
        if let Some(kind) = infer2::get(&data) {
            output.mime_type = Some(kind.mime_type().to_string());
        }
        match decode_file_text(data) {
            Ok(decoded) => {
                output.encoding = decoded.encoding;
                let text = decoded.text;
                output.total_lines = Some(text.lines().count());
                if args.offset == 0 && args.limit == 0 {
                    output.content = text;
                } else if args.limit == 0 {
                    output.content = text
                        .lines()
                        .skip(args.offset)
                        .collect::<Vec<_>>()
                        .join("\n");
                } else {
                    output.content = text
                        .lines()
                        .skip(args.offset)
                        .take(args.limit)
                        .collect::<Vec<_>>()
                        .join("\n");
                }
                output.truncated =
                    truncate_inline_text(&mut output.content, MAX_INLINE_CONTENT_BYTES);
            }
            Err(mut bytes) => {
                // Cap binary previews as well; keep the length a multiple of 3 so the
                // base64 prefix decodes cleanly.
                let max_raw_bytes = MAX_INLINE_CONTENT_BYTES / 4 * 3;
                if bytes.len() > max_raw_bytes {
                    bytes.truncate(max_raw_bytes);
                    output.truncated = true;
                }
                output.content = ByteBufB64(bytes).to_base64();
                output.encoding = BASE64_ENCODING.to_string();
            }
        }

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
                .join(format!("anda-fs-read-test-{:016x}", rand::random::<u64>()));
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

    fn read_tool(workspace: &Path) -> ReadFileTool {
        ReadFileTool::new(workspace.to_path_buf())
    }

    struct RewritingReadHook;

    #[async_trait::async_trait]
    impl ToolHook<ReadFileArgs, ReadFileOutput> for RewritingReadHook {
        async fn before_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut args: ReadFileArgs,
        ) -> Result<ReadFileArgs, BoxError> {
            args.path = "hook.txt".to_string();
            args.offset = 1;
            args.limit = 1;
            Ok(args)
        }

        async fn after_tool_call(
            &self,
            _ctx: &BaseCtx,
            mut output: ToolOutput<ReadFileOutput>,
        ) -> Result<ToolOutput<ReadFileOutput>, BoxError> {
            output.output.content.push_str("\nhooked");
            Ok(output)
        }
    }

    #[tokio::test]
    async fn metadata_hooks_and_mime_detection_are_covered() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("hook.txt"), "zero\none\ntwo\n")
            .await
            .unwrap();
        tokio::fs::write(
            workspace.join("tiny.png"),
            [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
        )
        .await
        .unwrap();

        let tool = read_tool(&workspace).with_description("custom read".to_string());
        assert_eq!(tool.name(), ReadFileTool::NAME);
        assert_eq!(tool.description(), "custom read");
        let definition = tool.definition();
        assert_eq!(definition.name, ReadFileTool::NAME);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(
            definition.parameters["required"],
            json!(["path", "offset", "limit"])
        );

        let image = tool
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "tiny.png".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(image.output.encoding, BASE64_ENCODING);
        assert_eq!(image.output.mime_type.as_deref(), Some("image/png"));

        let ctx = mock_ctx();
        ctx.set_state(ReadFileHook::new(Arc::new(RewritingReadHook)));
        let hooked = tool
            .call(
                ctx,
                ReadFileArgs {
                    path: "ignored.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(hooked.output.content, "one\nhooked");
        assert_eq!(hooked.output.total_lines, Some(3));
    }

    #[tokio::test]
    async fn reads_from_default_workspace_when_meta_workspace_has_no_match() {
        let temp_dir = TestTempDir::new().await;
        let runtime_workspace = temp_dir.path().join("runtime");
        let home_workspace = temp_dir.path().join("home");
        tokio::fs::create_dir_all(&runtime_workspace).await.unwrap();
        tokio::fs::create_dir_all(&home_workspace).await.unwrap();
        tokio::fs::write(home_workspace.join("notes.txt"), "from home")
            .await
            .unwrap();

        let result = read_tool(&home_workspace)
            .call(
                mock_ctx_with_workspace(&runtime_workspace),
                ReadFileArgs {
                    path: "notes.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.content, "from home");
        assert_eq!(result.output.encoding, "utf8");
    }

    #[tokio::test]
    async fn applies_offset_when_limit_is_zero() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("notes.txt"), "zero\none\ntwo\nthree\n")
            .await
            .unwrap();

        let result = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "notes.txt".to_string(),
                    offset: 1,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.content, "one\ntwo\nthree");
        assert_eq!(result.output.encoding, "utf8");
    }

    #[tokio::test]
    async fn reads_requested_text_window() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("notes.txt"), "zero\none\ntwo\nthree\n")
            .await
            .unwrap();

        let result = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "notes.txt".to_string(),
                    offset: 1,
                    limit: 2,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.content, "one\ntwo");
        assert_eq!(result.output.size, 19);
    }

    #[tokio::test]
    async fn truncates_oversized_text_and_binary_content() {
        use crate::extension::fs::MAX_INLINE_CONTENT_BYTES;
        use std::str::FromStr;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let line = "0123456789abcdef\n";
        let total_lines = MAX_INLINE_CONTENT_BYTES / line.len() + 1024;
        let text = line.repeat(total_lines);
        tokio::fs::write(workspace.join("big.txt"), &text)
            .await
            .unwrap();

        let tool = read_tool(&workspace);
        let result = tool
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "big.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(result.output.truncated);
        assert!(result.output.content.len() <= MAX_INLINE_CONTENT_BYTES);
        assert!(result.output.content.ends_with('\n'));
        assert_eq!(result.output.total_lines, Some(total_lines));

        // Paging through the same file stays untruncated.
        let window = tool
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "big.txt".to_string(),
                    offset: total_lines - 2,
                    limit: 2,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(!window.output.truncated);
        assert_eq!(
            window.output.content,
            format!("{}\n{}", line.trim_end(), line.trim_end())
        );

        let mut binary = vec![0u8; MAX_INLINE_CONTENT_BYTES];
        binary[0] = 0xff;
        binary[1] = 0xfe;
        tokio::fs::write(workspace.join("big.bin"), &binary)
            .await
            .unwrap();
        let result = tool
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "big.bin".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();
        assert!(result.output.truncated);
        assert_eq!(result.output.encoding, BASE64_ENCODING);
        assert!(result.output.content.len() <= MAX_INLINE_CONTENT_BYTES);
        // The truncated base64 prefix still decodes to the head of the file.
        let decoded = ByteBufB64::from_str(&result.output.content).unwrap();
        assert_eq!(decoded.0.len(), MAX_INLINE_CONTENT_BYTES / 4 * 3);
        assert_eq!(&decoded.0[..2], &[0xff, 0xfe]);
    }

    #[tokio::test]
    async fn returns_base64_for_non_utf8_content() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let binary = vec![0xff, 0x00, 0x81, 0x7f];
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("payload.bin"), &binary)
            .await
            .unwrap();

        let result = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "payload.bin".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.content, ByteBufB64(binary).to_base64());
        assert_eq!(result.output.encoding, "base64");
        assert_eq!(result.output.size, 4);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reads_files_from_a_symlinked_workspace_root() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let workspace_link = temp_dir.path().join("workspace-link");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("notes.txt"), "hello\nworld\n")
            .await
            .unwrap();
        symlink(&workspace, &workspace_link).unwrap();

        let result = read_tool(&workspace_link)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "notes.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output.content, "hello\nworld\n");
        assert_eq!(result.output.encoding, "utf8");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_reading_through_symbolic_link_leaving_workspace() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("secret.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&external, "secret").await.unwrap();
        symlink(&external, workspace.join("secret-link.txt")).unwrap();

        // A workspace-local symlink whose target escapes the workspace must not expose host files.
        let err = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "secret-link.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("outside the workspace"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_reading_through_symbolic_linked_directory_leaving_workspace() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("external");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&external).await.unwrap();
        tokio::fs::write(external.join("secret.txt"), "secret")
            .await
            .unwrap();
        symlink(&external, workspace.join("linked-dir")).unwrap();

        // A symlinked directory that points outside the workspace must not expose its contents.
        let err = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "linked-dir/secret.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("outside the workspace"));
    }

    #[tokio::test]
    async fn rejects_absolute_path_outside_workspace() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("secret.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&external, "secret").await.unwrap();

        let err = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: external.to_string_lossy().into_owned(),
                    offset: 0,
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
    async fn rejects_parent_dir_escape_outside_workspace() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("secret.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&external, "secret").await.unwrap();

        let err = read_tool(&workspace)
            .call(
                mock_ctx(),
                ReadFileArgs {
                    path: "../secret.txt".to_string(),
                    offset: 0,
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
