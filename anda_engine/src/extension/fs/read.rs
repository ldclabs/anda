use anda_core::{BoxError, FunctionDefinition, Json, Resource, Tool, ToolOutput};
use ic_auth_types::ByteBufB64;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

use super::{BASE64_ENCODING, MAX_FILE_SIZE_BYTES, UTF8_ENCODING, has_multiple_hard_links};
use crate::{context::BaseCtx, hook::Hook};

/// Arguments for filesystem read operations.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FsReadArgs {
    /// Relative or absolute path to a file inside the workspace.
    pub path: String,
    /// Zero-based line offset for UTF-8 text output.
    #[serde(default)]
    pub offset: usize,
    /// Maximum number of UTF-8 lines to return. `0` means all remaining lines.
    #[serde(default)]
    pub limit: usize,
}

/// Normalized result returned by a filesystem read operation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FsReadOutput {
    /// File content as UTF-8 text or base64-encoded bytes for non-UTF-8 files.
    pub content: String,
    /// The encoding of the file content.
    pub encoding: String,
    pub size: u64,
    /// The MIME type of the file content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// The number of lines in the file content, if the content is UTF-8 text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_lines: Option<usize>,
}

#[derive(Clone)]
pub struct FsReadTool {
    work_dir: PathBuf,

    hook: Option<Arc<dyn Hook>>,
}

impl FsReadTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "fs_read";

    /// Create a new `FsReadTool` with the specified working directory.
    pub fn new(work_dir: PathBuf, hook: Option<Arc<dyn Hook>>) -> Self {
        Self { work_dir, hook }
    }
}

impl Tool<BaseCtx> for FsReadTool {
    type Args = FsReadArgs;
    type Output = Json;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        format!(
            "Read files from the filesystem in the workspace directory: {}",
            self.work_dir.display()
        )
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
                        "description": "Path to the file. Relative paths resolve from the workspace; paths outside the workspace are not allowed."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based line offset for UTF-8 text output (default: 0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of UTF-8 text lines to return (default: 0, all remaining lines)"
                    }
                },
                "required": ["path"]
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
        if let Some(hook) = &self.hook {
            hook.on_tool_start(&ctx, &self.name()).await?;
        }

        // Canonicalize the workspace root as well so symlinked workspace paths behave consistently.
        let resolved_work_dir = tokio::fs::canonicalize(&self.work_dir)
            .await
            .map_err(|err| format!("Failed to resolve workspace path: {err}"))?;

        let path = self.work_dir.join(&args.path);
        let resolved_path = match tokio::fs::canonicalize(&path).await {
            Ok(p) => p,
            Err(err) => return Err(format!("Failed to resolve file path: {err}").into()),
        };
        if !resolved_path.starts_with(&resolved_work_dir) {
            return Err("Access to paths outside the workspace is not allowed".into());
        }

        let meta = tokio::fs::metadata(&resolved_path)
            .await
            .map_err(|err| format!("Failed to read file metadata: {err}"))?;

        if has_multiple_hard_links(&meta) {
            return Err("Reading multiply-linked file is not allowed".into());
        }

        if !meta.is_file() {
            return Err("Path does not point to a regular file".into());
        }

        if meta.len() > MAX_FILE_SIZE_BYTES {
            return Err(format!(
                "File size {} exceeds maximum allowed size of {} bytes",
                meta.len(),
                MAX_FILE_SIZE_BYTES
            )
            .into());
        }

        let data = tokio::fs::read(&resolved_path)
            .await
            .map_err(|err| format!("Failed to read file: {err}"))?;
        let mut output = FsReadOutput {
            content: String::new(),
            encoding: UTF8_ENCODING.to_string(),
            size: meta.len(),
            ..Default::default()
        };
        if let Some(kind) = infer::get(&data) {
            output.mime_type = Some(kind.mime_type().to_string());
        }
        match String::from_utf8(data) {
            Ok(text) => {
                let all_lines = text.lines();
                output.total_lines = Some(all_lines.clone().count());
                if args.offset == 0 && args.limit == 0 {
                    output.content = text;
                } else if args.limit == 0 {
                    output.content = all_lines.skip(args.offset).collect::<Vec<_>>().join("\n");
                } else {
                    output.content = all_lines
                        .skip(args.offset)
                        .take(args.limit)
                        .collect::<Vec<_>>()
                        .join("\n");
                }
            }
            Err(v) => {
                output.content = ByteBufB64(v.into_bytes()).to_base64();
                output.encoding = BASE64_ENCODING.to_string();
            }
        }

        if let Some(hook) = &self.hook {
            return hook
                .on_tool_end(&ctx, &self.name(), ToolOutput::new(json!(output)))
                .await;
        }

        Ok(ToolOutput::new(json!(output)))
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

    fn read_tool(work_dir: &Path) -> FsReadTool {
        FsReadTool::new(work_dir.to_path_buf(), None)
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
                FsReadArgs {
                    path: "notes.txt".to_string(),
                    offset: 1,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["content"], "one\ntwo\nthree");
        assert_eq!(result.output["encoding"], "utf8");
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
                FsReadArgs {
                    path: "notes.txt".to_string(),
                    offset: 1,
                    limit: 2,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["content"], "one\ntwo");
        assert_eq!(result.output["size"], 19);
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
                FsReadArgs {
                    path: "payload.bin".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["content"], ByteBufB64(binary).to_base64());
        assert_eq!(result.output["encoding"], "base64");
        assert_eq!(result.output["size"], 4);
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
                FsReadArgs {
                    path: "notes.txt".to_string(),
                    offset: 0,
                    limit: 0,
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["content"], "hello\nworld\n");
        assert_eq!(result.output["encoding"], "utf8");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlink_escape_outside_workspace() {
        use std::os::unix::fs::symlink;

        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        let external = temp_dir.path().join("secret.txt");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(&external, "secret").await.unwrap();
        symlink(&external, workspace.join("secret-link.txt")).unwrap();

        let err = read_tool(&workspace)
            .call(
                mock_ctx(),
                FsReadArgs {
                    path: "secret-link.txt".to_string(),
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
