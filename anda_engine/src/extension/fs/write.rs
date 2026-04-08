use anda_core::{BoxError, FunctionDefinition, Json, Resource, Tool, ToolOutput};
use ic_auth_types::ByteBufB64;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use super::{BASE64_ENCODING, UTF8_ENCODING, default_write_encoding, has_multiple_hard_links};
use crate::{context::BaseCtx, hook::Hook};

/// Arguments for filesystem write operations.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FsWriteArgs {
    /// Relative or absolute path to a file inside the workspace.
    pub path: String,
    /// File content encoded as UTF-8 text or base64, depending on `encoding`.
    pub content: String,
    /// Content encoding. Supported values are `utf8` and `base64`.
    #[serde(default = "default_write_encoding")]
    pub encoding: String,
}

impl Default for FsWriteArgs {
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
pub struct FsWriteOutput {
    /// Number of bytes written to the target file.
    pub size: u64,
}

#[derive(Clone)]
pub struct FsWriteTool {
    work_dir: PathBuf,

    hook: Option<Arc<dyn Hook>>,
}

impl FsWriteTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "fs_write";

    /// Create a new `FsWriteTool` with the specified working directory.
    pub fn new(work_dir: PathBuf, hook: Option<Arc<dyn Hook>>) -> Self {
        Self { work_dir, hook }
    }
}

impl Tool<BaseCtx> for FsWriteTool {
    type Args = FsWriteArgs;
    type Output = Json;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        format!(
            "Write files to the filesystem in the workspace directory: {}",
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
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file. If encoding is 'base64', this should be base64-encoded data."
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Encoding of the content. Can be 'utf8' or 'base64'. Defaults to 'utf8'."
                    }
                },
                "required": ["path", "content"]
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

        let resolved_path = resolve_write_path(&self.work_dir, &args.path).await?;

        let data = decode_content(args.content, &args.encoding)?;

        match tokio::fs::metadata(&resolved_path).await {
            Ok(meta) => {
                if has_multiple_hard_links(&meta) {
                    return Err("Writing multiply-linked files is not allowed".into());
                }

                if !meta.is_file() {
                    return Err("Path does not point to a regular file".into());
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                if let Some(parent) = resolved_path.parent() {
                    // Ensure parent directories exist for newly created files.
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|err| format!("Failed to create parent directories: {err}"))?;
                }
            }
            Err(err) => return Err(format!("Failed to read file metadata: {err}").into()),
        }

        let size = data.len() as u64;
        tokio::fs::write(&resolved_path, data)
            .await
            .map_err(|err| format!("Failed to write file: {err}"))?;

        if let Some(hook) = &self.hook {
            return hook
                .on_tool_end(
                    &ctx,
                    &self.name(),
                    ToolOutput::new(json!(FsWriteOutput { size })),
                )
                .await;
        }

        Ok(ToolOutput::new(json!(FsWriteOutput { size })))
    }
}

/// Resolves a write target inside the workspace, even when the destination does not yet exist.
async fn resolve_write_path(work_dir: &Path, user_path: &str) -> Result<PathBuf, BoxError> {
    let resolved_work_dir = tokio::fs::canonicalize(work_dir)
        .await
        .map_err(|err| format!("Failed to resolve workspace path: {err}"))?;
    let path = work_dir.join(user_path);

    match tokio::fs::symlink_metadata(&path).await {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err("Writing to symbolic links is not allowed".into());
            }

            let resolved_path = tokio::fs::canonicalize(&path)
                .await
                .map_err(|err| format!("Failed to resolve file path: {err}"))?;
            if !resolved_path.starts_with(&resolved_work_dir) {
                return Err("Access to paths outside the workspace is not allowed".into());
            }

            Ok(resolved_path)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let (existing_ancestor, missing_components) = nearest_existing_ancestor(&path).await?;
            let resolved_ancestor = tokio::fs::canonicalize(&existing_ancestor)
                .await
                .map_err(|err| format!("Failed to resolve file path: {err}"))?;
            if !resolved_ancestor.starts_with(&resolved_work_dir) {
                return Err("Access to paths outside the workspace is not allowed".into());
            }

            Ok(missing_components
                .into_iter()
                .rev()
                .fold(resolved_ancestor, |acc, component| acc.join(component)))
        }
        Err(err) => Err(format!("Failed to inspect file path: {err}").into()),
    }
}

/// Finds the nearest existing path component and returns the missing tail components.
async fn nearest_existing_ancestor(path: &Path) -> Result<(PathBuf, Vec<OsString>), BoxError> {
    let mut current = path.to_path_buf();
    let mut missing_components = Vec::new();

    loop {
        match tokio::fs::symlink_metadata(&current).await {
            Ok(_) => return Ok((current, missing_components)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let file_name = current.file_name().ok_or_else(|| {
                    "Access to paths outside the workspace is not allowed".to_string()
                })?;
                missing_components.push(file_name.to_os_string());
                current = current
                    .parent()
                    .ok_or_else(|| {
                        "Access to paths outside the workspace is not allowed".to_string()
                    })?
                    .to_path_buf();
            }
            Err(err) => return Err(format!("Failed to inspect file path: {err}").into()),
        }
    }
}

/// Decodes content according to the requested encoding.
fn decode_content(content: String, encoding: &str) -> Result<Vec<u8>, BoxError> {
    match encoding {
        UTF8_ENCODING => Ok(content.into_bytes()),
        BASE64_ENCODING => ByteBufB64::from_str(&content)
            .map(|decoded| decoded.0)
            .map_err(|err| format!("Failed to decode base64 content: {err}").into()),
        other => Err(format!("Unsupported encoding {other:?}. Expected 'utf8' or 'base64'").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use serde_json::json;
    use std::path::{Path, PathBuf};

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

    fn write_tool(work_dir: &Path) -> FsWriteTool {
        FsWriteTool::new(work_dir.to_path_buf(), None)
    }

    #[tokio::test]
    async fn creates_new_file_with_missing_parent_directories() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let result = write_tool(&workspace)
            .call(
                mock_ctx(),
                FsWriteArgs {
                    path: "nested/dir/output.txt".to_string(),
                    content: "hello".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["size"], 5);
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
                FsWriteArgs {
                    path: "payload.bin".to_string(),
                    content: ByteBufB64(binary.clone()).to_base64(),
                    encoding: BASE64_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["size"], 4);
        let written = tokio::fs::read(workspace.join("payload.bin"))
            .await
            .unwrap();
        assert_eq!(written, binary);
    }

    #[tokio::test]
    async fn rejects_unsupported_encoding() {
        let temp_dir = TestTempDir::new().await;
        let workspace = temp_dir.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let err = write_tool(&workspace)
            .call(
                mock_ctx(),
                FsWriteArgs {
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
                FsWriteArgs {
                    path: "notes.txt".to_string(),
                    content: "hello".to_string(),
                    encoding: UTF8_ENCODING.to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.output["size"], 5);
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
                FsWriteArgs {
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
                FsWriteArgs {
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
