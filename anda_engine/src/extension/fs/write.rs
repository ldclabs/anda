use anda_core::{BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput};
use ic_auth_types::ByteBufB64;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{borrow::Cow, path::PathBuf, str::FromStr};

use super::{
    BASE64_ENCODING, UTF8_ENCODING, atomic_write_file, default_write_encoding, ensure_regular_file,
    resolve_write_path,
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
    /// File content encoded as UTF-8 text or base64, depending on `encoding`.
    pub content: String,
    /// Content encoding. Supported values are `utf8` and `base64`.
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

pub type WriteFileHook = DynToolHook<WriteFileArgs, WriteFileOutput>;

#[derive(Clone)]
pub struct WriteFileTool {
    work_dir: PathBuf,
    description: String,
}

impl WriteFileTool {
    /// Tool name used for registration and function definition.
    pub const NAME: &'static str = "write_file";

    /// Create a new `WriteFileTool` with the default working directory.
    /// You can override the working directory for each call by including a `work_dir` field in the tool call's context meta extra.
    pub fn new(work_dir: PathBuf) -> Self {
        let description =
            "Atomically write files to the filesystem in the workspace directory".to_string();
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

impl Tool<BaseCtx> for WriteFileTool {
    type Args = WriteFileArgs;
    type Output = WriteFileOutput;

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
        let hook = ctx.get_state::<WriteFileHook>();

        let args = if let Some(hook) = &hook {
            hook.before_tool_call(&ctx, args).await?
        } else {
            args
        };

        let work_dir = ctx
            .meta()
            .get_extra_as::<String>("work_dir")
            .map(PathBuf::from)
            .map(Cow::Owned)
            .unwrap_or_else(|| Cow::Borrowed(&self.work_dir));

        let resolved_path = resolve_write_path(&work_dir, &args.path).await?;

        let data = decode_content(args.content, &args.encoding)?;

        let existing_permissions = match tokio::fs::metadata(&resolved_path).await {
            Ok(meta) => {
                ensure_regular_file(&meta, "Writing multiply-linked files is not allowed")?;

                Some(meta.permissions())
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                if let Some(parent) = resolved_path.parent() {
                    // Ensure parent directories exist for newly created files.
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|err| format!("Failed to create parent directories: {err}"))?;
                }

                None
            }
            Err(err) => return Err(format!("Failed to read file metadata: {err}").into()),
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
    use crate::{
        engine::EngineBuilder,
        extension::fs::{commit_atomic_replace, write_temp_file_for_atomic_replace},
    };
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

    fn write_tool(work_dir: &Path) -> WriteFileTool {
        WriteFileTool::new(work_dir.to_path_buf())
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
