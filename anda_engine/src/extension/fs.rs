use anda_core::{
    BoxError, RequestMeta, platform_text_encoding, text_encoding_for_label, text_encoding_label,
    text_from_bytes_with_encoding,
};
use encoding_rs::Encoding;
use std::{
    ffi::OsString,
    fmt,
    fs::{Metadata, Permissions},
    path::{Component, Path, PathBuf},
};
use tokio::io::AsyncWriteExt;

mod edit;
mod read;
mod search;
mod write;

pub use edit::*;
pub use read::*;
pub use search::*;
pub use write::*;

pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum bytes of file content returned inline in a tool response. Larger
/// content is truncated so a single read cannot flood the model context.
pub(crate) const MAX_INLINE_CONTENT_BYTES: usize = 256 * 1024;

pub(crate) const UTF8_ENCODING: &str = "utf8";
pub(crate) const BASE64_ENCODING: &str = "base64";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedFileText {
    pub(crate) text: String,
    pub(crate) encoding: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileTextEncodeError {
    UnsupportedEncoding,
    UnmappableCharacters,
}

impl fmt::Display for FileTextEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEncoding => f.write_str("unsupported text encoding"),
            Self::UnmappableCharacters => f.write_str(
                "content contains characters not representable in the requested encoding",
            ),
        }
    }
}

impl std::error::Error for FileTextEncodeError {}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedFilePath {
    pub(crate) workspace: PathBuf,
    pub(crate) path: PathBuf,
}

pub(crate) fn normalize_workspaces<I>(workspaces: I) -> Vec<PathBuf>
where
    I: IntoIterator<Item = PathBuf>,
{
    let mut normalized = Vec::new();
    for workspace in workspaces {
        push_workspace(&mut normalized, workspace);
    }

    normalized
}

pub(crate) fn tool_workspaces(meta: &RequestMeta, defaults: &[PathBuf]) -> Vec<PathBuf> {
    let mut workspaces = Vec::new();

    if let Some(workspace) = meta.get_extra_as::<PathBuf>("workspace") {
        push_workspace(&mut workspaces, workspace);
    } else if let Some(extra_workspaces) = meta.get_extra_as::<Vec<PathBuf>>("workspace") {
        for workspace in extra_workspaces {
            push_workspace(&mut workspaces, workspace);
        }
    }

    if let Some(workspace) = meta.get_extra_as::<PathBuf>("workspaces") {
        push_workspace(&mut workspaces, workspace);
    } else if let Some(extra_workspaces) = meta.get_extra_as::<Vec<PathBuf>>("workspaces") {
        for workspace in extra_workspaces {
            push_workspace(&mut workspaces, workspace);
        }
    }

    for workspace in defaults {
        push_workspace(&mut workspaces, workspace.clone());
    }

    workspaces
}

pub(crate) fn format_workspaces(workspaces: &[PathBuf]) -> String {
    if workspaces.is_empty() {
        return "<none>".to_string();
    }

    workspaces
        .iter()
        .map(|workspace| workspace.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn push_workspace(workspaces: &mut Vec<PathBuf>, workspace: PathBuf) {
    if workspace.as_os_str().is_empty() {
        return;
    }

    if !workspaces.iter().any(|existing| existing == &workspace) {
        workspaces.push(workspace);
    }
}

pub(crate) async fn resolve_read_path_in_workspaces(
    workspaces: &[PathBuf],
    user_path: &str,
) -> Result<ResolvedFilePath, BoxError> {
    let mut errors = Vec::new();

    for workspace in workspaces {
        match resolve_read_path(workspace, user_path).await {
            Ok(path) => {
                return Ok(ResolvedFilePath {
                    workspace: workspace.clone(),
                    path,
                });
            }
            Err(err) => errors.push(format!("{}: {err}", workspace.display())),
        }
    }

    Err(workspace_access_error(
        "Path",
        "requested_path",
        user_path,
        workspaces,
        errors,
    ))
}

pub(crate) async fn resolve_write_path_in_workspaces(
    workspaces: &[PathBuf],
    user_path: &str,
) -> Result<ResolvedFilePath, BoxError> {
    let requested_path = Path::new(user_path);

    if requested_path.is_relative() {
        for workspace in workspaces {
            let candidate_path = workspace.join(requested_path);
            match tokio::fs::symlink_metadata(&candidate_path).await {
                Ok(_) => {
                    let path = resolve_write_path(workspace, user_path).await?;
                    return Ok(ResolvedFilePath {
                        workspace: workspace.clone(),
                        path,
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => {
                    return Err(format!(
                        "Failed to inspect file path (workspace: {}, requested_path: {}, candidate_path: {}): {err}",
                        workspace.display(),
                        user_path,
                        candidate_path.display()
                    )
                    .into());
                }
            }
        }
    }

    let mut errors = Vec::new();
    for workspace in workspaces {
        match resolve_write_path(workspace, user_path).await {
            Ok(path) => {
                return Ok(ResolvedFilePath {
                    workspace: workspace.clone(),
                    path,
                });
            }
            Err(err) => errors.push(format!("{}: {err}", workspace.display())),
        }
    }

    Err(workspace_access_error(
        "Path",
        "requested_path",
        user_path,
        workspaces,
        errors,
    ))
}

pub(crate) fn workspace_access_error(
    subject: &str,
    request_label: &str,
    requested_value: &str,
    workspaces: &[PathBuf],
    errors: Vec<String>,
) -> BoxError {
    let details = if errors.is_empty() {
        String::new()
    } else {
        format!("; errors: {}", errors.join("; "))
    };

    format!(
        "{subject} is not accessible from any configured workspace ({request_label}: {}, workspaces: [{}]){}",
        requested_value,
        format_workspaces(workspaces),
        details
    )
    .into()
}

/// Resolves an existing read target reachable from the workspace namespace.
pub async fn resolve_read_path(workspace: &Path, user_path: &str) -> Result<PathBuf, BoxError> {
    let resolved_workspace = resolve_workspace_path(workspace).await?;
    let requested_path = Path::new(user_path);
    let path = workspace.join(requested_path);

    if !path_contains_parent_reference(requested_path) {
        ensure_path_in_workspace_namespace(workspace, &resolved_workspace, &path)?;

        return tokio::fs::canonicalize(&path)
            .await
            .map_err(|err| {
                format!(
                    "Failed to resolve file path (workspace: {}, requested_path: {}, candidate_path: {}): {err}",
                    workspace.display(),
                    requested_path.display(),
                    path.display()
                )
                .into()
            });
    }

    let resolved_path = tokio::fs::canonicalize(&path)
        .await
        .map_err(|err| {
            format!(
                "Failed to resolve file path (workspace: {}, requested_path: {}, candidate_path: {}): {err}",
                workspace.display(),
                requested_path.display(),
                path.display()
            )
        })?;

    ensure_path_in_workspace(&resolved_workspace, &resolved_path)?;

    Ok(resolved_path)
}

/// Resolves a write target inside the workspace, even when the destination does not yet exist.
pub async fn resolve_write_path(workspace: &Path, user_path: &str) -> Result<PathBuf, BoxError> {
    let resolved_workspace = resolve_workspace_path(workspace).await?;
    let path = workspace.join(user_path);

    match tokio::fs::symlink_metadata(&path).await {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(format!(
                    "Writing to symbolic links is not allowed (workspace: {}, path: {})",
                    workspace.display(),
                    path.display()
                )
                .into());
            }

            let resolved_path = tokio::fs::canonicalize(&path)
                .await
                .map_err(|err| {
                    format!(
                        "Failed to resolve file path (workspace: {}, requested_path: {}, candidate_path: {}): {err}",
                        workspace.display(),
                        user_path,
                        path.display()
                    )
                })?;
            ensure_path_in_workspace(&resolved_workspace, &resolved_path)?;

            Ok(resolved_path)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let (existing_ancestor, missing_components) = nearest_existing_ancestor(&path).await?;
            let resolved_ancestor = tokio::fs::canonicalize(&existing_ancestor)
                .await
                .map_err(|err| {
                    format!(
                        "Failed to resolve file path ancestor (workspace: {}, requested_path: {}, ancestor_path: {}): {err}",
                        workspace.display(),
                        user_path,
                        existing_ancestor.display()
                    )
                })?;
            ensure_path_in_workspace(&resolved_workspace, &resolved_ancestor)?;

            Ok(missing_components
                .into_iter()
                .rev()
                .fold(resolved_ancestor, |acc, component| acc.join(component)))
        }
        Err(err) => Err(format!(
            "Failed to inspect file path (workspace: {}, path: {}): {err}",
            workspace.display(),
            path.display()
        )
        .into()),
    }
}

pub(crate) async fn resolve_workspace_path(workspace: &Path) -> Result<PathBuf, BoxError> {
    tokio::fs::canonicalize(workspace).await.map_err(|err| {
        format!(
            "Failed to resolve workspace path (workspace: {}): {err}",
            workspace.display()
        )
        .into()
    })
}

pub(crate) fn ensure_path_in_workspace(
    resolved_workspace: &Path,
    resolved_path: &Path,
) -> Result<(), BoxError> {
    if !resolved_path.starts_with(resolved_workspace) {
        return Err(format!(
            "Access to paths outside the workspace is not allowed (resolved_workspace: {}, resolved_path: {})",
            resolved_workspace.display(),
            resolved_path.display()
        )
        .into());
    }

    Ok(())
}

/// Returns true when the requested path contains a parent directory traversal.
pub(crate) fn path_contains_parent_reference(path: &Path) -> bool {
    path.components()
        .any(|component| matches!(component, Component::ParentDir))
}

/// Ensures the requested path stays within the workspace namespace before following symlinks.
pub(crate) fn ensure_path_in_workspace_namespace(
    workspace: &Path,
    resolved_workspace: &Path,
    requested_path: &Path,
) -> Result<(), BoxError> {
    if requested_path.starts_with(workspace) || requested_path.starts_with(resolved_workspace) {
        return Ok(());
    }

    Err(format!(
        "Access to paths outside the workspace is not allowed (workspace: {}, resolved_workspace: {}, requested_path: {})",
        workspace.display(),
        resolved_workspace.display(),
        requested_path.display()
    )
    .into())
}

/// Returns the default encoding used for file writes.
pub(crate) fn default_write_encoding() -> String {
    UTF8_ENCODING.to_string()
}

pub(crate) fn decode_file_text(bytes: Vec<u8>) -> Result<DecodedFileText, Vec<u8>> {
    decode_file_text_with_fallback(bytes, platform_text_encoding())
}

fn decode_file_text_with_fallback(
    bytes: Vec<u8>,
    fallback_encoding: Option<&'static Encoding>,
) -> Result<DecodedFileText, Vec<u8>> {
    // Take ownership on success so valid UTF-8 content is not copied.
    let bytes = match String::from_utf8(bytes) {
        Ok(text) => {
            return Ok(DecodedFileText {
                text,
                encoding: UTF8_ENCODING.to_string(),
            });
        }
        Err(err) => err.into_bytes(),
    };

    let Some(encoding) = fallback_encoding else {
        return Err(bytes);
    };
    if encoding.name() == "UTF-8" {
        return Err(bytes);
    }

    let text = match text_from_bytes_with_encoding(&bytes, Some(encoding)) {
        Some(text) => text.into_owned(),
        None => return Err(bytes),
    };
    if !is_text_like(&text) {
        return Err(bytes);
    }

    Ok(DecodedFileText {
        text,
        encoding: text_encoding_label(encoding),
    })
}

pub(crate) fn encode_file_text(
    content: &str,
    encoding_label: &str,
) -> Result<Vec<u8>, FileTextEncodeError> {
    let encoding =
        text_encoding_for_label(encoding_label).ok_or(FileTextEncodeError::UnsupportedEncoding)?;
    let (bytes, _, had_errors) = encoding.encode(content);
    if had_errors {
        return Err(FileTextEncodeError::UnmappableCharacters);
    }
    Ok(bytes.into_owned())
}

fn is_text_like(text: &str) -> bool {
    text.chars()
        .all(|ch| matches!(ch, '\n' | '\r' | '\t' | '\u{000c}') || !ch.is_control())
}

/// Truncates `content` to at most `max_bytes`, preferring a line boundary and
/// falling back to a character boundary. Returns true when content was cut.
pub(crate) fn truncate_inline_text(content: &mut String, max_bytes: usize) -> bool {
    if content.len() <= max_bytes {
        return false;
    }

    let mut end = max_bytes;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    let cut = match content[..end].rfind('\n') {
        // Keep whole lines when possible; a single oversized line is cut at `end`.
        Some(idx) if idx > 0 => idx + 1,
        _ => end,
    };
    content.truncate(cut);
    true
}

/// Returns true when a file has multiple hard links.
///
/// Multiple links can allow path-based workspace guards to be bypassed by
/// linking a workspace path to external sensitive content.
pub(crate) fn has_multiple_hard_links(metadata: &Metadata) -> bool {
    link_count(metadata) > 1
}

pub(crate) fn ensure_regular_file(
    metadata: &Metadata,
    path: &Path,
    hard_link_error: &str,
) -> Result<(), BoxError> {
    if has_multiple_hard_links(metadata) {
        return Err(format!("{} (path: {})", hard_link_error, path.display()).into());
    }

    if !metadata.is_file() {
        return Err(format!(
            "Path does not point to a regular file (path: {})",
            path.display()
        )
        .into());
    }

    Ok(())
}

pub(crate) fn ensure_file_size_within_limit(
    metadata: &Metadata,
    path: &Path,
    max_size_bytes: u64,
) -> Result<(), BoxError> {
    if metadata.len() > max_size_bytes {
        return Err(format!(
            "File size {} exceeds maximum allowed size of {} bytes (path: {})",
            metadata.len(),
            max_size_bytes,
            path.display()
        )
        .into());
    }

    Ok(())
}

#[cfg(unix)]
fn link_count(metadata: &Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.nlink()
}

#[cfg(windows)]
fn link_count(_metadata: &Metadata) -> u64 {
    // Rust stable does not currently expose a portable, stable Windows hard-link
    // count API on `std::fs::Metadata`. Returning 1 avoids false positive blocks
    // and keeps Windows builds stable until a supported API is available.
    1
}

#[cfg(not(any(unix, windows)))]
fn link_count(_metadata: &Metadata) -> u64 {
    1
}

/// Atomically writes data to a file by first writing to a temporary file and then renaming it into place.
pub async fn atomic_write_file(
    target_path: &Path,
    data: &[u8],
    existing_permissions: Option<&Permissions>,
) -> Result<(), BoxError> {
    let temp_path =
        write_temp_file_for_atomic_replace(target_path, data, existing_permissions).await?;

    if let Err(err) = commit_atomic_replace(&temp_path, target_path).await {
        let _ = tokio::fs::remove_file(&temp_path).await;
        return Err(err);
    }

    Ok(())
}

pub(crate) async fn write_temp_file_for_atomic_replace(
    target_path: &Path,
    data: &[u8],
    existing_permissions: Option<&Permissions>,
) -> Result<PathBuf, BoxError> {
    for _ in 0..16 {
        let temp_path = atomic_temp_path(target_path)?;
        let mut file = match tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)
            .await
        {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(format!(
                    "Failed to create temporary file (target_path: {}, temp_path: {}): {err}",
                    target_path.display(),
                    temp_path.display()
                )
                .into());
            }
        };

        let write_result = async {
            file.write_all(data)
                .await
                .map_err(|err| {
                    format!(
                        "Failed to write temporary file (target_path: {}, temp_path: {}): {err}",
                        target_path.display(),
                        temp_path.display()
                    )
                })?;

            if let Some(permissions) = existing_permissions {
                tokio::fs::set_permissions(&temp_path, permissions.clone())
                    .await
                    .map_err(|err| {
                        format!(
                            "Failed to apply file permissions (target_path: {}, temp_path: {}): {err}",
                            target_path.display(),
                            temp_path.display()
                        )
                    })?;
            }

            file.sync_all()
                .await
                .map_err(|err| {
                    format!(
                        "Failed to sync temporary file (target_path: {}, temp_path: {}): {err}",
                        target_path.display(),
                        temp_path.display()
                    )
                })?;

            Ok::<(), BoxError>(())
        }
        .await;
        drop(file);

        if let Err(err) = write_result {
            let _ = tokio::fs::remove_file(&temp_path).await;
            return Err(err);
        }

        return Ok(temp_path);
    }

    Err(format!(
        "Failed to allocate unique temporary file for atomic write (target_path: {})",
        target_path.display()
    )
    .into())
}

pub(crate) async fn commit_atomic_replace(
    temp_path: &Path,
    target_path: &Path,
) -> Result<(), BoxError> {
    tokio::fs::rename(temp_path, target_path)
        .await
        .map_err(|err| {
            format!(
                "Failed to atomically replace file (temp_path: {}, target_path: {}): {err}",
                temp_path.display(),
                target_path.display()
            )
            .into()
        })
}

fn atomic_temp_path(target_path: &Path) -> Result<PathBuf, BoxError> {
    let parent = target_path.parent().ok_or_else(|| {
        format!(
            "Failed to determine parent directory for write target (target_path: {})",
            target_path.display()
        )
    })?;
    let file_name = target_path.file_name().ok_or_else(|| {
        format!(
            "Failed to determine file name for write target (target_path: {})",
            target_path.display()
        )
    })?;

    let mut temp_name = OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(".anda-tmp-{:016x}", rand::random::<u64>()));

    Ok(parent.join(temp_name))
}

/// Finds the nearest existing path component and returns the missing tail components.
pub(crate) async fn nearest_existing_ancestor(
    path: &Path,
) -> Result<(PathBuf, Vec<OsString>), BoxError> {
    let mut current = path.to_path_buf();
    let mut missing_components = Vec::new();

    loop {
        match tokio::fs::symlink_metadata(&current).await {
            Ok(_) => return Ok((current, missing_components)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let file_name = current.file_name().ok_or_else(|| {
                    format!(
                        "Access to paths outside the workspace is not allowed while resolving ancestor (requested_path: {}, current_path: {})",
                        path.display(),
                        current.display()
                    )
                })?;
                missing_components.push(file_name.to_os_string());
                current = current
                    .parent()
                    .ok_or_else(|| {
                        format!(
                            "Access to paths outside the workspace is not allowed while resolving ancestor (requested_path: {}, current_path: {})",
                            path.display(),
                            current.display()
                        )
                    })?
                    .to_path_buf();
            }
            Err(err) => {
                return Err(format!(
                    "Failed to inspect file path while resolving ancestor (requested_path: {}, current_path: {}): {err}",
                    path.display(),
                    current.display()
                )
                .into())
            }
        }
    }
}

pub(crate) fn normalize_relative_path(path: &Path) -> String {
    let value = path
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/");
    if value.is_empty() {
        ".".to_string()
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::RequestMeta;
    use serde_json::json;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("anda-fs-{name}-{:016x}", rand::random::<u64>()))
    }

    #[test]
    fn workspace_helpers_normalize_dedupe_and_report_empty_sets() {
        let first = PathBuf::from("/tmp/one");
        let second = PathBuf::from("/tmp/two");
        let third = PathBuf::from("/tmp/three");
        let fourth = PathBuf::from("/tmp/four");

        assert_eq!(
            normalize_workspaces(vec![
                PathBuf::new(),
                first.clone(),
                first.clone(),
                second.clone()
            ]),
            vec![first.clone(), second.clone()]
        );
        assert_eq!(format_workspaces(&[]), "<none>");
        assert_eq!(
            workspace_access_error("Path", "requested_path", "file.txt", &[], Vec::new())
                .to_string(),
            "Path is not accessible from any configured workspace (requested_path: file.txt, workspaces: [<none>])"
        );

        let mut meta = RequestMeta::default();
        meta.extra.insert(
            "workspace".to_string(),
            json!([first, "", second.clone(), second]),
        );
        meta.extra
            .insert("workspaces".to_string(), json!(third.clone()));

        let workspaces = tool_workspaces(&meta, &[third, fourth.clone()]);
        assert_eq!(
            workspaces,
            vec![
                PathBuf::from("/tmp/one"),
                PathBuf::from("/tmp/two"),
                PathBuf::from("/tmp/three"),
                fourth
            ]
        );
    }

    #[test]
    fn file_text_encoding_decodes_legacy_text_and_rejects_binary() {
        let gbk = vec![0xd6, 0xd0, 0xce, 0xc4, b'.', b't', b'x', b't', b'\n'];

        let decoded = decode_file_text_with_fallback(gbk.clone(), Some(encoding_rs::GBK)).unwrap();
        assert_eq!(
            decoded,
            DecodedFileText {
                text: "中文.txt\n".to_string(),
                encoding: "gbk".to_string(),
            }
        );

        let utf8 = decode_file_text_with_fallback(
            "中文.txt\n".as_bytes().to_vec(),
            Some(encoding_rs::GBK),
        )
        .unwrap();
        assert_eq!(utf8.text, "中文.txt\n");
        assert_eq!(utf8.encoding, UTF8_ENCODING);

        let binary = vec![0xff, 0x00, 0x81, 0x7f];
        assert_eq!(
            decode_file_text_with_fallback(binary.clone(), Some(encoding_rs::GBK)).unwrap_err(),
            binary
        );
    }

    #[test]
    fn file_text_encoding_encodes_legacy_text() {
        let gbk = vec![0xd6, 0xd0, 0xce, 0xc4, b'.', b't', b'x', b't', b'\n'];

        assert_eq!(encode_file_text("中文.txt\n", "gbk").unwrap(), gbk);
        assert_eq!(
            encode_file_text("hello", "utf-8").unwrap(),
            b"hello".to_vec()
        );
        assert_eq!(
            encode_file_text("hello", "not-an-encoding").unwrap_err(),
            FileTextEncodeError::UnsupportedEncoding
        );
    }

    #[test]
    fn truncate_inline_text_prefers_line_then_char_boundaries() {
        let mut text = "short".to_string();
        assert!(!truncate_inline_text(&mut text, 10));
        assert_eq!(text, "short");

        let mut text = "line one\nline two\nline three".to_string();
        assert!(truncate_inline_text(&mut text, 20));
        assert_eq!(text, "line one\nline two\n");

        // A single oversized line is cut at a char boundary instead of dropped.
        let mut text = "中文内容没有换行".to_string();
        assert!(truncate_inline_text(&mut text, 10));
        assert_eq!(text, "中文内");

        // A leading newline does not produce an empty result.
        let mut text = "\nabcdefghijklmnop".to_string();
        assert!(truncate_inline_text(&mut text, 8));
        assert_eq!(text, "\nabcdefg");
    }

    #[test]
    fn file_metadata_guards_reject_non_regular_large_and_hardlinked_files() {
        let root = temp_dir("metadata");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("file.txt");
        std::fs::write(&file, b"abcd").unwrap();

        let file_meta = std::fs::metadata(&file).unwrap();
        ensure_file_size_within_limit(&file_meta, &file, 4).unwrap();
        assert!(
            ensure_file_size_within_limit(&file_meta, &file, 3)
                .unwrap_err()
                .to_string()
                .contains("exceeds maximum")
        );

        let dir_meta = std::fs::symlink_metadata(&root).unwrap();
        assert!(
            ensure_regular_file(&dir_meta, &root, "hard links blocked")
                .unwrap_err()
                .to_string()
                .contains("Path does not point to a regular file")
                || ensure_regular_file(&dir_meta, &root, "hard links blocked")
                    .unwrap_err()
                    .to_string()
                    .contains("hard links blocked")
        );

        #[cfg(unix)]
        {
            let link = root.join("link.txt");
            std::fs::hard_link(&file, &link).unwrap();
            let linked_meta = std::fs::metadata(&file).unwrap();
            assert!(has_multiple_hard_links(&linked_meta));
            assert!(
                ensure_regular_file(&linked_meta, &file, "hard links blocked")
                    .unwrap_err()
                    .to_string()
                    .contains("hard links blocked")
            );
        }

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn resolve_helpers_cover_parent_paths_missing_tails_and_errors() {
        let root = temp_dir("resolve");
        tokio::fs::create_dir_all(root.join("dir")).await.unwrap();
        tokio::fs::write(root.join("dir/file.txt"), b"ok")
            .await
            .unwrap();

        let parent_read = resolve_read_path(&root, "dir/../dir/file.txt")
            .await
            .unwrap();
        assert_eq!(
            parent_read,
            tokio::fs::canonicalize(root.join("dir/file.txt"))
                .await
                .unwrap()
        );

        let canonical_root = tokio::fs::canonicalize(&root).await.unwrap();
        let write_path = resolve_write_path(&root, "new/nested/file.txt")
            .await
            .unwrap();
        assert_eq!(write_path, canonical_root.join("new/nested/file.txt"));

        let selected = resolve_write_path_in_workspaces(
            &[root.join("missing"), root.clone()],
            "new/nested/file.txt",
        )
        .await
        .unwrap();
        assert_eq!(selected.workspace, root);
        assert!(selected.path.ends_with("new/nested/file.txt"));

        let read_err = resolve_read_path_in_workspaces(&[], "missing.txt")
            .await
            .unwrap_err();
        assert!(read_err.to_string().contains("workspaces: [<none>]"));

        let missing_workspace = resolve_workspace_path(Path::new("/definitely/missing/anda"))
            .await
            .unwrap_err();
        assert!(
            missing_workspace
                .to_string()
                .contains("Failed to resolve workspace path")
        );

        assert!(path_contains_parent_reference(Path::new("a/../b")));
        assert!(!path_contains_parent_reference(Path::new("a/b")));
        assert!(
            ensure_path_in_workspace_namespace(
                Path::new("/tmp/work"),
                Path::new("/tmp/work"),
                Path::new("/tmp/other/file.txt"),
            )
            .unwrap_err()
            .to_string()
            .contains("outside the workspace")
        );

        let _ = tokio::fs::remove_dir_all(selected.workspace).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn atomic_write_helpers_commit_cleanup_and_path_formatting() {
        let root = temp_dir("atomic");
        tokio::fs::create_dir_all(&root).await.unwrap();
        let target = root.join("file.txt");

        atomic_write_file(&target, b"first", None).await.unwrap();
        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"first");

        let permissions = tokio::fs::metadata(&target).await.unwrap().permissions();
        atomic_write_file(&target, b"second", Some(&permissions))
            .await
            .unwrap();
        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"second");

        let temp = write_temp_file_for_atomic_replace(&target, b"third", None)
            .await
            .unwrap();
        assert!(
            temp.file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".anda-tmp-")
        );
        commit_atomic_replace(&temp, &target).await.unwrap();
        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"third");

        let missing_temp = root.join("missing-temp");
        assert!(
            commit_atomic_replace(&missing_temp, &target)
                .await
                .unwrap_err()
                .to_string()
                .contains("Failed to atomically replace file")
        );
        assert!(
            atomic_write_file(&root, b"cannot replace a directory", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("Failed to atomically replace file")
        );
        assert!(
            write_temp_file_for_atomic_replace(&root.join("missing/file.txt"), b"bad", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("Failed to create temporary file")
        );
        assert!(
            write_temp_file_for_atomic_replace(Path::new(""), b"bad", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("Failed to determine")
        );

        let (ancestor, missing) = nearest_existing_ancestor(&root.join("a/b/c.txt"))
            .await
            .unwrap();
        assert_eq!(ancestor, root);
        assert_eq!(missing.len(), 3);
        assert!(
            nearest_existing_ancestor(Path::new(""))
                .await
                .unwrap_err()
                .to_string()
                .contains("outside the workspace")
        );
        assert_eq!(normalize_relative_path(Path::new("")), ".");
        assert_eq!(normalize_relative_path(Path::new("a/b")), "a/b");
        assert_eq!(default_write_encoding(), UTF8_ENCODING);

        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn deterministic_error_branches_for_read_and_metadata_guards() {
        let root = temp_dir("fs-errors");
        tokio::fs::create_dir_all(&root).await.unwrap();

        let err = resolve_read_path(&root, "missing/../missing.txt")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Failed to resolve file path"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let link = root.join("link");
            symlink(root.join("missing-target"), &link).unwrap();
            let meta = std::fs::symlink_metadata(&link).unwrap();
            assert!(
                ensure_regular_file(&meta, &link, "hard links blocked")
                    .unwrap_err()
                    .to_string()
                    .contains("Path does not point to a regular file")
            );
        }

        let _ = tokio::fs::remove_dir_all(root).await;
    }
}
