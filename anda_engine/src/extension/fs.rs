//! Workspace-scoped filesystem tool support.
//!
//! This module contains shared path resolution, text decoding/encoding, size
//! limits, and atomic write helpers used by the read, write, search, and edit
//! filesystem tools. Public tool structs are re-exported from the submodules.
//!
//! All workspace file tools report the capability group via [`fs_tool_group_info`],
//! so the discovery layer presents them to the model as one workspace bundle.

use anda_core::{
    BoxError, RequestMeta, ToolGroupInfo, platform_text_encoding, text_encoding_for_label,
    text_encoding_label, text_from_bytes_with_encoding,
};
use encoding_rs::Encoding;
use std::{
    ffi::OsString,
    fmt,
    fs::{Metadata, Permissions},
    path::{Component, Path, PathBuf},
};
use tokio::io::AsyncReadExt;

mod access;
mod locks;
mod patch;
mod patch_parser;
pub use patch::*;

mod edit;
mod read;
mod search;
mod write;

pub use edit::*;
pub use read::*;
pub use search::*;
pub use write::*;

/// Stable id of the filesystem workspace capability group.
pub const FS_TOOL_GROUP_ID: &str = "fs_workspace";

/// Returns the shared [`ToolGroupInfo`] for the filesystem workspace tools.
///
/// Each read, search, edit, write, and apply_patch tool reports
/// this so the registry presents them as one bundle. The registry fills in the
/// member list from the tools actually registered.
pub fn fs_tool_group_info() -> ToolGroupInfo {
    ToolGroupInfo {
        id: FS_TOOL_GROUP_ID.to_string(),
        title: "Filesystem workspace".to_string(),
        description: "Read, search, edit, and write files within the agent's sandboxed workspace directories.".to_string(),
        instructions: Some(
            "Use the registered members of this group for workspace-scoped file operations. Relative paths resolve in configured root order; request metadata can prioritize a subdirectory but does not grant access or revoke the default roots. search_file matches paths, not file contents.".to_string(),
        ),
    }
}

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

/// Sandboxed workspace roots for one tool call, in priority order.
///
/// Built once per call from the tool's configured roots plus the request's
/// untrusted narrowing hints ([`RequestMeta.extra`]), so a request can
/// prioritize a subdirectory of a configured root but never escape it. Every
/// filesystem operation the workspace tools perform goes through this scope:
/// multi-root resolution order, symlink and hard-link re-checks, size limits,
/// and error text all live behind it.
#[derive(Debug, Clone)]
pub(crate) struct WorkspaceScope {
    workspaces: Vec<PathBuf>,
}

/// An existing regular file resolved inside a workspace.
///
/// Produced by [`WorkspaceScope::open_read`] / [`WorkspaceScope::open_edit`]
/// with the regular-file, hard-link, and size-limit checks already applied.
#[derive(Debug)]
pub(crate) struct ReadTarget {
    pub(crate) workspace: PathBuf,
    pub(crate) path: PathBuf,
    pub(crate) metadata: Metadata,
    file: tokio::fs::File,
}

/// A write destination resolved inside a workspace.
///
/// Produced by [`WorkspaceScope::open_write`]; `existing` carries the current
/// file's metadata when the destination already exists (its permissions are
/// preserved by [`WriteTarget::write_atomic`]).
#[derive(Debug)]
pub(crate) struct WriteTarget {
    pub(crate) workspace: PathBuf,
    pub(crate) path: PathBuf,
    pub(crate) existing: Option<Metadata>,
}

impl WorkspaceScope {
    /// Resolves the workspace roots for one tool call from the configured
    /// defaults and the request's narrowing hints.
    pub(crate) async fn for_call(meta: &RequestMeta, defaults: &[PathBuf]) -> Self {
        Self {
            workspaces: tool_workspaces(meta, defaults).await,
        }
    }

    /// The workspace roots in priority order.
    pub(crate) fn roots(&self) -> &[PathBuf] {
        &self.workspaces
    }

    /// Consumes the scope and returns the highest-priority root, if any.
    pub(crate) fn into_primary(self) -> Option<PathBuf> {
        self.workspaces.into_iter().next()
    }

    /// Human-readable list of the roots for descriptions and error text.
    pub(crate) fn display(&self) -> String {
        format_workspaces(&self.workspaces)
    }

    /// Opens an existing file for reading.
    ///
    /// Resolves `user_path` against the roots in priority order (following the
    /// read rules: the target must exist and canonicalize inside a root), then
    /// opens it through `access::open_read`, which enforces the regular-file,
    /// hard-link, and size-limit checks on the opened handle.
    pub(crate) async fn open_read(&self, user_path: &str) -> Result<ReadTarget, BoxError> {
        let resolved = resolve_read_path_in_workspaces(&self.workspaces, user_path).await?;
        let (file, metadata) = access::open_read(&resolved.path).await?;

        Ok(ReadTarget {
            workspace: resolved.workspace,
            path: resolved.path,
            metadata,
            file,
        })
    }

    /// Opens an existing file for in-place editing.
    ///
    /// Resolves `user_path` with the write rules (symlink targets are refused)
    /// but requires the destination to already exist, then opens it with the
    /// same handle checks as [`Self::open_read`].
    pub(crate) async fn open_edit(&self, user_path: &str) -> Result<ReadTarget, BoxError> {
        let resolved = resolve_write_path_in_workspaces(&self.workspaces, user_path).await?;
        let (file, metadata) = access::open_read(&resolved.path).await.map_err(|err| {
            let missing = err
                .downcast_ref::<std::io::Error>()
                .is_some_and(|err| err.kind() == std::io::ErrorKind::NotFound);
            if missing {
                format!(
                    "Path does not point to an existing file (requested_path: {user_path}): {err}"
                )
                .into()
            } else {
                err
            }
        })?;

        Ok(ReadTarget {
            workspace: resolved.workspace,
            path: resolved.path,
            metadata,
            file,
        })
    }

    /// Opens a write destination, existing or not.
    ///
    /// Resolves `user_path` with the write rules; when the destination exists
    /// it must be a regular, singly-linked file whose permissions are carried
    /// into [`WriteTarget::write_atomic`].
    pub(crate) async fn open_write(&self, user_path: &str) -> Result<WriteTarget, BoxError> {
        let resolved = resolve_write_path_in_workspaces(&self.workspaces, user_path).await?;
        let existing = match tokio::fs::metadata(&resolved.path).await {
            Ok(metadata) => {
                ensure_regular_file(
                    &metadata,
                    &resolved.path,
                    "Writing multiply-linked files is not allowed",
                )?;
                Some(metadata)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
            Err(err) => {
                return Err(metadata_error(&resolved, user_path, err));
            }
        };

        Ok(WriteTarget {
            workspace: resolved.workspace,
            path: resolved.path,
            existing,
        })
    }
}

fn metadata_error(resolved: &ResolvedFilePath, user_path: &str, err: std::io::Error) -> BoxError {
    format!(
        "Failed to read file metadata (workspace: {}, requested_path: {}, resolved_path: {}): {err}",
        resolved.workspace.display(),
        user_path,
        resolved.path.display()
    )
    .into()
}

impl ReadTarget {
    /// Read from the validated handle, bounding actual bytes even if the file grows.
    pub(crate) async fn read_bytes(&mut self) -> Result<Vec<u8>, BoxError> {
        let mut bytes = Vec::new();
        (&mut self.file)
            .take(MAX_FILE_SIZE_BYTES + 1)
            .read_to_end(&mut bytes)
            .await?;
        if bytes.len() as u64 > MAX_FILE_SIZE_BYTES {
            return Err("File grew beyond the maximum file size of 10 MiB".into());
        }
        Ok(bytes)
    }

    /// Atomically replaces the file's content, preserving its permissions.
    pub(crate) async fn write_atomic(&self, data: &[u8]) -> Result<(), BoxError> {
        access::replace(&self.path, data, Some(&self.metadata.permissions())).await
    }
}

/// Read an absolute, normalized path through the no-follow file access layer.
/// Callers must resolve their trusted root first and validate relative components.
pub(crate) async fn read_regular_file_bounded(
    path: &Path,
    limit: u64,
) -> Result<Vec<u8>, BoxError> {
    let (file, metadata) = access::open_read(path).await?;
    ensure_file_size_within_limit(&metadata, path, limit)?;
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() as u64 > limit {
        return Err(format!(
            "File exceeds maximum size of {limit} bytes: {}",
            path.display()
        )
        .into());
    }
    Ok(bytes)
}

impl WriteTarget {
    /// Atomically writes the destination.
    ///
    /// For a new file the missing parent directories are created first and the
    /// file gets default permissions; an existing file keeps its permissions.
    pub(crate) async fn write_atomic(&self, data: &[u8]) -> Result<(), BoxError> {
        let permissions = self
            .existing
            .as_ref()
            .map(|metadata| metadata.permissions());
        access::replace(&self.path, data, permissions.as_ref()).await
    }
}

fn normalize_workspaces<I>(workspaces: I) -> Vec<PathBuf>
where
    I: IntoIterator<Item = PathBuf>,
{
    let mut normalized = Vec::new();
    for workspace in workspaces {
        push_workspace(&mut normalized, workspace);
    }

    normalized
}

/// Resolves the workspace roots for a tool call.
///
/// `RequestMeta.extra` is flattened straight off the RPC body, so a requested workspace is
/// caller-controlled and untrusted. A request may only ever *narrow* the configured roots by
/// prioritizing a subdirectory of one of them; a requested root that does not resolve inside a
/// configured root is dropped, so the configured roots always bound what the tool can reach.
async fn tool_workspaces(meta: &RequestMeta, defaults: &[PathBuf]) -> Vec<PathBuf> {
    let mut requested = Vec::new();

    if let Some(workspace) = meta.get_extra_as::<PathBuf>("workspace") {
        push_workspace(&mut requested, workspace);
    } else if let Some(extra_workspaces) = meta.get_extra_as::<Vec<PathBuf>>("workspace") {
        for workspace in extra_workspaces {
            push_workspace(&mut requested, workspace);
        }
    }

    if let Some(workspace) = meta.get_extra_as::<PathBuf>("workspaces") {
        push_workspace(&mut requested, workspace);
    } else if let Some(extra_workspaces) = meta.get_extra_as::<Vec<PathBuf>>("workspaces") {
        for workspace in extra_workspaces {
            push_workspace(&mut requested, workspace);
        }
    }

    let mut workspaces = Vec::new();
    if !requested.is_empty() {
        let resolved_defaults = resolve_workspace_paths(defaults).await;
        for workspace in requested {
            if is_within_workspaces(&workspace, &resolved_defaults).await {
                push_workspace(&mut workspaces, workspace);
            } else {
                log::warn!(
                    "ignoring requested workspace {:?} outside the configured workspaces {}",
                    workspace.display().to_string(),
                    format_workspaces(defaults),
                );
            }
        }
    }

    for workspace in defaults {
        push_workspace(&mut workspaces, workspace.clone());
    }

    workspaces
}

/// Canonicalizes each workspace, dropping the ones that cannot be resolved.
async fn resolve_workspace_paths(workspaces: &[PathBuf]) -> Vec<PathBuf> {
    let mut resolved = Vec::with_capacity(workspaces.len());
    for workspace in workspaces {
        if let Ok(path) = resolve_workspace_path(workspace).await {
            push_workspace(&mut resolved, path);
        }
    }

    resolved
}

/// Returns true when `candidate` canonicalizes inside one of the already-resolved roots.
async fn is_within_workspaces(candidate: &Path, resolved_workspaces: &[PathBuf]) -> bool {
    let Ok(resolved) = resolve_workspace_path(candidate).await else {
        return false;
    };

    resolved_workspaces
        .iter()
        .any(|root| ensure_path_in_workspace(root, &resolved).is_ok())
}

fn format_workspaces(workspaces: &[PathBuf]) -> String {
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

async fn resolve_read_path_in_workspaces(
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

async fn resolve_write_path_in_workspaces(
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

fn workspace_access_error(
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

        let resolved_path = tokio::fs::canonicalize(&path)
            .await
            .map_err(|err| -> BoxError {
                format!(
                    "Failed to resolve file path (workspace: {}, requested_path: {}, candidate_path: {}): {err}",
                    workspace.display(),
                    requested_path.display(),
                    path.display()
                )
                .into()
            })?;

        // The requested path itself stays inside the workspace, but it may pass through a
        // symbolic link that resolves outside of it. Re-check the canonicalized target so a
        // workspace-local symlink cannot be used to read arbitrary host files.
        ensure_path_in_workspace(&resolved_workspace, &resolved_path)?;

        return Ok(resolved_path);
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

async fn resolve_workspace_path(workspace: &Path) -> Result<PathBuf, BoxError> {
    tokio::fs::canonicalize(workspace).await.map_err(|err| {
        format!(
            "Failed to resolve workspace path (workspace: {}): {err}",
            workspace.display()
        )
        .into()
    })
}

fn ensure_path_in_workspace(
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
fn path_contains_parent_reference(path: &Path) -> bool {
    path.components()
        .any(|component| matches!(component, Component::ParentDir))
}

/// Ensures the requested path stays within the workspace namespace before following symlinks.
fn ensure_path_in_workspace_namespace(
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

/// Truncates `content` to at most `max_bytes`, preferring a line boundary and falling back to a
/// grapheme-cluster boundary so a multibyte character or emoji cluster is never split. Returns true
/// when content was cut.
pub(crate) fn truncate_inline_text(content: &mut String, max_bytes: usize) -> bool {
    if content.len() <= max_bytes {
        return false;
    }

    // Grapheme-cluster-safe byte cutoff within the budget (shared with truncate_utf8_to_max_bytes).
    let end = crate::grapheme_safe_cutoff(content, max_bytes);
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
    // Metadata alone has no stable link-count accessor on Windows. Workspace
    // reads and mutations additionally validate the opened handle's link count
    // in access.rs; metadata-only callers must not rely on this fallback.
    1
}

#[cfg(not(any(unix, windows)))]
fn link_count(_metadata: &Metadata) -> u64 {
    1
}

/// Atomically writes data to a file by first writing to a temporary file and then renaming it into place.
/// Relative paths, including a bare filename, resolve against the current directory.
pub async fn atomic_write_file(
    target_path: &Path,
    data: &[u8],
    existing_permissions: Option<&Permissions>,
) -> Result<(), BoxError> {
    let parent = target_path.parent().ok_or("Missing parent directory")?;
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let parent = tokio::fs::canonicalize(parent).await?;
    let name = target_path.file_name().ok_or("Missing file name")?;
    access::replace(&parent.join(name), data, existing_permissions)
        .await
        .map_err(|err| format!("Failed to atomically replace file: {err}").into())
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
    fn fs_tools_form_one_capability_group() {
        use crate::context::BaseCtx;
        use anda_core::ToolSet;
        use std::sync::Arc;

        let workspace = PathBuf::from("/tmp/anda-fs-group");
        let mut tools = ToolSet::<BaseCtx>::new();
        tools
            .add(Arc::new(ReadFileTool::new(workspace.clone())))
            .unwrap();
        tools
            .add(Arc::new(WriteFileTool::new(workspace.clone())))
            .unwrap();
        tools
            .add(Arc::new(EditFileTool::new(workspace.clone())))
            .unwrap();
        tools.add(Arc::new(SearchFileTool::new(workspace))).unwrap();

        let groups = tools.groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, FS_TOOL_GROUP_ID);
        // All four registered tools land in the group, sorted by name.
        assert_eq!(
            groups[0].members,
            vec![
                "edit_file".to_string(),
                "read_file".to_string(),
                "search_file".to_string(),
                "write_file".to_string(),
            ]
        );
        assert!(groups[0].instructions.is_some());
    }

    #[test]
    fn workspace_helpers_normalize_dedupe_and_report_empty_sets() {
        let first = PathBuf::from("/tmp/one");
        let second = PathBuf::from("/tmp/two");

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
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tool_workspaces_only_accepts_requests_inside_configured_roots() {
        let root = temp_dir("tool_workspaces");
        let configured = root.join("configured");
        let nested = configured.join("nested");
        let outside = root.join("outside");
        tokio::fs::create_dir_all(&nested).await.unwrap();
        tokio::fs::create_dir_all(&outside).await.unwrap();

        // No request metadata: the configured roots are used as-is.
        assert_eq!(
            tool_workspaces(&RequestMeta::default(), std::slice::from_ref(&configured)).await,
            vec![configured.clone()]
        );

        // A request may narrow to a subdirectory of a configured root, which then takes priority.
        let mut meta = RequestMeta::default();
        meta.extra
            .insert("workspace".to_string(), json!([nested, "", nested]));
        assert_eq!(
            tool_workspaces(&meta, std::slice::from_ref(&configured)).await,
            vec![nested.clone(), configured.clone()]
        );

        // A request outside the configured roots is dropped, not prioritized.
        let mut meta = RequestMeta::default();
        meta.extra.insert("workspace".to_string(), json!(outside));
        assert_eq!(
            tool_workspaces(&meta, std::slice::from_ref(&configured)).await,
            vec![configured.clone()]
        );

        // Filesystem root, the widest possible escape, is likewise rejected.
        let mut meta = RequestMeta::default();
        meta.extra.insert("workspaces".to_string(), json!("/"));
        assert_eq!(
            tool_workspaces(&meta, std::slice::from_ref(&configured)).await,
            vec![configured.clone()]
        );

        // A path that does not exist cannot be resolved and is dropped.
        let mut meta = RequestMeta::default();
        meta.extra.insert(
            "workspace".to_string(),
            json!(configured.join("does-not-exist")),
        );
        assert_eq!(
            tool_workspaces(&meta, std::slice::from_ref(&configured)).await,
            vec![configured.clone()]
        );

        let _ = tokio::fs::remove_dir_all(&root).await;
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

        // A single oversized line is cut at a grapheme boundary instead of dropped.
        let mut text = "中文内容没有换行".to_string();
        assert!(truncate_inline_text(&mut text, 10));
        assert_eq!(text, "中文内");

        // A leading newline does not produce an empty result.
        let mut text = "\nabcdefghijklmnop".to_string();
        assert!(truncate_inline_text(&mut text, 8));
        assert_eq!(text, "\nabcdefg");

        // A multi-codepoint grapheme cluster (family emoji joined by ZWJ, 25 bytes) on a single
        // oversized line is never split: a budget landing mid-cluster backs off to the previous
        // cluster boundary.
        let family = "👨‍👩‍👧‍👦";
        let mut text = family.repeat(3); // 75 bytes, no newline
        assert!(truncate_inline_text(&mut text, 60));
        assert_eq!(text, family.repeat(2));
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

    #[tokio::test]
    async fn atomic_write_accepts_a_bare_relative_filename() {
        // Keep the process-wide current directory unchanged while testing the public API.
        let target = PathBuf::from(format!(
            ".anda-atomic-relative-{:032x}",
            rand::random::<u128>()
        ));
        atomic_write_file(&target, b"first", None).await.unwrap();
        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"first");
        atomic_write_file(&target, b"second", None).await.unwrap();
        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"second");
        tokio::fs::remove_file(target).await.unwrap();
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

        assert!(
            atomic_write_file(&root, b"cannot replace a directory", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("Failed to atomically replace file")
        );
        assert!(
            atomic_write_file(&root.join("missing/file.txt"), b"bad", None)
                .await
                .is_err()
        );
        assert!(
            atomic_write_file(Path::new(""), b"bad", None)
                .await
                .is_err()
        );
        // Failed writes leave no temporary files behind.
        let mut entries = tokio::fs::read_dir(&root).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            assert_eq!(entry.file_name(), "file.txt");
        }

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
