use anda_core::BoxError;
use std::{
    ffi::OsString,
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

pub(crate) const UTF8_ENCODING: &str = "utf8";
pub(crate) const BASE64_ENCODING: &str = "base64";

/// Resolves an existing read target reachable from the workspace namespace.
pub async fn resolve_read_path(workspace: &Path, user_path: &str) -> Result<PathBuf, BoxError> {
    let resolved_workspace = resolve_workspace_path(workspace).await?;
    let requested_path = Path::new(user_path);
    let path = workspace.join(requested_path);

    if !path_contains_parent_reference(requested_path) {
        ensure_path_in_workspace_namespace(workspace, &resolved_workspace, &path)?;

        return tokio::fs::canonicalize(&path)
            .await
            .map_err(|err| format!("Failed to resolve file path: {err}").into());
    }

    let resolved_path = tokio::fs::canonicalize(&path)
        .await
        .map_err(|err| format!("Failed to resolve file path: {err}"))?;

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
                return Err("Writing to symbolic links is not allowed".into());
            }

            let resolved_path = tokio::fs::canonicalize(&path)
                .await
                .map_err(|err| format!("Failed to resolve file path: {err}"))?;
            ensure_path_in_workspace(&resolved_workspace, &resolved_path)?;

            Ok(resolved_path)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let (existing_ancestor, missing_components) = nearest_existing_ancestor(&path).await?;
            let resolved_ancestor = tokio::fs::canonicalize(&existing_ancestor)
                .await
                .map_err(|err| format!("Failed to resolve file path: {err}"))?;
            ensure_path_in_workspace(&resolved_workspace, &resolved_ancestor)?;

            Ok(missing_components
                .into_iter()
                .rev()
                .fold(resolved_ancestor, |acc, component| acc.join(component)))
        }
        Err(err) => Err(format!("Failed to inspect file path: {err}").into()),
    }
}

pub(crate) async fn resolve_workspace_path(workspace: &Path) -> Result<PathBuf, BoxError> {
    tokio::fs::canonicalize(workspace)
        .await
        .map_err(|err| format!("Failed to resolve workspace path: {err}").into())
}

pub(crate) fn ensure_path_in_workspace(
    resolved_workspace: &Path,
    resolved_path: &Path,
) -> Result<(), BoxError> {
    if !resolved_path.starts_with(resolved_workspace) {
        return Err("Access to paths outside the workspace is not allowed".into());
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

    Err("Access to paths outside the workspace is not allowed".into())
}

/// Returns the default encoding used for file writes.
pub(crate) fn default_write_encoding() -> String {
    UTF8_ENCODING.to_string()
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
    hard_link_error: &str,
) -> Result<(), BoxError> {
    if has_multiple_hard_links(metadata) {
        return Err(hard_link_error.to_string().into());
    }

    if !metadata.is_file() {
        return Err("Path does not point to a regular file".into());
    }

    Ok(())
}

pub(crate) fn ensure_file_size_within_limit(
    metadata: &Metadata,
    max_size_bytes: u64,
) -> Result<(), BoxError> {
    if metadata.len() > max_size_bytes {
        return Err(format!(
            "File size {} exceeds maximum allowed size of {} bytes",
            metadata.len(),
            max_size_bytes
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
            Err(err) => return Err(format!("Failed to create temporary file: {err}").into()),
        };

        let write_result = async {
            file.write_all(data)
                .await
                .map_err(|err| format!("Failed to write temporary file: {err}"))?;

            if let Some(permissions) = existing_permissions {
                tokio::fs::set_permissions(&temp_path, permissions.clone())
                    .await
                    .map_err(|err| format!("Failed to apply file permissions: {err}"))?;
            }

            file.sync_all()
                .await
                .map_err(|err| format!("Failed to sync temporary file: {err}"))?;

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

    Err("Failed to allocate unique temporary file for atomic write".into())
}

pub(crate) async fn commit_atomic_replace(
    temp_path: &Path,
    target_path: &Path,
) -> Result<(), BoxError> {
    tokio::fs::rename(temp_path, target_path)
        .await
        .map_err(|err| format!("Failed to atomically replace file: {err}").into())
}

fn atomic_temp_path(target_path: &Path) -> Result<PathBuf, BoxError> {
    let parent = target_path
        .parent()
        .ok_or_else(|| "Failed to determine parent directory for write target".to_string())?;
    let file_name = target_path
        .file_name()
        .ok_or_else(|| "Failed to determine file name for write target".to_string())?;

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
