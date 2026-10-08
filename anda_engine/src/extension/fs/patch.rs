//! Prevalidated workspace patches with exact matching and explicit partial results.

use super::{
    patch_parser::{self, Action},
    *,
};
use crate::{
    context::BaseCtx,
    extension::shell::preview_output,
    extension::{hooked_call, tool_definition},
};
use anda_core::{FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Optional precondition for a patch source or existing destination.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct FileVersion {
    /// Workspace path, as it appears in the patch.
    pub path: String,
    /// SHA-256 of the original encoded bytes, returned by dry_run. Use "missing" for a new file.
    pub sha256: String,
}

/// JSON wrapper keeps patch support available to all function-calling providers.
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct ApplyPatchArgs {
    /// Text between *** Begin Patch and *** End Patch. Supports Add File, Delete File, Update File and Move to; updates use @@ chunks with exact space, '+' or '-' line prefixes (a blank line is empty context). Ambiguous context is rejected.
    pub patch: String,
    /// Validate and return the proposed diff and source versions without writing.
    #[serde(default)]
    pub dry_run: bool,
    /// Optional original byte hashes. Every supplied path must belong to this patch.
    #[serde(default)]
    pub expected_versions: Vec<FileVersion>,
}

/// A planned or committed file change.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PatchChange {
    /// Original path (or the newly created path).
    pub path: String,
    /// Destination of a move, if any.
    pub destination: Option<String>,
    /// Original byte hash; "missing" means the file did not exist.
    pub original_sha256: String,
    /// Updated byte hash; "missing" means the file is deleted.
    pub sha256: String,
    /// True only after the whole operation completed. A failed move can leave its destination behind.
    pub applied: bool,
}

/// Patch result. Atomicity is per file, not across the whole patch.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ApplyPatchOutput {
    /// Planned changes with explicit commit status.
    pub changes: Vec<PatchChange>,
    /// Bounded diff preview.
    pub diff: String,
    /// Whether the diff preview was truncated.
    pub diff_truncated: bool,
    /// Failure during commit, after successful prevalidation. Earlier writes may remain.
    pub error: Option<String>,
}

/// Dedicated coding edit tool. Reads and searches can remain shell operations.
#[derive(Clone)]
pub struct ApplyPatchTool {
    workspaces: Vec<PathBuf>,
}

impl ApplyPatchTool {
    /// Creates a patch tool confined to one workspace.
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_workspaces([workspace])
    }
    /// Creates a patch tool confined to the supplied roots in priority order.
    pub fn with_workspaces(workspaces: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            workspaces: normalize_workspaces(workspaces),
        }
    }
}

struct Prepared {
    path: PathBuf,
    destination: Option<PathBuf>,
    original: Option<Vec<u8>>,
    updated: Option<Vec<u8>>,
    permissions: Option<Permissions>,
    change: PatchChange,
    diff: String,
}

fn hash(bytes: Option<&[u8]>) -> String {
    bytes.map_or_else(
        || "missing".into(),
        |bytes| hex::encode(Sha256::digest(bytes)),
    )
}

impl Tool<BaseCtx> for ApplyPatchTool {
    type Args = ApplyPatchArgs;
    type Output = ApplyPatchOutput;
    fn name(&self) -> String {
        "apply_patch".into()
    }
    fn description(&self) -> String {
        "Apply a workspace-scoped multi-file patch with exact context matching, encoding preservation and per-file atomic writes. Add and move never overwrite existing files. Prevalidates every change, supports dry_run and expected SHA-256 versions. Commit failures can leave earlier changes applied.".into()
    }
    fn group(&self) -> Option<ToolGroupInfo> {
        Some(fs_tool_group_info())
    }
    fn definition(&self) -> FunctionDefinition {
        tool_definition::<ApplyPatchArgs>(self.name(), self.description())
    }
    async fn call(
        &self,
        ctx: BaseCtx,
        args: ApplyPatchArgs,
        _: Vec<Resource>,
    ) -> Result<ToolOutput<ApplyPatchOutput>, BoxError> {
        let ctx = &ctx;
        hooked_call(ctx, args, |args| async move {
            let patches = patch_parser::parse(&args.patch)?;
            let scope = WorkspaceScope::for_call(ctx.meta(), &self.workspaces).await;
            let mut paths = Vec::new();
            for patch in &patches {
                paths.push(
                    resolve_write_path_in_workspaces(scope.roots(), &patch.path)
                        .await?
                        .path,
                );
                if let Action::Update {
                    destination: Some(path),
                    ..
                } = &patch.action
                {
                    paths.push(
                        resolve_write_path_in_workspaces(scope.roots(), path)
                            .await?
                            .path,
                    );
                }
            }
            let distinct = paths.iter().collect::<BTreeSet<_>>();
            if distinct.len() != paths.len() {
                return Err(
                    "Each file may occur only once in a patch (including move destinations)".into(),
                );
            }
            let cancellation = ctx.cancellation_token();
            let _guards =
                locks::lock_paths(paths.iter().map(PathBuf::as_path), &cancellation).await?;
            let mut prepared = Vec::new();
            let mut total_bytes = 0usize;
            for patch in patches {
                let target = scope.open_write(&patch.path).await?;
                let permissions = target.existing.as_ref().map(Metadata::permissions);
                let original = if target.existing.is_some() {
                    Some(scope.open_edit(&patch.path).await?.read_bytes().await?)
                } else {
                    None
                };
                let original_text = match (&patch.action, &original) {
                    (Action::Update { .. }, Some(bytes)) => Some(
                        decode_file_text(bytes.clone())
                            .map_err(|_| "Patch requires a supported text encoding")?,
                    ),
                    // Delete needs no text (and Add rejects existing files below), so
                    // undecodable content only changes the preview.
                    (_, Some(bytes)) => decode_file_text(bytes.clone()).ok(),
                    (_, None) => None,
                };
                let before = original_text
                    .as_ref()
                    .map_or("", |decoded| decoded.text.as_str());
                let encoding = original_text
                    .as_ref()
                    .map_or(UTF8_ENCODING, |decoded| decoded.encoding.as_str());
                let (destination_name, after) = match patch.action {
                    Action::Add(content) => {
                        if original.is_some() {
                            return Err("Add File refuses to overwrite an existing file".into());
                        }
                        (None, Some(content))
                    }
                    Action::Delete => {
                        if original.is_none() {
                            return Err("Delete File requires an existing file".into());
                        }
                        (None, None)
                    }
                    Action::Update {
                        destination,
                        chunks,
                    } => {
                        if original.is_none() {
                            return Err("Update File requires an existing file".into());
                        }
                        (destination, Some(patch_parser::update(before, &chunks)?))
                    }
                };
                let destination = match &destination_name {
                    Some(path) => {
                        let target = scope.open_write(path).await?;
                        if target.existing.is_some() {
                            return Err("Move refuses to overwrite an existing file".into());
                        }
                        Some(target.path)
                    }
                    None => None,
                };
                let updated = after
                    .as_ref()
                    .map(|text| encode_file_text(text, encoding))
                    .transpose()?;
                if updated
                    .as_ref()
                    .is_some_and(|bytes| bytes.len() as u64 > MAX_FILE_SIZE_BYTES)
                {
                    return Err("Patch result exceeds maximum file size".into());
                }
                total_bytes = total_bytes
                    .saturating_add(original.as_ref().map_or(0, Vec::len))
                    .saturating_add(updated.as_ref().map_or(0, Vec::len));
                if total_bytes > 64 * 1024 * 1024 {
                    return Err("Patch exceeds 64 MiB working-set budget".into());
                }
                let change = PatchChange {
                    path: patch.path,
                    destination: destination_name,
                    original_sha256: hash(original.as_deref()),
                    sha256: hash(updated.as_deref()),
                    applied: false,
                };
                let diff = if original.is_some() && original_text.is_none() {
                    format!("--- {0}\n+++ {0}\nBinary file deleted\n", change.path)
                } else {
                    diff(
                        &change.path,
                        change.destination.as_deref(),
                        before,
                        after.as_deref().unwrap_or(""),
                    )
                };
                prepared.push(Prepared {
                    path: target.path,
                    destination,
                    original,
                    updated,
                    permissions,
                    change,
                    diff,
                });
            }
            for expected in &args.expected_versions {
                let actual = prepared
                    .iter()
                    .find_map(|entry| {
                        if entry.change.path == expected.path {
                            Some(entry.change.original_sha256.as_str())
                        } else if entry.change.destination.as_deref()
                            == Some(expected.path.as_str())
                        {
                            Some("missing")
                        } else {
                            None
                        }
                    })
                    .ok_or("Version precondition path is not in the patch")?;
                if actual != expected.sha256 {
                    return Err(format!("Stale file version: {}", expected.path).into());
                }
            }
            let preview = preview_output(
                &prepared
                    .iter()
                    .map(|entry| entry.diff.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                32 * 1024,
            );
            let mut result = ApplyPatchOutput {
                diff: preview.text,
                diff_truncated: preview.omitted_bytes > 0,
                ..Default::default()
            };
            for mut entry in prepared {
                if !args.dry_run && result.error.is_none() {
                    let outcome = commit(&scope, &entry, &cancellation).await;
                    match outcome {
                        Ok(()) => entry.change.applied = true,
                        Err(error) => {
                            result.error = Some(format!("{}: {error}", entry.change.path))
                        }
                    }
                }
                result.changes.push(entry.change);
            }
            Ok(ToolOutput {
                is_error: result.error.is_some().then_some(true),
                ..ToolOutput::new(result)
            })
        })
        .await
    }
}

async fn commit(
    scope: &WorkspaceScope,
    entry: &Prepared,
    cancellation: &anda_core::CancellationToken,
) -> Result<(), BoxError> {
    if cancellation.is_cancelled() {
        return Err("Patch cancelled".into());
    }
    // Detect external changes before committing. This is optimistic validation,
    // not an atomic compare-and-swap against arbitrary external writers.
    let current = scope.open_write(&entry.change.path).await?;
    if current.path != entry.path {
        return Err("Resolved path changed during patch".into());
    }
    let bytes = if current.existing.is_some() {
        Some(
            scope
                .open_edit(&entry.change.path)
                .await?
                .read_bytes()
                .await?,
        )
    } else {
        None
    };
    if bytes != entry.original {
        return Err("File changed after patch preparation".into());
    }
    if let Some(destination) = &entry.destination {
        let target = scope
            .open_write(entry.change.destination.as_deref().unwrap_or_default())
            .await?;
        if target.path != *destination || target.existing.is_some() {
            return Err("Move destination changed after preparation".into());
        }
    }
    match &entry.updated {
        Some(bytes) => {
            let path = entry.destination.as_deref().unwrap_or(&entry.path);
            if entry.destination.is_some() || entry.original.is_none() {
                access::create(path, bytes, entry.permissions.as_ref()).await?;
            } else {
                access::replace(path, bytes, entry.permissions.as_ref()).await?;
            }
            if entry.destination.is_some() {
                access::remove(&entry.path).await.map_err(|error| {
                    format!("Move destination was written, but removing the source failed: {error}")
                })?;
            }
        }
        None => access::remove(&entry.path).await?,
    }
    Ok(())
}

fn diff(path: &str, destination: Option<&str>, before: &str, after: &str) -> String {
    let old = before.lines().collect::<Vec<_>>();
    let new = after.lines().collect::<Vec<_>>();
    let common = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let suffix = old[common..]
        .iter()
        .rev()
        .zip(new[common..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = format!(
        "--- {path}\n+++ {}\n@@ -{},{} +{},{} @@\n",
        destination.unwrap_or(path),
        common + 1,
        old.len() - common - suffix,
        common + 1,
        new.len() - common - suffix
    );
    for (prefix, lines) in [
        ('-', &old[common..old.len() - suffix]),
        ('+', &new[common..new.len() - suffix]),
    ] {
        for line in lines {
            out.push(prefix);
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}
