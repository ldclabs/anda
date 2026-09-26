//! Bounded filesystem discovery and shared skill/package validation.

use super::catalog::{Entry, digest};
use super::{Skill, SkillLimits, SkillLoadReport, SkillMetadata, parse_skill_md};
use anda_core::BoxError;
use std::{
    collections::VecDeque,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

pub(super) const MAX_SKILL_FILE_BYTES: u64 = 512 * 1024;
pub(super) const MAX_RESOURCE_BYTES: u64 = 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 32 * 1024;

pub(super) struct Scan {
    pub files: Vec<PathBuf>,
    pub report: SkillLoadReport,
}

pub(super) async fn scan_files(root: &Path, limits: &SkillLimits) -> Scan {
    let mut scan = Scan {
        files: Vec::new(),
        report: SkillLoadReport::default(),
    };
    let mut pending = VecDeque::from([(root.to_path_buf(), 0)]);
    let mut directories = 1;
    let mut entries_seen = 0;
    while let Some((directory, depth)) = pending.pop_front() {
        let mut reader = match tokio::fs::read_dir(&directory).await {
            Ok(reader) => reader,
            Err(error) if directory == root && error.kind() == std::io::ErrorKind::NotFound => {
                continue;
            }
            Err(error) => {
                scan.report.note("io", directory, error);
                continue;
            }
        };
        let mut entries = Vec::new();
        loop {
            let entry = match reader.next_entry().await {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(error) => {
                    scan.report.note("io", directory.clone(), error);
                    break;
                }
            };
            entries_seen += 1;
            if entries_seen > limits.max_entries {
                scan.report.truncated = true;
                scan.report
                    .note("limit", root.into(), "Directory entry limit reached");
                return scan;
            }
            entries.push(entry);
        }
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let kind = match entry.file_type().await {
                Ok(kind) => kind,
                Err(error) => {
                    scan.report.note("io", path, error);
                    continue;
                }
            };
            if kind.is_dir() {
                if entry.file_name().to_string_lossy().starts_with('.') {
                    continue;
                }
                if depth >= limits.max_depth || directories >= limits.max_directories {
                    scan.report.truncated = true;
                    scan.report
                        .note("limit", path, "Directory depth or count limit reached");
                } else {
                    directories += 1;
                    pending.push_back((path, depth + 1));
                }
            } else if entry.file_name() == "SKILL.md" {
                if kind.is_file() {
                    if scan.files.len() == limits.max_skills {
                        scan.report.truncated = true;
                        scan.report
                            .note("limit", root.into(), "Skill count limit reached");
                        return scan;
                    }
                    scan.files.push(path);
                } else {
                    scan.report.rejected += 1;
                    scan.report.note(
                        "invalid",
                        path,
                        "SKILL.md must be a regular, non-symlink file",
                    );
                }
            }
        }
    }
    scan.files.sort();
    scan
}

/// Relative resources use portable slash-separated components, never ambient paths.
pub(super) fn validate_resource(resource: &str) -> Result<(), BoxError> {
    if resource.is_empty()
        || resource.len() > 1_024
        || resource.contains(['\\', '\0', ':'])
        || resource
            .split('/')
            .any(|part| matches!(part, "" | "." | ".."))
        || Path::new(resource)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(
            "Skill resource must be a relative package path without parent components".into(),
        );
    }
    Ok(())
}

pub(super) async fn read_text(
    root: &Path,
    relative: &Path,
    limit: u64,
) -> Result<String, BoxError> {
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("Skill path must remain inside its configured root".into());
    }
    // The loader already resolved the trusted root. Do not canonicalize again: a directory
    // swapped for a symlink after discovery must fail the no-follow handle walk.
    let bytes =
        crate::extension::fs::read_regular_file_bounded(&root.join(relative), limit).await?;
    let content = super::types::decode_skill_md_bytes(bytes).map_err(|_| -> BoxError {
        "Only UTF-8 or supported text-encoded skill files are supported".into()
    })?;
    if content.len() as u64 > limit {
        return Err(format!("Decoded skill file exceeds maximum size of {limit} bytes").into());
    }
    Ok(content)
}

pub(super) async fn read_entry(
    root: &Path,
    path: &Path,
    rank: usize,
) -> Result<(Entry, String, usize), BoxError> {
    let relative = path.strip_prefix(root)?;
    let canonical_root = tokio::fs::canonicalize(root).await?;
    let content = read_text(&canonical_root, relative, MAX_SKILL_FILE_BYTES).await?;
    let base_dir = path.parent().ok_or("Skill has no parent directory")?;
    let skill = parse_skill_md(base_dir.to_path_buf(), &content)?;
    let sidecar = base_dir.join("agents/openai.yaml");
    let mut metadata = frontmatter_metadata(&skill)?;
    let mut size = content.len();
    match tokio::fs::symlink_metadata(canonical_root.join(sidecar.strip_prefix(root)?)).await {
        Ok(_) => {
            let contents = read_text(
                &canonical_root,
                sidecar.strip_prefix(root)?,
                MAX_METADATA_BYTES,
            )
            .await?;
            size += contents.len();
            // Parse failures reject this skill rather than silently removing an explicit-only policy.
            let overlay: serde_json::Value = serde_saphyr::from_str(&contents)?;
            let mut base = serde_json::to_value(&metadata)?;
            if let (Some(base), Some(overlay)) = (base.as_object_mut(), overlay.as_object()) {
                for (key, value) in overlay {
                    base.insert(key.clone(), value.clone());
                }
            } else {
                return Err("Skill metadata must be a mapping".into());
            }
            metadata = serde_json::from_value(base)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    validate_metadata(&metadata)?;
    let identity = canonical_root.join(relative);
    let fingerprint = digest(&serde_json::to_vec(&(&content, &metadata))?);
    Ok((
        Entry {
            id: format!(
                "skill://{}",
                digest(identity.as_os_str().as_encoded_bytes())
            ),
            source_id: format!(
                "root:{}",
                digest(canonical_root.as_os_str().as_encoded_bytes())
            ),
            rank,
            root: root.to_path_buf(),
            canonical_root,
            skill: Arc::new(skill),
            metadata,
            fingerprint,
        },
        content,
        size,
    ))
}

fn frontmatter_metadata(skill: &Skill) -> Result<SkillMetadata, BoxError> {
    let mut value = serde_json::Map::new();
    for key in ["interface", "policy", "dependencies"] {
        if let Some(v) = skill.frontmatter.metadata.get(key) {
            value.insert(key.into(), v.clone());
        }
    }
    let mut metadata: SkillMetadata = serde_json::from_value(value.into())?;
    if metadata.interface.short_description.is_none() {
        metadata.interface.short_description = skill
            .frontmatter
            .metadata
            .get("short-description")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
    }
    Ok(metadata)
}
fn validate_metadata(metadata: &SkillMetadata) -> Result<(), BoxError> {
    let fields = [
        &metadata.interface.display_name,
        &metadata.interface.short_description,
        &metadata.interface.default_prompt,
    ];
    if fields
        .into_iter()
        .flatten()
        .any(|value| value.chars().count() > 1_024)
        || metadata.dependencies.tools.len() > 32
        || metadata.dependencies.tools.iter().any(|tool| {
            tool.kind.trim().is_empty()
                || tool.kind.len() > 64
                || tool.value.trim().is_empty()
                || tool.value.len() > 256
                || tool
                    .description
                    .as_ref()
                    .is_some_and(|text| text.chars().count() > 1_024)
        })
    {
        return Err("Skill presentation or dependency metadata exceeds its limits".into());
    }
    Ok(())
}
