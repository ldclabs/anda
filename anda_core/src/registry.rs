//! Shared mechanics for the name-keyed registries ([`ToolSet`](crate::ToolSet),
//! [`AgentSet`](crate::AgentSet)): group aggregation and name-filtered
//! selection over a lowercase-keyed map.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ToolGroup, ToolGroupInfo};

/// Aggregates per-entry group declarations into assembled [`ToolGroup`]s.
///
/// Entries that declare the same [`ToolGroupInfo::id`] are collected into one
/// group whose `members` are exactly the declaring entry names, sorted for
/// determinism. Group metadata is taken from the first entry (by lowercase name
/// order) that declares the id.
pub(crate) fn collect_groups<'a>(
    entries: impl Iterator<Item = (&'a String, Option<ToolGroupInfo>)>,
) -> Vec<ToolGroup> {
    let mut grouped: BTreeMap<String, (ToolGroupInfo, Vec<String>)> = BTreeMap::new();
    for (name, info) in entries {
        if let Some(info) = info {
            grouped
                .entry(info.id.clone())
                .or_insert_with(|| (info, Vec::new()))
                .1
                .push(name.clone());
        }
    }

    grouped
        .into_values()
        .map(|(info, mut members)| {
            members.sort();
            ToolGroup::from_info(info, members)
        })
        .collect()
}

/// Projects all entries, or the entries selected by `names`, into a vector.
///
/// Requested names are matched case-insensitively against the map's lowercase
/// keys and deduplicated, so repeated requested names do not emit duplicate
/// projections (some model providers reject duplicate schemas).
pub(crate) fn select_by_names<'a, V, T>(
    set: &'a BTreeMap<String, V>,
    names: Option<&[String]>,
    project: impl Fn(&'a V) -> T,
) -> Vec<T> {
    match names {
        None => set.values().map(project).collect(),
        Some(names) => {
            let mut seen = BTreeSet::new();
            names
                .iter()
                .filter_map(|name| {
                    let key = name.to_ascii_lowercase();
                    set.get(&key)
                        .and_then(|entry| seen.insert(key).then(|| project(entry)))
                })
                .collect()
        }
    }
}
