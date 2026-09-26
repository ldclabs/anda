//! Strict, bounded parser for the familiar apply_patch text format.

use anda_core::BoxError;

#[derive(Debug)]
pub(super) struct Patch {
    pub path: String,
    pub action: Action,
}
#[derive(Debug)]
pub(super) enum Action {
    Add(String),
    Delete,
    Update {
        destination: Option<String>,
        chunks: Vec<Chunk>,
    },
}
#[derive(Debug)]
pub(super) struct Chunk {
    pub context: Option<String>,
    pub old: Vec<String>,
    pub new: Vec<String>,
    pub eof: bool,
}

pub(super) fn parse(text: &str) -> Result<Vec<Patch>, BoxError> {
    if text.len() > 1024 * 1024 {
        return Err("Patch exceeds 1 MiB".into());
    }
    // Whitespace around the envelope carries no meaning; lines inside it stay exact.
    let lines = text.trim().lines().collect::<Vec<_>>();
    if lines.first() != Some(&"*** Begin Patch") || lines.last() != Some(&"*** End Patch") {
        return Err("Expected *** Begin Patch and *** End Patch markers".into());
    }
    let mut index = 1;
    let mut patches = Vec::new();
    while index + 1 < lines.len() {
        let header = lines[index];
        index += 1;
        let (path, action) = if let Some(path) = header.strip_prefix("*** Add File: ") {
            let mut content = String::new();
            while index + 1 < lines.len() && !lines[index].starts_with("*** ") {
                content.push_str(
                    lines[index]
                        .strip_prefix('+')
                        .ok_or("Added lines must start with +")?,
                );
                content.push('\n');
                index += 1;
            }
            (path, Action::Add(content))
        } else if let Some(path) = header.strip_prefix("*** Delete File: ") {
            (path, Action::Delete)
        } else if let Some(path) = header.strip_prefix("*** Update File: ") {
            let destination = lines
                .get(index)
                .and_then(|line| line.strip_prefix("*** Move to: "))
                .map(str::to_owned);
            if destination.is_some() {
                index += 1;
            }
            let mut chunks = Vec::new();
            while index + 1 < lines.len() && lines[index].starts_with("@@") {
                let header = lines[index];
                index += 1;
                let context = if header == "@@" {
                    None
                } else {
                    Some(
                        header
                            .strip_prefix("@@ ")
                            .ok_or("Invalid patch context header")?
                            .to_owned(),
                    )
                };
                let mut chunk = Chunk {
                    context,
                    old: Vec::new(),
                    new: Vec::new(),
                    eof: false,
                };
                while index + 1 < lines.len()
                    && !lines[index].starts_with("@@")
                    && !lines[index].starts_with("*** ")
                {
                    let line = lines[index];
                    index += 1;
                    if line.is_empty() {
                        // An empty context line whose leading space was stripped.
                        chunk.old.push(String::new());
                        chunk.new.push(String::new());
                    } else if let Some(line) = line.strip_prefix(' ') {
                        chunk.old.push(line.to_owned());
                        chunk.new.push(line.to_owned());
                    } else if let Some(line) = line.strip_prefix('-') {
                        chunk.old.push(line.to_owned());
                    } else if let Some(line) = line.strip_prefix('+') {
                        chunk.new.push(line.to_owned());
                    } else {
                        return Err("Patch lines must start with a space, + or -".into());
                    }
                }
                if lines.get(index) == Some(&"*** End of File") {
                    chunk.eof = true;
                    index += 1;
                }
                if chunk.old.is_empty() && chunk.new.is_empty() {
                    return Err("Empty update chunk".into());
                }
                chunks.push(chunk);
                if chunks.len() > 256 {
                    return Err("Too many update chunks".into());
                }
            }
            if chunks.is_empty() {
                return Err("Update requires at least one @@ chunk".into());
            }
            (
                path,
                Action::Update {
                    destination,
                    chunks,
                },
            )
        } else {
            return Err(format!("Invalid patch header: {header}").into());
        };
        if path.is_empty() || path.len() > 4096 || path.contains('\0') {
            return Err("Invalid patch path".into());
        }
        patches.push(Patch {
            path: path.to_owned(),
            action,
        });
        if patches.len() > 32 {
            return Err("A patch may touch at most 32 source files".into());
        }
    }
    if patches.is_empty() {
        return Err("Patch contains no file changes".into());
    }
    Ok(patches)
}

/// Match exact line content, ignoring only line terminators. Ambiguity is an error.
pub(super) fn update(text: &str, chunks: &[Chunk]) -> Result<String, BoxError> {
    let raw = text.split_inclusive('\n').take(200_001).collect::<Vec<_>>();
    if raw.len() > 200_000 {
        return Err("Patch file exceeds the 200000-line budget".into());
    }
    let lines = raw
        .iter()
        .map(|line| {
            line.strip_suffix('\n')
                .unwrap_or(line)
                .strip_suffix('\r')
                .unwrap_or(line.strip_suffix('\n').unwrap_or(line))
        })
        .collect::<Vec<_>>();
    let mut replacements = Vec::new();
    let mut cursor = 0;
    let mut projected = text.len();
    for chunk in chunks {
        if let Some(context) = &chunk.context {
            let found = (cursor..lines.len())
                .filter(|&i| lines[i] == context)
                .collect::<Vec<_>>();
            if found.len() != 1 {
                return Err("Patch context is missing or ambiguous".into());
            }
            cursor = found[0] + 1;
        }
        let index = if chunk.old.is_empty() {
            lines.len()
        } else {
            if chunk.old.len() > lines.len() {
                return Err("Patch context was not found".into());
            }
            let found = (cursor..=lines.len() - chunk.old.len())
                .filter(|&start| {
                    (!chunk.eof || start + chunk.old.len() == lines.len())
                        && lines[start..start + chunk.old.len()]
                            .iter()
                            .zip(&chunk.old)
                            .all(|(a, b)| *a == b)
                })
                .take(2)
                .collect::<Vec<_>>();
            if found.len() != 1 {
                return Err(
                    "Patch context is missing or ambiguous; include more exact surrounding lines"
                        .into(),
                );
            }
            found[0]
        };
        let ending = raw
            .get(index)
            .or_else(|| raw.last())
            .filter(|line| line.ends_with("\r\n"))
            .map_or("\n", |_| "\r\n");
        let mut replacement = chunk.new.join(ending);
        if !chunk.new.is_empty()
            && (index + chunk.old.len() < lines.len()
                || text.ends_with('\n')
                || chunk.old.is_empty())
        {
            replacement.push_str(ending);
        }
        if chunk.old.is_empty() && !text.is_empty() && !text.ends_with('\n') {
            replacement.insert_str(0, ending);
        }
        let removed: usize = raw[index..index + chunk.old.len()]
            .iter()
            .map(|line| line.len())
            .sum();
        projected = projected
            .saturating_sub(removed)
            .saturating_add(replacement.len());
        if projected as u64 > super::MAX_FILE_SIZE_BYTES {
            return Err("Patch result exceeds maximum file size".into());
        }
        replacements.push((index, chunk.old.len(), replacement));
        cursor = index + chunk.old.len();
    }
    let mut out = String::with_capacity(projected);
    let mut cursor = 0;
    for (index, count, replacement) in replacements {
        for line in &raw[cursor..index] {
            out.push_str(line);
        }
        out.push_str(&replacement);
        cursor = index + count;
    }
    for line in &raw[cursor..] {
        out.push_str(line);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chunks(patch: &str) -> Vec<Chunk> {
        let mut parsed = parse(&format!(
            "*** Begin Patch\n*** Update File: file\n{patch}\n*** End Patch"
        ))
        .unwrap();
        match parsed.remove(0).action {
            Action::Update { chunks, .. } => chunks,
            _ => unreachable!(),
        }
    }

    #[test]
    fn updates_preserve_missing_newline_and_match_eof_exactly() {
        assert_eq!(
            update("before", &chunks("@@\n-before\n+after")).unwrap(),
            "after"
        );
        assert_eq!(
            update(
                "same\nother\nsame\n",
                &chunks("@@\n-same\n+last\n*** End of File")
            )
            .unwrap(),
            "same\nother\nlast\n"
        );
        assert!(update("    indented\n", &chunks("@@\n-indented\n+wrong")).is_err());
        assert_eq!(
            update("a\r\nb\r\n", &chunks("@@\n a\n-b\n+c")).unwrap(),
            "a\r\nc\r\n"
        );
    }

    #[test]
    fn blank_context_lines_and_envelope_whitespace_are_accepted() {
        assert_eq!(
            update("a\n\nb\n", &chunks("@@\n a\n\n-b\n+c")).unwrap(),
            "a\n\nc\n"
        );
        assert!(parse("\n*** Begin Patch\n*** Delete File: file\n*** End Patch\n\n").is_ok());
    }
}
