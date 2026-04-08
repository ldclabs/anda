use std::fs::Metadata;

mod read;
mod write;

pub use read::*;
pub use write::*;

pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

pub(crate) const UTF8_ENCODING: &str = "utf8";
pub(crate) const BASE64_ENCODING: &str = "base64";

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
