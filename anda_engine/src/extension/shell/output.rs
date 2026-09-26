//! Bounded model-facing output, independent of process capture and log retention.

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

/// Default combined stdout/stderr budget for the session shell tools.
pub const DEFAULT_OUTPUT_BYTES: usize = 32 * 1024;

/// A bounded text preview with explicit omission metadata.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct OutputPreview {
    /// Text retaining both the beginning and end when truncated.
    pub text: String,
    /// Bytes omitted from the supplied decoded text.
    pub omitted_bytes: usize,
}

/// Limits decoded text without splitting grapheme clusters. The omission marker
/// counts against the budget; very small budgets may contain only a marker.
pub fn preview_output(text: &str, max_bytes: usize) -> OutputPreview {
    if text.len() <= max_bytes {
        return OutputPreview {
            text: text.to_owned(),
            omitted_bytes: 0,
        };
    }
    const MARKER: &str = "\n... [output omitted] ...\n";
    if max_bytes < MARKER.len() {
        return OutputPreview {
            text: MARKER[..max_bytes].to_owned(),
            omitted_bytes: text.len(),
        };
    }
    let available = max_bytes - MARKER.len();
    let head = crate::grapheme_safe_cutoff(text, available / 2);
    let tail_budget = available - head;
    let mut tail = text.len();
    for (index, _) in text.grapheme_indices(true).rev() {
        if text.len() - index > tail_budget {
            break;
        }
        tail = index;
    }
    OutputPreview {
        text: format!("{}{MARKER}{}", &text[..head], &text[tail..]),
        omitted_bytes: tail - head,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_keeps_error_tail_and_respects_every_budget() {
        let text = format!("START{}FAILED", "👩‍💻e\u{301}".repeat(100));
        for budget in 0..text.len() {
            let preview = preview_output(&text, budget);
            assert!(preview.text.len() <= budget);
            assert!(preview.omitted_bytes > 0);
        }
        let preview = preview_output(&text, 100);
        assert!(preview.text.starts_with("START"));
        assert!(preview.text.ends_with("FAILED"));
        assert!(!preview.text.contains('\u{fffd}'));
    }
}
