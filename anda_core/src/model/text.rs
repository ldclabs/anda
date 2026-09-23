//! Text decoding for untrusted bytes: UTF-8 first, an optional legacy-encoding
//! fallback, and the encoding-label helpers used by filesystem and shell tools.

use encoding_rs::{Encoding, UTF_8};
use std::borrow::Cow;

/// Attempts to decode the given byte slice as UTF-8 text and checks if it looks like text content.
fn utf8_text_from_bytes(data: &[u8]) -> Option<&str> {
    let text = std::str::from_utf8(data).ok()?;
    looks_like_text(text).then_some(text)
}

/// Attempts to decode bytes as text and checks if it looks like text content.
///
/// UTF-8 is always preferred. On Windows, non-UTF-8 bytes fall back to the
/// system ANSI code page used by many legacy text files.
pub fn text_from_bytes(data: &[u8]) -> Option<Cow<'_, str>> {
    text_from_bytes_with_encoding(data, platform_text_encoding())
}

/// Attempts to decode bytes as text using UTF-8 first and an explicit fallback encoding second.
pub fn text_from_bytes_with_encoding<'a>(
    data: &'a [u8],
    fallback_encoding: Option<&'static Encoding>,
) -> Option<Cow<'a, str>> {
    if let Some(text) = utf8_text_from_bytes(data) {
        return Some(Cow::Borrowed(text));
    }

    let encoding = fallback_encoding?;
    if encoding.name() == UTF_8.name() {
        return None;
    }

    let (text, _, had_errors) = encoding.decode(data);
    // `Encoding::decode` sniffs UTF-16 BOMs even when the requested fallback is
    // a legacy code page. A BOM-only non-empty byte slice decodes to an empty
    // string and should not become an empty prompt document.
    if had_errors || (!data.is_empty() && text.is_empty()) || !looks_like_text(&text) {
        return None;
    }

    Some(Cow::Owned(text.into_owned()))
}

/// Resolves a text encoding label, accepting both `utf8` and standard Encoding Standard labels.
pub fn text_encoding_for_label(label: &str) -> Option<&'static Encoding> {
    let label = label.trim();
    if label.eq_ignore_ascii_case("utf8") {
        return Some(UTF_8);
    }
    Encoding::for_label(label.as_bytes())
}

/// Returns the normalized label used by Anda for a decoded text encoding.
pub fn text_encoding_label(encoding: &'static Encoding) -> String {
    if encoding.name() == UTF_8.name() {
        "utf8".to_string()
    } else {
        encoding.name().to_ascii_lowercase()
    }
}

/// Returns the platform-local text encoding used for legacy text files.
///
/// On Windows this is the ANSI code page returned by `GetACP`; on other
/// platforms there is no legacy fallback and UTF-8 remains the only implicit
/// text encoding.
pub fn platform_text_encoding() -> Option<&'static Encoding> {
    #[cfg(target_os = "windows")]
    {
        windows_code_page_encoding(windows_file_code_page())
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// Maps a Windows code page number to an Encoding Standard decoder when supported.
pub fn windows_code_page_encoding(code_page: u32) -> Option<&'static Encoding> {
    match code_page {
        65001 => Some(UTF_8),
        936 => Some(encoding_rs::GBK),
        950 => Some(encoding_rs::BIG5),
        932 => Some(encoding_rs::SHIFT_JIS),
        949 => Some(encoding_rs::EUC_KR),
        1250 => Some(encoding_rs::WINDOWS_1250),
        1251 => Some(encoding_rs::WINDOWS_1251),
        1252 => Some(encoding_rs::WINDOWS_1252),
        1253 => Some(encoding_rs::WINDOWS_1253),
        1254 => Some(encoding_rs::WINDOWS_1254),
        1255 => Some(encoding_rs::WINDOWS_1255),
        1256 => Some(encoding_rs::WINDOWS_1256),
        1257 => Some(encoding_rs::WINDOWS_1257),
        1258 => Some(encoding_rs::WINDOWS_1258),
        _ => {
            let windows_label = format!("windows-{code_page}");
            Encoding::for_label(windows_label.as_bytes()).or_else(|| {
                let cp_label = format!("cp{code_page}");
                Encoding::for_label(cp_label.as_bytes())
            })
        }
    }
}

pub(crate) fn resource_text_from_bytes<'a>(
    data: &'a [u8],
    mime_type: Option<&str>,
) -> Option<Cow<'a, str>> {
    resource_text_from_bytes_with_encoding(data, mime_type, platform_text_encoding())
}

fn resource_text_from_bytes_with_encoding<'a>(
    data: &'a [u8],
    mime_type: Option<&str>,
    fallback_encoding: Option<&'static Encoding>,
) -> Option<Cow<'a, str>> {
    // Some binary formats (notably uncompressed PDFs) contain valid UTF-8.
    // Preserve their modality instead of exposing the file's source as text.
    if let Some(mime_type) = mime_type {
        let essence = mime_type
            .split(';')
            .next()
            .unwrap_or(mime_type)
            .trim()
            .to_ascii_lowercase();
        if essence == "application/pdf"
            || essence.starts_with("audio/")
            || essence.starts_with("video/")
            || (essence.starts_with("image/") && !essence.ends_with("+xml"))
        {
            return None;
        }
    }

    if let Some(text) = utf8_text_from_bytes(data) {
        return Some(Cow::Borrowed(text));
    }

    if let Some(mime_type) = mime_type
        && !mime_type_allows_text_fallback(mime_type)
    {
        return None;
    }

    text_from_bytes_with_encoding(data, fallback_encoding)
}

fn mime_type_allows_text_fallback(mime_type: &str) -> bool {
    let essence = mime_type
        .split(';')
        .next()
        .unwrap_or(mime_type)
        .trim()
        .to_ascii_lowercase();

    essence.is_empty()
        || essence.starts_with("text/")
        || essence.ends_with("+json")
        || essence.ends_with("+xml")
        || matches!(
            essence.as_str(),
            "application/json"
                | "application/xml"
                | "application/javascript"
                | "application/x-javascript"
                | "application/x-ndjson"
                | "application/yaml"
                | "application/x-yaml"
                | "application/toml"
                | "application/x-www-form-urlencoded"
        )
}

#[cfg(target_os = "windows")]
fn windows_file_code_page() -> u32 {
    // Legacy Windows text files commonly use the ANSI code page rather than the
    // OEM console code page used by `cmd.exe` output.
    unsafe { windows_sys::Win32::Globalization::GetACP() }
}

fn looks_like_text(text: &str) -> bool {
    let mut sampled = 0usize;
    let mut suspicious = 0usize;
    for ch in text.chars().take(4096) {
        sampled += 1;
        if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            suspicious += 1;
        }
    }

    sampled == 0 || suspicious * 100 / sampled <= 5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_from_bytes_decodes_utf8_and_legacy_fallback() {
        let decoded =
            text_from_bytes_with_encoding("中文.txt".as_bytes(), Some(encoding_rs::GBK)).unwrap();
        assert!(matches!(&decoded, Cow::Borrowed(_)));
        assert_eq!(decoded.as_ref(), "中文.txt");

        let (gbk, _, had_errors) = encoding_rs::GBK.encode("中文.txt");
        assert!(!had_errors);
        let decoded = text_from_bytes_with_encoding(&gbk, Some(encoding_rs::GBK)).unwrap();
        assert!(matches!(&decoded, Cow::Owned(_)));
        assert_eq!(decoded.as_ref(), "中文.txt");

        assert!(text_from_bytes_with_encoding(&gbk, Some(UTF_8)).is_none());
        assert!(text_from_bytes_with_encoding(&[0x81, 0x30], Some(encoding_rs::GBK)).is_none());
        assert!(
            resource_text_from_bytes_with_encoding(
                &[0xff, 0xfe],
                None,
                Some(encoding_rs::WINDOWS_1252),
            )
            .is_none()
        );
    }

    #[test]
    fn test_resource_text_fallback_respects_binary_mime_type() {
        let binary_header = [0xff, 0xd8, 0xff];
        assert!(
            resource_text_from_bytes_with_encoding(
                &binary_header,
                Some("image/jpeg"),
                Some(encoding_rs::WINDOWS_1252),
            )
            .is_none()
        );

        let (legacy_text, _, had_errors) = encoding_rs::WINDOWS_1252.encode("café");
        assert!(!had_errors);
        let decoded = resource_text_from_bytes_with_encoding(
            &legacy_text,
            Some("text/plain; charset=windows-1252"),
            Some(encoding_rs::WINDOWS_1252),
        )
        .unwrap();
        assert_eq!(decoded.as_ref(), "café");
    }

    #[test]
    fn resource_text_preserves_utf8_detection_for_text_and_unspecified_types() {
        for mime in [
            None,
            Some("application/octet-stream"),
            Some("text/plain"),
            Some("application/json"),
            Some("application/xml"),
            Some("image/svg+xml"),
            Some("application/example+json"),
        ] {
            let text = resource_text_from_bytes_with_encoding(b"UTF-8 text", mime, None).unwrap();
            assert!(matches!(text, Cow::Borrowed("UTF-8 text")));
        }
    }
}
