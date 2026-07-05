//! Runtime implementation for the Anda agent framework.
//!
//! `anda_engine` turns the traits and data contracts from `anda_core` into a
//! runnable agent engine. It provides the execution context, model routing,
//! storage, hooks, memory tools, remote engine integration, and built-in
//! extensions used by Anda agents.
//!
//! # Main modules
//! - [`engine`]: engine construction, agent execution, tool calls, and remote
//!   engine metadata.
//! - [`context`]: runtime contexts passed to agents and tools, including cache,
//!   storage, HTTP, Web3, cancellation, and state features.
//! - [`model`]: model provider adapters and label-based model routing.
//! - [`extension`]: reusable tools such as filesystem (read, write, search,
//!   edit), shell, fetch, notes, skills, and todos.
//! - [`memory`]: persistent conversation and resource memory backed by AndaDB
//!   and the Cognitive Nexus.
//! - [`store`]: object storage abstraction used by engine contexts.

use anda_core::Json;
use candid::Principal;
use chrono::prelude::*;
use rand::RngExt;
use unicode_segmentation::UnicodeSegmentation;

pub mod context;
pub mod engine;
pub mod extension;
pub mod hook;
pub mod management;
pub mod memory;
pub mod model;
pub mod store;
pub mod subagent;

/// Returns the current Unix timestamp in milliseconds.
pub use structured_logger::unix_ms;

/// Generates cryptographically secure random bytes.
pub fn rand_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    rand::fill(&mut bytes);
    bytes
}

/// This is used to represent unauthenticated or anonymous users in the system.
pub const ANONYMOUS: Principal = Principal::anonymous();

/// User agent string used by Anda Engine HTTP clients.
pub static APP_USER_AGENT: &str = concat!(
    "Mozilla/5.0 anda.bot ",
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
);

/// Generates a random number within the given range.
pub fn rand_number<T, R>(range: R) -> T
where
    T: rand::distr::uniform::SampleUniform,
    R: rand::distr::uniform::SampleRange<T>,
{
    let mut rng = rand::rng();
    rng.random_range(range)
}

/// Returns the current UTC datetime in RFC 3339 format with millisecond precision.
pub fn rfc3339_datetime_now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Converts a Unix timestamp in milliseconds to an RFC 3339 UTC datetime string.
pub fn rfc3339_datetime(now_ms: u64) -> Option<String> {
    let datetime = DateTime::<Utc>::from_timestamp_millis(now_ms as i64);
    datetime.map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

/// Sets the Unix timestamp in milliseconds for each JSON object in the vector.
pub fn json_set_unix_ms_timestamp(mut vals: Vec<Json>, timestamp_ms: u64) -> Vec<Json> {
    for val in vals.iter_mut() {
        if let Some(obj) = val.as_object_mut() {
            obj.insert("timestamp".into(), timestamp_ms.into());
        }
    }
    vals
}

/// Converts each JSON object's `timestamp` field from Unix milliseconds to RFC 3339.
pub fn json_convert_rfc3339_timestamp(mut vals: Vec<Json>) -> Vec<Json> {
    for val in vals.iter_mut() {
        if let Some(obj) = val.as_object_mut()
            && let Some(timestamp_ms) = obj.get("timestamp").and_then(Json::as_u64)
        {
            obj.insert("timestamp".into(), rfc3339_datetime(timestamp_ms).into());
        }
    }
    vals
}

/// Converts a Unix timestamp in milliseconds to a local datetime string in the format "YYYY-MM-DD HH(AM/PM) ±TZ".
/// Example: 1970-01-01 08AM +08:00
pub fn local_date_hour(now_ms: u64) -> Option<String> {
    let local_datetime: Option<DateTime<Local>> =
        DateTime::<Utc>::from_timestamp_millis(now_ms as i64).map(|d| d.with_timezone(&Local));
    local_datetime.map(|dt| dt.format("%Y-%m-%d %I%p %:z").to_string())
}

/// Returns the largest byte length `<= max_bytes` that falls on a grapheme-cluster boundary of
/// `text`, so multi-codepoint characters — emoji ZWJ sequences (e.g. 👨‍👩‍👧‍👦), regional-indicator
/// flags, skin-tone modifiers, and combining marks — are never split.
///
/// The result is always a valid UTF-8 boundary. It is `text.len()` when the whole string fits, and
/// `0` when not even the first grapheme cluster fits within `max_bytes`.
pub fn grapheme_safe_cutoff(text: &str, max_bytes: usize) -> usize {
    if text.len() <= max_bytes {
        return text.len();
    }

    text.grapheme_indices(true)
        .map(|(idx, g)| idx + g.len())
        .take_while(|&end| end <= max_bytes)
        .last()
        .unwrap_or(0)
}

/// Truncates a UTF-8 string in place to at most `max_bytes`, backing off to the nearest
/// grapheme-cluster boundary via [`grapheme_safe_cutoff`] so multi-codepoint characters are never
/// split.
///
/// Returns the new byte length when truncation happened, or `None` when the string already fit
/// within `max_bytes`. When not even the first grapheme cluster fits, the string is truncated to
/// empty.
pub fn truncate_utf8_to_max_bytes(text: &mut String, max_bytes: usize) -> Option<usize> {
    if text.len() <= max_bytes {
        return None;
    }

    let cutoff = grapheme_safe_cutoff(text, max_bytes);
    text.truncate(cutoff);
    Some(cutoff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::DateTime;
    use serde_json::json;

    #[test]
    fn rand_number_returns_value_in_range() {
        let v: u8 = rand_number(3..10);
        assert!((3..10).contains(&v));
    }

    #[test]
    fn rfc3339_datetime_now_is_valid_rfc3339() {
        let now = rfc3339_datetime_now();
        let parsed = DateTime::parse_from_rfc3339(&now).expect("must be valid rfc3339 datetime");
        assert_eq!(parsed.offset().local_minus_utc(), 0);
    }

    #[test]
    fn rfc3339_datetime_handles_valid_and_invalid_timestamp() {
        let invalid_ms = i64::MIN as u64;

        assert_eq!(
            rfc3339_datetime(0),
            Some("1970-01-01T00:00:00.000Z".to_string())
        );
        assert_eq!(rfc3339_datetime(invalid_ms), None);
    }

    #[test]
    fn json_set_unix_ms_timestamp_updates_only_objects() {
        let vals = vec![json!({"k": 1}), json!("txt"), json!(null)];
        let out = json_set_unix_ms_timestamp(vals, 1234);

        assert_eq!(out[0]["timestamp"], json!(1234));
        assert_eq!(out[0]["k"], json!(1));
        assert_eq!(out[1], json!("txt"));
        assert_eq!(out[2], json!(null));
    }

    #[test]
    fn json_convert_rfc3339_timestamp_converts_numeric_timestamp() {
        let vals = vec![
            json!({"timestamp": 0, "name": "a"}),
            json!({"timestamp": "bad"}),
            json!(42),
        ];
        let out = json_convert_rfc3339_timestamp(vals);

        assert_eq!(out[0]["timestamp"], json!("1970-01-01T00:00:00.000Z"));
        assert_eq!(out[0]["name"], json!("a"));
        assert_eq!(out[1]["timestamp"], json!("bad"));
        assert_eq!(out[2], json!(42));
    }

    #[test]
    fn local_date_hour_returns_expected_shape_and_none_for_invalid() {
        let invalid_ms = i64::MIN as u64;
        let s = local_date_hour(0).expect("epoch should produce local datetime");
        println!("Local datetime for epoch: {}", s);
        // 1970-01-01 08AM +08:00
        let parts: Vec<&str> = s.split(' ').collect();

        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "1970-01-01");
        assert_eq!(parts[1].len(), 4);
        assert!(parts[1].ends_with("AM") || parts[1].ends_with("PM"));
        assert!(parts[1][..2].chars().all(|c| c.is_ascii_digit()));
        assert!(parts[2].starts_with('+') || parts[2].starts_with('-'));
        assert_eq!(parts[2].len(), 6);

        assert_eq!(local_date_hour(invalid_ms), None);
    }

    #[test]
    fn truncate_utf8_to_max_bytes_respects_budget_and_grapheme_boundaries() {
        // Within budget: untouched.
        let mut text = "hello".to_string();
        assert_eq!(truncate_utf8_to_max_bytes(&mut text, 5), None);
        assert_eq!(text, "hello");

        // ASCII over budget: truncated exactly to the byte budget.
        let mut text = "hello world".to_string();
        assert_eq!(truncate_utf8_to_max_bytes(&mut text, 5), Some(5));
        assert_eq!(text, "hello");

        // A multibyte codepoint is never split: "héllo" is 6 bytes (é = 2 bytes). A 2-byte budget
        // backs off to the boundary after "h".
        let mut text = "héllo".to_string();
        assert_eq!(truncate_utf8_to_max_bytes(&mut text, 2), Some(1));
        assert_eq!(text, "h");

        // A multi-codepoint grapheme cluster (family emoji joined by ZWJ, 25 bytes) is never split.
        // A budget that lands mid-cluster backs off to the previous cluster boundary.
        let family = "👨‍👩‍👧‍👦";
        assert_eq!(family.len(), 25);
        let mut text = family.repeat(3); // 75 bytes
        let cutoff = truncate_utf8_to_max_bytes(&mut text, 60).unwrap();
        assert_eq!(cutoff, 50);
        assert_eq!(text, family.repeat(2));

        // Budget smaller than the first cluster: truncated to empty rather than split.
        let mut text = family.to_string();
        assert_eq!(truncate_utf8_to_max_bytes(&mut text, 10), Some(0));
        assert!(text.is_empty());
    }
}
