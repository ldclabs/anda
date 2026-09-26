//! Project remote results without sending application metadata or base64 as prose.

use super::McpLimits;
use anda_core::{ByteBufB64, Json, ToolMedia, ToolPresentation};
use std::str::FromStr;

pub(super) fn present_result(result: &Json, limits: &McpLimits) -> ToolPresentation {
    let mut view = ToolPresentation::default();
    let mut media_bytes = 0usize;
    let mut truncated = false;
    // Reserve space for an unambiguous truncation marker.
    let budget = limits.output_text_bytes.saturating_sub(64);
    if let Some(value) = result.get("structured_content").filter(|v| !v.is_null()) {
        append_text(&mut view.text, &value.to_string(), budget, &mut truncated);
    }
    for content in result
        .get("content")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        match content.get("type").and_then(Json::as_str) {
            Some("text") => {
                if let Some(text) = content.get("text").and_then(Json::as_str) {
                    append_text(&mut view.text, text, budget, &mut truncated);
                }
            }
            Some("image" | "audio") => {
                let mime = content.get("mimeType").and_then(Json::as_str).unwrap_or("");
                let encoded = content.get("data").and_then(Json::as_str).unwrap_or("");
                let remaining = limits.output_media_bytes.saturating_sub(media_bytes);
                let valid_mime = mime.len() <= 128
                    && (matches!(
                        mime,
                        "image/png" | "image/jpeg" | "image/webp" | "image/gif"
                    ) || mime.starts_with("audio/"));
                let decoded = (valid_mime
                    && view.media.len() < limits.output_media_items
                    && encoded.len()
                        <= remaining
                            .saturating_mul(4)
                            .saturating_div(3)
                            .saturating_add(4))
                .then(|| ByteBufB64::from_str(encoded).ok())
                .flatten();
                if let Some(data) =
                    decoded.filter(|data| !data.is_empty() && data.len() <= remaining)
                {
                    media_bytes += data.len();
                    view.media.push(ToolMedia {
                        mime_type: mime.into(),
                        data,
                    });
                } else {
                    append_text(
                        &mut view.text,
                        "[tool media omitted: invalid encoding, type, or media budget exceeded]",
                        budget,
                        &mut truncated,
                    );
                }
            }
            Some("resource") => {
                if let Some(resource) = content.get("resource") {
                    if let Some(text) = resource.get("text").and_then(Json::as_str) {
                        append_text(&mut view.text, text, budget, &mut truncated);
                    } else if let Some(uri) = resource.get("uri").and_then(Json::as_str) {
                        append_text(
                            &mut view.text,
                            &format!("Resource: {uri}"),
                            budget,
                            &mut truncated,
                        );
                    }
                }
            }
            Some("resource_link") => {
                if let Some(uri) = content.get("uri").and_then(Json::as_str) {
                    append_text(
                        &mut view.text,
                        &format!("Resource: {uri}"),
                        budget,
                        &mut truncated,
                    );
                }
            }
            _ => append_text(
                &mut view.text,
                "[unsupported MCP content retained in raw result]",
                budget,
                &mut truncated,
            ),
        }
    }
    if truncated {
        view.text
            .push_str("\n[tool output truncated; full result retained by host]");
    }
    view
}

fn append_text(output: &mut String, text: &str, limit: usize, truncated: &mut bool) {
    if !output.is_empty() && output.len() < limit {
        output.push('\n');
    }
    let mut end = text.len().min(limit.saturating_sub(output.len()));
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    output.push_str(&text[..end]);
    *truncated |= end < text.len();
}
