//! Chat message content: [`Message`], [`ContentPart`], and their wire codecs.
//!
//! `ContentPart` is the persisted, provider-neutral content model (see its
//! docs for the `raw_history` contract). The hand-written `Deserialize`
//! implementations keep CBOR byte strings intact, so RPC bodies round-trip
//! without a lossy `serde_json::Value` intermediate.

use candid::Principal;
use ic_auth_types::ByteBufB64;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use std::str::FromStr;

use super::text::resource_text_from_bytes;
use crate::{Json, Resource, ToolCall};

fn deserialize_content<'de, D>(deserializer: D) -> Result<Vec<ContentPart>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Deserialize directly instead of routing through `serde_json::Value`, whose
    // visitor cannot represent CBOR byte strings. Untagged buffering keeps byte
    // payloads (e.g. `InlineData.data`) intact so CBOR RPC bodies round-trip.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Content {
        Text(String),
        Parts(Vec<ContentPart>),
    }

    match Option::<Content>::deserialize(deserializer)? {
        None => Ok(Vec::new()),
        Some(Content::Text(s)) => Ok(vec![ContentPart::Text { text: s }]),
        Some(Content::Parts(parts)) => Ok(parts),
    }
}

/// Chat message sent to or returned by an LLM provider.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Message {
    /// Message role: "system", "user", "assistant", "tool".
    pub role: String,

    /// Message content parts.
    #[serde(default, deserialize_with = "deserialize_content")]
    pub content: Vec<ContentPart>,

    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    /// This field is not used by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The user ID of the message sender.
    /// This field is not used by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<Principal>,

    /// The timestamp of the message.
    /// This field is not used by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

impl Message {
    /// Returns all text content parts joined with blank lines.
    pub fn text(&self) -> Option<String> {
        let mut texts: Vec<&str> = Vec::new();
        for part in &self.content {
            if let ContentPart::Text { text } = part {
                texts.push(text);
            }
        }
        if texts.is_empty() {
            return None;
        }
        Some(texts.join("\n\n"))
    }

    /// Returns all reasoning content parts joined with blank lines.
    pub fn thoughts(&self) -> Option<String> {
        let mut thoughts: Vec<&str> = Vec::new();
        for part in &self.content {
            if let ContentPart::Reasoning { text } = part {
                thoughts.push(text);
            }
        }
        if thoughts.is_empty() {
            return None;
        }
        Some(thoughts.join("\n\n"))
    }

    /// Extracts tool calls from this message.
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for part in &self.content {
            if let ContentPart::ToolCall {
                name,
                args,
                call_id,
            } = part
            {
                tool_calls.push(ToolCall {
                    name: name.clone(),
                    args: args.clone(),
                    call_id: call_id.clone(),
                    result: None,
                    remote_id: None,
                });
            }
        }
        tool_calls
    }

    /// Removes non-visible content parts and appends a short pruning notice.
    pub fn prune_content(&mut self) -> usize {
        let original_len = self.content.len();
        self.content.retain(|part| {
            matches!(
                part,
                ContentPart::Text { .. }
                    | ContentPart::Reasoning { .. }
                    | ContentPart::Action { .. }
            )
        });
        let pruned = original_len - self.content.len();
        if pruned > 0 {
            self.content.push(ContentPart::Text {
                text: format!(
                    "[{} items (tool calls or files) pruned due to limits]",
                    pruned
                ),
            });
        }
        pruned
    }
}

/// A single content item inside a chat message.
///
/// The enum supports Anda's normalized content types while preserving unknown
/// provider-specific JSON payloads in [`ContentPart::Any`].
///
/// # This is the persisted, provider-neutral view
///
/// `ContentPart` is what gets **stored** as conversation history and what crosses the engine
/// API boundary. It intentionally does **not** carry a provider's per-turn intermediate
/// state — Anthropic's `thinking.signature`, Gemini's `thoughtSignature`, and similar opaque
/// tokens. Those are only meaningful inside the reasoning round that produced them, so a
/// stored conversation has no use for them.
///
/// Within a round they are not lost: the provider-native messages travel in
/// [`CompletionRequest::raw_history`](crate::model::CompletionRequest::raw_history), which
/// every model adapter sends ahead of the converted `chat_history`, so they never pass
/// through this lossy conversion. See that field's docs for the full contract.
///
/// **Do not add provider-specific fields to this enum** to "preserve" such state — that is
/// what `raw_history` is for. A conversion that *drops* one on the way in is correct by
/// design. What is not correct is a conversion that, on the way back *out*, emits something
/// the provider rejects (for example an Anthropic `thinking` block with an empty signature);
/// omit the block instead.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all_fields = "camelCase")]
pub enum ContentPart {
    /// Visible text content.
    Text {
        /// Text body.
        text: String,
    },
    /// Provider reasoning or thinking text.
    Reasoning {
        /// Reasoning text body.
        text: String,
    },
    /// File content referenced by URI.
    FileData {
        /// URI pointing to the file data.
        file_uri: String,

        /// MIME type if known.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    /// Inline binary data with an explicit MIME type.
    InlineData {
        /// MIME type for the inline bytes.
        mime_type: String,
        /// Base64-encoded binary payload.
        data: ByteBufB64,
    },
    /// Tool call requested by a model.
    ToolCall {
        /// Tool function name.
        name: String,
        /// JSON arguments for the tool call.
        args: Json,

        /// Provider call identifier used to correlate tool outputs.
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
    },
    /// Tool output returned to a model.
    ToolOutput {
        /// Tool function name.
        name: String,
        /// JSON output payload.
        output: Json,

        /// Whether the tool output represents an error.
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,

        /// Provider call identifier this output answers.
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,

        /// Remote engine principal when the tool call was delegated.
        #[serde(skip_serializing_if = "Option::is_none")]
        remote_id: Option<Principal>,
    },
    /// Signed action payload emitted by an agent.
    Action {
        /// Action name.
        name: String,
        /// Action-specific payload.
        payload: Json,

        /// Principals that should receive the action.
        #[serde(skip_serializing_if = "Option::is_none")]
        recipients: Option<Vec<Principal>>,

        /// Optional signature over the action payload.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<ByteBufB64>,
    },
    /// Provider-specific content part preserved as raw JSON.
    #[serde(untagged)]
    Any(Json),
}

impl ContentPart {
    /// Creates a content part of type `Any` with the given type tag and value.
    ///
    /// The type tag is only added when `val` serializes to a JSON object.
    pub fn any_from<T>(ty: &str, val: T) -> Self
    where
        T: Serialize,
    {
        let mut val = json!(val);
        if let Some(map) = val.as_object_mut() {
            map.insert("type".to_string(), ty.into());
        }
        ContentPart::Any(val)
    }

    /// Attempts to convert this content part of type `Any` into the specified type if the type tag matches.
    pub fn any_into<T>(self, ty: &str) -> Result<T, Box<Self>>
    where
        T: DeserializeOwned,
    {
        if let ContentPart::Any(val) = &self
            && let Some(t) = val.get("type").and_then(|x| x.as_str())
            && t == ty
        {
            T::deserialize(val).map_err(|_| Box::new(self))
        } else {
            Err(Box::new(self))
        }
    }

    /// Estimates the number of tokens in this content part for usage accounting and pruning.
    pub fn estimated_tokens(&self) -> usize {
        match self {
            ContentPart::Text { text } | ContentPart::Reasoning { text } => estimate_tokens(text),
            ContentPart::FileData {
                file_uri,
                mime_type,
            } => estimate_tokens(file_uri)
                .saturating_add(mime_type.as_deref().map_or(0, estimate_tokens)),
            ContentPart::InlineData { mime_type, data } => {
                // Base64 expands bytes by ~4/3 and ~4 base64 chars ≈ 1 token, so
                // the encoded payload is roughly `len / 3` tokens.
                estimate_tokens(mime_type).saturating_add(data.len().div_ceil(3))
            }
            ContentPart::ToolCall {
                name,
                args,
                call_id,
            } => estimate_tokens(name)
                .saturating_add(estimate_tokens(&args.to_string()))
                .saturating_add(call_id.as_deref().map_or(0, estimate_tokens)),
            ContentPart::ToolOutput {
                name,
                output,
                call_id,
                ..
            } => estimate_tokens(name)
                .saturating_add(estimate_tokens(&output.to_string()))
                .saturating_add(call_id.as_deref().map_or(0, estimate_tokens)),
            ContentPart::Action { name, payload, .. } => {
                estimate_tokens(name).saturating_add(estimate_tokens(&payload.to_string()))
            }
            ContentPart::Any(value) => estimate_tokens(&value.to_string()),
        }
    }
}

/// Converts a content part with inline data to a data URL string.
///
/// See <https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Schemes/data>.
pub fn part_to_data_url(data: &ByteBufB64, mime_type: Option<&str>) -> String {
    format!(
        "data:{};base64,{}",
        mime_type.unwrap_or(""),
        data.to_base64()
    )
}

/// Parses a data URL string and extracts the inline data and MIME type, if applicable.
///
/// When the data URL omits the media type (e.g. `data:;base64,...`), the MIME
/// type is inferred from the decoded bytes, falling back to
/// `application/octet-stream` for base64 payloads and `text/plain` for
/// percent-encoded payloads.
pub fn inline_data_from_data_url(data_url: &str) -> Option<(ByteBufB64, String)> {
    if let Some(stripped) = data_url.strip_prefix("data:") {
        let (meta, data_part) = stripped.split_once(",")?;

        if let Some(mime_part) = meta.strip_suffix(";base64") {
            if let Ok(data) = ByteBufB64::from_str(data_part) {
                let mime_type = if mime_part.is_empty() {
                    infer2::get(&data)
                        .map(|t| t.mime_type().to_string())
                        .unwrap_or_else(|| "application/octet-stream".to_string())
                } else {
                    mime_part.to_string()
                };
                Some((data, mime_type))
            } else {
                None
            }
        } else {
            let data = decode_percent_encoded_bytes(data_part)?;
            let mime_type = if meta.is_empty() {
                infer2::get(&data)
                    .map(|t| t.mime_type().to_string())
                    .unwrap_or_else(|| "text/plain".to_string())
            } else {
                meta.to_string()
            };
            Some((data, mime_type))
        }
    } else if let Ok(data) = ByteBufB64::from_str(data_url) {
        let mime_type = infer2::get(&data).map(|t| t.mime_type().to_string());
        Some((
            data,
            mime_type.unwrap_or_else(|| "application/octet-stream".to_string()),
        ))
    } else {
        None
    }
}

/// A `Principal` that decodes from either a text string (human-readable formats)
/// or raw bytes (binary formats), tolerant of the format serde's untagged/tagged
/// buffering exposes.
///
/// Serde's buffering deserializes fields with `is_human_readable() == true`
/// regardless of the wire format, which drives `candid::Principal` into its
/// Candid-framed byte path and rejects raw CBOR principal bytes. Decoding via
/// `deserialize_any` with a visitor that accepts both text and raw bytes avoids
/// that mismatch.
struct PrincipalCompat(Principal);

impl<'de> Deserialize<'de> for PrincipalCompat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct PrincipalCompatVisitor;

        impl serde::de::Visitor<'_> for PrincipalCompatVisitor {
            type Value = Principal;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a principal as text or raw bytes")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Principal, E> {
                Principal::from_text(v).map_err(E::custom)
            }

            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Principal, E> {
                Principal::try_from(v).map_err(E::custom)
            }

            fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Principal, E> {
                Principal::try_from(v.as_slice()).map_err(E::custom)
            }
        }

        deserializer
            .deserialize_any(PrincipalCompatVisitor)
            .map(PrincipalCompat)
    }
}

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // The known content types mirror the derived `Serialize` tag/rename rules
        // so both JSON and CBOR round-trip. `Typed` is buffered by serde's
        // untagged machinery, which (unlike `serde_json::Value`) preserves CBOR
        // byte strings such as `InlineData.data`, `Action.signature`,
        // `ToolOutput.remote_id`, and `Action.recipients`. Principals use
        // [`PrincipalCompat`] so their raw-byte encoding survives buffering.
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all_fields = "camelCase")]
        enum Typed {
            Text {
                text: String,
            },
            Reasoning {
                text: String,
            },
            FileData {
                file_uri: String,
                mime_type: Option<String>,
            },
            InlineData {
                mime_type: String,
                data: ByteBufB64,
            },
            ToolCall {
                name: String,
                args: Json,
                call_id: Option<String>,
            },
            ToolOutput {
                name: String,
                output: Json,
                is_error: Option<bool>,
                call_id: Option<String>,
                remote_id: Option<PrincipalCompat>,
            },
            Action {
                name: String,
                payload: Json,
                recipients: Option<Vec<PrincipalCompat>>,
                signature: Option<ByteBufB64>,
            },
        }

        // A bare string is text; a tagged object with a known type is that
        // variant; anything else (including a known tag with mismatched fields)
        // is preserved verbatim as `Any`, matching `From<Json>` semantics.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Str(String),
            Typed(Typed),
            Any(Json),
        }

        Ok(match Helper::deserialize(deserializer)? {
            Helper::Str(text) => ContentPart::Text { text },
            Helper::Any(value) => ContentPart::Any(value),
            Helper::Typed(typed) => match typed {
                Typed::Text { text } => ContentPart::Text { text },
                Typed::Reasoning { text } => ContentPart::Reasoning { text },
                Typed::FileData {
                    file_uri,
                    mime_type,
                } => ContentPart::FileData {
                    file_uri,
                    mime_type,
                },
                Typed::InlineData { mime_type, data } => {
                    ContentPart::InlineData { mime_type, data }
                }
                Typed::ToolCall {
                    name,
                    args,
                    call_id,
                } => ContentPart::ToolCall {
                    name,
                    args,
                    call_id,
                },
                Typed::ToolOutput {
                    name,
                    output,
                    is_error,
                    call_id,
                    remote_id,
                } => ContentPart::ToolOutput {
                    name,
                    output,
                    is_error,
                    call_id,
                    remote_id: remote_id.map(|p| p.0),
                },
                Typed::Action {
                    name,
                    payload,
                    recipients,
                    signature,
                } => ContentPart::Action {
                    name,
                    payload,
                    recipients: recipients.map(|list| list.into_iter().map(|p| p.0).collect()),
                    signature,
                },
            },
        })
    }
}

impl From<String> for ContentPart {
    fn from(text: String) -> Self {
        Self::Text { text }
    }
}

impl From<Json> for ContentPart {
    fn from(val: Json) -> Self {
        // Reuse the `Deserialize` logic so both paths agree: known tags become
        // the matching variant and everything else is preserved as `Any`.
        match ContentPart::deserialize(&val) {
            Ok(part) => part,
            Err(_) => ContentPart::Any(val),
        }
    }
}

impl TryFrom<Resource> for ContentPart {
    type Error = Resource;
    fn try_from(res: Resource) -> Result<Self, Self::Error> {
        if res.blob.as_ref().map(|v| !v.0.is_empty()).unwrap_or(false)
            && let Some(data) = res.blob
        {
            match resource_text_from_bytes(&data.0, res.mime_type.as_deref()) {
                Some(text) => Ok(ContentPart::Text {
                    text: text.into_owned(),
                }),
                None => {
                    let data: ByteBufB64 = data.0.into();
                    let mime_type = res.mime_type.unwrap_or_else(|| {
                        infer2::get(&data)
                            .map(|t| t.mime_type())
                            .unwrap_or("application/octet-stream")
                            .to_string()
                    });
                    Ok(ContentPart::InlineData { mime_type, data })
                }
            }
        } else if res
            .uri
            .as_ref()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
            && let Some(file_uri) = res.uri
        {
            Ok(ContentPart::FileData {
                file_uri,
                mime_type: res.mime_type,
            })
        } else {
            Err(res)
        }
    }
}

/// Estimates token count using a small, provider-independent heuristic.
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count()).saturating_add(3) / 4
}

fn decode_percent_encoded_bytes(input: &str) -> Option<ByteBufB64> {
    fn decode_hex(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' => {
                let hi = *bytes.get(index + 1)?;
                let lo = *bytes.get(index + 2)?;
                decoded.push((decode_hex(hi)? << 4) | decode_hex(lo)?);
                index += 3;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }

    Some(decoded.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::resource;
    use serde_json::Map;

    #[test]
    fn test_data_url_helpers_round_trip_and_invalid_inputs() {
        let data: ByteBufB64 = b"hello".to_vec().into();
        let mime_type = "text/plain".to_string();

        let data_url = part_to_data_url(&data, Some(&mime_type));
        assert_eq!(data_url, "data:text/plain;base64,aGVsbG8=");

        let (decoded, decoded_mime_type) = inline_data_from_data_url(&data_url).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_mime_type, "text/plain");

        let (decoded, _) = inline_data_from_data_url("aGVsbG8=").unwrap();
        assert_eq!(decoded, data);

        let html_url = "data:text/html,%3Ch1%3EHello%2C%20World%21%3C%2Fh1%3E";
        let (decoded, decoded_mime_type) = inline_data_from_data_url(html_url).unwrap();
        let expected_html: ByteBufB64 = b"<h1>Hello, World!</h1>".to_vec().into();
        assert_eq!(decoded, expected_html);
        assert_eq!(decoded_mime_type, "text/html");

        let (decoded, decoded_mime_type) =
            inline_data_from_data_url("data:text/plain,hello").unwrap();
        let expected_text: ByteBufB64 = b"hello".to_vec().into();
        assert_eq!(decoded, expected_text);
        assert_eq!(decoded_mime_type, "text/plain");

        assert!(inline_data_from_data_url("data:text/plain,%GG").is_none());
        assert!(inline_data_from_data_url("not-base64%%%").is_none());
    }

    #[test]
    fn test_inline_data_from_data_url_infers_missing_mime_type() {
        // base64 data URL without media type: infer from magic bytes
        let jpeg_header: ByteBufB64 = vec![0xff, 0xd8, 0xff, 0xe0].into();
        let data_url = format!("data:;base64,{}", jpeg_header.to_base64());
        let (decoded, mime_type) = inline_data_from_data_url(&data_url).unwrap();
        assert_eq!(decoded, jpeg_header);
        assert_eq!(mime_type, "image/jpeg");

        // base64 data URL without media type and no recognizable magic bytes
        let (decoded, mime_type) = inline_data_from_data_url("data:;base64,aGVsbG8=").unwrap();
        assert_eq!(decoded, ByteBufB64::from(b"hello".to_vec()));
        assert_eq!(mime_type, "application/octet-stream");

        // percent-encoded data URL without media type defaults to text/plain
        let (decoded, mime_type) = inline_data_from_data_url("data:,Hello%20World").unwrap();
        assert_eq!(decoded, ByteBufB64::from(b"Hello World".to_vec()));
        assert_eq!(mime_type, "text/plain");
    }

    #[test]
    fn test_content_part_try_from_resource_variants() {
        let text = Resource {
            blob: Some(b"hello".to_vec().into()),
            ..resource(1, &["text"])
        };
        assert_eq!(
            ContentPart::try_from(text).unwrap(),
            ContentPart::Text {
                text: "hello".into(),
            }
        );

        let binary = Resource {
            blob: Some(vec![0xff, 0xd8, 0xff].into()),
            mime_type: Some("image/jpeg".into()),
            ..resource(2, &["image"])
        };
        assert_eq!(
            ContentPart::try_from(binary).unwrap(),
            ContentPart::InlineData {
                mime_type: "image/jpeg".into(),
                data: vec![0xff, 0xd8, 0xff].into(),
            }
        );

        let file = Resource {
            uri: Some("file:///tmp/a.txt".into()),
            mime_type: Some("text/plain".into()),
            ..resource(3, &["text"])
        };
        assert_eq!(
            ContentPart::try_from(file).unwrap(),
            ContentPart::FileData {
                file_uri: "file:///tmp/a.txt".into(),
                mime_type: Some("text/plain".into()),
            }
        );

        let empty_blob = Resource {
            blob: Some(Vec::<u8>::new().into()),
            ..resource(4, &["text"])
        };
        assert!(ContentPart::try_from(empty_blob).is_err());

        let empty_uri = Resource {
            uri: Some("   ".into()),
            ..resource(5, &["text"])
        };
        assert!(ContentPart::try_from(empty_uri).is_err());
    }

    #[test]
    fn test_message_content_deserialize_rejects_non_string_non_array() {
        assert!(
            serde_json::from_value::<Message>(json!({
                "role": "user",
                "content": 123,
            }))
            .is_err()
        );
    }

    #[test]
    fn test_content_part_text_serde_and_from() {
        let part: ContentPart = "hello".to_string().into();
        assert_eq!(
            part,
            ContentPart::Text {
                text: "hello".into()
            }
        );

        // serde round-trip
        let v = serde_json::to_value(&part).unwrap();
        assert_eq!(v.get("type").unwrap(), "Text");
        assert_eq!(v.get("text").unwrap(), "hello");

        let back: ContentPart = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back, part);
        let back: ContentPart = v.into();
        assert_eq!(back, part);

        let part: Vec<ContentPart> = serde_json::from_str(
            r#"
            [
                "hello",
                {
                    "type": "Text",
                    "text": "world"
                }
            ]
            "#,
        )
        .unwrap();
        assert_eq!(
            part,
            vec![
                ContentPart::Text {
                    text: "hello".into()
                },
                ContentPart::Text {
                    text: "world".into()
                }
            ]
        );
    }

    #[test]
    fn test_content_part_filedata_serde_optional() {
        // mime_type = None -> not serialized
        let part = ContentPart::FileData {
            file_uri: "gs://bucket/file".into(),
            mime_type: None,
        };
        let v = serde_json::to_value(&part).unwrap();
        assert_eq!(v.get("type").unwrap(), "FileData");
        // fields use camelCase
        assert_eq!(v.get("fileUri").unwrap(), "gs://bucket/file");
        assert!(v.get("mimeType").is_none());

        // mime_type = Some -> present
        let part2 = ContentPart::FileData {
            file_uri: "gs://bucket/file2".into(),
            mime_type: Some("image/png".into()),
        };
        let v2 = serde_json::to_value(&part2).unwrap();
        assert_eq!(v2.get("type").unwrap(), "FileData");
        assert_eq!(v2.get("fileUri").unwrap(), "gs://bucket/file2");
        assert_eq!(v2.get("mimeType").unwrap(), "image/png");

        // deserialization check
        let back: ContentPart = serde_json::from_value(v2.clone()).unwrap();
        assert_eq!(back, part2);
        let back: ContentPart = v2.into();
        assert_eq!(back, part2);
    }

    #[test]
    fn test_content_part_inlinedata_serde() {
        let part = ContentPart::InlineData {
            mime_type: "text/plain".into(),
            data: b"hello".to_vec().into(),
        };
        let v = serde_json::to_value(&part).unwrap();
        assert_eq!(v.get("type").unwrap(), "InlineData");
        assert_eq!(v.get("mimeType").unwrap(), "text/plain");
        assert_eq!(v.get("data").unwrap(), "b64:aGVsbG8=");

        let back: ContentPart = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back, part);
        let back: ContentPart = v.into();
        assert_eq!(back, part);
    }

    #[test]
    fn test_content_part_any_serde() {
        let v = json!({
            "type": "text/plain",
            "data": "aGVsbG8=",
        });
        let part: ContentPart = v.clone().into();
        assert_eq!(part, ContentPart::Any(v));
        let v2 = serde_json::to_value(&part).unwrap();
        assert_eq!(v2.get("type").unwrap(), "text/plain");
        assert_eq!(v2.get("data").unwrap(), "aGVsbG8=");

        let part = ContentPart::Any(json!({
            "data": "aGVsbG8=",
        }));
        let v2 = serde_json::to_value(&part).unwrap();
        assert!(v2.get("type").is_none());
        assert_eq!(v2.get("data").unwrap(), "aGVsbG8=");
    }

    #[test]
    fn test_content_part_any_supports_resource_serde() {
        let mut metadata = Map::new();
        metadata.insert("source".into(), json!("upload"));
        metadata.insert("priority".into(), json!(3));

        let resource = Resource {
            _id: 42,
            name: "note.txt".into(),
            tags: vec!["text".into(), "note".into()],
            description: Some("A note resource".into()),
            uri: Some("file:///tmp/note.txt".into()),
            mime_type: Some("text/plain".into()),
            blob: Some(b"hello world".to_vec().into()),
            size: Some(11),
            metadata: Some(metadata),
            ..Default::default()
        };

        let resource_json = json!(resource);
        let part: ContentPart = resource_json.clone().into();
        assert_eq!(part, ContentPart::Any(resource_json.clone()));

        let serialized = serde_json::to_value(&part).unwrap();
        assert_eq!(serialized, resource_json);

        let back: ContentPart = serde_json::from_value(serialized.clone()).unwrap();
        assert_eq!(back, ContentPart::Any(resource_json));

        let resource_back: Resource = serde_json::from_value(serialized).unwrap();
        assert_eq!(resource_back._id, 42);
        assert_eq!(resource_back.name, "note.txt");
        assert_eq!(resource_back.tags, vec!["text", "note"]);
        assert_eq!(
            resource_back.description.as_deref(),
            Some("A note resource")
        );
        assert_eq!(resource_back.uri.as_deref(), Some("file:///tmp/note.txt"));
        assert_eq!(resource_back.mime_type.as_deref(), Some("text/plain"));
        assert_eq!(resource_back.blob, Some(b"hello world".to_vec().into()));
        assert_eq!(resource_back.size, Some(11));
        assert_eq!(
            resource_back
                .metadata
                .as_ref()
                .and_then(|meta| meta.get("source")),
            Some(&json!("upload"))
        );
        assert_eq!(
            resource_back
                .metadata
                .as_ref()
                .and_then(|meta| meta.get("priority")),
            Some(&json!(3))
        );
    }

    #[test]
    fn test_content_part_any_from_and_any_into_resource() {
        let mut metadata = Map::new();
        metadata.insert("source".into(), json!("upload"));
        metadata.insert("priority".into(), json!(3));

        let resource = Resource {
            _id: 42,
            name: "note.txt".into(),
            tags: vec!["text".into(), "note".into()],
            description: Some("A note resource".into()),
            uri: Some("file:///tmp/note.txt".into()),
            mime_type: Some("text/plain".into()),
            blob: Some(b"hello world".to_vec().into()),
            size: Some(11),
            metadata: Some(metadata),
            ..Default::default()
        };

        let part = ContentPart::any_from("Resource", &resource);
        let expected = json!({
            "type": "Resource",
            "_id": 42,
            "name": "note.txt",
            "tags": ["text", "note"],
            "description": "A note resource",
            "uri": "file:///tmp/note.txt",
            "mime_type": "text/plain",
            "blob": "b64:aGVsbG8gd29ybGQ=",
            "size": 11,
            "metadata": {
                "source": "upload",
                "priority": 3
            }
        });
        assert_eq!(part, ContentPart::Any(expected));

        let resource_back = part.clone().any_into::<Resource>("Resource").unwrap();
        assert_eq!(resource_back._id, resource._id);
        assert_eq!(resource_back.name, resource.name);
        assert_eq!(resource_back.tags, resource.tags);
        assert_eq!(resource_back.description, resource.description);
        assert_eq!(resource_back.uri, resource.uri);
        assert_eq!(resource_back.mime_type, resource.mime_type);
        assert_eq!(resource_back.blob, resource.blob);
        assert_eq!(resource_back.size, resource.size);
        assert_eq!(resource_back.metadata, resource.metadata);

        assert_eq!(
            part.clone().any_into::<Resource>("OtherType"),
            Err(Box::new(part.clone()))
        );

        let invalid = ContentPart::any_from("Resource", "plain-text");
        assert_eq!(
            invalid.clone().any_into::<Resource>("Resource"),
            Err(Box::new(invalid))
        );
    }

    #[test]
    fn test_content_part_toolcall_and_tooloutput_serde() {
        let call = ContentPart::ToolCall {
            name: "sum".into(),
            args: serde_json::json!({"x":1, "y":2}),
            call_id: None,
        };
        let v_call = serde_json::to_value(&call).unwrap();
        assert_eq!(v_call.get("type").unwrap(), "ToolCall");
        assert_eq!(v_call.get("name").unwrap(), "sum");
        assert_eq!(
            v_call.get("args").unwrap(),
            &serde_json::json!({"x":1, "y":2})
        );
        // callId omitted
        assert!(v_call.get("callId").is_none());
        let back_call: ContentPart = serde_json::from_value(v_call.clone()).unwrap();
        assert_eq!(back_call, call);
        let back: ContentPart = v_call.into();
        assert_eq!(back, call);

        let out = ContentPart::ToolOutput {
            name: "sum".into(),
            output: serde_json::json!({"result":3}),
            is_error: None,
            call_id: Some("c1".into()),
            remote_id: None,
        };
        let v_out = serde_json::to_value(&out).unwrap();
        assert_eq!(v_out.get("type").unwrap(), "ToolOutput");
        assert_eq!(v_out.get("name").unwrap(), "sum");
        assert_eq!(
            v_out.get("output").unwrap(),
            &serde_json::json!({"result":3})
        );
        // callId present
        assert_eq!(v_out.get("callId").unwrap(), "c1");
        let back_out: ContentPart = serde_json::from_value(v_out.clone()).unwrap();
        assert_eq!(back_out, out);
        let back: ContentPart = v_out.into();
        assert_eq!(back, out);
    }

    /// A message carrying every byte-string payload (`InlineData.data`,
    /// `ToolOutput.remote_id`, `Action.recipients`/`signature`) must survive a
    /// CBOR round-trip. CBOR encodes these as byte strings, which a
    /// `serde_json::Value` intermediate cannot decode, so this guards the
    /// non-`Value` deserialization path used on the RPC wire.
    #[test]
    fn test_message_cbor_round_trip_with_byte_payloads() {
        let principal = Principal::from_slice(&[1, 2, 3, 4, 5]);
        let message = Message {
            role: "assistant".into(),
            content: vec![
                ContentPart::Text { text: "hi".into() },
                ContentPart::InlineData {
                    mime_type: "image/png".into(),
                    data: vec![0u8, 159, 146, 150, 255].into(),
                },
                ContentPart::ToolOutput {
                    name: "delegate".into(),
                    output: json!({"ok": true}),
                    is_error: Some(false),
                    call_id: Some("c1".into()),
                    remote_id: Some(principal),
                },
                ContentPart::Action {
                    name: "notify".into(),
                    payload: json!({"n": 1}),
                    recipients: Some(vec![principal]),
                    signature: Some(vec![9u8, 8, 7, 0, 255].into()),
                },
                ContentPart::Any(json!({"provider": "x", "n": 2})),
            ],
            name: Some("$system".into()),
            user: Some(principal),
            timestamp: Some(42),
        };

        // CBOR (non-human-readable): byte strings must round-trip.
        let cbor = cbor2::to_canonical_vec(&message).unwrap();
        let from_cbor: Message = cbor2::from_slice(&cbor).unwrap();
        assert_eq!(from_cbor, message);

        // JSON (human-readable) must still round-trip.
        let json = serde_json::to_vec(&message).unwrap();
        let from_json: Message = serde_json::from_slice(&json).unwrap();
        assert_eq!(from_json, message);
    }

    /// A `ContentPart` with a known `type` tag but mismatched fields falls back
    /// to `Any`, matching `From<Json>` semantics rather than erroring the whole
    /// message.
    #[test]
    fn test_content_part_unknown_and_malformed_fall_back_to_any() {
        // Known tag, wrong field shape (missing `data`): preserved as `Any`.
        let malformed = json!({"type": "InlineData", "mimeType": "image/png"});
        let part: ContentPart = serde_json::from_value(malformed.clone()).unwrap();
        assert_eq!(part, ContentPart::Any(malformed.clone()));
        assert_eq!(
            ContentPart::from(malformed.clone()),
            ContentPart::Any(malformed)
        );

        // Unknown tag: preserved as `Any`.
        let unknown = json!({"type": "Custom", "x": 1});
        let part: ContentPart = serde_json::from_value(unknown.clone()).unwrap();
        assert_eq!(part, ContentPart::Any(unknown));
    }

    #[test]
    fn test_message_text_collects_only_text_parts_in_order() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![
                ContentPart::Reasoning {
                    text: "first thought".into(),
                },
                ContentPart::Text {
                    text: "first text".into(),
                },
                ContentPart::ToolCall {
                    name: "sum".into(),
                    args: serde_json::json!({"x":1, "y":2}),
                    call_id: Some("call_1".into()),
                },
                ContentPart::Text {
                    text: "second text".into(),
                },
                ContentPart::Action {
                    name: "notify".into(),
                    payload: serde_json::json!({"ok": true}),
                    recipients: None,
                    signature: None,
                },
            ],
            ..Default::default()
        };

        assert_eq!(msg.text().as_deref(), Some("first text\n\nsecond text"));

        let no_text = Message {
            role: "assistant".into(),
            content: vec![ContentPart::Reasoning {
                text: "thought only".into(),
            }],
            ..Default::default()
        };
        assert_eq!(no_text.text(), None);
    }

    #[test]
    fn test_message_thoughts_collects_only_reasoning_parts_in_order() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![
                ContentPart::Text {
                    text: "visible text".into(),
                },
                ContentPart::Reasoning {
                    text: "first thought".into(),
                },
                ContentPart::ToolOutput {
                    name: "sum".into(),
                    output: serde_json::json!({"result": 3}),
                    is_error: None,
                    call_id: Some("call_1".into()),
                    remote_id: None,
                },
                ContentPart::Reasoning {
                    text: "second thought".into(),
                },
            ],
            ..Default::default()
        };

        assert_eq!(
            msg.thoughts().as_deref(),
            Some("first thought\n\nsecond thought")
        );

        let no_reasoning = Message {
            role: "assistant".into(),
            content: vec![ContentPart::Text {
                text: "text only".into(),
            }],
            ..Default::default()
        };
        assert_eq!(no_reasoning.thoughts(), None);
    }

    #[test]
    fn test_message_tool_calls_extract_from_content_parts() {
        let parts = vec![
            ContentPart::Text {
                text: "hello".into(),
            },
            ContentPart::ToolCall {
                name: "sum".into(),
                args: serde_json::json!({"x":1, "y": 2}),
                call_id: Some("abc".into()),
            },
            ContentPart::ToolCall {
                name: "echo".into(),
                args: serde_json::json!({"text":"hi"}),
                call_id: None,
            },
            ContentPart::ToolOutput {
                name: "sum".into(),
                output: serde_json::json!({"result": 3}),
                is_error: None,
                call_id: Some("abc".into()),
                remote_id: None,
            },
        ];
        let msg = Message {
            role: "assistant".into(),
            content: parts,
            ..Default::default()
        };

        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "sum");
        assert_eq!(calls[0].args, serde_json::json!({"x":1, "y":2}));
        assert_eq!(calls[0].call_id.as_deref(), Some("abc"));
        assert!(calls[0].result.is_none());
        assert!(calls[0].remote_id.is_none());
        assert_eq!(calls[1].name, "echo");
        assert_eq!(calls[1].args, serde_json::json!({"text":"hi"}));
        assert!(calls[1].call_id.is_none());
        assert!(calls[1].result.is_none());
        assert!(calls[1].remote_id.is_none());
    }

    #[test]
    fn test_message_prune_content_keeps_visible_parts_and_is_idempotent() {
        let action = ContentPart::Action {
            name: "delegate".into(),
            payload: serde_json::json!({"agent": "planner"}),
            recipients: None,
            signature: None,
        };
        let mut msg = Message {
            role: "assistant".into(),
            content: vec![
                ContentPart::Text {
                    text: "visible text".into(),
                },
                ContentPart::ToolCall {
                    name: "sum".into(),
                    args: serde_json::json!({"x":1, "y":2}),
                    call_id: Some("call_1".into()),
                },
                ContentPart::Reasoning {
                    text: "visible thought".into(),
                },
                ContentPart::FileData {
                    file_uri: "file:///tmp/a.txt".into(),
                    mime_type: None,
                },
                action.clone(),
                ContentPart::ToolOutput {
                    name: "sum".into(),
                    output: serde_json::json!({"result": 3}),
                    is_error: None,
                    call_id: Some("call_1".into()),
                    remote_id: None,
                },
            ],
            ..Default::default()
        };

        assert_eq!(msg.prune_content(), 3);
        assert_eq!(
            msg.content,
            vec![
                ContentPart::Text {
                    text: "visible text".into(),
                },
                ContentPart::Reasoning {
                    text: "visible thought".into(),
                },
                action,
                ContentPart::Text {
                    text: "[3 items (tool calls or files) pruned due to limits]".into(),
                },
            ]
        );

        let pruned = msg.content.clone();
        assert_eq!(msg.prune_content(), 0);
        assert_eq!(msg.content, pruned);
    }

    #[test]
    fn test_message_content_deserialize_from_string() {
        // content as a plain string
        let msg: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "hello world"
        }))
        .unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.len(), 1);
        assert_eq!(
            msg.content[0],
            ContentPart::Text {
                text: "hello world".into()
            }
        );

        // content as an array still works
        let msg2: Message = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [{"type": "Text", "text": "hi"}]
        }))
        .unwrap();
        assert_eq!(msg2.content.len(), 1);
        assert_eq!(msg2.content[0], ContentPart::Text { text: "hi".into() });

        // missing content defaults to empty vec
        let msg3: Message = serde_json::from_value(serde_json::json!({
            "role": "system"
        }))
        .unwrap();
        assert!(msg3.content.is_empty());

        // null content is treated as empty content for provider compatibility
        let msg4: Message = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": null
        }))
        .unwrap();
        assert!(msg4.content.is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("abcdef"), 2);
        assert_eq!(estimate_tokens(""), 0);
    }
}
