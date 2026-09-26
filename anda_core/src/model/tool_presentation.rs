//! A portable model view of a tool result, independent of its full audit payload.

use super::{ByteBufB64, Json};
use serde::{Deserialize, Serialize};

/// Inline media belonging to a tool result, never a new user message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolMedia {
    /// Media type of the decoded bytes.
    pub mime_type: String,
    /// Inline media bytes, encoded as base64 in JSON.
    pub data: ByteBufB64,
}

/// Bounded model-facing result. Serialized inside `ContentPart::ToolOutput.output`
/// with an explicit type tag so persisted histories retain the tool boundary.
/// Provider adapters project supported media and describe unsupported media as text.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub struct ToolPresentation {
    /// Text chosen for the model (without application-only metadata).
    pub text: String,
    /// Inline images/audio associated with the same tool call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<ToolMedia>,
}

impl ToolPresentation {
    /// Reads only explicitly tagged presentations; ordinary JSON stays unchanged.
    pub fn from_output(output: &Json) -> Option<Self> {
        if output.get("type").and_then(Json::as_str) != Some("ToolPresentation") {
            return None;
        }
        serde_json::from_value(output.clone()).ok()
    }

    /// Encodes this view for provider-neutral persisted tool history.
    pub fn into_output(self) -> Json {
        // This type contains no fallible JSON map keys or non-finite numbers.
        serde_json::to_value(self).expect("tool presentation is JSON serializable")
    }

    /// Text fallback for providers that cannot receive media in tool responses.
    pub fn text_fallback(&self) -> String {
        let mut text = self.text.clone();
        for media in &self.media {
            text.push_str(&format!(
                "\n[{} tool media omitted: unsupported by this model API]",
                media.mime_type
            ));
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presentation_round_trips_and_old_results_stay_ordinary_json() {
        let view = ToolPresentation {
            text: "hello".into(),
            media: vec![ToolMedia {
                mime_type: "audio/wav".into(),
                data: ByteBufB64::from(vec![1, 2, 3]),
            }],
        };
        assert_eq!(
            ToolPresentation::from_output(&view.clone().into_output()),
            Some(view.clone())
        );
        assert!(view.text_fallback().contains("audio/wav"));
        assert!(!view.text_fallback().contains("AQID"));
        assert!(ToolPresentation::from_output(&serde_json::json!({"text":"ordinary"})).is_none());
        let mut bytes = Vec::new();
        cbor2::to_writer(&view, &mut bytes).unwrap();
        let decoded: ToolPresentation = cbor2::from_slice(&bytes).unwrap();
        assert_eq!(decoded, view);
    }
}
