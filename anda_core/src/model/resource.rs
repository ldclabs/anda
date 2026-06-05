use candid::Principal;
use chrono::prelude::*;
use ic_auth_types::{ByteArrayB64, ByteBufB64};
use ic_cose_types::cose::sha3_256;
use serde::Serialize;
use std::collections::BTreeSet;

use anda_db_schema::{Json, Map};

pub use anda_db_schema::Resource;

/// Borrowed view of a [`Resource`] suitable for serialization.
#[derive(Debug, Serialize)]
pub struct ResourceRef<'a> {
    /// The unique identifier for this resource in the Anda DB collection.
    pub _id: u64,

    /// A list of tags that identifies the type of this resource.
    /// "text", "image", "audio", "video", etc.
    pub tags: &'a [String],

    /// A human-readable name for this resource.
    pub name: &'a String,

    /// A description of what this resource represents.
    /// This can be used by clients to improve the LLM's understanding of available resources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'a String>,

    /// The URI of this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<&'a String>,

    /// MIME type. See <https://developer.mozilla.org/en-US/docs/Web/HTTP/MIME_types/Common_types>.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<&'a String>,

    /// The binary data of this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<&'a ByteBufB64>,

    /// The size of the resource in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,

    /// The SHA3-256 hash of the resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<&'a ByteArrayB64<32>>,

    /// Metadata associated with this resource.
    /// This can include additional information such as creation date, author, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<&'a Map<String, Json>>,
}

impl<'a> From<&'a Resource> for ResourceRef<'a> {
    fn from(resource: &'a Resource) -> Self {
        Self {
            _id: resource._id,
            tags: &resource.tags,
            name: &resource.name,
            description: resource.description.as_ref(),
            uri: resource.uri.as_ref(),
            mime_type: resource.mime_type.as_ref(),
            blob: resource.blob.as_ref(),
            size: resource.size,
            hash: resource.hash.as_ref(),
            metadata: resource.metadata.as_ref(),
        }
    }
}

/// Updates resource metadata before persistence.
///
/// Binary resources receive a SHA3-256 hash. New resources (`_id == 0`) also
/// receive `user` and `created_at` metadata.
pub fn update_resources(user: &Principal, resources: Vec<Resource>) -> Vec<Resource> {
    let user = user.to_string();
    let utc = Utc::now().to_rfc3339();
    resources
        .into_iter()
        .map(|mut r| {
            if let Some(blob) = &r.blob {
                r.hash = Some(sha3_256(blob).into());
            }

            if r._id == 0 {
                let meta = r.metadata.get_or_insert_with(Map::new);
                meta.insert("user".to_string(), user.clone().into());
                meta.insert("created_at".to_string(), utc.clone().into());
            }
            r
        })
        .collect()
}

/// Removes and returns resources matching any of the supported tags.
///
/// The order of selected resources and remaining resources is preserved. If the
/// first tag is `*`, all resources are selected.
pub fn select_resources(resources: &mut Vec<Resource>, tags: &[String]) -> Vec<Resource> {
    if tags.is_empty() {
        return Vec::new();
    }

    if tags.first().map(|s| s.as_str()) == Some("*") {
        return std::mem::take(resources);
    }

    let tag_set: BTreeSet<&str> = tags.iter().map(String::as_str).collect();
    let mut selected = Vec::new();
    let mut remaining = Vec::with_capacity(resources.len());

    for resource in std::mem::take(resources) {
        if resource
            .tags
            .iter()
            .any(|tag| tag_set.contains(tag.as_str()))
        {
            selected.push(resource);
        } else {
            remaining.push(resource);
        }
    }

    *resources = remaining;
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(id: u64, tags: &[&str]) -> Resource {
        Resource {
            _id: id,
            name: format!("resource-{id}"),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        }
    }

    fn metadata(key: &str, value: &str) -> Map<String, Json> {
        let mut map = Map::new();
        map.insert(key.to_string(), Json::from(value));
        map
    }

    #[test]
    fn select_resources_preserves_selected_and_remaining_order() {
        let mut resources = vec![
            resource(1, &["text"]),
            resource(2, &["image"]),
            resource(3, &["text", "code"]),
            resource(4, &["audio"]),
        ];
        let tags = vec!["text".to_string(), "audio".to_string()];

        let selected = select_resources(&mut resources, &tags);

        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![1, 3, 4]
        );
        assert_eq!(
            resources
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn select_resources_wildcard_takes_all_resources() {
        let mut resources = vec![resource(1, &["text"]), resource(2, &["image"])];
        let tags = vec!["*".to_string()];

        let selected = select_resources(&mut resources, &tags);

        assert_eq!(
            selected
                .iter()
                .map(|resource| resource._id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(resources.is_empty());
    }

    #[test]
    fn update_resources_adds_metadata_and_hashes_binary_resources() {
        let user = Principal::from_text("aaaaa-aa").unwrap();
        let existing = Resource {
            _id: 7,
            name: "existing".to_string(),
            tags: vec!["text".to_string()],
            metadata: Some(metadata("kept", "yes")),
            ..Default::default()
        };
        let new_binary = Resource {
            _id: 0,
            name: "new".to_string(),
            tags: vec!["image".to_string()],
            blob: Some(ByteBufB64(vec![1, 2, 3])),
            ..Default::default()
        };

        let updated = update_resources(&user, vec![existing, new_binary]);

        assert_eq!(updated[0]._id, 7);
        assert_eq!(
            updated[0]
                .metadata
                .as_ref()
                .and_then(|meta| meta.get("kept"))
                .and_then(|value| value.as_str()),
            Some("yes")
        );
        assert!(
            updated[0]
                .metadata
                .as_ref()
                .and_then(|meta| meta.get("created_at"))
                .is_none()
        );

        let new_meta = updated[1].metadata.as_ref().unwrap();
        assert_eq!(
            new_meta.get("user").and_then(|value| value.as_str()),
            Some(user.to_string().as_str())
        );
        assert!(
            new_meta
                .get("created_at")
                .and_then(|value| value.as_str())
                .is_some_and(|value| DateTime::parse_from_rfc3339(value).is_ok())
        );
        assert_eq!(updated[1].hash, Some(sha3_256(&[1, 2, 3]).into()));
    }

    #[test]
    fn resource_ref_serializes_optional_metadata_without_owned_blob_copy() {
        let hash: ByteArrayB64<32> = [9u8; 32].into();
        let resource = Resource {
            _id: 42,
            tags: vec!["text".to_string()],
            name: "doc".to_string(),
            description: Some("description".to_string()),
            uri: Some("file://doc.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            blob: Some(ByteBufB64(b"hello".to_vec())),
            size: Some(5),
            hash: Some(hash.clone()),
            metadata: Some(metadata("source", "unit")),
        };

        let view = ResourceRef::from(&resource);
        assert_eq!(view._id, 42);
        assert_eq!(view.tags, &["text".to_string()]);
        assert_eq!(view.description.map(String::as_str), Some("description"));
        assert_eq!(view.uri.map(String::as_str), Some("file://doc.txt"));
        assert_eq!(view.mime_type.map(String::as_str), Some("text/plain"));
        assert_eq!(view.blob.unwrap().0.as_slice(), b"hello");
        assert_eq!(view.size, Some(5));
        assert_eq!(view.hash, Some(&hash));
        assert_eq!(
            view.metadata
                .unwrap()
                .get("source")
                .and_then(|value| value.as_str()),
            Some("unit")
        );

        let json = serde_json::to_value(view).unwrap();
        assert_eq!(json["_id"], 42);
        assert_eq!(json["name"], "doc");
        assert_eq!(json["size"], 5);
    }
}
