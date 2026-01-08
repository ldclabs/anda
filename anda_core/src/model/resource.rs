use candid::Principal;
use chrono::prelude::*;
use ic_auth_types::{ByteArrayB64, ByteBufB64};
use ic_cose_types::cose::sha3_256;
use serde::Serialize;

use anda_db_schema::{Json, Map};

pub use anda_db_schema::Resource;

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

    /// MIME type, https://developer.mozilla.org/zh-CN/docs/Web/HTTP/MIME_types/Common_types
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<&'a String>,

    /// The binary data of this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<&'a ByteBufB64>,

    /// The size of the resource in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<usize>,

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

/// Extracts resources with the given tags from the list of resources.
pub fn select_resources(resources: &mut Vec<Resource>, tags: &[String]) -> Vec<Resource> {
    if tags.is_empty() {
        return Vec::new();
    }

    if tags.first().map(|s| s.as_str()) == Some("*") {
        return std::mem::take(resources);
    }

    // nightly feature:
    // {
    //     let res: Vec<Resource> = resources
    //         .extract_if(.., |r| tags.contains(&r.tag.as_str()))
    //         .collect();
    //     if res.is_empty() { None } else { Some(res) }
    // }

    {
        let mut res = Vec::new();
        let mut i = 0;

        while i < resources.len() {
            if resources[i].tags.iter().any(|tag| tags.contains(tag)) {
                res.push(resources.remove(i));
            } else {
                i += 1;
            }
        }
        res
    }
}
