//! In-memory caching system for AI Agent components.
//!
//! This module provides a thread-safe, in-memory LRU cache implementation with expiration policies
//! for storing serialized data. The cache is primarily used by AI Agents and Tools to store
//! frequently accessed data with configurable expiration policies.
//!
//! # Key Features
//! - LRU (Least Recently Used) eviction policy;
//! - Configurable maximum capacity;
//! - Time-to-Idle (TTI) and Time-to-Live (TTL) expiration policies;
//! - Thread-safe operations;
//! - Automatic serialization/deserialization using CBOR format.
//!
//! # Usage
//! The cache is isolated per agent/tool using path-based namespacing. Each agent/tool has its own
//! isolated cache storage within the shared cache instance.
//!
//! # Performance Characteristics
//! - O(1) time complexity for get/set operations;
//! - Memory usage scales with cache capacity and item sizes;
//! - Automatic eviction of expired items.
//!
//! # Limitations
//! - Data is not persisted across system restarts;
//! - Maximum cache size is limited by available memory;
//! - Serialization/deserialization overhead for large objects.

use anda_core::BoxError;
use anda_core::context::CacheExpiry;
use bytes::Bytes;
use ciborium::from_reader;
use ic_auth_types::deterministic_cbor_into_vec;
use moka::{future::Cache, policy::Expiry};
use object_store::path::Path;
use serde::{Serialize, de::DeserializeOwned};
use std::collections::BTreeSet;
use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub(crate) struct CacheService {
    #[allow(clippy::type_complexity)]
    cache_store: HashMap<Path, Cache<String, Arc<(Bytes, Option<CacheExpiry>)>>>,
}

/// CacheService provides an in-memory LRU cache with expiration for AI Agent system's agents and tools.
///
/// In the Anda Engine implementation, the `path` parameter is derived from agents' or tools' `name`,
/// ensuring that each agent or tool has isolated cache storage.
///
/// Note: Data is cached only in memory and will be lost upon system restart.
/// For persistent storage, use `StoreFeatures`.
impl CacheService {
    /// Creates a new CacheService instance with specified maximum capacity.
    ///
    /// # Arguments
    /// * `max_capacity` - Maximum number of items the cache can hold (u64);
    /// * `names` - Set of base paths for cache namespacing.
    ///
    /// # Default Behavior
    /// - Maximum time-to-idle (TTI): 7 days;
    /// - Uses custom expiration policy based on CacheExpiry.
    pub fn new(max_capacity: u64, names: BTreeSet<Path>) -> Self {
        Self {
            cache_store: names
                .into_iter()
                .map(|k| {
                    (
                        k,
                        Cache::builder()
                            .max_capacity(max_capacity)
                            // max TTI is 7 days
                            .time_to_idle(Duration::from_secs(3600 * 24 * 7))
                            .expire_after(CacheServiceExpiry)
                            .build(),
                    )
                })
                .collect(),
        }
    }
}

impl CacheService {
    /// Checks if a key exists in the cache.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key. It is used to isolate cache storage for each agent/tool.
    /// * It will panic if the cache is not created for the given path.
    /// * `key` - The key to check.
    ///
    /// # Returns
    /// `true` if key exists, `false` otherwise.
    pub fn contains(&self, path: &Path, key: &str) -> bool {
        self.cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .contains_key(key)
    }

    /// Retrieves a cached value by key.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key;
    /// * `key` - The key to retrieve.
    ///
    /// # Returns
    /// Result containing deserialized value if successful, error otherwise.
    pub async fn get<T>(&self, path: &Path, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        if let Some(val) = self
            .cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .get(key)
            .await
        {
            from_reader(&val.0[..]).map_err(|err| err.into())
        } else {
            Err(format!("key {} not found", key).into())
        }
    }

    /// Gets a cached value or initializes it if missing.
    ///
    /// If key doesn't exist, calls init function to create value and cache it.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key;
    /// * `key` - The key to retrieve or initialize;
    /// * `init` - Async function that returns the value and optional expiry.
    ///
    /// # Returns
    /// Result containing deserialized value if successful, error otherwise.
    pub async fn get_with<T, F>(&self, path: &Path, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        futures_util::pin_mut!(init);
        match self
            .cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .try_get_with_by_ref(key, async move {
                match init.await {
                    Ok((val, expiry)) => {
                        let data = deterministic_cbor_into_vec(&val)?;
                        Ok(Arc::new((data.into(), expiry)))
                    }
                    Err(e) => Err(e),
                }
            })
            .await
        {
            Ok(val) => from_reader(&val.0[..]).map_err(|e| e.into()),
            Err(err) => Err(format!("key {} init failed: {}", key, err).into()),
        }
    }

    /// Sets a value in cache with optional expiration policy.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key;
    /// * `key` - The key to set;
    /// * `value` - Tuple containing value and optional expiry policy.
    pub async fn set<T>(&self, path: &Path, key: &str, value: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        let data = deterministic_cbor_into_vec(&value.0).unwrap();
        self.cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .insert(key.to_string(), Arc::new((data.into(), value.1)))
            .await;
    }

    /// Sets a value in cache if key doesn't exist.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key;
    /// * `key` - The key to set;
    /// * `value` - Tuple containing value and optional expiry policy.
    pub async fn set_if_not_exists<T>(
        &self,
        path: &Path,
        key: &str,
        value: (T, Option<CacheExpiry>),
    ) -> bool
    where
        T: Sized + Serialize + Send,
    {
        let data = deterministic_cbor_into_vec(&value.0).unwrap();
        let entry = self
            .cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .entry_by_ref(key)
            .or_optionally_insert_with(async { Some(Arc::new((data.into(), value.1))) })
            .await;
        entry.map(|v| v.is_fresh()).unwrap_or(false)
    }

    /// Deletes a cached value by key.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key;
    /// * `key` - The key to delete.
    ///
    /// # Returns
    /// `true` if key existed and was deleted, `false` otherwise.
    pub async fn delete(&self, path: &Path, key: &str) -> bool {
        self.cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .remove(key)
            .await
            .is_some()
    }

    /// Returns an iterator over the cache entries for a given path.
    pub fn iter(
        &self,
        path: &Path,
    ) -> impl Iterator<Item = (Arc<String>, Arc<(Bytes, Option<CacheExpiry>)>)> {
        self.cache_store
            .get(path)
            .expect("CacheService: cache not found")
            .iter()
    }
}

struct CacheServiceExpiry;

impl Expiry<String, Arc<(Bytes, Option<CacheExpiry>)>> for CacheServiceExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &Arc<(Bytes, Option<CacheExpiry>)>,
        _created_at: Instant,
    ) -> Option<Duration> {
        match value.1 {
            Some(CacheExpiry::TTL(du)) => Some(du),
            Some(CacheExpiry::TTI(du)) => Some(du),
            None => None,
        }
    }

    fn expire_after_read(
        &self,
        _key: &String,
        value: &Arc<(Bytes, Option<CacheExpiry>)>,
        _read_at: Instant,
        duration_until_expiry: Option<Duration>,
        _last_modified_at: Instant,
    ) -> Option<Duration> {
        match value.1 {
            Some(CacheExpiry::TTL(_)) => duration_until_expiry,
            Some(CacheExpiry::TTI(du)) => Some(du),
            None => None,
        }
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &Arc<(Bytes, Option<CacheExpiry>)>,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        match value.1 {
            Some(CacheExpiry::TTL(du)) => Some(du),
            Some(CacheExpiry::TTI(du)) => Some(du),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
    struct Profile {
        name: String,
        age: Option<u8>,
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_cache_service() {
        let path1 = Path::from("path1");
        let path2 = Path::from("path2");
        let cache = CacheService::new(100, BTreeSet::from([path1.clone(), path2.clone()]));
        assert!(!cache.contains(&path1, "key"));
        assert!(cache.get::<Profile>(&path2, "key").await.is_err());

        let profile = Profile {
            name: "Anda".to_string(),
            age: Some(18),
        };
        let p1 = profile.clone();
        let res = cache
            .get_with(&path1, "key", async move {
                Ok((p1, Some(CacheExpiry::TTI(Duration::from_secs(10)))))
            })
            .await
            .unwrap();
        assert_eq!(res, profile);

        let res = cache.get::<Profile>(&path1, "key").await.unwrap();
        assert_eq!(res, profile);
        assert!(cache.get::<Profile>(&path2, "key").await.is_err());

        cache
            .set(
                &path1,
                "key",
                (
                    Profile {
                        name: "Anda".to_string(),
                        age: Some(19),
                    },
                    Some(CacheExpiry::TTI(Duration::from_secs(10))),
                ),
            )
            .await;
        let res = cache.get::<Profile>(&path1, "key").await.unwrap();
        assert_ne!(res, profile);
        assert_eq!(res.age, Some(19));

        cache.delete(&path1, "key").await;
        assert!(cache.get::<Profile>(&path1, "key").await.is_err());
    }
}
