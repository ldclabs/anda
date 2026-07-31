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
use cbor2::{from_slice, to_canonical_vec};
use moka::{future::Cache, policy::Expiry};
use object_store::path::Path;
use parking_lot::RwLock;
use serde::{Serialize, de::DeserializeOwned};
use std::collections::BTreeSet;
use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

type CacheValue = Arc<(Bytes, Option<CacheExpiry>)>;
type NamespaceCache = Cache<String, CacheValue>;

#[derive(Debug)]
pub(crate) struct CacheService {
    max_capacity: u64,
    cache_store: RwLock<HashMap<Path, NamespaceCache>>,
}

/// CacheService provides an in-memory LRU cache with expiration for AI Agent system's agents and tools.
///
/// In the Anda Engine implementation, the `path` parameter is derived from agents' or tools' `name`,
/// ensuring that each agent or tool has isolated cache storage.
///
/// Note: Data is cached only in memory and will be lost upon system restart.
/// For persistent storage, use `StoreFeatures`.
impl CacheService {
    /// Returns the namespace cache for `path`, creating it on first use.
    ///
    /// Namespaces cannot be fully enumerated up front: provider-backed tools (MCP) are
    /// discovered after the engine is built, and subagents can be registered at runtime.
    /// Requiring pre-registration made every cache operation in those contexts fail silently
    /// — `get_with` returned an error *without running the initializer*, and
    /// `set_if_not_exists` returned `false`, which lease-style callers read as "already
    /// held". Creating on demand keeps the per-agent/per-tool isolation while removing that
    /// whole failure class.
    fn cache(&self, path: &Path) -> NamespaceCache {
        // `moka::future::Cache` is a cheap handle over shared state, so cloning it out of
        // the lock lets callers await without holding the lock.
        if let Some(cache) = self.cache_store.read().get(path) {
            return cache.clone();
        }

        let mut store = self.cache_store.write();
        store
            .entry(path.clone())
            .or_insert_with(|| Self::new_namespace(self.max_capacity))
            .clone()
    }

    fn new_namespace(max_capacity: u64) -> NamespaceCache {
        Cache::builder()
            .max_capacity(max_capacity)
            // max TTI is 7 days
            .time_to_idle(Duration::from_secs(3600 * 24 * 7))
            .expire_after(CacheServiceExpiry)
            .build()
    }

    /// Creates a new CacheService instance with specified maximum capacity.
    ///
    /// # Arguments
    /// * `max_capacity` - Maximum number of items the cache can hold (u64);
    /// * `names` - Base paths to pre-create. Any other namespace is created on first use.
    ///
    /// # Default Behavior
    /// - Maximum time-to-idle (TTI): 7 days;
    /// - Uses custom expiration policy based on CacheExpiry.
    pub fn new(max_capacity: u64, names: BTreeSet<Path>) -> Self {
        Self {
            max_capacity,
            cache_store: RwLock::new(
                names
                    .into_iter()
                    .map(|k| (k, Self::new_namespace(max_capacity)))
                    .collect(),
            ),
        }
    }
}

impl CacheService {
    /// Checks if a key exists in the cache.
    ///
    /// # Arguments
    /// * `path` - The namespace for the key. It is used to isolate cache storage for each agent/tool.
    /// * `key` - The key to check.
    ///
    /// # Returns
    /// `true` if key exists, `false` otherwise, including when the cache namespace is missing.
    pub fn contains(&self, path: &Path, key: &str) -> bool {
        self.cache(path).contains_key(key)
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
        match self.cache(path).get(key).await {
            Some(val) => from_slice(&val.0[..]).map_err(|err| err.into()),
            None => Err(format!("key {} not found", key).into()),
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
        let cache = self.cache(path);
        futures_util::pin_mut!(init);
        match cache
            .try_get_with_by_ref(key, async move {
                match init.await {
                    Ok((val, expiry)) => {
                        let data = to_canonical_vec(&val)?;
                        Ok(Arc::new((data.into(), expiry)))
                    }
                    Err(e) => Err(e),
                }
            })
            .await
        {
            Ok(val) => from_slice(&val.0[..]).map_err(|e| e.into()),
            // Preserve the underlying error (and its `source()` chain / retryable
            // / status signals) instead of flattening it to a string.
            Err(err) => Err(Box::new(CacheInitError {
                key: key.to_string(),
                source: err,
            })),
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
        let cache = self.cache(path);
        let data = match to_canonical_vec(&value.0) {
            Ok(data) => data,
            Err(err) => {
                log::error!("CacheService failed to serialize value, key: {key}, error: {err}");
                return;
            }
        };
        cache
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
        let cache = self.cache(path);
        let data = match to_canonical_vec(&value.0) {
            Ok(data) => data,
            Err(err) => {
                log::error!("CacheService failed to serialize value, key: {key}, error: {err}");
                return false;
            }
        };
        let entry = cache
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
        self.cache(path).remove(key).await.is_some()
    }

    /// Returns an iterator over the cache entries for a given path.
    ///
    /// Moka's iterator borrows the namespace handle, which is created on demand and owned by
    /// this call, so entries are collected eagerly. Namespaces are per-agent/per-tool and
    /// capacity-bounded, so the snapshot stays small.
    pub fn iter(
        &self,
        path: &Path,
    ) -> impl Iterator<Item = (Arc<String>, Arc<(Bytes, Option<CacheExpiry>)>)> {
        self.cache(path).iter().collect::<Vec<_>>().into_iter()
    }
}

/// Error returned when a [`CacheService::get_with`] initializer fails.
///
/// Wrapping the initializer error preserves its `source()` chain and any
/// downcastable status/retryable signals that a plain string would lose.
#[derive(Debug)]
struct CacheInitError {
    key: String,
    source: Arc<BoxError>,
}

impl std::fmt::Display for CacheInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "key {} init failed: {}", self.key, self.source)
    }
}

impl std::error::Error for CacheInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&**self.source)
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

    #[tokio::test(flavor = "current_thread")]
    async fn unregistered_namespace_is_created_on_demand() {
        // Namespaces cannot be enumerated up front: MCP tools are discovered after the
        // engine is built, and subagents can be registered at runtime. An unregistered path
        // must behave like any other namespace rather than failing every operation — in
        // particular `get_with` must run its initializer, and `set_if_not_exists` must
        // report `true` on the first insert (lease callers read `false` as "already held").
        let path1 = Path::from("path1");
        let path2 = Path::from("path2");
        let cache = CacheService::new(100, BTreeSet::from([path1.clone()]));
        let profile = Profile {
            name: "Anda".to_string(),
            age: Some(18),
        };

        assert!(!cache.contains(&path2, "key"));
        assert!(cache.get::<Profile>(&path2, "key").await.is_err());

        let p1 = profile.clone();
        let initialized = cache
            .get_with(&path2, "key", async move { Ok((p1, None)) })
            .await
            .expect("the initializer must run on an on-demand namespace");
        assert_eq!(initialized, profile);
        assert!(cache.contains(&path2, "key"));

        assert!(cache.delete(&path2, "key").await);
        assert!(
            cache
                .set_if_not_exists(&path2, "key", (profile.clone(), None))
                .await,
            "the first insert must claim the key"
        );
        assert!(
            !cache
                .set_if_not_exists(&path2, "key", (profile.clone(), None))
                .await,
            "a second insert must observe the existing key"
        );
        assert_eq!(cache.iter(&path2).count(), 1);

        // Namespaces stay isolated: writing to `path2` does not populate `path1`.
        assert!(cache.get::<Profile>(&path1, "key").await.is_err());
    }
}
