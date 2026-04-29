//! Execution context traits for agents and tools.
//!
//! This module defines the capability traits that an Anda runtime exposes to
//! agents and tools. Context implementations provide identity, cancellation,
//! cryptographic keys, isolated storage, caching, HTTP calls, and canister
//! access without requiring each agent or tool to know how those services are
//! implemented.
//!
//! The traits are split by capability so custom runtimes can implement only one
//! coherent execution surface while still keeping the public API explicit:
//!
//! - [`BaseContext`] combines the capabilities available to agents and tools.
//! - [`AgentContext`] extends [`BaseContext`] with completion and orchestration
//!   features used by agents.
//! - [`StateFeatures`], [`KeysFeatures`], [`StoreFeatures`], [`CacheFeatures`],
//!   and [`HttpFeatures`] describe individual groups of runtime services.
//! - [`CacheStoreFeatures`] provides convenience methods for values that should
//!   be cached in memory and persisted to object storage.
//!
//! The `anda_engine` `context` module provides the default runtime
//! implementation. Other runtimes can implement these traits for specialized
//! environments such as tests, embedded workers, or alternative TEE backends.

use async_trait::async_trait;
use bytes::Bytes;
use ciborium::from_reader;
use ic_auth_types::deterministic_cbor_into_vec;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{future::Future, sync::Arc, time::Duration};

pub use anda_db_schema::Json;
pub use candid::Principal;
pub use ic_cose_types::CanisterCaller;
pub use ic_oss_types::object_store::UpdateVersion;
pub use object_store::{ObjectMeta, PutMode, PutResult, UpdateVersion as OsVersion, path::Path};
pub use tokio_util::sync::CancellationToken;

use crate::BoxError;
use crate::model::*;

/// Execution environment available to agents.
///
/// `AgentContext` combines the base runtime capabilities with model completion
/// and orchestration methods for calling local or remote agents and tools.
pub trait AgentContext: BaseContext + CompletionFeatures {
    /// Returns definitions for available local tools.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific tool names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools.
    fn tool_definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Returns definitions for tools exposed by remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint;
    /// * `names` - Optional filter for specific tool names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools.
    fn remote_tool_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[String]>,
    ) -> impl Future<Output = Result<Vec<FunctionDefinition>, BoxError>> + Send;

    /// Removes and returns resources supported by the named tool.
    fn select_tool_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> impl Future<Output = Vec<Resource>> + Send;

    /// Returns definitions for available local agents.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific agent names;
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents.
    fn agent_definitions(&self, names: Option<&[String]>) -> Vec<FunctionDefinition>;

    /// Returns definitions for agents exposed by remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint;
    /// * `names` - Optional filter for specific agent names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents.
    fn remote_agent_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[String]>,
    ) -> impl Future<Output = Result<Vec<FunctionDefinition>, BoxError>> + Send;

    /// Removes and returns resources supported by the named agent.
    fn select_agent_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> impl Future<Output = Vec<Resource>> + Send;

    /// Returns definitions for all available tools and agents, including remote ones.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific tool or agent names;
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools and agents.
    fn definitions(
        &self,
        names: Option<&[String]>,
    ) -> impl Future<Output = Vec<FunctionDefinition>> + Send;

    /// Executes a local tool call.
    ///
    /// # Arguments
    /// * `args` - Tool input arguments, [`ToolInput`].
    ///
    /// # Returns
    /// [`ToolOutput`] containing the final result.
    fn tool_call(
        &self,
        args: ToolInput<Json>,
    ) -> impl Future<Output = Result<(ToolOutput<Json>, Option<Principal>), BoxError>> + Send;

    /// Runs a local agent.
    ///
    /// # Arguments
    /// * `args` - Agent input arguments, [`AgentInput`].
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    fn agent_run(
        self,
        args: AgentInput,
    ) -> impl Future<Output = Result<(AgentOutput, Option<Principal>), BoxError>> + Send;

    /// Runs a remote agent via HTTP RPC.
    ///
    /// # Arguments
    /// * `endpoint` - Remote endpoint URL;
    /// * `args` - Agent input arguments, [`AgentInput`]. The `meta` field will be set by the runtime.
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    fn remote_agent_run(
        &self,
        endpoint: &str,
        args: AgentInput,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Core execution environment available to both agents and tools.
///
/// `BaseContext` groups state, cryptographic, storage, caching, HTTP, and ICP
/// canister capabilities behind a single trait bound.
pub trait BaseContext:
    Sized + StateFeatures + KeysFeatures + StoreFeatures + CacheFeatures + HttpFeatures + CanisterCaller
{
    /// Executes a remote tool call via HTTP RPC.
    ///
    /// # Arguments
    /// * `endpoint` - Remote endpoint URL
    /// * `args` - Tool input arguments, [`ToolInput`].
    ///
    /// # Returns
    /// [`ToolOutput`] containing the final result.
    fn remote_tool_call(
        &self,
        endpoint: &str,
        args: ToolInput<Json>,
    ) -> impl Future<Output = Result<ToolOutput<Json>, BoxError>> + Send;
}

/// Context metadata available during an agent or tool call.
pub trait StateFeatures: Sized {
    /// Returns the engine principal.
    fn engine_id(&self) -> &Principal;

    /// Returns the engine name.
    fn engine_name(&self) -> &str;

    /// Returns the verified caller principal if available.
    /// A non-anonymous principal indicates that the request was verified
    /// using ICP blockchain's signature verification algorithm.
    /// Details: <https://github.com/ldclabs/ic-auth>
    fn caller(&self) -> &Principal;

    /// Returns metadata attached to the current request.
    fn meta(&self) -> &RequestMeta;

    /// Returns the cancellation token for the current execution context.
    /// Each call level has its own token scope.
    /// For example, when an agent calls a tool, the tool receives
    /// a child token of the agent's token.
    /// Cancelling the agent token cancels all child calls, while cancelling a
    /// child token does not affect the parent context.
    fn cancellation_token(&self) -> CancellationToken;

    /// Returns the time elapsed since the context was created.
    fn time_elapsed(&self) -> Duration;
}

/// Cryptographic key operations available to agents and tools.
///
/// Runtime implementations derive isolated AES, Ed25519, and Secp256k1 keys
/// from their root key material. The active agent or tool namespace is included
/// in derivation paths so identical user-supplied paths remain isolated across
/// components.
pub trait KeysFeatures: Sized {
    /// Derives a 256-bit AES-GCM key from the given derivation path.
    fn a256gcm_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 32], BoxError>> + Send;

    /// Signs a message using Ed25519 signature scheme from the given derivation path.
    fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], BoxError>> + Send;

    /// Verifies an Ed25519 signature from the given derivation path.
    fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), BoxError>> + Send;

    /// Returns the Ed25519 public key for the given derivation path.
    fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 32], BoxError>> + Send;

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path.
    fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], BoxError>> + Send;

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path.
    fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), BoxError>> + Send;

    /// Signs a SHA-256 digest using Secp256k1 ECDSA from the given derivation path.
    /// The message will be hashed with SHA-256 before signing.
    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], BoxError>> + Send;

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path.
    fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], BoxError>> + Send;

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path.
    fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), BoxError>> + Send;

    /// Returns the compressed SEC1-encoded Secp256k1 public key for the given derivation path.
    fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 33], BoxError>> + Send;
}

/// Persistent object storage available to agents and tools.
///
/// Provides persistent storage capabilities for Agents and Tools to store and manage data.
/// All operations are asynchronous and return Result types with custom error handling.
pub trait StoreFeatures: Sized {
    /// Retrieves data from storage at the specified path.
    fn store_get(
        &self,
        path: &Path,
    ) -> impl Future<Output = Result<(bytes::Bytes, ObjectMeta), BoxError>> + Send;

    /// Lists objects in storage with optional prefix and offset filters.
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results;
    /// * `offset` - Optional path to start listing from (exclude).
    fn store_list(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> impl Future<Output = Result<Vec<ObjectMeta>, BoxError>> + Send;

    /// Stores data at the specified path with a given write mode.
    ///
    /// # Arguments
    /// * `path` - Target storage path;
    /// * `mode` - Write mode (Create, Overwrite, etc.);
    /// * `value` - Data to store as bytes.
    fn store_put(
        &self,
        path: &Path,
        mode: PutMode,
        value: bytes::Bytes,
    ) -> impl Future<Output = Result<PutResult, BoxError>> + Send;

    /// Renames a storage object if the target path doesn't exist.
    ///
    /// # Arguments
    /// * `from` - Source path;
    /// * `to` - Destination path.
    fn store_rename_if_not_exists(
        &self,
        from: &Path,
        to: &Path,
    ) -> impl Future<Output = Result<(), BoxError>> + Send;

    /// Deletes data at the specified path.
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete.
    fn store_delete(&self, path: &Path) -> impl Future<Output = Result<(), BoxError>> + Send;
}

/// Cache expiration policy for cached items.
#[derive(Debug, Clone)]
pub enum CacheExpiry {
    /// Time-to-Live: Entry expires after duration from when it was set.
    TTL(Duration),
    /// Time-to-Idle: Entry expires after duration from last access.
    TTI(Duration),
}

/// In-memory cache storage available to agents and tools.
///
/// Provides isolated in-memory cache storage with TTL/TTI expiration.
/// Cache data is ephemeral and will be lost on engine restart.
pub trait CacheFeatures: Sized {
    /// Checks if a key exists in the cache.
    fn cache_contains(&self, key: &str) -> bool;

    /// Gets a cached value by key, returns error if not found or deserialization fails.
    fn cache_get<T>(&self, key: &str) -> impl Future<Output = Result<T, BoxError>> + Send
    where
        T: DeserializeOwned;

    /// Gets a cached value or initializes it if missing.
    ///
    /// If key doesn't exist, calls init function to create value and cache it.
    fn cache_get_with<T, F>(
        &self,
        key: &str,
        init: F,
    ) -> impl Future<Output = Result<T, BoxError>> + Send
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static;

    /// Sets a value in cache with optional expiration policy.
    fn cache_set<T>(
        &self,
        key: &str,
        val: (T, Option<CacheExpiry>),
    ) -> impl Future<Output = ()> + Send
    where
        T: Sized + Serialize + Send;

    /// Sets a value in cache if key doesn't exist, returns true if set.
    fn cache_set_if_not_exists<T>(
        &self,
        key: &str,
        val: (T, Option<CacheExpiry>),
    ) -> impl Future<Output = bool> + Send
    where
        T: Sized + Serialize + Send;

    /// Deletes a cached value by key, returns true if key existed.
    fn cache_delete(&self, key: &str) -> impl Future<Output = bool> + Send;

    /// Returns an iterator over all cached items with raw value.
    fn cache_raw_iter(
        &self,
    ) -> impl Iterator<Item = (Arc<String>, Arc<(Bytes, Option<CacheExpiry>)>)>;
}

/// HTTP request capabilities available to agents and tools.
///
/// All HTTP requests are managed and scheduled by the runtime. Since agents may
/// run in WASM containers, implementations should not
/// implement HTTP requests directly.
pub trait HttpFeatures: Sized {
    /// Makes an HTTPS request.
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`;
    /// * `method` - HTTP method (GET, POST, etc.);
    /// * `headers` - Optional HTTP headers;
    /// * `body` - Optional request body (default empty).
    fn https_call(
        &self,
        url: &str,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> impl Future<Output = Result<reqwest::Response, BoxError>> + Send;

    /// Makes a signed HTTPS request with message authentication.
    ///
    /// # Arguments
    /// * `url` - Target URL;
    /// * `method` - HTTP method (GET, POST, etc.);
    /// * `message_digest` - 32-byte message digest for signing;
    /// * `headers` - Optional HTTP headers;
    /// * `body` - Optional request body (default empty).
    fn https_signed_call(
        &self,
        url: &str,
        method: http::Method,
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>,
    ) -> impl Future<Output = Result<reqwest::Response, BoxError>> + Send;

    /// Makes a signed CBOR-encoded RPC call.
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to;
    /// * `method` - RPC method name to call;
    /// * `args` - Arguments to serialize as CBOR and send with the request.
    fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        args: impl Serialize + Send,
    ) -> impl Future<Output = Result<T, BoxError>> + Send
    where
        T: DeserializeOwned;
}

#[derive(Clone, Deserialize, Serialize)]
struct CacheStoreValue<T>(T, UpdateVersion);

/// Convenience methods for values backed by both cache and object storage.
#[async_trait]
pub trait CacheStoreFeatures: StoreFeatures + CacheFeatures + Send + Sync + 'static {
    /// Initializes a cached value from storage, or creates it with `init` if missing.
    async fn cache_store_init<T, F>(&self, key: &str, init: F) -> Result<(), BoxError>
    where
        T: DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<T, BoxError>> + Send + 'static,
    {
        let p = Path::from(key);
        match self.store_get(&p).await {
            Ok((v, meta)) => {
                let val: T = from_reader(&v[..])?;
                self.cache_set(
                    key,
                    (
                        CacheStoreValue(
                            val,
                            UpdateVersion {
                                e_tag: meta.e_tag,
                                version: meta.version,
                            },
                        ),
                        None,
                    ),
                )
                .await;
                Ok(())
            }
            Err(_) => {
                let val: T = init.await?;
                let data = deterministic_cbor_into_vec(&val)?;
                let res = self.store_put(&p, PutMode::Create, data.into()).await?;
                self.cache_set(
                    key,
                    (
                        CacheStoreValue(
                            val,
                            UpdateVersion {
                                e_tag: res.e_tag,
                                version: res.version,
                            },
                        ),
                        None,
                    ),
                )
                .await;
                Ok(())
            }
        }
    }

    /// Returns a value and its storage version, loading it into cache if needed.
    async fn cache_store_get<T>(&self, key: &str) -> Result<(T, UpdateVersion), BoxError>
    where
        T: DeserializeOwned + Serialize + Send,
    {
        match self.cache_get::<CacheStoreValue<T>>(key).await {
            Ok(CacheStoreValue(val, ver)) => Ok((val, ver)),
            Err(_) => {
                // fetch from store and set in cache
                let p = Path::from(key);
                let (v, meta) = self.store_get(&p).await?;
                let val: T = from_reader(&v[..])?;
                let version = UpdateVersion {
                    e_tag: meta.e_tag,
                    version: meta.version,
                };
                self.cache_set(key, (CacheStoreValue(val, version.clone()), None))
                    .await;
                let val: T = from_reader(&v[..])?;
                Ok((val, version))
            }
        }
    }

    /// Persists a value to storage and updates the cache on success.
    ///
    /// When `version` is provided, the write uses an atomic update against that
    /// storage version. Without a version, the value is written with overwrite
    /// semantics.
    async fn cache_store_set<T>(
        &self,
        key: &str,
        val: T,
        version: Option<UpdateVersion>,
    ) -> Result<UpdateVersion, BoxError>
    where
        T: DeserializeOwned + Serialize + Send,
    {
        let data = deterministic_cbor_into_vec(&val)?;
        let p = Path::from(key);
        if let Some(ver) = version {
            // atomic update
            let res = self
                .store_put(
                    &p,
                    PutMode::Update(OsVersion {
                        e_tag: ver.e_tag.clone(),
                        version: ver.version.clone(),
                    }),
                    data.into(),
                )
                .await?;
            // we can set the cache value after atomic update succeeded
            let ver = UpdateVersion {
                e_tag: res.e_tag,
                version: res.version,
            };
            self.cache_set(key, (CacheStoreValue(val, ver.clone()), None))
                .await;
            Ok(ver)
        } else {
            let res = self.store_put(&p, PutMode::Overwrite, data.into()).await?;
            let ver = UpdateVersion {
                e_tag: res.e_tag,
                version: res.version,
            };
            self.cache_set(key, (CacheStoreValue(val, ver.clone()), None))
                .await;
            Ok(ver)
        }
    }

    /// Deletes a value from both cache and storage.
    async fn cache_store_delete(&self, key: &str) -> Result<(), BoxError> {
        let p = Path::from(key);
        self.cache_delete(key).await;
        self.store_delete(&p).await
    }
}

/// Prefixes a derivation path with the current context path.
pub fn derivation_path_with(path: &Path, derivation_path: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut dp = Vec::with_capacity(derivation_path.len() + 1);
    dp.push(path.as_ref().as_bytes().to_vec());
    dp.extend(derivation_path);
    dp
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::{
        collections::BTreeMap,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    type TestCacheValue = Arc<(Bytes, Option<CacheExpiry>)>;
    type TestCacheMap = BTreeMap<String, TestCacheValue>;

    #[derive(Default)]
    struct TestCacheStore {
        cache: Mutex<TestCacheMap>,
        store: Mutex<BTreeMap<String, (Bytes, UpdateVersion)>>,
        store_gets: AtomicUsize,
        versions: AtomicUsize,
    }

    impl TestCacheStore {
        fn put_serialized(&self, key: &str, value: Vec<u8>, version: UpdateVersion) {
            self.store
                .lock()
                .unwrap()
                .insert(key.to_string(), (value.into(), version));
        }

        fn next_version(&self) -> UpdateVersion {
            let version = self.versions.fetch_add(1, Ordering::SeqCst) + 1;
            UpdateVersion {
                e_tag: Some(format!("etag-{version}")),
                version: Some(version.to_string()),
            }
        }
    }

    impl CacheFeatures for TestCacheStore {
        fn cache_contains(&self, key: &str) -> bool {
            self.cache.lock().unwrap().contains_key(key)
        }

        async fn cache_get<T>(&self, key: &str) -> Result<T, BoxError>
        where
            T: DeserializeOwned,
        {
            let value = self
                .cache
                .lock()
                .unwrap()
                .get(key)
                .cloned()
                .ok_or_else(|| format!("key {key} not found"))?;
            from_reader(&value.0[..]).map_err(|err| err.into())
        }

        async fn cache_get_with<T, F>(&self, key: &str, init: F) -> Result<T, BoxError>
        where
            T: Sized + DeserializeOwned + Serialize + Send,
            F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
        {
            if let Some(value) = self.cache.lock().unwrap().get(key).cloned() {
                return from_reader(&value.0[..]).map_err(|err| err.into());
            }

            let (value, expiry) = init.await?;
            let data = deterministic_cbor_into_vec(&value)?;
            self.cache
                .lock()
                .unwrap()
                .insert(key.to_string(), Arc::new((data.into(), expiry)));
            Ok(value)
        }

        async fn cache_set<T>(&self, key: &str, val: (T, Option<CacheExpiry>))
        where
            T: Sized + Serialize + Send,
        {
            let data = deterministic_cbor_into_vec(&val.0).unwrap();
            self.cache
                .lock()
                .unwrap()
                .insert(key.to_string(), Arc::new((data.into(), val.1)));
        }

        async fn cache_set_if_not_exists<T>(&self, key: &str, val: (T, Option<CacheExpiry>)) -> bool
        where
            T: Sized + Serialize + Send,
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.contains_key(key) {
                return false;
            }

            let data = deterministic_cbor_into_vec(&val.0).unwrap();
            cache.insert(key.to_string(), Arc::new((data.into(), val.1)));
            true
        }

        async fn cache_delete(&self, key: &str) -> bool {
            self.cache.lock().unwrap().remove(key).is_some()
        }

        fn cache_raw_iter(
            &self,
        ) -> impl Iterator<Item = (Arc<String>, Arc<(Bytes, Option<CacheExpiry>)>)> {
            self.cache
                .lock()
                .unwrap()
                .iter()
                .map(|(key, value)| (Arc::new(key.clone()), value.clone()))
                .collect::<Vec<_>>()
                .into_iter()
        }
    }

    impl StoreFeatures for TestCacheStore {
        async fn store_get(&self, path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
            self.store_gets.fetch_add(1, Ordering::SeqCst);
            let (value, version) = self
                .store
                .lock()
                .unwrap()
                .get(path.as_ref())
                .cloned()
                .ok_or_else(|| format!("path {path} not found"))?;

            Ok((
                value.clone(),
                ObjectMeta {
                    location: path.clone(),
                    last_modified: chrono::Utc::now(),
                    size: value.len() as u64,
                    e_tag: version.e_tag,
                    version: version.version,
                },
            ))
        }

        async fn store_list(
            &self,
            _prefix: Option<&Path>,
            _offset: &Path,
        ) -> Result<Vec<ObjectMeta>, BoxError> {
            Ok(Vec::new())
        }

        async fn store_put(
            &self,
            path: &Path,
            mode: PutMode,
            value: bytes::Bytes,
        ) -> Result<PutResult, BoxError> {
            let key = path.as_ref().to_string();
            let mut store = self.store.lock().unwrap();
            match mode {
                PutMode::Create if store.contains_key(&key) => {
                    return Err(format!("path {path} already exists").into());
                }
                PutMode::Update(expected) => {
                    let Some((_, current)) = store.get(&key) else {
                        return Err(format!("path {path} not found").into());
                    };
                    if current.e_tag != expected.e_tag || current.version != expected.version {
                        return Err(format!("path {path} version mismatch").into());
                    }
                }
                _ => {}
            }

            let version = self.next_version();
            store.insert(key, (value, version.clone()));
            Ok(PutResult {
                e_tag: version.e_tag,
                version: version.version,
            })
        }

        async fn store_rename_if_not_exists(&self, from: &Path, to: &Path) -> Result<(), BoxError> {
            let mut store = self.store.lock().unwrap();
            let to = to.as_ref().to_string();
            if store.contains_key(&to) {
                return Err(format!("path {to} already exists").into());
            }
            let value = store
                .remove(from.as_ref())
                .ok_or_else(|| format!("path {from} not found"))?;
            store.insert(to, value);
            Ok(())
        }

        async fn store_delete(&self, path: &Path) -> Result<(), BoxError> {
            self.store.lock().unwrap().remove(path.as_ref());
            Ok(())
        }
    }

    impl CacheStoreFeatures for TestCacheStore {}

    #[test]
    fn cache_store_get_populates_cache_without_second_store_read() {
        let ctx = TestCacheStore::default();
        let stored_version = UpdateVersion {
            e_tag: Some("etag-stored".to_string()),
            version: Some("1".to_string()),
        };
        let data = deterministic_cbor_into_vec(&123_u32).unwrap();
        ctx.put_serialized("answer", data, stored_version.clone());

        let (value, version) = block_on(ctx.cache_store_get::<u32>("answer")).unwrap();
        assert_eq!(value, 123);
        assert_eq!(version.e_tag, stored_version.e_tag);
        assert_eq!(version.version, stored_version.version);
        assert_eq!(ctx.store_gets.load(Ordering::SeqCst), 1);

        let (value, _) = block_on(ctx.cache_store_get::<u32>("answer")).unwrap();
        assert_eq!(value, 123);
        assert_eq!(ctx.store_gets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_store_set_overwrite_updates_cache() {
        let ctx = TestCacheStore::default();

        let version = block_on(ctx.cache_store_set("answer", 42_u32, None)).unwrap();
        assert_eq!(ctx.store_gets.load(Ordering::SeqCst), 0);

        let (value, cached_version) = block_on(ctx.cache_store_get::<u32>("answer")).unwrap();
        assert_eq!(value, 42);
        assert_eq!(cached_version.e_tag, version.e_tag);
        assert_eq!(cached_version.version, version.version);
        assert_eq!(ctx.store_gets.load(Ordering::SeqCst), 0);
    }
}
