//! Execution context traits for agents and tools.
//!
//! This module defines the capability traits that an Anda runtime exposes to
//! agents and tools. Context implementations provide identity, cancellation,
//! cryptographic keys, isolated storage, caching, and HTTP calls without
//! requiring each agent or tool to know how those services are implemented.
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

use cbor2::{from_slice, to_canonical_vec};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{future::Future, time::Duration};

pub use anda_db_schema::Json;
pub use candid::Principal;
pub use ic_oss_types::object_store::UpdateVersion;
pub use object_store::{ObjectMeta, PutMode, PutResult, UpdateVersion as OsVersion, path::Path};
pub use tokio_util::sync::CancellationToken;

use crate::model::*;
use crate::{BoxError, path_lowercase};

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
/// `BaseContext` groups state, cryptographic, storage, caching, and HTTP
/// capabilities behind a single trait bound. Canister access is intentionally
/// not part of this bound: runtimes that need it implement
/// [`CanisterCaller`](ic_cose_types::CanisterCaller) separately on their context
/// type.
pub trait BaseContext:
    Sized + StateFeatures + KeysFeatures + StoreFeatures + CacheFeatures + HttpFeatures
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

    /// Signs a message using Secp256k1 ECDSA from the given derivation path.
    /// The message will be hashed with SHA-256 before signing.
    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], BoxError>> + Send;

    /// Signs a 32-byte digest using Secp256k1 ECDSA from the given derivation path.
    fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], BoxError>> + Send;

    /// Verifies a Secp256k1 ECDSA signature against a 32-byte digest.
    /// Use SHA-256 of the message for signatures from `secp256k1_sign_message_ecdsa`.
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
    ///
    /// Missing objects should return [`object_store::Error::NotFound`], directly
    /// or in the error's source chain, so [`CacheStoreFeatures`] can distinguish
    /// absence from a failed read.
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

fn is_store_not_found(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if matches!(
            error.downcast_ref::<object_store::Error>(),
            Some(object_store::Error::NotFound { .. })
        ) {
            return true;
        }
        match error.source() {
            Some(source) => error = source,
            None => return false,
        }
    }
}

/// Convenience methods for values backed by both cache and object storage.
///
/// Keys are namespace-relative and converted to object-store paths, then
/// ASCII-lowercased with [`path_lowercase`]. The same path is used for storage and as the
/// cache key. Direct cache access to these entries must use that canonical key.
///
/// # Consistency
///
/// These helpers coordinate a cache and a store as two separate operations and
/// do **not** provide cross-task linearizability, even with versioned writes.
/// An in-flight cache fill can overwrite a newer cached value or repopulate an
/// entry after deletion. These helpers set no expiry, so stale entries may
/// persist until eviction or an explicit refresh. Callers needing consistency
/// under concurrency must serialize the entire read/fill, write, and delete
/// operations for a key, or use [`StoreFeatures`] directly with versioned writes.
pub trait CacheStoreFeatures: StoreFeatures + CacheFeatures + Send + Sync + 'static {
    /// Initializes a cached value from storage, or creates it with `init` if missing.
    /// Read errors other than [`object_store::Error::NotFound`] are propagated
    /// without running `init`.
    fn cache_store_init<T, F>(
        &self,
        key: &str,
        init: F,
    ) -> impl Future<Output = Result<(), BoxError>> + Send
    where
        T: DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<T, BoxError>> + Send + 'static,
    {
        async move {
            let p = path_lowercase(&Path::from(key));
            let (val, version) = match self.store_get(&p).await {
                Ok((v, meta)) => {
                    let val: T = from_slice(&v[..])?;
                    (
                        val,
                        UpdateVersion {
                            e_tag: meta.e_tag,
                            version: meta.version,
                        },
                    )
                }
                Err(error) if is_store_not_found(error.as_ref()) => {
                    let val: T = init.await?;
                    let data = to_canonical_vec(&val)?;
                    let res = self.store_put(&p, PutMode::Create, data.into()).await?;
                    (
                        val,
                        UpdateVersion {
                            e_tag: res.e_tag,
                            version: res.version,
                        },
                    )
                }
                Err(error) => return Err(error),
            };
            self.cache_set(p.as_ref(), (CacheStoreValue(val, version), None))
                .await;
            Ok(())
        }
    }

    /// Returns a value and its storage version, loading it into cache if needed.
    fn cache_store_get<T>(
        &self,
        key: &str,
    ) -> impl Future<Output = Result<(T, UpdateVersion), BoxError>> + Send
    where
        T: DeserializeOwned + Serialize + Send + Sync,
    {
        async move {
            let p = path_lowercase(&Path::from(key));
            let key = p.as_ref();
            if let Ok(CacheStoreValue(val, ver)) = self.cache_get::<CacheStoreValue<T>>(key).await {
                return Ok((val, ver));
            }

            // Cache miss (or undecodable entry): fetch from store and refill the cache.
            let (v, meta) = self.store_get(&p).await?;
            let val: T = from_slice(&v[..])?;
            let version = UpdateVersion {
                e_tag: meta.e_tag,
                version: meta.version,
            };
            self.cache_set(key, (CacheStoreValue(&val, version.clone()), None))
                .await;
            Ok((val, version))
        }
    }

    /// Persists a value to storage and updates the cache on success.
    ///
    /// When `version` is provided, the write uses an atomic compare-and-swap
    /// against that storage version. Without a version, the store write uses
    /// overwrite semantics. Neither mode makes the cache update atomic with
    /// storage; see the trait's consistency notes.
    fn cache_store_set<T>(
        &self,
        key: &str,
        val: T,
        version: Option<UpdateVersion>,
    ) -> impl Future<Output = Result<UpdateVersion, BoxError>> + Send
    where
        T: Serialize + Send,
    {
        async move {
            let data = to_canonical_vec(&val)?;
            let p = path_lowercase(&Path::from(key));
            let mode = version.map_or(PutMode::Overwrite, |ver| {
                PutMode::Update(OsVersion {
                    e_tag: ver.e_tag,
                    version: ver.version,
                })
            });
            let res = self.store_put(&p, mode, data.into()).await?;
            let ver = UpdateVersion {
                e_tag: res.e_tag,
                version: res.version,
            };
            self.cache_set(p.as_ref(), (CacheStoreValue(val, ver.clone()), None))
                .await;
            Ok(ver)
        }
    }

    /// Deletes a value from both cache and storage.
    ///
    /// The cache is evicted only after storage deletion succeeds. An already
    /// in-flight read can still repopulate it afterward; callers must coordinate
    /// concurrent operations when deletion needs to be immediately visible.
    fn cache_store_delete(&self, key: &str) -> impl Future<Output = Result<(), BoxError>> + Send {
        async move {
            let p = path_lowercase(&Path::from(key));
            self.store_delete(&p).await?;
            self.cache_delete(p.as_ref()).await;
            Ok(())
        }
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
    use bytes::Bytes;
    use futures::executor::block_on;
    use http::Extensions;
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
        read_error: Mutex<Option<BoxError>>,
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
            from_slice(&value.0[..]).map_err(|err| err.into())
        }

        async fn cache_get_with<T, F>(&self, key: &str, init: F) -> Result<T, BoxError>
        where
            T: Sized + DeserializeOwned + Serialize + Send,
            F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
        {
            if let Some(value) = self.cache.lock().unwrap().get(key).cloned() {
                return from_slice(&value.0[..]).map_err(|err| err.into());
            }

            let (value, expiry) = init.await?;
            let data = to_canonical_vec(&value)?;
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
            let data = to_canonical_vec(&val.0).unwrap();
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

            let data = to_canonical_vec(&val.0).unwrap();
            cache.insert(key.to_string(), Arc::new((data.into(), val.1)));
            true
        }

        async fn cache_delete(&self, key: &str) -> bool {
            self.cache.lock().unwrap().remove(key).is_some()
        }
    }

    impl StoreFeatures for TestCacheStore {
        async fn store_get(&self, path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
            self.store_gets.fetch_add(1, Ordering::SeqCst);
            if let Some(error) = self.read_error.lock().unwrap().take() {
                return Err(error);
            }
            let (value, version) = self
                .store
                .lock()
                .unwrap()
                .get(path.as_ref())
                .cloned()
                .ok_or_else(|| object_store::Error::NotFound {
                    path: path.to_string(),
                    source: std::io::Error::from(std::io::ErrorKind::NotFound).into(),
                })?;

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
                extensions: Extensions::default(),
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
    fn cache_store_keys_use_the_same_canonical_path_in_both_layers() {
        block_on(async {
            let ctx = TestCacheStore::default();
            ctx.cache_store_init("Folder/Foo*", async { Ok::<_, BoxError>(1_u32) })
                .await
                .unwrap();
            assert_eq!(
                ctx.cache_store_get::<u32>("folder/foo*").await.unwrap().0,
                1
            );
            ctx.cache_store_set("folder/foo*", 2_u32, None)
                .await
                .unwrap();
            assert_eq!(
                ctx.cache_store_get::<u32>("Folder/Foo*").await.unwrap().0,
                2
            );
            let key = path_lowercase(&Path::from("Folder/Foo*"));
            assert!(ctx.cache_contains(key.as_ref()));
            assert_eq!(
                ctx.store.lock().unwrap().keys().collect::<Vec<_>>(),
                vec![&key.to_string()]
            );
            ctx.cache_delete(key.as_ref()).await;
            assert_eq!(
                ctx.cache_store_get::<u32>("Folder/Foo*").await.unwrap().0,
                2
            );
            ctx.cache_store_delete("FOLDER/FOO*").await.unwrap();
            assert!(!ctx.cache_contains(key.as_ref()));
            assert!(ctx.cache_store_get::<u32>("folder/foo*").await.is_err());
        });
    }

    #[test]
    fn cache_store_init_propagates_read_failures_without_initializing() {
        block_on(async {
            for kind in [
                std::io::ErrorKind::TimedOut,
                std::io::ErrorKind::PermissionDenied,
            ] {
                let ctx = TestCacheStore::default();
                *ctx.read_error.lock().unwrap() = Some(std::io::Error::from(kind).into());
                let error = ctx
                    .cache_store_init::<u32, _>("key", async {
                        panic!("initializer must not run on read failure")
                    })
                    .await
                    .unwrap_err();
                assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), kind);
                assert!(ctx.store.lock().unwrap().is_empty());
                assert!(!ctx.cache_contains("key"));
            }
        });
    }

    #[test]
    fn cache_store_init_recognizes_wrapped_not_found_errors() {
        #[derive(Debug, thiserror::Error)]
        #[error("store read failed: {0}")]
        struct WrappedError(#[source] object_store::Error);

        block_on(async {
            let ctx = TestCacheStore::default();
            *ctx.read_error.lock().unwrap() =
                Some(Box::new(WrappedError(object_store::Error::NotFound {
                    path: "key".into(),
                    source: std::io::Error::from(std::io::ErrorKind::NotFound).into(),
                })));
            ctx.cache_store_init("key", async { Ok::<_, BoxError>(7_u32) })
                .await
                .unwrap();
            assert_eq!(ctx.cache_store_get::<u32>("key").await.unwrap().0, 7);
        });
    }

    #[test]
    fn cache_store_get_populates_cache_without_second_store_read() {
        let ctx = TestCacheStore::default();
        let stored_version = UpdateVersion {
            e_tag: Some("etag-stored".to_string()),
            version: Some("1".to_string()),
        };
        let data = to_canonical_vec(&123_u32).unwrap();
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

    #[test]
    fn cache_store_init_loads_existing_value_and_skips_initializer() {
        let ctx = TestCacheStore::default();
        let stored_version = UpdateVersion {
            e_tag: Some("etag-existing".to_string()),
            version: Some("7".to_string()),
        };
        let data = to_canonical_vec(&"stored".to_string()).unwrap();
        ctx.put_serialized("message", data, stored_version.clone());

        block_on(ctx.cache_store_init("message", async {
            Err::<String, BoxError>("initializer should not run".into())
        }))
        .unwrap();

        let (value, version) = block_on(ctx.cache_store_get::<String>("message")).unwrap();
        assert_eq!(value, "stored");
        assert_eq!(version.e_tag, stored_version.e_tag);
        assert_eq!(version.version, stored_version.version);
        assert_eq!(ctx.store_gets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_store_init_creates_missing_value_and_delete_clears_layers() {
        let ctx = TestCacheStore::default();

        block_on(ctx.cache_store_init("message", async {
            Ok::<_, BoxError>("created".to_string())
        }))
        .unwrap();
        assert!(ctx.cache_contains("message"));
        assert!(ctx.store.lock().unwrap().contains_key("message"));

        let (value, _) = block_on(ctx.cache_store_get::<String>("message")).unwrap();
        assert_eq!(value, "created");

        block_on(ctx.cache_store_delete("message")).unwrap();
        assert!(!ctx.cache_contains("message"));
        assert!(!ctx.store.lock().unwrap().contains_key("message"));
    }

    #[test]
    fn cache_store_set_update_enforces_expected_version() {
        let ctx = TestCacheStore::default();

        let version = block_on(ctx.cache_store_set("answer", 1_u32, None)).unwrap();
        let updated = block_on(ctx.cache_store_set("answer", 2_u32, Some(version))).unwrap();
        let (value, cached_version) = block_on(ctx.cache_store_get::<u32>("answer")).unwrap();
        assert_eq!(value, 2);
        assert_eq!(cached_version.version, updated.version);

        let err = block_on(ctx.cache_store_set(
            "answer",
            3_u32,
            Some(UpdateVersion {
                e_tag: Some("wrong".to_string()),
                version: Some("wrong".to_string()),
            }),
        ))
        .unwrap_err();
        assert!(err.to_string().contains("version mismatch"));
    }

    /// The default methods must yield `Send` futures so runtimes can spawn them.
    #[test]
    fn cache_store_futures_are_send() {
        fn assert_send<T: Send>(_: T) {}

        let ctx = TestCacheStore::default();
        assert_send(ctx.cache_store_init("key", async { Ok::<_, BoxError>(1_u32) }));
        assert_send(ctx.cache_store_get::<u32>("key"));
        assert_send(ctx.cache_store_set("key", 1_u32, None));
        assert_send(ctx.cache_store_delete("key"));
    }

    #[test]
    fn derivation_path_with_prefixes_current_path() {
        let path = Path::from("agent/main");
        let derivation_path = derivation_path_with(&path, vec![b"child".to_vec()]);
        assert_eq!(
            derivation_path,
            vec![b"agent/main".to_vec(), b"child".to_vec()]
        );
    }
}
