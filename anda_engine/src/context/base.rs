//! Base context implementation for the Anda Engine.
//!
//! This module provides the core context implementation that serves as the foundation
//! for all operations in the system. The [`BaseCtx`] struct implements various traits
//! that provide access to:
//! - [`StateFeatures`]: Context state management;
//! - [`KeysFeatures`]: Cryptographic key operations;
//! - [`StoreFeatures`]: Persistent storage operations;
//! - [`CacheFeatures`]: Caching mechanisms;
//! - [`CanisterCaller`]: Canister interaction capabilities;
//! - [`HttpFeatures`]: HTTPs communication features.
//!
//! The context is designed to be:
//! - Thread-safe through Arc-based sharing of resources;
//! - Cloneable with each clone maintaining its own state;
//! - Hierarchical through child context creation;
//! - Cancellable through CancellationToken integration.
//!
//! Key features:
//! - Context depth limiting to prevent infinite nesting;
//! - TEE (Trusted Execution Environment) integration for secure operations;
//! - Unified interface for cryptographic operations with multiple algorithms;
//! - Consistent error handling through BoxError;
//! - Time tracking for operation duration.

use anda_core::{
    BaseContext, BoxError, CacheExpiry, CacheFeatures, CacheStoreFeatures, CancellationToken,
    CanisterCaller, HttpFeatures, Json, KeysFeatures, ObjectMeta, Path, PutMode, PutResult,
    RequestMeta, StateFeatures, StoreFeatures, ToolInput, ToolOutput, derivation_path_with,
};
use bytes::Bytes;
use candid::{CandidType, Principal, utils::ArgumentEncoder};
use http::Extensions;
use parking_lot::RwLock;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeSet,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

const CONTEXT_MAX_DEPTH: u8 = 42;
const CACHE_MAX_CAPACITY: u64 = 1000000;

use super::{
    RemoteEngines,
    cache::CacheService,
    web3::{Web3Client, Web3SDK},
};
use crate::store::Store;

#[derive(Clone)]
pub struct BaseCtx {
    pub(crate) id: Principal,
    pub(crate) name: String,
    pub(crate) caller: Principal,
    pub(crate) path: Path,
    pub(crate) cancellation_token: CancellationToken,
    pub(crate) start_at: Instant,
    pub(crate) depth: u8,
    pub(crate) web3: Arc<Web3SDK>,
    /// Registered remote engines for tool and agent execution.
    pub(crate) remote: Arc<RemoteEngines>,
    pub(crate) state: Arc<RwLock<Extensions>>,
    pub(crate) meta: RequestMeta,

    cache: Arc<CacheService>,
    store: Store,
}

/// Base context [`BaseContext`] implementation providing core functionality for the engine.
///
/// This struct serves as the foundation for all operations in the system,
/// providing access to:
/// - User authentication and authorization;
/// - Cryptographic operations;
/// - Storage operations;
/// - Caching mechanisms;
/// - Canister communication;
/// - HTTP operations.
///
/// The context is designed to be thread-safe and cloneable, with each clone
/// maintaining its own state while sharing underlying resources.
impl BaseCtx {
    /// Creates a new BaseCtx instance.
    pub(crate) fn new(
        id: Principal,
        name: String,
        cancellation_token: CancellationToken,
        names: BTreeSet<Path>,
        web3: Arc<Web3SDK>,
        store: Store,
        remote: Arc<RemoteEngines>,
    ) -> Self {
        let caller = Principal::anonymous();
        Self {
            id,
            name: name.clone(),
            caller,
            path: Path::default(),
            cancellation_token,
            start_at: Instant::now(),
            cache: Arc::new(CacheService::new(CACHE_MAX_CAPACITY, names)),
            store,
            web3,
            depth: 0,
            remote,
            state: Arc::new(RwLock::new(Extensions::default())),
            meta: RequestMeta::default(),
        }
    }

    /// Creates a child context with a new path.
    ///
    /// This is used to create nested contexts while maintaining the parent's state.
    /// The child context inherits all properties from the parent but with:
    /// - A new path;
    /// - A child cancellation token;
    /// - Incremented depth.
    ///
    /// # Arguments
    /// * `path` - New path for the child context.
    ///
    /// # Errors
    /// Returns an error if the context depth exceeds CONTEXT_MAX_DEPTH.
    pub(crate) fn child(&self, path: String) -> Result<Self, BoxError> {
        let path = Path::parse(path)?;
        let child = Self {
            id: self.id,
            name: self.name.clone(),
            caller: self.caller,
            path,
            cancellation_token: self.cancellation_token.child_token(),
            start_at: self.start_at,
            cache: self.cache.clone(),
            store: self.store.clone(),
            web3: self.web3.clone(),
            depth: self.depth + 1,
            remote: self.remote.clone(),
            state: self.state.clone(),
            meta: self.meta.clone(),
        };

        if child.depth >= CONTEXT_MAX_DEPTH {
            return Err("Context depth limit exceeded".into());
        }
        Ok(child)
    }

    /// Creates a child context with additional user and caller information.
    ///
    /// Similar to `child()`, but allows specifying user and caller information
    /// for the new context.
    ///
    /// # Arguments
    /// * `path` - New path for the child context;
    /// * `caller` - caller principal (or ANONYMOUS);
    /// * `user` - user state;
    /// * `meta` - Metadata for the new context.
    ///
    /// # Errors
    /// Returns an error if the context depth exceeds CONTEXT_MAX_DEPTH.
    pub(crate) fn child_with(
        &self,
        caller: Principal,
        path: String,
        meta: RequestMeta,
    ) -> Result<Self, BoxError> {
        let path = Path::parse(path)?;
        let child = Self {
            id: self.id,
            name: self.name.clone(),
            caller,
            path,
            cancellation_token: self.cancellation_token.child_token(),
            start_at: Instant::now(),
            cache: self.cache.clone(),
            store: self.store.clone(),
            web3: self.web3.clone(),
            depth: self.depth + 1,
            remote: self.remote.clone(),
            state: self.state.clone(),
            meta,
        };

        if child.depth >= CONTEXT_MAX_DEPTH {
            return Err("Context depth limit exceeded".into());
        }
        Ok(child)
    }

    pub(crate) fn self_meta(&self, target: Principal) -> RequestMeta {
        RequestMeta {
            engine: Some(target),
            user: Some(self.name.clone()),
            extra: Default::default(),
        }
    }

    pub fn get_state<T>(&self) -> Option<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        self.state.read().get::<T>().cloned()
    }

    pub fn set_state<T>(&self, v: T) -> Option<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        self.state.write().insert(v)
    }
}

impl BaseContext for BaseCtx {
    /// Executes a remote tool call via HTTP RPC.
    ///
    /// # Arguments
    /// * `endpoint` - Remote endpoint URL;
    /// * `args` - Tool input arguments, [`ToolInput`].
    ///
    /// # Returns
    /// [`ToolOutput`] containing the final result.
    async fn remote_tool_call(
        &self,
        endpoint: &str,
        mut args: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        let target = self
            .remote
            .get_id_by_endpoint(endpoint)
            .ok_or_else(|| format!("remote engine endpoint {} not found", endpoint))?;
        args.meta = Some(self.self_meta(target));
        self.https_signed_rpc(endpoint, "tool_call", &(&args,))
            .await
    }
}

impl CacheStoreFeatures for BaseCtx {}

impl StateFeatures for BaseCtx {
    fn engine_id(&self) -> &Principal {
        &self.id
    }

    fn engine_name(&self) -> &str {
        &self.name
    }

    fn caller(&self) -> &Principal {
        &self.caller
    }

    fn meta(&self) -> &RequestMeta {
        &self.meta
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    fn time_elapsed(&self) -> Duration {
        self.start_at.elapsed()
    }
}

impl KeysFeatures for BaseCtx {
    /// Derives a 256-bit AES-GCM key from the given derivation path.
    async fn a256gcm_key(&self, derivation_path: Vec<Vec<u8>>) -> Result<[u8; 32], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.a256gcm_key(derivation_path_with(&self.path, derivation_path))
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.a256gcm_key(derivation_path_with(&self.path, derivation_path))
                    .await
            }
        }
    }

    /// Signs a message using Ed25519 signature scheme from the given derivation path.
    async fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.ed25519_sign_message(derivation_path_with(&self.path, derivation_path), message)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.ed25519_sign_message(derivation_path_with(&self.path, derivation_path), message)
                    .await
            }
        }
    }

    /// Verifies an Ed25519 signature from the given derivation path.
    async fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.ed25519_verify(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                    signature,
                )
                .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.ed25519_verify(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                    signature,
                )
                .await
            }
        }
    }

    /// Gets the public key for Ed25519 from the given derivation path.
    async fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 32], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.ed25519_public_key(derivation_path_with(&self.path, derivation_path))
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.ed25519_public_key(derivation_path_with(&self.path, derivation_path))
                    .await
            }
        }
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path.
    async fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_sign_message_bip340(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                )
                .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_sign_message_bip340(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                )
                .await
            }
        }
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path.
    async fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_verify_bip340(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                    signature,
                )
                .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_verify_bip340(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                    signature,
                )
                .await
            }
        }
    }

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path.
    /// The message will be hashed with SHA-256 before signing.
    async fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_sign_message_ecdsa(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                )
                .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_sign_message_ecdsa(
                    derivation_path_with(&self.path, derivation_path),
                    message,
                )
                .await
            }
        }
    }

    /// Signs a message hash using Secp256k1 ECDSA signature from the given derivation path.
    async fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_sign_digest_ecdsa(
                    derivation_path_with(&self.path, derivation_path),
                    message_hash,
                )
                .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_sign_digest_ecdsa(
                    derivation_path_with(&self.path, derivation_path),
                    message_hash,
                )
                .await
            }
        }
    }

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path.
    async fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_verify_ecdsa(
                    derivation_path_with(&self.path, derivation_path),
                    message_hash,
                    signature,
                )
                .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_verify_ecdsa(
                    derivation_path_with(&self.path, derivation_path),
                    message_hash,
                    signature,
                )
                .await
            }
        }
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path.
    async fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 33], BoxError> {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_public_key(derivation_path_with(&self.path, derivation_path))
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_public_key(derivation_path_with(&self.path, derivation_path))
                    .await
            }
        }
    }
}

impl StoreFeatures for BaseCtx {
    /// Retrieves data from storage at the specified path.
    async fn store_get(&self, path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        self.store.store_get(&self.path, path).await
    }

    /// Lists objects in storage with optional prefix and offset filters.
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results;
    /// * `offset` - Optional path to start listing from (exclude).
    async fn store_list(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> Result<Vec<ObjectMeta>, BoxError> {
        self.store.store_list(&self.path, prefix, offset).await
    }

    /// Stores data at the specified path with a given write mode.
    ///
    /// # Arguments
    /// * `path` - Target storage path;
    /// * `mode` - Write mode (Create, Overwrite, etc.);
    /// * `value` - Data to store as bytes.
    async fn store_put(
        &self,
        path: &Path,
        mode: PutMode,
        value: bytes::Bytes,
    ) -> Result<PutResult, BoxError> {
        self.store.store_put(&self.path, path, mode, value).await
    }

    /// Renames a storage object if the target path doesn't exist.
    ///
    /// # Arguments
    /// * `from` - Source path;
    /// * `to` - Destination path.
    async fn store_rename_if_not_exists(&self, from: &Path, to: &Path) -> Result<(), BoxError> {
        self.store
            .store_rename_if_not_exists(&self.path, from, to)
            .await
    }

    /// Deletes data at the specified path.
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete.
    async fn store_delete(&self, path: &Path) -> Result<(), BoxError> {
        self.store.store_delete(&self.path, path).await
    }
}

impl CacheFeatures for BaseCtx {
    /// Checks if a key exists in the cache.
    fn cache_contains(&self, key: &str) -> bool {
        self.cache.contains(&self.path, key)
    }

    /// Gets a cached value by key, returns error if not found or deserialization fails.
    async fn cache_get<T>(&self, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.cache.get(&self.path, key).await
    }

    /// Gets a cached value or initializes it if missing.
    ///
    /// If key doesn't exist, calls init function to create value and cache it.
    async fn cache_get_with<T, F>(&self, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        self.cache.get_with(&self.path, key, init).await
    }

    /// Sets a value in cache with optional expiration policy.
    async fn cache_set<T>(&self, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        self.cache.set(&self.path, key, val).await
    }

    /// Sets a value in cache if key doesn't exist, returns true if set.
    async fn cache_set_if_not_exists<T>(&self, key: &str, val: (T, Option<CacheExpiry>)) -> bool
    where
        T: Sized + Serialize + Send,
    {
        self.cache.set_if_not_exists(&self.path, key, val).await
    }

    /// Deletes a cached value by key, returns true if key existed.
    async fn cache_delete(&self, key: &str) -> bool {
        self.cache.delete(&self.path, key).await
    }

    /// Returns an iterator over all cached items with raw value.
    fn cache_raw_iter(
        &self,
    ) -> impl Iterator<Item = (Arc<String>, Arc<(Bytes, Option<CacheExpiry>)>)> {
        self.cache.iter(&self.path)
    }
}

impl CanisterCaller for BaseCtx {
    /// Performs a query call to a canister (read-only, no state changes).
    ///
    /// # Arguments
    /// * `canister` - Target canister principal;
    /// * `method` - Method name to call;
    /// * `args` - Input arguments encoded in Candid format.
    async fn canister_query<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        self.web3
            .as_ref()
            .canister_query(canister, method, args)
            .await
    }

    /// Performs an update call to a canister (may modify state).
    ///
    /// # Arguments
    /// * `canister` - Target canister principal;
    /// * `method` - Method name to call;
    /// * `args` - Input arguments encoded in Candid format.
    async fn canister_update<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        self.web3
            .as_ref()
            .canister_update(canister, method, args)
            .await
    }
}

impl HttpFeatures for BaseCtx {
    /// Makes an HTTPs request.
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`;
    /// * `method` - HTTP method (GET, POST, etc.);
    /// * `headers` - Optional HTTP headers;
    /// * `body` - Optional request body (default empty).
    async fn https_call(
        &self,
        url: &str,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>,
    ) -> Result<reqwest::Response, BoxError> {
        self.web3
            .as_ref()
            .https_call(url, method, headers, body)
            .await
    }

    /// Makes a signed HTTPs request with message authentication.
    ///
    /// # Arguments
    /// * `url` - Target URL;
    /// * `method` - HTTP method (GET, POST, etc.);
    /// * `message_digest` - 32-byte message digest for signing;
    /// * `headers` - Optional HTTP headers;
    /// * `body` - Optional request body (default empty).
    async fn https_signed_call(
        &self,
        url: &str,
        method: http::Method,
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> Result<reqwest::Response, BoxError> {
        self.web3
            .as_ref()
            .https_signed_call(url, method, message_digest, headers, body)
            .await
    }

    /// Makes a signed CBOR-encoded RPC call.
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to;
    /// * `method` - RPC method name to call;
    /// * `args` - Arguments to serialize as CBOR and send with the request.
    async fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        args: impl Serialize + Send,
    ) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.web3
            .as_ref()
            .https_signed_rpc(endpoint, method, args)
            .await
    }
}
