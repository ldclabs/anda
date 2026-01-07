//! # Context Module
//!
//! This module defines the core context interfaces and traits that provide the execution environment
//! for Agents and Tools in the ANDA system. It includes:
//!
//! - **AgentContext**: The primary interface combining all core functionality and AI-specific features.
//! - **BaseContext**: Fundamental operations available to all Agents and Tools.
//! - **Feature Traits**: Modular capabilities including state management, cryptographic operations,
//!   storage, caching, and HTTP communication.
//!
//! The context system is designed to be:
//! - **Modular**: Features are separated into distinct traits for better organization and flexibility.
//! - **Asynchronous**: All operations are async to support efficient I/O operations.
//! - **Extensible**: New features can be added as separate traits while maintaining compatibility.
//! - **Secure**: Includes cryptographic operations and verified caller information.
//!
//! ## Key Components
//!
//! ### Core Traits
//! - [`AgentContext`]: Main interface combining all capabilities;
//! - [`BaseContext`]: Fundamental operations required by all contexts;
//!
//! ### Feature Sets
//! - [`StateFeatures`]: Contextual information about the execution environment;
//! - [`KeysFeatures`]: Cryptographic operations and key management;
//! - [`StoreFeatures`]: Persistent storage capabilities;
//! - [`CacheFeatures`]: In-memory caching with expiration policies;
//! - [`HttpFeatures`]: HTTP communication capabilities;
//! - [`VectorSearchFeatures`]: Semantic search functionality.
//!
//! ## Usage
//! Implement these traits to create custom execution contexts for Agents and Tools. The `anda_engine` [`context`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/context/mod.rs) module provides.
//! a complete implementation, but custom implementations can be created for specialized environments.

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

/// AgentContext provides the execution environment for Agents.
/// It combines core functionality with AI-specific features:
/// - [`BaseContext`]`: Fundamental operations;
/// - [`CompletionFeatures`]: LLM completions and function calling;
/// - [`EmbeddingFeatures`]: Text embeddings.
pub trait AgentContext: BaseContext + CompletionFeatures + EmbeddingFeatures {
    /// Retrieves definitions for available tools.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific tool names.
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools.
    fn tool_definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition>;

    /// Retrieves definitions for available tools in the remote engines.
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
        names: Option<&[&str]>,
    ) -> impl Future<Output = Result<Vec<FunctionDefinition>, BoxError>> + Send;

    /// Extracts resources from the provided list based on the tool's supported tags.
    fn select_tool_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> impl Future<Output = Vec<Resource>> + Send;

    /// Retrieves definitions for available agents.
    ///
    /// # Arguments
    /// * `names` - Optional filter for specific agent names;
    /// * `with_prefix` - Flag to add the prefix `LA_` to agent names to distinguish from tools.
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents.
    fn agent_definitions(
        &self,
        names: Option<&[&str]>,
        with_prefix: bool,
    ) -> Vec<FunctionDefinition>;

    /// Retrieves definitions for available agents in the remote engines.
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
        names: Option<&[&str]>,
    ) -> impl Future<Output = Result<Vec<FunctionDefinition>, BoxError>> + Send;

    /// Extracts resources from the provided list based on the agent's supported tags.
    fn select_agent_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> impl Future<Output = Vec<Resource>> + Send;

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
    /// * `args` - Tool input arguments, [`AgentInput`].
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    fn agent_run(
        &self,
        args: AgentInput,
    ) -> impl Future<Output = Result<(AgentOutput, Option<Principal>), BoxError>> + Send;

    /// Runs a remote agent via HTTP RPC.
    ///
    /// # Arguments
    /// * `endpoint` - Remote endpoint URL;
    /// * `args` - Tool input arguments, [`AgentInput`]. The `meta` field will be set to the current agent's metadata.
    ///
    /// # Returns
    /// [`AgentOutput`] containing the result of the agent execution.
    fn remote_agent_run(
        &self,
        endpoint: &str,
        args: AgentInput,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// BaseContext is the core context interface available when calling Agent or Tool.
/// It provides access to various feature sets including:
/// - [`StateFeatures`]: User, caller, time, and cancellation token.
/// - [`KeysFeatures`]: Cryptographic key operations.
/// - [`StoreFeatures`]: Persistent storage.
/// - [`CacheFeatures`]: In-memory caching.
/// - [`HttpFeatures`]: HTTP request capabilities.
/// - [`CanisterCaller`]: ICP blockchain smart contract interactions.
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

/// StateFeatures is one of the context feature sets available when calling Agent or Tool.
pub trait StateFeatures: Sized {
    /// Gets the engine ID
    fn engine_id(&self) -> &Principal;

    /// Gets the engine name。
    fn engine_name(&self) -> &str;

    /// Gets the verified caller principal if available.
    /// A non anonymous principal indicates the request has been verified
    /// using ICP blockchain's signature verification algorithm.
    /// Details: https://github.com/ldclabs/ic-auth
    fn caller(&self) -> &Principal;

    /// Gets the matadata of the request。
    fn meta(&self) -> &RequestMeta;

    /// Gets the cancellation token for the current execution context.
    /// Each call level has its own token scope.
    /// For example, when an agent calls a tool, the tool receives
    /// a child token of the agent's token.
    /// Cancelling the agent's token will cancel all its child calls,
    /// but cancelling a tool's token won't affect its parent agent.
    fn cancellation_token(&self) -> CancellationToken;

    /// Gets the time elapsed since the original context was created.
    fn time_elapsed(&self) -> Duration;
}

/// Provides vector search capabilities for semantic similarity search.
pub trait VectorSearchFeatures: Sized {
    /// Performs a semantic search to find top n most similar documents.
    /// Returns a list of deserialized json document.
    fn top_n(
        &self,
        query: &str,
        n: usize,
    ) -> impl Future<Output = Result<Vec<String>, BoxError>> + Send;

    /// Performs a semantic search but returns only document IDs.
    /// More efficient when only document identifiers are needed.
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<String>, BoxError>> + Send;
}

/// KeysFeatures is one of the context feature sets available when calling Agent or Tool.
///
/// The Agent engine running in TEE has a permanent fixed 48-bit root key,
/// from which AES, Ed25519, Secp256k1 keys are derived.
/// The Agent/Tool name is included in key derivation, ensuring isolation
/// even with the same derivation path.
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

    /// Gets the public key for Ed25519 from the given derivation path.
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

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path.
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

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path.
    fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 33], BoxError>> + Send;
}

/// StoreFeatures is one of the context feature sets available when calling Agent or Tool.
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

/// CacheFeatures is one of the context feature sets available when calling Agent or Tool.
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

/// HttpFeatures provides HTTP request capabilities for Agents and Tools.
///
/// All HTTP requests are managed and scheduled by the Agent engine.
/// Since Agents may run in WASM containers, implementations should not
/// implement HTTP requests directly.
pub trait HttpFeatures: Sized {
    /// Makes an HTTPs request.
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

    /// Makes a signed HTTPs request with message authentication.
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

/// CacheStoreFeatures combines Store and Cache features for efficient data management.
#[async_trait]
pub trait CacheStoreFeatures: StoreFeatures + CacheFeatures + Send + Sync + 'static {
    /// Initializes a cache value from store if missing.
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

    /// Gets a value from cache, initializes from store if missing.
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

    /// Sets a value to cache and persists it in store.
    /// Optionally provides an update version for atomic updates.
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
            // delete cache for fetching on next get
            self.cache_delete(key).await;
            Ok(UpdateVersion {
                e_tag: res.e_tag,
                version: res.version,
            })
        }
    }

    /// Deletes a value from cache and store
    async fn cache_store_delete(&self, key: &str) -> Result<(), BoxError> {
        let p = Path::from(key);
        self.cache_delete(key).await;
        self.store_delete(&p).await
    }
}

/// Derives a derivation path with the given path and derivation path.
pub fn derivation_path_with(path: &Path, derivation_path: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut dp = Vec::with_capacity(derivation_path.len() + 1);
    dp.push(path.as_ref().as_bytes().to_vec());
    dp.extend(derivation_path);
    dp
}
