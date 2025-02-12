//! Trusted Execution Environment (TEE) Client Implementation
//!
//! This module provides a client for interacting with Trusted Execution Environment (TEE) services,
//! offering cryptographic operations and secure communication capabilities. The TEEClient implements
//! multiple interfaces including CoseSDK, CanisterCaller, and HttpFeatures to provide a comprehensive
//! set of security features.
//!
//! # Key Features
//! - Cryptographic key derivation and management
//! - Digital signature generation and verification (Ed25519, Secp256k1)
//! - Secure communication with canisters
//! - HTTPS request handling with message authentication
//! - CBOR-encoded RPC calls with signing
//!
//! # Security Considerations
//! - All cryptographic operations are performed within the TEE
//! - HTTPS-only communication enforced
//! - Message authentication for all signed requests
//! - Timeouts and keep-alive settings configured for secure connections
//!
//! # Interfaces Implemented
//! - [`CoseSDK`]: For COSE (CBOR Object Signing and Encryption) operations
//! - [`CanisterCaller`]: For secure ICP canisters communication
//! - [`HttpFeatures`]: For secure HTTP operations with signing capabilities

use anda_core::{
    canister_rpc, cbor_rpc, http_rpc, BoxError, HttpFeatures, HttpRPCError, Path, RPCRequest,
    CONTENT_TYPE_CBOR,
};
use candid::{utils::ArgumentEncoder, CandidType, Principal};
use ciborium::from_reader;
use ic_cose::client::CoseSDK;
use ic_cose_types::{
    cose::{
        ed25519::ed25519_verify,
        k256::{secp256k1_verify_bip340, secp256k1_verify_ecdsa},
        sha3_256,
    },
    to_cbor_bytes, CanisterCaller,
};
use reqwest::Client;
use serde::{de::DeserializeOwned, Serialize};
use serde_bytes::ByteArray;
use std::{collections::HashMap, time::Duration};

use crate::APP_USER_AGENT;

/// Client for interacting with Trusted Execution Environment (TEE) services
///
/// Provides cryptographic operations, canister communication, and HTTP features
/// through a secure TEE interface. Manages both internal and external HTTP clients
/// with different configurations for secure communication.
#[derive(Clone)]
pub struct TEEClient {
    pub http: Client,
    pub outer_http: Client,
    pub cose_canister: Principal,
    endpoint_keys: String,
    endpoint_identity: String,
    endpoint_canister_query: String,
    endpoint_canister_update: String,
}

impl TEEClient {
    /// Creates a new TEEClient instance
    ///
    /// # Arguments
    /// * `tee_host` - Base URL of the TEE service
    /// * `basic_token` - Authentication token for TEE access
    /// * `cose_canister` - Principal of the COSE canister
    ///
    /// # Returns
    /// Configured TEEClient instance with initialized HTTP clients and endpoints
    pub fn new(tee_host: &str, basic_token: &str, cose_canister: Principal) -> Self {
        let http = reqwest::Client::builder()
            .http2_keep_alive_interval(Some(Duration::from_secs(25)))
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .http2_keep_alive_while_idle(true)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(20))
            .user_agent(APP_USER_AGENT)
            .default_headers({
                let mut headers = http::header::HeaderMap::with_capacity(3);
                let ct: http::HeaderValue = CONTENT_TYPE_CBOR.parse().unwrap();
                headers.insert(http::header::CONTENT_TYPE, ct.clone());
                headers.insert(http::header::ACCEPT, ct);
                if !basic_token.is_empty() {
                    headers.insert(http::header::AUTHORIZATION, basic_token.parse().unwrap());
                }

                headers
            })
            .build()
            .expect("Anda reqwest client should build");

        let outer_http = reqwest::Client::builder()
            .use_rustls_tls()
            .https_only(true)
            .http2_keep_alive_interval(Some(Duration::from_secs(25)))
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .http2_keep_alive_while_idle(true)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .user_agent(APP_USER_AGENT)
            .build()
            .expect("Anda reqwest client should build");

        Self {
            http,
            outer_http,
            cose_canister,
            endpoint_keys: format!("{}/keys", tee_host),
            endpoint_identity: format!("{}/identity", tee_host),
            endpoint_canister_query: format!("{}/canister/query", tee_host),
            endpoint_canister_update: format!("{}/canister/update", tee_host),
        }
    }

    /// Derives a 256-bit AES-GCM key from the given derivation path
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    ///
    /// # Returns
    /// Result containing the derived 256-bit key or an error
    pub async fn a256gcm_key(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
    ) -> Result<[u8; 32], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<32> =
            http_rpc(&self.http, &self.endpoint_keys, "a256gcm_key", &(dp,)).await?;
        Ok(res.into_array())
    }

    /// Signs a message using Ed25519 signature scheme
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Message to be signed
    ///
    /// # Returns
    /// Result containing the 64-byte signature or an error
    pub async fn ed25519_sign_message(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<64> = http_rpc(
            &self.http,
            &self.endpoint_keys,
            "ed25519_sign_message",
            &(dp, message),
        )
        .await?;
        Ok(res.into_array())
    }

    /// Verifies an Ed25519 signature
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Original message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// Result indicating success or failure of verification
    pub async fn ed25519_verify(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        let pk = self.ed25519_public_key(path, derivation_path).await?;
        ed25519_verify(&pk, message, signature).map_err(|e| e.into())
    }

    /// Gets the public key for Ed25519
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    ///
    /// # Returns
    /// Result containing the 32-byte public key or an error
    pub async fn ed25519_public_key(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
    ) -> Result<[u8; 32], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: (ByteArray<32>, ByteArray<32>) = http_rpc(
            &self.http,
            &self.endpoint_keys,
            "ed25519_public_key",
            &(dp,),
        )
        .await?;
        Ok(res.0.into_array())
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Message to be signed
    ///
    /// # Returns
    /// Result containing the 64-byte signature or an error
    pub async fn secp256k1_sign_message_bip340(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<64> = http_rpc(
            &self.http,
            &self.endpoint_keys,
            "secp256k1_sign_message_bip340",
            &(dp, message),
        )
        .await?;
        Ok(res.into_array())
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Original message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// Result indicating success or failure of verification
    pub async fn secp256k1_verify_bip340(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        let pk = self.secp256k1_public_key(path, derivation_path).await?;
        secp256k1_verify_bip340(&pk, message, signature).map_err(|e| e.into())
    }

    /// Signs a message using Secp256k1 ECDSA signature
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Message to be signed
    ///
    /// # Returns
    /// Result containing the 64-byte signature or an error
    pub async fn secp256k1_sign_message_ecdsa(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<64> = http_rpc(
            &self.http,
            &self.endpoint_keys,
            "secp256k1_sign_message_ecdsa",
            &(dp, message),
        )
        .await?;
        Ok(res.into_array())
    }

    /// Verifies a Secp256k1 ECDSA signature
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Original message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// Result indicating success or failure of verification
    pub async fn secp256k1_verify_ecdsa(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        let pk = self.secp256k1_public_key(path, derivation_path).await?;
        secp256k1_verify_ecdsa(&pk, message, signature).map_err(|e| e.into())
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    ///
    /// # Returns
    /// Result containing the 33-byte public key or an error
    pub async fn secp256k1_public_key(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
    ) -> Result<[u8; 33], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: (ByteArray<33>, ByteArray<32>) = http_rpc(
            &self.http,
            &self.endpoint_keys,
            "secp256k1_public_key",
            &(dp,),
        )
        .await?;
        Ok(res.0.into_array())
    }
}

/// Implements the `CoseSDK` trait for TEEClient to enable IC-COSE canister API calls
///
/// This implementation provides the necessary interface to interact with the
/// [IC-COSE](https://github.com/ldclabs/ic-cose) canister, allowing cryptographic
/// operations through the COSE (CBOR Object Signing and Encryption) protocol.
impl CoseSDK for TEEClient {
    fn canister(&self) -> &Principal {
        &self.cose_canister
    }
}

impl CanisterCaller for TEEClient {
    /// Performs a query call to a canister (read-only, no state changes)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    async fn canister_query<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        let res = canister_rpc(
            &self.http,
            &self.endpoint_canister_query,
            canister,
            method,
            args,
        )
        .await?;
        Ok(res)
    }

    /// Performs an update call to a canister (may modify state)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    async fn canister_update<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        let res = canister_rpc(
            &self.http,
            &self.endpoint_canister_update,
            canister,
            method,
            args,
        )
        .await?;
        Ok(res)
    }
}

impl HttpFeatures for TEEClient {
    /// Makes an HTTPs request
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    async fn https_call(
        &self,
        url: &str,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> Result<reqwest::Response, BoxError> {
        if !url.starts_with("https://") {
            return Err("Invalid URL, must start with https://".into());
        }
        let mut req = self.outer_http.request(method, url);
        if let Some(headers) = headers {
            req = req.headers(headers);
        }
        if let Some(body) = body {
            req = req.body(body);
        }

        req.send().await.map_err(|e| e.into())
    }

    /// Makes a signed HTTPs request with message authentication
    ///
    /// # Arguments
    /// * `url` - Target URL
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `message_digest` - 32-byte message digest for signing
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    async fn https_signed_call(
        &self,
        url: &str,
        method: http::Method,
        message_digest: &[u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> Result<reqwest::Response, BoxError> {
        let res: HashMap<String, String> = http_rpc(
            &self.http,
            &self.endpoint_identity,
            "sign_http",
            &(message_digest,),
        )
        .await?;
        let mut headers = headers.unwrap_or_default();
        res.into_iter().for_each(|(k, v)| {
            headers.insert(
                http::HeaderName::try_from(k).expect("invalid header name"),
                http::HeaderValue::try_from(v).expect("invalid header value"),
            );
        });
        self.https_call(url, method, Some(headers), body).await
    }

    /// Makes a signed CBOR-encoded RPC call
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to
    /// * `method` - RPC method name to call
    /// * `params` - Parameters to serialize as CBOR and send with the request
    async fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        params: impl Serialize + Send,
    ) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        let params = to_cbor_bytes(&params);
        let req = RPCRequest {
            method,
            params: &params.into(),
        };
        let body = to_cbor_bytes(&req);
        let digest: [u8; 32] = sha3_256(&body);
        let res: HashMap<String, String> =
            http_rpc(&self.http, &self.endpoint_identity, "sign_http", &(digest,)).await?;
        let mut headers = http::HeaderMap::new();
        res.into_iter().for_each(|(k, v)| {
            headers.insert(
                http::HeaderName::try_from(k).expect("invalid header name"),
                http::HeaderValue::try_from(v).expect("invalid header value"),
            );
        });

        let res = cbor_rpc(&self.outer_http, endpoint, method, Some(headers), body).await?;
        let res = from_reader(&res[..]).map_err(|e| HttpRPCError::ResultError {
            endpoint: endpoint.to_string(),
            path: method.to_string(),
            error: e.into(),
        })?;
        Ok(res)
    }
}
