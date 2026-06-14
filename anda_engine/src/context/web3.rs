//! Web3 capability adapter used by engine contexts.
//!
//! The engine can run against either the TEE gateway client or a generic Web3
//! client supplied by host applications. This module normalizes both options
//! into the [`Web3SDK`] enum and implements the core cryptographic, canister,
//! and HTTP traits against that abstraction.

use anda_core::{BoxError, BoxPinFut, CanisterCaller, HttpFeatures, KeysFeatures};
use candid::{
    CandidType, Decode, Principal,
    utils::{ArgumentEncoder, encode_args},
};
use cbor2::{from_slice, to_canonical_vec};
use ic_auth_verifier::envelope::SignedEnvelope;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;

pub use ic_tee_gateway_sdk::client::{Client as TEEClient, ClientBuilder as TEEClientBuilder};

/// Represents a Web3 client for interacting with the Internet Computer and other services.
pub enum Web3SDK {
    /// TEE gateway SDK client.
    Tee(Arc<TEEClient>),
    /// Host-provided Web3 client implementation.
    Web3(Web3Client),
}

impl Web3SDK {
    /// Wraps a TEE gateway client.
    pub fn from_tee(client: Arc<TEEClient>) -> Self {
        Self::Tee(client)
    }

    /// Wraps a generic Web3 client implementation.
    pub fn from_web3(client: Arc<dyn Web3ClientFeatures>) -> Self {
        Self::Web3(Web3Client { client })
    }

    /// Returns a placeholder client whose operations fail with `not implemented`.
    pub fn not_implemented() -> Self {
        Self::Web3(Web3Client::not_implemented())
    }

    /// Returns the principal associated with the underlying client.
    pub fn get_principal(&self) -> Principal {
        match self {
            Web3SDK::Tee(cli) => cli.get_principal(),
            Web3SDK::Web3(Web3Client { client }) => client.get_principal(),
        }
    }
}

/// Object-safe Web3 capability surface required by [`Web3SDK`].
pub trait Web3ClientFeatures: Send + Sync + 'static {
    /// Returns the principal associated with the client identity.
    fn get_principal(&self) -> Principal;

    /// Signs a digest and returns an authorization envelope.
    fn sign_envelope(
        &self,
        message_digest: [u8; 32],
    ) -> BoxPinFut<Result<SignedEnvelope, BoxError>>;

    /// Derives a 256-bit AES-GCM key from the given derivation path
    fn a256gcm_key(&self, derivation_path: Vec<Vec<u8>>) -> BoxPinFut<Result<[u8; 32], BoxError>>;

    /// Signs a message using Ed25519 signature scheme from the given derivation path
    fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>>;

    /// Verifies an Ed25519 signature from the given derivation path
    fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>>;

    /// Gets the public key for Ed25519 from the given derivation path
    fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> BoxPinFut<Result<[u8; 32], BoxError>>;

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path
    fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>>;

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path
    fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>>;

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path
    /// The message will be hashed with SHA-256 before signing
    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>>;

    /// Signs a message hash using Secp256k1 ECDSA signature from the given derivation path
    fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>>;

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path
    fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>>;

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path
    fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> BoxPinFut<Result<[u8; 33], BoxError>>;

    /// Performs a query call to a canister (read-only, no state changes)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    fn canister_query_raw(
        &self,
        canister: Principal,
        method: String,
        args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>>;

    /// Performs an update call to a canister (may modify state)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    fn canister_update_raw(
        &self,
        canister: Principal,
        method: String,
        args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>>;

    /// Makes an HTTPs request
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    fn https_call(
        &self,
        url: String,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>>;

    /// Makes a signed HTTPs request with message authentication
    ///
    /// # Arguments
    /// * `url` - Target URL
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `message_digest` - 32-byte message digest for signing
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    fn https_signed_call(
        &self,
        url: String,
        method: http::Method,
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>>;

    /// Makes a signed CBOR-encoded RPC call
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to
    /// * `method` - RPC method name to call
    /// * `args` - Arguments to serialize as CBOR and send with the request
    fn https_signed_rpc_raw(
        &self,
        endpoint: String,
        method: String,
        args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>>;
}

struct NotImplemented;

impl Web3ClientFeatures for NotImplemented {
    fn get_principal(&self) -> Principal {
        Principal::anonymous()
    }

    fn sign_envelope(
        &self,
        _message_digest: [u8; 32],
    ) -> BoxPinFut<Result<SignedEnvelope, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn a256gcm_key(&self, _derivation_path: Vec<Vec<u8>>) -> BoxPinFut<Result<[u8; 32], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn ed25519_sign_message(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn ed25519_verify(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
        _signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn ed25519_public_key(
        &self,
        _derivation_path: Vec<Vec<u8>>,
    ) -> BoxPinFut<Result<[u8; 32], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn secp256k1_sign_message_bip340(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn secp256k1_verify_bip340(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
        _signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn secp256k1_sign_message_ecdsa(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn secp256k1_sign_digest_ecdsa(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message_hash: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn secp256k1_verify_ecdsa(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message_hash: &[u8],
        _signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn secp256k1_public_key(
        &self,
        _derivation_path: Vec<Vec<u8>>,
    ) -> BoxPinFut<Result<[u8; 33], BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn canister_query_raw(
        &self,
        _canister: Principal,
        _method: String,
        _args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn canister_update_raw(
        &self,
        _canister: Principal,
        _method: String,
        _args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn https_call(
        &self,
        _url: String,
        _method: http::Method,
        _headers: Option<http::HeaderMap>,
        _body: Option<Vec<u8>>, // default is empty
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn https_signed_call(
        &self,
        _url: String,
        _method: http::Method,
        _message_digest: [u8; 32],
        _headers: Option<http::HeaderMap>,
        _body: Option<Vec<u8>>, // default is empty
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn https_signed_rpc_raw(
        &self,
        _endpoint: String,
        _method: String,
        _params: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

/// Shared object-safe Web3 client wrapper.
#[derive(Clone)]
pub struct Web3Client {
    /// Shared object-safe Web3 capability implementation.
    pub client: Arc<dyn Web3ClientFeatures>,
}

impl Web3Client {
    /// Creates a placeholder client whose operations fail with `not implemented`.
    pub fn not_implemented() -> Self {
        Self {
            client: Arc::new(NotImplemented),
        }
    }
}

impl CanisterCaller for &Web3SDK {
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
        match self {
            Web3SDK::Tee(cli) => cli.canister_query(canister, method, args).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                let input = encode_args(args)?;
                let res = cli
                    .canister_query_raw(canister.to_owned(), method.to_string(), input)
                    .await?;
                let output = Decode!(res.as_slice(), Out)?;
                Ok(output)
            }
        }
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
        match self {
            Web3SDK::Tee(cli) => cli.canister_update(canister, method, args).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                let input = encode_args(args)?;
                let res = cli
                    .canister_update_raw(canister.to_owned(), method.to_string(), input)
                    .await?;
                let output = Decode!(res.as_slice(), Out)?;
                Ok(output)
            }
        }
    }
}

impl HttpFeatures for &Web3SDK {
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
        match self {
            Web3SDK::Tee(cli) => cli.https_call(url, method, headers, body).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.https_call(url.to_string(), method, headers, body).await
            }
        }
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
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> Result<reqwest::Response, BoxError> {
        match self {
            Web3SDK::Tee(cli) => {
                cli.https_signed_call(url, method, message_digest, headers, body)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.https_signed_call(url.to_string(), method, message_digest, headers, body)
                    .await
            }
        }
    }

    /// Makes a signed CBOR-encoded RPC call
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to
    /// * `method` - RPC method name to call
    /// * `args` - Arguments to serialize as CBOR and send with the request
    async fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        args: impl Serialize + Send,
    ) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        match self {
            Web3SDK::Tee(cli) => cli.https_signed_rpc(endpoint, method, args).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                let args = to_canonical_vec(&args)?;
                let res = cli
                    .https_signed_rpc_raw(endpoint.to_string(), method.to_string(), args)
                    .await?;
                let res = from_slice(&res[..])?;
                Ok(res)
            }
        }
    }
}

impl KeysFeatures for &Web3SDK {
    async fn a256gcm_key(&self, derivation_path: Vec<Vec<u8>>) -> Result<[u8; 32], BoxError> {
        match self {
            Web3SDK::Tee(cli) => cli.a256gcm_key(derivation_path).await,
            Web3SDK::Web3(Web3Client { client: cli }) => cli.a256gcm_key(derivation_path).await,
        }
    }

    async fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self {
            Web3SDK::Tee(cli) => cli.ed25519_sign_message(derivation_path, message).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.ed25519_sign_message(derivation_path, message).await
            }
        }
    }

    async fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        match self {
            Web3SDK::Tee(cli) => cli.ed25519_verify(derivation_path, message, signature).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.ed25519_verify(derivation_path, message, signature).await
            }
        }
    }

    async fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 32], BoxError> {
        match self {
            Web3SDK::Tee(cli) => cli.ed25519_public_key(derivation_path).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.ed25519_public_key(derivation_path).await
            }
        }
    }

    async fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_sign_message_bip340(derivation_path, message)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_sign_message_bip340(derivation_path, message)
                    .await
            }
        }
    }

    async fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        match self {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_verify_bip340(derivation_path, message, signature)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_verify_bip340(derivation_path, message, signature)
                    .await
            }
        }
    }

    async fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_sign_message_ecdsa(derivation_path, message)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_sign_message_ecdsa(derivation_path, message)
                    .await
            }
        }
    }

    async fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        match self {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_sign_digest_ecdsa(derivation_path, message_hash)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_sign_digest_ecdsa(derivation_path, message_hash)
                    .await
            }
        }
    }

    async fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        match self {
            Web3SDK::Tee(cli) => {
                cli.secp256k1_verify_ecdsa(derivation_path, message_hash, signature)
                    .await
            }
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_verify_ecdsa(derivation_path, message_hash, signature)
                    .await
            }
        }
    }

    async fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 33], BoxError> {
        match self {
            Web3SDK::Tee(cli) => cli.secp256k1_public_key(derivation_path).await,
            Web3SDK::Web3(Web3Client { client: cli }) => {
                cli.secp256k1_public_key(derivation_path).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candid::encode_args;

    struct MockWeb3Client {
        principal: Principal,
    }

    impl MockWeb3Client {
        fn new(principal: Principal) -> Self {
            Self { principal }
        }
    }

    impl Web3ClientFeatures for MockWeb3Client {
        fn get_principal(&self) -> Principal {
            self.principal
        }

        fn sign_envelope(
            &self,
            _message_digest: [u8; 32],
        ) -> BoxPinFut<Result<SignedEnvelope, BoxError>> {
            Box::pin(futures::future::ready(Err(
                "mock envelope unavailable".into()
            )))
        }

        fn a256gcm_key(
            &self,
            _derivation_path: Vec<Vec<u8>>,
        ) -> BoxPinFut<Result<[u8; 32], BoxError>> {
            Box::pin(futures::future::ready(Ok([1; 32])))
        }

        fn ed25519_sign_message(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
        ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
            Box::pin(futures::future::ready(Ok([2; 64])))
        }

        fn ed25519_verify(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
            _signature: &[u8],
        ) -> BoxPinFut<Result<(), BoxError>> {
            Box::pin(futures::future::ready(Ok(())))
        }

        fn ed25519_public_key(
            &self,
            _derivation_path: Vec<Vec<u8>>,
        ) -> BoxPinFut<Result<[u8; 32], BoxError>> {
            Box::pin(futures::future::ready(Ok([3; 32])))
        }

        fn secp256k1_sign_message_bip340(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
        ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
            Box::pin(futures::future::ready(Ok([4; 64])))
        }

        fn secp256k1_verify_bip340(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
            _signature: &[u8],
        ) -> BoxPinFut<Result<(), BoxError>> {
            Box::pin(futures::future::ready(Ok(())))
        }

        fn secp256k1_sign_message_ecdsa(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message: &[u8],
        ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
            Box::pin(futures::future::ready(Ok([5; 64])))
        }

        fn secp256k1_sign_digest_ecdsa(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message_hash: &[u8],
        ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
            Box::pin(futures::future::ready(Ok([6; 64])))
        }

        fn secp256k1_verify_ecdsa(
            &self,
            _derivation_path: Vec<Vec<u8>>,
            _message_hash: &[u8],
            _signature: &[u8],
        ) -> BoxPinFut<Result<(), BoxError>> {
            Box::pin(futures::future::ready(Ok(())))
        }

        fn secp256k1_public_key(
            &self,
            _derivation_path: Vec<Vec<u8>>,
        ) -> BoxPinFut<Result<[u8; 33], BoxError>> {
            Box::pin(futures::future::ready(Ok([7; 33])))
        }

        fn canister_query_raw(
            &self,
            _canister: Principal,
            method: String,
            _args: Vec<u8>,
        ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
            Box::pin(futures::future::ready(Ok(encode_args((format!(
                "query:{method}"
            ),))
            .unwrap())))
        }

        fn canister_update_raw(
            &self,
            _canister: Principal,
            method: String,
            _args: Vec<u8>,
        ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
            Box::pin(futures::future::ready(Ok(encode_args((format!(
                "update:{method}"
            ),))
            .unwrap())))
        }

        fn https_call(
            &self,
            url: String,
            _method: http::Method,
            _headers: Option<http::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
            Box::pin(futures::future::ready(
                Err(format!("no http: {url}").into()),
            ))
        }

        fn https_signed_call(
            &self,
            url: String,
            _method: http::Method,
            _message_digest: [u8; 32],
            _headers: Option<http::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
            Box::pin(futures::future::ready(Err(format!(
                "no signed http: {url}"
            )
            .into())))
        }

        fn https_signed_rpc_raw(
            &self,
            _endpoint: String,
            method: String,
            _args: Vec<u8>,
        ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
            Box::pin(futures::future::ready(
                to_canonical_vec(&format!("rpc:{method}")).map_err(|err| err.into()),
            ))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn web3_sdk_delegates_to_mock_client_and_decodes_results() {
        let principal = Principal::self_authenticating([1; 32]);
        let sdk = Web3SDK::from_web3(Arc::new(MockWeb3Client::new(principal)));

        assert_eq!(sdk.get_principal(), principal);
        let Web3SDK::Web3(Web3Client { client }) = &sdk else {
            panic!("expected web3 client");
        };
        assert_eq!(
            client.a256gcm_key(vec![b"path".to_vec()]).await.unwrap(),
            [1; 32]
        );
        assert_eq!(
            client
                .ed25519_sign_message(vec![b"path".to_vec()], b"message")
                .await
                .unwrap(),
            [2; 64]
        );
        client
            .ed25519_verify(vec![b"path".to_vec()], b"message", &[0; 64])
            .await
            .unwrap();
        assert_eq!(
            client
                .ed25519_public_key(vec![b"path".to_vec()])
                .await
                .unwrap(),
            [3; 32]
        );
        assert_eq!(
            client
                .secp256k1_sign_message_bip340(vec![b"path".to_vec()], b"message")
                .await
                .unwrap(),
            [4; 64]
        );
        client
            .secp256k1_verify_bip340(vec![b"path".to_vec()], b"message", &[0; 64])
            .await
            .unwrap();
        assert_eq!(
            client
                .secp256k1_sign_message_ecdsa(vec![b"path".to_vec()], b"message")
                .await
                .unwrap(),
            [5; 64]
        );
        assert_eq!(
            client
                .secp256k1_sign_digest_ecdsa(vec![b"path".to_vec()], &[0; 32])
                .await
                .unwrap(),
            [6; 64]
        );
        client
            .secp256k1_verify_ecdsa(vec![b"path".to_vec()], &[0; 32], &[0; 64])
            .await
            .unwrap();
        assert_eq!(
            client
                .secp256k1_public_key(vec![b"path".to_vec()])
                .await
                .unwrap(),
            [7; 33]
        );

        let query: String = (&sdk)
            .canister_query(&Principal::anonymous(), "status", ())
            .await
            .unwrap();
        assert_eq!(query, "query:status");
        let update: String = (&sdk)
            .canister_update(&Principal::anonymous(), "commit", ())
            .await
            .unwrap();
        assert_eq!(update, "update:commit");
        let rpc: String = (&sdk)
            .https_signed_rpc("https://example.test/rpc", "ping", &("arg",))
            .await
            .unwrap();
        assert_eq!(rpc, "rpc:ping");

        assert!(
            (&sdk)
                .https_call("https://example.test", http::Method::GET, None, None)
                .await
                .unwrap_err()
                .to_string()
                .contains("no http")
        );
        assert!(
            (&sdk)
                .https_signed_call(
                    "https://example.test",
                    http::Method::POST,
                    [0; 32],
                    None,
                    None,
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("no signed http")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn not_implemented_web3_client_reports_errors_for_all_operations() {
        let sdk = Web3SDK::not_implemented();
        assert_eq!(sdk.get_principal(), Principal::anonymous());
        let Web3SDK::Web3(Web3Client { client }) = &sdk else {
            panic!("expected web3 client");
        };

        assert!(client.sign_envelope([0; 32]).await.is_err());
        assert!(client.a256gcm_key(Vec::new()).await.is_err());
        assert!(client.ed25519_sign_message(Vec::new(), b"m").await.is_err());
        assert!(
            client
                .ed25519_verify(Vec::new(), b"m", &[0; 64])
                .await
                .is_err()
        );
        assert!(client.ed25519_public_key(Vec::new()).await.is_err());
        assert!(
            client
                .secp256k1_sign_message_bip340(Vec::new(), b"m")
                .await
                .is_err()
        );
        assert!(
            client
                .secp256k1_verify_bip340(Vec::new(), b"m", &[0; 64])
                .await
                .is_err()
        );
        assert!(
            client
                .secp256k1_sign_message_ecdsa(Vec::new(), b"m")
                .await
                .is_err()
        );
        assert!(
            client
                .secp256k1_sign_digest_ecdsa(Vec::new(), &[0; 32])
                .await
                .is_err()
        );
        assert!(
            client
                .secp256k1_verify_ecdsa(Vec::new(), &[0; 32], &[0; 64])
                .await
                .is_err()
        );
        assert!(client.secp256k1_public_key(Vec::new()).await.is_err());
        assert!(
            client
                .canister_query_raw(Principal::anonymous(), "q".to_string(), Vec::new())
                .await
                .is_err()
        );
        assert!(
            client
                .canister_update_raw(Principal::anonymous(), "u".to_string(), Vec::new())
                .await
                .is_err()
        );
        assert!(
            client
                .https_call(
                    "https://example.test".to_string(),
                    http::Method::GET,
                    None,
                    None
                )
                .await
                .is_err()
        );
        assert!(
            client
                .https_signed_call(
                    "https://example.test".to_string(),
                    http::Method::POST,
                    [0; 32],
                    None,
                    None,
                )
                .await
                .is_err()
        );
        assert!(
            client
                .https_signed_rpc_raw(
                    "https://example.test".to_string(),
                    "rpc".to_string(),
                    Vec::new(),
                )
                .await
                .is_err()
        );

        let query: Result<String, BoxError> = (&sdk)
            .canister_query(&Principal::anonymous(), "status", ())
            .await;
        assert!(query.is_err());
        let update: Result<String, BoxError> = (&sdk)
            .canister_update(&Principal::anonymous(), "commit", ())
            .await;
        assert!(update.is_err());
        let rpc: Result<String, BoxError> = (&sdk)
            .https_signed_rpc("https://example.test/rpc", "ping", &())
            .await;
        assert!(rpc.is_err());
    }
}
