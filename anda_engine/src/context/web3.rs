use anda_core::{BoxError, BoxPinFut, CanisterCaller, HttpFeatures};
use candid::{
    CandidType, Decode, Principal,
    utils::{ArgumentEncoder, encode_args},
};
use ciborium::from_reader;
use ic_auth_types::deterministic_cbor_into_vec;
use ic_auth_verifier::envelope::SignedEnvelope;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;

pub use ic_tee_gateway_sdk::client::{Client as TEEClient, ClientBuilder as TEEClientBuilder};

/// Represents a Web3 client for interacting with the Internet Computer and other services.
pub enum Web3SDK {
    Tee(Arc<TEEClient>),
    Web3(Web3Client),
}

impl Web3SDK {
    pub fn from_tee(client: Arc<TEEClient>) -> Self {
        Self::Tee(client)
    }

    pub fn from_web3(client: Arc<dyn Web3ClientFeatures>) -> Self {
        Self::Web3(Web3Client { client })
    }

    pub fn not_implemented() -> Self {
        Self::Web3(Web3Client::not_implemented())
    }

    pub fn get_principal(&self) -> Principal {
        match self {
            Web3SDK::Tee(cli) => cli.get_principal(),
            Web3SDK::Web3(Web3Client { client }) => client.get_principal(),
        }
    }
}

pub trait Web3ClientFeatures: Send + Sync + 'static {
    fn get_principal(&self) -> Principal;

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

#[derive(Clone)]
pub struct Web3Client {
    pub client: Arc<dyn Web3ClientFeatures>,
}

impl Web3Client {
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
                let args = deterministic_cbor_into_vec(&args)?;
                let res = cli
                    .https_signed_rpc_raw(endpoint.to_string(), method.to_string(), args)
                    .await?;
                let res = from_reader(&res[..])?;
                Ok(res)
            }
        }
    }
}
