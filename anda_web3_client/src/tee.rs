//! TEE gateway-backed Web3 client. Enabled by the `tee` feature.
//!
//! [`TeeClient`](crate::tee::TeeClient) adapts an [`ic_tee_gateway_sdk`] client to
//! the object-safe [`Web3ClientFeatures`](anda_engine::context::Web3ClientFeatures)
//! trait so it can be handed to the engine
//! through [`anda_engine::context::Web3SDK::from_web3`]. Key derivation, envelope
//! signing, and attestation use the gateway SDK. Signed RPC uses the engine's
//! bounded response reader while leaving envelope signing to the gateway.
//! Signed calls reject malformed URLs and embedded userinfo before any signing;
//! the gateway and its HTTP client keep enforcing HTTPS.
//!
//! Canister access is not part of
//! [`Web3ClientFeatures`](anda_engine::context::Web3ClientFeatures). When needed,
//! use the gateway client via [`TeeClient::gateway`](crate::tee::TeeClient::gateway), which
//! implements `CanisterCaller` itself.
//!
//! ```
//! use std::sync::Arc;
//! use anda_engine::context::Web3SDK;
//! use anda_web3_client::tee::{TeeClient, TeeGatewayClient};
//!
//! // Supply a gateway already connected to the TEE.
//! fn web3_sdk(gateway: Arc<TeeGatewayClient>) -> Web3SDK {
//!     Web3SDK::from_web3(Arc::new(TeeClient::new(gateway)))
//! }
//! ```

use crate::{
    ecdsa_digest,
    request::{check_url, rpc_body},
};
use anda_cloud_cdk::{TEEInfo, TEEKind};
use anda_core::{BoxError, BoxPinFut, cbor_rpc};
use anda_engine::context::Web3ClientFeatures;
use candid::Principal;
use ic_auth_types::ByteBufB64;
use ic_auth_verifier::{envelope::SignedEnvelope, sha3_256};
use ic_tee_cdk::AttestationRequest;
use std::{future::ready, sync::Arc};

pub use ic_tee_gateway_sdk::client::{
    Client as TeeGatewayClient, ClientBuilder as TeeGatewayClientBuilder,
};

/// Adapts a TEE gateway client to the engine's [`Web3ClientFeatures`] trait.
#[derive(Clone)]
pub struct TeeClient {
    gateway: Arc<TeeGatewayClient>,
}

impl TeeClient {
    /// Wraps a TEE gateway client.
    pub fn new(gateway: Arc<TeeGatewayClient>) -> Self {
        Self { gateway }
    }

    /// Returns the underlying TEE gateway client.
    pub fn gateway(&self) -> &Arc<TeeGatewayClient> {
        &self.gateway
    }
}

impl Web3ClientFeatures for TeeClient {
    fn get_principal(&self) -> Principal {
        self.gateway.get_principal()
    }

    fn sign_envelope(
        &self,
        message_digest: [u8; 32],
    ) -> BoxPinFut<Result<SignedEnvelope, BoxError>> {
        let gw = self.gateway.clone();
        Box::pin(async move { gw.sign_envelope(message_digest).await })
    }

    fn tee_attestation(
        &self,
        public_key: ByteBufB64,
        nonce: Vec<u8>,
    ) -> BoxPinFut<Result<Option<TEEInfo>, BoxError>> {
        let gw = self.gateway.clone();
        Box::pin(async move {
            let info = gw.tee_info().ok_or("TEE not available")?;
            let tee = gw
                .sign_attestation(AttestationRequest {
                    public_key: Some(public_key),
                    user_data: None,
                    nonce: Some(nonce.into()),
                })
                .await?;
            Ok(Some(TEEInfo {
                id: info.id,
                kind: TEEKind::try_from(tee.kind.as_str())?,
                url: info.url,
                attestation: Some(tee.attestation),
            }))
        })
    }

    fn a256gcm_key(&self, derivation_path: Vec<Vec<u8>>) -> BoxPinFut<Result<[u8; 32], BoxError>> {
        let gw = self.gateway.clone();
        Box::pin(async move { gw.a256gcm_key(derivation_path).await })
    }

    fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let gw = self.gateway.clone();
        let message = message.to_vec();
        Box::pin(async move { gw.ed25519_sign_message(derivation_path, &message).await })
    }

    fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        let gw = self.gateway.clone();
        let message = message.to_vec();
        let signature = signature.to_vec();
        Box::pin(async move {
            gw.ed25519_verify(derivation_path, &message, &signature)
                .await
        })
    }

    fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> BoxPinFut<Result<[u8; 32], BoxError>> {
        let gw = self.gateway.clone();
        Box::pin(async move { gw.ed25519_public_key(derivation_path).await })
    }

    fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let gw = self.gateway.clone();
        let message = message.to_vec();
        Box::pin(async move {
            gw.secp256k1_sign_message_bip340(derivation_path, &message)
                .await
        })
    }

    fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        let gw = self.gateway.clone();
        let message = message.to_vec();
        let signature = signature.to_vec();
        Box::pin(async move {
            gw.secp256k1_verify_bip340(derivation_path, &message, &signature)
                .await
        })
    }

    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let gw = self.gateway.clone();
        let message = message.to_vec();
        Box::pin(async move {
            gw.secp256k1_sign_message_ecdsa(derivation_path, &message)
                .await
        })
    }

    fn secp256k1_sign_digest_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let gw = self.gateway.clone();
        let message_hash = match ecdsa_digest(message_hash) {
            Ok(digest) => *digest,
            Err(err) => return Box::pin(ready(Err(err))),
        };
        Box::pin(async move {
            gw.secp256k1_sign_digest_ecdsa(derivation_path, &message_hash)
                .await
        })
    }

    fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message_hash: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        let gw = self.gateway.clone();
        let message_hash = match ecdsa_digest(message_hash) {
            Ok(digest) => *digest,
            Err(err) => return Box::pin(ready(Err(err))),
        };
        let signature = signature.to_vec();
        Box::pin(async move {
            gw.secp256k1_verify_ecdsa(derivation_path, &message_hash, &signature)
                .await
        })
    }

    fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> BoxPinFut<Result<[u8; 33], BoxError>> {
        let gw = self.gateway.clone();
        Box::pin(async move { gw.secp256k1_public_key(derivation_path).await })
    }

    fn https_call(
        &self,
        url: String,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>,
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
        let gw = self.gateway.clone();
        Box::pin(async move { gw.https_call(&url, method, headers, body).await })
    }

    fn https_signed_call(
        &self,
        url: String,
        method: http::Method,
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>,
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
        if let Err(err) = check_url(&url, true) {
            return Box::pin(ready(Err(err)));
        }
        let gw = self.gateway.clone();
        Box::pin(async move {
            gw.https_signed_call(&url, method, message_digest, headers, body)
                .await
        })
    }

    fn https_signed_rpc_raw(
        &self,
        endpoint: String,
        method: String,
        args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        if let Err(err) = check_url(&endpoint, true) {
            return Box::pin(ready(Err(err)));
        }
        let gw = self.gateway.clone();
        Box::pin(async move {
            let body = rpc_body(&method, args)?;
            let envelope = gw.sign_envelope(sha3_256(&body)).await?;
            let mut headers = http::HeaderMap::new();
            envelope.to_authorization(&mut headers)?;
            let res = cbor_rpc(&gw.outer_http, &endpoint, &method, Some(headers), body).await?;
            Ok(res.into_vec())
        })
    }
}
