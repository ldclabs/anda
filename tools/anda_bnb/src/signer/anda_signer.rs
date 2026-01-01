use alloy::consensus::SignableTransaction;
use alloy::hex;
use alloy::primitives::{Address, B256, ChainId, U256};
use alloy::signers::{
    self as alloy_signer, Error, Result, Signature, Signer, sign_transaction_with_chain_id,
};
use anda_core::KeysFeatures;
use anda_engine::context::BaseCtx;
use async_trait::async_trait;
use std::fmt;

/// Anda signer that uses a remote TEE service via a web client.
pub struct AndaEvmSigner {
    /// The derivation path
    derivation: Vec<Vec<u8>>,
    /// The chain ID
    chain_id: Option<ChainId>,
    /// The Ethereum address
    address: Address,
    /// The public key
    pubkey: [u8; 33],
    /// The base context for communicating with the TEE service
    client: BaseCtx,
}

impl fmt::Debug for AndaEvmSigner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AndaEvmSigner")
            .field("derivation", &self.derivation)
            .field("chain_id", &self.chain_id)
            .field("address", &self.address)
            .field("pubkey", &hex::encode(self.pubkey))
            .finish()
    }
}

/// Errors thrown by [`AndaEvmSigner`].
#[derive(Debug, thiserror::Error)]
pub enum AndaSignerError {
    /// Web client error
    #[error("web client error: {0}")]
    WebClient(String),

    /// Address derivation error
    #[error("address derivation error: {0}")]
    AddressDerivation(String),

    /// Public key convertion error
    #[error("public key convertion error: {0}")]
    PubKeyConvertion(String),

    /// Signature error
    #[error("signature error: {0}")]
    SignatureError(String),
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl alloy::network::TxSigner<Signature> for AndaEvmSigner {
    fn address(&self) -> Address {
        self.address
    }

    async fn sign_transaction(
        &self,
        tx: &mut dyn SignableTransaction<Signature>,
    ) -> Result<Signature> {
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Anda signing transaction: {:#?}", tx);
        }
        sign_transaction_with_chain_id!(self, tx, self.sign_hash(&tx.signature_hash()).await)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Signer for AndaEvmSigner {
    async fn sign_hash(&self, hash: &B256) -> Result<Signature> {
        let sig_ic = self
            .client
            .secp256k1_sign_digest_ecdsa(self.derivation.clone(), hash.as_slice())
            .await
            .map_err(|e| Error::other(AndaSignerError::SignatureError(e.to_string())))?;

        let signature = Signature::new(
            U256::from_be_slice(&sig_ic[0..32]),  // r
            U256::from_be_slice(&sig_ic[32..64]), // s
            y_parity(hash.as_slice(), &sig_ic, self.pubkey.as_slice())?,
        );

        Ok(signature)
    }

    #[inline]
    fn address(&self) -> Address {
        self.address
    }

    #[inline]
    fn chain_id(&self) -> Option<ChainId> {
        self.chain_id
    }

    #[inline]
    fn set_chain_id(&mut self, chain_id: Option<ChainId>) {
        self.chain_id = chain_id;
    }
}

alloy::network::impl_into_wallet!(AndaEvmSigner);

/// Represents the AndaEvmSigner, which is responsible for signing operations
/// using a TEE (Trusted Execution Environment) service.
impl AndaEvmSigner {
    /// Creates a new instance of `AndaEvmSigner`.
    ///
    /// This method performs the following steps:
    /// 1. Fetches the public key from the TEE service using the provided derivation path.
    /// 2. Derives the Ethereum address from the fetched public key.
    /// 3. Initializes the `AndaEvmSigner` instance with the derived address, public key, and other parameters.
    ///
    /// ### Parameters
    /// - `client`: An instance of `BaseCtx` used to interact with the TEE service.
    /// - `derivation`: A vector of vector byte representing the derivation path for the key.
    /// - `chain_id`: An optional chain ID for the Ethereum network.
    ///
    /// ### Returns
    /// - `Result<Self, AndaSignerError>`: On success, returns an instance of `AndaEvmSigner`. On failure, returns an `AndaSignerError`.
    pub async fn new(
        client: BaseCtx,
        derivation: Vec<Vec<u8>>,
        chain_id: Option<ChainId>,
    ) -> Result<Self, AndaSignerError> {
        // Fetch the public key from the TEE service
        let pubkey = client
            .secp256k1_public_key(derivation.clone())
            .await
            .map_err(|e| AndaSignerError::WebClient(e.to_string()))?;

        // Convert the public key to an Ethereum address
        let address =
            derive_address_from_pubkey(&pubkey).map_err(AndaSignerError::AddressDerivation)?;
        log::debug!(
            "Signer pubkey: {:?}, Signer EVM address: {:?}",
            hex::encode(pubkey),
            address
        );

        Ok(Self {
            derivation,
            chain_id,
            address,
            pubkey,
            client,
        })
    }
}

/// Helper function to derive an Ethereum address from a public key
///
/// # Arguments
///
/// * `pubkey` - A byte slice representing the public key in SEC1 format.
///
/// # Returns
///
/// The Ethereum address derived from the public key, or an error if conversion fails.
pub fn derive_address_from_pubkey(pubkey: &[u8]) -> Result<Address, String> {
    let key = k256::ecdsa::VerifyingKey::from_sec1_bytes(pubkey).map_err(|e| e.to_string())?;
    Ok(alloy::signers::utils::public_key_to_address(&key))
}

/// Computes the parity bit allowing to recover the public key from the signature.
///
/// # Arguments
///
/// * `prehash` - The prehash of the message.
/// * `sig` - The signature.
/// * `pubkey` - The public key.
///
/// # Returns
///
/// The parity bit.
fn y_parity(prehash: &[u8], sig: &[u8], pubkey: &[u8]) -> Result<bool> {
    use alloy::signers::k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

    let orig_key = VerifyingKey::from_sec1_bytes(pubkey).expect("failed to parse the pubkey");
    let signature = Signature::try_from(sig)?;
    for parity in [0u8, 1] {
        let recid = RecoveryId::try_from(parity)?;
        let recovered_key = match VerifyingKey::recover_from_prehash(prehash, &signature, recid) {
            Ok(k) => k,
            Err(_) => continue, // 尝试另一 parity
        };
        if recovered_key == orig_key {
            return Ok(parity == 1);
        }
    }

    panic!(
        "failed to recover the parity bit from a signature; sig: {}, pubkey: {}",
        hex::encode(sig),
        hex::encode(pubkey)
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ledger::DRVT_PATH;
    use anda_engine::{
        context::Web3SDK,
        engine::{AgentInfo, EngineBuilder},
        extension::extractor::Extractor,
    };
    use anda_web3_client::client::Client as Web3Client;
    use rand::Rng;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;

    // public static chain id of BNB
    pub static CHAIN_ID: u64 = 97; // BNB testnet

    #[tokio::test]
    async fn test_sign_message() {
        // Create an agent for testing
        #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
        struct TestStruct {
            name: String,
            age: Option<u8>,
        }
        let agent = Extractor::<TestStruct>::default();

        // Generate random bytes for root secret
        let mut rng = rand::rng();
        let random_bytes: Vec<u8> = (0..48).map(|_| rng.random()).collect();
        let root_secret: [u8; 48] = random_bytes
            .clone()
            .try_into()
            .map_err(|_| format!("invalid root_secret: {:?}", &random_bytes))
            .unwrap();

        // Initialize Web3 client
        let web3 = Web3Client::builder()
            .with_root_secret(root_secret)
            .build()
            .await
            .unwrap();

        // Create a context for testing
        let engine_ctx = EngineBuilder::new()
            .with_info(AgentInfo {
                handle: "bnb_test".to_string(),
                name: "BNB_TEST".to_string(),
                description: "Test BNB Engine".to_string(),
                endpoint: "https://localhost:8443/default".to_string(),
                ..Default::default()
            })
            .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3))))
            .register_agent(agent)
            .unwrap()
            .mock_ctx();
        let ctx = engine_ctx.base;

        let signer = AndaEvmSigner::new(
            ctx,
            DRVT_PATH.iter().map(|&s| s.to_vec()).collect(),
            Some(CHAIN_ID),
        )
        .await
        .unwrap();

        let message = vec![0, 1, 2, 3];
        let sig = signer.sign_message(&message).await.unwrap();
        assert_eq!(
            sig.recover_address_from_msg(message).unwrap(),
            signer.address()
        );
    }
}
