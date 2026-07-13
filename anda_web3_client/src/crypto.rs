//! Deterministic key derivation for the non-TEE Web3 [`Client`](crate::client::Client).
//!
//! These helpers mirror the derivation scheme of the TEE gateway
//! (`ic_tee_gateway_sdk::crypto`) exactly, so a key derived here from a given
//! root secret is byte-for-byte identical to the one the gateway derives from
//! the same secret. They are reimplemented locally so the default (non-TEE)
//! client depends on neither `ic_tee_gateway_sdk` nor any other `ic_tee_*`
//! crate.

use ic_cose_types::cose::{kdf::derive_a256gcm_key, sha3_256};

/// Derives a 256-bit AES-GCM key from a root secret and derivation path.
///
/// The derivation path is hashed with SHA3-256 to form the HKDF salt.
pub fn a256gcm_key(root_secret: &[u8], derivation_path: Vec<Vec<u8>>) -> [u8; 32] {
    let salt = derivation_path_to_context(&derivation_path);
    derive_a256gcm_key(root_secret, Some(&salt))
}

/// Signs a message using the Ed25519 signature scheme.
pub fn ed25519_sign_message(
    root_secret: &[u8],
    derivation_path: Vec<Vec<u8>>,
    msg: &[u8],
) -> [u8; 64] {
    let sk = ic_ed25519::PrivateKey::generate_from_seed(root_secret);
    let path = ic_ed25519::DerivationPath::new(
        derivation_path
            .into_iter()
            .map(ic_ed25519::DerivationIndex)
            .collect(),
    );
    let (sk, _) = sk.derive_subkey(&path);
    sk.sign_message(msg)
}

/// Derives an Ed25519 public key (32 bytes) and chain code (32 bytes).
pub fn ed25519_public_key(
    root_secret: &[u8],
    derivation_path: Vec<Vec<u8>>,
) -> ([u8; 32], [u8; 32]) {
    let sk = ic_ed25519::PrivateKey::generate_from_seed(root_secret);
    let path = ic_ed25519::DerivationPath::new(
        derivation_path
            .into_iter()
            .map(ic_ed25519::DerivationIndex)
            .collect(),
    );
    let pk = sk.public_key();
    let (pk, chain_code) = pk.derive_subkey(&path);
    (pk.serialize_raw(), chain_code)
}

/// Signs a message using the BIP-340 Schnorr signature scheme for secp256k1.
pub fn secp256k1_sign_message_bip340(
    root_secret: &[u8],
    derivation_path: Vec<Vec<u8>>,
    msg: &[u8],
) -> [u8; 64] {
    let sk = ic_secp256k1::PrivateKey::generate_from_seed(root_secret);
    let path = ic_secp256k1::DerivationPath::new(
        derivation_path
            .into_iter()
            .map(ic_secp256k1::DerivationIndex)
            .collect(),
    );
    let (sk, _) = sk.derive_subkey(&path);
    sk.sign_message_with_bip340_no_rng(msg)
}

/// Signs a message using ECDSA for secp256k1. The message is hashed with SHA-256.
pub fn secp256k1_sign_message_ecdsa(
    root_secret: &[u8],
    derivation_path: Vec<Vec<u8>>,
    msg: &[u8],
) -> [u8; 64] {
    let sk = ic_secp256k1::PrivateKey::generate_from_seed(root_secret);
    let path = ic_secp256k1::DerivationPath::new(
        derivation_path
            .into_iter()
            .map(ic_secp256k1::DerivationIndex)
            .collect(),
    );
    let (sk, _) = sk.derive_subkey(&path);
    sk.sign_message_with_ecdsa(msg)
}

/// Signs a pre-computed digest using ECDSA for secp256k1.
pub fn secp256k1_sign_digest_ecdsa(
    root_secret: &[u8],
    derivation_path: Vec<Vec<u8>>,
    message_hash: &[u8],
) -> [u8; 64] {
    let sk = ic_secp256k1::PrivateKey::generate_from_seed(root_secret);
    let path = ic_secp256k1::DerivationPath::new(
        derivation_path
            .into_iter()
            .map(ic_secp256k1::DerivationIndex)
            .collect(),
    );
    let (sk, _) = sk.derive_subkey(&path);
    sk.sign_digest_with_ecdsa(message_hash)
}

/// Derives a compressed SEC1 secp256k1 public key (33 bytes) and chain code (32 bytes).
pub fn secp256k1_public_key(
    root_secret: &[u8],
    derivation_path: Vec<Vec<u8>>,
) -> ([u8; 33], [u8; 32]) {
    let sk = ic_secp256k1::PrivateKey::generate_from_seed(root_secret);
    let path = ic_secp256k1::DerivationPath::new(
        derivation_path
            .into_iter()
            .map(ic_secp256k1::DerivationIndex)
            .collect(),
    );
    let pk = sk.public_key();
    let (pk, chain_code) = pk.derive_subkey(&path);
    let pk = pk.serialize_sec1(true);
    let pk: [u8; 33] = pk
        .try_into()
        .expect("secp256k1_public_key: invalid SEC1 public key");
    (pk, chain_code)
}

/// Hashes the concatenation of the derivation path segments with SHA3-256.
///
/// Equivalent to feeding each segment sequentially into a SHA3-256 hasher, which
/// is what the TEE gateway does, so the resulting salt matches byte-for-byte.
///
/// Note: segment boundaries are **not** encoded into the salt — the segments are
/// concatenated with no length prefix or separator, so `[b"ab", b"c"]` and
/// `[b"a", b"bc"]` hash to the same salt (and thus the same AES-GCM key). This is
/// intentional, to stay byte-for-byte compatible with the TEE gateway. Callers
/// that need distinct keys for distinct paths must not rely on a variable-length
/// segment boundary alone to separate them. (Ed25519/secp256k1 derivation is
/// unaffected: it preserves segment boundaries via `DerivationPath`.)
fn derivation_path_to_context(derivation_path: &[Vec<u8>]) -> Vec<u8> {
    let mut data = Vec::new();
    for path in derivation_path {
        data.extend_from_slice(path);
    }
    sha3_256(&data).to_vec()
}
