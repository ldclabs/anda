use anda_web3_client::crypto;

// Captured from ic_tee_gateway_sdk 0.7.2. Keep these fixed across dependency updates.
#[test]
fn derived_keys_chain_codes_and_signatures_are_stable() {
    let root = [3; 48];
    let path = || vec![b"agent".to_vec(), b"wallet".to_vec()];
    let (ed_pk, ed_chain) = crypto::ed25519_public_key(&root, path());
    let (secp_pk, secp_chain) = crypto::secp256k1_public_key(&root, path());
    let vectors = [
        (
            "aes",
            hex::encode(crypto::a256gcm_key(&root, path())),
            "d05a79fc4a01ffcfe161a2c4d451722b7e5f80b858a02606b5c7e4ad8ce86b5f",
        ),
        (
            "ed25519 public key",
            hex::encode(ed_pk),
            "e6270f673064bc4a2fdee5eb47b231e5b7839dba7f9eb3971698df40120d4eaf",
        ),
        (
            "ed25519 chain code",
            hex::encode(ed_chain),
            "7396b17d22eece8c401d725452fcbd81ec95ac831e6367d8dc106dd35a80fb12",
        ),
        (
            "secp256k1 public key",
            hex::encode(secp_pk),
            "038b280f39569515385d59f2ad4f45abd825936032e346b91b0ab2b83ba904309c",
        ),
        (
            "secp256k1 chain code",
            hex::encode(secp_chain),
            "e5f73570a480e0a622b82f2fadc736d1db31774ee01db94b42df289975ce1c93",
        ),
        (
            "ed25519 signature",
            hex::encode(crypto::ed25519_sign_message(&root, path(), b"hello")),
            "1a1dbf9ea1c147ef69d989cab403f22eb34271564182ae656981e4d2ce63be04b1ae5f10962613c9b4591b33947cc105ac6d3c1e314f69caa1dc13ad86948602",
        ),
        (
            "bip340 signature",
            hex::encode(crypto::secp256k1_sign_message_bip340(
                &root,
                path(),
                b"hello",
            )),
            "11206670daea72d027253a2b02a82bccc6cd69cbdaef4985325bac1d94d80c7800f1063a38f8cbf10775e0cd4d4f07fdf20e4392b9f5c6e08094d0d4930c0801",
        ),
        (
            "ecdsa message signature",
            hex::encode(crypto::secp256k1_sign_message_ecdsa(
                &root,
                path(),
                b"hello",
            )),
            "1447211ff7d577a120e1be9589837594a7169dc1a4ee82011f22e0d40d7c4645610f6da198d1e430441ae595ce100529ed226f906bf78fe5a6068829f9ea6e33",
        ),
        (
            "ecdsa digest signature",
            hex::encode(crypto::secp256k1_sign_digest_ecdsa(&root, path(), &[7; 32])),
            "b54d059be45e04476a8e9237139f7bb1d2803dcff2c3cc615026eedecf3cfb2c271b8996d448c651bb568f7d6d6d7d6f0f5e6d0ba8075691e0d881a4423b60fe",
        ),
    ];
    for (name, actual, expected) in vectors {
        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn aes_path_concatenation_remains_gateway_compatible() {
    let root = [3; 48];
    let left = vec![b"ab".to_vec(), b"c".to_vec()];
    let right = vec![b"a".to_vec(), b"bc".to_vec()];
    assert_eq!(
        crypto::a256gcm_key(&root, left.clone()),
        crypto::a256gcm_key(&root, right.clone())
    );
    assert_ne!(
        crypto::ed25519_public_key(&root, left.clone()),
        crypto::ed25519_public_key(&root, right.clone())
    );
    assert_ne!(
        crypto::secp256k1_public_key(&root, left),
        crypto::secp256k1_public_key(&root, right)
    );
}
