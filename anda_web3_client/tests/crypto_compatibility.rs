use anda_web3_client::crypto;
use ic_tee_gateway_sdk::crypto as gateway;

#[test]
fn all_derivations_match_gateway() {
    let root = [3; 48];
    for path in [
        vec![],
        vec![vec![]],
        vec![b"agent".to_vec(), b"wallet".to_vec()],
        vec![vec![0, 255, 1]],
    ] {
        assert_eq!(
            crypto::a256gcm_key(&root, path.clone()),
            gateway::a256gcm_key(&root, path.clone())
        );
        assert_eq!(
            crypto::ed25519_public_key(&root, path.clone()),
            gateway::ed25519_public_key(&root, path.clone())
        );
        assert_eq!(
            crypto::secp256k1_public_key(&root, path.clone()),
            gateway::secp256k1_public_key(&root, path.clone())
        );
        assert_eq!(
            crypto::ed25519_sign_message(&root, path.clone(), b"hello"),
            gateway::ed25519_sign_message(&root, path.clone(), b"hello")
        );
        assert_eq!(
            crypto::secp256k1_sign_message_bip340(&root, path.clone(), b"hello"),
            gateway::secp256k1_sign_message_bip340(&root, path.clone(), b"hello")
        );
        assert_eq!(
            crypto::secp256k1_sign_message_ecdsa(&root, path.clone(), b"hello"),
            gateway::secp256k1_sign_message_ecdsa(&root, path.clone(), b"hello")
        );
        assert_eq!(
            crypto::secp256k1_sign_digest_ecdsa(&root, path.clone(), &[7; 32]),
            gateway::secp256k1_sign_digest_ecdsa(&root, path, &[7; 32])
        );
    }
}
