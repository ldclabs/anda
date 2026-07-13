use anda_core::{HttpFeatures, RPCResponse};
use anda_engine::context::Web3ClientFeatures;
use anda_web3_client::{
    Agent, Client, Identity, identity_from_pem, identity_from_secret, load_identity,
};
use axum::{Router, body::Bytes, http::StatusCode, response::IntoResponse, routing::post};
use candid::Principal;
use cbor2::to_canonical_vec;
use ic_auth_types::ByteBufB64;
use ic_cose::client::CoseSDK;
use ic_cose_types::CanisterCaller;
use std::sync::Arc;

const SECP256K1_IDENTITY_PEM: &str = "-----BEGIN EC PARAMETERS-----
BgUrgQQACg==
-----END EC PARAMETERS-----
-----BEGIN EC PRIVATE KEY-----
MHQCAQEEIAgy7nZEcVHkQ4Z1Kdqby8SwyAiyKDQmtbEHTIM+WNeBoAcGBSuBBAAK
oUQDQgAEgO87rJ1ozzdMvJyZQ+GABDqUxGLvgnAnTlcInV3NuhuPv4O3VGzMGzeB
N3d26cRxD99TPtm8uo2OuzKhSiq6EQ==
-----END EC PRIVATE KEY-----
";

fn boxed_identity(secret: [u8; 32]) -> Arc<dyn Identity> {
    Arc::from(identity_from_secret(secret))
}

fn no_proxy_http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

async fn client_with_identity(allow_http: bool) -> Client {
    Client::builder()
        .with_identity(boxed_identity([7; 32]))
        .with_root_secret([9; 48])
        .with_http_client(no_proxy_http_client())
        .with_allow_http(allow_http)
        .build()
        .await
        .unwrap()
}

fn rpc_response(result: RPCResponse) -> Vec<u8> {
    to_canonical_vec(&result).unwrap()
}

async fn rpc_handler(body: Bytes) -> impl IntoResponse {
    if body.is_empty() {
        return (StatusCode::BAD_REQUEST, "missing body".as_bytes().to_vec());
    }
    let payload = to_canonical_vec(&"pong".to_string()).unwrap();
    (StatusCode::OK, rpc_response(Ok(ByteBufB64::from(payload))))
}

async fn echo_handler(body: Bytes) -> impl IntoResponse {
    if body.is_empty() {
        (StatusCode::OK, b"ok".to_vec())
    } else {
        (StatusCode::OK, body.to_vec())
    }
}

async fn spawn_server() -> String {
    let app = Router::new()
        .route("/echo", post(echo_handler))
        .route("/rpc", post(rpc_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test(flavor = "current_thread")]
async fn identities_and_builder_options_are_stable() {
    let anonymous = load_identity("Anonymous").unwrap();
    assert_eq!(anonymous.sender().unwrap(), Principal::anonymous());

    let secret_hex = hex::encode([3_u8; 32]);
    let from_hex = load_identity(&secret_hex).unwrap();
    assert_eq!(
        from_hex.sender().unwrap(),
        identity_from_secret([3_u8; 32]).sender().unwrap()
    );
    let invalid = match load_identity("abcd") {
        Ok(_) => panic!("expected invalid short hex secret to fail"),
        Err(err) => err,
    };
    assert!(invalid.to_string().contains("invalid id_secret"));
    assert!(identity_from_pem("/definitely/missing/identity.pem").is_err());

    let identity = boxed_identity([4; 32]);
    let agent = Agent::builder()
        .with_url("https://icp-api.io")
        .with_arc_identity(identity.clone())
        .build()
        .unwrap();
    let cose_canister = Principal::from_text("aaaaa-aa").unwrap();
    let client = Client::builder()
        .with_ic_host("https://icp-api.io")
        .with_root_secret([1; 48])
        .with_cose_canister(cose_canister)
        .with_identity(identity.clone())
        .with_agent(agent)
        .with_http_client(no_proxy_http_client())
        .with_allow_http(true)
        .build()
        .await
        .unwrap();

    assert_eq!(client.get_principal(), identity.sender().unwrap());
    assert_eq!(CoseSDK::canister(&client), &cose_canister);
}

#[tokio::test(flavor = "current_thread")]
async fn pem_default_builder_and_canister_error_paths_are_exercised() {
    let pem_path = std::env::temp_dir().join(format!(
        "anda-web3-identity-{}-{}.pem",
        std::process::id(),
        "secp256k1"
    ));
    std::fs::write(&pem_path, SECP256K1_IDENTITY_PEM).unwrap();
    let from_pem = identity_from_pem(pem_path.to_str().unwrap()).unwrap();
    let loaded_from_pem = load_identity(pem_path.to_str().unwrap()).unwrap();
    assert_eq!(
        from_pem.sender().unwrap(),
        loaded_from_pem.sender().unwrap()
    );
    std::fs::remove_file(&pem_path).unwrap();

    let endpoint = spawn_server().await;
    let default_client = Client::builder()
        .with_ic_host(&endpoint)
        .with_allow_http(true)
        .build()
        .await
        .unwrap();
    assert_ne!(default_client.get_principal(), Principal::anonymous());

    let identity = boxed_identity([8; 32]);
    let agent = Agent::builder()
        .with_url(endpoint)
        .with_arc_identity(identity.clone())
        .with_verify_query_signatures(false)
        .build()
        .unwrap();
    let client = Client::builder()
        .with_identity(identity)
        .with_agent(agent)
        .with_http_client(no_proxy_http_client())
        .with_allow_http(true)
        .build()
        .await
        .unwrap();
    let canister = Principal::anonymous();

    let query: Result<String, _> =
        CanisterCaller::canister_query(&client, &canister, "greet", ("world",)).await;
    assert!(query.is_err());

    let update: Result<String, _> =
        CanisterCaller::canister_update(&client, &canister, "greet", ("world",)).await;
    assert!(update.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn crypto_feature_methods_sign_verify_and_reject_bad_signatures() {
    let client = client_with_identity(false).await;
    let path = vec![b"agent".to_vec(), b"tool".to_vec()];
    let message = b"message";

    assert_eq!(
        client
            .sign_envelope([1; 32])
            .await
            .unwrap()
            .digest
            .unwrap()
            .0,
        [1; 32]
    );
    assert_eq!(
        Web3ClientFeatures::get_principal(&client),
        client.get_principal()
    );
    assert_eq!(
        Web3ClientFeatures::sign_envelope(&client, [2; 32])
            .await
            .unwrap()
            .digest
            .unwrap()
            .0,
        [2; 32]
    );

    assert_eq!(
        Web3ClientFeatures::a256gcm_key(&client, path.clone())
            .await
            .unwrap()
            .len(),
        32
    );

    let ed_sig = Web3ClientFeatures::ed25519_sign_message(&client, path.clone(), message)
        .await
        .unwrap();
    Web3ClientFeatures::ed25519_verify(&client, path.clone(), message, &ed_sig)
        .await
        .unwrap();
    assert!(
        Web3ClientFeatures::ed25519_verify(&client, path.clone(), b"wrong", &ed_sig)
            .await
            .is_err()
    );
    assert_eq!(
        Web3ClientFeatures::ed25519_public_key(&client, path.clone())
            .await
            .unwrap()
            .len(),
        32
    );

    let schnorr = Web3ClientFeatures::secp256k1_sign_message_bip340(&client, path.clone(), message)
        .await
        .unwrap();
    Web3ClientFeatures::secp256k1_verify_bip340(&client, path.clone(), message, &schnorr)
        .await
        .unwrap();
    assert!(
        Web3ClientFeatures::secp256k1_verify_bip340(&client, path.clone(), b"wrong", &schnorr)
            .await
            .is_err()
    );

    let ecdsa_message =
        Web3ClientFeatures::secp256k1_sign_message_ecdsa(&client, path.clone(), message)
            .await
            .unwrap();
    assert_eq!(ecdsa_message.len(), 64);
    let digest = [5_u8; 32];
    let ecdsa_digest =
        Web3ClientFeatures::secp256k1_sign_digest_ecdsa(&client, path.clone(), &digest)
            .await
            .unwrap();
    Web3ClientFeatures::secp256k1_verify_ecdsa(&client, path.clone(), &digest, &ecdsa_digest)
        .await
        .unwrap();
    assert!(
        Web3ClientFeatures::secp256k1_verify_ecdsa(
            &client,
            path.clone(),
            &[6_u8; 32],
            &ecdsa_digest
        )
        .await
        .is_err()
    );
    assert_eq!(
        Web3ClientFeatures::secp256k1_public_key(&client, path)
            .await
            .unwrap()
            .len(),
        33
    );
}

#[tokio::test(flavor = "current_thread")]
async fn http_guards_local_calls_and_signed_rpc_paths_are_exercised() {
    let guarded = client_with_identity(false).await;
    assert!(
        HttpFeatures::https_call(
            &guarded,
            "http://example.test",
            http::Method::GET,
            None,
            None
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Invalid url")
    );
    assert!(
        HttpFeatures::https_signed_call(
            &guarded,
            "http://example.test",
            http::Method::GET,
            [0; 32],
            None,
            None,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Invalid url")
    );
    let guarded_rpc: Result<String, _> =
        HttpFeatures::https_signed_rpc(&guarded, "http://example.test/rpc", "ping", &()).await;
    assert!(guarded_rpc.unwrap_err().to_string().contains("Invalid url"));

    // Non-http(s) schemes and malformed URLs are rejected even when
    // `allow_http` is enabled, so an attacker cannot smuggle `file://` /
    // `data:` / metadata-style targets through a signed call. `allow_http` only
    // opens the plain `http` scheme.
    let permissive = client_with_identity(true).await;
    for bad in [
        "file:///etc/passwd",
        "ftp://example.test/x",
        "data:text/plain,hello",
        "ws://example.test/socket",
        "not-a-url",
        "https://",
    ] {
        let err = HttpFeatures::https_call(&permissive, bad, http::Method::GET, None, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Invalid url"),
            "expected {bad:?} to be rejected, got: {err}"
        );
    }

    assert!(
        Web3ClientFeatures::https_call(
            &guarded,
            "http://example.test".to_string(),
            http::Method::GET,
            None,
            None,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Invalid url")
    );
    assert!(
        Web3ClientFeatures::https_signed_call(
            &guarded,
            "http://example.test".to_string(),
            http::Method::GET,
            [0; 32],
            None,
            None,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Invalid url")
    );
    assert!(
        Web3ClientFeatures::https_signed_rpc_raw(
            &guarded,
            "http://example.test/rpc".to_string(),
            "ping".to_string(),
            Vec::new(),
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Invalid url")
    );

    let endpoint = spawn_server().await;
    let client = client_with_identity(true).await;
    let mut headers = http::HeaderMap::new();
    headers.insert("x-test", "1".parse().unwrap());

    let res = HttpFeatures::https_call(
        &client,
        &format!("{endpoint}/echo"),
        http::Method::POST,
        Some(headers.clone()),
        Some(b"plain".to_vec()),
    )
    .await
    .unwrap();
    assert_eq!(res.text().await.unwrap(), "plain");

    let res = HttpFeatures::https_signed_call(
        &client,
        &format!("{endpoint}/echo"),
        http::Method::POST,
        [8; 32],
        Some(headers.clone()),
        Some(b"signed".to_vec()),
    )
    .await
    .unwrap();
    assert_eq!(res.text().await.unwrap(), "signed");

    let res = Web3ClientFeatures::https_call(
        &client,
        format!("{endpoint}/echo"),
        http::Method::POST,
        Some(headers.clone()),
        None,
    )
    .await
    .unwrap();
    assert_eq!(res.text().await.unwrap(), "ok");

    let res = Web3ClientFeatures::https_signed_call(
        &client,
        format!("{endpoint}/echo"),
        http::Method::POST,
        [9; 32],
        Some(headers),
        Some(b"raw-signed".to_vec()),
    )
    .await
    .unwrap();
    assert_eq!(res.text().await.unwrap(), "raw-signed");

    let rpc: String =
        HttpFeatures::https_signed_rpc(&client, &format!("{endpoint}/rpc"), "ping", &("arg",))
            .await
            .unwrap();
    assert_eq!(rpc, "pong");

    let raw = Web3ClientFeatures::https_signed_rpc_raw(
        &client,
        format!("{endpoint}/rpc"),
        "ping".into(),
        vec![1, 2],
    )
    .await
    .unwrap();
    assert!(!raw.is_empty());
}
