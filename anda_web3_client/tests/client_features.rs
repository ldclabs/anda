mod support;

use anda_core::{HttpFeatures, RPCRequest, RPCResponse};
use anda_engine::context::Web3ClientFeatures;
use anda_web3_client::{
    Agent, Client, Identity, identity_from_pem, identity_from_secret, load_identity,
};
use axum::{Router, body::Bytes, http::StatusCode, response::IntoResponse, routing::post};
use candid::Principal;
use cbor2::to_canonical_vec;
use ic_auth_types::ByteBufB64;
use ic_auth_verifier::{envelope::SignedEnvelope, sha3_256};
use ic_cose::client::CoseSDK;
use ic_cose_types::CanisterCaller;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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

async fn rpc_handler(headers: http::HeaderMap, body: Bytes) -> impl IntoResponse {
    if body.is_empty() {
        return (StatusCode::BAD_REQUEST, "missing body".as_bytes().to_vec());
    }
    let envelope = SignedEnvelope::from_authorization(&headers).unwrap();
    envelope.verify(0, None, Some(&sha3_256(&body))).unwrap();
    assert_eq!(envelope.sender(), boxed_identity([7; 32]).sender().unwrap());
    let request: RPCRequest = cbor2::from_slice(&body).unwrap();
    assert_eq!(request.method, "ping");
    assert_eq!(
        request.params.as_slice(),
        to_canonical_vec(&("arg",)).unwrap()
    );
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
        .with_root_secret([9; 48])
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
    // The userinfo forms below read as the trusted host but resolve to `evil.test`;
    // accepting them would deliver an identity-signed request to the attacker.
    let permissive = client_with_identity(true).await;
    for bad in [
        "file:///etc/passwd",
        "ftp://example.test/x",
        "data:text/plain,hello",
        "ws://example.test/socket",
        "not-a-url",
        "https://",
        "https://example.test@evil.test/rpc",
        "https://example.test:token@evil.test/rpc",
        "http://example.test@evil.test/rpc",
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
        to_canonical_vec(&("arg",)).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(cbor2::from_slice::<String>(&raw).unwrap(), "pong");
}

#[tokio::test]
async fn missing_secrets_fail_without_breaking_identity_only_requests() {
    assert!(
        Client::builder()
            .build()
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("configure an identity")
    );
    assert!(
        Client::builder()
            .with_root_secret([0; 48])
            .build()
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("all zeros")
    );
    let client = Client::builder()
        .with_identity(boxed_identity([7; 32]))
        .with_http_client(no_proxy_http_client())
        .with_allow_http(true)
        .build()
        .await
        .unwrap();
    let signature = [0; 64];
    let results = [
        client.a256gcm_key(vec![]).await.map(|_| ()),
        client.ed25519_public_key(vec![]).await.map(|_| ()),
        client
            .ed25519_sign_message(vec![], b"message")
            .await
            .map(|_| ()),
        client.ed25519_verify(vec![], b"message", &signature).await,
        client.secp256k1_public_key(vec![]).await.map(|_| ()),
        client
            .secp256k1_sign_message_bip340(vec![], b"message")
            .await
            .map(|_| ()),
        client
            .secp256k1_verify_bip340(vec![], b"message", &signature)
            .await,
        client
            .secp256k1_sign_message_ecdsa(vec![], b"message")
            .await
            .map(|_| ()),
        client
            .secp256k1_sign_digest_ecdsa(vec![], &[7; 32])
            .await
            .map(|_| ()),
        client
            .secp256k1_verify_ecdsa(vec![], &[7; 32], &signature)
            .await,
    ];
    for result in results {
        assert!(result.unwrap_err().to_string().contains("with_root_secret"));
    }
    support::assert_rpc_protocol(&client).await;
}

#[tokio::test]
async fn mismatched_agent_identity_is_rejected() {
    let agent = Agent::builder()
        .with_url("https://icp-api.io")
        .with_arc_identity(boxed_identity([2; 32]))
        .build()
        .unwrap();
    let error = Client::builder()
        .with_identity(boxed_identity([1; 32]))
        .with_agent(agent)
        .build()
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("agent principal does not match"));
}

#[tokio::test]
async fn ecdsa_digest_contract_matches_message_signing() {
    let client = client_with_identity(false).await;
    for length in [0, 16, 31, 33, 64] {
        let digest = vec![7; length];
        assert!(
            client
                .secp256k1_sign_digest_ecdsa(vec![], &digest)
                .await
                .unwrap_err()
                .to_string()
                .contains("32-byte digest")
        );
        assert!(
            client
                .secp256k1_verify_ecdsa(vec![], &digest, &[0; 64])
                .await
                .unwrap_err()
                .to_string()
                .contains("32-byte digest")
        );
    }
    let message = b"message";
    let digest = ic_auth_verifier::sha256(message);
    let from_message = client
        .secp256k1_sign_message_ecdsa(vec![], message)
        .await
        .unwrap();
    let from_digest = client
        .secp256k1_sign_digest_ecdsa(vec![], &digest)
        .await
        .unwrap();
    assert_eq!(from_message, from_digest);
    client
        .secp256k1_verify_ecdsa(vec![], &digest, &from_message)
        .await
        .unwrap();
}

fn replica_status() -> Vec<u8> {
    use cbor2::Value;
    to_canonical_vec(&Value::Map(vec![
        (Value::Text("root_key".into()), Value::Bytes(vec![1; 96])),
        (
            Value::Text("replica_health_status".into()),
            Value::Text("healthy".into()),
        ),
    ]))
    .unwrap()
}

#[tokio::test]
async fn replica_root_key_failure_is_reported_and_rebuilding_recovers() {
    let healthy = Arc::new(AtomicBool::new(false));
    let state = healthy.clone();
    let server = support::serve(Router::new().route(
        "/api/v2/status",
        axum::routing::get(move || {
            let state = state.clone();
            async move {
                if state.load(Ordering::SeqCst) {
                    (StatusCode::OK, replica_status())
                } else {
                    (StatusCode::BAD_REQUEST, vec![])
                }
            }
        }),
    ))
    .await;
    let make_client = || {
        Client::builder()
            .with_ic_host(&server.url)
            .with_root_secret([9; 48])
    };
    let error = make_client().build().await.err().unwrap();
    assert!(
        error
            .to_string()
            .contains("failed to fetch replica root key")
    );
    healthy.store(true, Ordering::SeqCst);
    assert!(make_client().build().await.is_ok());
}

#[tokio::test(start_paused = true)]
async fn replica_root_key_fetch_has_a_setup_deadline() {
    // Keep the port open but never send an HTTP response.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let start = tokio::time::Instant::now();
    let error = Client::builder()
        .with_ic_host(&url)
        .with_root_secret([9; 48])
        .build()
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("timed out after 10 seconds"));
    assert_eq!(start.elapsed(), std::time::Duration::from_secs(10));
}

#[tokio::test]
async fn default_agent_verifies_queries_and_custom_agent_can_opt_out() {
    let read_state_called = Arc::new(AtomicBool::new(false));
    let state = read_state_called.clone();
    let server = support::serve(
        Router::new()
            .route(
                "/api/v2/status",
                axum::routing::get(|| async { replica_status() }),
            )
            .route(
                "/api/v3/canister/{id}/query",
                post(|| async {
                    use cbor2::Value;
                    to_canonical_vec(&Value::Map(vec![
                        (Value::Text("status".into()), Value::Text("replied".into())),
                        (
                            Value::Text("reply".into()),
                            Value::Map(vec![(
                                Value::Text("arg".into()),
                                Value::Bytes(candid::encode_one("unsigned").unwrap()),
                            )]),
                        ),
                    ]))
                    .unwrap()
                }),
            )
            .fallback(move || {
                let state = state.clone();
                async move {
                    state.store(true, Ordering::SeqCst);
                    StatusCode::NOT_FOUND
                }
            }),
    )
    .await;
    let identity = boxed_identity([7; 32]);
    let client = Client::builder()
        .with_ic_host(&server.url)
        .with_identity(identity.clone())
        .build()
        .await
        .unwrap();
    let result: Result<String, _> =
        CanisterCaller::canister_query(&client, &Principal::anonymous(), "greet", ()).await;
    assert!(result.is_err());
    assert!(read_state_called.load(Ordering::SeqCst));

    read_state_called.store(false, Ordering::SeqCst);
    let agent = Agent::builder()
        .with_url(&server.url)
        .with_arc_identity(identity.clone())
        .with_verify_query_signatures(false)
        .build()
        .unwrap();
    let client = Client::builder()
        .with_identity(identity)
        .with_agent(agent)
        .build()
        .await
        .unwrap();
    let result: String =
        CanisterCaller::canister_query(&client, &Principal::anonymous(), "greet", ())
            .await
            .unwrap();
    assert_eq!(result, "unsigned");
    assert!(!read_state_called.load(Ordering::SeqCst));
}
