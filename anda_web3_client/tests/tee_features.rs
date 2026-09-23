mod support;

use anda_engine::context::Web3ClientFeatures;
use anda_web3_client::tee::{TeeClient, TeeGatewayClientBuilder};
use ic_auth_verifier::identity::BasicIdentity;
use std::sync::Arc;

#[tokio::test]
async fn tee_rpc_preserves_wire_bytes_verifies_signatures_and_bounds_responses() {
    let gateway = TeeGatewayClientBuilder::default()
        .with_identity(Arc::new(BasicIdentity::from_raw_key(&[7; 32])))
        .with_http_client(reqwest::Client::builder().no_proxy().build().unwrap())
        .build();
    let client = TeeClient::new(Arc::new(gateway));
    support::assert_rpc_protocol(&client).await;
}

#[tokio::test]
async fn tee_rejects_invalid_digests_before_contacting_gateway() {
    let client = TeeClient::new(Arc::new(
        TeeGatewayClientBuilder::default()
            .with_tee_host("http://127.0.0.1:1")
            .build(),
    ));
    for length in [0, 16, 31, 33, 64] {
        let digest = vec![7; length];
        let err = client
            .secp256k1_sign_digest_ecdsa(vec![], &digest)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("32-byte digest"));
        let err = client
            .secp256k1_verify_ecdsa(vec![], &digest, &[0; 64])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("32-byte digest"));
    }
}

#[tokio::test]
async fn tee_signed_calls_reject_smuggled_hosts_before_signing() {
    // Without a local identity every signature is a gateway round trip, which
    // would fail with a connection error rather than the URL guard.
    let client = TeeClient::new(Arc::new(
        TeeGatewayClientBuilder::default()
            .with_tee_host("http://127.0.0.1:1")
            .build(),
    ));
    for url in [
        "https://trusted.example@evil.test/rpc",
        "https://trusted.example:token@evil.test/rpc",
        "file:///etc/passwd",
        "not-a-url",
    ] {
        let err = client
            .https_signed_call(url.into(), http::Method::POST, [7; 32], None, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Invalid url"), "{url}: {err}");
        let err = client
            .https_signed_rpc_raw(url.into(), "ping".into(), Vec::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Invalid url"), "{url}: {err}");
    }
}

#[tokio::test]
async fn tee_rpc_can_delegate_envelope_signing_to_the_gateway() {
    use anda_core::{RPCRequest, RPCResponse};
    use axum::{Router, body::Bytes, routing::post};
    use ic_auth_types::ByteBufB64;
    use ic_auth_verifier::{envelope::SignedEnvelope, sha3_256};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let identity = Arc::new(BasicIdentity::from_raw_key(&[7; 32]));
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded_calls = calls.clone();
    let server = support::serve(
        Router::new()
            .route(
                "/identity",
                post(move |body: Bytes| {
                    let calls = recorded_calls.clone();
                    let request: RPCRequest = cbor2::from_slice(&body).unwrap();
                    assert_eq!(request.method, "sign_http");
                    let (digest,): (ByteBufB64,) = cbor2::from_slice(&request.params).unwrap();
                    let envelope =
                        SignedEnvelope::sign_digest(identity.as_ref(), digest.into_vec()).unwrap();
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let result: RPCResponse =
                            Ok(cbor2::to_canonical_vec(&envelope).unwrap().into());
                        cbor2::to_canonical_vec(&result).unwrap()
                    }
                }),
            )
            .route(
                "/rpc",
                post(|headers: http::HeaderMap, body: Bytes| async move {
                    let envelope = SignedEnvelope::from_authorization(&headers).unwrap();
                    envelope.verify(0, None, Some(&sha3_256(&body))).unwrap();
                    let request: RPCRequest = cbor2::from_slice(&body).unwrap();
                    assert_eq!(request.method, "ping");
                    let result: RPCResponse = Ok(request.params);
                    cbor2::to_canonical_vec(&result).unwrap()
                }),
            ),
    )
    .await;
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut gateway = TeeGatewayClientBuilder::default()
        .with_tee_host(&server.url)
        .with_http_client(http.clone())
        .build();
    gateway.http = http;
    let client = TeeClient::new(Arc::new(gateway));
    let args = cbor2::to_canonical_vec(&("payload",)).unwrap();
    let result = client
        .https_signed_rpc_raw(format!("{}/rpc", server.url), "ping".into(), args.clone())
        .await
        .unwrap();
    assert_eq!(result, args);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
