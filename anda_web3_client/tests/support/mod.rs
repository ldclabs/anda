use anda_core::{MAX_RPC_RESPONSE_BYTES, RPCRequest, RPCResponse};
use anda_engine::context::Web3ClientFeatures;
use axum::{Router, body::Bytes, http::HeaderMap, routing::post};
use ic_auth_types::ByteBufB64;
use ic_auth_verifier::{envelope::SignedEnvelope, sha3_256};

pub struct Server {
    pub url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub async fn serve(app: Router) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Server { url, task }
}

pub async fn assert_rpc_protocol(client: &dyn Web3ClientFeatures) {
    let principal = client.get_principal();
    // A valid non-canonical CBOR item must survive the raw boundary unchanged.
    let params = vec![0x81, 0x18, 0x01];
    let expected_params = params.clone();
    let success = post(move |headers: HeaderMap, body: Bytes| {
        let expected_params = expected_params.clone();
        async move {
            assert_eq!(headers["content-type"], "application/cbor");
            assert_eq!(headers["accept"], "application/cbor");
            let envelope = SignedEnvelope::from_authorization(&headers).unwrap();
            assert_eq!(envelope.sender(), principal);
            envelope.verify(0, None, Some(&sha3_256(&body))).unwrap();
            let request: RPCRequest = cbor2::from_slice(&body).unwrap();
            assert_eq!(request.method, "raw_probe");
            assert_eq!(request.params.as_slice(), expected_params);
            let result: RPCResponse = Ok(ByteBufB64::from(vec![0x18, 0x01]));
            cbor2::to_canonical_vec(&result).unwrap()
        }
    });
    let server = serve(
        Router::new()
            .route("/ok", success)
            .route(
                "/remote-error",
                post(|| async {
                    cbor2::to_canonical_vec(&RPCResponse::Err("denied".into())).unwrap()
                }),
            )
            .route("/invalid", post(|| async { vec![0xff] }))
            .route(
                "/large",
                post(|| async { vec![0; MAX_RPC_RESPONSE_BYTES + 1] }),
            ),
    )
    .await;

    let raw = client
        .https_signed_rpc_raw(
            format!("{}/ok", server.url),
            "raw_probe".into(),
            params.clone(),
        )
        .await
        .unwrap();
    assert_eq!(raw, [0x18, 0x01]);
    for (path, expected) in [
        ("remote-error", "denied"),
        ("invalid", "parse result"),
        ("large", "too large"),
    ] {
        let error = client
            .https_signed_rpc_raw(
                format!("{}/{path}", server.url),
                "raw_probe".into(),
                params.clone(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
