use anda_core::{AgentInput, AgentOutput, RPCRequest, RPCResponse, http_rpc};
use anda_core::{CONTENT_TYPE_CBOR, CONTENT_TYPE_JSON, HttpFeatures};
use anda_engine::engine::{AgentInfo, EchoEngineInfo, Engine, EngineCard};
use anda_engine::management::{BaseManagement, Visibility};
use anda_engine_server::{ServerBuilder, middleware::ApiKeyMiddleware, types::AppInformation};
use anda_web3_client::client::Client as Web3Client;
use candid::Principal;
use ic_auth_types::ByteBufB64;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

async fn build_engine() -> Arc<Engine> {
    let info = AgentInfo {
        handle: "anda".to_string(),
        name: "Anda".to_string(),
        description: "Test engine".to_string(),
        endpoint: "https://localhost:8443/default".to_string(),
        ..Default::default()
    };
    let engine = Engine::builder()
        .with_info(info.clone())
        .with_management(Arc::new(BaseManagement {
            controller: Principal::anonymous(),
            managers: BTreeSet::new(),
            visibility: Visibility::Public,
        }))
        .register_agent(Arc::new(EchoEngineInfo::new(info)), None)
        .unwrap()
        .build("anda".to_string())
        .await
        .unwrap();
    Arc::new(engine)
}

async fn spawn_server(builder: ServerBuilder) -> String {
    let app = builder.build_router().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn spawn_default_server() -> (String, Principal) {
    let engine = build_engine().await;
    let id = engine.id();
    let endpoint =
        spawn_server(ServerBuilder::new().with_engines(BTreeMap::from([(id, engine)]), None)).await;
    (endpoint, id)
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn build_router_validates_engines() {
    let err = ServerBuilder::new().build_router().unwrap_err();
    assert!(err.to_string().contains("no engines registered"));

    let engine = build_engine().await;
    let id = engine.id();
    let other = Principal::from_text("aaaaa-aa").unwrap();
    let err = ServerBuilder::new()
        .with_engines(BTreeMap::from([(id, engine)]), Some(other))
        .build_router()
        .unwrap_err();
    assert!(err.to_string().contains("default engine not found"));
}

#[tokio::test(flavor = "current_thread")]
async fn information_endpoints_serve_json_and_cbor() {
    let (endpoint, id) = spawn_default_server().await;
    let client = http_client();

    // default content type is JSON, caller is anonymous
    for path in ["/", "/.well-known/information", "/.well-known/agents"] {
        let res = client
            .get(format!("{endpoint}{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let info: AppInformation = res.json().await.unwrap();
        assert_eq!(info.default_engine, id);
        assert_eq!(info.caller, Principal::anonymous());
        assert_eq!(info.engines.len(), 1);
        assert_eq!(info.engines[0].handle, "anda");
    }

    // CBOR via Accept header
    let res = client
        .get(format!("{endpoint}/"))
        .header(http::header::ACCEPT, CONTENT_TYPE_CBOR)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(
        res.headers().get(http::header::CONTENT_TYPE).unwrap(),
        CONTENT_TYPE_CBOR
    );
    let body = res.bytes().await.unwrap();
    let info: AppInformation = cbor2::from_slice(&body[..]).unwrap();
    assert_eq!(info.default_engine, id);

    // engine information endpoint
    for path in [
        "/.well-known/agents/default".to_string(),
        format!("/.well-known/agents/{}", id.to_text()),
    ] {
        let res = client
            .get(format!("{endpoint}{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let card: EngineCard = res.json().await.unwrap();
        assert_eq!(card.id, id);
        assert_eq!(card.info.handle, "anda");
        assert!(card.agents.iter().any(|f| f.definition.name == "anda"));
    }

    let res = client
        .get(format!("{endpoint}/.well-known/agents/not-a-principal"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 400);
    assert!(res.text().await.unwrap().contains("invalid engine id"));

    let res = client
        .get(format!("{endpoint}/.well-known/agents/aaaaa-aa"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 404);
    assert!(res.text().await.unwrap().contains("not found"));
}

#[tokio::test(flavor = "current_thread")]
async fn rpc_handles_cbor_requests_and_errors() {
    let (endpoint, id) = spawn_default_server().await;
    let client = http_client();
    let url = format!("{endpoint}/default");

    let card: EngineCard = http_rpc(&client, &url, "information", &()).await.unwrap();
    assert_eq!(card.id, id);

    let output: AgentOutput = http_rpc(
        &client,
        &url,
        "agent_run",
        &(AgentInput::new("".to_string(), "hello".to_string()),),
    )
    .await
    .unwrap();
    assert!(output.content.contains("anda"));

    // engine id in the path is also supported
    let card: EngineCard = http_rpc(
        &client,
        &format!("{endpoint}/{}", id.to_text()),
        "information",
        &(),
    )
    .await
    .unwrap();
    assert_eq!(card.id, id);

    let err = http_rpc::<EngineCard>(&client, &url, "bogus_method", &())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not implemented"));

    let err = http_rpc::<EngineCard>(&client, &format!("{endpoint}/aaaaa-aa"), "information", &())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"));

    // invalid engine id in the path is rejected before dispatch
    let body = cbor2::to_canonical_vec(&RPCRequest {
        method: "information".to_string(),
        params: ByteBufB64::default(),
    })
    .unwrap();
    let res = client
        .post(format!("{endpoint}/not-a-principal"))
        .header(http::header::CONTENT_TYPE, CONTENT_TYPE_CBOR)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 400);
    assert!(res.text().await.unwrap().contains("invalid engine id"));
}

#[tokio::test(flavor = "current_thread")]
async fn rpc_handles_json_requests() {
    let (endpoint, id) = spawn_default_server().await;
    let client = http_client();

    let params =
        serde_json::to_vec(&(AgentInput::new("".to_string(), "hello".to_string()),)).unwrap();
    let req = RPCRequest {
        method: "agent_run".to_string(),
        params: ByteBufB64::from(params),
    };
    let res = client
        .post(format!("{endpoint}/default"))
        .header(http::header::CONTENT_TYPE, CONTENT_TYPE_JSON)
        .body(serde_json::to_vec(&req).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(
        res.headers().get(http::header::CONTENT_TYPE).unwrap(),
        CONTENT_TYPE_JSON
    );
    let body = res.bytes().await.unwrap();
    let res: RPCResponse = serde_json::from_slice(&body).unwrap();
    let output: AgentOutput = serde_json::from_slice(&res.unwrap()).unwrap();
    assert!(output.content.contains("anda"));

    let req = RPCRequest {
        method: "information".to_string(),
        params: ByteBufB64::from(serde_json::to_vec(&()).unwrap()),
    };
    let res = client
        .post(format!("{endpoint}/default"))
        .header(http::header::CONTENT_TYPE, CONTENT_TYPE_JSON)
        .body(serde_json::to_vec(&req).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let body = res.bytes().await.unwrap();
    let res: RPCResponse = serde_json::from_slice(&body).unwrap();
    let card: EngineCard = serde_json::from_slice(&res.unwrap()).unwrap();
    assert_eq!(card.id, id);
}

#[tokio::test(flavor = "current_thread")]
async fn signed_requests_identify_the_caller() {
    let (endpoint, _id) = spawn_default_server().await;
    let web3 = Web3Client::builder()
        .with_allow_http(true)
        .with_http_client(http_client())
        .with_root_secret([7; 48])
        .build()
        .await
        .unwrap();
    let principal = web3.get_principal();
    assert_ne!(principal, Principal::anonymous());

    // signed GET: the server verifies the envelope and reports the caller
    let res = web3
        .https_signed_call(
            &format!("{endpoint}/"),
            http::Method::GET,
            [0; 32],
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let info: AppInformation = res.json().await.unwrap();
    assert_eq!(info.caller, principal);

    // signed RPC: body digest is bound to the envelope
    let output: AgentOutput = web3
        .https_signed_rpc(
            &format!("{endpoint}/default"),
            "agent_run",
            &(AgentInput::new("".to_string(), "hello".to_string()),),
        )
        .await
        .unwrap();
    assert!(output.content.contains("anda"));
}

#[tokio::test(flavor = "current_thread")]
async fn signed_rpc_rejects_envelope_without_digest() {
    use ic_auth_verifier::envelope::SignedEnvelope;

    let (endpoint, _id) = spawn_default_server().await;
    let client = http_client();

    // An envelope that carries no committed digest must be rejected on the RPC
    // path (before signature verification), so a signature can never be bound
    // to a server-computed body hash instead of the body the client signed.
    let envelope = SignedEnvelope {
        pubkey: ByteBufB64::from(vec![1u8; 32]),
        signature: ByteBufB64::from(vec![2u8; 64]),
        digest: None,
        delegation: None,
    };
    let mut auth_headers = http::HeaderMap::new();
    envelope.to_authorization(&mut auth_headers).unwrap();
    let auth = auth_headers
        .get(http::header::AUTHORIZATION)
        .unwrap()
        .clone();

    let body = cbor2::to_canonical_vec(&RPCRequest {
        method: "information".to_string(),
        params: ByteBufB64::default(),
    })
    .unwrap();
    let res = client
        .post(format!("{endpoint}/default"))
        .header(http::header::CONTENT_TYPE, CONTENT_TYPE_CBOR)
        .header(http::header::AUTHORIZATION, auth)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 401);
    assert!(
        res.text()
            .await
            .unwrap()
            .contains("missing the content digest")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn api_key_middleware_guards_requests() {
    let engine = build_engine().await;
    let id = engine.id();
    let endpoint = spawn_server(
        ServerBuilder::new()
            .with_engines(BTreeMap::from([(id, engine)]), None)
            .with_middleware(ApiKeyMiddleware::new("secret-key").exempt_path("/")),
    )
    .await;
    let client = http_client();

    // exempt path works without a key
    let res = client.get(format!("{endpoint}/")).send().await.unwrap();
    assert_eq!(res.status(), 200);

    let guarded = format!("{endpoint}/.well-known/information");
    let res = client.get(&guarded).send().await.unwrap();
    assert_eq!(res.status(), 401);

    let res = client
        .get(&guarded)
        .header("x-api-key", "wrong-key")
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 401);

    let res = client
        .get(&guarded)
        .header("x-api-key", "secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
}

#[tokio::test(flavor = "current_thread")]
async fn api_key_middleware_exempts_prefix() {
    let engine = build_engine().await;
    let id = engine.id();
    let endpoint = spawn_server(
        ServerBuilder::new()
            .with_engines(BTreeMap::from([(id, engine)]), None)
            .with_middleware(ApiKeyMiddleware::new("secret-key").exempt_prefix("/.well-known/")),
    )
    .await;
    let client = http_client();

    // Both the static and the dynamic-segment discovery routes under the
    // exempt prefix bypass the key.
    for path in [
        "/.well-known/information".to_string(),
        "/.well-known/agents".to_string(),
        format!("/.well-known/agents/{}", id.to_text()),
    ] {
        let res = client
            .get(format!("{endpoint}{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200, "path {path} should be exempt");
    }

    // A path outside the prefix is still guarded.
    let res = client.get(format!("{endpoint}/")).send().await.unwrap();
    assert_eq!(res.status(), 401);
}
