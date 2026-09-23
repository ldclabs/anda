use anda_core::{
    AgentInput, AgentOutput, BoxError, FunctionDefinition, Json, RPCRequest, RPCResponse, Resource,
    Tool, ToolInput, ToolOutput, http_rpc,
};
use anda_core::{CONTENT_TYPE_CBOR, CONTENT_TYPE_JSON, HttpFeatures};
use anda_engine::context::{BaseCtx, Web3SDK};
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
    build_configured_engine(Visibility::Public, Principal::anonymous(), None).await
}

async fn build_configured_engine(
    visibility: Visibility,
    controller: Principal,
    web3: Option<Arc<Web3Client>>,
) -> Arc<Engine> {
    let info = AgentInfo {
        handle: "anda".to_string(),
        name: "Anda".to_string(),
        description: "Test engine".to_string(),
        endpoint: "https://localhost:8443/default".to_string(),
        ..Default::default()
    };
    let mut builder = Engine::builder()
        .with_info(info.clone())
        .with_management(Arc::new(BaseManagement {
            controller,
            managers: BTreeSet::new(),
            visibility,
        }))
        .register_tool(Arc::new(EchoTool("echo_tool")))
        .unwrap()
        .register_tool(Arc::new(EchoTool("hidden_tool")))
        .unwrap()
        .export_tools(vec!["echo_tool".into()])
        .register_agent(Arc::new(EchoEngineInfo::new(info)), None)
        .unwrap();
    if let Some(web3) = web3 {
        builder = builder.with_web3_client(Arc::new(Web3SDK::from_web3(web3)));
    }
    let engine = builder.build("anda".to_string()).await.unwrap();
    Arc::new(engine)
}

struct EchoTool(&'static str);

impl Tool<BaseCtx> for EchoTool {
    type Args = String;
    type Output = String;

    fn name(&self) -> String {
        self.0.into()
    }
    fn description(&self) -> String {
        "Echoes a string".into()
    }
    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: serde_json::json!({"type": "string"}),
            strict: None,
        }
    }
    async fn call(
        &self,
        _ctx: BaseCtx,
        args: String,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<String>, BoxError> {
        if args == "fail" {
            return Err("requested failure".into());
        }
        Ok(ToolOutput::new(args))
    }
}

async fn web3_client(seed: u8) -> Arc<Web3Client> {
    Arc::new(
        Web3Client::builder()
            .with_root_secret([seed; 48])
            .with_allow_http(true)
            .with_http_client(http_client())
            .build()
            .await
            .unwrap(),
    )
}

fn encode(json: bool, value: &impl serde::Serialize) -> Vec<u8> {
    if json {
        serde_json::to_vec(value).unwrap()
    } else {
        cbor2::to_canonical_vec(value).unwrap()
    }
}

fn decode<T: serde::de::DeserializeOwned>(json: bool, bytes: &[u8]) -> T {
    if json {
        serde_json::from_slice(bytes).unwrap()
    } else {
        cbor2::from_slice(bytes).unwrap()
    }
}

async fn call_rpc(
    endpoint: &str,
    json: bool,
    method: &str,
    args: &impl serde::Serialize,
    signer: Option<&Web3Client>,
) -> RPCResponse {
    let body = encode(
        json,
        &RPCRequest {
            method: method.into(),
            params: encode(json, args).into(),
        },
    );
    let mut headers = http::HeaderMap::new();
    if let Some(signer) = signer {
        signer
            .sign_envelope(ic_cose_types::cose::sha3_256(&body))
            .await
            .unwrap()
            .to_authorization(&mut headers)
            .unwrap();
    }
    let content_type = if json {
        CONTENT_TYPE_JSON
    } else {
        CONTENT_TYPE_CBOR
    };
    let response = http_client()
        .post(endpoint)
        .headers(headers)
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], content_type);
    decode(json, &response.bytes().await.unwrap())
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

    let engine = build_engine().await;
    let err = ServerBuilder::new()
        .with_engines(BTreeMap::from([(other, engine)]), None)
        .build_router()
        .unwrap_err();
    assert!(err.to_string().contains(&other.to_text()));
    assert!(err.to_string().contains(&id.to_text()));
    assert!(err.to_string().contains("does not match"));
}

#[tokio::test(flavor = "current_thread")]
async fn multiple_engines_resolve_the_selected_default_and_principal_paths() {
    let first = build_configured_engine(
        Visibility::Public,
        Principal::anonymous(),
        Some(web3_client(11).await),
    )
    .await;
    let second = build_configured_engine(
        Visibility::Public,
        Principal::anonymous(),
        Some(web3_client(12).await),
    )
    .await;
    let default = second.id();
    let ids = [first.id(), second.id()];
    let endpoint = spawn_server(ServerBuilder::new().with_engines(
        BTreeMap::from([(first.id(), first), (second.id(), second)]),
        Some(default),
    ))
    .await;
    let info: AppInformation = http_client()
        .get(format!("{endpoint}/"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(info.default_engine, default);
    assert_eq!(info.engines.len(), 2);
    for (path, expected) in [
        ("default".into(), default),
        (ids[0].to_text(), ids[0]),
        (ids[1].to_text(), ids[1]),
    ] {
        let card: EngineCard = http_rpc(
            &http_client(),
            &format!("{endpoint}/{path}"),
            "information",
            &(),
        )
        .await
        .unwrap();
        assert_eq!(card.id, expected);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_honors_accept_weights_wildcards_and_exclusions() {
    let (endpoint, _) = spawn_default_server().await;
    let client = http_client();
    let cases = [
        (
            "application/json, application/cbor;q=0",
            Some(CONTENT_TYPE_JSON),
        ),
        (
            "application/json;q=0.2, application/cbor;q=0.8",
            Some(CONTENT_TYPE_CBOR),
        ),
        (
            "application/json;q=0.8, application/cbor;q=0.2",
            Some(CONTENT_TYPE_JSON),
        ),
        ("Application/CBOR;Q=1.000", Some(CONTENT_TYPE_CBOR)),
        ("*/*", Some(CONTENT_TYPE_JSON)),
        (
            "application/*;q=0.8, application/json;q=0",
            Some(CONTENT_TYPE_CBOR),
        ),
        ("application/cbor;q=0, */*;q=1", Some(CONTENT_TYPE_JSON)),
        ("application/json;q=0, application/cbor;q=0", None),
        ("text/plain", None),
        ("application/cbor;q=1.001", None),
    ];
    for path in [
        "/",
        "/.well-known/information",
        "/.well-known/agents",
        "/.well-known/agents/default",
    ] {
        for (accept, expected) in cases {
            let response = client
                .get(format!("{endpoint}{path}"))
                .header("accept", accept)
                .send()
                .await
                .unwrap();
            assert_eq!(response.headers()["vary"], "Accept");
            if path != "/.well-known/agents/default" {
                assert_eq!(response.headers()["cache-control"], "no-store");
            }
            match expected {
                Some(content_type) => {
                    assert_eq!(response.status(), 200, "{path}: {accept}");
                    assert_eq!(response.headers()["content-type"], content_type, "{accept}");
                }
                None => assert_eq!(response.status(), 406, "{path}: {accept}"),
            }
        }
    }
    let response = client
        .get(format!("{endpoint}/"))
        .header("accept", "application/json;q=0.1")
        .header("accept", "application/cbor;q=0.9")
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["content-type"], CONTENT_TYPE_CBOR);
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_preserves_metadata_and_vary_under_compression() {
    use anda_engine_server::middleware::CompressionMiddleware;
    let engine = build_engine().await;
    let extra = BTreeMap::from([(
        "nested".into(),
        serde_json::json!({"items": [1, true, "value"]}),
    )]);
    let endpoint = spawn_server(
        ServerBuilder::new()
            .with_engines(BTreeMap::from([(engine.id(), engine)]), None)
            .with_extra_info(extra.clone())
            .with_middleware(CompressionMiddleware::default()),
    )
    .await;
    let client = http_client();
    for json in [false, true] {
        let content_type = if json {
            CONTENT_TYPE_JSON
        } else {
            CONTENT_TYPE_CBOR
        };
        let response = client
            .get(format!("{endpoint}/"))
            .header("accept", content_type)
            .header("accept-encoding", "gzip")
            .send()
            .await
            .unwrap();
        let info: AppInformation = decode(json, &response.bytes().await.unwrap());
        assert_eq!(info.extra_info, extra);
    }
    let response = client
        .get(format!("{endpoint}/.well-known/agents/default"))
        .header("accept-encoding", "gzip")
        .send()
        .await
        .unwrap();
    let vary: Vec<_> = response
        .headers()
        .get_all("vary")
        .iter()
        .flat_map(|value| value.to_str().unwrap().split(',').map(str::trim))
        .collect();
    assert!(
        vary.iter()
            .any(|value| value.eq_ignore_ascii_case("accept"))
    );
    assert!(
        vary.iter()
            .any(|value| value.eq_ignore_ascii_case("accept-encoding"))
    );
    let card: EngineCard = response.json().await.unwrap();
    assert!(
        card.tools
            .iter()
            .any(|tool| tool.definition.name == "echo_tool")
    );
    assert!(
        !card
            .tools
            .iter()
            .any(|tool| tool.definition.name == "hidden_tool")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn tool_rpc_preserves_success_errors_and_visibility_in_both_formats() {
    let (endpoint, _) = spawn_default_server().await;
    for json in [false, true] {
        let input = ToolInput::new("echo_tool".into(), serde_json::json!("hello"));
        let bytes = call_rpc(
            &format!("{endpoint}/default"),
            json,
            "tool_call",
            &(input,),
            None,
        )
        .await
        .unwrap();
        let output: ToolOutput<Json> = decode(json, &bytes);
        assert_eq!(output.output, "hello");
        for (name, args, error) in [
            ("echo_tool", serde_json::json!("fail"), "requested failure"),
            ("echo_tool", serde_json::json!(123), "invalid args"),
            ("missing_tool", Json::Null, "not found"),
            ("hidden_tool", Json::Null, "not found"),
        ] {
            let result = call_rpc(
                &format!("{endpoint}/default"),
                json,
                "tool_call",
                &(ToolInput::new(name.into(), args),),
                None,
            )
            .await;
            assert!(result.unwrap_err().contains(error));
        }
        let result = call_rpc(
            &format!("{endpoint}/default"),
            json,
            "agent_run",
            &(AgentInput::new("missing_agent".into(), "hello".into()),),
            None,
        )
        .await;
        assert!(result.unwrap_err().contains("not found"));
        // Invalid outer argument shape must remain an RPC error in each codec.
        for method in ["tool_call", "agent_run"] {
            let result = call_rpc(&format!("{endpoint}/default"), json, method, &(), None).await;
            assert!(result.unwrap_err().contains("failed to decode params"));
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn private_engine_rejects_anonymous_and_other_users_but_accepts_controller() {
    let controller = web3_client(21).await;
    let user = web3_client(22).await;
    let engine =
        build_configured_engine(Visibility::Private, controller.get_principal(), None).await;
    let endpoint = spawn_server(
        ServerBuilder::new().with_engines(BTreeMap::from([(engine.id(), engine)]), None),
    )
    .await;
    for json in [false, true] {
        for (signer, allowed) in [
            (None, false),
            (Some(user.as_ref()), false),
            (Some(controller.as_ref()), true),
        ] {
            let result = call_rpc(
                &format!("{endpoint}/default"),
                json,
                "agent_run",
                &(AgentInput::new("".into(), "hello".into()),),
                signer,
            )
            .await;
            assert_eq!(result.is_ok(), allowed);
            let result = call_rpc(
                &format!("{endpoint}/default"),
                json,
                "tool_call",
                &(ToolInput::new(
                    "echo_tool".into(),
                    serde_json::json!("hello"),
                ),),
                signer,
            )
            .await;
            assert_eq!(result.is_ok(), allowed);
        }
    }
    let card: EngineCard = http_rpc(
        &http_client(),
        &format!("{endpoint}/default"),
        "information",
        &(),
    )
    .await
    .unwrap();
    assert!(
        card.tools
            .iter()
            .all(|tool| tool.definition.name != "hidden_tool")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn signed_rpc_accepts_scheme_variants_and_rejects_tampered_bodies() {
    let (endpoint, _) = spawn_default_server().await;
    let signer = web3_client(31).await;
    for json in [false, true] {
        let body = encode(
            json,
            &RPCRequest {
                method: "information".into(),
                params: encode(json, &()).into(),
            },
        );
        let envelope = signer
            .sign_envelope(ic_cose_types::cose::sha3_256(&body))
            .await
            .unwrap();
        for scheme in ["ICP", "icp", "IcP "] {
            let content_type = if json {
                CONTENT_TYPE_JSON
            } else {
                CONTENT_TYPE_CBOR
            };
            let response = http_client()
                .post(format!("{endpoint}/default"))
                .header("content-type", content_type)
                .header(
                    "authorization",
                    format!("{scheme} {}", envelope.to_base64()),
                )
                .body(body.clone())
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let result: RPCResponse = decode(json, &response.bytes().await.unwrap());
            assert!(result.is_ok());
        }
        let tampered = encode(
            json,
            &RPCRequest {
                method: "agent_run".into(),
                params: encode(json, &(AgentInput::new("".into(), "tampered".into()),)).into(),
            },
        );
        let response = http_client()
            .post(format!("{endpoint}/default"))
            .header(
                "content-type",
                if json {
                    CONTENT_TYPE_JSON
                } else {
                    CONTENT_TYPE_CBOR
                },
            )
            .header("authorization", format!("ICP {}", envelope.to_base64()))
            .body(tampered)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
        assert!(response.text().await.unwrap().contains("digest"));
    }
    // Custom authentication headers must also identify the caller without caching it.
    let mut headers = http::HeaderMap::new();
    signer
        .sign_envelope([0; 32])
        .await
        .unwrap()
        .to_headers(&mut headers)
        .unwrap();
    let response = http_client()
        .get(format!("{endpoint}/"))
        .headers(headers)
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["cache-control"], "no-store");
    assert_eq!(
        response.json::<AppInformation>().await.unwrap().caller,
        signer.get_principal()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn cwt_authentication_checks_signatures_subjects_and_validity() {
    use ic_cose_types::cose::{
        cwt::ClaimsSet,
        ed25519::{Signer, SigningKey},
        sign1::{EdDSA, cose_sign1},
    };
    fn token(key: &SigningKey, claims: &ClaimsSet) -> String {
        let mut message = cose_sign1(claims.to_vec().unwrap(), EdDSA, None).unwrap();
        let signature = key.sign(&message.prepare_signature(None, None, None).unwrap());
        message
            .set_signature(signature.to_bytes().to_vec())
            .unwrap();
        ByteBufB64::from(message.to_vec().unwrap()).to_string()
    }
    let key = SigningKey::from_bytes(&[41; 32]);
    let other = SigningKey::from_bytes(&[42; 32]);
    let subject = Principal::self_authenticating([43; 32]);
    let now = ic_auth_verifier::unix_timestamp().as_secs();
    let valid = ClaimsSet {
        subject: Some(subject.to_text()),
        expiration: Some((now + 3600).into()),
        ..Default::default()
    };
    let engine = build_configured_engine(Visibility::Private, subject, None).await;
    let endpoint = spawn_server(
        ServerBuilder::new()
            .with_engines(BTreeMap::from([(engine.id(), engine)]), None)
            .with_ed25519_pubkeys(vec![key.verifying_key()]),
    )
    .await;
    for scheme in ["Bearer", "bearer", "BEARER", "Bearer "] {
        let response = http_client()
            .get(format!("{endpoint}/"))
            .header("authorization", format!("{scheme} {}", token(&key, &valid)))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json::<AppInformation>().await.unwrap().caller,
            subject
        );
    }
    for json in [false, true] {
        let body = encode(
            json,
            &RPCRequest {
                method: "agent_run".into(),
                params: encode(json, &(AgentInput::new("".into(), "hello".into()),)).into(),
            },
        );
        let response = http_client()
            .post(format!("{endpoint}/default"))
            .header(
                "content-type",
                if json {
                    CONTENT_TYPE_JSON
                } else {
                    CONTENT_TYPE_CBOR
                },
            )
            .header("authorization", format!("bearer {}", token(&key, &valid)))
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert!(decode::<RPCResponse>(json, &response.bytes().await.unwrap()).is_ok());
    }
    let invalid = [
        token(&other, &valid),
        token(
            &key,
            &ClaimsSet {
                expiration: None,
                ..valid.clone()
            },
        ),
        token(
            &key,
            &ClaimsSet {
                expiration: Some((now - 600).into()),
                ..valid.clone()
            },
        ),
        token(
            &key,
            &ClaimsSet {
                not_before: Some((now + 600).into()),
                ..valid.clone()
            },
        ),
        token(
            &key,
            &ClaimsSet {
                subject: None,
                ..valid.clone()
            },
        ),
        token(
            &key,
            &ClaimsSet {
                subject: Some("not-a-principal".into()),
                ..valid
            },
        ),
    ];
    for token in invalid {
        let response = http_client()
            .get(format!("{endpoint}/"))
            .header("authorization", format!("bearer {token}"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
        assert_eq!(response.headers()["cache-control"], "no-store");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn request_body_limit_and_middleware_order_remain_intact() {
    let engine = build_engine().await;
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut builder =
        ServerBuilder::new().with_engines(BTreeMap::from([(engine.id(), engine)]), None);
    for name in ["A", "B"] {
        let events = events.clone();
        builder = builder.with_request_middleware(move |req, next| {
            let events = events.clone();
            async move {
                events.lock().unwrap().push((name, "before"));
                let response = next.run(req).await;
                events.lock().unwrap().push((name, "after"));
                response
            }
        });
    }
    let endpoint = spawn_server(builder).await;
    assert_eq!(
        http_client()
            .get(format!("{endpoint}/"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            ("B", "before"),
            ("A", "before"),
            ("A", "after"),
            ("B", "after")
        ]
    );
    let response = http_client()
        .post(format!("{endpoint}/default"))
        .header("content-type", CONTENT_TYPE_JSON)
        .body(vec![b' '; 2 * 1024 * 1024 + 1])
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 413);
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
async fn unparseable_credential_is_rejected_not_downgraded_to_anonymous() {
    let (endpoint, _id) = spawn_default_server().await;
    let client = http_client();

    let body = cbor2::to_canonical_vec(&RPCRequest {
        method: "information".to_string(),
        params: ByteBufB64::default(),
    })
    .unwrap();

    // A credential that is present but does not parse must be rejected. Both envelope
    // parsers return `None` for malformed input exactly as they do for an absent header, so
    // falling through to anonymous would let an on-path attacker strip or corrupt one header
    // to launder an authenticated call into an unattributable anonymous one that still runs.
    for credential in [
        // Malformed `ICP` envelope: base64 that is not a valid CBOR envelope.
        ("authorization", "ICP bm90LWFuLWVudmVsb3Bl"),
        // A bearer token when no trusted CWT key is configured.
        ("authorization", "Bearer some-token"),
        // A partial `ic-auth-*` credential (pubkey without signature or digest).
        ("ic-auth-pubkey", "AQEBAQ"),
    ] {
        let res = client
            .post(format!("{endpoint}/default"))
            .header(http::header::CONTENT_TYPE, CONTENT_TYPE_CBOR)
            .header(credential.0, credential.1)
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            401,
            "credential {credential:?} must be rejected, not downgraded to anonymous"
        );
    }

    // No credential at all is still anonymous, which the public engine accepts.
    let res = client
        .post(format!("{endpoint}/default"))
        .header(http::header::CONTENT_TYPE, CONTENT_TYPE_CBOR)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
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
