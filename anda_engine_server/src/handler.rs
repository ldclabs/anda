//! Axum handlers for engine information and signed RPC calls.
//!
//! The handlers verify CWT or signed-envelope credentials, resolve the target
//! engine, and dispatch typed CBOR/JSON payloads into the engine runtime.

use anda_core::{
    AgentInput, CONTENT_TYPE_CBOR, CONTENT_TYPE_JSON, Json, RPCRequest, RPCResponse, ToolInput,
};
use anda_engine::engine::Engine;
use axum::{
    body::Bytes,
    extract::{FromRequest, Path, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use candid::Principal;
use cbor2::{from_slice, to_canonical_vec};
use http::{
    HeaderValue,
    header::{ACCEPT, AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE, VARY},
};
use ic_auth_types::ByteBufB64;
use ic_auth_verifier::envelope::{
    ANONYMOUS_PRINCIPAL, HEADER_IC_AUTH_CONTENT_DIGEST, HEADER_IC_AUTH_DELEGATION,
    HEADER_IC_AUTH_PUBKEY, HEADER_IC_AUTH_SIGNATURE, SignedEnvelope,
};
use ic_cose_types::cose::{
    SIGN1_TAG,
    cwt::{ClaimsSet, cwt_from},
    ed25519::VerifyingKey,
    sha3_256,
    sign1::cose_sign1_from,
    skip_prefix,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, str::FromStr, sync::Arc};
use structured_logger::unix_ms;

use crate::types::*;

/// Shared axum application state for engine server routes.
#[derive(Clone)]
pub struct AppState {
    /// Registered engines keyed by principal.
    pub engines: Arc<BTreeMap<Principal, Arc<Engine>>>,
    /// Default engine used by discovery routes.
    pub default_engine: Principal,
    /// Server start timestamp in milliseconds.
    pub start_time_ms: u64,
    /// Additional metadata returned from information endpoints.
    pub extra_info: Arc<BTreeMap<String, Json>>,
    /// Trusted Ed25519 public keys for bearer CWT verification.
    pub ed25519_pubkeys: Arc<Vec<VerifyingKey>>,
}

impl AppState {
    /// Resolves the caller principal from the request headers.
    ///
    /// Credentials are checked in the following order:
    /// 1. A `Bearer` CWT token signed by one of the trusted `ed25519_pubkeys`.
    ///    Bearer tokens are not bound to a single request, so `expect_target`
    ///    and `expect_digest` do not apply to this path. This path is only
    ///    attempted when at least one trusted key is configured. Because the
    ///    token's lifetime is its only containment, a token without an `exp`
    ///    claim is rejected.
    /// 2. A [`SignedEnvelope`] from the `Authorization` header or the
    ///    `ic-auth-*` headers, verified against `expect_target` and
    ///    `expect_digest`.
    ///
    /// Authentication schemes are case-insensitive and allow one or more spaces
    /// before the token. On signed RPCs the envelope must carry a `digest` that
    /// matches the received body. Direct-key signatures provide neither freshness
    /// nor automatic engine binding; clients should set [`anda_core::RequestMeta::engine`].
    /// See the [authentication contract](https://github.com/ldclabs/anda/blob/main/anda_engine_server/README.md#authentication-contract)
    /// for replay, delegation, and bearer-token boundaries.
    ///
    /// Returns the anonymous principal only when no credential is present. When a
    /// credential is present but fails to verify (bad signature, wrong target,
    /// missing or tampered body digest, or expired token), an error is returned
    /// so the caller can reject the request instead of silently downgrading to
    /// anonymous access.
    pub fn verify_user(
        &self,
        headers: &http::HeaderMap,
        now_ms: u64,
        expect_target: Option<Principal>,
        expect_digest: Option<&[u8]>,
    ) -> Result<Principal, String> {
        // A bearer attempt must verify when trusted keys are configured.
        if !self.ed25519_pubkeys.is_empty()
            && let Some(token) = authorization_token(headers, "Bearer")
        {
            let cwt = self
                .verify_cwt_token(token, now_ms)
                .ok_or_else(|| "invalid or expired bearer token".to_string())?;
            return cwt
                .subject
                .and_then(|s| Principal::from_text(&s).ok())
                .ok_or_else(|| "bearer token has no valid subject".to_string());
        }

        // Signed-envelope path from the `Authorization` header or the `ic-auth-*` headers.
        if let Some(se) = authorization_token(headers, "ICP")
            .and_then(|token| SignedEnvelope::from_base64(token).ok())
            .or_else(|| SignedEnvelope::from_headers(headers))
        {
            // Require the client to commit to the body hash explicitly.
            if expect_digest.is_some() && se.digest.is_none() {
                return Err("signed request is missing the content digest".to_string());
            }
            return match se.verify(now_ms, expect_target, expect_digest) {
                Ok(_) => Ok(se.sender()),
                Err(err) => Err(format!("invalid request credential: {err}")),
            };
        }

        // Malformed credentials must not silently become anonymous requests.
        if has_credential_headers(headers) {
            return Err("unparseable request credential".to_string());
        }

        // No credential supplied: treat as anonymous.
        Ok(ANONYMOUS_PRINCIPAL)
    }

    /// Verifies a raw `Bearer` CWT token string against the trusted `ed25519_pubkeys`.
    /// Returns `None` when verification fails.
    ///
    /// A bearer token is not bound to a request body or a target engine, so its only
    /// containment is its lifetime. [`cwt_from`] treats `exp` as optional and accepts a
    /// token with no time claims at all, which would grant an unbounded, unrevocable
    /// credential; this wrapper therefore rejects a token that carries no `exp`.
    fn verify_cwt_token(&self, token: &str, now_ms: u64) -> Option<ClaimsSet> {
        let data = ByteBufB64::from_str(token).ok()?;
        let data = skip_prefix(&SIGN1_TAG, &data);
        let cs1 = cose_sign1_from(data, &[], &[], &self.ed25519_pubkeys).ok()?;
        let claims = cwt_from(&cs1.payload.unwrap_or_default(), (now_ms / 1000) as i64).ok()?;
        claims.expiration.as_ref()?;
        Some(claims)
    }
}

fn authorization_token<'a>(headers: &'a http::HeaderMap, scheme: &str) -> Option<&'a str> {
    let (actual, token) = headers.get(AUTHORIZATION)?.to_str().ok()?.split_once(' ')?;
    actual
        .eq_ignore_ascii_case(scheme)
        .then(|| token.trim_start_matches(' '))
}

/// Returns true when the request carries any credential header, whether or not it parses.
///
/// Used to distinguish absent credentials from ones the parsers could not decode.
fn has_credential_headers(headers: &http::HeaderMap) -> bool {
    headers.contains_key(AUTHORIZATION)
        || headers.contains_key(&HEADER_IC_AUTH_PUBKEY)
        || headers.contains_key(&HEADER_IC_AUTH_SIGNATURE)
        || headers.contains_key(&HEADER_IC_AUTH_CONTENT_DIGEST)
        || headers.contains_key(&HEADER_IC_AUTH_DELEGATION)
}

/// GET /.well-known/information
///
/// Server-level discovery endpoint. Open to anonymous callers by design; it
/// verifies any supplied credential only to echo the resolved `caller`, and
/// exposes each engine's public [`AgentInfo`](anda_engine::engine::AgentInfo)
/// summary rather than any private capability.
pub async fn get_information(
    State(app): State<AppState>,
    headers: http::HeaderMap,
) -> impl IntoResponse {
    let mut response = match app.verify_user(&headers, unix_ms(), None, None) {
        Ok(caller) => discovery_response(
            &headers,
            AppInformationRef {
                engines: app.engines.values().map(|e| e.info()).collect(),
                default_engine: app.default_engine,
                start_time_ms: app.start_time_ms,
                caller,
                extra_info: &app.extra_info,
            },
        ),
        Err(err) => (StatusCode::UNAUTHORIZED, err).into_response(),
    };
    // The caller also varies with custom ic-auth-* headers, which HTTP caches
    // do not automatically treat like Authorization.
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

/// GET /.well-known/agents/{id}
///
/// Discovery endpoint. Following the RFC 8615 `.well-known` convention, it is
/// intentionally open to anonymous callers and does not run `check_visibility`.
/// It returns only the engine's public [`EngineCard`](anda_engine::engine::EngineCard),
/// which exposes exported agents/tools (see [`Engine::information`]); private,
/// non-exported capabilities are never included. Enforcement of per-caller
/// access happens on the RPC path (`agent_run`/`tool_call`) inside the engine.
pub async fn get_engine_information(
    State(app): State<AppState>,
    headers: http::HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let id = match resolve_engine_id(&app, &id) {
        Ok(id) => id,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };

    match app.engines.get(&id) {
        Some(engine) => discovery_response(&headers, engine.information()),
        None => (
            StatusCode::NOT_FOUND,
            format!("engine {} not found", id.to_text()),
        )
            .into_response(),
    }
}

/// POST /{*id}
pub async fn anda_engine(
    State(app): State<AppState>,
    headers: http::HeaderMap,
    Path(id): Path<String>,
    body: RPCRequestBody,
) -> impl IntoResponse {
    let id = match resolve_engine_id(&app, &id) {
        Ok(id) => id,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };

    let RPCRequestBody {
        codec,
        request,
        digest,
    } = body;
    let caller = match app.verify_user(&headers, unix_ms(), Some(id), Some(&digest)) {
        Ok(caller) => caller,
        Err(err) => return (StatusCode::UNAUTHORIZED, err).into_response(),
    };

    let res = engine_run(codec, request, &app, caller, id).await;
    codec.respond(res)
}

/// RPC request extracted from a CBOR or JSON `POST` body.
///
/// The codec is selected from `Content-Type` (`application/cbor`,
/// `application/json`, or a `+cbor`/`+json` suffix) and is also used for the
/// response. The body's SHA3-256 digest is kept for signed-envelope
/// verification. Other content types are rejected with `415`, undecodable
/// bodies with `400`, and bodies over the router's body limit with `413`.
pub struct RPCRequestBody {
    codec: Codec,
    request: RPCRequest,
    digest: [u8; 32],
}

impl<S: Send + Sync> FromRequest<S> for RPCRequestBody {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let codec = Codec::from_content_type(req.headers())
            .ok_or_else(|| StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response())?;
        let body = Bytes::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;
        let request = codec
            .decode(&body)
            .map_err(|err| (StatusCode::BAD_REQUEST, err).into_response())?;
        Ok(Self {
            codec,
            request,
            digest: sha3_256(&body),
        })
    }
}

/// Resolves an engine path segment: either the literal `default` or an engine
/// principal in text format.
fn resolve_engine_id(app: &AppState, id: &str) -> Result<Principal, String> {
    if id == "default" {
        Ok(app.default_engine)
    } else {
        Principal::from_text(id).map_err(|_| format!("invalid engine id: {id:?}"))
    }
}

/// Wire format negotiated from the request content type.
#[derive(Clone, Copy)]
enum Codec {
    Cbor,
    Json,
}

fn discovery_response(headers: &http::HeaderMap, value: impl Serialize) -> Response {
    let mut response = match Codec::from_accept(headers) {
        Some(codec) => codec.respond(value),
        None => (
            StatusCode::NOT_ACCEPTABLE,
            "supported response types: application/json, application/cbor",
        )
            .into_response(),
    };
    // Append so outer compression middleware can retain both selection fields.
    response
        .headers_mut()
        .append(VARY, HeaderValue::from_static("Accept"));
    response
}

/// Parses HTTP quality values as thousandths, without floating-point rounding.
fn parse_quality(value: &str) -> Option<u16> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    if fraction.len() > 3 || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    match whole {
        "0" => Some(
            fraction
                .bytes()
                .fold(0, |n, b| n * 10 + u16::from(b - b'0'))
                * 10u16.pow(3 - fraction.len() as u32),
        ),
        "1" if fraction.bytes().all(|b| b == b'0') => Some(1000),
        _ => None,
    }
}

impl Codec {
    fn from_content_type(headers: &http::HeaderMap) -> Option<Self> {
        let mime: mime::Mime = headers.get(CONTENT_TYPE)?.to_str().ok()?.parse().ok()?;
        if mime.type_() != mime::APPLICATION {
            return None;
        }
        let is = |name: &str| mime.subtype() == name || mime.suffix().is_some_and(|s| s == name);
        if is("cbor") {
            Some(Self::Cbor)
        } else if is("json") {
            Some(Self::Json)
        } else {
            None
        }
    }

    fn from_accept(headers: &http::HeaderMap) -> Option<Self> {
        if !headers.contains_key(ACCEPT) {
            return Some(Self::Json);
        }

        // A specific range overrides a wildcard, including a specific q=0.
        // The server emits unparameterized JSON/CBOR, so other media parameters
        // do not match these representations.
        let mut scores: [Option<(u8, u16)>; 2] = [None, None];
        for value in headers.get_all(ACCEPT) {
            for range in value.to_str().ok()?.split(',') {
                let Ok(range) = range.trim().parse::<mime::Mime>() else {
                    continue;
                };
                if range.params().any(|(name, _)| name != "q") {
                    continue;
                }
                let quality = match range.get_param("q") {
                    Some(q) => match parse_quality(q.as_str()) {
                        Some(q) => q,
                        None => continue,
                    },
                    None => 1000,
                };
                for (index, subtype) in ["json", "cbor"].iter().enumerate() {
                    let specificity = match (range.type_().as_str(), range.subtype().as_str()) {
                        ("*", "*") => 0,
                        ("application", "*") => 1,
                        ("application", actual) if actual == *subtype => 2,
                        _ => continue,
                    };
                    scores[index] = scores[index].max(Some((specificity, quality)));
                }
            }
        }
        let [json, cbor] = scores.map(|score| score.map_or(0, |(_, q)| q));
        match (json, cbor) {
            (0, 0) => None,
            _ if cbor > json => Some(Self::Cbor),
            _ => Some(Self::Json),
        }
    }

    fn respond(self, value: impl Serialize) -> Response {
        let (content_type, body) = match self {
            Self::Cbor => (
                CONTENT_TYPE_CBOR,
                cbor2::to_vec(&value).map_err(|err| err.to_string()),
            ),
            Self::Json => (
                CONTENT_TYPE_JSON,
                serde_json::to_vec(&value).map_err(|err| err.to_string()),
            ),
        };
        match body {
            Ok(body) => (
                [(CONTENT_TYPE, HeaderValue::from_static(content_type))],
                body,
            )
                .into_response(),
            Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
        }
    }

    // Errors use `Display` (not `Debug`) so they report the parser's own
    // message without echoing raw request bytes back to the caller.
    fn decode<T: DeserializeOwned>(self, data: &[u8]) -> Result<T, String> {
        match self {
            Codec::Cbor => from_slice(data).map_err(|err| err.to_string()),
            Codec::Json => serde_json::from_slice(data).map_err(|err| err.to_string()),
        }
    }

    // Consuming the encoded buffer releases it before agent/tool execution awaits.
    fn decode_params<T: DeserializeOwned>(self, params: ByteBufB64) -> Result<T, String> {
        self.decode(params.as_slice())
            .map_err(|err| format!("failed to decode params: {err}"))
    }

    fn encode_result<T: Serialize>(self, value: &T) -> Result<ByteBufB64, String> {
        match self {
            Codec::Cbor => to_canonical_vec(value).map_err(|err| err.to_string()),
            Codec::Json => serde_json::to_vec(value).map_err(|err| err.to_string()),
        }
        .map(ByteBufB64::from)
        .map_err(|err| format!("failed to encode result: {err}"))
    }
}

async fn engine_run(
    codec: Codec,
    req: RPCRequest,
    app: &AppState,
    caller: Principal,
    id: Principal,
) -> RPCResponse {
    let engine = app
        .engines
        .get(&id)
        .ok_or_else(|| format!("engine {} not found", id.to_text()))?;

    let start = std::time::Instant::now();
    let RPCRequest { method, params } = req;
    let (name, res) = match method.as_str() {
        "agent_run" => {
            let (input,): (AgentInput,) = codec.decode_params(params)?;
            let name = input.name.clone();
            let res = engine
                .agent_run(caller, input)
                .await
                .map_err(|err| format!("failed to run agent: {err}"));
            (name, res.and_then(|output| codec.encode_result(&output)))
        }
        "tool_call" => {
            let (input,): (ToolInput<Json>,) = codec.decode_params(params)?;
            let name = input.name.clone();
            let res = engine
                .tool_call(caller, input)
                .await
                .map_err(|err| format!("failed to call tool: {err}"));
            (name, res.and_then(|output| codec.encode_result(&output)))
        }
        // Discovery method: like the `.well-known` routes, it is intentionally
        // open to anonymous callers and returns only the public `EngineCard`
        // (exported agents/tools). Per-caller access is enforced by `agent_run`
        // and `tool_call` via the engine's `check_visibility`.
        "information" => return codec.encode_result(&engine.information()),
        method => {
            return Err(format!(
                "{method} on engine {} not implemented",
                id.to_text()
            ));
        }
    };

    log::info!(
        method = method,
        agent = id.to_text(),
        caller = caller.to_text(),
        elapsed = start.elapsed().as_millis(),
        name = name,
        error = res.as_ref().err();
        "",
    );
    res
}
