//! Axum handlers for engine information and signed RPC calls.
//!
//! The handlers verify CWT or signed-envelope credentials, resolve the target
//! engine, and dispatch typed CBOR/JSON payloads into the engine runtime.

use anda_core::{AgentInput, Json, RPCRequest, ToolInput};
use anda_engine::engine::Engine;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use candid::Principal;
use cbor2::{from_slice, to_canonical_vec};
use http::header::AUTHORIZATION;
use ic_auth_types::ByteBufB64;
use ic_auth_verifier::{
    envelope::{ANONYMOUS_PRINCIPAL, SignedEnvelope},
    unix_timestamp,
};
use ic_cose_types::cose::{
    SIGN1_TAG,
    cwt::{ClaimsSet, cwt_from},
    ed25519::VerifyingKey,
    sign1::cose_sign1_from,
    skip_prefix,
};
use ic_tee_agent::{
    RPCResponse,
    http::{Content, ContentWithSHA3},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, str::FromStr, sync::Arc};

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
    ///    attempted when at least one trusted key is configured.
    /// 2. A [`SignedEnvelope`] from the `Authorization` header or the
    ///    `ic-auth-*` headers, verified against `expect_target` and
    ///    `expect_digest`.
    ///
    /// On the signed-RPC path (`expect_digest` is `Some`) the envelope must
    /// carry its own `digest`; a digest-less envelope is rejected instead of
    /// letting [`SignedEnvelope::verify`] fall back to the server-computed body
    /// hash. This is a fail-closed hygiene check: it forces the client to
    /// explicitly commit to a body hash so the server always exercises the
    /// `digest == body_hash` equality path.
    ///
    /// It is **not**, on its own, a cryptographic defense. The signature is
    /// verified over the same 32-byte body hash whether or not `digest` is
    /// present, and the anda RPC payload (`{method, params}`) binds neither the
    /// target engine nor a domain tag. An attacker able to obtain a signature
    /// over that hash (e.g. a signing oracle sharing the key) can still pass by
    /// echoing the hash as `digest`. Genuine cross-protocol / oracle resistance
    /// requires domain separation in the signature scheme (`ic_auth_verifier`),
    /// which is out of scope for this crate.
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
        // Bearer CWT path, only when trusted keys are configured. A `Bearer ` token present
        // here is a CWT attempt and must verify.
        if !self.ed25519_pubkeys.is_empty()
            && let Some(token) = headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
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
        if let Some(se) = SignedEnvelope::from_authorization(headers)
            .or_else(|| SignedEnvelope::from_headers(headers))
        {
            // Fail-closed on a body-bound RPC: require the client to present its
            // own committed digest rather than leaning on the server-computed
            // body hash. This is hygiene, not a standalone crypto defense — see
            // the `verify_user` docs for why the signing scheme must add domain
            // separation to actually resist an oracle sharing the key.
            if expect_digest.is_some() && se.digest.is_none() {
                return Err("signed request is missing the content digest".to_string());
            }
            return match se.verify(now_ms, expect_target, expect_digest) {
                Ok(_) => Ok(se.sender()),
                Err(err) => Err(format!("invalid request credential: {err}")),
            };
        }

        // No credential supplied: treat as anonymous.
        Ok(ANONYMOUS_PRINCIPAL)
    }

    /// Verifies a raw `Bearer` CWT token string against the trusted `ed25519_pubkeys`.
    /// Returns `None` when no trusted key is configured or verification fails.
    fn verify_cwt_token(&self, token: &str, now_ms: u64) -> Option<ClaimsSet> {
        if self.ed25519_pubkeys.is_empty() {
            return None;
        }
        let data = ByteBufB64::from_str(token).ok()?;
        let data = skip_prefix(&SIGN1_TAG, &data);
        let cs1 = cose_sign1_from(data, &[], &[], &self.ed25519_pubkeys).ok()?;
        cwt_from(&cs1.payload.unwrap_or_default(), (now_ms / 1000) as i64).ok()
    }
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
    let caller = match app.verify_user(&headers, unix_timestamp().as_millis() as u64, None, None) {
        Ok(caller) => caller,
        Err(err) => return (StatusCode::UNAUTHORIZED, err).into_response(),
    };

    let info = AppInformation {
        engines: app.engines.values().map(|e| e.info().clone()).collect(),
        default_engine: app.default_engine,
        start_time_ms: app.start_time_ms,
        caller,
        extra_info: app.extra_info.as_ref().clone(),
    };

    match Content::from(&headers) {
        Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
        _ => Content::JSON(info, None).into_response(),
    }
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
        Some(engine) => {
            let info = engine.information();
            match Content::from(&headers) {
                Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
                _ => Content::JSON(info, None).into_response(),
            }
        }
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
    ct: ContentWithSHA3<RPCRequest>,
) -> impl IntoResponse {
    let id = match resolve_engine_id(&app, &id) {
        Ok(id) => id,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };

    let (codec, req, hash) = match &ct {
        ContentWithSHA3::CBOR(req, hash) => (Codec::Cbor, req, hash),
        ContentWithSHA3::JSON(req, hash) => (Codec::Json, req, hash),
    };

    let caller = match app.verify_user(
        &headers,
        unix_timestamp().as_millis() as u64,
        Some(id),
        Some(hash.as_slice()),
    ) {
        Ok(caller) => caller,
        Err(err) => return (StatusCode::UNAUTHORIZED, err).into_response(),
    };

    let res = engine_run(codec, req, &app, caller, id).await;
    match codec {
        Codec::Cbor => Content::CBOR(res, None).into_response(),
        Codec::Json => Content::JSON(res, None).into_response(),
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

impl Codec {
    fn decode_params<T: DeserializeOwned>(self, params: &[u8]) -> Result<T, String> {
        // Use `Display` (not `Debug`) so decode errors report the parser's own
        // message without echoing raw request bytes back to the caller.
        match self {
            Codec::Cbor => {
                from_slice(params).map_err(|err| format!("failed to decode params: {err}"))
            }
            Codec::Json => serde_json::from_slice(params)
                .map_err(|err| format!("failed to decode params: {err}")),
        }
    }

    fn encode_result<T: Serialize>(self, value: &T) -> Result<ByteBufB64, String> {
        match self {
            Codec::Cbor => to_canonical_vec(value)
                .map(ByteBufB64::from)
                .map_err(|err| format!("failed to encode result: {err:?}")),
            Codec::Json => serde_json::to_vec(value)
                .map(ByteBufB64::from)
                .map_err(|err| format!("failed to encode result: {err:?}")),
        }
    }
}

fn log_rpc(
    method: &str,
    engine: &Principal,
    caller: &Principal,
    start: std::time::Instant,
    name: &str,
    error: Option<&str>,
) {
    log::info!(
        method = method,
        agent = engine.to_text(),
        caller = caller.to_text(),
        elapsed = start.elapsed().as_millis(),
        name = name,
        error = error;
        "",
    );
}

async fn engine_run(
    codec: Codec,
    req: &RPCRequest,
    app: &AppState,
    caller: Principal,
    id: Principal,
) -> RPCResponse {
    let engine = app
        .engines
        .get(&id)
        .ok_or_else(|| format!("engine {} not found", id.to_text()))?;

    let start = std::time::Instant::now();
    match req.method.as_str() {
        "agent_run" => {
            let args: (AgentInput,) = codec.decode_params(req.params.as_slice())?;
            let name = args.0.name.clone();
            let res = engine
                .agent_run(caller, args.0)
                .await
                .map_err(|err| format!("failed to run agent: {err:?}"));
            log_rpc(
                req.method.as_str(),
                &id,
                &caller,
                start,
                &name,
                res.as_ref().err().map(String::as_str),
            );

            codec.encode_result(&res?)
        }
        "tool_call" => {
            let args: (ToolInput<Json>,) = codec.decode_params(req.params.as_slice())?;
            let name = args.0.name.clone();
            let res = engine
                .tool_call(caller, args.0)
                .await
                .map_err(|err| format!("failed to call tool: {err:?}"));
            log_rpc(
                req.method.as_str(),
                &id,
                &caller,
                start,
                &name,
                res.as_ref().err().map(String::as_str),
            );

            codec.encode_result(&res?)
        }
        // Discovery method: like the `.well-known` routes, it is intentionally
        // open to anonymous callers and returns only the public `EngineCard`
        // (exported agents/tools). Per-caller access is enforced by `agent_run`
        // and `tool_call` below via the engine's `check_visibility`.
        "information" => codec.encode_result(&engine.information()),
        method => Err(format!(
            "{method} on engine {} not implemented",
            id.to_text()
        )),
    }
}
