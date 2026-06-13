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
    ///    and `expect_digest` do not apply to this path.
    /// 2. A [`SignedEnvelope`] from the `Authorization` header or the
    ///    `ic-auth-*` headers, verified against `expect_target` and
    ///    `expect_digest`.
    ///
    /// Falls back to the anonymous principal when no credential verifies.
    pub fn verify_user(
        &self,
        headers: &http::HeaderMap,
        now_ms: u64,
        expect_target: Option<Principal>,
        expect_digest: Option<&[u8]>,
    ) -> Principal {
        if let Some(cwt) = self.verify_cwt(headers, now_ms) {
            cwt.subject
                .and_then(|s| Principal::from_text(&s).ok())
                .unwrap_or(ANONYMOUS_PRINCIPAL)
        } else if let Some(se) = SignedEnvelope::from_authorization(headers)
            .or_else(|| SignedEnvelope::from_headers(headers))
        {
            match se.verify(now_ms, expect_target, expect_digest) {
                Ok(_) => se.sender(),
                Err(_) => ANONYMOUS_PRINCIPAL,
            }
        } else {
            ANONYMOUS_PRINCIPAL
        }
    }

    /// Verifies a `Bearer` CWT token against the trusted `ed25519_pubkeys`.
    /// Returns `None` when no trusted key is configured, the token is missing,
    /// or verification fails.
    pub fn verify_cwt(&self, headers: &http::HeaderMap, now_ms: u64) -> Option<ClaimsSet> {
        if !self.ed25519_pubkeys.is_empty()
            && let Some(token) = headers.get(AUTHORIZATION)
            && let Ok(token) = token.to_str()
            && let Some(token) = token.strip_prefix("Bearer ")
        {
            let data = ByteBufB64::from_str(token).ok()?;
            let data = skip_prefix(&SIGN1_TAG, &data);
            let cs1 = cose_sign1_from(data, &[], &[], &self.ed25519_pubkeys).ok()?;
            cwt_from(&cs1.payload.unwrap_or_default(), (now_ms / 1000) as i64).ok()
        } else {
            None
        }
    }
}

/// GET /.well-known/information
pub async fn get_information(
    State(app): State<AppState>,
    headers: http::HeaderMap,
) -> impl IntoResponse {
    let caller = app.verify_user(&headers, unix_timestamp().as_millis() as u64, None, None);

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

    let caller = app.verify_user(
        &headers,
        unix_timestamp().as_millis() as u64,
        Some(id),
        Some(hash.as_slice()),
    );

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
        match self {
            Codec::Cbor => {
                from_slice(params).map_err(|err| format!("failed to decode params: {err:?}"))
            }
            Codec::Json => serde_json::from_slice(params)
                .map_err(|err| format!("failed to decode params: {err:?}")),
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
        "information" => codec.encode_result(&engine.information()),
        method => Err(format!(
            "{method} on engine {} not implemented",
            id.to_text()
        )),
    }
}
