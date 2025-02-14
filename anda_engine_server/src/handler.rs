use anda_engine::engine::Engine;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use candid::Principal;
use ciborium::from_reader;
use ic_cose_types::to_cbor_bytes;
use ic_tee_agent::{
    http::{Content, UserSignature, ANONYMOUS_PRINCIPAL},
    RPCRequest, RPCResponse,
};
use serde_bytes::ByteBuf;
use std::collections::BTreeMap;
use std::sync::Arc;
use structured_logger::unix_ms;

use crate::{ic_sig_verifier::verify_sig, types::*};

#[derive(Clone)]
pub struct AppState {
    pub(crate) engines: Arc<BTreeMap<Principal, Engine>>,
    pub(crate) default_engine: Principal,
    pub(crate) start_time_ms: u64,
}

/// GET /.well-known/information
pub async fn get_information(
    State(app): State<AppState>,
    headers: http::HeaderMap,
) -> impl IntoResponse {
    let caller = if let Some(sig) = UserSignature::try_from(&headers) {
        match sig.verify_with(app.default_engine, unix_ms(), verify_sig) {
            Ok(_) => sig.user,
            Err(_) => ANONYMOUS_PRINCIPAL,
        }
    } else {
        ANONYMOUS_PRINCIPAL
    };

    let info = AppInformation {
        engines: app.engines.iter().map(|(_, e)| e.information()).collect(),
        default_engine: app.default_engine,
        start_time_ms: app.start_time_ms,
        caller,
    };

    match Content::from(&headers) {
        Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
        _ => Content::JSON(AppInformationJSON::from(info), None).into_response(),
    }
}

/// POST /{*id}
pub async fn anda_engine(
    State(app): State<AppState>,
    headers: http::HeaderMap,
    Path(id): Path<String>,
    ct: Content<RPCRequest>,
) -> impl IntoResponse {
    let id = Principal::from_text(&id).unwrap_or(app.default_engine);
    let caller = if let Some(sig) = UserSignature::try_from(&headers) {
        match sig.verify_with(id, unix_ms(), verify_sig) {
            Ok(_) => sig.user,
            Err(_) => ANONYMOUS_PRINCIPAL,
        }
    } else {
        ANONYMOUS_PRINCIPAL
    };

    match ct {
        Content::CBOR(req, _) => {
            let res = engine_run(&req, &app, caller, id).await;
            Content::CBOR(res, None).into_response()
        }
        _ => StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response(),
    }
}

async fn engine_run(
    req: &RPCRequest,
    app: &AppState,
    caller: Principal,
    id: Principal,
) -> RPCResponse {
    let engine = app
        .engines
        .get(&id)
        .ok_or_else(|| format!("engine {id} not found"))?;

    match req.method.as_str() {
        "agent_run" => {
            let args: (String, String, Option<ByteBuf>) = from_reader(req.params.as_slice())
                .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine
                .agent_run(
                    Some(args.0),
                    args.1,
                    args.2.map(|v| v.into_vec()),
                    None,
                    Some(caller),
                )
                .await
                .map_err(|err| format!("failed to run agent: {err:?}"))?;
            Ok(to_cbor_bytes(&res).into())
        }
        "tool_call" => {
            let args: (String, String) = from_reader(req.params.as_slice())
                .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine
                .tool_call(args.0, args.1, None, Some(caller))
                .await
                .map_err(|err| format!("failed to call tool: {err:?}"))?;
            Ok(to_cbor_bytes(&res).into())
        }
        _ => Err(format!("engine {id} not implemented")),
    }
}
