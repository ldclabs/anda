//! HTTP utilities for CBOR and Candid RPC calls.
//!
//! This module provides functionality for:
//! - Making CBOR-encoded RPC calls;
//! - Making Candid-encoded canister calls;
//! - Handling HTTP requests and responses;
//! - Error handling for RPC operations.
//!
//! The main types are:
//! - [`RPCRequest`]: Represents a generic RPC request with CBOR-encoded parameters;
//! - [`CanisterRequestRef`]: Represents a canister-specific request with Candid-encoded parameters;
//! - [`RPCResponse`]: Represents a response from an RPC call;
//! - [`HttpRPCError`]: Represents possible errors during RPC operations.
//!
//! The main functions are:
//! - [`http_rpc`]: Makes a generic CBOR-encoded RPC call;
//! - [`canister_rpc`]: Makes a canister-specific RPC call with Candid encoding;
//! - [`cbor_rpc`]: Internal function for making CBOR-encoded HTTP requests.

use candid::{CandidType, Principal, decode_args, encode_args, utils::ArgumentEncoder};
use ciborium::from_reader;
use http::header;
use ic_auth_types::{ByteBufB64, deterministic_cbor_into_vec};
use reqwest::Client;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt::Display;

pub const CONTENT_TYPE_CBOR: &str = "application/cbor";
pub const CONTENT_TYPE_JSON: &str = "application/json";
pub const CONTENT_TYPE_TEXT: &str = "text/plain";

/// Owned RPC request with a method name and CBOR-encoded parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RPCRequest {
    /// The method name to call.
    pub method: String,

    /// CBOR-encoded parameters for the RPC call.
    /// Parameters should be provided as a tuple, where each element represents a single argument.
    /// Examples:
    /// - `()`: No arguments;
    /// - `(1,)`: Single argument;
    /// - `(1, "hello", 3.14)`: Three arguments.
    pub params: ByteBufB64,
}

/// Borrowed RPC request with a method name and CBOR-encoded parameters.
#[derive(Clone, Debug, Serialize)]
pub struct RPCRequestRef<'a> {
    /// The method name to call.
    pub method: &'a str,
    /// CBOR-encoded parameters for the RPC call.
    /// Parameters should be provided as a tuple, where each element represents a single argument.
    /// Examples:
    /// - `()`: No arguments;
    /// - `(1,)`: Single argument;
    /// - `(1, "hello", 3.14)`: Three arguments.
    pub params: &'a ByteBufB64,
}

/// Borrowed request to an ICP canister with Candid-encoded parameters.
#[derive(Clone, Debug, Serialize)]
pub struct CanisterRequestRef<'a> {
    /// The target canister's principal ID
    pub canister: &'a Principal,
    /// The method name to call on the canister
    pub method: &'a str,
    /// Candid-encoded parameters for the canister call.
    /// Parameters should be provided as a tuple, where each element represents a single argument.
    /// Examples:
    /// - `()`: No arguments;
    /// - `(1,)`: Single argument;
    /// - `(1, "hello", 3.14)`: Three arguments.
    pub params: &'a ByteBufB64,
}

/// RPC response payload returned by remote engines.
///
/// `Ok` contains the CBOR or Candid encoded success payload. `Err` contains a
/// remote error message.
pub type RPCResponse = Result<ByteBufB64, String>;

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ListPagination {
//     pub id: String,
//     pub page_token: Option<String>,
//     pub page_size: Option<u16>,
// }

/// Paginated list response.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ListObject<T> {
    pub data: Vec<T>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_size: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// Errors returned by [`http_rpc`], [`canister_rpc`], and [`cbor_rpc`].
#[derive(Debug, thiserror::Error)]
pub enum HttpRPCError {
    #[error("http_rpc({endpoint:?}, {path:?}): send error: {error}")]
    RequestError {
        endpoint: String,
        path: String,
        error: String,
    },

    #[error("http_rpc({endpoint:?}, {path:?}): response status {status}, error: {error}")]
    ResponseError {
        endpoint: String,
        path: String,
        status: u16,
        error: String,
    },

    #[error("http_rpc({endpoint:?}, {path:?}): parse result error: {error}")]
    ResultError {
        endpoint: String,
        path: String,
        error: String,
    },
}

/// Calls a remote CBOR RPC method and decodes its CBOR response payload.
///
/// # Arguments
/// * `client` - HTTP client to use for the request.
/// * `endpoint` - URL endpoint to send the request to.
/// * `method` - RPC method name to call.
/// * `args` - Arguments to serialize as CBOR and send with the request.
///
/// # Returns
/// Result with either the deserialized response or an [`HttpRPCError`].
pub async fn http_rpc<T>(
    client: &Client,
    endpoint: &str,
    method: &str,
    args: &impl Serialize,
) -> Result<T, HttpRPCError>
where
    T: DeserializeOwned,
{
    let args = deterministic_cbor_into_vec(args).map_err(|e| HttpRPCError::RequestError {
        endpoint: endpoint.to_string(),
        path: method.to_string(),
        error: format!("{e:?}"),
    })?;
    let req = RPCRequestRef {
        method,
        params: &args.into(),
    };
    let req = deterministic_cbor_into_vec(&req).map_err(|e| HttpRPCError::RequestError {
        endpoint: endpoint.to_string(),
        path: method.to_string(),
        error: format!("{e:?}"),
    })?;

    let res = cbor_rpc(client, endpoint, method, None, req).await?;
    from_reader(&res[..]).map_err(|e| HttpRPCError::ResultError {
        endpoint: endpoint.to_string(),
        path: method.to_string(),
        error: format!("{e:?}"),
    })
}

/// Calls a canister method through a remote endpoint using Candid-encoded arguments.
///
/// # Arguments
/// * `client` - HTTP client to use for the request.
/// * `endpoint` - URL endpoint to send the request to.
/// * `canister` - Target canister's principal ID.
/// * `method` - Method name to call on the canister.
/// * `args` - Arguments to encode using Candid.
///
/// # Returns
/// Result with either the deserialized response or an [`HttpRPCError`].
pub async fn canister_rpc<In, Out>(
    client: &Client,
    endpoint: &str,
    canister: &Principal,
    method: &str,
    args: In,
) -> Result<Out, HttpRPCError>
where
    In: ArgumentEncoder,
    Out: CandidType + for<'a> candid::Deserialize<'a>,
{
    let args = encode_args(args).map_err(|e| HttpRPCError::RequestError {
        endpoint: format!("{endpoint}/{canister}"),
        path: method.to_string(),
        error: format!("{e:?}"),
    })?;
    let req = deterministic_cbor_into_vec(&CanisterRequestRef {
        canister,
        method,
        params: &ByteBufB64::from(args),
    })
    .map_err(|e| HttpRPCError::RequestError {
        endpoint: endpoint.to_string(),
        path: method.to_string(),
        error: format!("{e:?}"),
    })?;
    let res = cbor_rpc(client, endpoint, canister, None, req).await?;
    let res: (Out,) = decode_args(&res).map_err(|e| HttpRPCError::ResultError {
        endpoint: format!("{endpoint}/{canister}"),
        path: method.to_string(),
        error: format!("{e:?}"),
    })?;
    Ok(res.0)
}

/// Sends a raw CBOR RPC request and returns the remote payload.
///
/// # Arguments
/// * `client` - HTTP client to use for the request.
/// * `endpoint` - URL endpoint to send the request to.
/// * `path` - Path or identifier for the request.
/// * `headers` - Optional headers to include in the request.
/// * `body` - CBOR-encoded request body.
///
/// # Returns
/// Result with either the raw ByteBuf response or an [`HttpRPCError`].
pub async fn cbor_rpc(
    client: &Client,
    endpoint: &str,
    path: impl Display,
    headers: Option<http::HeaderMap>,
    body: Vec<u8>,
) -> Result<ByteBufB64, HttpRPCError> {
    let mut headers = headers.unwrap_or_default();
    let ct: http::HeaderValue = http::HeaderValue::from_static(CONTENT_TYPE_CBOR);
    headers.insert(header::CONTENT_TYPE, ct.clone());
    headers.insert(header::ACCEPT, ct);
    let res = client
        .post(endpoint)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|e| HttpRPCError::RequestError {
            endpoint: endpoint.to_string(),
            path: path.to_string(),
            error: format!("{e:?}"),
        })?;
    let status = res.status().as_u16();
    if status != 200 {
        return Err(HttpRPCError::ResponseError {
            endpoint: endpoint.to_string(),
            path: path.to_string(),
            status,
            error: res.text().await.unwrap_or_default(),
        });
    }

    let data = res.bytes().await.map_err(|e| HttpRPCError::ResultError {
        endpoint: endpoint.to_string(),
        path: path.to_string(),
        error: format!("{e:?}"),
    })?;
    let res: RPCResponse = from_reader(&data[..]).map_err(|e| HttpRPCError::ResultError {
        endpoint: endpoint.to_string(),
        path: path.to_string(),
        error: format!("{e:?}"),
    })?;
    res.map_err(|e| HttpRPCError::ResultError {
        endpoint: endpoint.to_string(),
        path: path.to_string(),
        error: format!("{e:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Bytes, extract::State, response::IntoResponse, routing::post};
    use http::{HeaderMap, StatusCode};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Clone)]
    struct ResponseSpec {
        status: StatusCode,
        body: Vec<u8>,
    }

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        headers: HeaderMap,
        body: Vec<u8>,
    }

    type SharedState = Arc<Mutex<(ResponseSpec, Option<RecordedRequest>)>>;

    struct FailingSerialize;

    impl Serialize for FailingSerialize {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(serde::ser::Error::custom("serialize failed"))
        }
    }

    struct FailingArgs;

    impl ArgumentEncoder for FailingArgs {
        fn encode(self, _ser: &mut candid::ser::IDLBuilder) -> candid::Result<()> {
            Err(candid::Error::msg("encode failed"))
        }

        fn encode_ref(&self, _ser: &mut candid::ser::IDLBuilder) -> candid::Result<()> {
            Err(candid::Error::msg("encode failed"))
        }
    }

    async fn handler(
        State(state): State<SharedState>,
        headers: HeaderMap,
        body: Bytes,
    ) -> impl IntoResponse {
        let mut state = state.lock().unwrap();
        state.1 = Some(RecordedRequest {
            headers,
            body: body.to_vec(),
        });
        (state.0.status, state.0.body.clone())
    }

    async fn spawn_server(status: StatusCode, body: Vec<u8>) -> (String, SharedState) {
        let state = Arc::new(Mutex::new((
            ResponseSpec { status, body },
            None::<RecordedRequest>,
        )));
        let app = Router::new()
            .route("/", post(handler))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(std::future::IntoFuture::into_future(axum::serve(
            listener, app,
        )));
        (format!("http://{addr}"), state)
    }

    async fn spawn_truncated_body_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut req = [0_u8; 1024];
            let _ = stream.read(&mut req).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/cbor\r\ncontent-length: 64\r\n\r\npartial",
                )
                .await
                .unwrap();
            stream.shutdown().await.unwrap();
        });
        format!("http://{addr}")
    }

    fn rpc_response(result: RPCResponse) -> Vec<u8> {
        deterministic_cbor_into_vec(&result).unwrap()
    }

    fn rpc_success<T: Serialize>(value: &T) -> Vec<u8> {
        let payload = deterministic_cbor_into_vec(value).unwrap();
        rpc_response(Ok(ByteBufB64::from(payload)))
    }

    fn client() -> Client {
        Client::builder().no_proxy().build().unwrap()
    }

    fn recorded(state: &SharedState) -> RecordedRequest {
        state.lock().unwrap().1.clone().unwrap()
    }

    #[tokio::test]
    async fn http_rpc_sends_cbor_request_and_decodes_response() {
        let (endpoint, state) =
            spawn_server(StatusCode::OK, rpc_success(&"pong".to_string())).await;
        let output: String = http_rpc(&client(), &endpoint, "ping", &("arg", 7_u8))
            .await
            .unwrap();
        assert_eq!(output, "pong");

        let req = recorded(&state);
        assert_eq!(
            req.headers.get(header::CONTENT_TYPE).unwrap(),
            CONTENT_TYPE_CBOR
        );
        assert_eq!(req.headers.get(header::ACCEPT).unwrap(), CONTENT_TYPE_CBOR);
        let decoded: RPCRequest = from_reader(&req.body[..]).unwrap();
        assert_eq!(decoded.method, "ping");
        let args: (String, u8) = from_reader(&decoded.params.0[..]).unwrap();
        assert_eq!(args, ("arg".to_string(), 7));
    }

    #[tokio::test]
    async fn canister_rpc_sends_cbor_wrapped_candid_and_decodes_response() {
        let encoded = encode_args(("hello".to_string(),)).unwrap();
        let (endpoint, state) =
            spawn_server(StatusCode::OK, rpc_response(Ok(ByteBufB64::from(encoded)))).await;
        let canister = Principal::anonymous();

        let output: String = canister_rpc(&client(), &endpoint, &canister, "greet", ("anda",))
            .await
            .unwrap();
        assert_eq!(output, "hello");

        let req = recorded(&state);
        let value: ciborium::value::Value = from_reader(&req.body[..]).unwrap();
        let text = format!("{value:?}");
        assert!(text.contains("greet"));
    }

    #[tokio::test]
    async fn cbor_rpc_reports_http_remote_and_decode_errors() {
        let (endpoint, _) = spawn_server(StatusCode::BAD_REQUEST, b"bad request".to_vec()).await;
        let err = cbor_rpc(&client(), &endpoint, "path", None, Vec::new())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            HttpRPCError::ResponseError {
                status: 400,
                error,
                ..
            } if error == "bad request"
        ));

        let (endpoint, _) = spawn_server(
            StatusCode::OK,
            rpc_response(Err("remote failed".to_string())),
        )
        .await;
        let err = cbor_rpc(&client(), &endpoint, "path", None, Vec::new())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            HttpRPCError::ResultError { error, .. } if error.contains("remote failed")
        ));

        let (endpoint, _) = spawn_server(StatusCode::OK, b"not cbor".to_vec()).await;
        let err = cbor_rpc(&client(), &endpoint, "path", None, Vec::new())
            .await
            .unwrap_err();
        assert!(matches!(err, HttpRPCError::ResultError { .. }));
    }

    #[tokio::test]
    async fn http_and_canister_rpc_report_payload_decode_errors() {
        let (endpoint, _) = spawn_server(StatusCode::OK, rpc_success(&"not a number")).await;
        let err = http_rpc::<u64>(&client(), &endpoint, "number", &()).await;
        assert!(matches!(err, Err(HttpRPCError::ResultError { .. })));

        let encoded = encode_args(("not a number".to_string(),)).unwrap();
        let (endpoint, _) =
            spawn_server(StatusCode::OK, rpc_response(Ok(ByteBufB64::from(encoded)))).await;
        let err =
            canister_rpc::<_, u64>(&client(), &endpoint, &Principal::anonymous(), "number", ())
                .await;
        assert!(matches!(err, Err(HttpRPCError::ResultError { .. })));
    }

    #[tokio::test]
    async fn request_encoding_errors_are_reported_before_sending() {
        let err = http_rpc::<String>(
            &client(),
            "http://127.0.0.1:1",
            "serialize",
            &FailingSerialize,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            err,
            HttpRPCError::RequestError {
                path,
                error,
                ..
            } if path == "serialize" && error.contains("serialize failed")
        ));

        let err = canister_rpc::<_, String>(
            &client(),
            "http://127.0.0.1:1",
            &Principal::anonymous(),
            "encode",
            FailingArgs,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            err,
            HttpRPCError::RequestError {
                path,
                error,
                ..
            } if path == "encode" && error.contains("encode failed")
        ));
    }

    #[tokio::test]
    async fn cbor_rpc_reports_body_read_errors() {
        let endpoint = spawn_truncated_body_server().await;
        let err = cbor_rpc(&client(), &endpoint, "body", None, Vec::new())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            HttpRPCError::ResultError { ref path, .. } if path == "body"
        ));
    }

    #[tokio::test]
    async fn cbor_rpc_reports_send_errors() {
        let err = cbor_rpc(
            &client(),
            "http://127.0.0.1:1",
            "unreachable",
            None,
            Vec::new(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, HttpRPCError::RequestError { .. }));
        assert!(err.to_string().contains("unreachable"));
    }
}
