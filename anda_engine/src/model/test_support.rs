//! Shared HTTP mock scaffolding for model adapter tests.

use axum::{Router, body::Bytes, extract::State, response::IntoResponse, routing::any};
use http::{HeaderMap, HeaderValue, Method, StatusCode, Uri};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct MockResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct RecordedRequest {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

pub(crate) type MockState = Arc<Mutex<(MockResponse, Option<RecordedRequest>)>>;

async fn mock_handler(
    State(state): State<MockState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let mut state = state.lock().unwrap();
    state.1 = Some(RecordedRequest {
        method,
        uri,
        headers,
        body: body.to_vec(),
    });
    let mut response = (state.0.status, state.0.body.clone()).into_response();
    for (name, value) in state.0.headers.iter() {
        response.headers_mut().insert(name, value.clone());
    }
    response
}

/// Serves one canned response for every request and records the last request.
pub(crate) async fn spawn_mock_server(
    status: StatusCode,
    headers: HeaderMap,
    body: impl Into<Vec<u8>>,
) -> (String, MockState) {
    let state = Arc::new(Mutex::new((
        MockResponse {
            status,
            headers,
            body: body.into(),
        },
        None,
    )));
    let app = Router::new()
        .fallback(any(mock_handler))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), state)
}

/// Second slot counts the requests served so retry behavior can be asserted.
pub(crate) type RetryState = Arc<Mutex<(VecDeque<MockResponse>, usize)>>;

async fn retry_mock_handler(
    State(state): State<RetryState>,
    _method: Method,
    _body: Bytes,
) -> impl IntoResponse {
    let mut state = state.lock().unwrap();
    state.1 += 1;
    let mock = state.0.pop_front().expect("mock response should exist");
    let mut response = (mock.status, mock.body).into_response();
    for (name, value) in mock.headers.iter() {
        response.headers_mut().insert(name, value.clone());
    }
    response
}

/// Serves the given responses in order, one per request.
pub(crate) async fn spawn_retry_mock_server(responses: Vec<MockResponse>) -> (String, RetryState) {
    let state = Arc::new(Mutex::new((responses.into(), 0)));
    let app = Router::new()
        .fallback(any(retry_mock_handler))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), state)
}

pub(crate) fn no_proxy_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

pub(crate) fn recorded(state: &MockState) -> RecordedRequest {
    state.lock().unwrap().1.clone().unwrap()
}

pub(crate) fn sse_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers
}
