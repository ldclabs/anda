//! HTTP byte policy under rmcp's lifecycle, request routing, and SSE transport.
//! All protocol messages use rmcp types; this adapter only handles HTTP responses.

use anda_core::BoxError;
use futures::{StreamExt, TryStreamExt};
use http::{HeaderName, HeaderValue};
use reqwest::{Client, Response, StatusCode};
use rmcp::{
    model::{ClientJsonRpcMessage, ClientRequest, ErrorData, ServerJsonRpcMessage},
    transport::{
        common::client_side_sse::BoxedSseResponse,
        streamable_http_client::{
            AuthRequiredError, InsufficientScopeError, StreamableHttpClient, StreamableHttpError,
            StreamableHttpPostResponse,
        },
    },
};
use std::{collections::HashMap, sync::Arc, time::Duration};

#[derive(Debug, thiserror::Error)]
pub(super) enum HttpError {
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error("MCP HTTP response exceeds the byte budget")]
    Limit,
    #[error("MCP HTTP status {0}")]
    Status(u16),
}

type Error = StreamableHttpError<HttpError>;

#[derive(Clone)]
pub(super) struct McpHttpClient {
    client: Client,
    limit: usize,
}
impl McpHttpClient {
    pub fn new(limit: usize) -> Result<Self, BoxError> {
        Ok(Self {
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(15))
                .build()?,
            limit,
        })
    }

    fn request(
        &self,
        method: reqwest::Method,
        uri: &str,
        session: Option<&str>,
        token: Option<&str>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<reqwest::RequestBuilder, Error> {
        let mut request = self
            .client
            .request(method, uri)
            .header("accept", "application/json, text/event-stream");
        for (name, value) in headers {
            if matches!(
                name.as_str(),
                "accept" | "mcp-session-id" | "last-event-id" | "content-length" | "host"
            ) {
                return Err(StreamableHttpError::ReservedHeaderConflict(
                    name.to_string(),
                ));
            }
            request = request.header(name, value);
        }
        if let Some(session) = session {
            request = request.header("mcp-session-id", session);
        }
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        Ok(request)
    }

    fn check_auth(response: &Response) -> Result<(), Error> {
        let status = response.status();
        if !matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
            return Ok(());
        }
        let Some(header) = response
            .headers()
            .get("www-authenticate")
            .and_then(|h| h.to_str().ok())
        else {
            return Err(StreamableHttpError::Client(HttpError::Status(
                status.as_u16(),
            )));
        };
        if status == StatusCode::UNAUTHORIZED {
            Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(
                header.to_string(),
            )))
        } else {
            Err(StreamableHttpError::InsufficientScope(
                InsufficientScopeError::new(header.to_string(), None),
            ))
        }
    }

    async fn body(&self, response: Response) -> Result<Vec<u8>, Error> {
        if response
            .content_length()
            .is_some_and(|n| n > self.limit as u64)
        {
            return Err(StreamableHttpError::Client(HttpError::Limit));
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|err| StreamableHttpError::Client(HttpError::Request(err)))?;
            if chunk.len() > self.limit.saturating_sub(bytes.len()) {
                return Err(StreamableHttpError::Client(HttpError::Limit));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    fn events(&self, response: Response) -> BoxedSseResponse {
        let limit = self.limit;
        let mut budget = EventBudget::default();
        let bounded = response.bytes_stream().map(move |chunk| {
            let chunk = chunk.map_err(HttpError::Request)?;
            budget.observe(&chunk, limit)?;
            Ok::<_, HttpError>(chunk)
        });
        // Move the decoded fields into rmcp's public SSE type at the transport boundary.
        sse_stream::SseByteStream::new(bounded)
            .map_ok(|event| rmcp_sse_stream::Sse {
                event: event.event,
                data: event.data,
                id: event.id,
                retry: event.retry,
            })
            .map_err(|err| match err {
                sse_stream::Error::Body(source) => rmcp_sse_stream::Error::Body(source),
                sse_stream::Error::Utf8Parse(source) => rmcp_sse_stream::Error::Utf8Parse(source),
                _ => rmcp_sse_stream::Error::InvalidLine,
            })
            .boxed()
    }
}

fn media_type_is(header: &str, expected: &str) -> bool {
    header
        .split(';')
        .next()
        .is_some_and(|ty| ty.trim().eq_ignore_ascii_case(expected))
}

/// Count raw event bytes before the SSE parser can accumulate them. CR, LF and
/// split CRLF are supported. Comments count toward the same conservative budget.
#[derive(Default)]
pub(super) struct EventBudget {
    line: usize,
    event: usize,
    was_cr: bool,
}
impl EventBudget {
    pub(super) fn observe(&mut self, bytes: &[u8], limit: usize) -> Result<(), HttpError> {
        for &byte in bytes {
            if self.was_cr {
                self.was_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            if byte == b'\r' || byte == b'\n' {
                if self.line == 0 {
                    self.event = 0;
                } else {
                    self.event = self.event.saturating_add(self.line).saturating_add(1);
                }
                self.line = 0;
                self.was_cr = byte == b'\r';
            } else {
                self.line = self.line.saturating_add(1);
            }
            if self.event.saturating_add(self.line) > limit {
                return Err(HttpError::Limit);
            }
        }
        Ok(())
    }
}

impl StreamableHttpClient for McpHttpClient {
    type Error = HttpError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session: Option<Arc<str>>,
        token: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, Error> {
        let response = self
            .request(
                reqwest::Method::POST,
                &uri,
                session.as_deref(),
                token.as_deref(),
                headers,
            )?
            .json(&message)
            .send()
            .await
            .map_err(|err| StreamableHttpError::Client(HttpError::Request(err)))?;
        Self::check_auth(&response)?;
        let status = response.status();
        if matches!(status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == StatusCode::NOT_FOUND && session.is_some() {
            return Err(StreamableHttpError::SessionExpired);
        }
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let returned_session = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if status.is_success() && media_type_is(&content_type, "text/event-stream") {
            return Ok(StreamableHttpPostResponse::Sse(
                self.events(response),
                returned_session,
            ));
        }
        let body = self.body(response).await?;
        if !status.is_success() {
            // Let the SDK classify legacy discovery rejection, including middleware
            // that uses a different id. Authorization and 5xx never trigger this path.
            if session.is_none()
                && status.is_client_error()
                && let ClientJsonRpcMessage::Request(request) = &message
                && matches!(request.request, ClientRequest::DiscoverRequest(_))
            {
                let error = match serde_json::from_slice::<ServerJsonRpcMessage>(&body) {
                    Ok(ServerJsonRpcMessage::Error(error)) => error.error,
                    _ => ErrorData::invalid_request(
                        format!("discovery rejected: HTTP {status}"),
                        None,
                    ),
                };
                return Ok(StreamableHttpPostResponse::Json(
                    ServerJsonRpcMessage::error(error, Some(request.id.clone())),
                    None,
                ));
            }
            if media_type_is(&content_type, "application/json")
                && let Ok(error @ ServerJsonRpcMessage::Error(_)) = serde_json::from_slice(&body)
            {
                return Ok(StreamableHttpPostResponse::Json(error, returned_session));
            }
            return Err(StreamableHttpError::Client(HttpError::Status(
                status.as_u16(),
            )));
        }
        if !matches!(message, ClientJsonRpcMessage::Request(_)) && body.is_empty() {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if !media_type_is(&content_type, "application/json") {
            return Err(StreamableHttpError::UnexpectedContentType(Some(
                content_type,
            )));
        }
        match serde_json::from_slice(&body) {
            Ok(message) => Ok(StreamableHttpPostResponse::Json(message, returned_session)),
            Err(_) if !matches!(message, ClientJsonRpcMessage::Request(_)) => {
                Ok(StreamableHttpPostResponse::Accepted)
            }
            Err(err) => Err(StreamableHttpError::Deserialize(err)),
        }
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session: Option<Arc<str>>,
        last_event_id: Option<String>,
        token: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxedSseResponse, Error> {
        let mut request = self.request(
            reqwest::Method::GET,
            &uri,
            session.as_deref(),
            token.as_deref(),
            headers,
        )?;
        if let Some(id) = last_event_id {
            request = request.header("last-event-id", id);
        }
        let response = request
            .send()
            .await
            .map_err(|err| StreamableHttpError::Client(HttpError::Request(err)))?;
        Self::check_auth(&response)?;
        if response.status() == StatusCode::METHOD_NOT_ALLOWED {
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        }
        if !response.status().is_success() {
            return Err(StreamableHttpError::Client(HttpError::Status(
                response.status().as_u16(),
            )));
        }
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !media_type_is(content_type, "text/event-stream") {
            return Err(StreamableHttpError::UnexpectedContentType(Some(
                content_type.into(),
            )));
        }
        Ok(self.events(response))
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session: Arc<str>,
        token: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), Error> {
        let response = self
            .request(
                reqwest::Method::DELETE,
                &uri,
                Some(&session),
                token.as_deref(),
                headers,
            )?
            .send()
            .await
            .map_err(|err| StreamableHttpError::Client(HttpError::Request(err)))?;
        Self::check_auth(&response)?;
        if response.status().is_success() || response.status() == StatusCode::METHOD_NOT_ALLOWED {
            Ok(())
        } else {
            Err(StreamableHttpError::Client(HttpError::Status(
                response.status().as_u16(),
            )))
        }
    }
}

/// Retry safe reads only, with the caller's outer deadline spanning every attempt.
pub(super) async fn retry_read<T, F, Fut>(mut operation: F) -> Result<T, BoxError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, rmcp::service::ServiceError>>,
{
    for delay in [250, 1_000] {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) if is_transient(&err) => {
                tokio::time::sleep(Duration::from_millis(delay)).await
            }
            Err(err) => return Err(err.into()),
        }
    }
    Ok(operation().await?)
}

pub(super) fn is_transient(error: &(dyn std::error::Error + 'static)) -> bool {
    use rmcp::service::{ClientInitializeError, ServiceError};
    if let Some(ServiceError::TransportSend(transport)) = error.downcast_ref::<ServiceError>() {
        return is_transient(transport.error.as_ref());
    }
    if let Some(error) = error.downcast_ref::<ClientInitializeError>() {
        return match error {
            ClientInitializeError::TransportError { error, .. } => {
                is_transient(error.error.as_ref())
            }
            ClientInitializeError::LegacyFallbackFailed { fallback, .. } => {
                is_transient(fallback.as_ref())
            }
            _ => false,
        };
    }
    match error.downcast_ref::<StreamableHttpError<HttpError>>() {
        Some(StreamableHttpError::Client(HttpError::Status(408 | 429 | 500 | 502 | 503 | 504))) => {
            true
        }
        Some(StreamableHttpError::Client(HttpError::Request(error))) => {
            error.is_connect() || error.is_timeout()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_types_match_case_insensitively_and_in_full() {
        assert!(media_type_is(
            "Application/JSON; charset=utf-8",
            "application/json"
        ));
        assert!(media_type_is("TEXT/EVENT-STREAM", "text/event-stream"));
        assert!(!media_type_is("application/json-seq", "application/json"));
    }

    #[tokio::test]
    async fn sdk_receives_complete_decoded_events_and_budget_errors() {
        let client = McpHttpClient::new(64).unwrap();
        let chunks = [
            b"id: 42\r".as_slice(),
            b"\nevent: message\nretry: 250\ndata: first\ndata: second\n\n",
        ];
        let body = reqwest::Body::wrap_stream(futures::stream::iter(
            chunks.into_iter().map(Ok::<_, std::io::Error>),
        ));
        let mut events = client.events(Response::from(http::Response::new(body)));
        let event = events.next().await.unwrap().unwrap();
        assert_eq!(event.event.as_deref(), Some("message"));
        assert_eq!(event.data.as_deref(), Some("first\nsecond"));
        assert_eq!(event.id.as_deref(), Some("42"));
        assert_eq!(event.retry, Some(250));
        assert!(events.next().await.is_none());

        let comments = ":a keep-alive comment\n".repeat(4);
        let mut events = client.events(Response::from(http::Response::new(comments)));
        assert!(
            events
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("byte budget")
        );
        assert!(events.next().await.is_none());
    }
}
