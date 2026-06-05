//! Fetch Resources Extension for Anda Engine
//!
//! This module provides functionality to fetch resources from URLs, allowing
//! the engine to retrieve content from web endpoints and return it as strings.
//!
//! # Features
//! - Fetch resources from any HTTPS URL
//! - Automatic content type handling
//! - UTF-8 string conversion with base64 fallback for binary content
//! - Integration with Anda's HTTP features
//!
//! # Usage
//! ```rust,ignore
//! let fetch_tool = FetchResourcesTool::new();
//! // Manual invocation within an agent
//! let content = FetchResourcesTool::fetch(ctx, "https://example.com/api/data").await?;
//! // Or register with Engine for automatic invocation
//! let engine = Engine::builder()
//!     .with_name("MyEngine".to_string())
//!     .register_tool(fetch_tool)?
//!     .register_agent(my_agent, None)?
//!     .build("default_agent".to_string())?;
//! ```

use anda_core::{
    BoxError, FunctionDefinition, HttpFeatures, Json, Resource, Tool, ToolOutput, gen_schema_for,
};
use encoding_rs::Encoding;
use http::header;
use ic_auth_types::ByteBufB64;
use mime::Mime;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::context::BaseCtx;

/// Arguments for fetching resources from a URL
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct FetchWebResourcesArgs {
    /// The URL to fetch resources from
    pub url: String,
}

/// Fetch Resources Tool implementation
///
/// Provides functionality to fetch content from web URLs and return it as a string.
/// If the content is not valid UTF-8, it will be base64-url encoded.
///
/// # Content Handling
/// - UTF-8 text content is returned as-is
/// - Binary content is automatically base64-url encoded
/// - Supports various content types including HTML, JSON, and binary data
///
/// # HTTP Features
/// - Uses GET method for all requests
/// - Sets appropriate Accept headers for broad compatibility
/// - Handles HTTP status codes and error responses
#[derive(Debug, Clone)]
pub struct FetchWebResourcesTool {
    /// JSON schema for the fetch arguments
    schema: Json,
}

impl Default for FetchWebResourcesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl FetchWebResourcesTool {
    pub const NAME: &'static str = "fetch_web_resources";

    /// Creates a new FetchWebResourcesTool instance
    pub fn new() -> Self {
        let schema = gen_schema_for::<FetchWebResourcesArgs>();
        Self { schema }
    }

    /// Fetches content from the specified URL
    ///
    /// # Arguments
    /// * `ctx` - HTTP context for making requests
    /// * `url` - The URL to fetch content from
    ///
    /// # Returns
    /// Response headers and raw bytes of the fetched content or an error
    pub async fn fetch(
        ctx: &impl HttpFeatures,
        url: &str,
    ) -> Result<(header::HeaderMap, Vec<u8>), BoxError> {
        let mut headers = header::HeaderMap::new();

        headers.insert(
            header::ACCEPT,
            "application/json, text/*, */*;q=0.9"
                .parse()
                .expect("invalid header value"),
        );

        let response = ctx
            .https_call(url, http::Method::GET, Some(headers), None)
            .await?;

        if !response.status().is_success() {
            return Err(format!("Fetch failed with status: {}", response.status()).into());
        }
        let headers = response.headers().clone();
        let body = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read response body: {}", e))?;

        Ok((headers, body.to_vec()))
    }

    /// Fetches content from the specified URL and returns it as text (base64-url encoded if not UTF-8)
    ///
    /// # Arguments
    /// * `ctx` - HTTP context for making requests
    /// * `url` - The URL to fetch content from
    ///
    /// # Returns
    /// String content (UTF-8 or base64-url encoded) or an error
    pub async fn fetch_as_text(ctx: &impl HttpFeatures, url: &str) -> Result<String, BoxError> {
        let (headers, body) = Self::fetch(ctx, url).await?;
        match Self::decode_text(&headers, &body) {
            Some(text) => Ok(text),
            None => match String::from_utf8(body) {
                Ok(text) => Ok(text),
                Err(e) => Ok(ByteBufB64(e.into_bytes()).to_string()),
            },
        }
    }

    /// Fetches content from the specified URL and returns it as a byte buffer.
    /// If the content is text and character encoding is not UTF-8, it will be converted to UTF-8.
    ///
    /// # Arguments
    /// * `ctx` - HTTP context for making requests
    /// * `url` - The URL to fetch content from
    ///
    /// # Returns
    /// Base64-url encoded byte buffer or an error
    pub async fn fetch_as_bytes(
        ctx: &impl HttpFeatures,
        url: &str,
    ) -> Result<ByteBufB64, BoxError> {
        let (headers, body) = Self::fetch(ctx, url).await?;
        match Self::decode_text(&headers, &body) {
            Some(text) => Ok(ByteBufB64(text.into_bytes())),
            None => Ok(ByteBufB64(body)),
        }
    }

    /// Decodes text content from bytes using the specified encoding.
    /// If the content is text and character encoding is not UTF-8, it will be converted to UTF-8.
    /// The non-UTF-8 content will be base64-url encoded.
    ///
    /// # Arguments
    /// * `headers` - HTTP headers containing the content type
    /// * `data` - Raw byte data to decode
    ///
    /// # Returns
    /// UTF-8 encoded string if successful, None otherwise
    pub fn decode_text(headers: &header::HeaderMap, data: &[u8]) -> Option<String> {
        let content_type = headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<Mime>().ok());
        if let Some(encoding_name) = content_type
            .as_ref()
            .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()))
            && let Some(encoding) = Encoding::for_label(encoding_name.as_bytes())
        {
            let (text, _, had_errors) = encoding.decode(data);
            if !had_errors {
                return Some(text.into_owned());
            }
        }
        None
    }
}

impl Tool<BaseCtx> for FetchWebResourcesTool {
    type Args = FetchWebResourcesArgs;
    type Output = String;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Fetches resources from a given URL and returns the content as text (base64-url encoded if not UTF-8)".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    /// Executes the fetch operation
    ///
    /// # Arguments
    /// * `ctx` - Base context
    /// * `args` - Fetch arguments containing the URL
    /// * `_resources` - Unused resources parameter
    ///
    /// # Returns
    /// String content (UTF-8 or base64-url encoded) or an error
    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let text = FetchWebResourcesTool::fetch_as_text(&ctx, &args.url).await?;
        Ok(ToolOutput::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;
    use axum::{Router, routing::get};
    use parking_lot::Mutex;
    use serde::de::DeserializeOwned;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct ErrorHttp {
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl HttpFeatures for ErrorHttp {
        async fn https_call(
            &self,
            url: &str,
            method: http::Method,
            headers: Option<header::HeaderMap>,
            body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            assert_eq!(method, http::Method::GET);
            assert!(headers.unwrap().contains_key(header::ACCEPT));
            assert!(body.is_none());
            self.calls.lock().push(url.to_string());
            Err("http disabled".into())
        }

        async fn https_signed_call(
            &self,
            _url: &str,
            _method: http::Method,
            _message_digest: [u8; 32],
            _headers: Option<header::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            Err("not used".into())
        }

        async fn https_signed_rpc<T>(
            &self,
            _endpoint: &str,
            _method: &str,
            _args: impl Serialize + Send,
        ) -> Result<T, BoxError>
        where
            T: DeserializeOwned,
        {
            Err("not used".into())
        }
    }

    #[derive(Clone)]
    struct ReqwestHttp {
        client: reqwest::Client,
    }

    impl ReqwestHttp {
        fn new() -> Self {
            Self {
                client: reqwest::Client::builder().no_proxy().build().unwrap(),
            }
        }
    }

    impl HttpFeatures for ReqwestHttp {
        async fn https_call(
            &self,
            url: &str,
            method: http::Method,
            headers: Option<header::HeaderMap>,
            body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            let mut request = self.client.request(method, url);
            if let Some(headers) = headers {
                request = request.headers(headers);
            }
            if let Some(body) = body {
                request = request.body(body);
            }
            Ok(request.send().await?)
        }

        async fn https_signed_call(
            &self,
            _url: &str,
            _method: http::Method,
            _message_digest: [u8; 32],
            _headers: Option<header::HeaderMap>,
            _body: Option<Vec<u8>>,
        ) -> Result<reqwest::Response, BoxError> {
            Err("not used".into())
        }

        async fn https_signed_rpc<T>(
            &self,
            _endpoint: &str,
            _method: &str,
            _args: impl Serialize + Send,
        ) -> Result<T, BoxError>
        where
            T: DeserializeOwned,
        {
            Err("not used".into())
        }
    }

    async fn spawn_fetch_server() -> String {
        let app = Router::new()
            .route(
                "/latin1",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/plain; charset=windows-1252")],
                        vec![0xE9],
                    )
                }),
            )
            .route(
                "/utf8",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/plain")],
                        "plain utf8".as_bytes().to_vec(),
                    )
                }),
            )
            .route(
                "/binary",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "application/octet-stream")],
                        vec![0xFF, 0xFE],
                    )
                }),
            )
            .route(
                "/missing",
                get(|| async { (http::StatusCode::NOT_FOUND, "missing") }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[test]
    fn decode_text_respects_declared_charset_and_ignores_invalid_headers() {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            "text/plain; charset=windows-1252".parse().unwrap(),
        );
        assert_eq!(
            FetchWebResourcesTool::decode_text(&headers, &[0xE9]).as_deref(),
            Some("é")
        );

        headers.insert(header::CONTENT_TYPE, "text/plain".parse().unwrap());
        assert!(FetchWebResourcesTool::decode_text(&headers, &[0xE9]).is_none());
        headers.insert(
            header::CONTENT_TYPE,
            "text/plain; charset=unknown".parse().unwrap(),
        );
        assert!(FetchWebResourcesTool::decode_text(&headers, b"plain").is_none());
        headers.insert(
            header::CONTENT_TYPE,
            "text/plain; charset=utf-8".parse().unwrap(),
        );
        assert!(FetchWebResourcesTool::decode_text(&headers, &[0xFF]).is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fetch_tool_definition_and_http_error_paths_are_stable() {
        let tool = FetchWebResourcesTool::default();
        assert_eq!(tool.name(), FetchWebResourcesTool::NAME);
        assert!(tool.description().contains("Fetches resources"));
        let definition = tool.definition();
        assert_eq!(definition.name, FetchWebResourcesTool::NAME);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(definition.parameters["type"], "object");

        let http = ErrorHttp::default();
        assert!(
            FetchWebResourcesTool::fetch(&http, "https://example.test/data")
                .await
                .unwrap_err()
                .to_string()
                .contains("http disabled")
        );
        assert!(
            FetchWebResourcesTool::fetch_as_text(&http, "https://example.test/text")
                .await
                .unwrap_err()
                .to_string()
                .contains("http disabled")
        );
        assert!(
            FetchWebResourcesTool::fetch_as_bytes(&http, "https://example.test/bin")
                .await
                .unwrap_err()
                .to_string()
                .contains("http disabled")
        );
        assert_eq!(
            http.calls.lock().clone(),
            vec![
                "https://example.test/data",
                "https://example.test/text",
                "https://example.test/bin"
            ]
        );

        let ctx = EngineBuilder::new().mock_ctx().base;
        assert!(
            tool.call(
                ctx,
                FetchWebResourcesArgs {
                    url: "https://example.test/data".to_string(),
                },
                Vec::new(),
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("not implemented")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fetch_success_paths_decode_text_bytes_and_status_errors() {
        let endpoint = spawn_fetch_server().await;
        let http = ReqwestHttp::new();

        let (headers, body) = FetchWebResourcesTool::fetch(&http, &format!("{endpoint}/utf8"))
            .await
            .unwrap();
        assert_eq!(body, b"plain utf8");
        assert_eq!(
            headers
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/plain")
        );

        assert_eq!(
            FetchWebResourcesTool::fetch_as_text(&http, &format!("{endpoint}/latin1"))
                .await
                .unwrap(),
            "é"
        );
        assert_eq!(
            FetchWebResourcesTool::fetch_as_text(&http, &format!("{endpoint}/utf8"))
                .await
                .unwrap(),
            "plain utf8"
        );
        assert_eq!(
            FetchWebResourcesTool::fetch_as_text(&http, &format!("{endpoint}/binary"))
                .await
                .unwrap(),
            ByteBufB64(vec![0xFF, 0xFE]).to_string()
        );

        assert_eq!(
            FetchWebResourcesTool::fetch_as_bytes(&http, &format!("{endpoint}/latin1"))
                .await
                .unwrap()
                .0,
            "é".as_bytes()
        );
        assert_eq!(
            FetchWebResourcesTool::fetch_as_bytes(&http, &format!("{endpoint}/binary"))
                .await
                .unwrap()
                .0,
            vec![0xFF, 0xFE]
        );

        let err = FetchWebResourcesTool::fetch(&http, &format!("{endpoint}/missing"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("404 Not Found"));
    }
}
