//! Google Search Extension for Anda Engine
//!
//! This module provides integration with Google's Custom Search API, allowing
//! the engine to perform web searches and retrieve results.
//!
//! # Features
//! - Perform web searches using Google's Custom Search API
//! - Parse and return structured search results
//! - Configurable number of results
//! - Integration with Anda's HTTP features
//!
//! # Configuration
//! Requires:
//! - Google API Key
//! - Custom Search Engine ID
//!
//! # Usage
//! ```rust,ignore
//! let google = GoogleSearchTool::new(api_key, search_engine_id, Some(5));
//! // Manual invocation within an agent
//! let results = google.search(ctx, SearchArgs { query: "ICPanda" }).await?;
//! // Or register with Engine for automatic invocation
//! let engine = Engine::builder()
//!     .with_name("MyEngine".to_string())
//!     .register_tool(google_search)?
//!     .register_agent(my_agent, None)?
//!     .build("default_agent".to_string())?;
//! ```

use anda_core::{
    BoxError, FunctionDefinition, HttpFeatures, Resource, Tool, ToolOutput, gen_schema_for,
};
use http::header;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use crate::context::BaseCtx;

/// Arguments for Google search query
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct SearchArgs {
    /// The search query string
    pub query: String,
}

/// Represents a single search result item
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct SearchResultItem {
    /// Title of the search result
    pub title: String,
    /// URL of the search result
    pub link: String,
    /// Short description snippet of the result
    pub snippet: String,
}

/// Google Search Tool implementation
///
/// Provides functionality to perform web searches using Google's Custom Search API.
///
/// # Prerequisites
/// - Enable Custom Search API at <https://console.cloud.google.com/>
/// - Obtain API Key and Custom Search Engine ID
///
/// # API Reference
/// - Official documentation: <https://developers.google.com/custom-search/v1/using_rest>
#[derive(Debug, Clone)]
pub struct GoogleSearchTool {
    /// Google API key for authentication
    api_key: String,
    /// Custom Search Engine ID
    search_engine_id: String,
    /// Number of results to return
    result_number: u8,
    /// JSON schema for the search arguments
    schema: Value,
}

impl GoogleSearchTool {
    pub const NAME: &'static str = "google_web_search";
    /// Creates a new GoogleSearchTool instance
    ///
    /// # Arguments
    /// * `api_key` - Google API key
    /// * `search_engine_id` - Custom Search Engine ID
    /// * `result_number` - Optional number of results to return (defaults to 5)
    pub fn new(api_key: String, search_engine_id: String, result_number: Option<u8>) -> Self {
        let schema = gen_schema_for::<SearchArgs>();

        GoogleSearchTool {
            api_key,
            search_engine_id,
            result_number: result_number.unwrap_or(5),
            schema,
        }
    }

    /// Performs a Google search using the provided query
    ///
    /// # Arguments
    /// * `ctx` - HTTP context for making requests
    /// * `args` - Search arguments containing the query
    ///
    /// # Returns
    /// Vector of search result items or an error
    pub async fn search(
        &self,
        ctx: &impl HttpFeatures,
        args: SearchArgs,
    ) -> Result<Vec<SearchResultItem>, BoxError> {
        let mut url = Url::parse("https://www.googleapis.com/customsearch/v1")?;
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            "application/json".parse().expect("invalid header value"),
        );
        headers.insert(
            header::ACCEPT_ENCODING,
            "gzip".parse().expect("invalid header value"),
        );

        url.query_pairs_mut()
            .append_pair("key", &self.api_key)
            .append_pair("cx", &self.search_engine_id)
            .append_pair("num", self.result_number.to_string().as_str())
            .append_pair("q", args.query.as_str());

        let response = ctx
            .https_call(url.as_str(), http::Method::GET, Some(headers), None)
            .await?;

        if !response.status().is_success() {
            return Err(format!(
                "Google customsearch API returned status: {}",
                response.status()
            )
            .into());
        }

        let json: Value = response.json().await?;
        let mut res = Vec::new();
        if let Some(items) = json.get("items").and_then(|v| v.as_array()) {
            for item in items {
                if let (Some(title), Some(link), Some(snippet)) = (
                    item.get("title").and_then(|v| v.as_str()),
                    item.get("link").and_then(|v| v.as_str()),
                    item.get("snippet").and_then(|v| v.as_str()),
                ) {
                    res.push(SearchResultItem {
                        title: title.to_string(),
                        link: link.to_string(),
                        snippet: snippet.to_string(),
                    });
                }
            }
        }

        Ok(res)
    }
}

impl Tool<BaseCtx> for GoogleSearchTool {
    type Args = SearchArgs;
    type Output = Vec<SearchResultItem>;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Performs a google web search for your query then returns a string of the top search results.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    /// Executes the search operation
    ///
    /// # Arguments
    /// * `ctx` - Base context
    /// * `args` - Search arguments
    ///
    /// # Returns
    /// Vector of search results or an error
    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let res = self.search(&ctx, args).await?;
        Ok(ToolOutput::new(res))
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use anda_core::Json;
    use axum::{Json as AxumJson, Router, http::StatusCode, routing::get};
    use parking_lot::Mutex;
    use serde::de::DeserializeOwned;
    use serde_json::json;
    use std::sync::Arc;

    use crate::engine::EngineBuilder;

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
            let headers = headers.unwrap();
            assert_eq!(
                headers.get(header::CONTENT_TYPE).unwrap(),
                "application/json"
            );
            assert_eq!(headers.get(header::ACCEPT_ENCODING).unwrap(), "gzip");
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
        base: String,
    }

    impl ReqwestHttp {
        fn new(base: String) -> Self {
            Self {
                client: reqwest::Client::builder().no_proxy().build().unwrap(),
                base,
            }
        }

        fn rewrite_google_url(&self, url: &str) -> String {
            let parsed = Url::parse(url).unwrap();
            let query = parsed.query().unwrap_or_default();
            format!("{}/customsearch/v1?{}", self.base, query)
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
            let mut request = self.client.request(method, self.rewrite_google_url(url));
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

    async fn spawn_google_server() -> String {
        let app = Router::new()
            .route(
                "/customsearch/v1",
                get(|| async {
                    AxumJson(json!({
                        "items": [
                            {
                                "title": "Anda",
                                "link": "https://anda.example",
                                "snippet": "Search result"
                            },
                            {
                                "title": "Missing snippet",
                                "link": "https://skip.example"
                            },
                            {
                                "title": "Second",
                                "link": "https://second.example",
                                "snippet": "Another result"
                            }
                        ]
                    }))
                }),
            )
            .route(
                "/customsearch/status",
                get(|| async { (StatusCode::TOO_MANY_REQUESTS, "quota") }),
            )
            .route(
                "/customsearch/invalid",
                get(|| async { ([(header::CONTENT_TYPE, "application/json")], "not-json") }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn google_search_definition_url_and_error_paths_are_stable() {
        let tool = GoogleSearchTool::new("api-key".to_string(), "engine-id".to_string(), Some(3));
        assert_eq!(tool.name(), GoogleSearchTool::NAME);
        assert!(tool.description().contains("google web search"));
        let definition = tool.definition();
        assert_eq!(definition.name, GoogleSearchTool::NAME);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(
            definition.parameters["type"],
            Json::String("object".to_string())
        );

        let http = ErrorHttp::default();
        assert!(
            tool.search(
                &http,
                SearchArgs {
                    query: "anda rust".to_string(),
                },
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("http disabled")
        );
        let captured = http.calls.lock().pop().unwrap();
        let url = Url::parse(&captured).unwrap();
        let query: std::collections::BTreeMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(query.get("key").map(String::as_str), Some("api-key"));
        assert_eq!(query.get("cx").map(String::as_str), Some("engine-id"));
        assert_eq!(query.get("num").map(String::as_str), Some("3"));
        assert_eq!(query.get("q").map(String::as_str), Some("anda rust"));
        assert!(
            http.https_signed_call("unused", http::Method::GET, [0; 32], None, None)
                .await
                .unwrap_err()
                .to_string()
                .contains("not used")
        );
        let signed_rpc: Result<String, _> = http.https_signed_rpc("unused", "method", &()).await;
        assert!(signed_rpc.unwrap_err().to_string().contains("not used"));

        let default_tool =
            GoogleSearchTool::new("api-key".to_string(), "engine-id".to_string(), None);
        let http = ErrorHttp::default();
        let _ = default_tool
            .search(
                &http,
                SearchArgs {
                    query: "default".to_string(),
                },
            )
            .await;
        assert!(http.calls.lock()[0].contains("num=5"));

        let ctx = EngineBuilder::new().mock_ctx().base;
        assert!(
            tool.call(
                ctx,
                SearchArgs {
                    query: "anda".to_string(),
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
    async fn google_search_parses_success_status_errors_and_tool_call_output() {
        let endpoint = spawn_google_server().await;
        let http = ReqwestHttp::new(endpoint.clone());
        let tool = GoogleSearchTool::new("api-key".to_string(), "engine-id".to_string(), Some(2));

        let results = tool
            .search(
                &http,
                SearchArgs {
                    query: "anda".to_string(),
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Anda");
        assert_eq!(results[0].link, "https://anda.example");
        assert_eq!(results[0].snippet, "Search result");
        assert_eq!(results[1].title, "Second");
        assert!(
            http.https_call(
                "https://www.googleapis.com/customsearch/v1?key=api-key&cx=engine-id&num=1&q=body",
                http::Method::GET,
                None,
                Some(b"ignored".to_vec()),
            )
            .await
            .unwrap()
            .status()
            .is_success()
        );
        assert!(
            http.https_signed_call("unused", http::Method::GET, [0; 32], None, None)
                .await
                .unwrap_err()
                .to_string()
                .contains("not used")
        );
        let signed_rpc: Result<String, _> = http.https_signed_rpc("unused", "method", &()).await;
        assert!(signed_rpc.unwrap_err().to_string().contains("not used"));

        struct StatusHttp(ReqwestHttp);
        impl HttpFeatures for StatusHttp {
            async fn https_call(
                &self,
                url: &str,
                method: http::Method,
                headers: Option<header::HeaderMap>,
                body: Option<Vec<u8>>,
            ) -> Result<reqwest::Response, BoxError> {
                let rewritten = self
                    .0
                    .rewrite_google_url(url)
                    .replace("/customsearch/v1", "/customsearch/status");
                let mut request = self.0.client.request(method, rewritten);
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

        let endpoint = spawn_google_server().await;
        let status_http = StatusHttp(ReqwestHttp::new(endpoint));
        let status = tool
            .search(
                &status_http,
                SearchArgs {
                    query: "anda".to_string(),
                },
            )
            .await
            .unwrap_err();
        assert!(status.to_string().contains("429 Too Many Requests"));
        assert!(
            status_http
                .https_call(
                    "https://www.googleapis.com/customsearch/v1?key=api-key&cx=engine-id&num=1&q=body",
                    http::Method::GET,
                    None,
                    Some(b"ignored".to_vec()),
                )
                .await
                .unwrap()
                .status()
                .is_client_error()
        );
        assert!(
            status_http
                .https_signed_call("unused", http::Method::GET, [0; 32], None, None)
                .await
                .unwrap_err()
                .to_string()
                .contains("not used")
        );
        let signed_rpc: Result<String, _> =
            status_http.https_signed_rpc("unused", "method", &()).await;
        assert!(signed_rpc.unwrap_err().to_string().contains("not used"));
    }
}
