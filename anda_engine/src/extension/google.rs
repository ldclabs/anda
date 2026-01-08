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
/// - Enable Custom Search API at https://console.cloud.google.com/
/// - Obtain API Key and Custom Search Engine ID
///
/// # API Reference
/// - Official documentation: https://developers.google.com/custom-search/v1/using_rest
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
mod tests {
    use super::*;
    use crate::{engine::EngineBuilder, model::Model};

    #[tokio::test]
    #[ignore]
    async fn test_google_search_tool() {
        dotenv::dotenv().ok();

        let api_key = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY is not set");
        let search_engine_id =
            std::env::var("GOOGLE_SEARCH_ENGINE_ID").expect("GOOGLE_SEARCH_ENGINE_ID is not set");
        let tool = GoogleSearchTool::new(api_key, search_engine_id, Some(6));
        let definition = tool.definition();
        assert_eq!(tool.name(), "google_web_search");
        println!("{}", serde_json::to_string_pretty(&definition).unwrap());
        // {
        //     "name": "google_web_search",
        //     "description": "Performs a google web search for your query then returns a string of the top search results.",
        //     "parameters": {
        //       "description": "The search query to perform.",
        //       "properties": {
        //         "query": {
        //           "type": "string"
        //         }
        //       },
        //       "required": [
        //         "query"
        //       ],
        //       "title": "SearchArgs",
        //       "type": "object"
        //     },
        //     "strict": true
        // }

        let ctx = EngineBuilder::new()
            .with_model(Model::mock_implemented())
            .mock_ctx();
        let res = tool
            .search(
                &ctx,
                SearchArgs {
                    query: "ICPanda".to_string(),
                },
            )
            .await
            .unwrap();
        print!("{:?}", res);
    }
}
