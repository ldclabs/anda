//! Explicit resource access over the same authenticated SDK sessions.

use super::{McpToolProvider, catalog::collect_pages};
use anda_core::{BoxError, CancellationToken};
use rmcp::model::{ReadResourceRequestParams, ReadResourceResult, Resource, ResourceTemplate};
use std::time::Duration;

impl McpToolProvider {
    /// Lists a bounded live resource catalog. Requires the server's `resources` opt-in.
    pub async fn list_resources(
        &self,
        server_id: &str,
        cancellation: CancellationToken,
    ) -> Result<Vec<Resource>, BoxError> {
        let config = self.server_config(server_id)?;
        if !config.resources {
            return Err("MCP resources are disabled for this server".into());
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err("MCP resource listing cancelled".into()),
            _ = config.cancelled.cancelled() => Err("MCP server removed".into()),
            result = tokio::time::timeout(Duration::from_secs(config.timeouts.setup_secs + config.timeouts.list_secs), async {
                let session = self.ensure_session(&config).await?;
                let peer = session.service.lock().await.peer().clone();
                collect_pages(&config.limits, |params| { let peer = peer.clone(); async move {
                    let page = peer.list_resources(params).await?; Ok((page.resources, page.next_cursor))
                }}).await
            }) => result.map_err(|_| "MCP resource listing timed out")?,
        }
    }

    /// Lists bounded resource templates. No resource is read implicitly.
    pub async fn list_resource_templates(
        &self,
        server_id: &str,
        cancellation: CancellationToken,
    ) -> Result<Vec<ResourceTemplate>, BoxError> {
        let config = self.server_config(server_id)?;
        if !config.resources {
            return Err("MCP resources are disabled for this server".into());
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err("MCP resource listing cancelled".into()),
            _ = config.cancelled.cancelled() => Err("MCP server removed".into()),
            result = tokio::time::timeout(Duration::from_secs(config.timeouts.setup_secs + config.timeouts.list_secs), async {
                let session = self.ensure_session(&config).await?;
                let peer = session.service.lock().await.peer().clone();
                collect_pages(&config.limits, |params| { let peer = peer.clone(); async move {
                    let page = peer.list_resource_templates(params).await?; Ok((page.resource_templates, page.next_cursor))
                }}).await
            }) => result.map_err(|_| "MCP resource listing timed out")?,
        }
    }

    /// Reads an explicit URI from the selected server. Does not dereference it locally
    /// or forward conversation history. Results remain subject to the transport budget.
    pub async fn read_resource(
        &self,
        server_id: &str,
        uri: String,
        cancellation: CancellationToken,
    ) -> Result<ReadResourceResult, BoxError> {
        let config = self.server_config(server_id)?;
        if !config.resources {
            return Err("MCP resources are disabled for this server".into());
        }
        if uri.len() > config.limits.cursor_bytes {
            return Err("MCP resource URI limit exceeded".into());
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err("MCP resource read cancelled".into()),
            _ = config.cancelled.cancelled() => Err("MCP server removed".into()),
            result = tokio::time::timeout(config.timeouts.call(), async {
                let session = self.ensure_session(&config).await?;
                let peer = session.service.lock().await.peer().clone();
                Ok(peer.read_resource(ReadResourceRequestParams::new(uri)).await?)
            }) => result.map_err(|_| "MCP resource read timed out")?,
        }
    }
}
