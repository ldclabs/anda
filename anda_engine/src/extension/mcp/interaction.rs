//! Opt-in application callbacks for standard MCP elicitation.

use anda_core::{BoxError, CancellationToken};
use async_trait::async_trait;
use rmcp::model::{
    ElicitRequestParams, ElicitResult, ElicitationCapability, FormElicitationCapability,
};
use std::{sync::Arc, time::Duration};

/// Application-owned interaction. Implementations must treat requests as untrusted
/// and release their UI when `cancellation` fires. No browser or UI is opened here.
#[async_trait]
pub trait McpElicitationHandler: Send + Sync {
    /// Standard modes supported by this application; defaults to form input only.
    fn capabilities(&self) -> ElicitationCapability {
        ElicitationCapability::new().with_form(FormElicitationCapability::default())
    }
    /// Obtains an application response for one server request.
    async fn elicit(
        &self,
        server_id: &str,
        request: ElicitRequestParams,
        cancellation: CancellationToken,
    ) -> Result<ElicitResult, BoxError>;
}

#[derive(Clone)]
pub(super) struct ElicitationDispatcher {
    pub handler: Arc<dyn McpElicitationHandler>,
    pub server_id: String,
    pub timeout: Duration,
    pub session_cancelled: CancellationToken,
}
impl ElicitationDispatcher {
    pub async fn elicit(
        &self,
        request: ElicitRequestParams,
        cancellation: &CancellationToken,
    ) -> Result<ElicitResult, BoxError> {
        let capabilities = self.handler.capabilities();
        let supported = match &request {
            ElicitRequestParams::FormElicitationParams { .. } => capabilities.form.is_some(),
            ElicitRequestParams::UrlElicitationParams { .. } => capabilities.url.is_some(),
            _ => false,
        };
        if !supported {
            return Err("MCP elicitation mode is not enabled".into());
        }
        let child = cancellation.child_token();
        let _guard = child.clone().drop_guard();
        tokio::select! {
            biased;
            _ = self.session_cancelled.cancelled() => Err("MCP session closed during elicitation".into()),
            _ = cancellation.cancelled() => Err("MCP elicitation cancelled".into()),
            result = tokio::time::timeout(self.timeout, self.handler.elicit(&self.server_id, request, child)) =>
                result.map_err(|_| "MCP elicitation timed out")?,
        }
    }
}
