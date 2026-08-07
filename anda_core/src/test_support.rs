//! Shared fixtures for anda_core's own tests.

use candid::Principal;
use serde::Serialize;
use std::{future::Future, time::Duration};

use crate::{
    AgentContext, AgentInput, AgentOutput, BaseContext, BoxError, CacheExpiry, CacheFeatures,
    CancellationToken, CompletionFeatures, CompletionRequest, FunctionDefinition, HttpFeatures,
    Json, KeysFeatures, ObjectMeta, Path, PutMode, PutResult, RequestMeta, Resource, StateFeatures,
    StoreFeatures, ToolInput, ToolOutput,
};

/// Builds a tagged test resource with a stable name.
pub(crate) fn resource(id: u64, tags: &[&str]) -> Resource {
    Resource {
        _id: id,
        name: format!("resource-{id}"),
        tags: tags.iter().map(|tag| tag.to_string()).collect(),
        ..Default::default()
    }
}

/// Stub execution context implementing every context capability.
///
/// State reads succeed with fixed values; storage, cache, HTTP, and completion
/// calls return "not implemented" errors or inert defaults. Tests use it where
/// only the trait bounds matter, not runtime behavior.
#[derive(Clone)]
pub(crate) struct MockContext {
    engine_id: Principal,
    caller: Principal,
    meta: RequestMeta,
    cancellation_token: CancellationToken,
}

impl Default for MockContext {
    fn default() -> Self {
        Self {
            engine_id: Principal::management_canister(),
            caller: Principal::anonymous(),
            meta: RequestMeta::default(),
            cancellation_token: CancellationToken::new(),
        }
    }
}

impl StateFeatures for MockContext {
    fn engine_id(&self) -> &Principal {
        &self.engine_id
    }

    fn engine_name(&self) -> &str {
        "test-engine"
    }

    fn caller(&self) -> &Principal {
        &self.caller
    }

    fn meta(&self) -> &RequestMeta {
        &self.meta
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    fn time_elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

impl KeysFeatures for MockContext {
    async fn a256gcm_key(&self, _derivation_path: Vec<Vec<u8>>) -> Result<[u8; 32], BoxError> {
        Ok([0; 32])
    }

    async fn ed25519_sign_message(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        Ok([0; 64])
    }

    async fn ed25519_verify(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<(), BoxError> {
        Ok(())
    }

    async fn ed25519_public_key(
        &self,
        _derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 32], BoxError> {
        Ok([0; 32])
    }

    async fn secp256k1_sign_message_bip340(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        Ok([0; 64])
    }

    async fn secp256k1_verify_bip340(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<(), BoxError> {
        Ok(())
    }

    async fn secp256k1_sign_message_ecdsa(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        Ok([0; 64])
    }

    async fn secp256k1_sign_digest_ecdsa(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message_hash: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        Ok([0; 64])
    }

    async fn secp256k1_verify_ecdsa(
        &self,
        _derivation_path: Vec<Vec<u8>>,
        _message_hash: &[u8],
        _signature: &[u8],
    ) -> Result<(), BoxError> {
        Ok(())
    }

    async fn secp256k1_public_key(
        &self,
        _derivation_path: Vec<Vec<u8>>,
    ) -> Result<[u8; 33], BoxError> {
        Ok([0; 33])
    }
}

impl StoreFeatures for MockContext {
    async fn store_get(&self, _path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        Err("not implemented".into())
    }

    async fn store_list(
        &self,
        _prefix: Option<&Path>,
        _offset: &Path,
    ) -> Result<Vec<ObjectMeta>, BoxError> {
        Ok(Vec::new())
    }

    async fn store_put(
        &self,
        _path: &Path,
        _mode: PutMode,
        _value: bytes::Bytes,
    ) -> Result<PutResult, BoxError> {
        Err("not implemented".into())
    }

    async fn store_rename_if_not_exists(&self, _from: &Path, _to: &Path) -> Result<(), BoxError> {
        Err("not implemented".into())
    }

    async fn store_delete(&self, _path: &Path) -> Result<(), BoxError> {
        Ok(())
    }
}

impl CacheFeatures for MockContext {
    fn cache_contains(&self, _key: &str) -> bool {
        false
    }

    async fn cache_get<T>(&self, _key: &str) -> Result<T, BoxError>
    where
        T: serde::de::DeserializeOwned,
    {
        Err("not implemented".into())
    }

    async fn cache_get_with<T, F>(&self, _key: &str, _init: F) -> Result<T, BoxError>
    where
        T: Sized + serde::de::DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        Err("not implemented".into())
    }

    async fn cache_set<T>(&self, _key: &str, _val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
    }

    async fn cache_set_if_not_exists<T>(&self, _key: &str, _val: (T, Option<CacheExpiry>)) -> bool
    where
        T: Sized + Serialize + Send,
    {
        false
    }

    async fn cache_delete(&self, _key: &str) -> bool {
        false
    }
}

impl HttpFeatures for MockContext {
    async fn https_call(
        &self,
        _url: &str,
        _method: http::Method,
        _headers: Option<http::HeaderMap>,
        _body: Option<Vec<u8>>,
    ) -> Result<reqwest::Response, BoxError> {
        Err("not implemented".into())
    }

    async fn https_signed_call(
        &self,
        _url: &str,
        _method: http::Method,
        _message_digest: [u8; 32],
        _headers: Option<http::HeaderMap>,
        _body: Option<Vec<u8>>,
    ) -> Result<reqwest::Response, BoxError> {
        Err("not implemented".into())
    }

    async fn https_signed_rpc<T>(
        &self,
        _endpoint: &str,
        _method: &str,
        _args: impl Serialize + Send,
    ) -> Result<T, BoxError>
    where
        T: serde::de::DeserializeOwned,
    {
        Err("not implemented".into())
    }
}

impl BaseContext for MockContext {
    async fn remote_tool_call(
        &self,
        _endpoint: &str,
        _args: ToolInput<Json>,
    ) -> Result<ToolOutput<Json>, BoxError> {
        Err("not implemented".into())
    }
}

impl CompletionFeatures for MockContext {
    async fn completion(
        &self,
        _req: CompletionRequest,
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        Ok(AgentOutput::default())
    }

    fn model_name(&self) -> String {
        "test-model".to_string()
    }
}

impl AgentContext for MockContext {
    fn tool_definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
        Vec::new()
    }

    async fn remote_tool_definitions(
        &self,
        _endpoint: Option<&str>,
        _names: Option<&[String]>,
    ) -> Result<Vec<FunctionDefinition>, BoxError> {
        Ok(Vec::new())
    }

    async fn select_tool_resources(
        &self,
        _name: &str,
        _resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        Vec::new()
    }

    fn agent_definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
        Vec::new()
    }

    async fn remote_agent_definitions(
        &self,
        _endpoint: Option<&str>,
        _names: Option<&[String]>,
    ) -> Result<Vec<FunctionDefinition>, BoxError> {
        Ok(Vec::new())
    }

    async fn select_agent_resources(
        &self,
        _name: &str,
        _resources: &mut Vec<Resource>,
    ) -> Vec<Resource> {
        Vec::new()
    }

    async fn definitions(&self, _names: Option<&[String]>) -> Vec<FunctionDefinition> {
        Vec::new()
    }

    async fn tool_call(
        &self,
        _args: ToolInput<Json>,
    ) -> Result<(ToolOutput<Json>, Option<Principal>), BoxError> {
        Ok((ToolOutput::new(Json::Null), None))
    }

    async fn agent_run(
        self,
        _args: AgentInput,
    ) -> Result<(AgentOutput, Option<Principal>), BoxError> {
        Ok((AgentOutput::default(), None))
    }

    async fn remote_agent_run(
        &self,
        _endpoint: &str,
        _args: AgentInput,
    ) -> Result<AgentOutput, BoxError> {
        Ok(AgentOutput::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_context_features_cover_default_paths() {
        futures::executor::block_on(async {
            let ctx = MockContext::default();
            assert_eq!(*ctx.engine_id(), Principal::management_canister());
            assert_eq!(ctx.engine_name(), "test-engine");
            assert_eq!(*ctx.caller(), Principal::anonymous());
            assert!(ctx.meta().user.is_none());
            assert!(!ctx.cancellation_token().is_cancelled());
            assert_eq!(ctx.time_elapsed(), Duration::ZERO);

            assert_eq!(ctx.a256gcm_key(Vec::new()).await.unwrap(), [0; 32]);
            assert_eq!(
                ctx.ed25519_sign_message(Vec::new(), b"message")
                    .await
                    .unwrap(),
                [0; 64]
            );
            ctx.ed25519_verify(Vec::new(), b"message", &[0; 64])
                .await
                .unwrap();
            assert_eq!(ctx.ed25519_public_key(Vec::new()).await.unwrap(), [0; 32]);
            assert_eq!(
                ctx.secp256k1_sign_message_bip340(Vec::new(), b"message")
                    .await
                    .unwrap(),
                [0; 64]
            );
            ctx.secp256k1_verify_bip340(Vec::new(), b"message", &[0; 64])
                .await
                .unwrap();
            assert_eq!(
                ctx.secp256k1_sign_message_ecdsa(Vec::new(), b"message")
                    .await
                    .unwrap(),
                [0; 64]
            );
            assert_eq!(
                ctx.secp256k1_sign_digest_ecdsa(Vec::new(), &[0; 32])
                    .await
                    .unwrap(),
                [0; 64]
            );
            ctx.secp256k1_verify_ecdsa(Vec::new(), &[0; 32], &[0; 64])
                .await
                .unwrap();
            assert_eq!(ctx.secp256k1_public_key(Vec::new()).await.unwrap(), [0; 33]);

            assert!(ctx.store_get(&Path::from("missing")).await.is_err());
            assert!(
                ctx.store_list(None, &Path::default())
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                ctx.store_put(&Path::from("file"), PutMode::Overwrite, bytes::Bytes::new())
                    .await
                    .is_err()
            );
            assert!(
                ctx.store_rename_if_not_exists(&Path::from("a"), &Path::from("b"))
                    .await
                    .is_err()
            );
            ctx.store_delete(&Path::from("file")).await.unwrap();

            assert!(!ctx.cache_contains("key"));
            assert!(ctx.cache_get::<String>("key").await.is_err());
            assert!(
                ctx.cache_get_with("key", async { Ok(("value".to_string(), None)) })
                    .await
                    .is_err()
            );
            ctx.cache_set("key", ("value".to_string(), None)).await;
            assert!(
                !ctx.cache_set_if_not_exists("key", ("value".to_string(), None))
                    .await
            );
            assert!(!ctx.cache_delete("key").await);

            assert!(
                ctx.https_call("https://example.test", http::Method::GET, None, None)
                    .await
                    .is_err()
            );
            assert!(
                ctx.https_signed_call(
                    "https://example.test",
                    http::Method::POST,
                    [0; 32],
                    None,
                    None,
                )
                .await
                .is_err()
            );
            let rpc: Result<String, BoxError> = ctx
                .https_signed_rpc("https://example.test", "method", &())
                .await;
            assert!(rpc.is_err());

            assert!(
                ctx.remote_tool_call(
                    "https://example.test",
                    ToolInput::new("tool".to_string(), Json::Null),
                )
                .await
                .is_err()
            );
            assert_eq!(
                ctx.completion(CompletionRequest::default(), Vec::new())
                    .await
                    .unwrap()
                    .content,
                ""
            );
            assert_eq!(ctx.model_name(), "test-model");
            assert!(ctx.tool_definitions(None).is_empty());
            assert!(
                ctx.remote_tool_definitions(None, None)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                ctx.select_tool_resources("tool", &mut Vec::new())
                    .await
                    .is_empty()
            );
            assert!(ctx.agent_definitions(None).is_empty());
            assert!(
                ctx.remote_agent_definitions(None, None)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                ctx.select_agent_resources("agent", &mut Vec::new())
                    .await
                    .is_empty()
            );
            assert!(ctx.definitions(None).await.is_empty());
            assert!(
                ctx.tool_call(ToolInput::new("tool".to_string(), Json::Null))
                    .await
                    .unwrap()
                    .0
                    .output
                    .is_null()
            );
            assert!(
                ctx.clone()
                    .agent_run(AgentInput {
                        name: "agent".to_string(),
                        prompt: "prompt".to_string(),
                        ..Default::default()
                    })
                    .await
                    .unwrap()
                    .0
                    .content
                    .is_empty()
            );
            assert!(
                ctx.remote_agent_run(
                    "https://example.test",
                    AgentInput {
                        name: "agent".to_string(),
                        prompt: "prompt".to_string(),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
                .content
                .is_empty()
            );
        });
    }
}
