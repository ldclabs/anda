//! Middleware helpers for [`ServerBuilder`](crate::ServerBuilder).
//!
//! Callers can register arbitrary router middleware, request middleware, gzip
//! compression, or a simple API-key guard without making the server builder
//! generic over every tower layer.

use axum::{
    Router,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{future::Future, sync::Arc};

pub use crate::handler::AppState;
pub use tower_http::compression::CompressionLayer;
/// Axum router type carrying server application state.
pub type AppRouter = Router<AppState>;

/// Object-safe middleware trait for applying HTTP middleware to the server `Router`.
///
/// This is intentionally type-erased so callers can register arbitrary axum/tower
/// middleware without turning `ServerBuilder` into a giant generic type.
pub trait HttpMiddleware: Send + Sync + 'static {
    /// Applies this middleware to the router.
    fn apply(&self, router: AppRouter) -> AppRouter;
}

impl<F> HttpMiddleware for F
where
    F: Fn(AppRouter) -> AppRouter + Send + Sync + 'static,
{
    fn apply(&self, router: AppRouter) -> AppRouter {
        (self)(router)
    }
}

/// Middleware built from a function like axum::middleware::from_fn.
pub struct RequestFnMiddleware<F> {
    f: F,
}

impl<F> RequestFnMiddleware<F> {
    /// Wraps a request middleware function.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F, Fut> HttpMiddleware for RequestFnMiddleware<F>
where
    F: Fn(Request, Next) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Response> + Send + 'static,
{
    fn apply(&self, router: AppRouter) -> AppRouter {
        router.layer(axum::middleware::from_fn(self.f.clone()))
    }
}

/// Middleware that applies response compression using `tower-http`'s `CompressionLayer`.
pub struct CompressionMiddleware {
    layer: CompressionLayer,
}

impl CompressionMiddleware {
    /// Creates compression middleware from a specific tower-http layer.
    pub fn new(layer: CompressionLayer) -> Self {
        Self { layer }
    }
}

impl Default for CompressionMiddleware {
    fn default() -> Self {
        Self {
            layer: CompressionLayer::new(),
        }
    }
}

impl HttpMiddleware for CompressionMiddleware {
    fn apply(&self, router: AppRouter) -> AppRouter {
        router.layer(self.layer.clone())
    }
}

/// A simple API key middleware that validates `x-api-key` on every request.
///
/// Keys are compared in constant time to avoid leaking the expected key
/// through a timing side channel.
///
/// Use `exempt_path` to allow unauthenticated endpoints (e.g. health/info routes).
pub struct ApiKeyMiddleware {
    expected_key: String,
    exempt_paths: Vec<String>,
}

impl ApiKeyMiddleware {
    /// Creates an API-key middleware that expects `x-api-key` to match.
    pub fn new(expected_key: impl Into<String>) -> Self {
        Self {
            expected_key: expected_key.into(),
            exempt_paths: Vec::new(),
        }
    }

    /// Adds a path that bypasses API-key validation.
    pub fn exempt_path(mut self, path: impl Into<String>) -> Self {
        self.exempt_paths.push(path.into());
        self
    }
}

/// Compares two byte slices in constant time for equal-length inputs.
/// Only the length may be learned from timing, never the content.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl HttpMiddleware for ApiKeyMiddleware {
    fn apply(&self, router: AppRouter) -> AppRouter {
        let expected_key: Arc<str> = Arc::from(self.expected_key.as_str());
        let exempt_paths: Arc<[String]> = Arc::from(self.exempt_paths.as_slice());

        router.layer(axum::middleware::from_fn(
            move |req: Request, next: Next| {
                let expected_key = expected_key.clone();
                let exempt_paths = exempt_paths.clone();

                async move {
                    let path = req.uri().path();
                    if exempt_paths.iter().any(|p| p == path) {
                        return next.run(req).await;
                    }

                    match req.headers().get("x-api-key") {
                        Some(provided_key)
                            if constant_time_eq(
                                provided_key.as_bytes(),
                                expected_key.as_bytes(),
                            ) =>
                        {
                            next.run(req).await
                        }
                        _ => (
                            StatusCode::UNAUTHORIZED,
                            "missing or invalid x-api-key in headers",
                        )
                            .into_response(),
                    }
                }
            },
        ))
    }
}
