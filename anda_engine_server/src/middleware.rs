use axum::{
    Router,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{future::Future, sync::Arc};

use crate::handler::AppState;

/// Object-safe middleware trait for applying HTTP middleware to the server `Router`.
///
/// This is intentionally type-erased so callers can register arbitrary axum/tower
/// middleware without turning `ServerBuilder` into a giant generic type.
pub trait HttpMiddleware: Send + Sync + 'static {
    fn apply(&self, router: Router<AppState>) -> Router<AppState>;
}

impl<F> HttpMiddleware for F
where
    F: Fn(Router<AppState>) -> Router<AppState> + Send + Sync + 'static,
{
    fn apply(&self, router: Router<AppState>) -> Router<AppState> {
        (self)(router)
    }
}

/// Middleware built from a function like axum::middleware::from_fn.
pub struct RequestFnMiddleware<F> {
    f: F,
}

impl<F> RequestFnMiddleware<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F, Fut> HttpMiddleware for RequestFnMiddleware<F>
where
    F: Fn(Request, Next) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Response> + Send + 'static,
{
    fn apply(&self, router: Router<AppState>) -> Router<AppState> {
        router.layer(axum::middleware::from_fn(self.f.clone()))
    }
}

/// A simple API key middleware that validates `x-api-key` on every request.
///
/// Use `exempt_path` to allow unauthenticated endpoints (e.g. health/info routes).
pub struct ApiKeyMiddleware {
    expected_key: Arc<String>,
    exempt_paths: Arc<Vec<String>>,
}

impl ApiKeyMiddleware {
    pub fn new(expected_key: impl Into<String>) -> Self {
        Self {
            expected_key: Arc::new(expected_key.into()),
            exempt_paths: Arc::new(Vec::new()),
        }
    }

    pub fn exempt_path(mut self, path: impl Into<String>) -> Self {
        Arc::get_mut(&mut self.exempt_paths)
            .unwrap()
            .push(path.into());
        self
    }
}

impl HttpMiddleware for ApiKeyMiddleware {
    fn apply(&self, router: Router<AppState>) -> Router<AppState> {
        let expected_key = self.expected_key.clone();
        let exempt_paths = self.exempt_paths.clone();

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
                        Some(provided_key) if provided_key == expected_key.as_str() => {
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
