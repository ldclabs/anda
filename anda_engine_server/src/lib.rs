use anda_core::{BoxError, Json};
use anda_engine::engine::Engine;
use axum::{Router, routing};
use candid::Principal;
use std::{collections::BTreeMap, future::Future, net::SocketAddr, sync::Arc};
use structured_logger::unix_ms;
use tokio::signal;
use tokio_util::sync::CancellationToken;

mod handler;
mod middleware;
mod types;

use handler::*;
pub use middleware::{ApiKeyMiddleware, HttpMiddleware};

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct ServerBuilder {
    app_name: String,
    app_version: String,
    addr: String,
    origin: String,
    engines: BTreeMap<Principal, Engine>,
    default_engine: Option<Principal>,
    middlewares: Vec<Arc<dyn HttpMiddleware>>,
    extra_info: BTreeMap<String, Json>,
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating a new Server.
/// Example: https://github.com/ldclabs/anda/tree/main/examples/icp_ledger_agent
impl ServerBuilder {
    /// Creates a new ServerBuilder with default values.
    pub fn new() -> Self {
        ServerBuilder {
            app_name: APP_NAME.to_string(),
            app_version: APP_VERSION.to_string(),
            addr: "127.0.0.1:8042".to_string(),
            origin: "https://localhost:8443".to_string(),
            engines: BTreeMap::new(),
            default_engine: None,
            middlewares: Vec::new(),
            extra_info: BTreeMap::new(),
        }
    }

    pub fn with_app_name(mut self, app_name: String) -> Self {
        self.app_name = app_name;
        self
    }

    pub fn with_app_version(mut self, app_version: String) -> Self {
        self.app_version = app_version;
        self
    }

    pub fn with_addr(mut self, addr: String) -> Self {
        self.addr = addr;
        self
    }

    pub fn with_origin(mut self, origin: String) -> Self {
        self.origin = origin;
        self
    }

    pub fn with_extra_info(mut self, extra_info: BTreeMap<String, Json>) -> Self {
        self.extra_info = extra_info;
        self
    }

    pub fn with_engines(
        mut self,
        mut engines: BTreeMap<Principal, Engine>,
        default_engine: Option<Principal>,
    ) -> Self {
        for (id, engine) in engines.iter_mut() {
            engine.info_mut().endpoint = format!("{}/{}", self.origin, id.to_text());
        }

        self.engines = engines;
        self.default_engine = default_engine;
        self
    }

    /// Register a router middleware.
    ///
    /// This is the low-level API. The middleware will be applied to the internal
    /// axum `Router` (typically via `router.layer(...)`). Middlewares are applied
    /// in the order they are added.
    ///
    /// More details: https://docs.rs/axum/latest/axum/middleware/index.html#ordering
    ///
    /// If you want a middleware that looks like `axum::middleware::from_fn`
    /// (i.e. can operate on `(req, next)`), prefer [`with_request_middleware`].
    ///
    /// Example:
    /// ```ignore
    /// let server = ServerBuilder::new()
    ///   .with_middleware(|router| {
    ///     router.layer(axum::middleware::from_fn(|req, next| async move {
    ///       // custom auth / param checks here
    ///       next.run(req).await
    ///     }))
    ///   });
    /// ```
    pub fn with_middleware<M>(mut self, middleware: M) -> Self
    where
        M: HttpMiddleware,
    {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Register a request middleware like `axum::middleware::from_fn`.
    ///
    /// The middleware function runs for every incoming request, and can decide
    /// to short-circuit with a response or call `next.run(req)`.
    ///
    /// Example:
    /// ```ignore
    /// use axum::http::StatusCode;
    /// use axum::response::IntoResponse;
    ///
    /// let server = ServerBuilder::new()
    ///   .with_request_middleware(|req, next| async move {
    ///     // custom auth / param checks here
    ///     if req.headers().get("x-allow").is_none() {
    ///       return (StatusCode::UNAUTHORIZED, "missing x-allow").into_response();
    ///     }
    ///
    ///     next.run(req).await
    ///   });
    /// ```
    pub fn with_request_middleware<F, Fut>(self, f: F) -> Self
    where
        F: Fn(axum::extract::Request, axum::middleware::Next) -> Fut
            + Clone
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = axum::response::Response> + Send + 'static,
    {
        self.with_middleware(middleware::RequestFnMiddleware::new(f))
    }

    pub async fn serve(
        self,
        signal: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), BoxError> {
        if self.engines.is_empty() {
            return Err("no engines registered".into());
        }

        let default_engine = self
            .default_engine
            .unwrap_or_else(|| *self.engines.keys().next().unwrap());
        if !self.engines.contains_key(&default_engine) {
            return Err("default engine not found".into());
        }

        let state = AppState {
            engines: Arc::new(self.engines),
            default_engine,
            start_time_ms: unix_ms(),
            extra_info: Arc::new(self.extra_info),
        };

        // Build a router that is still "missing" an `AppState`.
        // We'll provide the state at the end (after applying middlewares) so we
        // end up with a `Router<()>` that can be passed to `axum::serve`.
        let mut app: Router<AppState> = Router::new()
            .route("/", routing::get(get_information))
            .route("/.well-known/information", routing::get(get_information))
            .route("/.well-known/agents", routing::get(get_information))
            .route(
                "/.well-known/agents/{id}",
                routing::get(get_engine_information),
            )
            .route("/{*id}", routing::post(anda_engine));

        for middleware in &self.middlewares {
            app = middleware.apply(app);
        }

        let app = app.with_state(state);

        let addr: SocketAddr = self.addr.parse()?;
        let listener = create_reuse_port_listener(addr).await?;
        log::warn!(
            "{}@{} listening on {:?}",
            self.app_name,
            self.app_version,
            addr
        );

        axum::serve(listener, app)
            .with_graceful_shutdown(signal)
            .await?;

        Ok(())
    }
}

pub async fn shutdown_signal(cancel_token: CancellationToken) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    log::warn!("received termination signal, starting graceful shutdown");
    cancel_token.cancel();
}

pub async fn create_reuse_port_listener(
    addr: SocketAddr,
) -> Result<tokio::net::TcpListener, BoxError> {
    let socket = match &addr {
        SocketAddr::V4(_) => tokio::net::TcpSocket::new_v4()?,
        SocketAddr::V6(_) => tokio::net::TcpSocket::new_v6()?,
    };

    socket.set_reuseport(true)?;
    socket.bind(addr)?;
    let listener = socket.listen(1024)?;
    Ok(listener)
}
