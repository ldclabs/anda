use anda_core::BoxError;
use anda_engine::engine::Engine;
use axum::{Router, routing};
use candid::Principal;
use std::{collections::BTreeMap, future::Future, net::SocketAddr, sync::Arc};
use structured_logger::unix_ms;
use tokio::signal;
use tokio_util::sync::CancellationToken;

mod handler;
mod types;

use handler::*;

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct ServerBuilder {
    app_name: String,
    app_version: String,
    addr: String,
    origin: String,
    engines: BTreeMap<Principal, Engine>,
    default_engine: Option<Principal>,
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
        };
        let app = Router::new()
            .route("/", routing::get(get_information))
            .route("/.well-known/information", routing::get(get_information))
            .route("/.well-known/agents", routing::get(get_information))
            .route(
                "/.well-known/agents/{id}",
                routing::get(get_engine_information),
            )
            .route("/{*id}", routing::post(anda_engine))
            .with_state(state);

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
