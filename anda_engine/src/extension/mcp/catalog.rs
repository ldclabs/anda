//! Bounded catalog collection and cancellation-safe publication.

use super::{McpLimits, McpServerConfig, McpServerMeta, McpServerStatus, McpSession, McpToolRoute};
use anda_core::{BoxError, CancellationToken};
use parking_lot::Mutex as SyncMutex;
use rmcp::model::PaginatedRequestParams;
use std::{
    collections::BTreeSet,
    future::Future,
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use tokio::sync::{Mutex, OwnedMutexGuard, RwLock};

#[derive(Debug)]
pub(super) struct Registration {
    pub config: McpServerConfig,
    pub generation: u64,
    pub connect: Mutex<()>,
    pub refresh: Arc<Mutex<()>>,
    pub catalog: RwLock<()>,
    pub calls: RwLock<()>,
    pub revision: AtomicU64,
    pub cancelled: CancellationToken,
    pub status: SyncMutex<McpServerStatus>,
}

impl Registration {
    pub fn new(config: McpServerConfig, generation: u64) -> Self {
        Self {
            config,
            generation,
            connect: Mutex::new(()),
            refresh: Arc::new(Mutex::new(())),
            catalog: RwLock::new(()),
            calls: RwLock::new(()),
            revision: AtomicU64::new(0),
            cancelled: CancellationToken::new(),
            status: SyncMutex::new(McpServerStatus::Disconnected),
        }
    }
}

impl Deref for Registration {
    type Target = McpServerConfig;
    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

pub(super) struct DirtyGuard {
    dirty: Arc<AtomicBool>,
    published: bool,
}

impl DirtyGuard {
    pub fn new(dirty: Arc<AtomicBool>) -> Self {
        Self {
            dirty,
            published: false,
        }
    }
    pub fn commit(&mut self) {
        self.published = true;
    }
}

impl Drop for DirtyGuard {
    fn drop(&mut self) {
        if !self.published {
            self.dirty.store(true, Ordering::SeqCst);
        }
    }
}

pub(super) struct Snapshot {
    pub registration: Arc<Registration>,
    pub session: Arc<McpSession>,
    pub routes: Vec<McpToolRoute>,
    pub meta: McpServerMeta,
    pub dirty: DirtyGuard,
    pub _refresh: OwnedMutexGuard<()>,
}

pub(super) async fn collect_pages<T, F, Fut>(
    limits: &McpLimits,
    mut fetch: F,
) -> Result<Vec<T>, BoxError>
where
    F: FnMut(Option<PaginatedRequestParams>) -> Fut,
    Fut: Future<Output = Result<(Vec<T>, Option<String>), BoxError>>,
{
    let mut items = Vec::new();
    let mut cursor = None;
    let mut seen = BTreeSet::new();
    for _ in 0..limits.catalog_pages {
        let (page, next) =
            fetch(cursor.map(|cursor| PaginatedRequestParams::default().with_cursor(Some(cursor))))
                .await?;
        if page.len() > limits.catalog_items.saturating_sub(items.len()) {
            return Err("MCP catalog item limit exceeded".into());
        }
        items.extend(page);
        let Some(next) = next else {
            return Ok(items);
        };
        if next.len() > limits.cursor_bytes {
            return Err("MCP pagination cursor limit exceeded".into());
        }
        if !seen.insert(next.clone()) {
            return Err("MCP repeated pagination cursor".into());
        }
        cursor = Some(next);
    }
    Err("MCP catalog page limit exceeded".into())
}

/// A cancelled setup must not leave an observational status stuck at Connecting.
pub(super) struct ConnectingGuard<'a>(pub &'a Registration);
impl Drop for ConnectingGuard<'_> {
    fn drop(&mut self) {
        let mut status = self.0.status.lock();
        if *status == McpServerStatus::Connecting {
            *status = McpServerStatus::Disconnected;
        }
    }
}
