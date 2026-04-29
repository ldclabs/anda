//! Object storage support for engine contexts.
//!
//! [`Store`] wraps an [`ObjectStore`] backend and applies namespace isolation for
//! agent and tool contexts. Each context receives a namespace derived from its
//! path, so shared storage backends can safely hold data for many agents and
//! tools.
//!
//! The module also defines [`VectorSearchFeaturesDyn`] and [`VectorStore`] for
//! integrations that expose vector retrieval alongside object storage.
//!
//! ## Examples
//! Basic usage:
//! ```rust,ignore
//! let store = Store::new(Arc::new(InMemory::new()));
//! store.store_put(&namespace, &path, PutMode::Create, data).await?;
//! let (content, meta) = store.store_get(&namespace, &path).await?;
//! ```

use anda_core::{BoxError, BoxPinFut, ObjectMeta, Path, PutMode, PutResult, path_join};
use futures::TryStreamExt;
use object_store::PutOptions;
use std::sync::Arc;

pub use object_store::{ObjectStore, ObjectStoreExt, local::LocalFileSystem, memory::InMemory};

/// Maximum object size accepted by store-backed context APIs.
pub const MAX_STORE_OBJECT_SIZE: usize = 1024 * 1024 * 2; // 2 MB

/// Object-safe vector search interface.
pub trait VectorSearchFeaturesDyn: Send + Sync + 'static {
    /// Finds the top `n` similar item identifiers for a query string.
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>>;

    /// Finds the top `n` similar internal IDs for a query string.
    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>>;
}

/// Cloneable wrapper around a vector search implementation.
#[derive(Clone)]
pub struct VectorStore {
    inner: Arc<dyn VectorSearchFeaturesDyn>,
}

impl VectorStore {
    /// Creates a vector store from an implementation.
    pub fn new(inner: Arc<dyn VectorSearchFeaturesDyn>) -> Self {
        Self { inner }
    }

    /// Creates a placeholder vector store that returns `not implemented` errors.
    pub fn not_implemented() -> Self {
        Self {
            inner: Arc::new(NotImplemented),
        }
    }
}

impl VectorSearchFeaturesDyn for VectorStore {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        self.inner.top_n(namespace, query, n)
    }

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        self.inner.top_n_ids(namespace, query, n)
    }
}

/// A placeholder for not implemented features.
#[derive(Clone, Debug)]
pub struct NotImplemented;

impl VectorSearchFeaturesDyn for NotImplemented {
    fn top_n(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn top_n_ids(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

/// Mock vector search implementation that returns empty result sets.
#[derive(Clone, Debug)]
pub struct MockImplemented;

impl VectorSearchFeaturesDyn for MockImplemented {
    fn top_n(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Ok(vec![])))
    }

    fn top_n_ids(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Ok(vec![])))
    }
}

/// Namespace-aware object storage facade used by engine contexts.
///
/// In Anda Engine, the namespace is derived from an agent or tool context path,
/// which isolates data for each registered component.
///
/// Any [`ObjectStore`] implementation can be used, including in-memory,
/// filesystem, cloud, and IC-COSE-backed stores.
///
/// You can find various implementations of [`ObjectStore`] at:
/// <https://github.com/apache/arrow-rs/tree/main/object_store>
///
/// Alternatively, you can use [IC-COSE](https://github.com/ldclabs/ic-cose)'s
/// [`ObjectStore`] implementation, which stores data on the ICP blockchain.
#[derive(Clone)]
pub struct Store {
    store: Arc<dyn ObjectStore>,
}

impl Store {
    /// Creates a storage facade from an object-store backend.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }

    /// Retrieves object bytes and metadata from a namespace-relative path.
    pub async fn store_get(
        &self,
        namespace: &Path,
        path: &Path,
    ) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        let path = path_join(namespace, path);
        let res = self.store.get_opts(&path, Default::default()).await?;
        let data = match res.payload {
            object_store::GetResultPayload::Stream(mut stream) => {
                let mut buf = bytes::BytesMut::new();
                while let Some(data) = stream.try_next().await? {
                    buf.extend_from_slice(&data);
                }
                buf.freeze() // Convert to immutable Bytes
            }
            _ => return Err("StoreFeatures: unexpected payload from get_opts".into()),
        };
        Ok((data, res.meta))
    }

    /// Lists objects with optional namespace-relative prefix and offset filters.
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results
    /// * `offset` - Optional path to start listing from (exclude)
    pub async fn store_list(
        &self,
        namespace: &Path,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> Result<Vec<ObjectMeta>, BoxError> {
        let prefix = prefix.map(|p| path_join(namespace, p));
        let offset = path_join(namespace, offset);

        let mut res = if offset.is_root() {
            self.store.list(prefix.as_ref())
        } else {
            self.store.list_with_offset(prefix.as_ref(), &offset)
        };
        let mut metas = Vec::new();
        while let Some(meta) = res.try_next().await? {
            metas.push(meta)
        }

        Ok(metas)
    }

    /// Stores bytes at a namespace-relative path with the given write mode.
    ///
    /// # Arguments
    /// * `path` - Target storage path
    /// * `mode` - Write mode (Create, Overwrite, etc.)
    /// * `val` - Data to store as bytes
    pub async fn store_put(
        &self,
        namespace: &Path,
        path: &Path,
        mode: PutMode,
        val: bytes::Bytes,
    ) -> Result<PutResult, BoxError> {
        let full_path = path_join(namespace, path);
        let res = self
            .store
            .put_opts(
                &full_path,
                val.into(),
                PutOptions {
                    mode,
                    ..Default::default()
                },
            )
            .await?;
        Ok(res)
    }

    /// Renames an object if the target path does not exist.
    ///
    /// # Arguments
    /// * `from` - Source path
    /// * `to` - Destination path
    pub async fn store_rename_if_not_exists(
        &self,
        namespace: &Path,
        from: &Path,
        to: &Path,
    ) -> Result<(), BoxError> {
        let from = path_join(namespace, from);
        let to = path_join(namespace, to);
        self.store.rename_if_not_exists(&from, &to).await?;
        Ok(())
    }

    /// Deletes an object at a namespace-relative path.
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete
    pub async fn store_delete(&self, namespace: &Path, path: &Path) -> Result<(), BoxError> {
        let path = path_join(namespace, path);
        self.store.delete(&path).await?;
        Ok(())
    }
}
