//! Fixed-size lock striping bounds registry memory across arbitrary model paths.

use anda_core::{BoxError, CancellationToken};
use std::{
    hash::{Hash, Hasher},
    path::Path,
    sync::{Arc, LazyLock},
};
use tokio::sync::{Mutex, OwnedMutexGuard};

static LOCKS: LazyLock<[Arc<Mutex<()>>; 256]> =
    LazyLock::new(|| std::array::from_fn(|_| Arc::new(Mutex::new(()))));

pub(super) async fn lock_paths<'a>(
    paths: impl IntoIterator<Item = &'a Path>,
    cancellation: &CancellationToken,
) -> Result<Vec<OwnedMutexGuard<()>>, BoxError> {
    let mut stripes = paths
        .into_iter()
        .map(|path| {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            #[cfg(any(windows, target_os = "macos"))]
            path.to_string_lossy().to_lowercase().hash(&mut hash);
            #[cfg(not(any(windows, target_os = "macos")))]
            path.hash(&mut hash);
            hash.finish() as usize % LOCKS.len()
        })
        .collect::<Vec<_>>();
    stripes.sort_unstable();
    stripes.dedup();
    let mut guards = Vec::with_capacity(stripes.len());
    for stripe in stripes {
        let guard = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err("File operation cancelled".into()),
            guard = LOCKS[stripe].clone().lock_owned() => guard,
        };
        guards.push(guard);
    }
    Ok(guards)
}
