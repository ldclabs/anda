//! Background-task registry for long-running tool and agent work.
//!
//! A producer that moves work off the request path registers a
//! [`BackgroundHandle`] (a per-task child [`CancellationToken`] plus optional
//! shared payload) in [`BackgroundTaskControls`]; subscribers observe, stop,
//! and finish tasks by ID. [`PrefixedId`] namespaces task identifiers.
//! These types are consumed by the hook traits in [`crate::hook`], but they
//! are a registry, not hooks: nothing here intercepts execution.

use anda_core::{BoxError, CancellationToken};
use core::{fmt, str::FromStr};
use parking_lot::RwLock;
use std::{any::Any, collections::HashMap, sync::Arc};
use structured_logger::unix_ms;

/// Identifier split into a namespace prefix and local ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrefixedId {
    /// Namespace or hook prefix.
    pub prefix: String,
    /// Local identifier within the prefix.
    pub id: String,
}

impl fmt::Display for PrefixedId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.prefix, self.id)
    }
}

impl FromStr for PrefixedId {
    type Err = BoxError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((prefix, id)) = s.split_once(':') else {
            return Err(format!("Invalid PrefixedId format: {}", s).into());
        };
        if prefix.trim().is_empty() || id.trim().is_empty() {
            return Err(format!("Prefix and ID cannot be empty: {}", s).into());
        }
        if prefix.trim() != prefix || id.trim() != id {
            return Err(format!(
                "Prefix and ID cannot have leading or trailing whitespace: {}",
                s
            )
            .into());
        }
        Ok(Self {
            prefix: prefix.to_string(),
            id: id.to_string(),
        })
    }
}

/// A cloneable capability handed to background-task subscribers so they can request
/// cancellation of a single background task without tearing down the whole request.
///
/// The handle wraps a per-task [`CancellationToken`] (a child of the task context's
/// token). Calling [`stop`](BackgroundHandle::stop) cancels only that task: the
/// producer's background loop observes the cancellation and terminates the task, while
/// sibling tasks and the enclosing request keep running. Request-level cancellation still
/// cascades down, because the per-task token is a child of the request token.
///
/// The handle also carries a creation timestamp (for observing run time) and an optional,
/// type-erased application payload attached with [`with_data`](BackgroundHandle::with_data).
/// The payload is shared by [`Arc`] across every clone (the subscriber's copy and the one
/// kept in [`BackgroundTaskControls`]), so a subscriber can keep per-task bookkeeping on the
/// handle itself instead of maintaining a parallel map. Use interior mutability (e.g.
/// [`Mutex`](std::sync::Mutex)) inside the payload when it must change over the task's lifetime.
#[derive(Clone)]
pub struct BackgroundHandle {
    task_id: String,
    token: CancellationToken,
    created_at: u64,
    data: Option<Arc<dyn Any + Send + Sync>>,
}

impl BackgroundHandle {
    /// Wraps a per-task cancellation token under its task ID. The creation timestamp is
    /// stamped now and no payload is attached.
    pub fn new(task_id: impl Into<String>, token: CancellationToken) -> Self {
        Self {
            task_id: task_id.into(),
            token,
            created_at: unix_ms(),
            data: None,
        }
    }

    /// Attaches an application-defined payload, wrapping it in an [`Arc`] so every clone of
    /// the handle observes the same value.
    pub fn with_data<T: Any + Send + Sync>(mut self, data: T) -> Self {
        self.data = Some(Arc::new(data));
        self
    }

    /// Attaches an already-shared payload, so the producer can keep its own [`Arc`] clone in
    /// addition to the copy carried by the handle.
    pub fn with_shared_data<T: Any + Send + Sync>(mut self, data: Arc<T>) -> Self {
        let data: Arc<dyn Any + Send + Sync> = data;
        self.data = Some(data);
        self
    }

    /// Downcasts the attached payload to `T`, returning a shared handle to it. Returns `None`
    /// when no payload is attached or it is of another type.
    pub fn data<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.data.clone()?.downcast::<T>().ok()
    }

    /// The background task's identifier.
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Wall-clock time in milliseconds (Unix epoch) when this handle, and thus the task, was
    /// created.
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Milliseconds elapsed since the task was created, for observing its run time.
    pub fn elapsed_ms(&self) -> u64 {
        unix_ms().saturating_sub(self.created_at)
    }

    /// Requests the producer stop this background task. Idempotent and safe to call
    /// after the task has already ended.
    pub fn stop(&self) {
        self.token.cancel();
    }

    /// Returns whether a stop has already been requested for this task.
    pub fn is_stopped(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// Registry of the [`BackgroundHandle`]s for a subscriber's live background tasks, keyed by
/// task ID.
///
/// Producers register a handle when a task starts and drop it (via
/// [`finish`](BackgroundTaskControls::finish)) when the task ends; subscribers call
/// [`stop_background_task`](BackgroundTaskControls::stop_background_task) to actively
/// terminate a still-running task, and [`get`](BackgroundTaskControls::get) /
/// [`handles`](BackgroundTaskControls::handles) to read the handles (and their payloads) for
/// progress reporting. Cloning shares the same underlying registry.
///
/// Stopping a task only signals its cancellation token; the handle stays registered until
/// [`finish`](BackgroundTaskControls::finish) removes it. This lets a late progress or end
/// callback — which only carries a task ID — still find the handle and observe that the task
/// was stopped, so it can suppress duplicate output instead of forwarding it.
#[derive(Clone, Default)]
pub struct BackgroundTaskControls {
    tasks: Arc<RwLock<HashMap<String, BackgroundHandle>>>,
}

impl BackgroundTaskControls {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a handle so the task can later be observed and stopped by ID.
    pub fn register(&self, handle: BackgroundHandle) {
        self.tasks
            .write()
            .insert(handle.task_id().to_string(), handle);
    }

    /// Returns a clone of the handle for `task_id`, if still registered. Gives callers access
    /// to the handle's application payload for progress and end bookkeeping.
    pub fn get(&self, task_id: &str) -> Option<BackgroundHandle> {
        self.tasks.read().get(task_id).cloned()
    }

    /// Downcasts the payload of the handle for `task_id` to `T` without cloning the whole
    /// handle. Returns `None` when the task is not registered or its payload is of another type.
    pub fn get_data<T: Any + Send + Sync>(&self, task_id: &str) -> Option<Arc<T>> {
        self.tasks.read().get(task_id)?.data::<T>()
    }

    /// Returns clones of every currently registered handle, for status reporting.
    pub fn handles(&self) -> Vec<BackgroundHandle> {
        self.tasks.read().values().cloned().collect()
    }

    /// Whether any task is still registered.
    pub fn is_empty(&self) -> bool {
        self.tasks.read().is_empty()
    }

    /// Forgets a task's handle once it has ended, without stopping it, returning the removed
    /// handle so a caller can read its payload one last time. This is the only path that removes
    /// a handle, so a stopped-but-not-yet-ended task stays observable to late progress and end
    /// callbacks.
    pub fn finish(&self, task_id: &str) -> Option<BackgroundHandle> {
        self.tasks.write().remove(task_id)
    }

    /// Forgets every registered handle at once, returning the removed handles so a caller can do
    /// final bookkeeping on their payloads. Like [`finish`](BackgroundTaskControls::finish) but
    /// for the whole registry; it does not stop the tasks, which a torn-down request cancels on
    /// its own through the context-token cascade.
    pub fn finish_all(&self) -> Vec<BackgroundHandle> {
        self.tasks
            .write()
            .drain()
            .map(|(_, handle)| handle)
            .collect()
    }

    /// Actively stops a single background task by ID.
    ///
    /// The handle stays registered so a later progress or end callback can still observe that
    /// the task was stopped; removal happens through [`finish`](BackgroundTaskControls::finish).
    /// Returns `true` if a task was found and signalled, `false` if no such task was registered
    /// (it already ended or never started).
    pub fn stop_background_task(&self, task_id: &str) -> bool {
        match self.tasks.read().get(task_id) {
            Some(handle) => {
                handle.stop();
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    #[test]
    fn background_handle_cancels_only_its_token() {
        let parent = CancellationToken::new();
        let task_token = parent.child_token();
        let handle = BackgroundHandle::new("shell:1", task_token.clone());

        assert_eq!(handle.task_id(), "shell:1");
        assert!(!handle.is_stopped());
        assert!(!task_token.is_cancelled());

        handle.stop();
        assert!(handle.is_stopped());
        assert!(task_token.is_cancelled());
        // Stopping one task must not cancel the enclosing request token.
        assert!(!parent.is_cancelled());
        // Idempotent.
        handle.stop();
    }

    #[test]
    fn background_handle_carries_payload_and_created_at() {
        let handle =
            BackgroundHandle::new("shell:1", CancellationToken::new()).with_data(Mutex::new(7u32));

        // The creation timestamp is stamped and elapsed time is observable.
        assert!(handle.created_at() > 0);
        let _ = handle.elapsed_ms();

        // The payload is shared across clones: a mutation through one is seen through the other.
        let clone = handle.clone();
        *handle.data::<Mutex<u32>>().unwrap().lock() = 42;
        assert_eq!(*clone.data::<Mutex<u32>>().unwrap().lock(), 42);

        // Downcasting to the wrong type, or reading a handle without payload, returns None.
        assert!(handle.data::<Mutex<String>>().is_none());
        let bare = BackgroundHandle::new("shell:2", CancellationToken::new());
        assert!(bare.data::<Mutex<u32>>().is_none());
    }

    #[test]
    fn background_task_controls_stop_finish_and_finish_all() {
        let controls = BackgroundTaskControls::new();
        let token_a = CancellationToken::new();
        let token_b = CancellationToken::new();
        controls.register(BackgroundHandle::new("a", token_a.clone()));
        controls.register(BackgroundHandle::new("b", token_b.clone()));

        // Stopping a live task signals its token and reports it was found, but keeps the handle
        // registered so a late progress/end callback can still observe the stop.
        assert!(controls.stop_background_task("a"));
        assert!(token_a.is_cancelled());
        assert!(controls.get("a").is_some());
        // Stop is idempotent and still reports the task as found until it is finished.
        assert!(controls.stop_background_task("a"));
        // Unknown task ids report not found.
        assert!(!controls.stop_background_task("missing"));

        // finish() forgets a handle and returns it, without cancelling.
        assert_eq!(controls.finish("a").unwrap().task_id(), "a");
        assert!(controls.get("a").is_none());
        assert!(!controls.stop_background_task("a"));

        controls.finish("b");
        assert!(!token_b.is_cancelled());
        assert!(!controls.stop_background_task("b"));

        // finish_all() drains every remaining handle without stopping the tasks.
        let token_c = CancellationToken::new();
        controls.register(BackgroundHandle::new("c", token_c.clone()));
        assert!(!controls.is_empty());
        let drained = controls.finish_all();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].task_id(), "c");
        assert!(!token_c.is_cancelled());
        assert!(controls.is_empty());
    }

    #[test]
    fn prefixed_id_display_and_parse_validate_shape() {
        let id: PrefixedId = "agent:run-1".parse().unwrap();
        assert_eq!(id.prefix, "agent");
        assert_eq!(id.id, "run-1");
        assert_eq!(id.to_string(), "agent:run-1");

        assert!("missing_separator".parse::<PrefixedId>().is_err());
        assert!(":empty_prefix".parse::<PrefixedId>().is_err());
        assert!("empty_id:".parse::<PrefixedId>().is_err());
        assert!(" agent:run".parse::<PrefixedId>().is_err());
        assert!("agent:run ".parse::<PrefixedId>().is_err());
    }
}
