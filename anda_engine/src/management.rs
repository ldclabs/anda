//! Engine visibility and caller management policies.
//!
//! Management policies decide which principals can administer an engine and who
//! can access exported agents and tools. The default [`BaseManagement`] policy
//! makes engines private to the controller and explicit managers.

use anda_core::BoxError;
use async_trait::async_trait;
use candid::Principal;
use ic_auth_verifier::ANONYMOUS_PRINCIPAL;
use std::collections::BTreeSet;

/// Root cache and storage namespace reserved for engine system data.
pub static SYSTEM_PATH: &str = "_";

/// Authorization policy used by [`Engine`](crate::engine::Engine).
#[async_trait]
pub trait Management: Send + Sync {
    /// Returns whether `caller` is the engine controller.
    fn is_controller(&self, caller: &Principal) -> bool;

    /// Returns whether `caller` can manage private engine state.
    fn is_manager(&self, caller: &Principal) -> bool;

    /// Validates access and returns the current engine visibility.
    fn check_visibility(&self, caller: &Principal) -> Result<Visibility, BoxError>;
}

/// Basic principal-list management policy for an engine.
pub struct BaseManagement {
    /// Principal that controls the engine.
    pub controller: Principal,

    /// Additional principals with manager privileges.
    pub managers: BTreeSet<Principal>,

    /// Public access level for non-manager callers.
    pub visibility: Visibility,
}

/// Engine visibility for exported agents and tools.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Only the controller and managers can access the engine.
    Private = 0,

    /// Managers can access the engine; non-manager access requires an external
    /// policy to grant permission before execution.
    Protected = 1,

    /// Any caller, including anonymous callers, can access exported functions.
    Public = 2,
}

#[async_trait]
impl Management for BaseManagement {
    /// Returns true if the caller is the controller of the engine.
    fn is_controller(&self, caller: &Principal) -> bool {
        caller == &self.controller
    }

    /// Returns true if the caller is the controller or a manager of the engine.
    fn is_manager(&self, caller: &Principal) -> bool {
        caller == &self.controller || self.managers.contains(caller)
    }

    /// Checks anonymous access and private visibility rules.
    fn check_visibility(&self, caller: &Principal) -> Result<Visibility, BoxError> {
        if self.visibility != Visibility::Public && caller == &ANONYMOUS_PRINCIPAL {
            return Err("anonymous caller not allowed".into());
        }

        if self.visibility == Visibility::Private && !self.is_manager(caller) {
            return Err("caller is not allowed".into());
        }

        Ok(self.visibility)
    }
}
