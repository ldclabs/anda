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
    ///
    /// The anonymous principal is never treated as an administrator, even when
    /// the engine was built without a Web3 identity and `controller` defaults to
    /// the anonymous principal.
    fn is_controller(&self, caller: &Principal) -> bool {
        caller != &ANONYMOUS_PRINCIPAL && caller == &self.controller
    }

    /// Returns true if the caller is the controller or a manager of the engine.
    ///
    /// The anonymous principal is never treated as a manager, even when the
    /// engine was built without a Web3 identity and `controller` defaults to the
    /// anonymous principal.
    fn is_manager(&self, caller: &Principal) -> bool {
        caller != &ANONYMOUS_PRINCIPAL
            && (caller == &self.controller || self.managers.contains(caller))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(seed: u8) -> Principal {
        Principal::self_authenticating([seed; 32])
    }

    #[test]
    fn base_management_checks_controller_managers_visibility_and_anonymous_access() {
        let controller = principal(1);
        let manager = principal(2);
        let user = principal(3);
        let mut managers = BTreeSet::new();
        managers.insert(manager);

        let policy = BaseManagement {
            controller,
            managers,
            visibility: Visibility::Private,
        };
        assert!(policy.is_controller(&controller));
        assert!(!policy.is_controller(&manager));
        assert!(policy.is_manager(&controller));
        assert!(policy.is_manager(&manager));
        assert!(!policy.is_manager(&user));
        assert_eq!(
            policy.check_visibility(&controller).unwrap() as u8,
            Visibility::Private as u8
        );
        assert!(policy.check_visibility(&manager).is_ok());
        let Err(err) = policy.check_visibility(&user) else {
            panic!("expected private visibility error");
        };
        assert!(err.to_string().contains("not allowed"));
        let Err(err) = policy.check_visibility(&ANONYMOUS_PRINCIPAL) else {
            panic!("expected anonymous visibility error");
        };
        assert!(err.to_string().contains("anonymous"));

        let protected = BaseManagement {
            controller,
            managers: BTreeSet::new(),
            visibility: Visibility::Protected,
        };
        assert_eq!(
            protected.check_visibility(&user).unwrap() as u8,
            Visibility::Protected as u8
        );
        assert!(protected.check_visibility(&ANONYMOUS_PRINCIPAL).is_err());

        let public = BaseManagement {
            controller,
            managers: BTreeSet::new(),
            visibility: Visibility::Public,
        };
        assert_eq!(
            public.check_visibility(&ANONYMOUS_PRINCIPAL).unwrap() as u8,
            Visibility::Public as u8
        );
    }
}
