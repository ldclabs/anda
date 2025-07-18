use anda_core::{ANONYMOUS, BoxError, Xid};
use async_trait::async_trait;
use candid::Principal;
use std::collections::BTreeSet;

mod db;
mod thread;
mod user;

pub use db::*;
pub use thread::*;
pub use user::*;

pub static SYSTEM_PATH: &str = "_";

#[async_trait]
pub trait Management: Send + Sync {
    fn is_controller(&self, caller: &Principal) -> bool;
    fn is_manager(&self, caller: &Principal) -> bool;
    fn check_visibility(&self, caller: &Principal) -> Result<Visibility, BoxError>;

    async fn get_user(&self, _caller: &Principal) -> Result<UserState, BoxError> {
        Err("`get_user` is not implemented".into())
    }

    async fn add_user(&self, _caller: Principal) -> Result<(), BoxError> {
        Err("`add_user` is not implemented".into())
    }

    async fn update_user(&self, _user: &UserState) -> Result<(), BoxError> {
        Err("`save_user` is not implemented".into())
    }

    async fn get_thread(&self, _id: &Xid) -> Result<Thread, BoxError> {
        Err("`get_thread` is not implemented".into())
    }

    // TODO: more management methods
}

/// Represents system management tools for the Anda engine.
pub struct BaseManagement {
    pub controller: Principal,
    pub managers: BTreeSet<Principal>,
    pub visibility: Visibility, // 0: private, 1: protected, 2: public
}

/// The visibility of the engine.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// private, can only be accessed by the controller and managers;
    Private = 0,

    /// protected, can be accessed by the controller, managers, and users who have permission;
    Protected = 1,

    /// public, can be accessed by anyone.
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

    fn check_visibility(&self, caller: &Principal) -> Result<Visibility, BoxError> {
        if self.visibility != Visibility::Public && caller == &ANONYMOUS {
            return Err("anonymous caller not allowed".into());
        }

        if self.visibility == Visibility::Private && !self.is_manager(caller) {
            return Err("caller is not allowed".into());
        }

        Ok(self.visibility)
    }
}
