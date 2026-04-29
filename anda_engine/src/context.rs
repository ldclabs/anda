//! Execution context system for Anda agents and tools.
//!
//! Contexts carry the runtime capabilities exposed to [`anda_core`] agents and
//! tools: caller identity, request metadata, cancellation, scoped cache, object
//! storage, HTTP and canister calls, Web3 signing, remote engine access, and
//! per-session state. Contexts are hierarchical so each agent or tool receives a
//! scoped namespace while sharing the same underlying runtime resources.

mod agent;
mod base;
mod cache;
mod engine;
mod subagent;
mod tool;
mod web3;

pub use agent::*;
pub use base::*;
pub use engine::*;
pub use subagent::*;
pub use tool::*;
pub use web3::*;

/// Mock implementations for tests.
///
/// This module provides mock implementations of core interfaces that allow
/// for controlled testing environments without requiring actual canister calls.
pub mod mock {
    use anda_core::{BoxError, CanisterCaller};
    use candid::{CandidType, Decode, Principal, encode_args, utils::ArgumentEncoder};

    /// A mock implementation of CanisterCaller for testing purposes.
    ///
    /// This struct allows you to simulate canister calls by providing a transformation function
    /// that takes the canister ID, method name, and arguments, and returns a response.
    ///
    /// # Example
    /// ```rust,ignore
    /// use anda_engine::context::mock::MockCanisterCaller;
    /// use anda_core::CanisterCaller;
    /// use candid::{encode_args, CandidType, Deserialize, Principal};
    ///
    /// #[derive(CandidType, Deserialize, Debug, PartialEq)]
    /// struct TestResponse {
    ///     canister: Principal,
    ///     method: String,
    ///     args: Vec<u8>,
    /// }
    ///
    /// #[tokio::test]
    /// async fn test_mock_canister_caller() {
    ///     let canister_id = Principal::anonymous();
    ///     let empty_args = encode_args(()).unwrap();
    ///
    ///     let caller = MockCanisterCaller::new(|canister, method, args| {
    ///         let response = TestResponse {
    ///             canister: canister.clone(),
    ///             method: method.to_string(),
    ///             args,
    ///         };
    ///         candid::encode_args((response,)).unwrap()
    ///     });
    ///
    ///     let res: TestResponse = caller
    ///         .canister_query(&canister_id, "canister_query", ())
    ///         .await
    ///         .unwrap();
    ///     assert_eq!(res.canister, canister_id);
    ///     assert_eq!(res.method, "canister_query");
    ///     assert_eq!(res.args, empty_args);
    ///
    ///     let res: TestResponse = caller
    ///         .canister_update(&canister_id, "canister_update", ())
    ///         .await
    ///         .unwrap();
    ///     assert_eq!(res.canister, canister_id);
    ///     assert_eq!(res.method, "canister_update");
    ///     assert_eq!(res.args, empty_args);
    /// }
    /// ```
    pub struct MockCanisterCaller<F: Fn(&Principal, &str, Vec<u8>) -> Vec<u8> + Send + Sync> {
        transform: F,
    }

    impl<F> MockCanisterCaller<F>
    where
        F: Fn(&Principal, &str, Vec<u8>) -> Vec<u8> + Send + Sync,
    {
        /// Creates a new MockCanisterCaller with the provided transformation function.
        ///
        /// # Arguments
        /// * `transform` - A function that takes (canister_id, method_name, args) and returns
        ///   a serialized response
        pub fn new(transform: F) -> Self {
            Self { transform }
        }
    }

    impl<F> CanisterCaller for MockCanisterCaller<F>
    where
        F: Fn(&Principal, &str, Vec<u8>) -> Vec<u8> + Send + Sync,
    {
        async fn canister_query<
            In: ArgumentEncoder + Send,
            Out: CandidType + for<'a> candid::Deserialize<'a>,
        >(
            &self,
            canister: &Principal,
            method: &str,
            args: In,
        ) -> Result<Out, BoxError> {
            let args = encode_args(args)?;
            let res = (self.transform)(canister, method, args);
            let output = Decode!(res.as_slice(), Out)?;
            Ok(output)
        }

        async fn canister_update<
            In: ArgumentEncoder + Send,
            Out: CandidType + for<'a> candid::Deserialize<'a>,
        >(
            &self,
            canister: &Principal,
            method: &str,
            args: In,
        ) -> Result<Out, BoxError> {
            let args = encode_args(args)?;
            let res = (self.transform)(canister, method, args);
            let output = Decode!(res.as_slice(), Out)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_core::CanisterCaller;
    use candid::{CandidType, Deserialize, Principal, encode_args};

    #[derive(CandidType, Deserialize, Debug, PartialEq)]
    struct TestResponse {
        canister: Principal,
        method: String,
        args: Vec<u8>,
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_mock_canister_caller() {
        let canister_id = Principal::anonymous();
        let empty_args = encode_args(()).unwrap();

        let caller = mock::MockCanisterCaller::new(|canister, method, args| {
            let response = TestResponse {
                canister: *canister,
                method: method.to_string(),
                args,
            };
            candid::encode_args((response,)).unwrap()
        });

        let res: TestResponse = caller
            .canister_query(&canister_id, "canister_query", ())
            .await
            .unwrap();
        assert_eq!(res.canister, canister_id);
        assert_eq!(res.method, "canister_query");
        assert_eq!(res.args, empty_args);

        let res: TestResponse = caller
            .canister_update(&canister_id, "canister_update", ())
            .await
            .unwrap();
        assert_eq!(res.canister, canister_id);
        assert_eq!(res.method, "canister_update");
        assert_eq!(res.args, empty_args);
    }
}
