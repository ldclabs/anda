/// Tests the balance retrieval functionality for BNB ledgers.
///
/// This async function demonstrates the process of:
/// - Loading cryptographic identity and secrets
/// - Initializing a Web3 client for BNB Chain
/// - Creating BNB ledgers and a balance query tool
/// - Deriving a user's EVM address
/// - Querying balances for different token symbols
///
/// # Examples
///
/// test_bnb_ledger_balance().await;
///
use std::sync::Arc;

use alloy::hex;
use anda_bnb::ledger::{BNBLedgers, BalanceOfArgs, BalanceOfTool, DRVT_PATH, bnb_rpc};
use anda_bnb::signer::derive_address_from_pubkey;
use anda_core::{KeysFeatures, Tool};
use anda_engine::{context::Web3SDK, engine::EngineBuilder, extension::extractor::Extractor};
use anda_web3_client::client::{Client as Web3Client, load_identity};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use structured_logger::Builder;

use crate::common::{CHAIN_ID, TOKEN_ADDR};
mod common;

pub async fn test_bnb_ledger_balance() {
    Builder::with_level("debug").init();

    // Read identity and root secret
    let id_secret = dotenv::var("ID_SECRET").unwrap();
    let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();

    // Parse and validate cryptographic secrets
    let identity = load_identity(&id_secret).unwrap();
    let root_secret = hex::decode(&root_secret_org).unwrap();
    let root_secret: [u8; 48] = root_secret
        .try_into()
        .map_err(|_| format!("invalid root_secret: {:?}", &root_secret_org))
        .unwrap();

    // Initialize Web3 client for BNB Chain network interaction
    let web3 = Web3Client::builder()
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build()
        .await
        .unwrap();

    let mut token_addrs = BTreeSet::new();
    token_addrs.insert(TOKEN_ADDR.to_string());
    let drvt_path: Vec<Vec<u8>> = DRVT_PATH.iter().map(|&s| s.to_vec()).collect();
    // Create ledgers instance
    let ledgers = BNBLedgers::load(bnb_rpc(), CHAIN_ID, drvt_path.clone(), token_addrs)
        .await
        .unwrap();
    let ledgers = Arc::new(ledgers);
    let tool = BalanceOfTool::new(ledgers.clone());
    let definition = tool.definition();
    assert_eq!(definition.name, "bnb_ledger_balance_of");

    // Create an agent for testing
    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }
    let agent = Extractor::<TestStruct>::default();

    // Create a context for testing
    let engine_ctx = EngineBuilder::new()
        // .with_name("BNB_TEST".to_string()).unwrap()
        .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3))))
        .register_agent(agent, None)
        .unwrap()
        .mock_ctx();
    let base_ctx = engine_ctx.base.clone();

    // Derive EVM address from derivation path
    let pubkey_bytes = base_ctx
        .secp256k1_public_key(drvt_path)
        .await
        .map_err(|e| {
            format!(
                "Failed to get public key from derivation path: {:?}. Error: {:?}",
                DRVT_PATH,
                e.to_string()
            )
        })
        .unwrap();
    let user_address = derive_address_from_pubkey(&pubkey_bytes).unwrap();
    log::debug!(
        "User pubkey: {:?}, User EVM address: {:?}",
        hex::encode(pubkey_bytes),
        user_address
    );

    // Iterate through the ledgers and perform balance queries
    for (symbol, _) in ledgers.ledgers.clone() {
        // Create arguments for balance query
        let args = BalanceOfArgs {
            account: user_address.to_string(),
            symbol: symbol,
        };

        // Call the tool to query balance
        let res = tool.call(base_ctx.clone(), args, vec![]).await;
        assert!(res.is_ok(), "Balance query failed: {:?}", res);
        println!("Balance query result: {:#?}", res.unwrap());
    }
}

// cargo run --example balance_test
// cargo run -p anda_bnb --example balance_test
#[tokio::main]
async fn main() {
    println!("Anda BNB query test!");
    test_bnb_ledger_balance().await;
}
