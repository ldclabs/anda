/// Tests the BNB ledger transfer functionality by performing token transfers across different ledgers.
///
/// This test function demonstrates the process of:
/// - Loading cryptographic identity and secrets
/// - Creating a BNB ledger instance
/// - Initializing a Web3 client
/// - Creating an engine context
/// - Iterating through ledgers and performing token transfers
///
/// # Panics
/// Panics if identity loading, ledger loading, or token transfers fail
///
/// # Environment Variables
/// - `ID_SECRET`: Cryptographic identity secret
/// - `ROOT_SECRET`: Root secret for Web3 client
///
use std::{collections::BTreeSet, sync::Arc};

use alloy::hex;
use alloy::primitives::address;
use anda_bnb::ledger::{BNBLedgers, DRVT_PATH, TransferToArgs, TransferTool, bnb_rpc};
use anda_core::Tool;
use anda_engine::{context::Web3SDK, engine::EngineBuilder, extension::extractor::Extractor};
use anda_web3_client::client::{Client as Web3Client, load_identity};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use structured_logger::Builder;

use crate::common::{CHAIN_ID, TOKEN_ADDR};
mod common;

pub async fn test_bnb_ledger_transfer() {
    Builder::with_level("debug").init();

    // Generate random bytes for identity and root secret
    let id_secret = dotenv::var("ID_SECRET").unwrap();
    let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();

    // Parse cryptographic secrets
    let identity = load_identity(&id_secret).unwrap();
    let root_secret = hex::decode(&root_secret_org).unwrap();
    let root_secret: [u8; 48] = root_secret
        .try_into()
        .map_err(|_| format!("invalid root_secret: {:?}", &root_secret_org))
        .unwrap();

    let mut token_addrs = BTreeSet::new();
    token_addrs.insert(TOKEN_ADDR.to_string());
    let drvt_path: Vec<Vec<u8>> = DRVT_PATH.iter().map(|&s| s.to_vec()).collect();

    // Create a BNB ledger instance
    let ledgers = BNBLedgers::load(bnb_rpc(), CHAIN_ID, drvt_path.clone(), token_addrs)
        .await
        .unwrap();
    let ledgers = Arc::new(ledgers);
    let tool = TransferTool::new(ledgers.clone());
    let definition = tool.definition();
    assert_eq!(definition.name, "bnb_ledger_transfer");
    assert!(
        tool.description()
            .contains(ledgers.ledgers.clone().first_key_value().unwrap().0)
    );

    // Create an agent for testing
    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }
    let agent = Extractor::<TestStruct>::default();

    // Initialize Web3 client for BNB network
    let web3 = Web3Client::builder()
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build()
        .await
        .unwrap();

    // Create a context for testing
    let engine_ctx = EngineBuilder::new()
        .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3))))
        .register_agent(agent, None)
        .unwrap()
        .mock_ctx();
    let base_ctx = engine_ctx.base;

    // Iterate through the ledgers and perform transfers
    for (symbol, _) in ledgers.ledgers.clone() {
        // Init transfer arguments
        let to_addr = address!("0xA8c4AAE4ce759072D933bD4a51172257622eF128"); // Receiver addr
        let transfer_amount = 0.00012;
        let transfer_to_args = TransferToArgs {
            account: to_addr.to_string(),
            symbol: symbol.clone(),
            amount: transfer_amount,
        };

        // Call tool to transfer tokens
        let res = tool.call(base_ctx.clone(), transfer_to_args, vec![]).await;
        assert!(res.is_ok(), "Transfer failed: {:?}", res);
        println!("Transfer result: {:#?}", res.unwrap());
    }
}

// cargo run --example transfer_test
// cargo run -p anda_bnb --example transfer_test
#[tokio::main]
async fn main() {
    println!("Anda BNB transfer test!");
    test_bnb_ledger_transfer().await;
}
