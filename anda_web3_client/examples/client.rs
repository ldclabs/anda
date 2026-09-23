//! Run with ANDA_ROOT_SECRET_HEX set to a persistent 48-byte secret in hex.
use anda_core::BoxError;
use anda_engine::context::Web3ClientFeatures;
use anda_web3_client::Client;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let secret = hex::decode(std::env::var("ANDA_ROOT_SECRET_HEX")?)?;
    let secret: [u8; 48] = secret.try_into().map_err(|_| "expected 48 bytes")?;
    let client = Client::builder().with_root_secret(secret).build().await?;
    let public_key = client.ed25519_public_key(vec![b"wallet".to_vec()]).await?;
    println!("Principal: {}", client.get_principal());
    println!("Wallet public key: {}", hex::encode(public_key));
    Ok(())
}
