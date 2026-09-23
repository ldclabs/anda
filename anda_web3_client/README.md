# `anda_web3_client`

`anda_web3_client` is a Rust SDK for Web3 integration in non-TEE environments.

## Overview

This crate provides a concrete `Web3ClientFeatures` implementation for Anda agents running outside of a TEE (Trusted Execution Environment). It backs the engine's Web3 context with an `ic-agent`-based client and local key derivation, so no TEE hardware is required.

## Features

- **ICP canister calls**: Query and update Internet Computer canisters through `ic-agent`.
- **Signed HTTP & CBOR-RPC**: Make plain and identity-signed HTTPS requests, plus signed CBOR-RPC calls (ICP-style signed envelopes).
- **Deterministic key derivation**: Derive AES-256-GCM keys and Ed25519 / secp256k1 keys (ECDSA and BIP-340 Schnorr) from a 48-byte root secret, byte-for-byte compatible with the TEE gateway.
- **COSE integration**: Interact with the [IC-COSE](https://github.com/ldclabs/ic-cose) canister via the `CoseSDK` trait.
- **Non-TEE compatible**: Designed for execution in standard environments.

Feature flags: `client` (the generic non-TEE client), `tee` (a TEE gateway-backed client), and `full` (both). The default build enables neither.

## Use Case

Use `anda_web3_client` (the `client` feature) when your agent needs ICP canister access, signed HTTP/RPC calls, and key derivation but does not require the security guarantees of a TEE. For TEE-protected interactions, enable `tee`, connect a gateway client, and wrap it with `tee::TeeClient` for `Web3SDK::from_web3`.

Endpoints passed to the signed HTTP/RPC calls must be trusted: requests are signed with the client identity before being sent, so a hostile endpoint receives a valid signed request. Both clients reject malformed URLs, non-HTTP(S) schemes, and embedded userinfo (`https://trusted@evil/`) before signing; this is a syntactic guard, not an SSRF filter.

## Configuration

Enable `client` in your dependency:

```toml
anda_web3_client = { version = "0.16", features = ["client"] }
```

Provide a persistent, securely generated 48-byte root secret when using key
operations. Generate it once, keep it in your secret store, and reuse it across
restarts; changing it changes every derived key. The following complete example
is also available in [`examples/client.rs`](examples/client.rs):

```rust
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
```

Run it with `cargo run -p anda_web3_client --features client --example client`,
with `ANDA_ROOT_SECRET_HEX` supplied by your environment or secret manager.

- `build()` requires an identity or a root secret and rejects all-zero root
  secrets. There is no implicit default identity or derivation secret.
- `with_identity(...)` alone supports canister calls, envelope signing, and
  signed HTTP/RPC. Key derivation and path-based signing/verification return an
  error until `with_root_secret(...)` is configured. The identity does not change
  the keys derived from a given root secret and path.
- `with_agent(...)` requires the agent's principal to match the client identity;
  the supplied agent's transport, root key, and verification settings are retained.
- The default agent verifies query signatures and uses `ic-agent`'s node-key
  cache. To opt out for a trusted endpoint, explicitly supply an agent built with
  `with_verify_query_signatures(false)` and the same identity.
- An `http://` IC host is treated as a local replica: root-key fetching must
  succeed within ten seconds. Startup errors are returned; retry construction
  after the replica becomes available. For other replica configurations, supply
  an agent with an explicitly configured root key. Only fetch root keys from a
  trusted development replica, never from an untrusted mainnet endpoint.
- `with_allow_http(true)` enables plain HTTP for the generic client's HTTP/RPC
  methods. A custom HTTP client controls timeouts, redirects, proxies, and TLS.
  TEE RPC uses the gateway's configured external HTTP client, whose default is
  HTTPS-only.
- Both RPC backends use Anda's bounded reader: success responses are limited to
  16 MiB and error diagnostics to 8 KiB. Raw RPC preserves encoded parameter and
  result bytes; callers supply CBOR-encoded params and decode the returned bytes.
- ECDSA digest signing and verification require exactly 32 bytes. For signatures
  from `secp256k1_sign_message_ecdsa`, verify with `SHA256(message)`. The low-level
  `crypto::secp256k1_sign_digest_ecdsa` helper accepts `&[u8; 32]`.

Existing nonzero root secrets retain their derived keys. AES derivation keeps
its gateway-compatible concatenation of path segments; segment boundaries alone
do not distinguish paths such as `[b"ab", b"c"]` and `[b"a", b"bc"]`.

## Validation

Run feature configurations independently to avoid workspace feature unification
hiding missing optional dependencies:

```sh
cargo test -p anda_web3_client --no-default-features
cargo test -p anda_web3_client --no-default-features --features client
cargo test -p anda_web3_client --no-default-features --features tee
cargo test -p anda_web3_client --features full
```

The default feature set has no runtime dependencies. Client and TEE transport
tests use local servers; compatibility tests compare local derivation with the
gateway implementation and fixed vectors without requiring TEE hardware.

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
