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

Use `anda_web3_client` (the `client` feature) when your agent needs ICP canister access, signed HTTP/RPC calls, and key derivation but does not require the security guarantees of a TEE. For TEE-protected interactions, use the `tee` feature or the full Anda engine with TEE support.

Endpoints passed to the signed HTTP/RPC calls must be trusted: requests are signed with the client identity before being sent, so a hostile endpoint receives a valid signed request.

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
