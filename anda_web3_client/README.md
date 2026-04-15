# `anda_web3_client`

`anda_web3_client` is a Rust SDK for Web3 integration in non-TEE environments.

## Overview

This crate provides Web3 functionality for Anda agents running outside of TEE (Trusted Execution Environment). It enables blockchain interactions without requiring TEE hardware.

## Features

- **Blockchain RPC**: Connect to various blockchain networks via HTTP RPC
- **Transaction Support**: Build and sign transactions
- **Contract Interaction**: Call smart contract methods
- **Non-TEE Compatible**: Designed for execution in standard environments

## Use Case

Use `anda_web3_client` when your agent needs to interact with blockchain networks but does not require the security guarantees of TEE. For TEE-protected interactions, use the full Anda engine with TEE support.

## License

Copyright © 2024-2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
