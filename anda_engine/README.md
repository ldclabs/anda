# `anda_engine`: Anda Agents Engine

![License](https://img.shields.io/crates/l/anda_engine.svg)
[![Crates.io](https://img.shields.io/crates/d/anda_engine.svg)](https://crates.io/crates/anda_engine)
[![Test](https://github.com/ldclabs/anda/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda/actions/workflows/test.yml)
[![Docs.rs](https://docs.rs/anda_engine/badge.svg)](https://docs.rs/anda_engine)
[![Latest Version](https://img.shields.io/crates/v/anda_engine.svg)](https://crates.io/crates/anda_engine)

Agents engine for Anda - A comprehensive framework for building and managing AI agents.

More information about this crate can be found in the [crate documentation][docs].

## Overview

`anda_engine` is a complete implementation of the [`anda_core`](https://github.com/ldclabs/anda/tree/main/anda_core) AI Agent definition, providing a robust foundation for creating, managing, and executing AI agents with the following core capabilities:

- **Agent Management**: Create, configure, and execute AI agents with customizable behaviors
- **Tool Integration**: Register and manage tools that agents can utilize
- **Context Management**: Handle execution contexts with cancellation support
- **Storage System**: Persistent storage with object and vector search capabilities
- **Model Integration**: Support for multiple AI model providers (OpenAI, DeepSeek, Cohere)
- **Extension System**: Additional capabilities including attention management and document processing

## License
Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `anda` by you, shall be licensed as MIT, without any
additional terms or conditions.

[docs]: https://docs.rs/anda_engine
[license]: ./../LICENSE-MIT