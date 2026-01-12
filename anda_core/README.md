# `anda_core`: Anda Core Library

![License](https://img.shields.io/crates/l/anda_core.svg)
[![Crates.io](https://img.shields.io/crates/d/anda_core.svg)](https://crates.io/crates/anda_core)
[![Test](https://github.com/ldclabs/anda/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda/actions/workflows/test.yml)
[![Docs.rs](https://docs.rs/anda_core/badge.svg)](https://docs.rs/anda_core)
[![Latest Version](https://img.shields.io/crates/v/anda_core.svg)](https://crates.io/crates/anda_core)

The Anda Core Library provides the fundamental building blocks for creating and managing AI agents and tools in a modular, secure, and extensible system.

More information about this crate can be found in the [crate documentation][docs].

## Key Features

- **Modular Architecture**: Separates concerns into distinct modules for agents, tools, context, and models
- **Type Safety**: Strongly typed interfaces for agent and tool definitions
- **Asynchronous Execution**: All operations are async for efficient I/O handling
- **Dynamic Dispatch**: Supports runtime polymorphism for agents and tools
- **Security Features**: Includes cryptographic operations and verified caller information
- **Extensibility**: New features can be added through modular trait implementations

## Core Modules

### 1. Agent Module [`agent.rs`](https://github.com/ldclabs/anda/blob/main/anda_core/src/agent.rs)
Provides core functionality for creating and managing AI agents:
- `Agent` trait for defining custom agents
- `AgentDyn` trait for runtime polymorphism
- `AgentSet` for managing multiple agents

### 2. Tool Module [`tool.rs`](https://github.com/ldclabs/anda/blob/main/anda_core/src/tool.rs)
Defines the core functionality for creating and managing tools:
- `Tool` trait for defining custom tools with typed arguments
- `ToolDyn` trait for runtime polymorphism
- `ToolSet` for managing multiple tools

### 3. Context Module [`context.rs`](https://github.com/ldclabs/anda/blob/main/anda_core/src/context.rs)
Provides the execution environment for agents and tools:
- `AgentContext` as the primary interface combining all capabilities
- `BaseContext` for fundamental operations
- Feature sets including:
  - State management
  - Cryptographic operations
  - Persistent storage
  - In-memory caching
  - HTTP communication
  - Blockchain interactions

### 4. Model Module [`model`](https://github.com/ldclabs/anda/blob/main/anda_core/src/model/mod.rs)
Defines core data structures and interfaces for LLMs:
- Agent output and message structures
- Function definitions with JSON schema support
- Knowledge and document handling
- Completion and embedding request/response structures
- Core AI capabilities traits

### 5. HTTP Module [`http.rs`](https://github.com/ldclabs/anda/blob/main/anda_core/src/http.rs)
Provides utilities for making RPC calls:
- CBOR-encoded RPC calls
- Candid-encoded canister calls
- HTTP request/response handling
- Error handling for RPC operations

## Key Concepts

### Agent System
- Agents implement specific capabilities through the `Agent` trait
- Agents can be dynamically selected and executed at runtime
- Agents can depend on multiple tools for functionality

### Tool System
- Tools provide specific functionality through the `Tool` trait
- Tools can be called with strongly-typed arguments
- Tools support both direct and JSON-based execution

### Context System
- Provides the execution environment for agents and tools
- Modular design allows for flexible feature composition
- Includes security features like cryptographic operations

### Knowledge Management
- Supports semantic search and document storage
- Allows adding and retrieving knowledge documents
- Provides both similarity-based and time-based retrieval

## Security Features
- Cryptographic key derivation and management
- Message signing and verification
- Secure storage operations
- Signed HTTP requests
- Caller verification

## License
Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `anda` by you, shall be licensed as MIT, without any
additional terms or conditions.

[docs]: https://docs.rs/anda_core
[license]: ./../LICENSE-MIT