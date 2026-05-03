# anda_engine

![License](https://img.shields.io/crates/l/anda_engine.svg)
[![Crates.io](https://img.shields.io/crates/d/anda_engine.svg)](https://crates.io/crates/anda_engine)
[![Test](https://github.com/ldclabs/anda/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda/actions/workflows/test.yml)
[![Docs.rs](https://docs.rs/anda_engine/badge.svg)](https://docs.rs/anda_engine)
[![Latest Version](https://img.shields.io/crates/v/anda_engine.svg)](https://crates.io/crates/anda_engine)

Runtime engine for [Anda](https://github.com/ldclabs/anda), a Rust framework for building autonomous AI agents powered by ICP identities and Trusted Execution Environments (TEEs).

`anda_engine` implements the runtime behind the traits and data contracts in [`anda_core`](https://github.com/ldclabs/anda/tree/main/anda_core). It wires together agent execution, tool dispatch, model providers, persistent storage, hooks, remote engines, and built-in extensions.

Full API documentation is available on [docs.rs][docs].

## What It Provides

`anda_engine` is designed as the embeddable runtime layer for applications that host Anda agents.

- Agent and tool registration with scoped execution contexts.
- Direct agent runs and direct tool calls with cancellation support.
- Label-based model routing with primary and fallback models.
- Built-in model adapters for OpenAI-compatible APIs, Anthropic, and Gemini.
- Object storage backed by the `object_store` ecosystem.
- Persistent memory tools built on AndaDB, Cognitive Nexus, and KIP.
- Remote engine discovery and cross-engine tool or agent calls.
- Hook APIs for observing and transforming agent and tool execution.
- Workspace tools for filesystem access, shell execution, web fetch, extraction, notes, skills, todos, and search.
- Web3 and TEE challenge signing through the Anda Web3 stack.

## Installation

```sh
cargo add anda_engine
```

The crate has no default optional features.

```toml
[dependencies]
anda_engine = "0.11"
```

Enable sandboxed shell execution when you need isolated command execution through Boxlite:

```toml
[dependencies]
anda_engine = { version = "0.11", features = ["sandbox"] }
```

## Quick Start

The example below builds an engine with the built-in `EchoEngineInfo` agent. Real applications usually register their own `anda_core::Agent` and `anda_core::Tool` implementations.

```rust,no_run
use anda_core::AgentInput;
use anda_engine::{
    ANONYMOUS,
    engine::{AgentInfo, EchoEngineInfo, Engine},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let echo_info = AgentInfo {
        handle: "echo".to_string(),
        name: "Echo Agent".to_string(),
        description: "Returns engine metadata as JSON.".to_string(),
        ..Default::default()
    };

    let engine = Engine::builder()
        .register_agent(Arc::new(EchoEngineInfo::new(echo_info)), None)?
        .build("echo".to_string())
        .await?;

    let output = engine
        .agent_run(
            ANONYMOUS,
            AgentInput::new("echo".to_string(), "hello".to_string()),
        )
        .await?;

    println!("{}", output.content);
    Ok(())
}
```

## Core Concepts

### Engine

`Engine` is the top-level runtime. It owns registered agents, tools, models, hooks, storage, management policy, Web3 or TEE identity, and remote engine metadata.

Use `EngineBuilder` to configure an engine, then call `build(default_agent)` to initialize tools and agents. The selected default agent is automatically exported.

### Contexts

Agents receive `AgentCtx`; tools receive `BaseCtx`. Contexts carry caller identity, request metadata, cancellation tokens, scoped cache and storage, shared state, HTTP and canister features, Web3 signing, and remote engine access.

Context namespaces are derived from agent and tool names so cache and object storage remain isolated between components.

### Models

`Models` is a thread-safe model registry. You can register concrete `Model` values under labels such as `primary`, `fallback`, `pro`, `flash`, or `lite`. Agents can route requests by label while applications remain free to change provider-specific model names.

Built-in provider adapters include:

- `openai`: OpenAI-compatible Responses API style completion.
- `anthropic`: Anthropic Messages API completion.
- `gemini`: Google Gemini completion.

Custom providers can implement `CompletionFeaturesDyn` and be wrapped with `Model::with_completer`.

### Tools and Extensions

The `extension` module provides reusable tools for common agent capabilities:

- `fetch`: signed HTTP fetching and resource loading.
- `fs`: workspace-scoped file read, write, search, and edit tools.
- `shell`: native or sandboxed shell command execution.
- `extractor`: structured data extraction.
- `note`: lightweight per-agent note storage.
- `skill`: file-backed skill loading and lifecycle management.
- `todo`: session-scoped task tracking.
- `google`: Google web search integration.

Filesystem and shell tools are intentionally workspace-scoped. Shell commands receive a restricted environment; only allowlisted host variables and explicitly configured keys are forwarded.

### Memory

The `memory` module stores conversations, resources, artifacts, usage, steering messages, and follow-up messages. It uses AndaDB collections and exposes KIP-backed tools for persistent agent memory through the Cognitive Nexus.

### Remote Engines

Engines can register other engines by endpoint. Remote metadata is fetched through signed RPC, and exported remote functions are exposed with prefixed names:

- Tools: `{handle}_{tool}`
- Agents: `{handle}_{agent}`

This lets agents discover and call capabilities hosted by other engines without linking them into the same process.

### Hooks

Engine-level hooks can observe or transform agent and tool execution. Typed hooks can be attached through context state for specific extensions, including background task lifecycle events.

`SingleThreadHook` is included for applications that want to limit each caller to one active prompt at a time.

## Feature Flags

| Feature   | Description                                                             |
| --------- | ----------------------------------------------------------------------- |
| `sandbox` | Enables the Boxlite-backed sandbox runtime for `extension::shell`.      |
| `full`    | Enables all optional runtime features currently provided by this crate. |

## Security Notes

- Engines are private by default. Configure `Management` when exposing an engine to external callers.
- Direct agent and tool calls validate request metadata and engine identity.
- Only exported agents and tools appear in `Engine::information` and are available to non-manager callers.
- Filesystem tools resolve paths under the configured workspace and reject unsafe writes through symlinks or multiply linked files.
- Shell output is truncated in responses when it exceeds the inline limit; full output can be written to a temporary file.

## Related Crates

- [`anda_core`](https://github.com/ldclabs/anda/tree/main/anda_core): core traits, request/response types, messages, resources, and tool schemas.
- [`anda_engine_server`](https://github.com/ldclabs/anda/tree/main/anda_engine_server): HTTP server for exposing one or more engines.
- [`anda_web3_client`](https://github.com/ldclabs/anda/tree/main/anda_web3_client): Web3 integration for non-TEE environments.

## Development

Useful checks while working on this crate:

```sh
cargo check -p anda_engine
cargo test -p anda_engine --lib
cargo clippy -p anda_engine --all-targets -- -D warnings
```

Optional sandbox checks:

```sh
cargo check -p anda_engine --features sandbox
cargo test -p anda_engine --lib --features sandbox
cargo clippy -p anda_engine --all-targets --features sandbox -- -D warnings
```

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `anda` by you, shall be licensed as MIT, without any
additional terms or conditions.

[docs]: https://docs.rs/anda_engine
[license]: ./../LICENSE-MIT