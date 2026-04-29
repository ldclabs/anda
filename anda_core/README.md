# anda_core

![License](https://img.shields.io/crates/l/anda_core.svg)
[![Crates.io](https://img.shields.io/crates/d/anda_core.svg)](https://crates.io/crates/anda_core)
[![Test](https://github.com/ldclabs/anda/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda/actions/workflows/test.yml)
[![Docs.rs](https://docs.rs/anda_core/badge.svg)](https://docs.rs/anda_core)
[![Latest Version](https://img.shields.io/crates/v/anda_core.svg)](https://crates.io/crates/anda_core)

`anda_core` defines the shared traits, data models, and protocol helpers used by the Anda agent framework. It is the stable contract between runtimes, agents, tools, model adapters, and clients.

This crate is intentionally small in scope: it does not run an engine, call model providers directly, or ship built-in tools. Runtime orchestration and provider integrations live in higher-level crates such as [`anda_engine`][anda-engine].

Full API documentation is available on [docs.rs][docs].

## Install

```toml
[dependencies]
anda_core = "0.11"
```

## What It Provides

- Strongly typed [`Agent`][agent-trait] and [`Tool`][tool-trait] traits.
- Dynamic registries through [`AgentSet`][agent-set] and [`ToolSet`][tool-set].
- Execution context capability traits for state, keys, storage, cache, HTTP, and canister calls.
- Provider-neutral model types such as [`Message`][message], [`ContentPart`][content-part], [`CompletionRequest`][completion-request], and [`AgentOutput`][agent-output].
- JSON schema helpers for LLM function calling.
- CBOR/Candid HTTP RPC helpers for remote engines and ICP canisters.

## Module Map

| Module                      | Purpose                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------- |
| [`agent`][agent-module]     | Defines agents, dynamic dispatch, dependency declarations, and agent registries.      |
| [`tool`][tool-module]       | Defines typed tools, raw JSON dispatch, supported resource tags, and tool registries. |
| [`context`][context-module] | Defines runtime capabilities exposed to agents and tools.                             |
| [`model`][model-module]     | Defines request/response, message, content, function-call, document, and usage types. |
| [`http`][http-module]       | Provides CBOR/Candid RPC request and response helpers.                                |
| [`json`][json-module]       | Generates compact JSON Schema values for tool and agent parameters.                   |

## Core Concepts

### Agents

An agent implements [`Agent<C>`][agent-trait] for a runtime context `C`. It declares a name, description, function definition, optional tool dependencies, optional supported resource tags, and an async `run` method.

Agent names are registered case-insensitively and must follow the same function-name rules used for LLM function calling: lowercase ASCII letters, digits, and underscores, starting with a lowercase letter.

### Tools

A tool implements [`Tool<C>`][tool-trait] with typed `Args` and `Output` associated types. The runtime accepts raw JSON tool calls, deserializes them into `Args`, executes the tool, then serializes the output back to JSON.

Tools can declare supported resource tags. During a call, the runtime removes matching resources from the request resource list and passes them to the tool.

### Contexts

[`BaseContext`][base-context] is the capability surface available to tools and agents. It combines:

- [`StateFeatures`][state-features] for engine identity, caller identity, metadata, cancellation, and elapsed time.
- [`KeysFeatures`][keys-features] for AES, Ed25519, and Secp256k1 key derivation, signing, verification, and public keys.
- [`StoreFeatures`][store-features] for isolated object storage.
- [`CacheFeatures`][cache-features] for isolated in-memory cache values with optional TTL/TTI expiration.
- [`HttpFeatures`][http-features] for runtime-managed HTTPS calls.
- `CanisterCaller` for ICP canister calls.

[`AgentContext`][agent-context] extends `BaseContext` with LLM completion and orchestration methods for local or remote agents and tools.

### Messages and Content

[`Message`][message] and [`ContentPart`][content-part] provide a normalized representation for text, reasoning, files, inline binary data, tool calls, tool outputs, signed actions, and provider-specific JSON payloads. Model adapters can preserve unknown provider content in `ContentPart::Any` while still exposing common behavior to the rest of the framework.

## Minimal Tool Example

```rust
use anda_core::{
    BaseContext, BoxError, FunctionDefinition, Resource, Tool, ToolOutput, gen_schema_for,
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoArgs {
    text: String,
}

#[derive(Default)]
struct EchoTool;

impl<C> Tool<C> for EchoTool
where
    C: BaseContext + Send + Sync,
{
    type Args = EchoArgs;
    type Output = String;

    fn name(&self) -> String {
        "echo".to_string()
    }

    fn description(&self) -> String {
        "Returns the provided text.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: gen_schema_for::<EchoArgs>(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        _ctx: C,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        Ok(ToolOutput::new(args.text))
    }
}
```

## Validation

Useful checks while working on this crate:

```sh
cargo test -p anda_core
cargo clippy -p anda_core --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p anda_core --no-deps
```

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `anda` by you shall be licensed as MIT, without any additional
terms or conditions.

[docs]: https://docs.rs/anda_core
[anda-engine]: https://docs.rs/anda_engine
[agent-module]: https://docs.rs/anda_core/latest/anda_core/agent/index.html
[tool-module]: https://docs.rs/anda_core/latest/anda_core/tool/index.html
[context-module]: https://docs.rs/anda_core/latest/anda_core/context/index.html
[model-module]: https://docs.rs/anda_core/latest/anda_core/model/index.html
[http-module]: https://docs.rs/anda_core/latest/anda_core/http/index.html
[json-module]: https://docs.rs/anda_core/latest/anda_core/json/index.html
[agent-trait]: https://docs.rs/anda_core/latest/anda_core/trait.Agent.html
[tool-trait]: https://docs.rs/anda_core/latest/anda_core/trait.Tool.html
[agent-set]: https://docs.rs/anda_core/latest/anda_core/struct.AgentSet.html
[tool-set]: https://docs.rs/anda_core/latest/anda_core/struct.ToolSet.html
[base-context]: https://docs.rs/anda_core/latest/anda_core/trait.BaseContext.html
[agent-context]: https://docs.rs/anda_core/latest/anda_core/trait.AgentContext.html
[state-features]: https://docs.rs/anda_core/latest/anda_core/trait.StateFeatures.html
[keys-features]: https://docs.rs/anda_core/latest/anda_core/trait.KeysFeatures.html
[store-features]: https://docs.rs/anda_core/latest/anda_core/trait.StoreFeatures.html
[cache-features]: https://docs.rs/anda_core/latest/anda_core/trait.CacheFeatures.html
[http-features]: https://docs.rs/anda_core/latest/anda_core/trait.HttpFeatures.html
[message]: https://docs.rs/anda_core/latest/anda_core/struct.Message.html
[content-part]: https://docs.rs/anda_core/latest/anda_core/enum.ContentPart.html
[completion-request]: https://docs.rs/anda_core/latest/anda_core/struct.CompletionRequest.html
[agent-output]: https://docs.rs/anda_core/latest/anda_core/struct.AgentOutput.html
[license]: ./../LICENSE-MIT