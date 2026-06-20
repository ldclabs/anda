# `Anda`

> A Rust framework for building composable AI agent runtimes.

## README Translations

[English readme](./README.md) | [中文说明](./README_CN.md) | [日本語の説明](./README_JA.md)

## Introduction

Anda is a Rust framework for building AI agents that can combine models, tools, memory, and other agents into a single runtime. It focuses on composability, type-safe extension points, asynchronous execution, and practical runtime control.

The core engine lets developers register agents and tools, route model requests by capability labels, call local or remote functions, isolate context state, and add optional persistence or memory layers when an application needs them.

![Anda Diagram](./anda_diagram.webp)

## Key Features

1. **Composable agents and tools**
   Agents and tools are registered through stable traits and function definitions, so specialized components can be combined into larger workflows without hard-coding one application shape.

2. **Model routing**
   The engine can route completion requests through labeled model tiers such as `primary`, `pro`, `flash`, or `lite`, while provider-specific adapters stay behind a common request and output contract.

3. **Runtime orchestration**
   `CompletionRunner` handles iterative model turns, tool calls, agent calls, usage accounting, artifacts, steering messages, follow-up messages, cancellation, and compact continuation handoffs for long-running sessions.

4. **Scoped execution context**
   `BaseCtx` and `AgentCtx` provide isolated state, cache, object storage, HTTP calls, signed calls, cancellation, and child contexts for each agent or tool.

5. **Extensible memory and skills**
   Optional extensions provide conversation storage, KIP-based memory tools, filesystem access, shell execution, fetch, notes, todos, and file-backed skills.

6. **Discovery-aware tool bundles**
   Static tools, dynamic providers, and MCP servers can expose capability groups so agents can survey related tool bundles with `tools_groups`, then expand a group with `tools_select` only when its schemas are needed.

## Project

Documents:
- [Anda Architecture](./docs/architecture.md)

### Project Structure

```sh
anda/
├── anda_cli/              # Command-line interface for Anda engine servers
├── anda_core/             # Core traits, types, and runtime contracts
├── anda_engine/           # Agent runtime, orchestration, contexts, models, and extensions
└── anda_engine_server/    # HTTP server for serving one or more Anda engines
```

### How to Use and Contribute

#### For application builders

Use `anda_cli` and `anda_engine_server` to run and interact with configured engines.

#### For developers

- Build custom agents and tools with the `anda_core` traits.
- Extend `anda_engine` with reusable runtime features.
- Improve model adapters, context capabilities, memory integrations, and server APIs.

### Products Built on Anda

- [Anda Brain](https://github.com/ldclabs/anda-brain): Persistent memory and cognition product built on the Anda framework.
- [Anda Bot](https://github.com/ldclabs/anda-bot): Personal AI assistant and application runtime built on the Anda framework.

### Related Projects

- [KIP](https://github.com/ldclabs/KIP): Knowledge Interaction Protocol used by Anda memory tools.

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See [LICENSE](./LICENSE-MIT) for the full license text.
