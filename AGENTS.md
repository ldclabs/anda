# AGENTS.md

Guidance for AI coding agents working in this repository.

## Scope

This file applies to the whole `ldclabs/anda` workspace. Follow more specific
instructions in nested `AGENTS.md` files if any are added later.

## Repository Shape

`anda` is a Rust workspace for composable AI agent runtimes.

- `anda_core`: stable traits, shared data models, context capabilities, JSON schema helpers, and RPC helpers.
- `anda_engine`: runtime orchestration, contexts, model adapters, tools, hooks, memory, subagents, and reusable extensions.
- `anda_engine_server`: HTTP server layer for exposing engines.
- `anda_cli`: CLI entrypoint for configured engine servers.
- `anda_web3_client`: Web3 integration for non-TEE environments.
- `docs/architecture.md`: source-level runtime architecture.
- `MCP_INTEGRATION.md`: MCP host/client design and boundaries.

Keep `anda_core` small and dependency-light. Runtime integrations belong in
`anda_engine` or higher-level crates unless the shared contract genuinely needs
to change.

## Working Rules

- Inspect the exact code and docs relevant to the request before editing.
- Preserve user or agent changes already present in the working tree. Do not
  revert unrelated dirty files.
- Prefer small, behavior-focused changes over broad refactors.
- Keep public API changes coherent across code, docs, examples, and tests.
- Use workspace dependencies from the root `Cargo.toml`; do not add duplicate
  per-crate versions unless there is a clear reason.
- Keep generated schemas and protocol-visible names stable. Agent, tool, and
  provider names must satisfy the function-name rules: lowercase ASCII letters,
  digits, underscores, and hyphens, starting with a lowercase letter, max 64
  bytes.
- When changing model/provider conversion code, preserve raw-history and
  tool/user message boundaries. Do not merge tool outputs into user messages or
  lose provider-specific content unless the existing code explicitly does so.

## Runtime Boundaries

- `Tool` and `ToolSet` are for static local tools.
- `ToolProvider` and `ToolProviderSet` are for runtime-discovered tools, such as
  MCP servers.
- `AgentCtx` is the scheduling surface for local tools, dynamic providers,
  local agents, subagents, and remote engines.
- `BaseCtx` is the capability surface for tools and agents. Use scoped child
  contexts so cache, store, metadata, and cancellation behavior remain isolated.
- `tools_search` and `tools_select` are normal agents, not side channels. If a
  change affects discovery, verify schema selection, request tool merging, and
  repeated-selection compaction.
- Remote engine prefixes and subagent prefixes are part of the runtime contract.
  Do not rename or reinterpret them without updating all call and discovery
  paths.

## MCP Rules

MCP support is implemented in `anda_engine::extension::mcp` and should remain a
generic engine capability. Product-level configuration, secret expansion,
approval UX, and launcher behavior belong in downstream applications such as
`anda-bot`.

- Use `rmcp` for MCP client behavior. Do not hand-roll JSON-RPC transports.
- Support MCP tools through `tools/list` and `tools/call`.
- Keep stdio transport as `command + args`; do not construct shell strings.
- Keep Streamable HTTP headers validated before connecting.
- Treat remote MCP descriptions, annotations, and output as untrusted metadata.
- Preserve the local-name to remote-name route table so original MCP tool names
  are used for calls while Anda exposes legal local function names.
- Do not integrate SEP-2577-deprecated capabilities: Roots, Sampling, or Logging
  control. In particular, do not advertise Roots, do not create Sampling
  messages, and do not call `logging/setLevel`.

## Security And Safety

- Engines are private by default. Be careful when changing `Management`,
  exported agents/tools, or direct `agent_run` / `tool_call` validation.
- Filesystem tools must remain workspace-scoped and must reject unsafe symlink,
  parent-directory, non-regular-file, or hardlink escapes.
- Shell tools must keep environment forwarding explicit and must not leak secret
  values in definitions, logs, or tool outputs.
- HTTP/model clients should preserve existing timeout, retry, body-size, and
  content-decoding behavior unless the task is specifically to change it.
- Never expose full conversation history to tools or remote systems unless the
  existing public contract requires it.

## Development Commands

Use the narrowest command that proves the change, then broaden when the touched
surface is shared.

```sh
cargo fmt
cargo test -p anda_core -p anda_engine
cargo clippy -p anda_core -p anda_engine --all-targets --all-features -- -D warnings
```

For crate-specific work:

```sh
cargo test -p anda_core
cargo test -p anda_engine
cargo test -p anda_engine_server
cargo test -p anda_cli
cargo test -p anda_web3_client
```

For public API or documentation-sensitive changes:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

If a command fails because dependencies or network access are unavailable, report
that clearly and keep the successful local checks in the final summary.

## Testing Expectations

- Add focused unit tests for new contracts, edge cases, and regressions.
- Add integration tests when behavior crosses crate, HTTP, transport, or runtime
  boundaries.
- For discovery or completion-runner changes, test both happy path and missing
  or malformed tool/agent paths.
- For MCP changes, test name mapping, include/exclude filters, schema conversion,
  call result conversion, and dynamic provider registration.
- For filesystem or shell changes, include negative tests for path escapes,
  encoding, symlinks, cancellation, and output limits where relevant.

## Documentation Expectations

Update docs when behavior changes user-facing or integrator-facing surfaces:

- Root `README.md` for high-level capability changes.
- Translated READMEs when the English README changes materially.
- Crate READMEs for crate-local public API changes.
- `docs/architecture.md` for runtime architecture changes.
- `MCP_INTEGRATION.md` for MCP host/client design changes.
- Rustdoc comments for new public types, methods, traits, and fields.

Keep examples copyable. Compile or test examples when they include API calls or
protocol bytes.

## Code Style

- Rust edition is 2024.
- Prefer typed APIs and serde/schemars helpers over ad hoc JSON or string
  manipulation.
- Keep `lib.rs` exports intentional and small.
- Avoid new abstractions unless they reduce real duplication or match existing
  registry/context patterns.
- Keep comments concise and explain non-obvious decisions, not line-by-line
  mechanics.
- Use `Arc` for shared runtime components following existing patterns.
- Prefer `BoxError` and existing error style unless introducing a local error
  enum materially improves call-site clarity.

## Before Finishing

Check the final diff for unrelated churn. In the final response, summarize:

- Files or modules changed.
- Behavioral impact.
- Tests and checks run.
- Any checks not run and why.
