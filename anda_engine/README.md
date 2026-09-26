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
- Label-based model routing with a primary model.
- Built-in model adapters for OpenAI-compatible APIs, Anthropic, and Gemini.
- Object storage backed by the `object_store` ecosystem.
- Persistent memory tools built on AndaDB, Cognitive Nexus, and KIP.
- Remote engine discovery and cross-engine tool or agent calls.
- Hook APIs for observing and transforming agent and tool execution.
- Workspace tools for filesystem access, shell execution, web fetch, notes, skills, todos, and search.
- MCP client support exposing remote MCP servers as runtime-discovered tools.
- Web3 and TEE challenge signing through the Anda Web3 stack.

## Installation

```sh
cargo add anda_engine
```

The crate has no default optional features.

```toml
[dependencies]
anda_engine = "0.16"
```

## Quick Start

The example below builds an explicitly public demo engine with the built-in `EchoEngineInfo` agent. Engines remain private by default. The same example is runnable with `cargo run -p anda_engine --example quick_start`. Real applications usually register their own `anda_core::Agent` and `anda_core::Tool` implementations.

```rust
use anda_core::AgentInput;
use anda_engine::{
    ANONYMOUS,
    engine::{AgentInfo, EchoEngineInfo, Engine},
    management::{BaseManagement, Visibility},
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

    // This demo explicitly permits anonymous callers; engines default to private.
    let engine = Engine::builder()
        .with_management(Arc::new(BaseManagement {
            controller: ANONYMOUS,
            managers: Default::default(),
            visibility: Visibility::Public,
        }))
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

Subagent sessions are scoped by caller principal and session ID. Use
`SubSessions::get_session_for`, `active_session_ids_for`, and `session_details_for`
for caller-facing controls and status. The caller-free lookup is intended for
host administration and returns `None` for an ambiguous ID. Shared callable
schemas do not include other callers' active session IDs. Background progress
uses the same `agent:session` task ID as start and end callbacks.

`CompletionStream` preserves follow-up and steering input submitted while a
model step is pending. Failed context compaction leaves the original runner
usable. Stop/cancel controls interrupt pending subagent work; native shell
cancellation cleans up the process tree and output readers.

Agent dependencies supplied by dynamic providers are verified after provider
initialization during engine build. Exact `SA_`, `RA_`, and `RT_` callable names
can be used when requesting definitions.


### Models

`Models` is a thread-safe model registry. You can register concrete `Model` values under labels such as `primary`, `pro`, `flash`, or `lite`. Agents can route requests by label while applications remain free to change provider-specific model names.

Built-in provider adapters include:

- `openai`: Responses API for model names starting with `gpt`, otherwise Chat Completions.
- `openai-response`: Explicit Responses API selection for any model name.
- `anthropic`: Anthropic Messages API completion.
- `gemini`: Google Gemini completion.

Custom providers can implement `CompletionFeaturesDyn` and be wrapped with `Model::with_completer`.

Completion adapter behavior:

- Chat Completions sends `max_completion_tokens` for an explicit output budget.
  A default request template containing only `max_tokens` opts into the legacy
  field for compatible endpoints; DeepSeek model names also use that field.
- Responses always uses streaming transport and `store: false`, then returns
  the aggregated `AgentOutput`. Its `with_stream` setting is retained for source
  compatibility but has no effect. Non-empty stop sequences return an error.
- Incomplete streams return retryable errors. Chat accepts an explicit finish
  reason or `[DONE]` (for compatible providers omitting the reason); Gemini
  requires a finish reason or prompt blocking; Anthropic requires both a stop
  reason and `message_stop`.
- Anthropic maps minimal/low effort to `low`, and medium/high/max to their
  corresponding API values. Gemini 2.5 maps minimal/low/medium/high/max to
  budgets of 0/1024/4096/16384/24576 tokens; Pro uses 128 for minimal and 32768
  for max. Gemini 3 uses thinking levels, mapping minimal to low on Pro and
  medium to high on the original Gemini 3 Pro. Use a default request template
  for other provider-specific settings. `ThinkingConfig::thinking_budget` is
  signed so `-1` can request dynamic thinking.
- Anthropic forwards `FunctionDefinition::strict` and closes objects in strict
  tool and output schemas while preserving optional properties. Common
  unsupported constraints (including numeric bounds and string lengths) return
  a local error rather than being discarded. Use supported schemas, or
  `strict: false` for tools that need schemas outside that subset.
- Chat audio inputs must contain inline WAV/MP3 data. Remote audio/file URLs
  and video input return errors; use Responses for remote file inputs.
  Anthropic inline PDFs use Base64 regardless of whether their bytes are UTF-8.
- OpenAI and Anthropic pair missing tool-call IDs when replaying neutral
  history, including repeated calls to the same function. Existing IDs and
  provider-native `raw_history` remain intact.

### Tools and Extensions

The `extension` module provides reusable tools for common agent capabilities:

- `fetch`: signed HTTP fetching and resource loading.
- `fs`: workspace-scoped file read, write, search, and edit tools.
- `shell`: native or sandboxed shell command execution.
- `mcp`: MCP servers as runtime-discovered tool providers.
- `note`: lightweight per-agent note storage.
- `skill`: file-backed skill loading and lifecycle management.
- `todo`: session-scoped task tracking.

Filesystem tools enforce configured workspace roots. A native shell's working directory alone does not confine the process: use an isolated host or opt into `NativeRuntime::with_sandbox`. Shell commands receive a restricted environment; only allowlisted host variables and explicitly configured keys are forwarded. Configured environment keys are normalized case-insensitively on Windows.

Use `extension::workspace::coding_tools` for a coding agent: it registers `shell`, `shell_session`, and `apply_patch`, with no redundant read/search tools. `readonly_file_tools` registers only `read_file` and `search_file`; `file_tools` preserves the four filesystem tools for hosts without a shell. Existing tools remain available individually.

The coding bundle uses `ShellCommandTool` in place of the legacy `ShellTool` (both have the name `shell`; do not register both). Its `CommandArgs` supports a scoped `cwd`, a wait duration separate from the process deadline, and a combined stdout/stderr preview budget. Optional stdin and Unix PTYs require host opt-in through `SessionLimits`. Shell selection is host-controlled through `NativeShell`; login profiles are not enabled. Legacy `ExecArgs`, `ExecOutput`, custom `Executor` implementations, and `ShellTool` remain source-compatible. The new session methods on `Executor` default to unsupported.

Before dispatching session tools, the host must install a `ShellSessionScope::new()` in parent context state. Reuse that capability only for the same conversation on subsequent turns. Sessions are also checked against the engine, verified caller, and agent; request metadata is not an authorization source. A session retains its launch runtime policy; changing runtime sandbox, shell, environment inheritance, log directory or session limits invalidates and cancels old sessions. No approval UI or persistent permission grants are implemented by the tool.

Session execution defaults to a 10-second foreground wait, a 10-minute total runtime limit, a 32KiB combined output preview (minimum request 128 bytes), 1MiB capture per stream, and a 32MiB combined log quota. Host limits cap caller requests. Output preserves the beginning and tail; `omitted_bytes` distinguishes a non-complete preview, while `log_complete` separately describes raw-log completeness. The log interleaves raw chunks with stdout/stderr labels. `shell_session` provides poll, write, close_stdin, stop, list, and bounded log reads by byte offset. PTYs combine both streams, and close_stdin is only supported for pipes. A write timeout or cancellation can leave a partially sent input; callers must not blindly retry it.

The default registry retains at most 64 sessions and refuses new launches when all slots are active. Completed sessions expire after five minutes or are evicted to admit a new command; expiration cleanup is lazy on the next session operation. Logs are deleted when their session is released. Dropping the runtime cancels retained sessions. Request cancellation also propagates to its processes. On Unix, when the launched shell exits, the session terminates descendants still in its process group. Use `background: true` to keep a supervised command running instead of appending `&`. Legacy shell raw-output files retain their previous caller-managed cleanup behavior, and legacy background commands retain their existing lifecycle; use the session bundle when total runtime limits and polling are required.

`apply_patch` accepts JSON containing a `patch` string, optional `dry_run`, and optional `expected_versions` (path and original SHA-256, or `missing`). It supports `*** Add File`, `*** Delete File`, `*** Update File`, `*** Move to`, `@@` chunks, and `*** End of File`. Exact context must identify one location; ambiguous matches and overwriting add/move targets are rejected. The patch is bounded to 1MiB input, 32 source files, 256 chunks per file, 200,000 lines per updated file, a 64MiB prepared byte budget, and a 32KiB diff preview. Existing encoding and LF/CRLF line endings are preserved. All files are prevalidated and locked before the first write; writes are atomic per file, not transactional across files. Commit failures return `is_error`, per-operation `applied` flags, and an explanation; a failed move can leave a written destination. SHA checks detect stale edits but are not an OS compare-and-swap against external writers.

The shared filesystem layer serializes cooperating writes across tool instances, enforces a 10MiB write/read/edit limit, and reads through validated file handles with a bound on actual bytes. Unix traversal and replacement use directory descriptors and no-follow operations. Windows checks file handles for reparse points and hardlink counts and pins parent directories during access. Locks do not serialize arbitrary shell scripts or external editors. Metadata workspace hints only prioritize permitted roots; they do not revoke access to the other configured roots.

`SandboxPolicy::workspace(root)` is an optional, host-owned process policy with explicit read/write roots and network denied by default. macOS uses `/usr/bin/sandbox-exec`; Linux requires `/usr/bin/bwrap` and user-namespace support. Common OS runtime paths are readable. Linux provides a private `/tmp`; on macOS, explicitly grant a dedicated scratch directory and configure `TMPDIR` when needed. Paths outside grants are unavailable, so build caches and SDKs may need explicit host grants. Missing or failing backends never trigger an unrestricted retry. Windows currently requires a custom isolated `Executor`; the built-in sandbox constructor fails there. Sandbox enforcement does not add application approval workflows or destination-level network filtering.

A compilable registration and polling example is available in [`examples/workspace_tools.rs`](examples/workspace_tools.rs):

```sh
cargo run -p anda_engine --example workspace_tools
```

### Memory

The `memory` module stores conversations, resources, artifacts, usage, steering messages, and follow-up messages. It uses AndaDB collections and exposes KIP-backed tools for persistent agent memory through the Cognitive Nexus.

### Remote Engines

Engines can register other engines by endpoint. Remote metadata is fetched through signed RPC, and exported remote functions are exposed with prefixed names:

- Tools: `RT_{handle}_{tool}`
- Agents: `RA_{handle}_{agent}`

This lets agents discover and call capabilities hosted by other engines without linking them into the same process.

### Hooks

Engine-level hooks can observe or transform agent and tool execution. Typed hooks can be attached through context state for specific extensions, including background task lifecycle events.

`SingleThreadHook` is included for applications that want to limit each caller to one active prompt at a time.

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

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `anda` by you, shall be licensed as MIT, without any
additional terms or conditions.

[docs]: https://docs.rs/anda_engine
[license]: ./../LICENSE-MIT
