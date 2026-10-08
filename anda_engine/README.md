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
- MCP client support with versioned catalogs, stable routes, input/output budgets, configurable concurrency/deadlines, OAuth refresh coordination, and opt-in elicitation/resources.
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

Subagent sessions are isolated by caller, host-created `SubAgentScope`, worker,
and alias. Install the same scope clone to continue a root task across engine
entry contexts. Use `get_session_in_scope` and `session_details_in_scope` for
model-facing access; legacy caller-only lookups are host administration helpers.
Turn-completion hooks and bounded event waits are independent of session closure.
Typed messages distinguish queued notifications from tasks that wake idle workers.
Shared limits govern residency, concurrent inference, queues and optional budgets.
Idle checkpoints and explicit provider-neutral handoffs are opt-in host capabilities.
See [subagent lifecycle and migration](../docs/subagents.md) and the runnable
[subagent_sessions example](examples/subagent_sessions.rs).

`CompletionStream` preserves follow-up and steering input submitted while a
model step is pending, and ends after it yields an error. Context compaction
moves pending and queued input to the replacement runner; a failed compaction
leaves the original runner usable. Stop/cancel controls interrupt pending
subagent work; native shell cancellation cleans up the process tree and output
readers.

A long-lived runner can send attachment bytes for one task only:
`set_transient_inline_data(true)` keeps `InlineData` (and `data:` `FileData`)
out of the neutral `chat_history`, and the provider raw history replaces them
with a short note once the runner goes idle or its task is stopped. Send a
reference alongside the bytes so the attachment can be found later.
`add_tools` offers more tool definitions mid-session without forgetting the
tools the model discovered, unlike `set_tools`.

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
- Anthropic maps `tool_choice_required` to `tool_choice: any`. Claude Opus 5.5,
  Claude Sonnet 5.5, and Claude Fable 5.1 reject forced tool choice with a 400,
  so leave it unset for those models and name the tool in the prompt instead.
- Anthropic forwards `FunctionDefinition::strict` and closes objects in strict
  tool and output schemas while preserving optional properties. A strict tool
  whose schema uses unsupported constraints (including numeric bounds and
  string lengths, such as `minimum: 0` on unsigned integers) is sent as a
  regular tool with its schema unchanged. In an output schema, such
  constraints return a local error rather than being discarded.
- Chat audio inputs must contain inline WAV/MP3 data. Remote audio/file URLs
  return errors; use Responses for remote file inputs. Video is sent as a
  `video_url` part for compatible providers; OpenAI itself rejects it.
  Anthropic inline PDFs use Base64 regardless of whether their bytes are UTF-8.
- OpenAI and Anthropic pair missing tool-call IDs when replaying neutral
  history, including repeated calls to the same function. Existing IDs,
  provider-native `raw_history`, and results whose calls exist only in
  `raw_history` remain intact.

### Tools and Extensions

The `extension` module provides reusable tools for common agent capabilities:

- `fetch`: signed HTTP fetching and resource loading.
- `fs`: workspace-scoped file read, write, search, and edit tools.
- `shell`: native or sandboxed shell command execution.
- `mcp`: MCP servers as runtime-discovered tool providers.
- `note`: per-agent persistent notes with bounded ID reads, listings, and substring search.
- `skill`: file-backed skills with bounded discovery, immutable catalog generations, stable identities, package resource reads, and optional delegated execution.
- `todo`: validated session tasks with incremental updates and bounded recovery after handoff.

Notes have an opt-in context index; task changes use existing typed hooks with an optional explanation. See [notes and tasks](../docs/note-todo.md) for pagination, limits, context lifecycle, and Rust API migration.

Filesystem tools enforce configured workspace roots. A native shell's working directory alone does not confine the process: use an isolated host or opt into `NativeRuntime::with_sandbox`. Shell commands receive a restricted environment; only allowlisted host variables and explicitly configured keys are forwarded. Configured environment keys are normalized case-insensitively on Windows.

Use `extension::workspace::coding_tools` for a coding agent: it registers `shell`, `shell_session`, and `apply_patch`, with no redundant read/search tools. `readonly_file_tools` registers only `read_file` and `search_file`; `file_tools` preserves the four filesystem tools for hosts without a shell. Existing tools remain available individually.

The coding bundle uses `ShellCommandTool` in place of the legacy `ShellTool` (both have the name `shell`; do not register both). Its `CommandArgs` supports a scoped `cwd`, a wait duration separate from the process deadline, and a combined stdout/stderr preview budget. Optional stdin and Unix PTYs require host opt-in through `SessionLimits`. Shell selection is host-controlled through `NativeShell`; login profiles are not enabled. Legacy `ExecArgs`, `ExecOutput`, custom `Executor` implementations, and `ShellTool` remain source-compatible. The new session methods on `Executor` default to unsupported.

Before dispatching session tools, the host must install a `ShellSessionScope::new()` in parent context state. Reuse that capability only for the same conversation on subsequent turns. Sessions are also checked against the engine, verified caller, and agent; request metadata is not an authorization source. A session retains its launch runtime policy; changing runtime sandbox, shell, environment inheritance, log directory or session limits invalidates and cancels old sessions. No approval UI or persistent permission grants are implemented by the tool. Argument hooks are typed on the session tools' own types (`ShellCommandToolHook` and `ShellSessionToolHook`). An approval gate installed as `ShellToolHook` for the legacy tool keeps gating `shell` after the switch: its `before_tool_call` sees the legacy fields (`command`, `env_keys`, `background`) just before execution and its rewrites apply, and its `after_tool_call` sees the legacy part of the result. It does not gate `shell_session`, whose `write` needs host-enabled stdin. `ShellToolHook` and `DynToolJsonHook` also receive background events.

Session execution defaults to a 10-second foreground wait (ending early only when the process exits), a 10-minute total runtime limit, a 32KiB combined output preview (minimum request 128 bytes), 1MiB capture per stream, and a 32MiB combined log quota. Host limits cap caller requests. Output preserves the beginning and tail; `omitted_bytes` distinguishes a non-complete preview, while `log_complete` separately describes raw-log completeness. The log keeps raw chunks in arrival order and labels each switch between stdout and stderr. `shell_session` provides poll, write, close_stdin, stop, list, and bounded log reads by byte offset. PTYs combine both streams, and close_stdin is only supported for pipes. A write timeout or cancellation can leave a partially sent input; callers must not blindly retry it.

The default registry retains at most 64 sessions and refuses new launches when all slots are active. Completed sessions expire after five minutes or are evicted to admit a new command; expiration cleanup is lazy on the next session operation. Logs are deleted when their session is released. Dropping the runtime cancels retained sessions, so run a call in a directory the host authorized separately on `NativeRuntime::for_workspace`, which shares the runtime's sessions and policy, rather than on a new runtime. On Windows, session processes start without a console window. Request cancellation also propagates to its processes. On Unix, when the launched shell exits, the session terminates descendants still in its process group. Use `background: true` to keep a supervised command running instead of appending `&`. Legacy shell raw-output files retain their previous caller-managed cleanup behavior, and legacy background commands retain their existing lifecycle; use the session bundle when total runtime limits and polling are required.

`apply_patch` accepts JSON containing a `patch` string, optional `dry_run`, and optional `expected_versions` (path and original SHA-256, or `missing`). It supports `*** Add File`, `*** Delete File`, `*** Update File`, `*** Move to`, `@@` chunks, and `*** End of File`. Exact context must identify one location; ambiguous matches and overwriting add/move targets are rejected. The patch is bounded to 1MiB input, 32 source files, 256 chunks per file, 200,000 lines per updated file, a 64MiB prepared byte budget, and a 32KiB diff preview. Existing encoding and LF/CRLF line endings are preserved. All files are prevalidated and locked before the first write; writes are atomic per file, not transactional across files. Commit failures return `is_error`, per-operation `applied` flags, and an explanation; a failed move can leave a written destination. SHA checks detect stale edits but are not an OS compare-and-swap against external writers.

The shared filesystem layer serializes cooperating writes across tool instances, enforces a 10MiB write/read/edit limit, and reads through validated file handles with a bound on actual bytes. Unix traversal and replacement use directory descriptors and no-follow operations. Windows checks file handles for reparse points and hardlink counts and pins parent directories during access. Locks do not serialize arbitrary shell scripts or external editors. Metadata workspace hints only prioritize permitted roots; they do not revoke access to the other configured roots.

`SandboxPolicy::workspace(root)` is an optional, host-owned process policy with explicit read/write roots and network denied by default. macOS uses `/usr/bin/sandbox-exec`; Linux requires `/usr/bin/bwrap` and user-namespace support. Common OS runtime paths are readable. Linux provides a private `/tmp`; on macOS, explicitly grant a dedicated scratch directory and configure `TMPDIR` when needed. Paths outside grants are unavailable, so build caches and SDKs may need explicit host grants. Missing or failing backends never trigger an unrestricted retry. Windows currently requires a custom isolated `Executor`; the built-in sandbox constructor fails there. Sandbox enforcement does not add application approval workflows or destination-level network filtering.

A compilable registration and polling example is available in [`examples/workspace_tools.rs`](examples/workspace_tools.rs):

```sh
cargo run -p anda_engine --example workspace_tools
```

### Skill catalogs and resources

`SkillManager` keeps `skills_manager({"name":"my-skill"})` for complete, small
`SKILL.md` reads. Register `manager.tools()` to include `skills_list` and
`skills_read` as a normal `skills` capability group. See
[`examples/skills.rs`](examples/skills.rs) for a runnable registration example.
The manager does not grant filesystem writes, shell execution, or install dependencies.
To expose skills declaring `execution: subagent`, additionally insert the same
`Arc<SkillManager>` into `engine.sub_agents_manager()`; inline remains the default.

- `skills_list({"query":null,"cursor":null})` returns at most 20 compact identities
  per page. Pass `next_cursor` back until it is null. Queries search names and
  descriptions; an exact name or ID also finds explicit-only skills.
- `skills_read({"skill":"my-skill","resource":null,"cursor":null})` reads
  `SKILL.md`. Use an ID from the list to distinguish duplicate names. Set
  `resource` to a package-relative text path such as `references/guide.md`.
  Read every page of an instruction document before acting on it. Changed content,
  package metadata, or resource identity invalidates a continuation; restart without
  a cursor. Reads revalidate current admission even when a cursor is supplied.
- `skills_manager` returns an error directing the model to `skills_read` when the
  complete JSON response would exceed the budget. It never returns partial instructions.

`catalog()` returns immutable host metadata, a generation and structured diagnostics.
Generation is unchanged by an identical reload. `reload()` / `load()` rescan membership;
`invalidate()` is a cheap hook for a host-owned filesystem watcher and causes the next
async list/read to refresh. Concurrent lazy refreshes coalesce. Reads revalidate the
selected `SKILL.md` and sidecar; a missing name also triggers a scan. Creating a duplicate
or changing an unrelated file requires invalidation/reload. Synchronous catalog and
callable lookups use the last published generation. Refreshing a completion request's
resident tool definitions remains the caller's responsibility; there is no background
watcher or automatic history rewrite.

Root order determines name precedence. Duplicate names inside the winning root have no
name-based reader or callable; each admitted copy remains readable by ID. Rejected copies
are filtered before resolving precedence. IDs are opaque hashes of the canonical file
location and survive frontmatter renames and root reordering; moving a file changes its ID.
Only unchanged winning identities retain live delegated sessions. A renamed/deleted skill
or a vanished root cannot leave a stale callable after reload. Scan failures omit
unverified files and report diagnostics rather than silently keeping their old callables.
Existing names retain `skill_*` callables; names longer than 58 characters use a stable
`skillh_*` hash so all valid 64-character skill names fit the function-name contract.

Defaults are: 6 descendant directory levels, 2,000 directories and 20,000 entries per root;
1,024 skill files and 32 MiB decoded skill/sidecar content across roots; 512 KiB per
`SKILL.md`, 32 KiB per sidecar, and 1 MiB per bundled text resource. Hidden descendant
directories and directory symlinks are not traversed. Configured root aliases are resolved
before use; descendant symlinks, hardlinks, nonregular files, and parent/path escapes are
rejected on the opened file handle. Actual bytes and decoded text are both bounded.
`SkillLimits` configures scan limits, an 8,000-byte resident catalog budget, and a
32 KiB serialized JSON response budget, within documented hard ceilings in the builder.
Descriptions are shortened before catalog entries are omitted. Host-supplied custom
introductory tool descriptions and hook-rewritten outputs are outside these budgets.

Optional `agents/openai.yaml` sections override corresponding frontmatter `metadata`
sections. Malformed or oversized sidecars reject the skill with a diagnostic, so a broken
explicit-only policy cannot silently become permissive. Supported text metadata includes
`interface.display_name`, `interface.short_description`, and `interface.default_prompt`;
`metadata.short-description` is also supported. UI hints are not automatically executed.

```yaml
policy:
  allow_implicit_invocation: false
interface:
  short_description: Publish a release when explicitly requested.
dependencies:
  tools:
    - type: tool
      value: shell
    - type: mcp
      value: releases
```

Explicit-only policy hides automatic catalog/definition listings; exact selection is still
possible. It is not an authorization boundary. `SkillFilter` controls actual admission.
`SkillSummary::preflight` compares dependencies with host-approved tool and provider name
sets and returns missing requirements. Unknown dependency kinds remain missing. Dependency
metadata never installs, connects, or grants a capability; launcher, OAuth and approval UX
belong to the application. `allowed-tools` continues to constrain delegated callables;
including `tools_select` also allows subsequent discovery under the completion runner's
existing rules. Inline skills retain the calling agent's permissions.

`SkillToolHook`, `SkillsListHook` and `SkillsReadHook` observe/customize ordinary tool calls.
The read hook includes the resolved skill identity, resource and content fingerprint, so
hosts can record usage without forwarding conversation history. Old catalog snapshots are
metadata only and do not authorize later reads. Existing copies of already running
subagents are not cancelled by removing their catalog entries; hosts own session shutdown.

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

### MCP runtime policies

Use `McpServerConfig` constructors, then configure `limits`, `timeouts`,
`concurrency`, `required`, and `startup`. Stdio forwards only platform essentials
plus explicit `env` by default (`inherit_env = true` restores full inheritance).
`routes()` preserves original tool metadata; `server_statuses()` is observational.
Raw MCP outputs remain auditable while the runner uses bounded
`ToolOutput::model_output` presentations, including supported images in tool
responses. Elicitation needs both a handler and per-server opt-in; resource APIs
are separately opt-in. See [MCP_INTEGRATION.md](../MCP_INTEGRATION.md) for defaults,
migration notes, lifecycle guarantees, model compatibility, and an example.
