# MCP Integration Design

This repository provides the reusable MCP host/client layer for Anda runtimes.
Product crates such as `anda-bot` should own user configuration, credential
expansion, and launcher UX, then pass concrete server configs into
`anda_engine::extension::mcp`.

## Scope

The first implementation supports MCP tools only:

- Transports: stdio child process and Streamable HTTP.
- Discovery: `tools/list` is cached as a dynamic `ToolProvider` snapshot. The
  server's `initialize` self-description (title and `instructions`) is captured
  alongside the tools so each server can be surfaced as a capability group.
- Invocation: `tools/call` is dispatched through the live MCP client session.
- Refresh: `notifications/tools/list_changed` marks a session dirty; the next
  async refresh or call refreshes the affected server. Concurrent callers race
  through an atomic claim so only one refresh runs per notification.
- Reconnect: a closed session (for example a crashed stdio child) is
  re-established on the next call. Connection setup is serialized per server so
  racing callers do not spawn duplicate sessions.
- Runtime add: callers can keep an `Arc<McpToolProvider>` after registering it
  with an engine, then call `add_server` to connect a new MCP server and expose
  its tools without rebuilding the `Engine`.

The implementation intentionally does not integrate SEP-2577-deprecated
capabilities: Roots, Sampling, and Logging control. Anda does not advertise
Roots, does not create Sampling messages, and does not call `logging/setLevel`.

## Runtime Shape

`anda_core` exposes a generic `ToolProvider<C>` contract next to the static
`Tool`/`ToolSet` path. A provider returns a synchronous discovery snapshot for
model-facing tool selection and async methods for initialization, refresh, and
execution.

`anda_engine::EngineBuilder` registers providers with
`register_tool_provider`. During `build`, the engine initializes static tools
first and then initializes dynamic providers. Provider initialization is
fault-tolerant: an MCP server that is unreachable at startup is logged and
skipped rather than failing the whole engine build, and its tools become
available once a later refresh succeeds. An explicit `refresh()` instead reports
per-server failures to the caller. Tool discovery and tool calls merge static
tools and provider-backed tools, with static tools retaining precedence if names
collide.

`anda_engine::extension::mcp::McpToolProvider` owns one MCP client session per
configured server. Each MCP server is isolated by id, transport, allowlist, and
denylist.

Servers configured at build time are loaded during provider initialization. New
servers can also be added later through the same provider instance. A successful
runtime add updates the provider snapshot used by `tools_select`,
`Engine::tools`, and tool calls. If the initial refresh fails, the server
registration is rolled back so callers do not observe a partially added server.

## Tool Naming

MCP tool names are not required to match Anda function naming rules. The
provider maps every remote tool to a stable local name:

```text
mcp_<server_id>_<remote_tool_name>
```

Every segment is lowercased and normalized to `a-z`, `0-9`, and `_`. Names that
exceed 64 characters or collide after normalization receive a short hash suffix.
The route keeps both names so calls use the original MCP tool name.

## Capability Groups

A flat tool list hides which tools belong together and how they combine. The
provider therefore exposes one `ToolGroup` per server through the generic
`ToolProvider::groups` contract:

```text
id:           mcp:<server_id>
title:        server title (falls back to `MCP server `<id>``)
description:  server description (falls back to a generic line)
instructions: server `instructions` from the initialize handshake (optional)
members:      every local tool name for that server
```

Groups are a *discovery-layer* concept; they are never sent to model providers
as function-calling schema (the completion API has no group concept). The
built-in discovery helpers expose them top-down:

- `tools_groups` lists the available groups as a compact directory (`id`,
  `title`, `description`, `member_count`) — no tool schemas — so the model can
  scan which bundles exist without flooding its context.
- `tools_select { group: "<id>" }` expands one group into the full schemas of
  all its member tools in a single call.
- `tools_search` / `tools_select` also attach the groups that the returned tools
  belong to, so discovering one tool reveals the bundle's purpose, the server's
  usage `instructions`, and the sibling member names.

The typical flow is therefore: `tools_groups` to survey bundles → `tools_select`
with a `group` id (or specific `tools`) to pull in schemas → call the tools.

The same group machinery also serves static tool bundles: built-in tools that
declare a `ToolGroupInfo` (for example the filesystem and persistent-memory
tools) are surfaced through the identical directory and expansion path.
Before discovery output is returned, group members are normalized against the
currently visible callable definitions. Stale members are dropped, provider
tools shadowed by static tools are hidden from the provider group, and duplicate
group ids are merged so `tools_select { group }` expands the whole visible
bundle deterministically.

## Security Boundaries

- MCP servers are never enabled implicitly by this crate.
- Stdio uses `command` plus `args`; it does not invoke a shell string.
- Streamable HTTP validates custom headers before connecting.
- Remote tool descriptions and annotations are treated as untrusted metadata.
- Server title and `instructions` are likewise untrusted: they are surfaced as
  group data the model reads, never as system instructions or runtime
  directives.
- Calls send only tool arguments and explicitly selected resources, not full
  conversation history.
- Tool results include `server_id` and the original MCP tool name for audit.

## Bot Integration

`anda-bot` should translate its YAML config into `McpServerConfig` values and
register one `McpToolProvider` with the engine builder. Bot-specific concerns
remain outside this repository layer:

- Environment variable and secret expansion.
- Default per-server working directories.
- User-facing approval UX.
- Commands such as `anda mcp list` or `anda mcp ping`.
