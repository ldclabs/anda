# MCP Integration Design

This repository provides the reusable MCP host/client layer for Anda runtimes.
Product crates such as `anda-bot` should own user configuration, credential
expansion, and launcher UX, then pass concrete server configs into
`anda_engine::extension::mcp`.

## Scope

The first implementation supports MCP tools only:

- Protocol: MCP `2026-07-28` plus the older `initialize`-based revisions. See
  [Protocol Revisions](#protocol-revisions).
- Transports: stdio child process and Streamable HTTP.
- Discovery: `tools/list` is cached as a dynamic `ToolProvider` snapshot. The
  server's self-description (title and `instructions`, from `server/discover` or
  the `initialize` handshake) is captured alongside the tools so each server can
  be surfaced as a capability group.
- Invocation: `tools/call` is dispatched through the live MCP client session,
  including the `2026-07-28` shapes that replace a direct result (MRTR
  `input_required` rounds and task handles).
- Refresh: `notifications/tools/list_changed` marks a session dirty; the next
  async refresh or call refreshes the affected server. Concurrent callers race
  through an atomic claim so only one refresh runs per notification. On
  `2026-07-28` peers the notification arrives on a `subscriptions/listen` stream
  the provider opens per session.
- Reconnect: a closed session (for example a crashed stdio child) is
  re-established on the next call. Connection setup is serialized per server so
  racing callers do not spawn duplicate sessions.
- Runtime add: callers can keep an `Arc<McpToolProvider>` after registering it
  with an engine, then call `add_server` to connect a new MCP server and expose
  its tools without rebuilding the `Engine`.
- Authorization: a Streamable HTTP server may use a static bearer token or
  OAuth 2.1. See [Authorization](#authorization).

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

## Protocol Revisions

MCP `2026-07-28` removed the session: there is no `initialize` handshake, no
`Mcp-Session-Id`, and every request carries its protocol version, client
identity, and capabilities in `_meta`. Servers advertise themselves through
`server/discover` instead. Because the opener decides which revisions are
reachable, `McpServerConfig::lifecycle` selects it per server:

```text
auto        probe `server/discover`, fall back to `initialize` (default)
discover    require the stateless lifecycle; fail on older servers
initialize  legacy handshake only, negotiating at most 2025-11-25
```

`auto` covers both server generations. A pre-2026 server usually does not answer
the discovery probe with a JSON-RPC "method not found" it can recover from — an
stdio child rejects the message and exits — so the provider retries once on a
fresh transport with the legacy handshake. An authorization failure is not a
lifecycle problem and is reported as-is instead of being retried. `2026-07-28` is
only ever negotiated through `server/discover`, which proves the peer implements
the stateless lifecycle; the legacy handshake proposes at most `2025-11-25`.

Three consequences of the revision shape the host behavior:

- **List changes are opt-in.** SEP-2575 replaced unsolicited server pushes with
  `subscriptions/listen`. For peers that negotiated `2026-07-28` and advertise
  `tools.listChanged`, the provider opens a `toolsListChanged` stream per session
  and drains it in a background task. Streams are not resumable, so an ended
  stream is reopened while the transport is up, and the session is marked dirty
  across the gap rather than trusting a possibly stale route table.
- **A tool call may not return a result.** An MRTR (SEP-2322) `input_required`
  round that carries only `requestState` is echoed back and the call continues;
  a round that actually requests sampling, elicitation, or roots comes back as a
  failed tool result, because this host advertises none of them. The model sees
  a tool-level error and can choose another path instead of losing the turn.
- **Results may be cacheable.** SEP-2549 `ttlMs` / `cacheScope` are honored by
  rmcp's client response cache. An explicit `refresh()` clears that cache first
  so a manual refresh always re-lists; the notification-driven path relies on
  rmcp invalidating the tool cache when `tools/list_changed` arrives.

### Tasks Extension

The SEP-2663 tasks extension (`io.modelcontextprotocol/tasks`) is opt-in per
server through `McpServerConfig::tasks`. Declaring it tells a server it may
answer `tools/call` with a task handle instead of a result; the provider then
polls `tasks/get` — honoring the server's suggested interval, clamped to
250 ms–10 s — until the task reaches a terminal state, so a long-running tool
does not have to hold its response open. The tool call still blocks, bounded by
`max_wait_secs` (default 300). A task this host walks away from — timeout, an
undeclared extension, or an in-task input request it cannot answer — is
cancelled best-effort so the server can release it.

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
instructions: server `instructions` from the handshake or discovery (optional)
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

## Authorization

Streamable HTTP servers may require credentials. `McpStreamableHttpTransport`
supports a static `bearer_token`, or `McpOAuthConfig` for OAuth 2.1:

- **Client Credentials** (`McpOAuthConfig::ClientCredentials`) is headless: the
  token is obtained during session setup with no human in the loop. Per
  RFC 6749 §4.4.3 this grant issues no refresh token, so the session carries the
  token's deadline and re-establishes itself shortly before expiry to mint a new
  one. `resource` (RFC 8707) is required by the MCP auth spec even though the
  field is optional in the config type.
- **Authorization Code** (`McpOAuthConfig::AuthorizationCode`) is interactive.
  `anda_engine` is a library and does not open a browser or receive the redirect:
  call `begin_authorization` to get the URL, present it however the application
  likes, then call `complete_authorization` with the returned code. A present
  RFC 9207 `iss` on the redirect is validated against the recorded issuer before
  the code is redeemed (SEP-2468). Refresh happens on demand from the stored
  credentials.
- `discover_http_oauth` probes an endpoint (RFC 9728 / RFC 8414) so an
  application can decide from a bare URL whether a flow is needed at all.

Credential persistence belongs to the application. Implement
`McpCredentialStore` to store tokens durably; `InMemoryMcpCredentialStore` is
provided for tests and short-lived processes, and loses tokens on restart.

### Headless And SSH Authorization

The Authorization Code flow needs a browser, but not on the machine hosting the
engine. `begin_authorization` returns a plain URL and `complete_authorization`
accepts the full redirect URL as a string, so an agent on a remote server can
run the flow entirely through its conversation channel:

1. The application surfaces the authorization URL as text (chat message,
   terminal output). The user opens it in the browser on their own machine.
2. The redirect comes back one of two ways:
   - **SSH port forwarding** — the redirect URI is a loopback address
     (`http://127.0.0.1:<port>/callback`), the application listens on that port
     on the server, and the user opens a tunnel alongside their interactive
     session so the local browser reaches that listener:

     ```sh
     ssh -N -L <port>:127.0.0.1:<port> user@your-server
     ```

     `-N` runs the tunnel without a remote shell (add `-f` to background it).
     Use the same port on both sides: the authorization server redirects the
     browser to the exact `redirect_uri` that was registered, so the local port
     must be the one that URI names.
   - **Manual paste, no listener at all** — with the same loopback redirect URI
     and nothing listening, the browser lands on "connection refused", but the
     address bar holds the complete redirect URL including `code` and `state`.
     The user copies that URL into the conversation and the application passes
     it verbatim to `complete_authorization`.

Both variants validate PKCE, the CSRF state, and a present RFC 9207 `iss`
in-process, so the pasted URL cannot complete a flow this provider did not
start. The only constraint is unchanged: `begin_authorization` and
`complete_authorization` must run on the same provider instance, because the
intermediate PKCE/CSRF state lives in memory.

### Re-Authorization And Sign-Out

`begin_authorization` can run at any time — it never consults the stored token,
so a re-consent for updated permission scopes works even while the current
access token is still valid. Completing the flow persists the new grant *and
drops the server's live session*: a session pins the credentials it connected
with, so without the drop the new grant would sit unused in the store until the
old session happened to die. The next refresh or tool call reconnects with the
new credentials; in-flight calls on the old session finish undisturbed.

Two related primitives:

- `disconnect_server(server_id)` drops the session while keeping the server
  registered and its routes intact; the next call reconnects from the store. It
  takes the same per-server connect lock `ensure_session` holds across a
  handshake, so a reconnect already in flight cannot install a session carrying
  the credentials being retired.
- `clear_credentials(server_id)` additionally deletes the persisted grant, so
  the next connection fails with `McpAuthorizationRequired` until the
  interactive flow runs again — the "sign out / force re-consent" primitive.

`McpAuthorizationRequired` is the single signal to act on. It covers both "no
grant is stored" and "the stored grant is no longer usable" — a refresh token the
authorization server has revoked surfaces as an auth failure during credential
acquisition or the handshake, and is reported as `McpAuthorizationRequired` with
the underlying cause logged, so an application that downcasts for it always knows
when to re-run the interactive flow.

To request different scopes than the configured ones, update the server config
(`remove_server` + `register_server` with the new `scopes`; removal leaves
stored credentials in place) and run the flow again.

## Security Boundaries

- MCP servers are never enabled implicitly by this crate.
- Stdio uses `command` plus `args`; it does not invoke a shell string.
- Streamable HTTP validates custom headers before connecting.
- Bearer tokens, OAuth client secrets, and stdio `env` values are redacted from
  `Debug` output, so a config can be logged without leaking expanded secrets.
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
