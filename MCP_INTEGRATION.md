# MCP Integration Design

This repository provides the reusable MCP host/client layer for Anda runtimes.
Product crates such as `anda-bot` should own user configuration, credential
expansion, and launcher UX, then pass concrete server configs into
`anda_engine::extension::mcp`.

## Scope

The implementation provides MCP tools plus opt-in resource and elicitation APIs:

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
  async refresh or call refreshes the affected server. Concurrent callers share
  a serialized refresh; cancelling before publication restores the dirty flag. On
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
fault-tolerant by default: an unreachable optional server is logged and skipped.
Set `required = true` to fail initialization instead. Optional servers can use
`startup = "background"`; their tools become visible only after live discovery
succeeds. No cached annotations or credentials are used as authorization. An explicit `refresh()` instead reports
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
  a round requesting sampling or roots comes back as a failed tool result.
  Standard elicitation is supported only when both the server config and an
  application handler opt in; otherwise it also produces a tool-level error. The model sees
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

The task deadline includes each `tasks/get` response wait; cancellation also
runs when a parent drops the polling future. A `tasks/cancel` acknowledgement
is bounded to two seconds. Per-server `McpTimeouts` defaults are 90 seconds for
setup including fallback, 30 for complete listing, 180 for one tool request,
600 for a logical call including queueing/setup/MRTR/tasks, and 300 for an
elicitation callback. Each connection attempt additionally retains the
45-second bound and the discovery probe its 10-second bound. The logical-call
wall deadline includes human interaction; an application can configure it up
to one day. A task's `max_wait_secs` remains an additional, independent bound.

HTTP handshake and `tools/list` transient failures receive at most two retries
(250 ms and 1 s), within their outer deadline. Tool calls are never retried by
Anda after a transport failure; rmcp retains its protocol/auth challenge handling.


## Tool Naming

MCP tool names are not required to match Anda function naming rules. The
provider maps every remote tool to a stable local name:

```text
mcp_<server_id>_<remote_tool_name>
```

Every segment is lowercased and normalized to `a-z`, `0-9`, and `_`. Names that
exceed 64 characters or collide after normalization receive a short hash suffix.
The route keeps both names so calls use the original MCP tool name. Initial
same-server normalization collisions are all hashed, independently of list
order. Published identities retain their assigned names across additions,
removals, and reordering for the life of the registration. The catalog item
budget also bounds these retained identities; explicitly re-register a server
to reset that budget. Existing unambiguous names are unchanged.
Bulk refresh fetches server listings concurrently and applies successful
snapshots in server-id order, so response timing does not decide which server
keeps an unsuffixed name when local tool names collide at startup.

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
  its member tools in a single call, within the discovery byte budget.
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

Setup reports `McpAuthorizationRequired`. An authorization failure during a
live tool call instead returns `is_error: true` and
`{"error":{"code":"authorization_required","server_id":"..."}}`, allowing the
model and application to recover without exposing transport credentials. The setup error covers both "no
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
- Stdio uses `command` plus `args`; it does not invoke a shell string. It inherits
  only platform essentials by default; put extra values in `env`, or explicitly
  opt into the old full-parent behavior with `inherit_env = true`. Unix child
  processes get a dedicated process group that is terminated when the session
  is released. Windows currently guarantees direct-child cleanup only.
- Streamable HTTP validates URLs and custom headers before connecting. Protocol,
  routing, and conflicting Authorization headers are rejected. Both static and
  OAuth data clients disable redirects, including same-origin redirects.
- Bearer tokens, OAuth client secrets, and stdio `env` values are redacted from
  `Debug` output, so a config can be logged without leaking expanded secrets.
- Remote tool descriptions and annotations are treated as untrusted metadata.
- Server title and `instructions` are likewise untrusted: they are surfaced as
  group data the model reads, never as system instructions or runtime
  directives.
- Tool calls send arguments only. Resource reads are explicit separate operations;
  neither operation forwards conversation history or arbitrary context metadata.
- Tool results include `server_id` and the original MCP tool name for audit.

## Bot Integration

`anda-bot` should translate its YAML config into `McpServerConfig` values and
register one `McpToolProvider` with the engine builder. Bot-specific concerns
remain outside this repository layer:

- Environment variable and secret expansion.
- Default per-server working directories.
- User-facing approval UX.
- Commands such as `anda mcp list` or `anda mcp ping`.

## Catalog Consistency And Budgets

Every server registration has an identity and cancellation token. Catalog
fetches are serialized through publication, and published routes carry a
revision plus the original MCP `Tool` (annotations, output schema and metadata).
Calls refresh dirty catalogs and validate their captured identity/definition
before execution. Publication waits for calls already using that catalog;
removal cancels the registration and prevents in-flight fetches from publishing.
Disconnect retires the session: calls already sent may finish, but queued calls
cannot begin on retired credentials. A reconnect refreshes the live catalog.

`McpConcurrency` defaults to `Serial`. `ReadOnlyParallel` permits concurrent
annotated reads while excluding writes; `Parallel` requires explicit host
configuration. Remote annotations are hints, never approval or permission grants.
Tools with MCP Apps visibility metadata must explicitly include `model` to enter
the model-facing catalog. `server_statuses()` observes state without connecting.

`McpLimits` defaults:

| Budget | Default |
| --- | --- |
| Catalog pages / items / cursor | 100 / 2,048 / 64 KiB |
| Input or output schema per tool | 64 KiB |
| Tool description / title | 8 KiB each |
| Combined server metadata | 32 KiB |
| Incoming stdio line, HTTP JSON body, SSE event | 8 MiB |
| Model result text / decoded media / media blocks | 32 KiB / 5 MiB / 8 |

Repeated cursors and duplicate raw tool names are rejected. Invalid or oversized
catalogs leave the last published snapshot intact and retryable. Schema,
description and title budgets apply only to tools that would be published, so an
excluded or app-only tool cannot fail its server. SSE budgets
include raw comments within each event. rmcp still owns lifecycle negotiation,
JSON-RPC decoding/routing, subscriptions and authentication; the byte/HTTP
adapters enforce bounds before SDK decoding. Header/body errors omit raw bodies.

Discovery outputs are limited to 256 KiB; complete schemas that do not fit are
omitted, and callers can select omitted tools by exact name. Accumulated
discovered schemas are additionally limited to 128 definitions and 256 KiB.
MCP schema adaptation only fills missing/null `properties` on object schemas;
it preserves constraints, compositions and references rather than using lossy
schema compaction.

## Results And Model Presentation

`ToolOutput.output` retains the complete audited MCP envelope, including
`server_id`, original tool name, structured content, content blocks and `_meta`.
`ToolOutput.model_output` contains a bounded `ToolPresentation`: visible text
and inline media. Top-level and content-block `_meta` do not enter that view.
Unknown content and unsupported media produce explicit text notices; oversized
text carries a truncation marker. Structured data and content images can coexist.
A text block that only repeats the structured content as serialized JSON (the
spec's compatibility copy) is shown once; other text blocks are kept.

The runner persists the explicitly tagged presentation inside the existing
`ContentPart::ToolOutput.output`, preserving call IDs and tool/user boundaries.
OpenAI Responses and Anthropic project images into native tool-result blocks;
Chat Completions falls back to text. Gemini 3 projects supported image/document
MIME types into `functionResponse.parts`; older or unknown Gemini model names
fall back to text. Audio remains available to callers and in the neutral view,
but these model APIs currently receive a text notice instead of audio bytes.
Media fallbacks affect only provider requests; persisted neutral history retains
the original presentation for replay with another model. Anthropic omits blank
text blocks from image-only results and omits content for empty presentations.
See [Gemini's function response restrictions](https://ai.google.dev/gemini-api/docs/generate-content/function-calling#multimodal-function-responses).

Hooks that rewrite a result must update `model_output` too, or clear it to use
their replacement raw output. The original MCP envelope is not itself truncated.

## Optional Interaction And Resources

Install `McpToolProviderBuilder::elicitation_handler(Arc<dyn McpElicitationHandler>)`
and set `server.elicitation = true`. The handler declares supported standard
form/URL modes and receives the server id, typed request and cancellation token.
The application owns UI, consent, URL presentation, input validation and approval.
Legacy server requests and modern tool-call MRTR both use that handler. An MRTR
round is checked for unsupported methods before any prompt is opened. Sampling,
Roots, Logging control and proprietary verification extensions remain disabled.
Task-level input requests still cancel the task and return a tool error.

Set `server.resources = true` to call `list_resources`,
`list_resource_templates`, and `read_resource` with an explicit server id and
cancellation token. These methods share the authenticated session and byte/time
budgets. They do not register new model tools or fetch returned resource URIs
from the local filesystem or an unrelated HTTP origin.

## Events

`list_events`, `subscribe_events`, `subscribe_webhook` and `unsubscribe_webhook`
implement the client side of the experimental MCP Events extension
(`events/list`, `events/poll`, `events/stream`, `events/subscribe`,
`events/unsubscribe`). They share the server's session, credentials and budgets.

- **Detection.** rmcp 3.5 drops `capabilities.events`, so `list_events` sends
  `events/list` and returns `None` on "method not found". A server that ignores
  unknown methods is given 10 seconds. Event types over the description or schema
  limits are skipped.
- **Poll and push.** `subscribe_events` runs one subscription on a background
  task and reports `McpEventSignal`s to an application `McpEventSink`: `Active`
  (with `truncated` when events were lost), `Events` with the cursor to resume
  after them, recoverable `Error`s, `ListChanged`, and a final `Terminated`.
  Poll follows `hasMore` and `nextPollMs` (clamped to 1 s – 5 min, 100 events a
  page). Push keeps one `events/stream` request open per subscription, outside the
  tool concurrency queue and without a request timeout; a stream silent for 60 s,
  a closed or retired session, or a full 256-notification buffer reopens it from
  the last accepted cursor. Cancelling sends `notifications/cancelled` (rmcp also
  aborts the response stream over HTTP).
- **Ordering.** rmcp dispatches each notification on its own task, which can
  reorder a burst. `EventTap` wraps every transport and hands
  `notifications/events/*` to the session's router as they are read, keyed by
  `_meta["io.modelcontextprotocol/subscriptionId"]`; notifications that beat the
  stream's registration are held briefly.
- **Delivery guarantee.** The provider keeps no event state. A cursor is used for
  the next request only after the sink accepted the signal carrying it, so delivery
  is at least once; the application persists cursors and deduplicates by `eventId`.
  A sink error restarts the subscription from the last accepted cursor after a
  backoff (2 s doubling to 5 min). Events over 256 KiB are skipped with an `Error`.
- **Webhooks.** The provider subscribes, refreshes (subscribe again before
  `refreshBefore`) and unsubscribes; the application supplies the callback URL and
  `whsec_` secret and receives and verifies the deliveries itself. A host that
  relays webhooks through another MCP server (such as dMsg) drives that server's
  endpoint tools with `call_server_tool`, which calls a tool by its remote name
  for the application rather than the model, whether or not the model sees it.
- **Errors.** `McpEventError` classifies both generations of the draft's codes
  (-32023..-32027 and the earlier -32011..-32015 that OpenAI documents), keeps
  `data.reason`/`data.kind`, and maps authorization failures to
  `AuthorizationRequired`. Not found, forbidden, unsupported, invalid arguments,
  authorization and a removed or re-registered server end a subscription; other
  failures are retried.

Event payloads, descriptions and schemas are untrusted server data, like tool
results. Receiving an event does not authorize any action.

## Credential Transactions

`McpCredentialStore::acquire_refresh_guard` is an optional extension with a
backward-compatible default. The adapter forwards it to rmcp, which holds it
across authoritative credential reread, token exchange and completed persistence.
The in-memory store coordinates managers sharing the same store/server id.
Persistent stores should implement equivalent process/database coordination;
`load`, `save`, and `clear` must not reacquire the same lock. Credential keys must
identify the intended server/account, and changing an endpoint must not reuse an
unrelated grant. OS keychains and browser/callback UX remain application-owned.

## Configuration Example

```rust,no_run
use std::sync::Arc;
use anda_engine::extension::mcp::{McpConcurrency, McpServerConfig, McpToolProvider};

# async fn example() -> Result<(), anda_core::BoxError> {
let mut server = McpServerConfig::streamable_http("catalog", "https://example.com/mcp");
server.required = true;
server.timeouts.call_secs = 120;
server.limits.catalog_items = 512;
server.concurrency = McpConcurrency::ReadOnlyParallel;
server.resources = true;
let provider = Arc::new(McpToolProvider::new(vec![server])?);
let builder = anda_engine::engine::Engine::builder().register_tool_provider(provider)?;
# let _ = builder;
# Ok(())
# }
```

For Rust struct literals, use the server constructors and transport `..Default::default()`
so new policy fields receive their defaults. `ToolOutput` literals must include
`model_output` or use `..Default::default()`; persisted older outputs deserialize
with no presentation override. Servers requiring inherited process secrets must
configure `env` or explicitly select full inheritance.
