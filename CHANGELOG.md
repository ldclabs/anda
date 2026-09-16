# Changelog

All notable changes to the Anda project will be documented in this file.

## [0.16.0] — 2026-09-16

This release moves persistent memory to KIP 2.0, adds caller-isolated memory
spaces, and aligns every workspace crate on version `0.16.0`.

### Changed — workspace

- **Workspace release alignment** — `anda_core`, `anda_engine`,
  `anda_engine_server`, `anda_web3_client`, and `anda_cli` now share version
  `0.16.0`; internal crate requirements have been raised to `0.16`.

- **Published dependency upgrades** — `anda_db`, `anda_db_schema`, and
  `anda_db_tfs` move from 0.11 to 0.13, `ic_cose` and `ic_cose_types` from 0.10
  to 0.11, and `anda_cloud_cdk` from 0.5 to 0.6. The temporary sibling-checkout
  patches for the KIP 2.0 stack are removed now that the 0.13 crates are
  published.

### Changed — anda_engine

- **KIP 1.0 → KIP 2.0** — Upgraded `anda_kip` and `anda_cognitive_nexus` from
  0.11 to 0.13, the release that rewrites both crates for KIP 2.0.

- **The KIP tools take a model-facing argument type** — `MemoryManagement` and
  `MemoryReadonly` previously took the wire `Request` as their `Tool::Args`,
  which made the model write the protocol tag, nest `dry_run` under `options`,
  and respect `deny_unknown_fields`. They now take the new `KipArgs`: a single
  `command` *or* an `operations` batch whose items may be bare command strings,
  with `dry_run` at the top level. `KipArgs::into_request` builds the envelope
  and is where the `command` / `operations` exclusion is rejected, since an
  envelope built from both would silently run only one of them. `KipOperation`
  is the string-or-object entry of that batch. Hosts that construct these tool
  calls themselves must migrate from `Request` to `KipArgs`; unknown fields are
  rejected instead of being silently ignored.

- **Tool definitions come from `anda_kip`** — `FUNCTION_DEFINITION` is now
  deserialized from `anda_kip::KIP_FUNCTION_DEFINITION` instead of being spelled
  out here, so the schema a model sees and the envelope the engine executes stay
  in step across protocol revisions. The read-only tool gains its own
  `READONLY_FUNCTION_DEFINITION` (from `KIP_READONLY_FUNCTION_DEFINITION`) rather
  than reusing the writable schema under a different name: KIP 2.0's read-only
  entry point offers no write vocabulary and no `execution` modes, so the shared
  schema had been advertising a batch mode the read path has no use for.
  `MemoryReadonly::with_kip_function_definitions` overrides it, matching the
  writable tool.

- **The read-only gate moved into the protocol crate** — the batch read-only
  check is now `anda_kip::execute_request_readonly`, which 0.13 added. The
  hand-rolled equivalent here parsed every operation, then let `execute_request`
  parse them all again, and failed the whole request when any one of them did not
  parse. The upstream gate parses once and reports an unparseable operation as
  its own result, so a caller can still correlate it by `op_id`. Admission is
  decided on what each command parses to, never on the `language` label an
  operation declares, so no envelope field can talk a write past the boundary.

- **Read-only batch defaults** — `KipArgs::into_readonly_request` supplies
  `independent` mode for a multi-read tool call, because the read-only tool
  schema does not expose execution settings. Writable calls and raw KIP wire
  requests still require an explicit batch mode; explicit modes are preserved.

- **`MemoryManagement::describe_system` → `describe_self`** — renamed for what it
  reads: the agent's own `SelfModel` Concept, not the host system.

### Added — anda_engine

- **Multi-tenant persistent memory (`Tenancy`)** — `MemoryManagement::with_tenancy`
  chooses how a caller is bound to an identity and a MemorySpace. The default,
  `Tenancy::Shared`, is what every existing engine already did: one brain, every
  call authorized as the engine's own system Principal. `Tenancy::PerCallerSpace`
  is for a host serving more than one caller — each verified caller runs as its
  own Governance Principal, in a MemorySpace it owns, created on its first KIP
  call.

  Isolation rests on ownership rather than on Grants the host has to write and
  keep correct: a Space's owner is authorized in it and nobody is authorized in a
  Space they do not own, so a caller reaches its own memory and no one else's
  with no policy to get wrong. The Space is written by the engine onto every
  request, over whatever the envelope held — `KipArgs` exposes no Space field and
  the shipped tool schemas advertise none, and overwriting keeps that true if
  either ever gains one.

  A new tenant Space inherits the Schema Environment in force in the default
  Space. A fresh Space otherwise resolves Core alone, and Core declares no
  Concept types, so a host that installed its own vocabulary would find the same
  command working for the owner and failing for every tenant.

  An **anonymous** caller gets no Space of its own. `StateFeatures::caller`
  returns the anonymous Principal precisely when nothing was verified, and one
  shared Space for all of them would pool strangers' memories; it runs as the
  Nexus's anonymous Principal on the default Space, where default deny gives it
  nothing until a Space policy says otherwise.

- **`MemoryManagement::query_as` / `execute_as` / `session_for`** — The
  caller-scoped counterparts of `query` / `execute`, for a host running commands
  on a tenant's behalf. `describe_caller` and `get_or_init_caller` now go through
  them, so a caller's own Person Concept lands in that caller's Space.
  `session_for` binds a supplied request to the caller's Space and returns the
  provisioned `Session` for executing it.

- **`MemoryManagement::query` / `execute`** — Run one KIP command and get its
  result value back, rather than an envelope to unpack. `query` goes through the
  read-only gate, `execute` through the state-capable one. Both are the *host*
  acting on the default Space under every tenancy.

## [0.15.1] — 2026-08-07

### Added — anda_engine v0.15.1

- **`SkillManager::set_skill_filter`** — A host can now install a `SkillFilter` predicate deciding which skills on disk the manager may hold. Rejecting a skill drops it everywhere at once: not loaded, not callable, absent from the resident catalog in the tool description, and reported as not found by the reader tool — so an application-level enable/disable switch cannot leave a hidden skill reachable by name. The predicate runs during `load` *before* duplicate resolution, so rejecting the copy in a higher-priority directory promotes the next directory's copy instead of dropping the name. Installing a filter also prunes what it rejects immediately, so the registry is never inconsistent with the policy while waiting for a reload. This exists so an embedding application does not have to keep a second skill registry of its own, which would inevitably disagree with this one about what is dispatchable.

- **`openai-response` provider family** — `ModelConfig` gains a new `family` that always routes through the OpenAI Responses API (`completion_model_v2`) with streaming and effort support, where the existing `openai` family decides by model name (`gpt*` → Responses API, otherwise Chat Completions). A config that must pin the Responses API no longer depends on a `gpt`-prefixed model name.

### Fixed — anda_engine v0.15.1

- **Reading a shadowed skill by name no longer fails as ambiguous** — `skills_manager` resolved a name present in several configured skill directories by refusing it, while `load` had already resolved the same collision by directory priority. A personal skill shadowing a bundled one was therefore loaded and callable but unreadable, which under the inline execution default means unusable. The read path now applies the same priority rule; an ambiguity *within* one directory still errors, since there is no priority there to break the tie.

## [0.15.0] — 2026-08-07

This release folds in the entries previously staged as `0.14.6`, which was
never published, and adds a deep-module restructuring of `anda_engine` and
`anda_core` guided by a full design review (kept as an internal working
document, not part of the repository). Behavior is preserved across the
restructuring except where noted below. Every workspace crate moves to
`0.15.0` because the restructuring changes public API in `anda_core` and
`anda_engine`.

### Added — anda_engine v0.15.0

- **MCP 2026-07-28** — The MCP host now speaks the stateless revision alongside the older `initialize`-based ones. `McpServerConfig::lifecycle` picks how a server is approached: `auto` (default) probes `server/discover` and falls back to the legacy handshake — including on a fresh transport, since a pre-2026 server often drops the connection rather than answering "method not found" — while `discover` and `initialize` pin one lifecycle. `2026-07-28` is only negotiated through discovery, so a server that merely echoes a proposed version cannot pull the host onto a revision it does not implement. Discovery metadata (title, description, `instructions`) feeds the same tool groups the handshake used to.
- **Subscription-based tool updates** — SEP-2575 removed unsolicited server pushes, so for `2026-07-28` peers that advertise `tools.listChanged` the provider opens a `subscriptions/listen` stream per session and drains it in the background. Streams are not resumable: an ended stream is reopened while the transport is up, and the session is marked dirty across the gap so a change announced while nothing was listening still triggers a re-list.
- **MRTR and tasks handling** — A `tools/call` no longer always returns a result. An `input_required` round (SEP-2322) carrying only `requestState` is echoed back and the call continues; one that actually asks for sampling, elicitation, or roots — none of which this host advertises — comes back as a failed tool result, so the model can pick another path instead of losing the turn. The SEP-2663 tasks extension is opt-in per server through `McpServerConfig::tasks`: the provider then polls `tasks/get` at the server's suggested interval (clamped to 250 ms–10 s) up to `max_wait_secs`, and cancels any task it walks away from.
- **MCP re-authorization** — `complete_authorization` now drops the server's live session after persisting the new grant, so re-running the OAuth flow (e.g. for updated scopes) takes effect on the next call instead of waiting for the old session to die. New `disconnect_server` drops a session while keeping the server and its routes — holding the per-server connect lock, so a reconnect already in flight cannot reinstate the retired credentials; new `clear_credentials` also deletes the persisted grant, forcing `McpAuthorizationRequired` and a fresh consent. A grant the authorization server has revoked now also surfaces as `McpAuthorizationRequired` instead of an opaque transport error, so applications learn to re-run the flow. The headless/SSH flow (present the URL as text, paste the redirect URL back) is now documented in `MCP_INTEGRATION.md`.
- **Skill execution modes** — Skills now run inline by default: the `skills_manager` tool returns the full SKILL.md and the calling agent follows it in its own context, keeping the conversation, the user, and the turn's resources in reach. A skill can opt into isolated execution with `execution: subagent` (or `metadata.execution: subagent`) in its frontmatter, exposing it as an `SA_<agent_name>` worker for long-running, parallelisable, or context-hungry procedures. The tool response now reports `execution`, a `callable` name for subagent skills, and `base_dir` for resolving bundled files. Skills that previously ran as subagents must declare the execution mode to keep that behavior.
- **`resource-tags` for skill subagents** — New frontmatter field narrows which offered resources a delegated skill receives; when absent it accepts every offered resource so the current turn's attachments reach it.
- **`model::testing::ScriptedCompleter`** — A programmable completion double
  (queued replies, closures, error injection, request recording, echo
  fallback) available to downstream crates, replacing the need to hand-roll a
  fake provider per test.
- **`subagent::ConversationRecords`** — The two-method persistence port the
  subagent conversation recorder actually needs (`create` / `update`).
  `memory::Conversations` is its AndaDB adapter; AndaDB field encoding no
  longer crosses into the subagent layer, and
  `SubAgentConversationRecorder::with_store` accepts custom implementations,
  so subagent persistence is testable without AndaDB.
- **`context::DiscoveredTools`** — Discovery-tool policy (observation, merge
  probing, output compaction) extracted from the completion runner into the
  module that owns the discovery vocabulary.
- **`BaseCtx::path()`** — Public accessor for the context namespace path
  (`a_<agent>` for an agent call, `t_<tool>` for a tool call). Tool hooks are
  keyed by argument/output types, so two tools with identical `Args`/`Output`
  share one hook slot; this is how a hook identifies which one invoked it.

### Changed — anda_core v0.15.0 (breaking)

- **Registries own their invariant** — The `set` map on `ToolSet`, `AgentSet`,
  and `ToolProviderSet` is now private, so the lowercase-key invariant can no
  longer be bypassed by direct mutation. The replacement interface: `add_dyn`
  (validated insert of a type-erased entry), `iter()` (`(lowercase_name,
  entry)` pairs in name order), and owned `IntoIterator` yielding entries in
  name order. `EngineBuilder::register_tools` / `register_tool_providers` /
  `register_agents` now merge through `add_dyn` instead of re-implementing the
  duplicate check.
- **`CacheFeatures::cache_raw_iter` removed** — It leaked the runtime's
  internal cache entry representation (`(Arc<String>, Arc<(Bytes,
  Option<CacheExpiry>)>)`) into the capability trait and had no production
  consumers. Context implementations simply drop the method; the engine keeps
  an equivalent test-only iterator on its internal `CacheService`.
- **Dead text-decoding entry points removed** — `text_from` and
  `utf8_text_from` (unused owned-`Vec` variants) are gone, and
  `utf8_text_from_bytes` is now private (it equals
  `text_from_bytes_with_encoding(data, None)`). Every entry point the engine
  actually calls is untouched: `text_from_bytes_with_encoding`,
  `text_encoding_for_label`, `text_encoding_label`, `platform_text_encoding`,
  and `windows_code_page_encoding`. `text_from_bytes` is also retained — it has
  no in-tree caller, but it is the only ergonomic platform-default entry point
  (`text_from_bytes_with_encoding(data, platform_text_encoding())`) and is
  plausibly used downstream, so it was kept rather than widening the break.

### Changed — anda_core v0.15.0

- **`model` split into cohesive submodules** — `model.rs` now hosts only the
  call-contract types (agent/tool inputs and outputs, usage, request metadata,
  function definitions). Chat content (`Message`, `ContentPart`, the
  CBOR-safe wire codecs, data-URL helpers) moved to `model::content`, prompt
  documents to `model::document`, and text decoding to `model::text`. All
  names are still re-exported from the crate root, so existing imports are
  unaffected.
- **Registry mechanics deduplicated** — Group aggregation and name-filtered
  definition/function selection are shared by `ToolSet` and `AgentSet`
  through one internal `registry` module instead of two hand-copied
  implementations.
- **`Documents` implements `IntoIterator`** — Consuming iteration over the
  contained documents; `CompletionRequest::append_documents` now uses it
  instead of reaching into private fields.
- **Shared test fixtures** — The two duplicated ~300-line mock contexts in
  `agent.rs` and `tool.rs` tests are now one `MockContext` in an internal
  test-support module, and the twin mock-coverage tests merged into one.

### Changed — anda_engine v0.15.0 (breaking)

- **Provider seam unified on `CompletionFeaturesDyn`** — The delegating
  `anda_core::CompletionFeatures` impls on the Anthropic and Gemini
  `CompletionModel`s are removed (they ignored `resources` and merely forwarded;
  OpenAI's models never had them). Code that called the models through that
  trait should call `CompletionFeaturesDyn::completion` instead.
- **Memory internals no longer public** — The `Arc<Collection>` fields on
  `memory::Conversations` and `memory::MemoryManagement` are now private, so the
  storage abstraction can no longer be bypassed; use the methods, or the new
  `ConversationRecords` port (below).

### Changed — anda_engine v0.15.0

- **One completion driver for all providers** — The four provider adapters
  (Anthropic, Gemini, OpenAI Chat, OpenAI Responses) now share a single
  `drive_completion` algorithm behind an internal `WireFormat` seam; each
  adapter contributes only its wire mapping. The raw-history ordering and
  skip/drain invariants are enforced structurally instead of being hand-copied
  four times. Only observable difference: the Chat adapter's request debug log
  message is now `"Completion request"` like the other adapters.
- **Raw-history pruning moved behind the provider seam** —
  `CompletionFeaturesDyn` gains `prune_unanswered_tool_calls` /
  `prune_tool_interactions` with conservative default implementations; the
  completion runner no longer hard-codes any provider's wire shapes. Custom
  providers can override both with their own typed knowledge.
- **Workspace sandbox is one implementation** — The filesystem tools now
  resolve every path through an internal `WorkspaceScope`; the shell runtime's
  separate narrowing logic is gone and shell now honors the same
  `workspace`/`workspaces` request hints (string, path, or array forms) as the
  filesystem tools — a strict widening, still bounded by the configured root.
  The path-resolution helpers other modules could previously call directly are
  now private to the filesystem module.
- **Module layout** — `context/agent.rs` (6.7k lines) is split into
  `context/agent.rs`, `context/runner.rs`, and shared test fixtures;
  `extension/mcp.rs` (3.6k lines) into `mcp/auth.rs` (OAuth protocol),
  `mcp/session.rs` (transports and lifecycle), and `mcp/router.rs` (name
  mapping and call rounds); the background-task registry moves from `hook.rs`
  to a new `background` module. All previous public paths keep working through
  re-exports.
- **Single engine assembly path** — `EngineBuilder::build`, `::empty`, and
  `::mock_ctx` share one assembly routine, removing three hand-synced copies.
  `mock_ctx` is now documented as the supported way for downstream agent and
  tool authors to obtain an `AgentCtx` in their own tests.
- **`MockImplemented::model_name()`** now returns `"mock_implemented"` instead
  of `"not_implemented"`, so mocks and the null model are distinguishable (the
  runner compares model names to detect live model switches).
- **One tool-call protocol for the built-in extension tools** — The nine
  extension tools (fetch, the four filesystem tools, todo, note, shell,
  skills manager) and the six memory tools now share
  `extension::tool_definition` (parameter schema derived from the typed
  argument struct, `strict: Some(true)`) and `extension::hooked_call`
  (cancellation gate plus `DynToolHook` before/after wiring), replacing ten
  hand-written `json!` schemas and seven copies of the hook boilerplate. The
  two memory schemas that stay hand-built have hard reasons recorded in
  comments (KIP definitions come from outside; `memory_api` flattens an
  internally tagged enum that `schemars` would render as `anyOf`).
  Observable differences:
  - Every tool now rejects a call whose context is already cancelled
    (previously only `search_file` and the shell background path honored the
    token); the fetch, skills-manager, and memory tools gain hook support
    (new `FetchToolHook` / `SkillToolHook` aliases; the two KIP tools share
    the `DynToolHook<Request, Response>` slot since hooks are keyed by
    argument/output types — a hook tells them apart through the new
    `BaseCtx::path()`).
  - The note tool's failed operations (missing items, size limit, unknown op)
    and the shell tool's executor-failure and timeout outcomes still resolve
    to `Ok` with the same typed output, but are now flagged with
    `is_error: Some(true)` so hooks, providers, and telemetry see the failure
    signal. A shell command that runs and exits non-zero remains regular
    output.
  - Derived schemas carry additional metadata the hand-written ones lacked
    (`default` values from serde defaults, `minimum: 0` on unsigned integers,
    descriptions on nested item objects); property shapes, enums, and
    required lists are unchanged.
- **`allowed-tools` is now an upper bound** — A skill that declares `allowed-tools` is granted exactly those tools as a subagent; only skills that declare nothing inherit the manager's default tool set. Previously the configured defaults were merged into every skill, which could escalate a restriction written in a third-party SKILL.md.

### Fixed — anda_engine v0.15.0

- **Rustdoc builds clean again** — Intra-doc links in `extension::mcp::auth`
  and the `extension` module docs were left unresolved by the MCP/hook
  restructuring and failed `RUSTDOCFLAGS="-D warnings" cargo doc`; they now
  resolve.
- **The OpenAI Chat adapter no longer discards the assistant message on a
  failure verdict** — A non-success `finish_reason` (`length`,
  `content_filter`, …) previously dropped the whole turn: the truncated text,
  reasoning, and tool calls vanished and only `failed_reason` survived. The
  Chat adapter now matches the Anthropic and Gemini adapters: the message is
  preserved in both `raw_history` and `chat_history`, while the extracted
  `content`/`thoughts`/`tool_calls` remain gated on success.
- **Interrupted tool calls no longer strand an OpenAI Responses `reasoning`
  item** — Dropping unanswered tool-call requests (after steering, discard, or
  stop) now takes the reasoning item that must immediately precede a pruned
  call out with it, the way completed-interaction pruning already did.
  Previously an interrupt mid-tool-call on the Responses adapter left an
  orphaned reasoning item in `raw_history`, and every subsequent request was
  rejected for a reasoning item without its required following item — wedging
  the conversation until the process restarted.
- **Live model switch replay** — `raw_history` belongs to the model that produced it: replaying one provider's native message JSON (OpenAI `input_text` parts, Anthropic content blocks, Gemini parts) through another provider makes the request unparseable and the provider rejects the whole call, wedging the conversation until restart. Each turn now re-resolves the routed model; when it changes, the engine drops `raw_history` and replays the provider-neutral `chat_history`.

## [0.14.5] — 2026-07-31

### Added — anda_cli v0.14.5

- **Explicit remote HTTP opt-in** — New global `--allow-http` flag is required before the CLI sends signed requests to a non-loopback `http://` endpoint. Local loopback endpoints continue to work without configuration.

### Changed — anda_core v0.14.5, anda_engine v0.14.5, anda_engine_server v0.14.5, anda_web3_client v0.14.5, anda_cli v0.14.5

- **AndaDB 0.10→0.11** — Upgraded `anda_db`, `anda_db_tfs`, `anda_cognitive_nexus`, `anda_db_schema`, and `anda_kip` to 0.11. Conversation pagination now uses the database's newest-first ID query while preserving stable newest-first results across pages.

### Changed — anda_engine v0.14.5

- **rmcp 2.2→3.0** — Updated the MCP client integration for rmcp 3.0, including its peer metadata and OAuth metadata APIs.
- **MCP server discovery** — Tool-provider groups now retain an MCP server's handshake title, description, and instructions so consumers can present a coherent server capability bundle instead of only a flat tool list.
- **On-demand cache namespaces** — Agent, tool, runtime-discovered MCP, and subagent cache namespaces are now created lazily while retaining per-namespace isolation and configured capacity limits.

### Fixed — anda_engine v0.14.5

- **Workspace-bound filesystem and shell tools** — Caller-controlled `workspace`/`workspaces` metadata can now only narrow to a resolved subdirectory of a configured workspace; it can no longer redirect file or shell operations outside the configured roots. File search additionally caps scanned entries and observes cancellation, and `edit_file` rejects a replacement result exceeding the file-size limit before allocation.
- **Conversation resource authorization** — `get_resource_content` now requires the owning conversation and verifies both conversation ownership and resource membership before returning a resource, closing global resource-ID disclosure. Conversation persistence also preserves concurrently queued steering/follow-up messages and clears stale failure reasons after a successful update.
- **Model-adapter edge cases** — OpenAI-compatible completions preserve messages when `finish_reason` is omitted and treat empty tool arguments as `{}`. Anthropic requests derive matching non-empty tool IDs when an upstream provider omitted one, omit persisted reasoning blocks lacking a valid signature, and bound provider-supplied streaming content-block indexes to prevent oversized allocations.
- **Callable routing and cancellation** — Allowlisted subagents and remote callables now match their advertised routing prefixes; duplicate callable names across local and remote sources are suppressed; cancellation closes visible in-flight tool calls with interruption outputs so persisted histories remain replayable.
- **Hook cleanup** — When a later agent/tool start hook rejects, all previously started hooks are unwound even if an intermediate end hook fails, preventing stateful hooks from stranding leases.
- **Skills and background tasks** — Loaded skills retain stable subagent session registries across lookups and reloads; directly reading a skill refreshes its materialized subagent. Background task IDs are namespaced by subagent, preventing same-named sessions from colliding.
- **MCP resilience and routing** — Tool-list change notifications cannot be lost during a refresh; failed lists retry instead of accepting a stale route table; local-name collisions are safely disambiguated; peer-closed transports reconnect; client-credentials sessions reconnect before expiry; and stdio transport environment values are redacted from `Debug` output.
- **Remote engine initialization** — `EngineBuilder::mock_ctx` now registers configured remote engines just as `build` does, so remote tools and agents are available in mock contexts.

### Fixed — anda_engine_server v0.14.5

- **Credential handling fails closed** — Requests carrying malformed authorization headers are rejected instead of silently becoming anonymous, and configured CWT bearer tokens must carry an expiration claim.

### Fixed — anda_web3_client v0.14.5

- **Signed endpoint URL guard** — Rejects embedded URL userinfo such as `https://trusted.example@attacker.example/`, which could otherwise send a signed request to a misleading host.

## [0.14.4] — 2026-07-17

### Fixed — anda_engine v0.14.4

- **Anthropic structured output** — `output_schema` was silently dropped for Anthropic models; now mapped to `output_config.format` with `json_schema` type, so structured output requests work across all three model families.

### Changed — anda_engine v0.14.4

- **SSE deserialization zero-copy** — Wire enums in Anthropic, OpenAI, and Gemini type layers now deserialize by reference (`&str`) instead of cloning the buffered `Value`, removing per-event deep copies on SSE streaming hot paths.
- **Reduced per-request cloning** — The full conversation is no longer cloned per request; request-log clones now live inside the log branch, raw history is built before converting content blocks, and v2 `output`/`parsed_output` are filled in a single pass.
- **Shared model helpers** — `null_default`, `resolve_endpoint`, and `string_enum_serde!` (now with input aliases) hoisted to `model.rs`; applied to five hand-written open string enums.
- **Consolidated test scaffolding** — Triplicated HTTP mock scaffolding merged into `model/test_support.rs`.
- **Deduplicated model infrastructure** — Model constructors, `Models::clone`/`replace`, error-chain walkers, and the OpenAI media content-part mapping are now de-duplicated.

## [0.14.3] — 2026-07-16

### Changed — anda_core v0.14.3, anda_engine v0.14.3, anda_engine_server v0.14.3, anda_web3_client v0.14.3

- **anda_db 0.9→0.10** — Upgraded `anda_db`, `anda_db_tfs`, `anda_cognitive_nexus`, `anda_db_schema`, and `anda_kip` to 0.10. No API changes required in this workspace.

## [0.14.2] — 2026-07-13

### Added — anda_engine v0.14.2

- **MCP OAuth 2.1 authorization** — `McpOAuthConfig` supports two flows for Streamable HTTP MCP servers: interactive Authorization Code with PKCE (`begin_authorization` / `complete_authorization` / `cancel_authorization`) and headless Client Credentials (SEP-1046). The library drives the protocol but the consuming application owns the browser, redirect callback, and credential store.
- **Pluggable credential persistence** — `McpCredentialStore` trait with `load`/`save`/`clear` lets applications back OAuth tokens with an encrypted store. `InMemoryMcpCredentialStore` is the default for development.
- **`McpAuthorizationRequired` error** — Typed error returned when a session needs interactive auth; consumers downcast to trigger the authorization flow.
- **`discover_http_oauth`** — Static method probes an HTTP MCP endpoint for OAuth capabilities (scopes, DCR support) without connecting.
- **`register_server` / `remove_server`** — Register an auth-requiring server without connecting, for deferred connection after the interactive OAuth flow completes.
- **Validation: no `bearer_token` + `auth` mixing** — `McpStreamableHttpTransport` rejects configs that set both.

### Changed — anda_engine v0.14.2

- **rmcp 1.7→2.2** — Upgraded with the `auth` feature, replacing the static bearer-token path with `AuthClient`-based transport when `McpOAuthConfig` is present. Transport config split into `base_transport_config` (for auth client injection) and `transport_config` (static bearer, preserved for backward compat).
- **`list_roots` removed** — The deprecated `list_roots` client handler is removed to match rmcp 2.x.

### Added — anda_core v0.14.2

- **CBOR RPC response body cap** — `MAX_RPC_RESPONSE_BYTES` (16 MiB) enforced with streaming chunk-by-chunk guard; oversized responses are rejected before full buffering, protecting memory-constrained TEE runtimes.
- **`RemoteError` error variant** — Split from `ResultError` so callers can distinguish transport-level decode failures from application-level remote errors.
- **Documents closing-tag injection guard** — `Documents::Display` neutralizes literal `</tag>` delimiters inside untrusted attachment content (case-insensitive), preventing document content from closing the block early.

### Added — anda_engine v0.14.2 (continued)

- **Subagent execution-time tool allowlist** — `CompletionRunner::set_allowed_callables` enforces the subagent's tool whitelist at dispatch time, not just in the definitions sent to the model. An empty allowlist rejects every call; discovered tools (from allowed discovery tools) are granted implicitly.
- **Completion response body cap** — `MAX_COMPLETION_RESPONSE_BYTES` (64 MiB) guards against a runaway or malicious provider streaming unbounded body. Enforced with streaming chunk-by-chunk guard; `Content-Length` is pre-checked before the first byte.
- **ModelConfig api_key redacted from Debug** — Custom `Debug` impl replaces the `api_key` field with `[REDACTED]` so a `{:?}` log line never leaks a credential.
- **Export name validation** — `EngineBuilder::check_exports` rejects misspelled agent names (hard error) and warns on unresolved tool names before the engine starts.
- **Root-level cache namespace** — `Path::default()` is registered on every engine so root-context `cache_get_with` calls (e.g. dynamic remote-engine resolution) hit memory instead of always falling through to the store.

### Changed — anda_engine v0.14.2 (continued)

- **RemoteTool / RemoteAgent cleanup** — Removed the stale `engine: Principal` field; the target engine is resolved from `endpoint` at call time. `RemoteAgent` no longer lowercases a caller-provided name (the caller is expected to supply an already-valid lowercase name, consistent with `RemoteTool`).
- **Resource selection fixed for remote tools/agents** — `select_tool_resources` / `select_agent_resources` now use the same longest-handle + exact-name resolution as endpoint routing, so resources always match the engine/tool the call is routed to even with overlapping handle prefixes.
- **Engine visibility checks hardened** — `ctx_with` and `ctx_with_base` now enforce anonymous/private/protected rules (previously only `agent_run` and `tool_call` did). `ctx_with_base` also validates the agent name before creating the context.
- **Anonymous principal excluded from management** — `is_controller` and `is_manager` reject the anonymous principal even when the engine was built without a Web3 identity and `controller` defaults to anonymous.
- **Agent/tool end hooks always paired with start** — `agent_run` and `tool_call` now invoke `on_agent_end` / `on_tool_end` on the failure path (with a placeholder output) so hooks that track leases (e.g. `SingleThreadHook`) release their accounting.
- **Challenge endorsement guard** — `Engine::challenge` validates that the request's `AgentInfo` matches the engine's own info (deterministic CBOR comparison) before signing, preventing registry entry hijacking via forged agent-info payloads.
- **File search always canonicalizes** — Every path match is now canonicalized and re-checked against the workspace root, closing the symlink escape: a workspace-internal directory symlink pointing outside can no longer enumerate external filenames.
- **ToolsSearch wildcard returns names only** — The `*` query now enumerates name + description (no parameter schema) and is capped at 64 results, keeping listing cheap and preventing context-window blowout.
- **Store list strips namespace prefix** — `store_list` strips the context namespace from `prefix`/`offset` so a returned `ObjectMeta.location` can be fed back as pagination `offset` on a non-root context without a doubled namespace.
- **Cache `get_with` preserves error source chain** — `CacheInitError` wraps the initializer error instead of flattening it to a string, so downstream code can downcast for retryable/status signals.
- **Models::from_configs** — Disabled models are skipped with `info!`; misconfigured models that fail to build are skipped with `warn!` instead of being swallowed silently.
- **OpenAI Chat Completions** — `response_format.json_schema` now wraps the schema in the required `{name, schema, strict}` envelope; streaming requests request `stream_options.include_usage` for billing tracking; `CompletionResponse` populates the `model` field.
- **Gemini** — `tool_choice_required` is honored via `FunctionCallingMode::Any`; `tool_use_prompt_token_count` moved from `output_tokens` to `input_tokens` (it is an input-side count, distinct from `prompt_token_count`).
- **Anthropic** — Empty/whitespace-only partial JSON in a tool-use block finalization now preserves the existing `{}` default instead of overwriting with an empty string, matching the official SDK's `JSON.parse(buf || \"{}\")` guard.

### Fixed — anda_engine v0.14.2

- **Context compaction recovery** — A transport failure during the summarization turn now restores the runner's tools, discovered tools, queued input, and unbound flag; a retry finds a usable runner instead of a permanently tool-less one.
- **Subagent allowlist survives compaction** — `handoff` now carries `allowed_callables` into the replacement runner. Without this the subagent tool whitelist was silently dropped on the first context compaction, letting the subagent call any callable in the engine afterwards.
- **Unanswered tool calls flushed in discard** — `discard_in_flight_request` now unconditionally closes unanswered tool calls in the visible history, fixing the case where the tool round executed but the follow-up model call failed (pending calls drained, but visible history still held a `ToolCall` with no result — unreplayable by providers).
- **MCP cross-server local-name collision** — When two MCP servers produce the same local tool name, the newcomer is disambiguated with a stable hash suffix. If the disambiguated name still collides, the tool is dropped with an error log instead of silently hijacking another server's route.
- **File search symlink escape** — Previously a workspace-internal directory symlink pointing outside let a plain pattern enumerate external filenames. All matches are now canonicalized against the workspace root.
- **Memory expiry no longer deletes shared resources** — `delete_expired_conversations` now leaves resources intact because they are content-deduplicated and may be shared by other active conversations; reclaiming orphans requires a dedicated reference-counted GC pass.
- **Gemini token miscount** — `tool_use_prompt_token_count` was incorrectly added to `output_tokens` instead of `input_tokens`, inflating the output count and undercounting input.

### Changed — anda_core v0.14.2

- **Serde buffering replaces `serde_json::Value` for `ContentPart`** — Deserialization now uses serde's untagged/type-tagged machinery directly, preserving CBOR byte strings (`InlineData.data`, `Principal`, `Action.signature`) across RPC round-trips that previously lost them through the JSON intermediate.
- **Path encoding hardened** — `path_lowercase` and `path_join` no longer double-encode `%` in already-encoded object-store keys. Re-joining a namespace with a `store_list` key is idempotent.
- **Definitions/functions deduplicated by lowercase name** — `AgentSet` and `ToolSet` suppress duplicate schemas when the same tool/agent is requested multiple times (some model providers reject repeated definition names).
- **`/ping` with arguments accepted** — `/ping now` resolves to `Ping` instead of being treated as an unknown command; bare `/` and `/ arg` are plain prompts rather than empty commands.
- **Blank `failed_reason` treated as success** — An all-whitespace failure reason is normalized to `None` in `AgentOutput::into_tool_output`.
- **MCP secrets redacted from `Debug`** — Custom `Debug` impls on `McpStreamableHttpTransport` and `OAuthClientCredentialsConfig` replace bearer tokens and client secrets with `[REDACTED]`.
- **`validate_function_name`: characters → bytes** — Name length is now checked in bytes rather than characters.
- **Dependency cleanup** — Removed unused `futures` and `serde_bytes` from `anda_core` production dependencies (`futures` kept for dev-dependencies).

### Fixed — anda_core v0.14.2

- **`%` double-encoding in object-store paths** — `From<String>` re-encodes `%` to `%25`, so `path_lowercase` and `path_join` switched to `Path::parse` / `Path::from_iter` to preserve already-encoded segments.
- **`cache_store_delete` ordering** — Store is now deleted before cache to prevent a concurrent `cache_store_get` from repopulating a ghost cache entry that survives the delete.
- **Inline data token estimation** — Switched from `(len + 3) / 4` to `len.div_ceil(3)` to match base64 expansion ratio (~4/3 chars per byte, not 3/4).

### Added — anda_engine_server v0.14.2

- **Signed envelope digest required on RPC** — `verify_user` now rejects a `SignedEnvelope` that omits its committed `digest` on body-bound RPC paths, instead of falling back to the server-computed body hash. This is a fail-closed hygiene check (the client must explicitly commit to the body hash); it is *not* a standalone defense against a signing oracle sharing the key — the signature is still verified over the same hash, and genuine resistance requires domain separation in the signature scheme.
- **`ApiKeyMiddleware::exempt_prefix`** — Prefix-based exemption for discovery subtrees with dynamic segments (e.g. `/.well-known/` covers both `/information` and `/agents/{id}`) that cannot be enumerated exactly with `exempt_path`.

### Changed — anda_engine_server v0.14.2

- **README rewritten** — Accurately describes the server as a thin, stateless forwarder; session management, tool integrations, and access control belong to `anda_engine`.
- **Removed unused `origin`** — Builder field and `with_origin` method removed (no consumer used it).
- **Removed unused public `verify_cwt`** — Only the internal `verify_cwt_token` is needed.
- **Decode error hygiene** — Param decode failures now use `Display` (not `Debug`) so the client sees the parser's error message without the raw request bytes.

### Added — anda_web3_client v0.14.2

- **URL smuggling guard** — `check_url` now parses with `reqwest::Url` instead of a string-prefix check, rejecting non-http(s) schemes (`file`, `ftp`, `data`, `ws`), bare strings, and URLs with no host. Previously a `file:///etc/passwd`-style target would pass the prefix guard and attempt a connection.
- **All-zero root secret warning** — `ClientBuilder::build` logs a `warn!` when the default all-zero secret is used; the derived identity and all sub-keys are public and predictable.
- **Identity load improvements** — `load_identity` detects existing file paths before falling back to hex decoding, so a PEM file with `Secp256k1Identity` errors now surfaces the real parse failure instead of being misinterpreted as a hex string. `identity_from_pem` no longer masks the Ed25519 parse error when a Secp256k1 parse fails.

### Changed — anda_web3_client v0.14.2

- **Root secret zeroized** — `root_secret` field wrapped in `Zeroizing<[u8; 48]>`; the long-lived copy is wiped from memory on `Drop`, reducing the exposure window.
- **Dependencies** — Added `zeroize` to workspace dependencies; `log` and `zeroize` are now optional dependencies gated behind the `client` feature.
- **Query-signature verification (clarified, not changed)** — Documented that the default `Agent` does not request node keys or verify query signatures — the behavior since 0.14.0, now spelled out in a code comment: a non-TEE client reads canister state through a trusted boundary node. Pass your own `Agent` via `with_agent` to enable verification.
- **README rewritten** — Accurate feature descriptions (ICP canister calls, signed HTTP/CBOR-RPC, deterministic key derivation), feature-flag documentation, and security caveats (endpoints passed to signed calls must be trusted).

## [0.14.0] — 2026-07-05

### Removed — anda 0.14.0

- **`CanisterCaller` from `BaseContext`** — The `CanisterCaller` bound is removed from `BaseContext`. Runtimes requiring canister access must implement the trait separately on their context type. All built-in impls (`AgentCtx`, `BaseCtx`) and the `MockCanisterCaller` test helper are removed.
- **Canister methods from `Web3ClientFeatures`** — `canister_query_raw` and `canister_update_raw` are removed from the Web3 client trait surface. Runtimes needing raw canister access should use their own client directly.
- **`Web3SDK` enum and `Web3Client` wrapper** — `Web3SDK` is now a plain struct wrapping `Arc<dyn Web3ClientFeatures>` instead of a `Tee`/`Web3` enum. All match-based dispatch is flattened to direct trait-object calls.

### Added — anda 0.14.0

- **Per-task background cancellation** — `BackgroundHandle` + `BackgroundTaskControls` hook primitives with per-task child tokens for shell commands. New `/stop_task <task_id>` subagent control command stops individual background tasks without disturbing sibling tasks or the session.
- **`tee_attestation()` on `Web3ClientFeatures`** — New trait method (default `Ok(None)`) so TEE-backed clients can attach attestation evidence. Engine's `challenge_response` now uses a single unified path for both TEE and non-TEE flows.
- **`anda_web3_client` feature flags** — `client` (non-TEE, `ic-agent` + local key derivation), `tee` (TEE gateway), `full` (both). Default build pulls neither `ic-agent` nor `ic_tee_*` crates.
- **`crypto` + `tee` modules in `anda_web3_client`** — Deterministic key derivation ported from `ic_tee_gateway_sdk::crypto` (byte-for-byte identical), and `TeeClient` adapting `ic_tee_gateway_sdk` to the engine's `Web3ClientFeatures` trait.

### Changed — anda 0.14.0

- **Dependency upgrades**: `ic-agent` 0.47→0.48, `ic_auth_types` 0.9→0.10, `ic_auth_verifier` 0.9→0.10; new: `ic-ed25519` 0.6, `ic-secp256k1` 0.3.
- **Engine dependency diet** — Removed `ic_cose`, `ic_tee_cdk`, `ic_tee_gateway_sdk` from `anda_engine`; `ic_auth_verifier` feature reduced from `full` to `envelope`.
- **`rand_bytes` self-implemented** — Uses `rand::fill` directly instead of re-exporting `ic_cose::rand_bytes`, eliminating the last `ic_cose` dependency from `anda_engine`.

### Fixed — anda 0.14.0

- **Base64 blob serialization** — Test assertions updated to match `b64:` prefix format for inline data and resource blob encoding.

## [0.14.1] — 2026-07-05

### Changed — anda_engine v0.14.1

- **Background handle typed payload** — `BackgroundHandle` now carries a typed `Arc<dyn Any + Send + Sync>` payload via `with_data()`/`data()` methods, replacing the parallel `background_tasks` map in `SubSession` with data stored directly on the handle. `created_at` and `elapsed_ms()` provide lifecycle observability on the handle itself.
- **`BackgroundTaskControls` ergonomics** — Internal `Mutex` upgraded to `RwLock`; new `get()` / `get_data()` / `handles()` / `is_empty()` accessors; `finish()` now returns the removed handle; `finish_all()` clears all tasks; `stop_all()` removed with stop logic inlined at call sites.

## [0.13.15] — 2026-07-05

### Added — anda_engine v0.13.15

- **Provider raw-history pruning** — `prune_req_raw_history()` reclaims context-window budget for long-lived subagent sessions by removing consumed tool calls and results from accumulated provider-native JSON history, operating directly on raw JSON to preserve provider-specific shapes (OpenAI Chat/Responses, Anthropic, Gemini) that a Message round-trip would lose.
- **SSRF protection for model-controlled HTTP fetches** — `validate_public_url()` blocks requests to loopback, private, link-local, metadata, and unspecified addresses before any outbound connection is made.
- **Shell process-group cleanup** — Spawned shell children now belong to their own process group; cancellation kills the entire group so background descendants are not left behind.

### Changed — anda_engine v0.13.15

- **anda_db 0.8 → 0.9** — Bumped anda_db, anda_db_tfs, anda_cognitive_nexus, anda_db_schema, and anda_kip dependencies from 0.8 to 0.9, adopting the hardened JSON serialization, KQL pagination, and full-scan capping from anda-db 0.9.0.
- **Auth failures now return 401** — Bad credentials (wrong signature, tampered body, expired token, wrong target) now return HTTP 401 instead of silently downgrading to anonymous access.
- **Store size enforcement** — Uploads exceeding `MAX_STORE_OBJECT_SIZE` are now rejected at put time with a clear client error.

### Fixed — anda_engine v0.13.15

- **Symlink escape via workspace** — Filesystem reads now re-verify the canonicalized path resides inside the workspace after resolving symlinks, preventing reads to host files through workspace-local symlinks.

## [0.13.14] — 2026-06-29

### Added — anda_engine v0.13.14

- **Subagent conversation persistence** — Engine builders can now install a subagent conversation recorder so blocking subagent calls and background sessions are persisted as conversations, expose their conversation IDs, and retain status, usage, artifacts, metadata, and failure information for operational audit.

## [0.13.13] — 2026-06-27

### Fixed — anda_engine v0.13.13

- **Subagent compaction with pending tool calls** — Subagent session compaction now executes pending tool calls and records their tool outputs before summarizing, preventing compacted histories from stranding unanswered tool-call requirements.

## [0.13.11] — 2026-06-25

### Fixed — anda_engine v0.13.11

- **Interrupted tool-call history** — Completion runners now append explicit error tool outputs when pending tool calls are discarded, stopped, or interrupted by steering, preventing follow-up requests from carrying dangling tool-call state.
- **Provider raw-history cleanup** — Raw provider histories now prune unanswered tool calls recursively across nested OpenAI Responses, shell, patch, MCP approval, and Gemini function-call shapes while preserving surrounding text and metadata context.
- **OpenAI minimal reasoning compatibility** — OpenAI model requests now map `ModelEffort::Minimal` to `low` reasoning effort instead of sending the unsupported `minimal` value.

## [0.13.10] — 2026-06-25

### Added — anda_core v0.13.10

- **Hyphenated function names** — Agent, tool, and function-name validation now allows hyphens (`-`) in addition to letters, digits, and underscores, with documentation and tests updated to match the accepted naming rules.

### Added — anda_engine v0.13.10

- **Mutable completion chat history access** — `CompletionRunner::chat_history_mut()` exposes ordered mutable access to accumulated chat history messages so callers can update recorded context in place without appending replacement messages.

### Fixed — anda_engine v0.13.10

- **MCP peer metadata compatibility** — MCP tool discovery now handles the current `rmcp` peer-info return shape when capturing server title, description, and instructions for capability groups.

## [0.13.9] — 2026-06-20

### Fixed — anda_engine v0.13.9

- **Child agent context namespaces** — Child agent contexts now switch to the child agent namespace while preserving inherited extension state, so nested agent and tool contexts use the correct agent ownership metadata.

## [0.13.8] — 2026-06-20

### Added — anda_engine v0.13.8

- **Reusable completion handoffs** — Completion runners can now summarize long conversations into compact continuation handoffs and restart from the resulting summary while preserving base tool configuration and accumulated usage metadata.

### Changed — anda_engine v0.13.8

- **Subagent compaction reuse** — Subagent session compaction now delegates to the shared completion-runner handoff path, reducing duplicated context-reset logic while preserving session artifacts and tool usage.
- **Model request retries** — Completion model requests now retry up to three times with a longer retry delay cap, improving resilience to transient provider failures.

## [0.13.7] — 2026-06-20

### Added — anda_core v0.13.7

- **Agent capability groups** — Agents can now declare `ToolGroupInfo` metadata so related agent callables are discoverable as coherent bundles alongside tool and provider groups.

### Added — anda_engine v0.13.7

- **Agent group discovery** — Discovery helpers now include grouped agents in `tools_groups` / `tools_select` outputs, preserving bundle instructions and sibling agent member lists when a grouped agent is selected.

### Changed — anda_engine v0.13.7

- **Subagent compaction scheduling** — Removed the extra idle-loop compaction trigger so subagent sessions rely on the pending-input compaction path instead of compacting again at every idle boundary.


## [0.13.6] — 2026-06-18

### Added — anda_core v0.13.6

- **Tool capability groups** — Tools and dynamic providers can now expose `ToolGroupInfo` / `ToolGroup` metadata so related callables are discoverable as bundles without changing model-provider function schemas.

### Added — anda_engine v0.13.6

- **Tool group discovery helpers** — Added `tools_groups` plus `tools_select { group }` expansion and group annotations in `tools_search` / `tools_select` outputs, letting agents survey capability bundles first and then load all member schemas on demand.
- **Built-in filesystem and memory groups** — Filesystem workspace tools and persistent-memory/conversation tools now advertise shared capability groups with usage guidance and complete member lists.
- **MCP server capability groups** — MCP tool providers now capture server title, description, and `instructions` from the initialize handshake and surface each server as one discovery-layer tool group.
- **Subagent live status polling** — Added read-only `/status` session polling and manager-level `status` catalog output with elapsed time, idle time, token usage, turns, latest progress, and active background tasks.
- **Grapheme-safe output truncation** — Shared truncation helpers now respect Unicode grapheme-cluster boundaries, so shell/status/filesystem inline previews do not split multi-codepoint emoji, flags, skin-tone modifiers, or combining marks.

### Fixed — anda_engine v0.13.6

- **Tool group discovery normalization** — `tools_groups` / `tools_select` now filter stale or shadowed provider group members and merge duplicate group ids before returning discovery output, so group expansion matches the callable schemas actually visible to the current model turn.
- **Streaming completion request timeout override** — Streaming model requests now set their own 10-minute total timeout at the request level, so downstream applications that inject a shared HTTP client with a shorter generic timeout do not abort long-but-progressing SSE completions before the model transport budget.

## [0.13.5] — 2026-06-18

### Added — anda_core v0.13.5

- **Dynamic tool provider contract** — Added `ToolProvider`, `ToolProviderSet`, and borrowed `BoxFut` support so runtimes can expose tools discovered at runtime while keeping static `Tool`/`ToolSet` behavior intact.

### Added — anda_engine v0.13.5

- **Runtime-discovered tool providers** — Engine builders can now register dynamic tool providers, merge provider-backed functions into tool discovery, route direct and agent-driven tool calls through providers, and initialize providers during engine build.
- **MCP tools extension** — Added `anda_engine::extension::mcp`, a reusable MCP host/client provider backed by `rmcp` with stdio and Streamable HTTP transports, tool allow/deny filters, legal Anda tool-name mapping, dirty refresh on `tools/list_changed`, audited tool outputs, and explicit exclusion of deprecated Roots/Sampling/Logging control capabilities.

### Fixed — anda_engine v0.13.5

- **Streaming completion timeout handling** — Model completion clients no longer use HTTP/2 keep-alive PINGs as the liveness detector for long SSE reasoning streams. Completion transport now relies on a per-read body idle timeout plus the existing total request timeout, preventing provider/CDN PING ACK delays from aborting streams that are still producing body chunks.


## [0.13.4] — 2026-06-17

### Fixed — anda_engine v0.13.4

- **Subagent compaction before oversized input batches** — Idle subagent sessions now compact before attaching large batched follow-up or steering inputs, preventing background-result bursts from overflowing the context window before summarization can run. Compaction also refreshes session activity so small idle timeouts do not immediately reclaim freshly compacted sessions.


## [0.13.3] — 2026-06-15

### Changed — anda_core v0.13.3

- **Slash command argument parsing** — Added `PromptCommand::command_argument()` so command handlers can consistently extract the user-provided text after a slash command prefix.

### Changed — anda_engine v0.13.3

- **Subagent stop and cancel semantics** — `/stop <reason>` now stops the current session task while keeping the session idle and reusable, while `/cancel <reason>` continues to end the session runner. Stopped background task output is suppressed so stale child results are not forwarded after a stop.
- **Subagent compaction tool discovery retention** — Session compaction now preserves whether discovered tool definitions should be merged into future completion requests, avoiding tool-discovery state loss after long session handoffs.


## [0.13.2] — 2026-06-14

### Added — anda_engine v0.13.2

- **Runtime model registry replacement** — Added `Models::replace` to atomically replace a model registry from another `Models` instance, enabling callers to reload model configuration without preserving stale labels from the previous registry.


## [0.13.1] — 2026-06-14

### Changed — anda_engine v0.13.1

- **Subagent session lifecycle controls** — Subagent definitions and the manager tool schema now support an `idle_timeout` setting for session mode, letting callers tune idle session reclamation while preserving the engine default when unset.
- **Subagent progress and compaction robustness** — Session runners now emit visible progress signals before the next idle boundary, filter signalless tool-call noise, preserve usage/artifacts across context compaction, and fail loudly instead of replacing history with an empty compaction summary.

### Fixed — anda_engine v0.13.1

- **Native shell environment injection** — Restored the native shell runtime identity so safe host environment variables such as `PATH` are forwarded correctly, while keeping background task IDs prefixed with the shell tool name.
- **Native shell finalization latency** — stdout and stderr reader shutdown checks now run concurrently, avoiding doubled grace-period waits when descendant processes keep both pipes open.

## [0.13.0] — 2026-06-13

### Changed

- **Workspace release and dependency alignment** — Bumped the Anda workspace crates to `0.13.0`, aligned internal crate dependencies on the `0.13` series, and upgraded the ICP/TEE/Anda data stack to `anda_db_tfs 0.8`, `anda_db_schema 0.8`, `anda_cloud_cdk 0.5`, `ic_cose 0.10`, `ic_cose_types 0.10`, `ic-oss-types 1.3`, `ic_auth_types 0.9`, `ic_auth_verifier 0.9`, and `ic_tee_* 0.7`.
- **CBOR serialization backend** — Replaced direct `ciborium` usage across the published workspace crates with `cbor2`, using canonical encoding for cache/store data, HTTP RPC payloads, signed Web3 requests, engine server RPC responses, notes, and subagent persistence.

### Fixed — anda_engine v0.12.37

- **Transport error diagnostics** — Completion transport errors now preserve their source chain and include upstream request IDs, received stream byte counts, and elapsed time for response-body and mid-stream SSE failures, making timeout and upstream-abort diagnosis more actionable.

## [0.12.36] — 2026-06-11

### Fixed — anda_engine v0.12.36

- **Bounded filesystem and shell tool output** — File reads now cap inline text and binary previews with explicit truncation markers, glob searches tolerate unreadable or dangling entries while enforcing a scan cap, and native shell execution bounds captured output, terminal-progress rendering, inherited-pipe waits, and cancellation cleanup for background processes.
- **Subagent session robustness** — Session calls now validate structured arguments, atomically claim session IDs to avoid duplicate runners, report inactive control commands cleanly, preserve usage and artifacts across compaction, forward background usage deltas, and buffer stream steering while a completion step is in flight.
- **Conversation and memory edge cases** — Conversation batch reads, pagination, search limits, expired-deletion loops, timestamp serialization, resource existence checks, and resource ownership checks are now hardened to avoid default-limit truncation, cursor overlap, dangling references, and cross-conversation resource reads.
- **Model transport and response handling** — Completion retries now include a short backoff honoring capped `Retry-After` hints, shared HTTP request timeout allows long reasoning calls, SSE `[DONE]` detection is line-anchored so generated text cannot truncate streams, and OpenAI Responses `incomplete`/`failed`/`cancelled` statuses surface as failure reasons while preserving partial content.

### Changed — anda_engine_server v0.12.1

- **Embeddable and testable HTTP server router** — `ServerBuilder::build_router()` exposes the configured Axum router for embedding and integration tests, while engine RPC dispatch now shares CBOR/JSON decoding and result encoding paths, resolves the `default` engine consistently, and keeps request logging behavior unified.
- **API key middleware hardening** — API key checks now compare equal-length keys in constant time and keep exempt-path configuration clone-friendly.

### Fixed — anda_web3_client v0.12.1

- **Signed HTTP/RPC client hardening** — The Web3 client now caches its principal at build time, centralizes HTTPS guard and signed-header construction, avoids echoing secret material in identity-load errors, and shares signed CBOR RPC request construction across async and trait-based call paths.

## [0.12.35] — 2026-06-10

### Fixed — anda_engine v0.12.35

- **Follow-up delivery during tool execution** — `CompletionRunner` now queues follow-up messages for the next safe user turn and delivers them after pending tool-call results finish instead of waiting for a fully idle boundary. Tool outputs are preserved in chat history before the follow-up is sent, while steering messages still take priority.

## [0.12.34] — 2026-06-10

### Changed — anda_engine v0.12.34

- **Todo tool operation API** — Replaced the `todos`/`merge` write contract with explicit `op=read|set|update` and `items` parameters. `set` replaces the list, `update` patches only changed ids, empty ids are ignored instead of materialized, and write calls now return summary counts while `read` returns the full item list. This reduces tool-call payload size during long-running work and keeps task-list updates focused on changed items.

- **Note tool operation API** — Replaced substring-based `action=add|replace|remove` writes with stable-id `op=read|set|upsert|delete` and `items` parameters. Notes now persist in the `notes_v2` store, writes return compact summary counts, and read/load operations return structured note items with usage summaries.

- **Deprecated extension cleanup** — Removed the deprecated `google` and `extractor` extension modules from `anda_engine`, eliminating the legacy Google Custom Search tool and generic extractor helper that were previously marked deprecated.

- **Unified child context paths** — Agent and tool child context paths now consistently use underscore-separated names such as `a_echo_agent` and `t_echo_tool` on every platform.

### Fixed — anda_engine v0.12.34

- **HTTP client response decoding defaults** — Restored `reqwest`'s default response decoding behavior for the shared model HTTP client instead of globally disabling gzip, Brotli, zstd, and deflate decoding.

- **Streaming completion body read resilience** — SSE completion readers now return immediately after receiving `data: [DONE]`, preserving completed OpenAI-compatible streams even if the server or proxy closes the HTTP body with a late transport error. `reqwest` decode errors are also classified as retryable model transport failures so upper layers can apply delayed retry behavior instead of treating them as permanent completion failures.

- **Structured subagent tool arguments** — Subagent calls now preserve full structured argument objects, including `session`, `model`, and `effort`, instead of collapsing any object with a `prompt` field down to the prompt string. This keeps asynchronous/session subagent calls working while preserving legacy plain-string and single-`prompt` agent behavior.


## [0.12.33] — 2026-06-09

### Changed — anda_engine v0.12.33

- **Adaptive discovered-tool request merging** — `CompletionRunner` now tracks schemas returned by `tools_search` and `tools_select`, promotes repeatedly selected discovered tools into subsequent request tool definitions, and compacts repeated discovery outputs once schema merging is enabled. This lets long-running agents call tools after repeated discovery without resending full schemas in every tool-output context.
- **Identity-encoded completion transport** — The shared model HTTP client now disables automatic response decompression so streaming readers can consume raw SSE bytes even when a provider or proxy mislabels `Content-Encoding`.

### Fixed — anda_engine v0.12.33

- **Failed in-flight tool-result cleanup** — Added `CompletionRunner::discard_in_flight_request()` to clear stale request content, pending tool calls, and dangling raw tool-call history after a transport-level model failure before processing newly queued input.

## [0.12.32] — 2026-06-06

### Changed — anda_core v0.12.8

- **Legacy text decoding support** — Added shared text encoding helpers that keep UTF-8 as the preferred path while allowing platform-local legacy text fallback on Windows. Resource-to-text conversion now uses MIME-aware fallback decoding for text-like resource blobs without treating binary media as text.

### Changed — anda_engine v0.12.32

- **Filesystem, shell, and skill tools handle platform-local text encodings** — File reads, writes, shell output previews/progress, and `SKILL.md` loading now decode or encode supported text encodings such as GBK when needed, preserving UTF-8 behavior by default and keeping binary/unsupported data on the base64 or error paths. Shell progress streaming now preserves multibyte character boundaries for both UTF-8 and common legacy multibyte encodings.
- **Streaming completion responses are more robust** — OpenAI-compatible, Anthropic, and Gemini streaming requests now ask providers for identity-encoded event streams and the shared parser accepts BOM-prefixed SSE, NDJSON, plain JSON event payloads, and JSON arrays. This avoids provider/proxy response-shape surprises while preserving existing SSE handling.
- **Windows-compatible agent and tool context paths** — Agent/tool child context paths now avoid colon separators on Windows while preserving existing colon-based namespaces on Unix-like systems, keeping existing deployments stable and making Windows storage paths valid.

### Fixed — anda_engine v0.12.32

- **Native shell final progress flush** — Background shell tasks now emit any final complete stdout/stderr progress lines before the background-end hook, so short-lived commands do not lose their last progress update when they exit before the next progress interval.
- **Portable skill output paths** — `skills_manager` now normalizes relative `SKILL.md` paths with `/` separators on Windows, matching the existing API output shape on Unix-like systems.

### Fixed — anda_core v0.12.8

- **BOM-only resource text is filtered out** — Legacy text decoding now rejects non-empty byte slices that decode to an empty string, preventing Windows fallback decoding from turning BOM-only binary or empty-looking resources into empty prompt documents.

## [0.12.31] — 2026-06-06

### Changed — anda_engine v0.12.31

- **Model completion retry and retry metadata** — Added a shared `ModelError` type and provider request helper that retry transient completion failures once across OpenAI-compatible, Anthropic, and Gemini adapters. Retryable HTTP statuses now include request timeout, rate limiting, 5xx gateway/server failures, and provider-specific `529`; exhausted retryable failures expose retryability, HTTP status, and `Retry-After` metadata for upper-layer delayed retry decisions. Streaming response read failures now use the same retryable transport error path, and Anthropic stream `overloaded_error` / `rate_limit_error` events are marked retryable.
- **Tool discovery wording clarified** — Updated `tools_search` and `tools_select` descriptions to make schema discovery explicit: returned callable schemas live in tool-output context and should be called directly instead of being dynamically inserted into subsequent `CompletionRequest::tools`. Added regression coverage that selected tool schemas remain available as context without expanding the request tool list.

## [0.12.30] — 2026-06-04

### Changed — anda_core v0.12.7

- **Recursive strict JSON Schema normalization** — `normalize_strict_schema()` now recursively applies `additionalProperties: false`, `properties: {}`, and `required: []` defaults to all nested object schemas, not just the root. Previously propertyless nested objects (e.g. `parameters` without explicit `properties`) were left open, breaking strict-mode contracts for tools with nested object parameters. Added tests for nested propertyless object closure and recursive normalization.

### Changed — anda_engine v0.12.30

- **Reasoning content merged into message text** — `ContentPart::Reasoning { text }` in `message_into()` is now appended as `ContentItem::Text` in the current message content block instead of being emitted as a separate `MessageItem::Reasoning`. This simplifies the OpenRouter-facing message structure and avoids injecting standalone reasoning items that providers may reject or misinterpret.

## [0.12.29] — 2026-06-04

### Added — anda_engine v0.12.29

- **Multiple skill directories in `SkillManager`** — Added `SkillManager::new_with_dirs()` so skill loading and lookup can scan the default skill creation directory plus additional read-only skill roots. Skill descriptions now list all configured directories, duplicate directory entries are deduplicated, duplicate skill names are skipped after the first load root, and displayed `SKILL.md` paths are made relative to the matching configured root. Added coverage for loading and reading skills across multiple directories.

### Fixed — anda_engine v0.12.29

- **Subagent session resource and compaction edge cases** — Resource-only session follow-ups now run instead of being dropped, resource attachments are converted independently so one invalid resource no longer discards all content, compaction handoff calls temporarily disable tool definitions while preserving them for subsequent turns, final session output falls back to the latest visible progress when compaction produces no reportable result, and background session start hooks are emitted only after acknowledgement hooks succeed. Added regression coverage for resource-only follow-ups, compaction finalization, tool restoration after compaction, and failed acknowledgement hooks.

- **Agent completion runner edge cases** — Fixed completion streams so a pending model future is retained across polls instead of being recreated, document-only requests now execute without requiring prompt/content text, steering after tool calls prunes only unanswered raw tool-call items while preserving prior raw history and assistant reasoning, completed runners ignore late steering/follow-up queues, and remote agent resource selection now uses the unprefixed agent name. Added regression coverage for the stream polling, raw-history pruning, steering, and document-only paths.

### Changed — anda_engine v0.12.29

- **Hardened OpenAI-compatible response parsing** — Chat Completions and Responses now tolerate DeepSeek-compatible variants: nullable usage detail objects and counters, unknown `service_tier` strings, `reasoning` aliases for reasoning content, `tool_call`/`tool_use`/`function_call` finish reasons, JSON object tool arguments/inputs, and missing message/tool-call role/type/id fields. Added tests for non-streaming and streaming compatibility shapes.
- **Expanded provider compatibility tolerance** — Anthropic streaming now ignores unknown stream events and content deltas instead of failing; Gemini now preserves unknown enum values, accepts nullable usage/safety metadata, and treats missing/unspecified finish reasons as successful candidate output. OpenAI Responses now preserves unknown statuses and streaming tool-call chunks can omit indexes. Added provider tests for these compatibility shapes.

## [0.12.28] — 2026-06-04

### Changed — anda_engine v0.12.28

- **Removed fallback model support** — Removed the `fallback_model` registry slot from `Models`, the `with_fallback_model()` builder API, the `fallback_model()` accessor, fallback routing from `get_model()` and `resolve()`, and all fallback-related completion logic in `CompletionRunner::step()`. A `fallback` label is now an ordinary label with no special runtime behavior. Model routing uses the primary model plus explicit labels only.
- **Refactored response body reading** — Replaced `response.text().await` + `serde_json::from_str` with `response.bytes().await` + `serde_json::from_slice` across OpenAI, Anthropic, and Gemini completions. Error messages now use `String::from_utf8_lossy` for safer non-UTF-8 body display.
- **Improved subagent output lifecycle** — Added `with_session()`, `latest_output()`, `finalize_output()`, and `record_failed_output()` helpers to `SubSessionRunner` for consistent session tracking and error reporting. Cancellation failures and compaction errors now properly record their output before returning errors. Fixed `needs_compaction()` panic on `context_window == 0` by treating it as unlimited. Added 3 new tests for output lifecycle and edge cases.

### Added — anda_core v0.12.6

- **`utf8_text_from_bytes()` and `utf8_text_from()`** — New text-detection helpers that check for excessive control characters (≤5%) in the first 4KB before treating a byte buffer as text. Prevents binary blobs from being incorrectly rendered as garbled text in `ContentPart` and `Document` conversions.
- **Refined blob-to-text conversion** — `ContentPart::TryFrom<Resource>` and `Document::From<&Resource>` now use the new text-detection functions instead of naive `String::from_utf8`. `Document` conversion also uses `ResourceRef` to exclude `blob` from metadata and now correctly includes blob text content when detected as text.

## [0.12.27] — 2026-06-01

### Changed — anda_engine v0.12.27

- **`ContentPart::Any` handling unified across providers** — Refactored the `ContentPart::Any` → typed-part conversion paths in OpenAI, Anthropic, and Gemini modules to consistently (1) attempt deserialization into the provider's known typed form, and (2) fall back to wrapping the original JSON as a text part. Each provider now uses a dedicated helper (`chat_completion_content_part_from_any` / `content_item_from_any` / `message_item_from_any` / `content_block_from_any` / `part_from_any`) and explicitly filters out the catch-all `Any` variant so a round-trip through `Any` never silently replaces a typed part. This guarantees that structured content (e.g. `input_image`, `web_search_call`, `tool_use`) round-trips with full semantics, while truly unknown JSON is preserved verbatim as a stringified text payload for downstream inspection.

### Added — anda_engine v0.12.27

- **`to_message_inputs_only_preserves_known_any_content_parts`** — New test in `anda_engine/src/model/openai.rs` asserts that `ContentPart::Any` JSON with a known `image_url` shape is preserved as-is, while unknown shapes are wrapped as text with the original JSON serialized inside.
- **`message_into_only_preserves_known_any_items`** — New test in `anda_engine/src/model/openai/types.rs` covers both `ContentItem` and `MessageItem` deserialization paths: a known `input_image` stays in the user message content, a known `web_search_call` becomes its own `MessageItem`, and an unknown shape falls back to a text payload carrying the original JSON.
- **Anthropic `ContentBlock::Any` test coverage** — Extended the existing `content_part_into_preserves_anthropic_specific_variants` test in `anda_engine/src/model/anthropic/types.rs` with three additional cases: raw `text` blocks round-trip as `ContentBlock::Text`, unknown `type` values fall back to text-wrapped JSON, and malformed `tool_use` shapes also fall back to text-wrapped JSON instead of erroring.
- **`content_part_any_only_preserves_known_gemini_parts`** — New test in `anda_engine/src/model/gemini/types.rs` covers the same fallback contract for Gemini parts: raw `text` shapes become `PartKind::Text`, unknown shapes are wrapped as text, and malformed `functionCall` shapes fall back to text-wrapped JSON.

## [0.12.26] — 2026-06-01

### Changed — anda_engine v0.12.26

- **`Models::resolve()` simplified fallback chain** — Removed the deprecated `fallback_model()` path and the arbitrary-first-model extraction from `resolve()`. The resolution order is now: exact label match → `get_model()` (which returns the primary model, or the first available if none is configured). This eliminates a code path that could silently return an unexpected model and aligns `resolve()` with the well-tested `get_model()` behavior. Updated doc comment and test expectation accordingly.

## [0.12.25] — 2026-05-31

### Added — anda_core v0.12.5

- **`ContentPart::any_from` and `any_into` helpers** — New typed constructors for `ContentPart::Any`: `any_from::<T>(ty, val)` builds a `ContentPart::Any` JSON value with a `"type"` field, and `any_into::<T>(ty)` deserializes it back, verifying the type tag and returning `Err(self)` on mismatch. Enables type-safe round-trips for structured JSON content (e.g. `Resource`) without routing through specific provider types.

- **`test_content_part_any_from_and_any_into_resource`** — New test in `anda_core/src/model.rs` validates round-trip serialization of `Resource` through `ContentPart::Any`, covering all fields including `blob` (base64), `metadata`, `uri`, and `mime_type`. Also verifies type mismatch returns `Err` and non-`Resource` input is rejected.

## [0.12.25] — 2026-05-31

### Added — anda_engine v0.12.25

- **Model-aware subagent manager descriptions** — `SubAgentManager` now carries a `models: Vec<String>` field set via `with_models()` builder. When models are registered, the tool description dynamically includes available model names for routing decisions (e.g. "This manager supports the following models for routing decisions: flash, pro, primary.").
- **`resources` field in conversation storage** — `Conversation::to_fields()` now persists the `resources` field alongside `messages` and `artifacts`. Previously, resources attached to a conversation were lost on save/load cycles.
- **`ContentPart::Any` Resource serde test** — New `test_content_part_any_supports_resource_serde` validates round-trip serialization of `Resource` through `ContentPart::Any`, covering all fields including `blob`, `metadata`, `uri`, and `mime_type`.

### Changed — anda_engine v0.12.25

- **Effort enum values: `xhigh` → `max`** — SubAgent and SubAgentManager function schemas now list `"max"` instead of `"xhigh"` in the `effort` enum, aligning with the `ModelEffort` rename in v0.12.24. Updated in three locations (subagent schema, manager schema, test assertion).
- **`model_names()` doc comment** — Added documentation for the `Models::model_names()` method describing its return value.
- **`ModelConfig.effort` doc: `xhigh` → `max`** — Documentation string updated to reflect the renamed variant.

## [0.12.24] — 2026-05-29

### Added — anda_core v0.12.4

- **`ModelEffort` — provider-agnostic reasoning/thinking effort** — New `ModelEffort` enum (`minimal`, `low`, `medium`, `high`, `max`) provides a single vocabulary for reasoning effort across all model providers. Includes `as_str()` and `Display` impl.
- **Per-request effort on `CompletionRequest`** — `CompletionRequest` now carries an `effort: Option<ModelEffort>` field, enabling callers (including subagents) to select reasoning effort per completion rather than only at model-config time.

### Added — anda_engine v0.12.24

- **Tool error propagation via `is_error`** — `ToolOutput` and `ContentPart::ToolOutput` now carry `is_error: Option<bool>`. Agent runner sets `is_error: Some(true)` on tool/agent execution failures and tool-not-found paths. Memory tools signal Nexus errors through the same flag. Anthropic passes `is_error` through to `ToolResult`; Gemini routes errors to `FunctionResponseValue.error` instead of the normal `output` field.
- **Subagent catalog, operation routing, and runtime state preservation** — `SubAgentManager` gains `operation` field (`"upsert"` or `"list"`), `catalog()` method returning all registered subagents with metadata, and `preserve_runtime_state()` to keep active subsessions alive across upserts. `SubAgent::definition_description()` now enriches descriptions with tags, allowed tools, output schema info, active sessions, default model, and default effort. Session-mode responses include session ID.
- **Provider-agnostic `ModelEffort` with per-request application** — `ModelEffort` (promoted from engine to core) maps to each provider's native reasoning level: Anthropic `OutputEffort` (saturates at `Max`), Gemini `ThinkingLevel` (saturates at `High`), OpenAI `ReasoningEffort` (now includes `Max` variant), and OpenAI Responses v2 `Reasoning`. All three providers now apply `req.effort` during `completion()`, not only model-config defaults.
- **Subagent model and effort selection** — `SubAgent` and `SubAgentArgs` gain `model: String` and `effort: Option<ModelEffort>` fields, with case-insensitive deserialization (accepts `"HIGH"`, `"max"`, `null`, empty string, or JSON value). Session compaction preserves model and effort across context cycles. Both `SubAgent` and `SubAgentManager` function schemas expose `model` and `effort` parameters.
- **Model label resolution** — `AgentCtx` now resolves models by label via `models.resolve(label)` instead of the previous name-only lookup. `CompletionRunner` gains `set_model()` and `set_effort()` methods for mid-run model/effort switching.
- **Subagent tests for model/effort selection** — New `subagent_run_allows_model_and_effort_selection` test verifies end-to-end flow: subagent receives a model label, the correct `CompletionRequest` reaches the provider with the expected model and effort. New `RecordingRequestCompleter` mock captures `CompletionRequest` for assertion. All existing tests updated for new fields.

### Changed — anda_engine v0.12.24

- **Anthropic error messages include model name** — Non-200 HTTP responses now include the model name in the error text for easier debugging of provider failures.


## [0.12.23] — 2026-05-27

### Changed — anda_engine v0.12.23

- **Shell foreground commands auto-move to background after 42 seconds** — Previously native foreground commands were waited on with `wait_with_output()`, and if they exceeded `SHELL_TIMEOUT_SECS` (180s) the tool would return a timeout error. Now `execute_command` uses `tokio::select!` with `SHELL_AUTO_BACKGROUND_SECS` (42s): if the process hasn't finished within that window, it is transparently moved to background execution with a "moved to background" message containing the task ID for hook delivery. The `RunningProcess` struct now bundles child/stdout/stderr/readers immediately after spawn, and extracted `finalize_process_output()` is shared between the foreground-completed and background-completed paths. Added test `execute_auto_moves_long_running_foreground_to_background` verifying the auto-transition and hook delivery.
- **Comprehensive sub-session compaction tests** — New mock infrastructure (`UsageCompleter`, `RecordingCompactionCompleter`, `RecordingAgentHook`, `request_text` helper) and three new tests: `needs_compaction_respects_usage_threshold` verifies compaction triggers at exactly 100,000 input tokens; `needs_compaction_triggers_at_turn_limit` verifies compaction after `MAX_TURNS_TO_COMPACT` turns with unbound runner; `subsession_runner_compacts_context_and_continues_from_handoff` exercises the full compaction flow — context is compacted into a single assistant handoff message, instructions/role/output_schema are preserved, and subsequent follow-up input resumes correctly from the compacted history.


## [0.12.21] — 2026-05-25

### Changed — anda_engine v0.12.21

- **SubAgentManager now isolates subagent storage under `subagents/` prefix** — Previously `store_list` was called with no prefix filter (loading everything from root), and subagents were stored directly at the root level. Now both `load()` and `save()` use `store_prefix() = "subagents"` as the listing and storage path prefix, separating subagent data from other store entries. Added test verifying that a legacy agent stored at root level is not loaded by `SubAgentManager::load`.
- **Fixed `ContentPart::FileData` conversion for non-remote URIs** — In Anthropic, Gemini, and OpenAI (both `to_message_inputs` and `message_into`), `ContentPart::FileData` now guards on `file_uri` starting with `data:` or `https://` before converting to provider-native blocks (Image, Document, File, Video, Audio). Non-remote URIs like `file://` now correctly fall back to JSON-serialized text content blocks instead of being sent as inaccessible document/file references to the API. Added tests across all four conversion paths.

## [0.12.20] — 2026-05-21

### Changed — anda_engine v0.12.20

- **Removed `model` field from `AgentCtx`** — The cached default model instance is no longer stored on the context. Instead `models.get_model()` is called inline at the point of use (`completion_runner`, `model_name()`). Child and spawn agent contexts no longer need to clone the parent's model. This reduces cloning and simplifies the context lifecycle.
- **Compaction threshold raised to 80% of context window** — `needs_compaction` now triggers at 80% (`saturating_mul(8) / 10`) instead of 50% (`saturating_div(2)`), with the minimum raised from 50,000 to 100,000 tokens. Fewer unnecessary compactions, preserving more context across turns.


## [0.12.19] — 2026-05-21

### Fixed — anda_core v0.12.3, anda_engine v0.12.19

- **`normalize_schema_object` now correctly handles `["object", "null"]` types** — The `is_object` check previously only matched `"type": "object"`, missing nullable object schemas like `["object", "null"]`. New `schema_type_contains_object()` helper checks both string and array type values. Also, `additionalProperties: false` is now set for *all* object schemas, not only those with `properties` — so `{"type": ["object", "null"]}` without properties also gets it. Added test `test_normalize_strict_schema_handles_nullable_objects` covering both cases.
- **OpenAI Responses v2: filter empty Reasoning and ItemReference from history** — `normalize_message_item` now returns `None` for `Reasoning` items with `encrypted_content: None` (empty reasoning blocks) and `ItemReference` items (by-ID references with no content). Previously these passed through as-is and could cause provider errors. `raw_history_into` now chains `.filter_map(normalize_message_item)` on the core `Message` history path too, which previously bypassed normalization.

### Changed — anda_engine v0.12.19

- **`output_schema` in subagents_manager now uses `["string", "null"]`** — The JSON schema is submitted as a JSON-encoded string instead of a nested object. This avoids the structured-output problems that nested object schemas cause with strict function-calling providers. Added `deserialize_optional_json_schema` custom deserializer that accepts JSON strings (parsed inline) or `null`. New tests: `subagents_manager_definition_uses_strict_safe_output_schema`, `subagents_manager_args_accept_json_encoded_output_schema`.


## [0.12.18] — 2026-05-21

### Changed — anda_engine v0.12.18

- **Stripped non-essential JSON Schema keywords from tool definitions** — Removed `default`, `uniqueItems`, `minLength`, `minLength` (input/description), `pattern`, `maxLength`, `minimum`, and `maximum` from all tool schemas. These keywords add unnecessary strictness in a world where `required` already lists every field; some providers may also fail to handle them correctly in strict function calling mode. Affected tools:
  - `tools_search` / `tools_select`: removed `default` and `uniqueItems`
  - `skills_manager`: removed `pattern`, `maxLength`
  - `list_conversations` / `search_conversations`: removed `default`, `minimum`, `maximum`
  - `subagents_manager` / `subagent`: removed `minLength`, `uniqueItems`, `pattern`


## [0.12.17] — 2026-05-21

### Changed — anda_engine v0.12.17

- **Removed automatic `max_output_tokens` injection** — `CompletionRunner` no longer applies `self.model.max_output` as a default `max_output_tokens` when the request omits it. This responsibility now belongs to each provider's completion method or the caller, giving more precise control over token limits per request. Previously the runner automatically set `req.max_output_tokens = Some(self.model.max_output)` for any request without an explicit limit.


## [0.12.16] — 2026-05-21

### Added — anda_core v0.12.2, anda_engine v0.12.16

- **Strict schema normalization** — `normalize_strict_schema()` in `anda_core::json` recursively normalizes JSON schemas for strict function calling: rewrites `required` to contain every key in `properties`, defaults `additionalProperties` to `false`, and traverses nested schemas (`items`, `$defs`, `allOf`/`anyOf`/`oneOf`, `if`/`then`/`else`, `not`, etc.). New `FunctionDefinition::normalize_strict_parameters()` method calls this normalization when `strict: true`. All tool definitions (`ToolDefinition::from`, `CompletionRequest` tool-building) now go through this normalization path — ensuring consistent strict schemas regardless of provider.
- **Model-level stream control** — Added `stream: bool` field to `ModelConfig` and `with_stream()` builder methods on all completion model types: Anthropic (`CreateMessageParams.stream`), Gemini (`GenerateContentRequest.stream`), OpenAI Chat Completions (`ChatCompletionRequest.stream`), and OpenAI Responses v2 (`CompletionRequest.stream`). Each provider's `CompletionModel::new()` now reads `self.stream` and sets the default request accordingly.
- **OpenAI Responses v2 defaults to streaming** — The v2 model now defaults to `stream: true` with `store: false` for stateless requests. The original `reasoning` default parameter was removed.
- **OpenAI Responses v2 message history normalization** — New `raw_history_into()` converts raw `Json` history into `MessageItem` values with proper role-aware normalization: core `Message` history is unwrapped via `message_into()`, legacy Responses `MessageItem::Message` records are normalized through `normalize_message_item()`, and unrecognized values pass through as `Any`. Assistant messages in history now use `output_text` content type (required by Responses API) rather than plain `text`. Content-type filtering distinguishes input (`text`) from output (`output_text`) based on role.
- **OpenAI Responses v2 stream aggregation** — `responses_response_from_stream_events()` now handles `ResponseOutputItemDone` events: when the final response has an empty `output` array, output items collected from `done` events fill in via `output_index` ordering.

### Changed — anda_core v0.12.2, anda_engine v0.12.16

- **All tool schemas now declare complete `required` arrays** — Every tool definition schema lists all property keys in `required` with `additionalProperties: false`, including optional fields. Previously many tools omitted optional fields from `required`, which triggers strict schema validation failures with providers. Affected tools: `tools_search`, `tools_select`, `list_conversations`, `search_conversations`, `memory`, `note`, `todo`, `subagents_manager`, `read_file`, `write_file`, `edit_file`, `search_file`, `shell`, `extractor` (SubmitTool).
- **`anyOf` avoidance in tool schemas** — `note` tool now uses `"type": ["string", "null"]` with inline `enum` containing `null` instead of `anyOf`-based union schemas. `todo` and `subagents_manager` tools similarly use `["array", "null"]` and `["object", "null"]` type arrays. These patterns avoid `anyOf` which many providers (especially Anthropic) reject in strict mode.
- **Memory tool gets hand-written schema** — Replaced auto-generated schema with a manually crafted one that lists all 9 operation types in a flat `type` enum with all optional fields explicit, avoiding the nested discriminator pattern that produced invalid schemas across providers.
- **Gemini default request** — Removed the hard-coded `top_p: 0.95` default from `CompletionModel::new()`.
- **Subagent `session` description** — Changed from "Omit session" to "Leave session empty" to match strict schema nullability.
- **All tests updated** — Test function definitions, schema assertions, and expected values updated to include full `required` arrays and `additionalProperties: false`.


## [0.12.15] — 2026-05-21

### Added — anda_engine v0.12.15

- **SSE streaming support across all backends** — Anthropic, Gemini, OpenAI Chat Completions, and OpenAI Responses backends now support true SSE (Server-Sent Events) streaming. Previously responses were always read as complete JSON payloads; now streaming responses are parsed and aggregated from `text/event-stream` chunks. Added generic `read_sse_json_events<T>()` in `model.rs` with proper line-buffering, UTF-8 validation, and multi-line `data:` event concatenation.
  - **Anthropic**: `response_from_stream_events()` reconstructs `CreateMessageResponse` from stream events — handles MessageStart, ContentBlock{Start,Delta,Stop}, MessageDelta, and MessageStop. Content blocks (text, thinking, tool_use, server_tool_use) are incrementally assembled; cursor deltas accumulate non-zero fields. Streaming enabled when request `stream=true`.
  - **Gemini**: `response_from_stream_chunks()` aggregates `:streamGenerateContent?alt=sse` SSE responses. Candidates with the same index are merged — text parts are concatenated, finish_reason/safety_ratings/citation_metadata from later chunks overwrite earlier, and non-empty fields from later chunks supersede earlier defaults.
  - **OpenAI Chat Completions**: `chat_completion_response_from_stream_chunks()` reconstructs `CompletionResponse` from `chat.completion.chunk` SSE events. Stream delta accumulation handles: text content concatenation, content parts extension, tool call incremental assembly (id/type/function/custom), reasoning_content, refusal, function_call, and finish_reason. Tool calls are built via `ToolCallStreamBuilder` with per-index `BTreeMap` tracking.
  - **OpenAI Responses**: `responses_response_from_stream_events()` extracts the most recent response from events — picks up `response.created`/in_progress/completed/failed/incomplete. Parses output after reconstruction.
- All streaming backends set `Accept: text/event-stream` header and conditionally dispatch to streaming vs non-streaming code paths based on request configuration.
- **Gemini: `stream` request flag** — Added `stream: bool` field to `GenerateContentRequest` (serde-skipped, local-only) for choosing the streaming endpoint.
- **Unit tests**: `aggregates_anthropic_stream_events` (text + tool_use with partial JSON), `aggregates_gemini_stream_chunks` (text concatenation across chunks), `aggregates_chat_completion_stream_chunks` (text + streaming tool calls), `aggregates_responses_stream_completed_event` (response.completed extraction).


## [0.12.14] — 2026-05-21

### Added — anda_engine v0.12.14

- **`CompletionRunner::is_idle()`** — New method that reports whether the completion runner has no pending work: prompt, content, documents, steering message, follow-up message, and pending tool calls are all empty. Useful for polling-based control loops that need to detect when the agent is ready for new input.

### Changed — anda_engine v0.12.14

- **Subagent `tags` description refined** — Added concrete examples (`image`, `text`, `audio`) to the `tags` field description in the `subagents_manager` tool schema.

## [0.12.13] — 2026-05-19

### Changed — anda_engine v0.12.13

- **Error messages include model name across all backends** — HTTP request send failures and response body read failures in Anthropic, Gemini, OpenAI, and OpenAI V2 backends now include the model name in the error message. Previously these relied on bare `?` propagation, yielding opaque errors like "connection refused" with no model context, making multi-model debugging difficult.
- **OpenAI: simplified model reference** — Removed redundant local `let model = self.model.clone()`. Logging and errors now use `r.model` consistently.

## [0.12.12] — 2026-05-17

### Fixed — anda_engine v0.12.12

- **OpenAI: Multi-tool-output splitting** — `to_message_input` refactored to `to_message_inputs`, now returning `Vec<MessageInput>`. When a single `Message` contains multiple `ToolOutput` content parts, each output is emitted as a separate tool-role message with its own `tool_call_id`. This fixes the case where multiple tool outputs were flattened into one message, which violated the OpenAI API contract (one tool message per call).
- **OpenAI: Non-text content part preservation** — `MessageOutput → Message` conversion rewritten. The old `text()` filter that discarded non-text parts is replaced with `chat_completion_content_into_parts` / `chat_completion_content_part_into` / `file_data_content_part` pipeline. Image URLs, input audio, files, video URLs, and refusals now survive the round-trip conversion as proper `ContentPart` variants instead of being silently dropped.
- **`MessageInput` now carries `name`** — `name: msg.name.clone()` is propagated to `MessageInput` during conversion.

## [0.12.11] — 2026-05-17

### Fixed — anda_engine v0.12.11

- **`implicit_context` injection timing** — When pending tool calls are executed mid-turn, the implicit context is no longer injected on the same request. Instead it's deferred to the next user-facing turn, preventing implicit context from being consumed on tool-result rounds where it would have no effect.
- **OpenAI: DeepSeek `tool_choice` compatibility** — Skip setting `tool_choice` for models whose name starts with `"deepseek"`, as DeepSeek's API does not support this parameter and returns errors when it is present.

## [0.12.10] — 2026-05-17

### Changed — anda_engine v0.12.10

- **`implicit_context` on `CompletionRunner`** — new `implicit_context(&mut self, message: Message)` method stores a context message that is automatically injected into the next request's `chat_history` and consumed on use. This enables steering/follow-up messages to pass contextual information without manual request manipulation.
- **Prompt ordering fix across all providers** — system prompt now uses `content.insert(0, req.prompt.into())` instead of `content.push()`, ensuring the prompt appears before the conversation content in all three model backends (Anthropic, Gemini, OpenAI, and OpenAI v2). This fixes cases where the prompt was appended after content instead of leading.
- **OpenAI: `model` field fix** — `CompletionModel` now explicitly sets `r.model = self.model.clone()` on each request, fixing a bug where the model field in the cloned `default_request` was not being updated to the current model selection.

## [0.12.8] — 2026-05-16

### Changed — anda_core v0.12.1

- **`Resource → ContentPart` conversion now uses `TryFrom` with MIME detection** — `impl TryFrom<Resource> for ContentPart` replaces the infallible `From` impl. Binary blobs now use `infer2` to detect the actual MIME type from bytes instead of defaulting to `application/octet-stream`. Resources with neither blob nor URI return `Err(res)` instead of serializing to text.
- **`inline_data_from_data_url`** — New helper to parse data URLs (`data:[<mime>][;base64],<data>`) and plain base64 strings into `(ByteBufB64, mime_type)` pairs. Handles both base64-encoded and percent-encoded payloads.
- **`decode_percent_encoded_bytes`** — Internal helper for percent-decoding URL-encoded byte sequences.
- **Comprehensive test suite for `anda_core::model`** — Added 10 test functions covering `AgentInput`, `ToolInput`, `PromptCommand`, `AgentOutput::into_tool_output`, data URL round-trips, `ContentPart::try_from(Resource)` edge cases, `RequestMeta`, `Usage::accumulate` overflow, `FunctionDefinition`, `Document`/`Documents`, and `Message` deserialization.

### Changed — anda_engine v0.12.8

- **Anthropic: Extended API surface** — Full support for the latest Anthropic Messages API: `SystemPrompt` enum (string or content blocks), `CacheControlEphemeral` for prompt caching, `OutputConfig` with `OutputEffort` and `JsonOutputFormat`, structured `StopDetails::Refusal`, `ToolChoice` constructors (`auto()`, `any()`, `tool()`), `ThinkingDisplay` and `ThinkingType::Adaptive/Disabled`, extended `ContentBlock` variants (document, search_result, server_tool_use, web_search/fetch results, code execution results, container_upload), `ToolResultContent` as text-or-blocks, `CitationsConfig`/`TextCitation`, `UsageServiceTier`/`CacheCreation`/`ServerToolUsage` on `Usage`, `Container` in responses.
- **Gemini: Extended API surface** — `SafetySetting`/`HarmBlockThreshold`, `cached_content`/`service_tier`/`store` on `GenerateContentRequest`, `ModelStatus`, `GroundingAttribution`/`GroundingMetadata` with rich chunk types (web, images, maps, retrieved context), `LogprobsResult`, `UrlContextMetadata`, `SpeechConfig`/`VoiceConfig`, `ImageConfig`/`MediaResolution`, `Modality` enum, `response_json_schema` fields, `seed`, `enable_enhanced_civic_answers`, extended `FinishReason` variants. `SatisfyRating` renamed to `SafetyRating` with backward-compatible alias. Fixed `satefy_ratings` typo with serde alias.
- **OpenAI: Full `ChatCompletionRequest` type** — Structured request builder replacing ad-hoc JSON construction. Supports `audio`, `modalities`, `reasoning_effort`, `response_format` (text/json_object/json_schema), `service_tier`, `stop` (string or array), `stream_options`, `tool_choice` (none/auto/required/allowed_tools/function/custom), `verbosity`, `web_search_options`, `prediction`, `prompt_cache_key`/`prompt_cache_retention`, `logprobs`/`top_logprobs`, `safety_identifier`, `seed`, `store`, `metadata`, `user`, `parallel_tool_calls`, `frequency_penalty`/`presence_penalty`.
- **OpenAI: Content types and refusal handling** — `ChatCompletionMessageContent` supports text, content parts (text/image_url/input_audio/file/refusal), and `null` deserialization. Refusal detection from both legacy `refusal` field and content-block refusals. `MessageOutput` now uses `ChatCompletionMessageContent` and provides `has_output()`/`has_refusal()` helpers.
- **OpenAI: Custom tool support** — `ToolDefinition` as enum with `Function` and `Custom` variants. `CustomToolDefinition` with text/grammar format. `ToolCallOutput` supports both `function` and `custom` call types. `CustomToolCall` with raw `input` string.
- **OpenAI: Tool calls extracted to `tool_calls` field** — Assistant messages now serialize tool calls in a top-level `tool_calls` array alongside `content`, matching the OpenAI API shape.
- **OpenAI: Usage details** — `CompletionTokensDetails` (reasoning_tokens, audio_tokens, accepted/rejected_prediction_tokens) and `PromptTokensDetails` (audio_tokens).
- **OpenAI: Media type routing** — File/image/audio/video `ContentPart` items now route to the correct content block type (`image_url`, `input_audio`, `video_url`, `file`) based on MIME type.
- **OpenAI Responses API v2: Extended types** — `StreamEvent` enum with 11 event types. `MessageItem` expanded with file_search_call, computer_call, web_search_call, tool_search, compaction, image_generation, code_interpreter, shell calls, apply_patch, MCP calls, custom tools. `ToolDefinition` expanded with file_search, computer, web_search, MCP, code_interpreter, image_generation, local_shell, shell, custom, namespace, tool_search, apply_patch. `ContextManagement` and `ResponseConversation` for conversation state.
- **Model routing** — OpenAI models starting with `gpt` now use `completion_model_v2` (Chat Completions API), while non-gpt models use the standard Responses API path.
- **SubAgent: `From` → `TryFrom` migration** — SubAgent resource-to-ContentPart conversion updated to use the new `TryFrom` impl.

## [0.12.7] — 2026-05-15

### Changed

- **Case-insensitive model label lookup** — `Models::get()`, `Models::contains()`, and `Models::resolve()` now normalize labels with `to_ascii_lowercase()` before lookup. Labels are stored lowercase in `inner_set`. This means `get("GPT-4")` and `get("gpt-4")` resolve to the same model.
- **Model names auto-registered as labels** — `inner_set()` now appends `model_name.to_ascii_lowercase()` to the label set. A model with `model_name = "primary"` is now findable via `get("primary")`, removing the need for manual label aliasing.

## [0.12.6] — 2026-05-12

### Removed

- **`CompletionRunner` tools_select auto-loading** — the completion loop no longer parses `tools_select` results to automatically inject selected tool definitions into the next turn. Tool selection/loading is now handled externally by the calling context, simplifying the runner's responsibility and removing ~25 lines of specialized handling code.
- **`is_tools_select_name()`** helper removed — no longer needed after the above simplification.
- **ToolsSelect/ToolsSearch integration tests** — `ToolsSelectFlowCompleter`, `ToolsSelectQueryFlowCompleter`, and `ToolSelectorCompleter` (~280 lines of test infrastructure) removed alongside the auto-loading behavior they tested.

### Changed

- **Relaxed ToolsSelect/ToolsSearch parameter constraints** — `minLength`, `minItems`, and `minimum` constraints removed from JSON schemas. These validations now happen at the implementation level, giving models more flexibility in parameter usage.
- **Optimized `select_requested_names_with_model` serialization** — new `ToolItemRef` struct serializes only `name` + `description` when passing candidates to the selector model, instead of full `FunctionDefinition` (which includes large parameter schemas). Reduces token usage.

### Added

- **`local_date_hour(now_ms: u64) -> Option<String>`** — converts a Unix millisecond timestamp to a local datetime string in `"YYYY-MM-DD HH(AM/PM) ±TZ"` format.
- **Test coverage for `lib.rs` utilities** — `rand_number`, `rfc3339_datetime`, `json_set_unix_ms_timestamp`, `json_convert_rfc3339_timestamp`, and `local_date_hour` now have comprehensive tests.

## [0.12.5] — 2026-05-11

### Features

- **`with_caller()` for context cloning** — `BaseCtx::with_caller(caller)` and `AgentCtx::with_caller(caller)` clone the context with a new caller principal while preserving all extensions and internal state. Useful for sub-operations that execute under a different identity.

### Changed

- **Background progress interval 3s → 5s** — reduces noise for long-running commands (e.g., model inference, large builds). New `NativeRuntime::background_progress_interval()` builder method allows per-runtime customization for environments that need faster or slower tick rates.

## [0.12.4] — 2026-05-11

### Features

- **2D terminal emulation for shell progress** — `TerminalProgressState` upgraded from single-line buffer to full 2D terminal model with multi-line scrolling, cursor row tracking, and dirty-row incremental output. Supports CSI cursor movement (A/B/C/D), absolute positioning (G/H/f), line/screen erase (J/K), and multi-line parallel progress bars — all rewritten lines across rows are reported together per progress tick.
- **Smarter rewrite-mode detection** — `has_rewrite_control()` now only activates rewrite mode for actual terminal-control CSI sequences, not passive styling (colors, decorations). Plain ANSI-styled output stays in line-buffered mode and is emitted on newline boundaries only.

### Refactors

- **Plain progress mode is line-buffered** — non-rewrite output accumulates until `\n`, then emits complete lines via `completed_lines` buffer. No more mid-line fragmentation in plain-text progress.

## [0.12.3] — 2026-05-11

### Features

- **Background shell progress hooks** — `on_background_progress()` hook delivers incremental stdout/stderr every ~3 seconds while a background command runs. `TerminalProgressState` normalizes rewritten terminal lines (`\r`, `\b`, ANSI `ESC[K`) to their latest visible text, so the model sees clean output instead of raw control characters. UTF-8 boundary-safe chunking via `complete_utf8_prefix_len()` prevents splitting multi-byte sequences across progress deliveries.
- **`insecure()` mode for NativeRuntime** — builder option to skip `env_clear()`, allowing the shell to inherit host environment variables.

### Refactors

- **`execute_command()` extracted as public method** on `NativeRuntime` — takes a `std::process::Command` directly, enabling non-shell invocations through the native runtime. `build_shell_command()` now returns `std::process::Command` (decoupled from tokio).
- **`ToolsSearch::NAME` and `ToolsSelect::NAME`** added as `pub const` — used in `name()` instead of raw constants for cleaner code.

### Breaking Changes

- **`Executor::temp_dir()` removed** from the trait — no longer part of the public executor interface. `NativeRuntime` gains a `temp_dir()` builder method instead.
- **Native executor `name()` changed** from `"native_shell"` to `"shell"`.
- **`tools_search` removed** from `DEFAULT_SKILL_TOOLS` — skill agents no longer receive `tools_search` by default.

### Housekeeping

- **Remove deprecated tests** — test modules removed from `extractor.rs` (106 lines) and `google.rs` (52 lines), both already marked `#[deprecated]` since 0.12.0.

## [0.12.2] — 2026-05-09

### Features

- **Multi-workspace file tools** — `ReadFileTool`, `EditFileTool`, `WriteFileTool`, and `SearchFileTool` now accept `workspaces: Vec<PathBuf>` instead of a single workspace. Context meta `workspace`/`workspaces` fields take precedence over defaults, with automatic fallback. `SearchFileTool` iterates all workspaces and merges results. New `with_workspaces()` constructor for multi-default setups.

### Improvements

- **Workspace-scoped error messages** — all filesystem tool errors now include workspace, requested path, and resolved path context for faster debugging. New `workspace_access_error()` helper produces consistent "not accessible from any configured workspace" messages.
- **Note storage limits doubled** — `NOTE_CHAR_LIMIT` 8 KB → 16 KB, `NOTE_MATCH_PREVIEW_LIMIT` 80 → 120 chars.

### Fixes

- **Skill cache** now correctly updates on successful load (`write().entry().insert_entry`).

### Deprecations

- **`google`, `extractor`** modules marked `#[deprecated(since = "0.12.0")]` — will be removed in a future release.

## [0.12.1] — 2026-05-08

### Refactors

- **Unified `ToolsOutput`** — `ToolsSearchOutput` and `ToolsSelectOutput` merged into single `ToolsOutput` struct with `Vec<FunctionDefinition>` (full tool definitions, not just name+description), so the model can invoke tools immediately after `tools_search`. Intermediate `ToolsSearchItem` struct removed; `rank_search_items` returns names directly, definitions resolved at output boundary. Default limits: search 0→10, select 0→5 with explicit `MAX` caps.

### Fixes

- **Agent failures as errors** — when an agent call fails, return a `ToolOutput` error instead of breaking the conversation, allowing the LLM to correct and recover.

### Breaking Changes

- **`workspace` field removed from `ExecArgs`** — shell tool no longer accepts a `workspace` parameter; commands always execute in the runtime's workspace directory. `join_current_dir` helper removed.

### Fixes

- **Shell spawn failures** now return structured `ExecOutput` errors instead of propagating as Rust errors, so the model can see and respond to command execution failures.

### Internal

- **Native shell runtime** renamed from `"native"` to `"native_shell"`; background shell tasks now return an immediate `task_id` output so callers can track long-running commands.

## [0.12.0] — 2026-05-07

### Breaking Changes

- **SubAgent extracted as top-level module** — `SubAgent`/`SubAgentSet`/`SubAgentManager` moved from `context::subagent` to top-level `crate::subagent`. `SubAgentManager` is now an Agent (via `AgentSet`) instead of a Tool, enabling proper `AgentHook` callbacks for subagent lifecycle. Session-based background execution with `SubSessions` tracking, idle timeout (10 min), background task wait (1 hr), and automatic compaction at 81+ turns.
- **SkillManager reduced to read-only inspector** — `SkillManager` no longer supports `create`/`patch`/`edit`/`delete`/`write_file`/`remove_file` disk management operations. Skills are now created and updated by editing files directly on disk via shell or file tools. `SkillArgs` simplified to `{name}` only (with `deny_unknown_fields`); output is now `SkillContentOutput`.
- **Rename `work_dir` → `workspace`** across all tools, runtimes, and context metadata. Shell output field changed from `work_dir` to `workspace`.
- **Drop legacy DeepSeek client** — replaced by multi-label model support (`labels: Vec<String>` on `ModelConfig`).
- **Remove sandbox feature and `boxlite` dependency** from engine crate. Shell extension now always uses `NativeRuntime`.
- **Rename Dyn traits** — `AgentDyn` → `DynAgent`, `ToolDyn` → `DynTool`; internal storage switched from `Box<dyn ...>` to `Arc<dyn ...>`.
- **Rename `step` → `turns`** in `CompletionRunner` API.
- **Remove `CompletionHook`** — superseded by `AgentHook` with background support.
- **Remove `prune_raw_history` pipeline** — trait method, 4 provider impls, `CompletionRunner` method, and `pruned_placeholder` helper all deprecated since 0.11.0 have been removed.
- **`WorkDir` renamed to `Workspace`** in `ExecArgs`, `ExecOutput`, and context metadata.
- **`MAX_OUTPUT_BYTES`** increased from 128 KB to 256 KB.
- **`ModelConfig::model()`** now returns `Result` instead of silently producing a `not_implemented` model.
- **Remove deprecated `evaluate_tokens`** (anda_core) and `build_model` (anda_engine) — superseded by `estimate_tokens` and `model()`.

### Features

- **Downcast support** — `as_any()`/`into_any()` on `DynAgent` and `DynTool` traits with `downcast_ref`/`downcast` convenience methods for type-safe concrete type recovery.
- **`PromptCommand` enum** with `From<String>` for slash-command parsing (`/ping`, `/command ...`, plain text).
- **`AgentOutput` improvements** — `thinking` → `thoughts` rename; new `session` field; `PartialAgentOutput` struct; `into_tool_output()` method.
- **`CustomEnv` with auto-inject** — shell tool environment variables can be marked as `default` for automatic injection; key/description metadata exposed to model providers without leaking values.
- **Per-agent model selection** — sub-agents and call-agents use their configured model from the `models` map, falling back to the parent model.
- **`CompletionRunner::finalize`** — clean unbound runner completion; steering interrupt; follow-up message consolidation.
- **`ConversationDelta`** — offset-based incremental conversation fetching for large conversations.
- **Per-tool usage tracking** — `AgentOutput` and `CompletionRunner` track per-tool `Usage`; `ToolOutput.tools_usage` field.
- **Multi-label model support** — `Models` now supports `HashMap<String, Vec<Model>>`, allowing multiple models per label.
- **Tool call statistics** — per-tool call counts tracked in `CompletionRunner.tool_call_stats`.
- **Safe env vars for native shell** — `SAFE_ENV_VARS` whitelist passes only functional host env vars (`PATH`, `HOME`, `TERM`, …) to shell commands, never secrets.
- **Prefix constants** — `REMOTE_TOOL_PREFIX` (`RT_`), `REMOTE_AGENT_PREFIX` (`RA_`), `SUB_AGENT_PREFIX` (`SA_`) defined as constants; prefixing centralized at `AgentCtx::definitions()`.
- **`CompletionRunner::unbound` builder** — enables unconstrained completion execution.
- **`EngineRef` for late binding** — switch to `Arc<Engine>` throughout.
- **Per-agent storage and `output_schema`** for `SubAgent` — refactored from single CBOR to individual files per agent.
- **Multiple steering/follow-up messages queue** — `Vec<String>` and `VecDeque<String>` replace single `Option<String>`.
- **Batch conversation retrieval** — `batch_get_conversations` for efficient multi-conversation loading.
- **Persistent note tool** — agent-scoped durable notes with add/replace/remove operations.
- **Todo tool** — session-scoped task list shared with subagents.
- **Strict mode** enabled for all tool definitions (`strict: Some(true)`).
- **`Conversation` extra field** — extensible metadata for conversations.
- **`Thoughts` method on `Message`** — extract reasoning content.
- **`Idle` status** added to `ConversationStatus`.
- **CWT verification** support in engine-server.

### Refactors

- **Subagent module** extracted from context to top-level with session-based background execution.
- **Hook system** enhanced — `PrefixedId`, `on_background_progress`, `ToolBackgroundHook`, `DynToolJsonHook`.
- **Model API simplified** — `set_model_by` → `set`, `get_model_by` → `get` + `resolve`; `Models::from_configs` helper.
- **Remote dispatch** — now checks function registration, prefers longest matching handle.
- **Native shell runtime** — removed shell detection infrastructure; always uses `sh` on Unix, `cmd.exe` on Windows.
- **Defined prefix constants** and centralized prefixing to outer layer.
- **`BaseCtx`** passed as parameter instead of stored in `SubAgentManager`.
- **Tool calls deferred** to next turn.
- **`EngineBuilder::empty()`** made async for proper initialization.
- **Child contexts** clone parent state instead of sharing `Arc<RwLock>`.
- **`SkillFrontmatter` metadata** upgraded from `BTreeMap<String, String>` to `BTreeMap<String, Json>`.
- **Tool errors** use JSON `{"error": "..."}` formatting instead of plain strings.

### Fixes

- Auto-fill `max_output_tokens` from model config when caller doesn't set it.
- Cap Anthropic max tokens at 64,000.
- Normalize model labels to lowercase for consistent matching.
- `cache_store_set` fix — cache was deleted instead of updated after `store_put`.
- `null` content in Message deserialization handled correctly.
- `select_resources` preserves ordering with `O(n)` single-pass algorithm.
- `SO_REUSEADDR` fallback for platforms without `SO_REUSEPORT`.
- `ToolsSelectOutput` deserialization fixed for nested `.content` field.
- All 4 model response parsers now consistently set `output.thoughts`.

### Documentation

- **anda_engine** — all module docs, README rewritten with Install, Quick Start, Core Concepts, Feature Flags, Security.
- **anda_core** — all module docs, README rewritten with module map, concepts, and minimal `Tool` implementation example.

### Internal / Chores

- Upgrade `infer` → `infer2` (v0.21), `boxlite` → 0.9 (crates.io).
- Search file default limit set to 1000.
- `DEFAULT_SKILL_TOOLS` expanded to include `todo`, `tools_search`, `tools_select`.
- User name max length relaxed from 32 to 96 chars.
- `SkillFrontmatter` gains `extra` field with `serde(flatten)` for forward compatibility.
