# Changelog

All notable changes to the Anda project will be documented in this file.

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
