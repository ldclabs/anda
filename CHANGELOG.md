# Changelog

All notable changes to the Anda project will be documented in this file.



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

