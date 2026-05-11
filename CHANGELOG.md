# Changelog

All notable changes to the Anda project will be documented in this file.

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
