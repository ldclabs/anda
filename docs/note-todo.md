# Notes and session tasks

`anda_engine::extension::note` and `anda_engine::extension::todo` are small,
optional local tools. Register them like other tools; neither requires a model
provider, a separate database, or an event service.

## Todo contract

`todo` supports `read`, `set`, and `update`. Tasks keep stable IDs and one of
`pending`, `in_progress`, `completed`, or `cancelled`. Operation and status names
are trimmed and normalized to lowercase. Writes validate the complete batch
before changing the list:

- `set` replaces the list and requires `items`; only `items: []` clears it.
- `update` patches existing IDs, preserving fields supplied as null. A new ID
  requires nonempty content and defaults to `pending` when status is null.
- Empty IDs/content, unknown operations/statuses/JSON fields, and invalid limits
  fail explicitly. They do not silently skip tasks or reset statuses.
- All duplicate entries must validate; the last occurrence of an ID wins.
- IDs are at most 128 UTF-8 bytes without control characters. Task content is
  at most 4096 bytes. A batch/list holds at most 256 tasks, and a stored list
  occupies at most 64 KiB as serialized JSON.
- There is no mandatory single `in_progress` invariant: nested agents share
  the session and may work concurrently. Prefer one active task per worker.

```json
{"op":"set","items":[{"id":"implement","content":"Implement the change","status":"in_progress"},{"id":"verify","content":"Run focused tests","status":"pending"}]}
```

```json
{"op":"update","items":[{"id":"implement","content":null,"status":"completed"}],"explanation":"Implementation is ready for validation"}
```

Writes return status counts and an optional explanation (maximum 2048 bytes).
Reads return the full bounded list. Domain failures have `error` and
`ToolOutput.is_error = Some(true)`; the list stays unchanged. No versioned
subscriptions, persistent event log, or background task scheduler is added.
Applications can keep using `TodoToolHook` to observe results. An after-call
hook can inspect `todo_session(ctx).snapshot()` for a current UI view; a
concurrent write can advance that view beyond the observed call.

A `TodoSession` is an in-memory handle shared through context state. The runner
seeds it before creating per-call children. Direct tool callers should install
it on their parent context before creating children. It is not automatically
persisted or restored after process restart.

Successful `CompletionRunner::handoff()` adds an active task snapshot beside
the generated continuation summary. The snapshot contains only pending and
in-progress tasks, at most 32 entries, at most 256 bytes of each description,
and at most 4096 UTF-8 bytes overall. Omitted tasks and shortened descriptions
point the model to `todo read`. Failed handoffs do not inject a snapshot;
repeated handoffs replace the prior window. This is historical task data,
not a new user instruction.

## Note contract

`note` retains its `id + content` storage format and agent isolation. There is
no implicit cross-agent or per-caller sharing change. Existing `set`, `upsert`,
`delete`, host-side `load_notes`, and explicit legacy-store loading remain.
The tool now also supports bounded retrieval:

| Operation | Input | Result |
| --- | --- | --- |
| `read` | Optional exact `ids` | Content in `items` |
| `list` | Optional exact `ids` | IDs, content lengths, and short excerpts in `entries` |
| `search` | Literal `query`, optional exact `ids` | First match per matching note, excerpt, and character positions in `entries` |

```json
{"op":"read","ids":["build_commands"]}
```

```json
{"op":"search","query":"cargo test","limit":10}
```

Search is a case-sensitive literal substring search, not semantic retrieval or
regular-expression matching. It examines the bounded note document; no search
index or additional storage system is created. Results preserve stored order.
Unknown requested IDs are errors, while a search with no matches succeeds with
an empty result. An explicit empty `ids` list selects nothing.

`limit` defaults to 20 and must be 1-100. Serialized query results default to a
16 KiB JSON byte cap, independently of storage capacity. A host can set
`NoteTool::with_response_bytes`, clamped to 2-64 KiB. This counts JSON escaping
and pagination metadata, not just the note text. Storage writes retain the
configurable character limit (default 163840, including IDs and separators),
and are also limited to 2048 items and 1 MiB of encoded item data. New IDs must
be nonempty, at most 128 bytes, and contain no control characters.

A truncated result has `truncated: true` and `next_cursor`. Repeat the same
operation, IDs, and query with that cursor. The limit may change. Cursors bind
to the agent, query, and exact stored content; edits invalidate old cursors,
which return an error asking the caller to restart retrieval.

Even a single large note can span read pages. On each read result,
`offset_chars` is the Unicode scalar offset of the first returned item's
content; later items start at zero. Concatenate fragments by ID in cursor
order. No UTF-8 character is split. A list/search entry has `chars` (total
content length), `excerpt_offset_chars`, and, for search, `match_offset_chars`.
Excerpts are bounded previews, not complete notes.

Reads do not accept mutation `items`; writes do not accept query fields. Invalid
calls preserve the store and return a typed failure. Writes remain compact and
skip storage for unchanged content. Updates through one `NoteTool` instance
and its clones serialize per document. Separate instances/processes are not
coordinated: hosts sharing a store must ensure writer ownership. There is no
conditional-write retry protocol or automatic append-only journal.

## Optional context index

Registering `NoteTool` does not automatically inject notes. To opt in, install a
`NoteContextConfig` on the agent context before creating its runner:

```rust
use anda_engine::{context::AgentCtx, extension::note::NoteContextConfig};

fn enable_note_index(ctx: &AgentCtx) {
    ctx.base.set_state(NoteContextConfig {
        max_bytes: 4096,
        ids: None,
    });
}
```

The runner loads one small index before its first regular model request and
again in a fresh window after successful handoff. It requires the local `note`
tool to be registered and permitted by the runner's callable allowlist. It
never inserts an index between a pending tool call and its response. The index
is preserved in neutral history for model changes. Subsequent turns do not
repeatedly append it.

The index consists of IDs and short escaped excerpts, not an LLM-generated
summary. It treats notes as historical data and points to `note read/search`.
The default budget is 4096 UTF-8 bytes, clamped to 512-8192, with at most 32
entries. Optional `ids` restrict the index to selected notes; unmatched IDs
contribute nothing. There is no automatic relevance ranking or extra inference.

Hosts can use `load_note_summary(ctx, config)` independently. For a complete
host-side snapshot with errors preserved, use `try_load_notes(ctx)`.
`load_notes(ctx)` remains a compatibility wrapper returning `None` on errors.
Opt-in runner injection propagates storage/decode errors before inference;
they are not mistaken for an empty note store.

## Rust API migration

- `TodoStore::set/update` and `TodoSession::set/update` now return
  `Result<Vec<TodoItem>, String>`; handle validation errors.
- `TodoArgs` gains `explanation`; `TodoOutput` gains `explanation` and `error`.
- `NoteArgs` gains `ids`, `query`, `cursor`, and `limit`; `NoteOutput` gains
  `entries`, `offset_chars`, `truncated`, and `next_cursor`.
- Use `..Default::default()` in argument/output struct literals when appropriate.
- Tool `note read` is now paged; follow `next_cursor` when the full content is
  needed. Host-side `load_notes`/`try_load_notes` still return complete items.
- Missing `todo set` items no longer clear tasks; explicitly send `items: []`.
  Invalid status strings no longer fall back to `pending`.
