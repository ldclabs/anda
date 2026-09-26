# Subagent execution and lifecycle

`SubAgent` remains a reusable worker definition, callable through `SA_<name>`.
A nonempty session alias runs it in the background; an empty alias uses blocking
mode. Instructions, selected resources, model routing, and the execution-time
callable allowlist continue to apply. Discovery through explicitly allowed
discovery agents retains its existing semantics.

## Root ownership and migration

Background sessions are keyed by **caller principal, host-created root scope,
worker definition, and session alias**. Every new engine entry context receives
a fresh `SubAgentScope`. Normal child contexts inherit it. Changing the caller with `AgentCtx::with_caller`
creates another fresh scope; retain the returned context for subsequent calls. Independent tasks from
the same caller can therefore safely use the same worker/session alias.

To continue a root task across application turns, install the same scope clone
on the new `Engine::ctx_with` context **before** running the agent. Install it in
an engine start hook when using `Engine::agent_run`. Scope IDs are host-owned
capabilities; never populate them from model text or unsigned request metadata.
A scope rejects use by a different caller. Separate processes must not run the
same restored root concurrently; this is an in-process coordinator, not a
lease-based distributed scheduler.

Use `get_session_in_scope` and `session_details_in_scope` for model-facing lookup.
Caller-only and caller-free helpers remain for host administration; lookup returns
`None` when an alias is ambiguous across roots. Existing `SA_` names and
`agent:session` background task IDs are unchanged. Each instance additionally has
an opaque execution ID and root/parent IDs; late callbacks from an old generation
cannot finish a newer child registered under the same background task ID.

## Completion, controls, and messages

`AgentHook::on_background_turn_end` reports a completed work turn while the
session stays usable. A work turn may batch several queued inputs and perform
multiple model requests. It completes only when the runner and its owned
background work are idle. It is distinct from the model-request count.
`on_background_end` retains its original session-closure meaning. Both callbacks
carry cumulative worker usage. Nested forwarding computes deltas and delivers
artifacts at completion without duplicating the final idle result at closure.

Existing `/status`, `/steer`, `/stop`, `/stop_task`, and `/cancel` controls remain.
`/status` includes execution identity, work-turn count, event cursor, and queued
notification count. A bounded terminal snapshot remains available after cleanup. A closed event makes
the terminal result observable; an alias may briefly reject reuse until its final
callbacks and reservation cleanup finish.
Stop/cancel controls are coalesced outside the ordinary input queue, with cancel
taking precedence. Stop idles current work; cancel closes the session.

Additional controls use the existing prompt field:

- `/message <text>` accepts attributed task data without waking an idle worker.
  Active workers receive it at a safe runner boundary. These messages do not
  grant user authorization.
- `/wait <after_sequence> [timeout_ms]` waits for this session's lifecycle events;
  the default is 30 seconds and the maximum is 60 seconds. Reuse the returned
  cursor. A cursor from a previous runtime generation also reports `lagged`. A timeout is not completion, and `lagged` means the caller must refresh
  status because earlier events left the retention window.

Hosts can use `SubSession::send` with typed `SubAgentMessage` and
`MessageDelivery::{QueueOnly, TriggerTurn}`. Idle notifications reserve one queue
slot for the follow-up that will consume them. Full or oversized inputs are
rejected immediately instead of blocking indefinitely. Explicit stop/cancel may
discard queued input. Acceptance is not evidence of model consumption or durable
message delivery. Background result overflow explicitly fails the receiving
session rather than silently dropping a final result or blocking its producer.

`SubAgentScope::events` and `wait` support one or multiple execution IDs. `Any`
returns on matching activity; `All` requires a terminal turn/session event for
every supplied ID after the cursor. An empty target list selects the tree for
`Any`. Waits observe events without consuming them, and can be cancelled.
Events contain compact previews, never entire conversation histories. Full
results remain available to hooks and optional conversation recording.

## Resource policy

Install a `SubAgentScope::new(SubAgentLimits { ... })` to customize host limits.
Defaults allow 64 resident sessions, 8 concurrent model requests, 42 pending
inputs per session, 64 KiB per input including resources, 128 retained events and
terminal snapshots, and one hour of terminal retention. There is no default
aggregate token/request budget or absolute deadline. Session aliases are capped
at 128 bytes. Root scopes and their registries should be released when the host
finishes the root task.

Session and model admission are atomic and return errors at capacity. RAII guards
release reservations when initialization or a model request fails or is cancelled.
The model permit covers only inference, so a parent awaiting child tools does not
hold a permit needed by that child. Root, child, and compaction inference through
`CompletionRunner` share one accounting point. Parent summaries of child usage
are never charged again. Request limits count admitted attempts, including failed
attempts; token accounting uses actual input plus output tokens returned by the
provider. In-flight requests can overshoot the token threshold. Direct provider
calls outside `CompletionRunner` are outside this accounting contract.

An optional absolute `deadline_ms` also interrupts pending runner/tool work.
It is separate from the per-definition idle timeout. Message/control queue bounds
and event retention apply independently of context compaction.

## Idle checkpoints and handoffs

`SubAgentConversationRecorder` remains an audit log. `SubAgentCheckpoints` is a
separate opt-in idle recovery store; it supports a custom `CheckpointStore` or
an object-store adapter with atomic replacement and a 4 MiB encoded size cap.
Keys hash the caller, root, worker, and alias, so aliases cannot inject paths.

Checkpoints save completed neutral history, identity, turn count, usage and
artifacts only at safe idle boundaries. A new task invalidates the old checkpoint
before performing work. Cancellation invalidates it; idle expiry leaves it
available. Restarting with the trusted root ID, a newly installed current worker
definition and the checkpoint store restores that idle conversation. Instructions,
whitelists, model choices and context capabilities come from current host
configuration, never from the snapshot. Ownership, format version and paired tool
history are validated. Failed checkpoint writes fail the session visibly.

Recovery does not replay interrupted tools, guarantee exactly-once external side
effects, or persist queued mailbox input. It does not automatically evict a live
worker to admit another; idle timeout reclaims capacity, and admission fails while
all slots are occupied. Hosts can restore root usage using `restore_usage` from a
trusted aggregate watermark. Checkpoints also retain the last observed root
watermark, but crash-safe billing requires the host to persist aggregate accounting
independently of idle checkpoints. Audit records are not executable checkpoints.

History sharing is explicit through `SubAgentHandoff::summary`, `messages`, or
`last_turns`. The default remains a self-contained prompt and selected resources.
Handoffs preserve paired tool interactions, reject incomplete selections and
oversized content, exclude system/developer policy and provider-specific raw
history, and identify inherited content as background rather than new user
authorization. They are consumed for one generation, not automatically propagated
to every descendant. No provider thinking signatures are added to `ContentPart`.

See [the local example](../anda_engine/examples/subagent_sessions.rs). Run it with:

```sh
cargo run -p anda_engine --example subagent_sessions
```
