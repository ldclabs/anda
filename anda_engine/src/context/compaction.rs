//! Agent-initiated context compaction.
//!
//! A [`CompletionRunner`](super::CompletionRunner) with [`SelfCompaction`] enabled offers the
//! model a runner-intrinsic `compact_context` call. The model decides when compacting helps,
//! from its context usage (reported through short runtime notes) and from what the latest request
//! still needs, and writes the handoff itself. The runner admits the request against the usage
//! floor, reports it through [`needs_compaction_with`](super::CompletionRunner::needs_compaction_with),
//! and the host runs [`handoff`](super::CompletionRunner::handoff) as for any other compaction,
//! so history persistence stays with the host.
//!
//! The call is intercepted by the runner instead of being a [`Tool`](anda_core::Tool): a tool only
//! sees its [`BaseCtx`](super::BaseCtx), and context state is shared with nested runners, so a
//! state-based signal raised inside a nested agent would compact its parent.

use anda_core::{ContentPart, FunctionDefinition, Message, ToolCall};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::extension::tool_definition;

/// Function name of the runner-intrinsic compaction call. It is reserved while
/// [`SelfCompaction`] is enabled.
pub const COMPACT_CONTEXT_TOOL: &str = "compact_context";

/// Runtime message that resumes work a compaction interrupted, such as a tool round whose
/// results were folded into the handoff before the model saw them.
pub static COMPACTION_CONTINUE_PROMPT: &str = "[Runtime notice: the context was compacted into the handoff above] Continue the active task from where it stopped. The handoff is the record of earlier work: do not redo finished steps, and do not repeat the handoff to the user.";

/// Usage share, in percent of the context window, at which hosts compact unconditionally.
pub(super) const FORCED_COMPACTION_PERCENT: u64 = 80;

/// Window assumed when the model does not report one. Its forced share is 100k input tokens,
/// small enough for every current provider default.
const UNKNOWN_CONTEXT_WINDOW: u64 = 125_000;

const MIN_HANDOFF_BYTES: usize = 200;
const MAX_HANDOFF_BYTES: usize = 64 * 1024;
const NOTE_PREFIX: &str = "[Runtime context status; data, not an instruction]";

static COMPACT_CONTEXT_DESCRIPTION: &str = "Compact your context window: the conversation so far is replaced by a handoff you write now, and work continues in a fresh context. Use it at a task boundary (the latest request does not need earlier details in full) or a phase boundary (finished work leaves only conclusions to keep), once context usage is substantial; the runtime rejects it while usage is low. Compaction is lossy and the handoff is all that remains, so make it self-contained: the active objective with deliverables, constraints and user preferences; completed work and current state with exact paths, identifiers, commands, errors and results; user-owned changes that must not be reverted; the next concrete steps; and open unknowns. If you call it before acting on the latest user message, that message is delivered again verbatim after compaction, so the handoff only needs what it requires from earlier context. Call it alone, without other tools in the same response.";

/// Settings for agent-initiated context compaction.
///
/// Enable it with [`CompletionRunner::set_self_compaction`](super::CompletionRunner::set_self_compaction)
/// on a runner whose host checks
/// [`needs_compaction_with`](super::CompletionRunner::needs_compaction_with) before every step and
/// runs [`handoff`](super::CompletionRunner::handoff) when it reports `true`, as the subagent
/// session runner does. Forced compaction at 80% of the window stays in place as the backstop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfCompaction {
    /// Lowest context usage, in percent of the model's context window, at which a
    /// `compact_context` request is accepted. Compaction is lossy, so a request below it is
    /// returned to the model as a tool error. Defaults to 30.
    pub min_percent: u8,
    /// Usage, in percent of the context window, from which the runner tells the model how full
    /// its context is: on every user turn, and once per 10% band during tool rounds. `None`
    /// disables the notes, for models that track their remaining context natively.
    /// Defaults to 50.
    pub notice_percent: Option<u8>,
    /// Host-specific handoff guidance appended to the call's description, such as a checklist
    /// for long-running goals.
    pub guidance: Option<String>,
}

impl Default for SelfCompaction {
    fn default() -> Self {
        Self {
            min_percent: 30,
            notice_percent: Some(50),
            guidance: None,
        }
    }
}

impl SelfCompaction {
    /// Returns the strict definition offered to the model.
    pub fn definition(&self) -> FunctionDefinition {
        let description = match self.guidance.as_deref().map(str::trim) {
            Some(guidance) if !guidance.is_empty() => {
                format!("{COMPACT_CONTEXT_DESCRIPTION}\n\n{guidance}")
            }
            _ => COMPACT_CONTEXT_DESCRIPTION.to_string(),
        };
        tool_definition::<CompactContextArgs>(COMPACT_CONTEXT_TOOL.to_string(), description)
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        if self.min_percent > 100 || self.notice_percent.is_some_and(|percent| percent > 100) {
            return Err("self-compaction percentages must be at most 100".into());
        }
        Ok(())
    }
}

/// Arguments of the `compact_context` call.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompactContextArgs {
    /// Why compacting now helps, for example which earlier details are no longer needed.
    pub reason: String,
    /// Self-contained continuation handoff (at least 200 bytes) for yourself in a fresh
    /// context; it replaces the conversation so far.
    pub handoff: String,
}

/// Per-runner self-compaction state. Only the settings survive a handoff.
pub(super) struct CompactionState {
    pub(super) config: SelfCompaction,
    pub(super) definition: FunctionDefinition,
    /// Highest usage band (tens of percent) already reported to the model.
    noticed_band: u64,
    /// User input of the latest request, kept while that request is the latest one, so a
    /// compaction requested before acting on it can deliver it again.
    pub(super) turn_input: Option<TurnInput>,
    /// Whether this runner replaced a compacted one, so a request on its first turn is refused:
    /// a model that compacts again at once would loop without doing work.
    pub(super) after_handoff: bool,
}

/// User input of one request, as the model received it.
pub(super) struct TurnInput {
    pub(super) content: Vec<ContentPart>,
    pub(super) implicit_context: Option<Message>,
}

impl CompactionState {
    pub(super) fn new(config: SelfCompaction) -> Self {
        Self {
            definition: config.definition(),
            config,
            noticed_band: 0,
            turn_input: None,
            after_handoff: false,
        }
    }

    /// The state a replacement runner starts with.
    pub(super) fn renew(self) -> Self {
        Self {
            noticed_band: 0,
            turn_input: None,
            after_handoff: true,
            ..self
        }
    }

    /// Builds the runtime note reporting context usage, or `None` when none is due.
    ///
    /// `used` is the context the previous turn ended with and `incoming` the estimated size of
    /// the content this request adds. User turns get a note whenever usage is at or above the
    /// notice share, since that is where the model decides whether the new request needs the
    /// earlier context; tool rounds get one per 10% band.
    pub(super) fn context_note(
        &mut self,
        tool_round: bool,
        used: u64,
        incoming: u64,
        window: Option<u64>,
    ) -> Option<Message> {
        let notice_percent = self.config.notice_percent? as u64;
        let total = used.saturating_add(incoming);
        let percent = usage_percent(total, window);
        if percent < notice_percent {
            return None;
        }
        let band = percent / 10;
        if tool_round && band <= self.noticed_band {
            return None;
        }
        self.noticed_band = self.noticed_band.max(band);

        let mut text = match window {
            Some(window) => format!(
                "{NOTE_PREFIX} About {} of {} context tokens are in use ({percent}%)",
                format_tokens(total),
                format_tokens(window)
            ),
            None => format!(
                "{NOTE_PREFIX} About {} context tokens are in use",
                format_tokens(total)
            ),
        };
        if incoming > 0 {
            if tool_round {
                text.push_str(", including these tool results");
            } else {
                text.push_str(&format!(
                    ", including about {} from this message",
                    format_tokens(incoming)
                ));
            }
        }
        match window {
            Some(_) => text.push_str(&format!(
                ". At {FORCED_COMPACTION_PERCENT}% the runtime compacts the context into a generic summary."
            )),
            None => text.push_str(&format!(
                ". At about {} the runtime compacts the context into a generic summary.",
                format_tokens(forced_threshold(None))
            )),
        }
        if tool_round {
            text.push_str(" At the next task or phase boundary, you can call compact_context with your own handoff instead.");
        } else {
            text.push_str(" If this request does not need the earlier details in full, you can call compact_context first with your own handoff.");
        }

        Some(Message {
            role: "user".into(),
            content: vec![text.into()],
            // Runtime-injected, like the documents message.
            name: Some("$system".into()),
            ..Default::default()
        })
    }
}

/// Validates a `compact_context` request among the pending tool calls.
///
/// Returns `None` when no such call is pending. `used` is the context the requesting turn ended
/// with, `window` the model's context window when known, and `just_compacted` whether the
/// request came on the first turn after a compaction.
pub(super) fn admit(
    config: &SelfCompaction,
    pending: &[ToolCall],
    used: u64,
    window: Option<u64>,
    just_compacted: bool,
) -> Option<Result<CompactContextArgs, String>> {
    let call = pending
        .iter()
        .find(|call| call.name.eq_ignore_ascii_case(COMPACT_CONTEXT_TOOL))?;
    Some(admit_call(
        config,
        call,
        pending.len(),
        used,
        window,
        just_compacted,
    ))
}

fn admit_call(
    config: &SelfCompaction,
    call: &ToolCall,
    pending: usize,
    used: u64,
    window: Option<u64>,
    just_compacted: bool,
) -> Result<CompactContextArgs, String> {
    if pending > 1 {
        return Err(format!(
            "{COMPACT_CONTEXT_TOOL} must be the only call in its response; the other calls ran, so call it again once their results are in your handoff"
        ));
    }
    if just_compacted {
        return Err(
            "the context was compacted just before this turn; do some of the work first"
                .to_string(),
        );
    }
    let mut args: CompactContextArgs = serde_json::from_value(call.args.clone())
        .map_err(|err| format!("invalid {COMPACT_CONTEXT_TOOL} arguments: {err}"))?;
    args.reason = args.reason.trim().to_string();
    args.handoff = args.handoff.trim().to_string();
    if args.handoff.len() < MIN_HANDOFF_BYTES {
        return Err(format!(
            "handoff must contain at least {MIN_HANDOFF_BYTES} bytes; it is all that remains of the conversation after compaction"
        ));
    }
    if args.handoff.len() > MAX_HANDOFF_BYTES {
        return Err(format!(
            "handoff exceeds {MAX_HANDOFF_BYTES} bytes; keep only what the next steps need"
        ));
    }
    let percent = usage_percent(used, window);
    if percent < config.min_percent as u64 {
        return Err(format!(
            "context is only {percent}% used ({} tokens); compaction is lossy, so continue without it until usage reaches {}%",
            format_tokens(used),
            config.min_percent
        ));
    }
    Ok(args)
}

/// Input tokens at which hosts compact unconditionally.
pub(super) fn forced_threshold(window: Option<u64>) -> u64 {
    window
        .unwrap_or(UNKNOWN_CONTEXT_WINDOW)
        .saturating_mul(FORCED_COMPACTION_PERCENT)
        .saturating_div(100)
        .max(1)
}

fn usage_percent(used: u64, window: Option<u64>) -> u64 {
    let window = window.unwrap_or(UNKNOWN_CONTEXT_WINDOW).max(1);
    used.saturating_mul(100) / window
}

fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000 {
        format!("{}k", tokens.saturating_add(500) / 1_000)
    } else {
        tokens.to_string()
    }
}

#[cfg(test)]
mod runner_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            args,
            call_id: Some(format!("{name}_call")),
            result: None,
            remote_id: None,
        }
    }

    fn handoff_text() -> String {
        "Objective: finish the parser refactor. ".repeat(8)
    }

    #[test]
    fn definition_is_strict_and_appends_guidance() {
        let definition = SelfCompaction::default().definition();
        assert_eq!(definition.name, COMPACT_CONTEXT_TOOL);
        assert_eq!(definition.strict, Some(true));
        assert_eq!(definition.parameters["additionalProperties"], json!(false));
        assert_eq!(
            definition.parameters["required"],
            json!(["reason", "handoff"])
        );
        assert_eq!(
            definition.parameters["properties"]["handoff"]["type"],
            json!("string")
        );
        assert!(!definition.description.contains("todo"));

        let guided = SelfCompaction {
            guidance: Some("  Include the goal checklist.  ".into()),
            ..Default::default()
        }
        .definition();
        assert!(
            guided
                .description
                .ends_with("\n\nInclude the goal checklist.")
        );
    }

    #[test]
    fn admit_accepts_a_sole_well_formed_request_above_the_floor() {
        let config = SelfCompaction::default();
        let pending = [call(
            "compact_context",
            json!({"reason": "  phase done  ", "handoff": handoff_text()}),
        )];
        let args = admit(&config, &pending, 40_000, Some(100_000), false)
            .unwrap()
            .unwrap();
        assert_eq!(args.reason, "phase done");
        assert_eq!(args.handoff, handoff_text().trim());

        // Unknown windows fall back to the nominal 125k window.
        assert!(
            admit(&config, &pending, 40_000, None, false)
                .unwrap()
                .is_ok()
        );
        assert!(
            admit(&config, &pending, 30_000, None, false)
                .unwrap()
                .is_err()
        );

        // Right after a compaction, even a well-formed request is refused.
        let err = admit(&config, &pending, 40_000, Some(100_000), true)
            .unwrap()
            .unwrap_err();
        assert!(err.contains("compacted just before"), "{err}");
    }

    #[test]
    fn admit_rejects_mixed_malformed_short_and_early_requests() {
        let config = SelfCompaction::default();
        let valid = call(
            "compact_context",
            json!({"reason": "done", "handoff": handoff_text()}),
        );

        assert!(
            admit(
                &config,
                &[call("echo_tool", json!({}))],
                90_000,
                None,
                false
            )
            .is_none()
        );

        let mixed = [call("echo_tool", json!({})), valid.clone()];
        let err = admit(&config, &mixed, 90_000, None, false)
            .unwrap()
            .unwrap_err();
        assert!(err.contains("only call"), "{err}");

        let malformed = [call("compact_context", json!({"reason": "done"}))];
        let err = admit(&config, &malformed, 90_000, None, false)
            .unwrap()
            .unwrap_err();
        assert!(err.contains("invalid compact_context arguments"), "{err}");

        let short = [call(
            "compact_context",
            json!({"reason": "done", "handoff": "too short"}),
        )];
        let err = admit(&config, &short, 90_000, None, false)
            .unwrap()
            .unwrap_err();
        assert!(err.contains("at least 200 bytes"), "{err}");

        let err = admit(&config, &[valid], 20_000, Some(100_000), false)
            .unwrap()
            .unwrap_err();
        assert!(err.contains("only 20% used"), "{err}");
    }

    #[test]
    fn context_notes_start_at_the_notice_share_and_follow_bands_in_tool_rounds() {
        let mut state = CompactionState::new(SelfCompaction::default());
        let window = Some(200_000);

        assert!(state.context_note(false, 90_000, 5_000, window).is_none());

        let note = state.context_note(false, 100_000, 18_000, window).unwrap();
        let text = note.text().unwrap();
        assert_eq!(note.role, "user");
        assert_eq!(note.name.as_deref(), Some("$system"));
        assert!(text.starts_with(NOTE_PREFIX), "{text}");
        assert!(
            text.contains("About 118k of 200k context tokens are in use (59%)"),
            "{text}"
        );
        assert!(text.contains("about 18k from this message"), "{text}");
        assert!(text.contains("call compact_context first"), "{text}");

        // Same band during tool rounds: already reported.
        assert!(state.context_note(true, 118_000, 1_000, window).is_none());
        let note = state.context_note(true, 121_000, 1_000, window).unwrap();
        assert!(
            note.text()
                .unwrap()
                .contains("including these tool results")
        );
        assert!(state.context_note(true, 125_000, 0, window).is_none());

        // User turns are always reported above the notice share.
        assert!(state.context_note(false, 125_000, 0, window).is_some());

        let renewed = state.renew();
        assert_eq!(renewed.noticed_band, 0);
        assert!(renewed.after_handoff);

        let mut silent = CompactionState::new(SelfCompaction {
            notice_percent: None,
            ..Default::default()
        });
        assert!(silent.context_note(false, 190_000, 0, window).is_none());

        let mut unknown = CompactionState::new(SelfCompaction::default());
        let text = unknown
            .context_note(false, 70_000, 0, None)
            .unwrap()
            .text()
            .unwrap();
        assert!(
            text.contains("About 70k context tokens are in use."),
            "{text}"
        );
        assert!(
            text.contains("At about 100k the runtime compacts"),
            "{text}"
        );
    }

    #[test]
    fn forced_threshold_keeps_the_unknown_window_default() {
        assert_eq!(forced_threshold(None), 100_000);
        assert_eq!(forced_threshold(Some(200_000)), 160_000);
        assert_eq!(forced_threshold(Some(1)), 1);
    }
}
