//! Provider raw-history classification and pruning.
//!
//! `CompletionRequest::raw_history` carries provider-native message JSON
//! (OpenAI Chat, OpenAI Responses, Anthropic, and Gemini shapes all differ), so
//! reclaiming context-window budget from it requires recognizing tool-call and
//! tool-output items in each provider's wire format. That knowledge belongs to
//! the model layer: this module is the single home for it, and
//! [`CompletionFeaturesDyn`](crate::model::CompletionFeaturesDyn) exposes it as
//! overridable default methods so a custom provider whose wire format is not
//! covered here can supply its own classification without any change to the
//! completion runner.
//!
//! The implementation here is a conservative union over all built-in provider
//! shapes. Items are rewritten in place rather than round-tripped through
//! [`Message`](anda_core::Message), which would drop or corrupt
//! provider-specific fields such as reasoning signatures.

use anda_core::Json;

/// Removes unanswered tool-call requests from `raw_history[start..]`.
///
/// Used after an interrupt (steering, discard, stop) so the next request does
/// not inherit a tool-call requirement the provider would reject as unanswered.
/// Visible text and reasoning stay untouched; wrapper items left without
/// meaningful content are removed entirely.
pub(crate) fn prune_unanswered_tool_calls(raw_history: &mut Vec<Json>, start: usize) {
    if start >= raw_history.len() {
        return;
    }

    let tail: Vec<Json> = raw_history.drain(start..).collect();
    raw_history.extend(prune_items(tail, is_tool_call_item));
}

/// Removes completed tool interactions (calls and results) from provider raw history.
pub(crate) fn prune_tool_interactions(raw_history: &mut Vec<Json>) {
    let items = std::mem::take(raw_history);
    *raw_history = prune_items(items, |value| {
        is_tool_call_item(value) || is_tool_output_item(value)
    });
}

/// Drops every item `is_pruned` classifies, recursing into wrapper items so a
/// wrapper keeps its non-tool context and disappears only once nothing
/// meaningful is left.
///
/// Walks backwards so an OpenAI Responses `reasoning` item can see whether the item it
/// must immediately precede survived; the API rejects a reasoning item whose required
/// following item is missing, so orphaned reasoning follows its pruned sibling out.
fn prune_items(items: Vec<Json>, is_pruned: impl Fn(&Json) -> bool + Copy) -> Vec<Json> {
    let mut retained: Vec<Json> = Vec::with_capacity(items.len());
    let mut next_kept = true;
    for value in items.into_iter().rev() {
        if !next_kept && value.get("type").and_then(|v| v.as_str()) == Some("reasoning") {
            continue;
        }
        match prune_item(value, is_pruned) {
            Some(value) => {
                retained.push(value);
                next_kept = true;
            }
            None => next_kept = false,
        }
    }
    retained.reverse();
    retained
}

fn prune_item(mut value: Json, is_pruned: impl Fn(&Json) -> bool + Copy) -> Option<Json> {
    if is_pruned(&value) {
        return None;
    }

    prune_nested(&mut value, is_pruned);
    if item_has_context(&value) {
        Some(value)
    } else {
        None
    }
}

fn prune_nested(value: &mut Json, is_pruned: impl Fn(&Json) -> bool + Copy) {
    match value {
        Json::Array(items) => {
            let retained: Vec<Json> = items
                .drain(..)
                .filter_map(|item| prune_item(item, is_pruned))
                .collect();
            *items = retained;
        }
        Json::Object(map) => {
            // OpenAI Chat Completions keeps tool calls as fields on an assistant message.
            // Remove the unanswered calls, but keep any text/reasoning fields on the same
            // message. Provider raw history may also wrap output items under arbitrary fields,
            // so recurse through every remaining value instead of only common content arrays.
            map.remove("tool_calls");
            map.remove("function_call");
            map.remove("functionCall");

            for value in map.values_mut() {
                prune_nested(value, is_pruned);
            }
        }
        _ => {}
    }
}

fn is_tool_call_item(value: &Json) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };

    if matches!(
        map.get("type").and_then(|v| v.as_str()),
        Some(
            "function_call"
                | "custom_tool_call"
                | "computer_call"
                | "tool_search_call"
                | "local_shell_call"
                | "shell_call"
                | "apply_patch_call"
                | "mcp_approval_request"
                | "tool_call"
                | "tool_use"
                | "ToolCall"
                | "toolCall"
        )
    ) {
        return true;
    }

    // Gemini function-call parts do not use a `type` field. Treat only the part itself as a
    // tool call; wrapper objects that also carry text or metadata should be pruned recursively
    // so their non-tool context survives.
    map.contains_key("functionCall")
        && map.keys().all(|key| {
            matches!(
                key.as_str(),
                "functionCall" | "thought" | "thoughtSignature"
            )
        })
}

fn is_tool_output_item(value: &Json) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };

    // OpenAI Chat Completions carries tool results as whole messages with the "tool" role.
    if map.get("role").and_then(|v| v.as_str()) == Some("tool") {
        return true;
    }

    if matches!(
        map.get("type").and_then(|v| v.as_str()),
        Some(
            "function_call_output"
                | "custom_tool_call_output"
                | "computer_call_output"
                | "local_shell_call_output"
                | "shell_call_output"
                | "apply_patch_call_output"
                | "mcp_approval_response"
                | "tool_result"
                | "tool_output"
                | "ToolOutput"
                | "toolOutput"
        )
    ) {
        return true;
    }

    // Gemini function-response parts do not use a `type` field; mirror the
    // `is_tool_call_item` handling for `functionCall` parts.
    map.contains_key("functionResponse")
        && map.keys().all(|key| {
            matches!(
                key.as_str(),
                "functionResponse" | "thought" | "thoughtSignature"
            )
        })
}

fn item_has_context(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(_) | Json::Number(_) => true,
        Json::String(text) => !text.is_empty(),
        Json::Array(items) => items.iter().any(item_has_context),
        Json::Object(map) => map.iter().any(|(key, value)| {
            !matches!(
                key.as_str(),
                "role" | "type" | "name" | "id" | "status" | "phase" | "timestamp"
            ) && item_has_context(value)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prunes_contextless_raw_history_items() {
        let mut raw_history = vec![
            json!(null),
            json!(""),
            json!({"role": "assistant", "id": "meta-only", "status": "ok"}),
            json!({"role": "assistant", "content": []}),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
                    {"text": "kept text"}
                ],
                "tool_calls": [{"id": "call_1"}],
                "function_call": {"name": "lookup"}
            }),
            json!({"type": "function_call", "call_id": "call_1"}),
            json!(42),
        ];

        prune_unanswered_tool_calls(&mut raw_history, 0);

        assert_eq!(raw_history.len(), 2);
        assert_eq!(raw_history[0]["content"], json!([{"text": "kept text"}]));
        assert!(raw_history[0].get("tool_calls").is_none());
        assert!(raw_history[0].get("function_call").is_none());
        assert_eq!(raw_history[1], json!(42));
    }

    #[test]
    fn prunes_deeply_nested_raw_tool_calls() {
        let sentinel = json!({"role": "user", "content": "prior"});
        let mut raw_history = vec![
            sentinel.clone(),
            json!({
                "role": "assistant",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "kept nested text"}
                        ]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_nested_function",
                        "name": "lookup",
                        "arguments": "{}"
                    },
                    {
                        "type": "local_shell_call",
                        "id": "lsh_1",
                        "action": {"type": "exec", "command": ["pwd"]},
                        "call_id": "call_nested_shell",
                        "status": "completed"
                    },
                    {
                        "type": "apply_patch_call",
                        "id": "ap_1",
                        "call_id": "call_nested_patch",
                        "operation": {"type": "update", "path": "src/lib.rs", "diff": "@@"},
                        "status": "completed"
                    }
                ],
                "metadata": {
                    "items": [
                        {
                            "type": "custom_tool_call",
                            "call_id": "call_nested_custom",
                            "input": "select 1",
                            "name": "sql"
                        },
                        {"note": "keep nested metadata"}
                    ]
                },
                "tool_calls": [{"id": "call_nested_chat"}],
                "function_call": {"name": "legacy"},
                "functionCall": {"name": "gemini"}
            }),
        ];

        prune_unanswered_tool_calls(&mut raw_history, 1);

        assert_eq!(raw_history.len(), 2);
        assert_eq!(raw_history[0], sentinel);
        let pruned = serde_json::to_string(&raw_history[1]).unwrap();
        assert!(pruned.contains("kept nested text"));
        assert!(pruned.contains("keep nested metadata"));
        assert!(!pruned.contains("call_nested"));
        assert!(raw_history[1].get("tool_calls").is_none());
        assert!(raw_history[1].get("function_call").is_none());
        assert!(raw_history[1].get("functionCall").is_none());
    }

    #[test]
    fn prunes_completed_tool_interactions_from_raw_history() {
        let mut raw_history = vec![
            // OpenAI Chat Completions shapes.
            json!({"role": "user", "content": "question"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "chat tool result"}),
            // OpenAI Responses shapes: orphaned reasoning must follow its pruned call out.
            json!({"type": "reasoning", "id": "rs_orphan", "encrypted_content": "opaque"}),
            json!({"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_2", "output": "ok"}),
            // Anthropic shapes.
            json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "kept thinking", "signature": "sig"},
                    {"type": "text", "text": "anthropic text"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
                ]
            }),
            json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "anthropic result"}
                ]
            }),
            // Gemini shapes.
            json!({
                "role": "model",
                "parts": [
                    {"text": "gemini text"},
                    {"functionCall": {"name": "lookup", "args": {}}}
                ]
            }),
            json!({
                "role": "user",
                "parts": [{"functionResponse": {"name": "lookup", "response": {"ok": true}}}]
            }),
            // Reasoning followed by a surviving item stays.
            json!({"type": "reasoning", "id": "rs_kept", "encrypted_content": "opaque"}),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "final answer"}]
            }),
        ];

        prune_tool_interactions(&mut raw_history);

        let pruned = serde_json::to_string(&raw_history).unwrap();
        assert_eq!(raw_history.len(), 5);
        assert_eq!(raw_history[0]["content"], json!("question"));
        assert!(pruned.contains("kept thinking"));
        assert!(pruned.contains("anthropic text"));
        assert!(pruned.contains("gemini text"));
        assert!(pruned.contains("rs_kept"));
        assert!(pruned.contains("final answer"));
        assert!(!pruned.contains("call_1"));
        assert!(!pruned.contains("call_2"));
        assert!(!pruned.contains("rs_orphan"));
        assert!(!pruned.contains("toolu_1"));
        assert!(!pruned.contains("chat tool result"));
        assert!(!pruned.contains("anthropic result"));
        assert!(!pruned.contains("functionCall"));
        assert!(!pruned.contains("functionResponse"));
    }

    #[test]
    fn prunes_only_unanswered_raw_tool_call_items() {
        let sentinel = json!({"role": "user", "content": "prior"});
        let mut raw_history = vec![
            sentinel.clone(),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "anthropic text"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
                ]
            }),
            json!({
                "role": "model",
                "parts": [
                    {"text": "gemini text"},
                    {"functionCall": {"name": "lookup", "args": {}}}
                ]
            }),
            json!({"type": "function_call", "call_id": "call_1"}),
            json!({"type": "custom_tool_call", "call_id": "call_2"}),
        ];

        prune_unanswered_tool_calls(&mut raw_history, 1);

        assert_eq!(raw_history.len(), 3);
        assert_eq!(raw_history[0], sentinel);
        assert_eq!(raw_history[1]["content"].as_array().unwrap().len(), 1);
        assert_eq!(raw_history[1]["content"][0]["text"], "anthropic text");
        assert_eq!(raw_history[2]["parts"].as_array().unwrap().len(), 1);
        assert_eq!(raw_history[2]["parts"][0]["text"], "gemini text");
    }

    #[test]
    fn prunes_reasoning_orphaned_by_an_unanswered_tool_call() {
        // OpenAI Responses emits `[reasoning, function_call]` for a tool turn; dropping the
        // unanswered call must take its reasoning sibling with it, or the next request is
        // rejected for a reasoning item without its required following item.
        let sentinel = json!({"role": "user", "content": "prior"});
        let mut raw_history = vec![
            sentinel.clone(),
            json!({"type": "reasoning", "id": "rs_kept", "encrypted_content": "opaque"}),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "planning"}]
            }),
            json!({"type": "reasoning", "id": "rs_orphan", "encrypted_content": "opaque"}),
            json!({"type": "function_call", "call_id": "call_1", "name": "lookup"}),
        ];

        prune_unanswered_tool_calls(&mut raw_history, 1);

        let pruned = serde_json::to_string(&raw_history).unwrap();
        assert_eq!(raw_history.len(), 3);
        assert_eq!(raw_history[0], sentinel);
        assert!(pruned.contains("rs_kept"));
        assert!(pruned.contains("planning"));
        assert!(!pruned.contains("rs_orphan"));
        assert!(!pruned.contains("call_1"));
    }
}
