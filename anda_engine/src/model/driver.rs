//! Shared completion driver for provider adapters.
//!
//! Every provider adapter answers a [`CompletionRequest`] with the same
//! fifteen-step algorithm — instructions, replayed raw history (recording the
//! skip point), converted chat history, documents message, prompt/content
//! message, sampling options, tools, execute with a stream/non-stream branch,
//! failure logging, drain of the replayed prefix, and conversion into
//! [`AgentOutput`]. Only the wire format differs. [`drive_completion`] owns
//! that algorithm once; a provider contributes a [`WireFormat`] implementation
//! that maps each step onto its own request and response types.
//!
//! The driver makes the seam's most fragile invariants structural instead of
//! copied prose:
//! - raw history is always appended **before** messages converted from
//!   `chat_history`, so provider per-turn state stays intact for the round;
//! - the skip point is always recorded right after the raw append, and the
//!   replayed prefix is always drained before [`WireFormat::into_output`], so
//!   the returned `raw_history` contains only this turn's additions.

use anda_core::{
    AgentOutput, BoxError, CompletionRequest, ContentPart, FunctionDefinition, Json, Message,
    ModelEffort,
};
use log::{Level::Debug, log_enabled};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::{HashMap, VecDeque};

use crate::model::{
    execute_completion_request_with_retry, read_completion_response_bytes, read_completion_stream,
    streaming_completion_request,
};
use crate::{rfc3339_datetime, unix_ms};

/// Sampling and output-shaping options forwarded to the provider.
///
/// A provider maps these options onto its own request fields and reports
/// unsupported options instead of silently ignoring them.
pub(crate) struct SamplingOptions {
    pub temperature: Option<f64>,
    pub max_output_tokens: Option<usize>,
    pub effort: Option<ModelEffort>,
    pub output_schema: Option<Json>,
    pub stop: Option<Vec<String>>,
}

/// Wire-format mapping a provider contributes to [`drive_completion`].
///
/// All methods are associated functions over the provider's own request and
/// response types; the driver owns ordering, execution, retry, and logging.
pub(crate) trait WireFormat {
    /// Provider request template, pre-seeded by the adapter (model name,
    /// defaults) before the driver runs.
    type Request: Serialize + Send + Sync;
    /// Typed provider response produced by parsing or stream aggregation.
    type Response: Send;
    /// One deserialized item of the provider's SSE stream.
    type StreamItem: DeserializeOwned + Send;

    /// Applies non-empty system instructions.
    ///
    /// A provider that carries instructions inside its message container must
    /// apply them here — before the raw-history append — so the skip point
    /// excludes them from the returned raw history.
    fn set_instructions(r: &mut Self::Request, instructions: String);

    /// Appends replayed provider-native history and returns the skip point:
    /// the container length right after the append. Everything before the skip
    /// point is drained by [`WireFormat::sent_messages`] and never re-enters
    /// the output raw history.
    fn append_raw_history(r: &mut Self::Request, raw_history: Vec<Json>) -> usize;

    /// Converts one provider-neutral [`Message`] and appends the result (one
    /// or more wire messages) to the request.
    fn push_message(r: &mut Self::Request, msg: Message) -> Result<(), BoxError>;

    /// Maps supported sampling options onto request fields.
    fn apply_sampling(
        r: &mut Self::Request,
        options: SamplingOptions,
        model: &str,
    ) -> Result<(), BoxError>;

    /// Maps a non-empty tool list (and the required-choice flag) onto the
    /// request.
    fn apply_tools(r: &mut Self::Request, tools: Vec<FunctionDefinition>, required: bool);

    /// Final request adjustments after all content is applied (for example
    /// forcing stream flags or stream usage options).
    fn finalize_request(_r: &mut Self::Request) -> Result<(), BoxError> {
        Ok(())
    }

    /// Whether this request executes as a stream.
    fn is_stream(r: &Self::Request) -> bool;

    /// Endpoint path for this request (may depend on the stream flag).
    fn endpoint(r: &Self::Request, model: &str) -> String;

    /// Aggregates deserialized stream items into a full response.
    fn aggregate_stream(
        items: Vec<Self::StreamItem>,
        done: bool,
    ) -> Result<Self::Response, BoxError>;

    /// Parses a non-streaming response body. The second tuple element is an
    /// optional provider-native assistant message captured verbatim for the
    /// output raw history (used by providers whose typed response round-trip
    /// would drop unknown fields).
    fn parse_response(model: &str, data: &[u8])
    -> Result<(Self::Response, Option<Json>), BoxError>;

    /// Whether the response should be logged as a possible failure.
    fn maybe_failed(res: &Self::Response) -> bool;

    /// Drains the replayed raw-history prefix and returns the request messages
    /// this call actually added, as provider-native JSON.
    fn sent_messages(r: Self::Request, skip_raw: usize) -> Vec<Json>;

    /// Converts the provider response into the neutral [`AgentOutput`],
    /// seeding its raw history from `sent_messages` (plus the optional
    /// captured assistant message) and its chat history from `chat_history`.
    fn into_output(
        res: Self::Response,
        sent_messages: Vec<Json>,
        chat_history: Vec<Message>,
        assistant_raw_message: Option<Json>,
    ) -> Result<AgentOutput, BoxError>;
}

/// Executes one completion round for the wire format `W`.
///
/// `post` builds a request for a provider-relative path with authentication
/// already applied; `r` is the provider request template with the model name
/// set.
pub(crate) async fn drive_completion<W: WireFormat>(
    model: String,
    post: impl Fn(&str) -> reqwest::RequestBuilder,
    mut r: W::Request,
    req: CompletionRequest,
) -> Result<AgentOutput, BoxError> {
    let timestamp = unix_ms();
    let mut chat_history: Vec<Message> = Vec::new();

    if !req.instructions.is_empty() {
        W::set_instructions(&mut r, req.instructions);
    }

    let skip_raw = W::append_raw_history(&mut r, req.raw_history);

    for msg in req.chat_history {
        W::push_message(&mut r, msg)?;
    }

    if let Some(mut msg) = req
        .documents
        .to_message(&rfc3339_datetime(timestamp).unwrap())
    {
        msg.timestamp = Some(timestamp);
        chat_history.push(msg.clone());
        W::push_message(&mut r, msg)?;
    }

    let mut content = req.content;
    if !req.prompt.is_empty() {
        content.insert(0, req.prompt.into());
    }
    if !content.is_empty() {
        let msg = Message {
            role: req.role.unwrap_or_else(|| "user".to_string()),
            content,
            timestamp: Some(timestamp),
            ..Default::default()
        };
        chat_history.push(msg.clone());
        W::push_message(&mut r, msg)?;
    }

    W::apply_sampling(
        &mut r,
        SamplingOptions {
            temperature: req.temperature,
            max_output_tokens: req.max_output_tokens,
            effort: req.effort,
            output_schema: req.output_schema,
            stop: req.stop,
        },
        &model,
    )?;

    if !req.tools.is_empty() {
        W::apply_tools(&mut r, req.tools, req.tool_choice_required);
    }

    W::finalize_request(&mut r)?;

    let stream = W::is_stream(&r);
    let path = W::endpoint(&r, &model);
    let (res, assistant_raw_message) = execute_completion_request_with_retry(
        &model,
        || {
            let mut request = post(&path).json(&r);
            if stream {
                request = streaming_completion_request(request);
            }
            request
        },
        |response| async {
            if stream {
                let (items, done) =
                    read_completion_stream::<W::StreamItem>(response, &model).await?;
                Ok((W::aggregate_stream(items, done)?, None))
            } else {
                let data = read_completion_response_bytes(response, &model).await?;
                W::parse_response(&model, &data)
            }
        },
    )
    .await?;

    let failed = W::maybe_failed(&res);
    let sent_messages = W::sent_messages(r, skip_raw);
    let output = W::into_output(res, sent_messages, chat_history, assistant_raw_message)?;
    // Conversation content and provider tool credentials never belong in routine logs.
    if failed {
        log::warn!(model = model, usage:serde = output.usage; "Completion maybe failed");
    } else if log_enabled!(Debug) {
        log::debug!(model = model, usage:serde = output.usage; "Completion response");
    }
    Ok(output)
}

/// Pairs provider-neutral tool history before replay to APIs requiring call IDs.
/// Native history is deliberately left untouched. Queues distinguish repeated
/// calls to the same function, including parallel calls without provider IDs.
/// A result without a call in the neutral history, such as one answering a call
/// carried by `raw_history` from a provider that omits IDs, is sent unchanged.
pub(crate) fn assign_tool_call_ids(req: &mut CompletionRequest) {
    let mut parts: Vec<_> = req
        .chat_history
        .iter_mut()
        .flat_map(|msg| &mut msg.content)
        .chain(&mut req.content)
        .collect();
    let mut pending = HashMap::<String, VecDeque<(usize, Option<String>)>>::new();
    for index in 0..parts.len() {
        match &*parts[index] {
            ContentPart::ToolCall { name, call_id, .. } => {
                pending
                    .entry(name.clone())
                    .or_default()
                    .push_back((index, call_id.clone().filter(|id| !id.is_empty())));
            }
            ContentPart::ToolOutput { name, call_id, .. } => {
                let result_id = call_id.clone().filter(|id| !id.is_empty());
                let queue = pending.entry(name.clone()).or_default();
                let matched = if let Some(id) = &result_id {
                    // A persisted result may already carry an ID although its
                    // original foreign call did not. Reuse that ID on replay.
                    queue
                        .iter()
                        .position(|(_, call_id)| call_id.as_ref() == Some(id))
                        .or_else(|| queue.iter().position(|(_, call_id)| call_id.is_none()))
                        .and_then(|position| queue.remove(position))
                } else {
                    queue.pop_front()
                };
                if let Some((call_index, call_id)) = matched {
                    let id = result_id
                        .or(call_id)
                        .unwrap_or_else(|| format!("call_anda_{:032x}", rand::random::<u128>()));
                    if let ContentPart::ToolCall { call_id, .. } = &mut parts[call_index] {
                        *call_id = Some(id.clone());
                    }
                    if let ContentPart::ToolOutput { call_id, .. } = &mut parts[index] {
                        *call_id = Some(id);
                    }
                }
            }
            _ => {}
        }
    }
    for part in parts {
        if let ContentPart::ToolCall { call_id, .. } = part
            && call_id.as_ref().is_none_or(String::is_empty)
        {
            *call_id = Some(format!("call_anda_{:032x}", rand::random::<u128>()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replay_reuses_ids_already_persisted_on_results() {
        let mut req = CompletionRequest {
            content: vec![
                ContentPart::ToolCall {
                    name: "lookup".into(),
                    args: json!({}),
                    call_id: None,
                },
                ContentPart::ToolOutput {
                    name: "lookup".into(),
                    output: json!("ok"),
                    call_id: Some("persisted".into()),
                    is_error: None,
                    remote_id: None,
                },
            ],
            ..Default::default()
        };
        assign_tool_call_ids(&mut req);
        assert!(
            matches!(&req.content[0], ContentPart::ToolCall {call_id:Some(id),..} if id == "persisted")
        );
        let first = req.content.clone();
        assign_tool_call_ids(&mut req);
        assert_eq!(req.content, first);
    }

    #[test]
    fn results_answering_native_calls_and_raw_history_are_untouched() {
        // An OpenAI-compatible provider without tool-call IDs: the call lives in
        // raw history and its result carries an empty ID.
        let raw = json!({"role":"assistant","tool_calls":[{"id":"","type":"function",
            "function":{"name":"lookup","arguments":"{}"}}]});
        let content = vec![
            ContentPart::ToolOutput {
                name: "lookup".into(),
                output: json!("ok"),
                call_id: Some(String::new()),
                is_error: None,
                remote_id: None,
            },
            ContentPart::ToolOutput {
                name: "legacy".into(),
                output: json!("ok"),
                call_id: None,
                is_error: None,
                remote_id: None,
            },
        ];
        let mut req = CompletionRequest {
            raw_history: vec![raw.clone()],
            content: content.clone(),
            ..Default::default()
        };
        assign_tool_call_ids(&mut req);
        assert_eq!(req.content, content);
        assert_eq!(req.raw_history, vec![raw]);
    }
}
