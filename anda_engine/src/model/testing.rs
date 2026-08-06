//! Programmable completion doubles for tests and examples.
//!
//! [`ScriptedCompleter`] is a [`CompletionFeaturesDyn`] implementation whose
//! behavior is scripted per call, replacing the hand-rolled fake completers
//! that engine, runner, and subagent tests otherwise each write from scratch.
//! It is a regular (non-`cfg(test)`) module so downstream crates can drive
//! their own agents against it:
//!
//! ```rust,ignore
//! use anda_engine::model::{Model, testing::ScriptedCompleter};
//!
//! let completer = ScriptedCompleter::new("scripted")
//!     .push_with(|req| {
//!         Ok(anda_core::AgentOutput {
//!             content: format!("saw: {}", req.prompt),
//!             ..Default::default()
//!         })
//!     });
//! let model = Model::with_completer(completer.into_arc());
//! ```

use anda_core::{AgentOutput, BoxError, BoxPinFut, CompletionRequest, ContentPart, Usage};
use parking_lot::Mutex;
use std::{collections::VecDeque, sync::Arc};

use super::CompletionFeaturesDyn;

type Reply = Box<dyn FnOnce(&CompletionRequest) -> Result<AgentOutput, BoxError> + Send>;

/// A programmable completion provider.
///
/// Each call pops the next scripted reply; when the script is exhausted the
/// completer echoes the request (prompt or textual content becomes the output
/// content, with a one-request usage record), which is what a plain
/// echo-completer test double needs. Every received request is recorded and
/// can be inspected with [`ScriptedCompleter::requests`].
pub struct ScriptedCompleter {
    name: String,
    script: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<CompletionRequest>>,
}

impl ScriptedCompleter {
    /// Creates a completer with the given model name and an empty script.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            script: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Queues a fixed output as the next reply.
    pub fn push_output(self, output: AgentOutput) -> Self {
        self.script.lock().push_back(Box::new(move |_| Ok(output)));
        self
    }

    /// Queues an error as the next reply.
    pub fn push_error(self, message: impl Into<String>) -> Self {
        let message = message.into();
        self.script
            .lock()
            .push_back(Box::new(move |_| Err(message.into())));
        self
    }

    /// Queues a closure that builds the next reply from the request.
    pub fn push_with<F>(self, reply: F) -> Self
    where
        F: FnOnce(&CompletionRequest) -> Result<AgentOutput, BoxError> + Send + 'static,
    {
        self.script.lock().push_back(Box::new(reply));
        self
    }

    /// Wraps the completer for [`Model::with_completer`](super::Model::with_completer).
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Returns clones of every request received so far, in call order.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.requests.lock().clone()
    }

    /// The echo reply used when the script is exhausted.
    fn echo(req: &CompletionRequest) -> AgentOutput {
        let content = if req.prompt.is_empty() {
            req.content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                        Some(text.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            req.prompt.clone()
        };

        AgentOutput {
            content,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
                requests: 1,
            },
            ..Default::default()
        }
    }
}

impl CompletionFeaturesDyn for ScriptedCompleter {
    fn model_name(&self) -> String {
        self.name.clone()
    }

    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        self.requests.lock().push(req.clone());
        let reply = self.script.lock().pop_front();
        let result = match reply {
            Some(reply) => reply(&req),
            None => Ok(Self::echo(&req)),
        };
        Box::pin(futures::future::ready(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;

    #[tokio::test]
    async fn scripted_completer_pops_replies_then_echoes_and_records() {
        let completer = ScriptedCompleter::new("scripted")
            .push_output(AgentOutput {
                content: "first".to_string(),
                ..Default::default()
            })
            .push_with(|req| {
                Ok(AgentOutput {
                    content: format!("saw: {}", req.prompt),
                    ..Default::default()
                })
            })
            .push_error("boom")
            .into_arc();
        let model = Model::with_completer(completer.clone());

        let req = |prompt: &str| CompletionRequest {
            prompt: prompt.to_string(),
            ..Default::default()
        };

        assert_eq!(model.completion(req("a")).await.unwrap().content, "first");
        assert_eq!(
            model.completion(req("b")).await.unwrap().content,
            "saw: b"
        );
        assert_eq!(
            model.completion(req("c")).await.unwrap_err().to_string(),
            "boom"
        );
        // Script exhausted: echo with a one-request usage record.
        let echoed = model.completion(req("tail")).await.unwrap();
        assert_eq!(echoed.content, "tail");
        assert_eq!(echoed.usage.requests, 1);

        let seen: Vec<String> = completer
            .requests()
            .into_iter()
            .map(|r| r.prompt)
            .collect();
        assert_eq!(seen, vec!["a", "b", "c", "tail"]);
    }
}
