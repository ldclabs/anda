//! Live checks against DeepSeek's Anthropic-compatible API.
//!
//! Run with `DEEPSEEK_API_KEY=… cargo test -p anda_engine --test deepseek_live -- --ignored
//! --nocapture`. Each check sends a few small requests to `deepseek-flash`.

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, FunctionDefinition, Json, ModelEffort,
    Tool, ToolOutput,
};
use anda_engine::{
    context::BaseCtx,
    engine::EngineBuilder,
    model::{CompletionFeaturesDyn, Model, anthropic},
};
use serde_json::json;
use std::sync::{Arc, Mutex};

const API_BASE: &str = "https://api.deepseek.com/anthropic";
const MODEL: &str = "deepseek-flash";
/// Provider cache blocks may leave a short uncached suffix of an unchanged prefix.
const CACHE_SUFFIX_TOKENS: u64 = 256;

fn api_key() -> String {
    std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY is required")
}

fn deepseek(effort: ModelEffort) -> anthropic::CompletionModel {
    anthropic::Client::new(&api_key(), Some(API_BASE.to_string()))
        .completion_model(MODEL)
        .with_stream(true)
        .with_effort(Some(effort))
        .with_max_output(8192)
}

/// What one request carried and what it cost.
#[derive(Debug, Clone)]
struct Round {
    tool_additions: Vec<String>,
    activations_replayed: usize,
    input_tokens: u64,
    cached_tokens: u64,
    tool_calls: Vec<String>,
    raw_history: Vec<Json>,
}

/// Forwards to the adapter and records each request; it never changes the wire.
struct Recorder {
    inner: anthropic::CompletionModel,
    rounds: Arc<Mutex<Vec<Round>>>,
}

impl CompletionFeaturesDyn for Recorder {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let inner = self.inner.clone();
        let rounds = self.rounds.clone();
        Box::pin(async move {
            let tool_additions = req.tool_additions.clone();
            let activations_replayed = req
                .raw_history
                .iter()
                .filter(|message| message["role"] == "system")
                .count();
            let output = CompletionFeaturesDyn::completion(&inner, req).await?;
            rounds.lock().unwrap().push(Round {
                tool_additions,
                activations_replayed,
                input_tokens: output.usage.input_tokens,
                cached_tokens: output.usage.cached_tokens,
                tool_calls: output.tool_calls.iter().map(|c| c.name.clone()).collect(),
                raw_history: output.raw_history.clone(),
            });
            Ok(output)
        })
    }

    fn model_name(&self) -> String {
        self.inner.model_name()
    }
}

struct LookupNote;

impl Tool<BaseCtx> for LookupNote {
    type Args = Json;
    type Output = Json;

    fn name(&self) -> String {
        "lookup_note".to_string()
    }

    fn description(&self) -> String {
        "Looks up a stored note by key.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"]
            }),
            strict: None,
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        _args: Self::Args,
        _resources: Vec<anda_core::Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        Ok(ToolOutput::new(json!({"note": "none"})))
    }
}

struct CacheSample {
    label: String,
}

impl Tool<BaseCtx> for CacheSample {
    type Args = Json;
    type Output = Json;

    fn name(&self) -> String {
        "cache_sample".to_string()
    }

    fn description(&self) -> String {
        "Returns the sample label for this run.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({"type": "object", "properties": {}}),
            strict: None,
        }
    }

    async fn call(
        &self,
        _ctx: BaseCtx,
        _args: Self::Args,
        _resources: Vec<anda_core::Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        Ok(ToolOutput::new(json!({"label": self.label})))
    }
}

/// Runs four turns: a long unique history, a warm-up, then a turn that adds
/// `cache_sample` and asks for it. Returns every request and the last reply.
async fn add_a_tool_mid_conversation(
    effort: ModelEffort,
    tool_additions: bool,
) -> (Vec<Round>, String, String) {
    let label = format!("CACHE_SAMPLE_{:032x}", rand::random::<u128>());
    let rounds = Arc::new(Mutex::new(Vec::new()));
    let recorder = Recorder {
        inner: deepseek(effort).with_tool_additions(tool_additions),
        rounds: rounds.clone(),
    };
    let sample = Arc::new(CacheSample {
        label: label.clone(),
    });
    let ctx = EngineBuilder::new()
        .with_model(Model::with_completer(Arc::new(recorder)))
        .register_tool(Arc::new(LookupNote))
        .unwrap()
        .register_tool(sample.clone())
        .unwrap()
        .mock_ctx();

    // Unique records make a hit on someone else's cached prompt impossible.
    let records = (0..96)
        .map(|i| format!("record-{i}: {:032x}", rand::random::<u128>()))
        .collect::<Vec<_>>()
        .join("\n");
    let mut runner = ctx
        .completion_iter(
            CompletionRequest {
                instructions:
                    "You are a careful assistant. Follow the user's instructions exactly."
                        .to_string(),
                prompt: format!(
                    "Keep these reference records. Do not call tools. Reply READY.\n{records}"
                ),
                tools: vec![LookupNote.definition()],
                ..Default::default()
            },
            Vec::new(),
        )
        .unbound();

    let turn = async |runner: &mut anda_engine::context::CompletionRunner| {
        let mut content = String::new();
        while let Some(output) = runner.next().await.unwrap() {
            assert!(output.failed_reason.is_none(), "{:?}", output.failed_reason);
            content = output.content.clone();
            if runner.is_idle() {
                break;
            }
        }
        content
    };
    turn(&mut runner).await;
    runner.follow_up("Keep the reference records. Do not call tools. Reply READY.".to_string());
    turn(&mut runner).await;
    runner.add_tools(vec![sample.definition()]);
    runner.follow_up(
        "Call cache_sample exactly once, then reply with the label it returns and nothing else."
            .to_string(),
    );
    let last = turn(&mut runner).await;
    let rounds = rounds.lock().unwrap().clone();
    (rounds, last, label)
}

#[tokio::test]
#[ignore = "calls the DeepSeek API; set DEEPSEEK_API_KEY and run with --ignored"]
async fn a_tool_added_mid_conversation_keeps_the_cached_prefix() {
    let (rounds, reply, label) = add_a_tool_mid_conversation(ModelEffort::Max, true).await;
    for (i, round) in rounds.iter().enumerate() {
        println!(
            "request {i}: input {} cached {} additions {:?} activations replayed {} calls {:?}",
            round.input_tokens,
            round.cached_tokens,
            round.tool_additions,
            round.activations_replayed,
            round.tool_calls
        );
    }
    let added = rounds
        .iter()
        .position(|round| !round.tool_additions.is_empty())
        .expect("one request adds cache_sample");
    assert_eq!(rounds[added].tool_additions, ["cache_sample"]);
    assert_eq!(rounds[added].raw_history[1]["role"], "system");
    assert_eq!(
        rounds[added].raw_history[1]["content"][0]["type"],
        "tool_addition"
    );
    // The request that adds the tool reads the whole previous request from the cache.
    let before = &rounds[added - 1];
    assert!(
        rounds[added].cached_tokens + CACHE_SUFFIX_TOKENS >= before.input_tokens,
        "added: cached {} of previous input {}",
        rounds[added].cached_tokens,
        before.input_tokens
    );
    // Below max effort the model often answers that the activated tool does not exist, which
    // is why only max-effort requests activate tools. At max it still did once in 46 runs;
    // rerun before suspecting the activation.
    assert!(
        rounds[added]
            .tool_calls
            .contains(&"cache_sample".to_string()),
        "the model did not call the added tool; reply {reply:?}"
    );
    assert!(reply.contains(&label), "reply {reply:?} lacks {label}");
    // The next request replays the activation and keeps reading from the cache.
    let after = &rounds[added + 1];
    assert!(after.tool_additions.is_empty());
    assert_eq!(after.activations_replayed, 1);
    assert!(
        after.cached_tokens + CACHE_SUFFIX_TOKENS >= rounds[added].input_tokens,
        "after: cached {} of previous input {}",
        after.cached_tokens,
        rounds[added].input_tokens
    );
}

#[tokio::test]
#[ignore = "calls the DeepSeek API; set DEEPSEEK_API_KEY and run with --ignored"]
async fn below_max_effort_an_added_tool_is_declared_as_usual() {
    let (rounds, reply, label) = add_a_tool_mid_conversation(ModelEffort::High, true).await;
    for (i, round) in rounds.iter().enumerate() {
        println!(
            "request {i}: input {} cached {} additions {:?} calls {:?}",
            round.input_tokens, round.cached_tokens, round.tool_additions, round.tool_calls
        );
    }
    let added = rounds
        .iter()
        .position(|round| !round.tool_additions.is_empty())
        .expect("one request adds cache_sample");
    assert!(rounds.iter().all(|round| round.activations_replayed == 0));
    assert!(
        rounds[added]
            .raw_history
            .iter()
            .all(|message| message["role"] != "system")
    );
    assert!(
        rounds[added]
            .tool_calls
            .contains(&"cache_sample".to_string()),
        "the model did not call the added tool; reply {reply:?}"
    );
    assert!(reply.contains(&label), "reply {reply:?} lacks {label}");
}

#[tokio::test]
#[ignore = "calls the DeepSeek API; set DEEPSEEK_API_KEY and run with --ignored"]
async fn without_tool_additions_an_added_tool_misses_the_cache() {
    let (rounds, reply, label) = add_a_tool_mid_conversation(ModelEffort::Max, false).await;
    for (i, round) in rounds.iter().enumerate() {
        println!(
            "request {i}: input {} cached {} additions {:?} calls {:?}",
            round.input_tokens, round.cached_tokens, round.tool_additions, round.tool_calls
        );
    }
    assert!(reply.contains(&label), "reply {reply:?} lacks {label}");
    assert!(rounds.iter().all(|round| round.activations_replayed == 0));
    // The baseline the extension exists for: a tool appended to the declarations changes the
    // start of the prompt, so the request that adds it re-reads the conversation uncached.
    let added = rounds
        .iter()
        .position(|round| !round.tool_additions.is_empty())
        .expect("one request adds cache_sample");
    assert!(
        rounds[added].cached_tokens * 2 < rounds[added - 1].input_tokens,
        "added: cached {} of previous input {}",
        rounds[added].cached_tokens,
        rounds[added - 1].input_tokens
    );
}

#[tokio::test]
#[ignore = "calls the DeepSeek API; set DEEPSEEK_API_KEY and run with --ignored"]
async fn deepseek_accepts_every_mapped_effort() {
    for effort in [
        ModelEffort::Minimal,
        ModelEffort::Low,
        ModelEffort::Medium,
        ModelEffort::High,
        ModelEffort::Max,
    ] {
        let model = deepseek(effort);
        let output = CompletionFeaturesDyn::completion(
            &model,
            CompletionRequest {
                prompt: "What is 17 + 25? Reply with the number only.".to_string(),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_else(|err| panic!("{effort}: {err}"));
        println!(
            "{effort}: content {:?} thoughts {} output tokens {}",
            output.content,
            output.thoughts.is_some(),
            output.usage.output_tokens
        );
        assert!(output.failed_reason.is_none(), "{effort}: {output:?}");
        assert!(
            output.content.contains("42"),
            "{effort}: {:?}",
            output.content
        );
        if effort == ModelEffort::Minimal {
            assert!(output.thoughts.is_none(), "minimal still thought");
        }
    }
}
