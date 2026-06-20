# Anda Engine Architecture

This document describes the current `anda_engine` runtime architecture from the source code. It replaces the older deployment-first narrative that treated ICP, TEE, and IC-TEE as mandatory runtime blocks.

In the current engine, those integrations are optional backing capabilities. The practical center of the system is the `Engine` runtime: it validates callers, creates scoped contexts, dispatches agents and tools, routes model calls, handles tool-call loops, and exposes selected local or remote functions.

Preview note: use [markdown-viewer/markdown-viewer-extension](https://github.com/markdown-viewer/markdown-viewer-extension) to preview this document with the embedded HTML architecture diagram and PlantUML sequence diagram rendered.

Source map:

- [`engine.rs`](../anda_engine/src/engine.rs): top-level `Engine`, `EngineBuilder`, exported APIs, management checks, hooks, challenge signing.
- [`context/agent.rs`](../anda_engine/src/context/agent.rs): `AgentCtx`, local/remote/subagent routing, `CompletionRunner`, `CompletionStream`.
- [`context/base.rs`](../anda_engine/src/context/base.rs): `BaseCtx`, scoped state, cache, store, keys, HTTP, signed RPC, cancellation.
- [`context/tool.rs`](../anda_engine/src/context/tool.rs): built-in discovery agents: `tools_groups`, `tools_search`, and `tools_select`.
- [`model.rs`](../anda_engine/src/model.rs): `Models` label router, provider adapters, retry and streaming helpers.
- [`subagent.rs`](../anda_engine/src/subagent.rs): reusable subagents, background sessions, compaction and handoff.
- [`memory.rs`](../anda_engine/src/memory.rs): conversation/resource storage and KIP/Cognitive Nexus tools.
- [`extension`](../anda_engine/src/extension.rs): built-in tool libraries such as filesystem, shell, fetch, skills, notes, todos, and memory.

## Runtime View

<style scoped>
.anda-arch{font-family:Inter,Segoe UI,Arial,sans-serif;color:#18202b;background:#f7f9fb;border:1px solid #d9e2ec;border-radius:8px;padding:18px;margin:16px 0;box-shadow:0 10px 28px rgba(24,32,43,.08)}
.anda-arch *{box-sizing:border-box}
.anda-title{font-size:22px;font-weight:760;letter-spacing:0;margin:0 0 4px;color:#101820}
.anda-subtitle{font-size:13px;color:#52606d;margin:0 0 16px}
.anda-layout{display:grid;grid-template-columns:minmax(168px,.75fr) minmax(420px,2.2fr) minmax(190px,.85fr);gap:12px;align-items:stretch}
.anda-panel{background:#ffffff;border:1px solid #d9e2ec;border-radius:8px;padding:10px}
.anda-panel-title{font-size:12px;text-transform:uppercase;letter-spacing:.04em;font-weight:780;color:#52606d;margin-bottom:8px}
.anda-main{display:flex;flex-direction:column;gap:10px}
.anda-layer{border:1px solid #d9e2ec;border-left-width:5px;border-radius:8px;background:#ffffff;padding:10px}
.anda-layer.entry{border-left-color:#3b82f6}
.anda-layer.control{border-left-color:#64748b}
.anda-layer.registry{border-left-color:#22a06b}
.anda-layer.runner{border-left-color:#d97706}
.anda-layer.context{border-left-color:#0f766e}
.anda-layer.model{border-left-color:#b45309}
.anda-layer.storage{border-left-color:#7c3aed}
.anda-layer-title{font-size:14px;font-weight:780;color:#18202b;margin-bottom:8px}
.anda-grid{display:grid;gap:8px}
.anda-grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.anda-grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.anda-box{background:#f8fafc;border:1px solid #d9e2ec;border-radius:7px;padding:9px;min-height:58px;font-size:12px;line-height:1.35;color:#243b53}
.anda-box strong{display:block;font-size:12.5px;color:#102a43;margin-bottom:3px}
.anda-box small{display:block;color:#627d98}
.anda-pill{display:inline-block;background:#edf2f7;border:1px solid #cbd5e1;border-radius:999px;padding:3px 7px;font-size:11px;color:#334155;margin:2px 3px 2px 0}
.anda-side-list{display:flex;flex-direction:column;gap:7px}
.anda-side-item{border:1px solid #d9e2ec;background:#f8fafc;border-radius:7px;padding:8px;font-size:12px;line-height:1.35;color:#334e68}
.anda-note{margin-top:12px;border-top:1px solid #d9e2ec;padding-top:10px;font-size:12px;color:#52606d}
@media (max-width:900px){.anda-layout{grid-template-columns:1fr}.anda-grid.two,.anda-grid.three{grid-template-columns:1fr}}
</style>
<div class="anda-arch">
<div class="anda-title">Anda Engine Runtime Architecture</div>
<div class="anda-subtitle">Current source-level view: agents, tools, contexts, models, memory, and optional external capabilities.</div>
<div class="anda-layout">
<div class="anda-panel">
<div class="anda-panel-title">Entrypoints</div>
<div class="anda-side-list">
<div class="anda-side-item"><strong>Host apps</strong><br>CLI, HTTP server, bot runtime, or embedded Rust application call the engine API.</div>
<div class="anda-side-item"><strong>Public API</strong><br><span class="anda-pill">agent_run</span><span class="anda-pill">tool_call</span><span class="anda-pill">information</span><span class="anda-pill">challenge</span></div>
<div class="anda-side-item"><strong>Remote peers</strong><br>Other engines can discover exported functions and call them through signed RPC.</div>
</div>
</div>
<div class="anda-main">
<div class="anda-layer entry">
<div class="anda-layer-title">Engine Boundary</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Engine</strong><small>Owns runtime state, default agent, export lists, hooks, management policy.</small></div>
<div class="anda-box"><strong>EngineBuilder</strong><small>Registers tools, agents, models, store, remote engines, subagents, and hooks.</small></div>
<div class="anda-box"><strong>EngineCard</strong><small>Publishes exported agent/tool definitions for remote discovery.</small></div>
</div>
</div>
<div class="anda-layer control">
<div class="anda-layer-title">Access Control and Observation</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Management</strong><small>Private, protected, or public visibility; controller and manager principals.</small></div>
<div class="anda-box"><strong>Hooks</strong><small>on_agent_start/end and on_tool_start/end can reject, observe, or transform outputs.</small></div>
<div class="anda-box"><strong>Cancellation</strong><small>Root and child cancellation tokens propagate through contexts and runners.</small></div>
</div>
</div>
<div class="anda-layer registry">
<div class="anda-layer-title">Callable Registries</div>
<div class="anda-grid three">
<div class="anda-box"><strong>AgentSet</strong><small>Local agents, including built-in `tools_groups`, `tools_search`, `tools_select`, and `subagents_manager`.</small></div>
<div class="anda-box"><strong>ToolSet / ToolProviderSet</strong><small>Static tools and runtime-discovered providers with function definitions, resource tags, and capability groups.</small></div>
<div class="anda-box"><strong>RemoteEngines</strong><small>Remote function metadata routed with `RA_` and `RT_` prefixes.</small></div>
</div>
</div>
<div class="anda-layer runner">
<div class="anda-layer-title">AgentCtx and CompletionRunner</div>
<div class="anda-grid three">
<div class="anda-box"><strong>AgentCtx</strong><small>Combines BaseCtx with models, tools, agents, subagents, and routing helpers.</small></div>
<div class="anda-box"><strong>CompletionRunner</strong><small>Iterates model turns, executes tool calls, accumulates usage/artifacts, and returns final output.</small></div>
<div class="anda-box"><strong>SubAgent sessions</strong><small>`SA_` workers support blocking calls or background sessions with progress/final hooks.</small></div>
</div>
</div>
<div class="anda-layer context">
<div class="anda-layer-title">BaseCtx Capability Surface</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Scoped state</strong><small>Caller, request meta, elapsed time, typed state extensions, and depth-limited children.</small></div>
<div class="anda-box"><strong>Store and cache</strong><small>Context-path namespaces isolate agent and tool data in object store and cache.</small></div>
<div class="anda-box"><strong>External calls</strong><small>HTTP, signed RPC, key derivation/signing, and canister calls through a configured Web3SDK.</small></div>
</div>
</div>
<div class="anda-layer model">
<div class="anda-layer-title">Model Routing and Provider Adapters</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Models</strong><small>Label map plus primary model. Labels such as `pro`, `flash`, or `lite` choose provider entries.</small></div>
<div class="anda-box"><strong>Adapters</strong><small>OpenAI-compatible, Anthropic, Gemini, and custom `CompletionFeaturesDyn` providers.</small></div>
<div class="anda-box"><strong>Reliability</strong><small>Request defaults, SSE/NDJSON parsing, one short retry, and retryable `ModelError` signals.</small></div>
</div>
</div>
<div class="anda-layer storage">
<div class="anda-layer-title">Optional Extensions and Persistence</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Built-in tools</strong><small>fetch, filesystem, shell, note, skill, todo, and memory tools register like any other tool.</small></div>
<div class="anda-box"><strong>Memory</strong><small>Conversation/resource records in AndaDB; KIP commands backed by Cognitive Nexus.</small></div>
<div class="anda-box"><strong>ObjectStore</strong><small>In-memory by default; local, cloud, or IC-COSE-compatible backends can be supplied.</small></div>
</div>
</div>
</div>
<div class="anda-panel">
<div class="anda-panel-title">External Capabilities</div>
<div class="anda-side-list">
<div class="anda-side-item"><strong>Model providers</strong><br>Completion APIs are reached only through registered model adapters.</div>
<div class="anda-side-item"><strong>Web3SDK</strong><br>Can be a TEE client, a Web3 client, or a not-implemented placeholder.</div>
<div class="anda-side-item"><strong>HTTP resources</strong><br>Fetch and remote-engine calls use the context HTTP/signed-RPC traits.</div>
<div class="anda-side-item"><strong>Databases</strong><br>AndaDB and Cognitive Nexus are used when memory tools are registered.</div>
</div>
</div>
</div>
<div class="anda-note">Key point: the engine does not require a blockchain, TEE, or specific storage backend to schedule agents. Those are replaceable integrations behind `Web3SDK`, `Store`, model providers, or memory extensions.</div>
</div>

## Request Sequence

```plantuml
@startuml
title Anda Engine agent_run and completion loop
skinparam backgroundColor #FFFFFF
skinparam sequenceArrowColor #334155
skinparam sequenceLifeLineBorderColor #CBD5E1
skinparam sequenceParticipantBorderColor #CBD5E1
skinparam sequenceParticipantBackgroundColor #F8FAFC
skinparam sequenceGroupBorderColor #94A3B8
skinparam sequenceGroupBackgroundColor #F8FAFC
skinparam noteBackgroundColor #FFF7ED
skinparam noteBorderColor #FDBA74
actor Caller
participant "Host API\n(server / CLI / app)" as Host
participant "Engine" as Engine
participant "Management\n+ Hooks" as Guard
participant "AgentCtx\n+ BaseCtx" as Ctx
participant "Agent" as Agent
participant "CompletionRunner" as Runner
participant "Models\nlabel router" as Models
participant "Model adapter\nOpenAI / Anthropic / Gemini" as Model
participant "Tool / Agent\nregistries" as Registry
participant "Local Tool\nor Agent" as Local
participant "Remote Engine" as Remote
participant "SubAgent\nsession runner" as SubSession
Caller -> Host : submit AgentInput
Host -> Engine : agent_run(caller, input)
Engine -> Engine : validate RequestMeta\nnormalize default agent
Engine -> Guard : check_visibility(caller)
Guard --> Engine : visibility accepted
Engine -> Ctx : ctx_with(caller, agent, label, meta)
Engine -> Guard : on_agent_start(ctx, agent)
Engine -> Agent : run(ctx, prompt, resources)
Agent -> Ctx : completion(req, resources)
Ctx -> Runner : completion_iter(req, resources)
loop until final output, failure, or cancellation
Runner -> Models : resolve(req.model or ctx.label)
Models --> Runner : Model
Runner -> Model : completion(CompletionRequest)
Model --> Runner : AgentOutput\ncontent, usage, raw_history, tool_calls
alt model returned tool_calls
Runner -> Registry : resolve each tool_call name\nlocal, RA_, RT_, or SA_
par each resolved call
alt local tool
Runner -> Ctx : child_base(tool)
Ctx -> Local : Tool::call(BaseCtx, args, selected resources)
Local --> Runner : ToolOutput
else local agent or subagent without session
Runner -> Ctx : child(agent)
Ctx -> Local : Agent::run(AgentCtx, prompt, resources)
Local --> Runner : AgentOutput as ToolOutput
else remote callable
Runner -> Ctx : remote_tool_call or remote_agent_run
Ctx -> Remote : https_signed_rpc(tool_call / agent_run)
Remote --> Runner : ToolOutput or AgentOutput
else subagent session mode
Runner -> Local : SubAgent::run(session args)
Local -> SubSession : claim session and spawn background runner
Local --> Runner : ack with session id
SubSession -> Guard : on_background_start
SubSession -> Runner : unbound completion loop
SubSession -> Guard : on_background_progress / on_background_end
end
end
Runner -> Runner : accumulate usage, artifacts, tools_usage\nappend tool outputs to next request
else no tool calls
Runner -> Runner : final_output()
end
opt steering or follow-up queued
Runner -> Runner : insert user content at safe boundary\nprune unanswered tool raw history if needed
end
end
Runner --> Agent : AgentOutput
Agent --> Engine : AgentOutput
Engine -> Guard : on_agent_end(ctx, agent, output)
Guard --> Engine : transformed output
Engine -> Engine : clear provider raw_history
Engine --> Host : AgentOutput
Host --> Caller : response
@enduml
```

## Component Notes

- `Engine` is the public runtime boundary. It enforces exported agent/tool lists for non-manager callers and always exports the default agent.
- `EngineBuilder` starts with in-memory storage, no implemented Web3 client, no external model, and built-in discovery/subagent control agents.
- `AgentCtx` is the main scheduling surface. It exposes local tools, dynamic tool providers, local agents, subagents, registered remote engines, and dynamic remote engines from cache.
- `CompletionRunner` is iterative. A model turn can return tool calls; the runner executes them and feeds tool outputs into the next model turn. Long-running runners can compact oversized history into a continuation handoff and resume from that summary.
- `tools_groups`, `tools_search`, and `tools_select` are agents, not side channels. `tools_groups` returns a compact directory of visible capability bundles; `tools_select` can expand one group into schemas, and discovered schemas stay in tool-output context while repeated payloads are compacted from conversation context.
- `BaseCtx` creates namespace-scoped child contexts. Agent paths use `a_<agent>`, tool paths use `t_<tool>`, and all store/cache operations are resolved under that path.
- `Models` routes by label first and then falls back to the primary/default model. Provider-specific names stay inside adapter configuration.
- `SubAgentManager` turns persisted or temporary `SubAgent` definitions into callable `SA_<name>` agents. Long-running subagent sessions use hooks to push progress and final output.
- Memory is an extension layer. Conversation/resource storage uses AndaDB collections, and persistent knowledge operations are exposed as KIP tools backed by Cognitive Nexus.
- Web3, TEE, ICP, and IC-COSE integrations are implementation choices behind `Web3SDK`, `HttpFeatures`, `KeysFeatures`, `CanisterCaller`, or `ObjectStore`. They are not mandatory architecture layers for the engine itself.
