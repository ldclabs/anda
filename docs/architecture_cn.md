# Anda Engine 架构

本文基于当前 `anda_engine` 源码描述运行时架构，替换旧文档中把 ICP、TEE、IC-TEE 作为必需主路径的部署叙述。

当前引擎的实际核心是 `Engine` 运行时：校验调用者、创建隔离上下文、调度 agents 和 tools、路由模型请求、处理模型返回的 tool calls，并把需要暴露的本地或远程函数发布出去。Web3、TEE、ICP、IC-COSE 这类能力只是可替换的后端集成，不是理解或运行 engine 的前置条件。

预览提示：建议使用 [markdown-viewer/markdown-viewer-extension](https://github.com/markdown-viewer/markdown-viewer-extension) 预览本文档，以正确渲染内嵌 HTML 架构图和 PlantUML 时序图。

源码索引：

- [`engine.rs`](../anda_engine/src/engine.rs)：顶层 `Engine`、`EngineBuilder`、导出 API、管理策略、hooks、challenge 签名。
- [`context/agent.rs`](../anda_engine/src/context/agent.rs)：`AgentCtx`、本地/远程/subagent 路由、`CompletionRunner`、`CompletionStream`。
- [`context/base.rs`](../anda_engine/src/context/base.rs)：`BaseCtx`、隔离 state、cache、store、keys、HTTP、signed RPC、cancellation。
- [`context/tool.rs`](../anda_engine/src/context/tool.rs)：内置发现 agents：`tools_groups`、`tools_search` 和 `tools_select`。
- [`model.rs`](../anda_engine/src/model.rs)：`Models` 标签路由、provider adapters、重试和 streaming 解析。
- [`subagent.rs`](../anda_engine/src/subagent.rs)：可复用 subagents、后台 sessions、compaction 和 handoff。
- [`memory.rs`](../anda_engine/src/memory.rs)：conversation/resource 存储和 KIP/Cognitive Nexus tools。
- [`extension`](../anda_engine/src/extension.rs)：内置工具库，例如 filesystem、shell、fetch、skills、notes、todos、memory。

## 运行时视图

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
<div class="anda-subtitle">当前源码视角：agents、tools、contexts、models、memory，以及可选外部能力。</div>
<div class="anda-layout">
<div class="anda-panel">
<div class="anda-panel-title">入口</div>
<div class="anda-side-list">
<div class="anda-side-item"><strong>宿主应用</strong><br>CLI、HTTP server、bot runtime，或嵌入式 Rust 应用调用 engine API。</div>
<div class="anda-side-item"><strong>公开 API</strong><br><span class="anda-pill">agent_run</span><span class="anda-pill">tool_call</span><span class="anda-pill">information</span><span class="anda-pill">challenge</span></div>
<div class="anda-side-item"><strong>远程 peers</strong><br>其他 engines 可以发现导出的函数，并通过 signed RPC 调用。</div>
</div>
</div>
<div class="anda-main">
<div class="anda-layer entry">
<div class="anda-layer-title">Engine 边界</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Engine</strong><small>持有 runtime state、default agent、export lists、hooks、management policy。</small></div>
<div class="anda-box"><strong>EngineBuilder</strong><small>注册 tools、agents、models、store、remote engines、subagents、hooks。</small></div>
<div class="anda-box"><strong>EngineCard</strong><small>把已导出的 agent/tool definitions 发布给远程发现。</small></div>
</div>
</div>
<div class="anda-layer control">
<div class="anda-layer-title">访问控制和观测</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Management</strong><small>Private、protected、public 可见性，以及 controller/manager principals。</small></div>
<div class="anda-box"><strong>Hooks</strong><small>on_agent_start/end 和 on_tool_start/end 可以拒绝、观测或改写输出。</small></div>
<div class="anda-box"><strong>Cancellation</strong><small>根 token 和 child tokens 通过 contexts 与 runners 传递。</small></div>
</div>
</div>
<div class="anda-layer registry">
<div class="anda-layer-title">可调用对象注册表</div>
<div class="anda-grid three">
<div class="anda-box"><strong>AgentSet</strong><small>本地 agents，包括内置 `tools_groups`、`tools_search`、`tools_select`、`subagents_manager`。</small></div>
<div class="anda-box"><strong>ToolSet / ToolProviderSet</strong><small>静态 tools 和运行时发现的 providers，提供 function definitions、resource tags 和 capability groups。</small></div>
<div class="anda-box"><strong>RemoteEngines</strong><small>远程函数元数据，通过 `RA_` 和 `RT_` 前缀路由。</small></div>
</div>
</div>
<div class="anda-layer runner">
<div class="anda-layer-title">AgentCtx 和 CompletionRunner</div>
<div class="anda-grid three">
<div class="anda-box"><strong>AgentCtx</strong><small>把 BaseCtx 与 models、tools、agents、subagents、routing helpers 组合起来。</small></div>
<div class="anda-box"><strong>CompletionRunner</strong><small>迭代模型回合，执行 tool calls，累计 usage/artifacts，生成 final output。</small></div>
<div class="anda-box"><strong>SubAgent sessions</strong><small>`SA_` workers 支持同步调用，也支持带 progress/final hooks 的后台 session。</small></div>
</div>
</div>
<div class="anda-layer context">
<div class="anda-layer-title">BaseCtx 能力面</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Scoped state</strong><small>caller、request meta、elapsed time、typed state extensions、depth-limited children。</small></div>
<div class="anda-box"><strong>Store and cache</strong><small>基于 context path 的命名空间隔离 agent/tool 数据。</small></div>
<div class="anda-box"><strong>External calls</strong><small>HTTP、signed RPC、key derivation/signing、canister calls 通过配置的 Web3SDK 提供。</small></div>
</div>
</div>
<div class="anda-layer model">
<div class="anda-layer-title">模型路由和 Provider Adapters</div>
<div class="anda-grid three">
<div class="anda-box"><strong>Models</strong><small>标签映射加 primary model。`pro`、`flash`、`lite` 等标签选择 provider entry。</small></div>
<div class="anda-box"><strong>Adapters</strong><small>OpenAI-compatible、Anthropic、Gemini，以及自定义 `CompletionFeaturesDyn` providers。</small></div>
<div class="anda-box"><strong>Reliability</strong><small>请求默认值、SSE/NDJSON 解析、一次短重试、retryable `ModelError` 信号。</small></div>
</div>
</div>
<div class="anda-layer storage">
<div class="anda-layer-title">可选 Extensions 和持久化</div>
<div class="anda-grid three">
<div class="anda-box"><strong>内置工具</strong><small>fetch、filesystem、shell、note、skill、todo、memory tools 和其他工具一样注册。</small></div>
<div class="anda-box"><strong>Memory</strong><small>Conversation/resource records 存在 AndaDB；KIP commands 由 Cognitive Nexus 支撑。</small></div>
<div class="anda-box"><strong>ObjectStore</strong><small>默认 in-memory；也可以替换为 local、cloud、IC-COSE-compatible backends。</small></div>
</div>
</div>
</div>
<div class="anda-panel">
<div class="anda-panel-title">外部能力</div>
<div class="anda-side-list">
<div class="anda-side-item"><strong>Model providers</strong><br>Completion APIs 只通过已注册 model adapters 访问。</div>
<div class="anda-side-item"><strong>Web3SDK</strong><br>可以是 TEE client、Web3 client，或 not-implemented placeholder。</div>
<div class="anda-side-item"><strong>HTTP resources</strong><br>Fetch 和 remote-engine calls 使用 context HTTP/signed-RPC traits。</div>
<div class="anda-side-item"><strong>Databases</strong><br>只有注册 memory tools 时才会用到 AndaDB 和 Cognitive Nexus。</div>
</div>
</div>
</div>
<div class="anda-note">关键点：engine 调度 agents 并不依赖 blockchain、TEE 或特定 storage backend。这些都只是 `Web3SDK`、`Store`、model providers、memory extensions 后面的可替换集成。</div>
</div>

## 请求时序

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

## 组件说明

- `Engine` 是公开运行时边界。它对非 manager 调用者执行 exported agent/tool lists 检查，并自动导出 default agent。
- `EngineBuilder` 默认使用 in-memory store、not-implemented Web3 client、无外部模型，并注册 discovery/subagent control agents。
- `AgentCtx` 是主要调度面。它暴露本地 tools、动态 tool providers、本地 agents、subagents、已注册远程 engines，以及从 cache 动态加载的远程 engines。
- `CompletionRunner` 是迭代式执行器。模型回合可以返回 tool calls；runner 执行它们，再把 tool outputs 回灌到下一轮模型请求。
- `tools_groups`、`tools_search` 和 `tools_select` 是 agents，不是旁路机制。`tools_groups` 返回当前可见 capability bundles 的紧凑目录；`tools_select` 可以把一个 group 展开成 schemas，发现到的 schemas 仍保留在 tool-output context 中，并压缩 conversation context 中重复的 schema payload。
- `BaseCtx` 创建命名空间隔离的 child contexts。Agent 路径使用 `a_<agent>`，tool 路径使用 `t_<tool>`，store/cache 操作都在该 path 下解析。
- `Models` 先按 label 路由，再回落到 primary/default model。provider 真实模型名留在 adapter 配置内部。
- `SubAgentManager` 把持久化或临时 `SubAgent` 定义转成可调用的 `SA_<name>` agents。长任务 session 通过 hooks 推送 progress 和 final output。
- Memory 是 extension 层。Conversation/resource 存储使用 AndaDB collections，长期知识操作作为 KIP tools 暴露，并由 Cognitive Nexus 支撑。
- Web3、TEE、ICP、IC-COSE 等集成只是 `Web3SDK`、`HttpFeatures`、`KeysFeatures`、`CanisterCaller` 或 `ObjectStore` 后面的实现选择，不是 engine 本身的必需架构层。
