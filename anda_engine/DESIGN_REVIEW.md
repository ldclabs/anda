# anda_engine 模块设计评审（深模块视角）

- 评审日期：2026-08-06（基于 main 分支，工作树干净）
- 评审方法：以"深模块"（deep module）视角评估各模块的接口/实现比例、接缝（seam）位置与可测性。四个并行子代理分别通读模型层、context 层、extension 层、引擎生命周期层，核心接缝（`anda_core` 契约、`CompletionRunner`、`CompletionFeaturesDyn`）另行人工复核。
- 与 `CODE_REVIEW.md` 的关系：那份清单是逐条缺陷账本；本文是结构层面的账本——多数未清缺陷（runner 17 条、subagent/memory 22 条）集中在下文标记的浅模块与接缝泄漏处，结构修复会让它们成批消失或变得容易修。
- 术语：**模块** = 接口 + 实现；**接口** = 调用方必须知道的一切（签名、不变量、错误模式、配置）；**深** = 小接口藏大实现；**接缝** = 接口所在的位置，适配器（adapter）在接缝处满足接口。

## 总评

**值得保护的深接缝**（改动时不要变宽）：

- `CompletionFeaturesDyn`（model.rs）：2 个核心方法 + 中立的 `CompletionRequest`，4 个内置适配器 + 37 个测试假实现都立在这个接缝上。本次新增的 2 个剪枝方法带默认实现，不增加实现者负担。
- `Tool<C>` / `ToolProvider<C>`（anda_core/tool.rs）：类型化工具 + 动态工具集的双接缝，MCP 整体作为一个 `ToolProvider` 适配器接入，形状正确。
- 工具只拿 `BaseCtx` 不拿 `AgentCtx`（context/agent.rs `child_base`）：整个 context 层最窄的一道口子。
- `Store`（store.rs，5 方法藏命名空间/限额/流式）、`Management`（management.rs，3 方法）。
- 测试基础健康：假 provider 立在正确的接缝上，`mock_ctx` 全内存，469 个测试不依赖真实 provider。

**结构债的集中地**：模型层四份手抄的请求组装/流式聚合（~1300 行同构代码）；`context/agent.rs` 一个文件挤了约五个模块；`memory.rs` 把 AndaDB 类型泄漏成公开接口且一半代码无调用者；`hook.rs` 四个互不相通的 hook 家族；extension 层 8 份手写 schema 与 hook 样板。

## 已实施（2026-08-06）

- [x] **D1. raw_history 剪枝移到 provider 接缝之后**（原 `context/agent.rs:1730-1933` → `model/raw.rs`）
  - 原状：runner 内 ~205 行硬编码 OpenAI Chat/Responses、Anthropic、Gemini 的 wire 形状（`"function_call"`、`"tool_use"`、`"functionCall"`……）做历史剪枝——provider 知识长在接缝错误的一侧，新增 provider 或 provider 新增条目类型时改动无法局限在适配器内（`CODE_REVIEW.md` 2.16 正是此泄漏的症状）。
  - 现状：`CompletionFeaturesDyn` 新增 `prune_unanswered_tool_calls` / `prune_tool_interactions` 两个**可覆盖的默认方法**（默认实现为保守并集，在 `model/raw.rs`），`Model` 透传，runner 经 `self.model.prune_*` 调用。自定义 provider 可用自己的类型化知识覆盖；2.16 类问题（如 Anthropic `server_tool_use` 自应答语义）今后可在对应适配器的覆盖里精确修，不动 runner。四个纯函数测试随实现迁至 `model/raw.rs`。
- [x] **D2. `EngineBuilder` 三个装配终端合并为单一 `assemble()`**（engine.rs）
  - 原状：`build`/`empty` 各 ~85 行近乎逐行相同，`mock_ctx` 是第三份；漂移已经发生过（`empty()` 曾漏调 `info.validate()`，见 CODE_REVIEW 可优化点）。
  - 现状：私有 `assemble(default_agent)` 承载唯一装配路径，`build`/`empty` 变为薄封装（各自保留 default-agent 校验与 `info.validate()` 的语义差异），`context_names()` 提取共享，`mock_ctx` 复用之。净删 ~170 行重复。
- [x] **D3. `MockImplemented::model_name()` 改返 `"mock_implemented"`**——此前与 `NotImplemented` 同名不可区分，而 runner 靠 `model_name()` 判定模型切换。

验证：`cargo test -p anda_engine` 469 passed；`cargo clippy -p anda_engine --all-targets` 无警告；`cargo check --workspace` 通过。

## P1 高杠杆（建议下一批）

- [x] **D4. 模型层：共享补全驱动器 + wire 接缝**（2026-08-06 完成）
  - 新增 `model/driver.rs`：`drive_completion<W: WireFormat>` 持有全部 15 步顺序与 `skip_raw` 不变量（raw 先于 chat_history、skip 点紧随 raw 追加记录、`into_output` 前必排水——此前只存在于 anda_core 文档散文 + 四份手抄），重试/流分支/失败日志同归驱动器。
  - 四个适配器（Anthropic、Gemini、OpenAI Chat、OpenAI Responses）各缩为一个 `WireFormat` 实现（~13 个关联函数，纯 wire 映射：字段名、endpoint、typed parse/aggregate、redaction）+ 3 行 `completion()` 转发。provider 特有怪癖留在各自实现内（deepseek tool_choice 抑制、Chat stream_options 注入、V2 强制 stream/store、Anthropic assistant_raw 捕获）。
  - 行为差异（日志文本级）：Chat 的请求调试日志从 "OpenAI completions request" 归一为 "Completion request"。
  - 后续跟进：Chat/V2 的 `redacted_for_log` 目前保持原状（不脱敏 system 消息），与 Anthropic/Gemini 不一致——统一脱敏是一个待定的行为决策。6.3（Chat 非成功 finish_reason 丢内容）现在可在 Chat 的 `try_into` 单点修。
- [x] **D5. 模型层：共享流式聚合器 →（细读后降级，不立项）**
  - 复核修正两点：(a) 所谓"OpenAI/Gemini 无上限 BTreeMap"并非放大型攻击面——map 条目数与已接收字节成正比，读取端已有 64 MiB 上限；真正的放大风险（Anthropic 单事件触发 `Vec::resize` 大分配）已由 `MAX_STREAM_CONTENT_BLOCKS` 封死。(b) 三家聚合器对应三种真实不同的事件模型（Anthropic typed SSE 事件流 / OpenAI Chat delta builder / OpenAI Responses last-response-wins / Gemini 整块 chunk 合并），强行统一为一个 `StreamAccumulator` 框架会得到泄漏的抽象。
  - 剩余可做的只有修饰性收敛（`merge_opt` 辅助替代 27 个连续 `if x.is_some()`），价值不足以单独立项；如后续动到聚合代码可顺带。
- [x] **D6. 拆分 `context/agent.rs` 文件**（2026-08-06 完成）
  - `context/agent.rs`（1453 行：AgentCtx + 路由词汇 + 自身测试）、`context/runner.rs`（3961 行：常量 + CompletionRunner + CompletionStream + 全部 runner/stream 测试）、`context/test_fixtures.rs`（994 行：共享测试假件，`#[cfg(test)] pub(crate)`）。`pub use runner::*` 保持公共路径不变。
  - 顺带的深化：`CompletionRunner::new()` 取代跨模块 24 字段结构体字面量；`stream_placeholder` 复用 `new()`（消除"手写全部字段易漏改"项）。
- [x] **D7. mcp.rs 三分**（2026-08-06 完成）
  - `mcp/auth.rs`（458 行）：OAuth 配置/凭据存储/`McpAuthorizationRequired`/错误分类，加上从 provider 抽出的 5 个协议自由函数（`discover_http_oauth`、`begin_authorization_manager`、`complete_authorization_exchange`、`authorize_from_store`、`authorize_client_credentials`）——OAuth 协议逻辑与工具分发彻底分离，provider 的授权方法缩为"注册表查找 + pending_auth 记账 + 委托"。
  - `mcp/session.rs`（458 行）：transport 构造（stdio/HTTP）、`McpSession`、协议版本协商、`serve_bounded`、`tools/list_changed` 订阅泵。
  - `mcp/router.rs`（308 行）：本地名映射（sanitize/hash 去碰撞）、MRTR/Tasks 调用轮次（`call_tool_rounds`/`await_task`）、`CallToolResult` → `ToolOutput` 适配。
  - 根 `mcp.rs`（2570 行，其中 ~1200 为原地未动的 42 个测试）：`McpToolProvider` 组合层 + 注册表/索引（互锁状态与锁设计未动）+ 配置类型。公共 API 路径经 re-export 完全不变。
  - 未做（记为后续）：把 `index`{routes,sessions,metas} 拆成独立的 SessionPool/Router 结构体需要重新设计原子替换语义，本次刻意不动；shell 脚本假服务器测试仍打 provider 整体。
- [x] **D8. `WorkspaceScope`：把 fs.rs 从工具袋变成深模块**（2026-08-06 完成）
  - fs.rs 新增 `WorkspaceScope`（`for_call(meta, defaults)` / `open_read` / `open_edit` / `open_write` / `roots` / `into_primary` / `display`）与 `ReadTarget`/`WriteTarget`（`write_atomic` 承担建父目录与权限保留）。四个 fs 工具的前置序列各缩为 2 行；多根优先级、meta 收窄、symlink/hardlink 复查、大小限额与错误文案全部移入 scope 之后。
  - **shell 并轨**：`shell/native.rs` 的第二套发散 sandbox 实现（只认 `workspace` 字符串键）删除，改走 `WorkspaceScope::for_call(...).into_primary()`。行为收敛为 fs 语义（现在也接受 `workspaces` 键与数组/PathBuf 形式——是放宽而非收紧，且与 fs 工具一致）。
  - **可见性收紧**：`tool_workspaces`、`resolve_*_path_in_workspaces`、`workspace_access_error`、`resolve_workspace_path`、`ensure_path_in_workspace(_namespace)`、`path_contains_parent_reference`、`nearest_existing_ancestor`、`format_workspaces`、`normalize_workspaces` 全部降为 fs 模块私有（子模块 search 仍可见）；crate 其他模块只能经 scope 使用 sandbox。`ensure_regular_file`/`ensure_file_size_within_limit`/`normalize_relative_path` 因 skill.rs 使用保持 pub(crate)（后续 skill 也可改走 scope）。
  - 已知的次序微移：write_file 在双重故障场景（stat 失败 + 编码非法同时发生）下现在先报 stat 错误（原先 decode 在 stat 之前）；单一故障路径的错误文案逐字未变。
  - 关联：4.1 的漂移根因消除；4.2、4.6、4.8、4.13 今后只需修 scope 一处。

## P2 中杠杆

- [x] **D9. memory.rs：领域接口 + 隔离 AndaDB 接缝**（2026-08-06 完成）
  - 实测消费面比调研更窄：recorder 只需要 **2 个操作**。消费者侧（`subagent/conversation.rs`）定义端口 `ConversationRecords { create(ConversationRef) -> u64, update(&Conversation) }`，错误归一 `BoxError`；`memory::Conversations` 作为其 AndaDB 适配器实现——`to_runner_changes()` 的 `Fv` 编码移入适配器内，AndaDB 类型不再跨缝进入 subagent。
  - `SubAgentConversationRecorder` 改持 `Arc<dyn ConversationRecords>`，新增 `with_store()` 供自定义存储；`new(Conversations)` 签名不变。`pub conversations: Arc<Collection>` 公开字段在 `Conversations` 与 `MemoryManagement` 上全部收私（绕过抽象的口子关闭；对外是 breaking，发版时记 CHANGELOG）。
  - 新增端口一致性测试 `subagent_records_conversation_through_custom_store`：BTreeMap 假存储跑通完整 subagent 会话记录，证明 subagent 层测试不再需要拉起 AndaDB。
  - 未做（记为后续）：`Conversations`/`MemoryManagement` 固有方法仍返回 `DBError`（公开 API，单独决策）；`Conversation` 仍兼任 schema 与领域对象；零调用的 KIP/CognitiveNexus 面已与被测路径脱钩，去留可独立决策。
  - 关联：3.10、3.16、3.18、3.19 的修复现在都可落在适配器单点。
- [x] **D10. hook.rs 拆分**（2026-08-06 完成拆分部分；两个子项按证据降级/延后）
  - 已做：`PrefixedId`/`BackgroundHandle`/`BackgroundTaskControls` + 其 4 个测试独立为 `src/background.rs`（326 行，模块文档明确"这是注册表，不是 hook"）；hook.rs 缩至 965 行，`pub use crate::background::*` 保持 `crate::hook::BackgroundHandle` 等公共路径不变，5 个消费文件零改动。
  - **降级：`ToolBackgroundHook` 并入 `ToolHook<Json, Json>`**——核算后不划算：两者有真实语义差（类型化视图的 `args: &I` vs 类型擦除视图的 `args: Json` 按值；后者供 dyn-safe 场景），合并需改 session/native/agent 三处实现 + 8 个 UFCS 测试消歧调用（SubSession 同时实现 AgentHook 与 ToolBackgroundHook，合并后消歧更啰嗦），删除测试：拿掉它反而迫使各处摆出 `ToolHook<Json, Json>` 的别扭形状。保留两个 trait。
  - **延后：可追加的类型化 hook 注册表**（真正解决 3.5 的单槽静默覆盖）——与 `BaseCtx::child` 的快照复制状态语义强耦合（hook 注册在长活 ctx 与 per-call 子 ctx 上的可见性规则需先定），应作为独立设计过一遍再动手；现状风险（两个功能争同一槽位互相覆盖）已知且有测试覆盖的路径未受影响。
  - 同样延后：完成循环内 tool 分发不触发引擎级 `on_tool_start/end` 的决策（与注册表统一是同一次设计）。
- [ ] **D11. Tool 适配器统一（`ToolShell<Args, Output>` 或 derive）**
  - 8 份手写 `json!` schema（`gen_schema_for` 已存在但只有 fetch 用）、8 份逐字相同的 hook 前后样板、4 种互不相认的截断信号、三选一的错误约定（`Err` / `is_error` / 输出内嵌 `error` 字段——shell 永不失败、note 在成功输出里报失败）。一个适配器统一 schema 派生、hook 协议、截断信号与错误政策；顺带删掉两个专门看守手写 schema 形状的测试（todo.rs:705、note.rs:890），cancellation 支持从 2/8 变为默认全有。
- [x] **D12. runner：`DiscoveredTools` 提取**（2026-08-06 完成）
  - `context/tool.rs` 新增 `DiscoveredTools`（`observe_output` / `merge_into_request` / `compact_output_for_context` / `contains` / `reset_definitions` / merge 策略读写），runner 的三个字段收敛为一个策略对象、三个私有方法变单行委托。runner 不再 import `TOOLS_SEARCH_NAME`/`TOOLS_SELECT_NAME`/`ToolsOutput`——发现语义（工具名、载荷形状、探测规则）完整归 tool.rs 所有，`runner ↔ tool` 知识循环切断。2.3、2.6、2.8 的修复今后落在这一个类型里。
- [ ] **D13. runner：`resolve_callable(name) -> Callable` 统一分类 →（复核后改为独立设计项）**
  - 复核发现 `agent_run`/`tool_call` 的前缀路由带 miss 回退（前缀命中失败落回本地查找，见 agent.rs 注释），2.17 的爆炸半径比声明的小：只在"跨命名空间同名 + 前缀拼写重叠"时出现劫持。统一分类必须同时选定唯一优先级（静态名优先 vs 前缀优先），无论选哪个都是重叠场景下的行为变更，且落在每次工具调用的最热路径上——需要连同影子矩阵测试（2.4/2.17/3.17 的组合场景）一起做，不适合顺带。
- [x] **D14. `Engine::agent_run`/`tool_call` 共享前置检查**（2026-08-06 完成）
  - 逐字相同的 engine-id + user 校验两段抽为 `Engine::validate_request_meta`。可见性/名字检查顺序保持各自现状（agent_run 的防枚举顺序是有意的；tool_call 的 export 检查已用同文案遮蔽未导出工具，人工复核不构成枚举信道），hook 配对收尾因输出类型不同各自保留。
- [x] **D15. subagent 层收敛（驱动循环部分，2026-08-06 完成）**
  - `SubAgent::run` 的两份阻塞驱动循环统一为 `drive_to_completion(runner, Option<&mut SubAgentConversationLog>)`（None 即无记录路径；plain 路径顺带获得与记录路径一致的提前终止判定）；6 处重复的 `after_agent_run` 尾巴收敛为 `finish_with_hook`（带失败记录的 match 变体属合理差异保留）。session.rs 的循环是真实不同的生命周期（idle 心跳/后台任务门控），不强并。
  - **决策：`SubAgentSet`/`SubAgentSetManager` 暂保留**——in-repo 只有一个实现者（投机泛化成立），但它是面向下游的公开扩展点，折叠属 breaking API 决策，归入 1.0 API 审查一并定。`SubAgent` 配置/分发器分离仍开放。关联：3.3、3.12、3.21 未动。

## P3 低成本清理

- [x] **D16.**（2026-08-06 完成）删除 Anthropic/Gemini 的 `CompletionFeatures` 空转发 impl（`resources` 参数本就被忽略、仅测试用它消歧 `model_name`）。provider 接缝从此唯一：`CompletionFeaturesDyn`；`CompletionFeatures` 归 `AgentCtx`。对外属 breaking（直接经该 trait 调用两个模型的下游需换用 `CompletionFeaturesDyn`），发版记 CHANGELOG。
- [x] **D17.**（2026-08-06 完成 completer 部分）新增公开模块 `model::testing::ScriptedCompleter`：按调用出队的可编程回复（固定输出/错误/闭包）+ 请求录制 + 脚本耗尽后回声兜底，覆盖现有 37 个手写假件的全部行为模式；非 `cfg(test)`，下游可用。存量假件可在触碰到时增量迁移，不做一次性替换。conformance suite（同一请求过四适配器）另立后续——需为四家各造最小合法 wire fixture，价值边际（各家已有过缝测试）。
- [ ] **D18.** `AgentCtx` 转发样板 →（宏方案复核后否决）这些转发方法签名各异（泛型、where 子句、`impl Future` 返回），`macro_rules` 必须逐个重复完整签名，每方法净省约 1 行，代价是 grep/IDE 可达性下降——负收益，保留显式转发。仍然成立的部分：`pub base`/`pub label` 收窄 + 把 `get_state`/`set_state`/`agent`/`cancellation_token` 提升为 `AgentCtx` 方法，属 breaking，归入下一个大版本。
- [x] **D19.**（2026-08-06 以降级形式完成）复核发现注入点已存在：`handoff(Option<String>)` 本就接受自定义压缩提示语，`needs_compaction_with` 单一调用者——`CompactionPolicy` 结构体是"一个适配器的假想接缝"，不引入。实际修复的是文档债：81 轮上限与 80% 阈值/100k 兜底的依据已写成注释（消掉 CODE_REVIEW"缺注释依据"项）。
- [ ] **D20.** 分层倒置（context → extension::todo）→（接受并记档）候选修复是"由调用方注入播种器"，但这把每个 agent 作者都变成必须记得播种的调用方——接口变宽换分层纯洁，负收益。todo 是引擎内置扩展、与 runner 生命周期天然耦合；维持现状，若未来出现第二个需要会话级播种的扩展，再上通用播种机制（那时是两个适配器，接缝才真实）。
- [x] **D21.**（2026-08-06 以决策形式完成）判定为"非死契约"：这组函数是 `ModelError` 面向嵌入方的公开检查面（`CompletionFeaturesDyn::completion` 文档明确指向它），内置适配器返回时已耗尽自身传输重试，引擎不再自动重试是有意设计——延迟重试是应用层策略。已在 `is_retryable_model_error` 文档写明预期消费者与这一设计意图，防止再被当死代码。
- [ ] **D22.** 时间不可注入（`unix_ms()` ~40 处直呼）→（延后）不是低成本项：Clock 注入触及 hook/session/memory 三层的签名或状态传递，应在下次触碰 subagent 计时行为（3.21 等）时一并做，单独扫尾不值。
- [x] **D23.**（2026-08-06 以反向决策完成）`mock_ctx` 不加门、**转正为受支持的测试 API**：它是下游 agent/tool 作者在自己测试里获得 `AgentCtx` 的唯一入口（与 D17 的 `model::testing` 同向），加 `cfg(test)` 会砍掉正当用途。已删除被注释的门、重写文档说明支持范围与非生产定位。

## 推进顺序建议

D6（纯机械拆文件）→ D4+D5（模型层收敛，互相独立可并行）→ D8（WorkspaceScope）→ D7（MCP 三分）→ D9（memory 接缝）→ D10（hook 统一）→ 其余按接触到的模块顺带清。每步都应满足：接口变小或不变、`cargo test -p anda_engine` 全绿、行为差异写进 CHANGELOG。
