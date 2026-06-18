# `Anda`

> 一个基于 Rust 构建的可组合 AI 智能体运行时框架。

## 说明文档翻译

[English readme](./README.md) | [中文说明](./README_CN.md) | [日本語の説明](./README_JA.md)

## 简介

Anda 是一个用于构建 AI 智能体的 Rust 框架，可以把模型、工具、记忆和其他智能体组合进同一个运行时。它关注可组合性、类型安全的扩展点、异步执行，以及可控的运行时调度。

核心引擎支持开发者注册 agents 和 tools，按能力标签路由模型请求，调用本地或远程函数，隔离上下文状态，并在应用需要时接入可选的持久化和记忆层。

![Anda Diagram](./anda_diagram.webp)

## 核心特性

1. **可组合的 agents 和 tools**
   Agents 和 tools 通过稳定的 traits 与 function definitions 注册，专用组件可以组合成更大的工作流，而不需要把运行时写死成某一种应用形态。

2. **模型路由**
   引擎可以通过 `primary`、`pro`、`flash`、`lite` 等能力标签路由 completion 请求，具体 provider adapter 隐藏在统一的请求和输出契约后面。

3. **运行时编排**
   `CompletionRunner` 负责迭代模型回合、执行 tool calls、调用 agents、累计 usage、汇总 artifacts、处理 steering/follow-up messages 和 cancellation。

4. **隔离的执行上下文**
   `BaseCtx` 和 `AgentCtx` 为每个 agent 或 tool 提供隔离 state、cache、object storage、HTTP 调用、signed calls、cancellation 和 child contexts。

5. **可扩展的记忆与技能**
   可选 extensions 提供 conversation storage、基于 KIP 的 memory tools、filesystem、shell、fetch、notes、todos，以及文件驱动的 skills。

6. **支持发现流程的工具组**
   静态 tools、动态 providers 和 MCP servers 可以暴露 capability groups，让 agents 先用 `tools_groups` 浏览相关工具包，再在需要 schema 时通过 `tools_select` 展开某个 group。

## 项目说明

文档：
- [Anda 架构设计](./docs/architecture_cn.md)

### 项目结构

```sh
anda/
├── anda_cli/              # 与 Anda 引擎服务交互的命令行工具
├── anda_core/             # 核心 traits、类型和运行时契约
├── anda_engine/           # 智能体运行时、编排、上下文、模型和扩展
└── anda_engine_server/    # 支持一个或多个 Anda 引擎的 HTTP 服务
```

### 如何使用和参与贡献

#### 应用构建者：

使用 `anda_cli` 和 `anda_engine_server` 运行并访问已配置的引擎。

#### 开发者：

- 使用 `anda_core` 的 traits 构建自定义 agents 和 tools。
- 为 `anda_engine` 扩展可复用的运行时能力。
- 改进模型 adapters、上下文能力、记忆集成和服务端 API。

### 基于 Anda 的产品

- [Anda Brain](https://github.com/ldclabs/anda-brain): 基于 Anda 框架构建的持久记忆与认知产品。
- [Anda Bot](https://github.com/ldclabs/anda-bot): 基于 Anda 框架构建的个人 AI 助手和应用运行时。

### 关联项目

- [KIP](https://github.com/ldclabs/KIP): Anda memory tools 使用的 Knowledge Interaction Protocol。

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See [LICENSE](./LICENSE-MIT) for the full license text.
