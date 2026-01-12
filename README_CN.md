# `Anda`

> 🤖 一个基于 Rust 构建的 AI 智能体框架，由 ICP 区块链和 TEE 环境赋能。

## 🌍 说明文档翻译

[English readme](./README.md) | [中文说明](./README_CN.md) | [日本語の説明](./README_JA.md)

## 🐼 简介

Anda 是一个基于 Rust 构建的 AI 智能体框架，集成了 ICP 区块链并支持 TEE 环境。
它旨在构建一个高度可组合、自主运行且具备持续记忆能力的 AI 智能体网络。
通过连接跨行业的智能体，Anda 致力于打造一个超级 AGI 系统，推动人工智能向更高层次发展。

![Anda Diagram](./anda_diagram.webp)

### ✨ 核心特性

1. **可组合性**：
   Anda 智能体专注于解决特定领域的问题，并通过灵活组合不同的智能体来应对复杂任务。当单个智能体无法独立解决问题时，它能够与其他智能体协作，形成强大的问题解决网络。这种模块化设计使得 Anda 能够灵活应对多样化的需求。

2. **简洁性**：
   Anda 的设计理念强调简洁易用，旨在帮助开发者快速构建功能强大且高效的智能体。同时，非开发者也可以通过简单的配置创建自己的智能体，降低了技术门槛，使更多人能够参与到智能体的开发与应用中。

3. **可信性**：
   Anda 智能体运行在基于可信执行环境（TEEs）的去中心化可信计算环境（dTEE）中，确保了智能体的安全性、隐私性和数据完整性。这种架构为智能体的运行提供了高度可信的基础设施，保障了数据和计算过程的安全。

4. **自主性**：
   Anda 智能体从 ICP 区块链获取永久身份和加密能力，并结合大语言模型的思考和决策能力，使其能够根据自身的经验和知识自主、高效地解决问题。这种自主性使智能体能够适应动态环境，并在复杂场景中做出高效决策。

5. **永久记忆**：
   Anda 智能体的记忆状态存储在 ICP 区块链和 dTEE 的可信存储网络中，确保其能够持续升级算法、积累知识并不断进化。这种永久记忆机制使智能体能够长久运行，甚至实现“永生”，为构建超级 AGI 系统奠定基础。

### 🧠 愿景与目标

Anda 的目标是通过创建和连接无数智能体，构建一个开放、安全、可信、高度协同的智能体网络，最终实现超级 AGI 系统。我们相信，Anda 将为各行各业带来革命性的变革，推动人工智能技术在更广泛的领域中落地应用，为人类社会创造更大的价值。

## 🐼 关于 ICPanda DAO

ICPanda DAO 是在互联网计算机协议（ICP）区块链上建立的 SNS DAO 组织，发行了 `PANDA` 代币。作为 `Anda` 框架的创造者，ICPanda DAO 致力于探索 Web3 与 AI 融合的未来。

- **官方网站**: [https://panda.fans/](https://panda.fans/)
- **永久链接**: [https://dmsg.net/PANDA](https://dmsg.net/PANDA)
- **ICP SNS**: [https://dashboard.internetcomputer.org/sns/d7wvo-iiaaa-aaaaq-aacsq-cai](https://dashboard.internetcomputer.org/sns/d7wvo-iiaaa-aaaaq-aacsq-cai)
- **代币**: ICP 网络上的 PANDA，[https://www.coingecko.com/en/coins/icpanda-dao](https://www.coingecko.com/en/coins/icpanda-dao)

## 🔎 项目说明

文档：
- [Anda 架构设计](./docs/architecture_cn.md)

### 项目结构

```sh
anda/
├── anda_cli/    # 与 Anda 引擎服务交互的命令行工具
├── anda_core/          # 核心库，包含基础类型与接口
├── anda_engine/        # 智能体运行时与管理引擎实现
├── anda_engine_server/ # 支持多个 Anda 引擎的 HTTP 服务
├── anda_web3_client/   # 用于在非 TEE 环境的 Rust 语言 Web3 SDK。
├── agents/             # 各类AI智能体实现
│   ├── anda_bot/       # 示例智能体：Anda ICP
│   └── .../            # 更多智能体将在后续版本推出
├── tools/              # 工具库集合
│   ├── anda_bnb/       # 提供与币安链（BNB）的集成工具
│   ├── anda_icp/       # 提供与互联网计算机（ICP）的集成工具
│   └── .../            # 更多工具将在后续版本推出
├── characters/         # 角色设定示例库
└── examples/           # AI agents 示例
```

### 如何使用和参与贡献

#### 非开发者：

可以关注 `agents` 目录下的智能体。比如 [`anda_bot`](https://github.com/ldclabs/anda/tree/main/agents/anda_bot)。
目前部署流程还比较复杂，未来我们会推出云服务，实现一键部署。

#### 开发者：

- 可以在 `tools` 添加更多与外界其它服务的集成工具；
- 也可以在 `agents` 添加更多智能体应用；
- 或者完善 `anda_core` 和 `anda_engine` 核心引擎。

### 关联项目

- [IC-TEE](https://github.com/ldclabs/ic-tee): 🔐 Make Trusted Execution Environments (TEEs) work with the Internet Computer.
- [IC-COSE](https://github.com/ldclabs/ic-cose): ⚙️ A decentralized COnfiguration service with Signing and Encryption on the Internet Computer.

## 📝 License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See [LICENSE](./LICENSE-MIT) for the full license text.
