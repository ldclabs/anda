# `Anda`

> 🤖 An AI agent framework built with Rust, powered by ICP and TEEs.

## 🌍 README Translations

[English readme](./README.md) | [中文说明](./README_CN.md) | [日本語の説明](./README_JA.md)

## 🤖 Introduction

Anda is an AI agent framework built with Rust, featuring ICP blockchain integration and TEE support.
It is designed to create a highly composable, autonomous, and perpetually memorizing network of AI agents.
By connecting agents across various industries, Anda aims to create a super AGI system, advancing artificial intelligence to higher levels.

![Anda Diagram](./anda_diagram.webp)

### ✨ Key Features

1. **Composability**:
   Anda agents specialize in solving domain-specific problems and can flexibly combine with other agents to tackle complex tasks. When a single agent cannot solve a problem alone, it collaborates with others to form a robust problem-solving network. This modular design allows Anda to adapt to diverse needs.

2. **Simplicity**:
   Anda emphasizes simplicity and ease of use, enabling developers to quickly build powerful and efficient agents. Non-developers can also create their own agents through simple configurations, lowering the technical barrier and inviting broader participation in agent development and application.

3. **Trustworthiness**:
   Anda agents operate within a decentralized trusted execution environment (dTEE) based on Trusted Execution Environments (TEEs), ensuring security, privacy, and data integrity. This architecture provides a highly reliable infrastructure for agent operations, safeguarding data and computational processes.

4. **Autonomy**:
   Anda agents derive permanent identities and cryptographic capabilities from the ICP blockchain, combined with the reasoning and decision-making abilities of large language models (LLMs). This allows agents to autonomously and efficiently solve problems based on their experiences and knowledge, adapting to dynamic environments and making effective decisions in complex scenarios.

5. **Perpetual Memory**:
   The memory states of Anda agents are stored on the ICP blockchain and within the trusted storage network of dTEE, ensuring continuous algorithm upgrades, knowledge accumulation, and evolution. This perpetual memory mechanism enables agents to operate indefinitely, even achieving "immortality", laying the foundation for a super AGI system.

### 🧠 Vision and Goals

Anda's goal is to create and connect countless agents, building an open, secure, trustworthy, and highly collaborative network of agents, ultimately realizing a super AGI system. We believe Anda will bring revolutionary changes across industries, driving the widespread application of AI technology and creating greater value for human society.

## 🐼 About ICPanda DAO

ICPanda DAO is an SNS DAO established on the Internet Computer Protocol (ICP) blockchain, issuing the `PANDA` token. As the creator of the `Anda` framework, ICPanda DAO is dedicated to exploring the future of Web3 and AI integration.

- **Website**: [https://panda.fans/](https://panda.fans/)
- **Permalink**: [https://dmsg.net/PANDA](https://dmsg.net/PANDA)
- **ICP SNS**: [https://dashboard.internetcomputer.org/sns/d7wvo-iiaaa-aaaaq-aacsq-cai](https://dashboard.internetcomputer.org/sns/d7wvo-iiaaa-aaaaq-aacsq-cai)
- **Token**: PANDA on ICP network, [https://www.coingecko.com/en/coins/icpanda-dao](https://www.coingecko.com/en/coins/icpanda-dao)

## 🔎 Project

Documents:
- [Anda Architecture](./docs/architecture.md)

### Project Structure

```sh
anda/
├── anda_cli/              # Command line interface for Anda engine server
├── anda_core/             # Core library containing base types and interfaces
├── anda_engine/           # Engine implementation for agent runtime and management
├── anda_engine_server/    # HTTP server to serve multiple Anda engines
└── anda_web3_client/      # Rust SDK for Web3 integration in non-TEE environments
```

### How to Use and Contribute

#### For Non-Developers:

The Anda framework provides a command-line interface in `anda_cli` for interacting with the Anda engine server.

#### For Developers:

- Enhance the core engines `anda_core` and `anda_engine`;
- Build custom agents and tools using the `anda_core` traits;
- Contribute to the `anda_engine_server` HTTP server implementation.

### Related Projects

- [IC-TEE](https://github.com/ldclabs/ic-tee): 🔐 Make Trusted Execution Environments (TEEs) work with the Internet Computer.
- [IC-COSE](https://github.com/ldclabs/ic-cose): ⚙️ A decentralized COnfiguration service with Signing and Encryption on the Internet Computer.


## ❓ FAQ

### General

**Q: What is Anda?**
Anda is an AI agent framework built with Rust, featuring ICP blockchain integration and Trusted Execution Environment (TEE) support. It enables creating composable, autonomous agents with perpetual memory.

**Q: What is dTEE?**
Decentralized Trusted Execution Environment (dTEE) combines Trusted Execution Environments with blockchain to provide a secure, private, and verifiable computation layer for AI agents. It ensures data integrity and computational trustworthiness.

**Q: Why ICP blockchain?**
ICP provides agents with permanent identities, cryptographic capabilities, and perpetual memory storage. This enables agents to operate autonomously and persistently across the network.

### Getting Started

**Q: How do I install Anda?**
```bash
# Clone the repository
git clone https://github.com/ldclabs/anda.git
cd anda

# Build with Cargo
cargo build --release
```
Ensure you have Rust 1.75+ installed. See [Rust installation](https://rustup.rs/).

**Q: What are the main components?**
- `anda_cli`: Command-line interface for interacting with the engine server
- `anda_core`: Core library with base types and agent traits
- `anda_engine`: Agent runtime and management engine
- `anda_engine_server`: HTTP server for multiple Anda engines
- `anda_web3_client`: Rust SDK for Web3 integration

### Agent Development

**Q: How do I create a custom agent?**
Implement the `anda_core` traits to define your agent's behavior, tools, and interaction patterns. See the [architecture documentation](./docs/architecture.md) for details.

**Q: Can I use Anda without TEE?**
Yes. The `anda_web3_client` provides Web3 integration for non-TEE environments, allowing agents to interact with ICP blockchain without hardware-based trust.

**Q: How does agent composability work?**
Anda agents specialize in domain-specific problems and can flexibly combine with other agents. When a single agent cannot solve a problem alone, it collaborates with others to form a robust problem-solving network.

### ICP Integration

**Q: How do agents get their identity?**
Agents derive permanent identities and cryptographic capabilities from the ICP blockchain, enabling autonomous and persistent operation.

**Q: What is perpetual memory?**
Agent memory states are stored on the ICP blockchain and within dTEE trusted storage, ensuring continuous knowledge accumulation and evolution even across restarts.

### Troubleshooting

**Q: Build fails with Rust version error**
Ensure you have Rust 1.75 or later:
```bash
rustup update stable
rustc --version  # Should show 1.75+
```

**Q: Cannot connect to ICP network**
- Verify your network connectivity to the ICP blockchain
- Check that the IC-TEE dependency is properly configured
- See [IC-TEE documentation](https://github.com/ldclabs/ic-tee) for setup details

**Q: Agent fails to start**
- Check the engine server logs for error details
- Verify the agent configuration matches the expected format
- Ensure all required dependencies (ICP, TEE) are accessible

**Q: High memory usage**
- Limit the number of concurrent agents
- Adjust the memory retention policy in agent configuration
- Monitor dTEE storage usage

## 📝 License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See [LICENSE](./LICENSE-MIT) for the full license text.
