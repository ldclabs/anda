# `Anda`

> 🤖 ICP と TEE によって強化された、Rust で構築された AI エージェントフレームワーク。

## 🌍 README の翻訳

[English readme](./README.md) | [中文说明](./README_CN.md) | [日本語の説明](./README_JA.md)

## 🤖 紹介

Anda は Rust で構築された AI エージェントフレームワークで、ICP ブロックチェーン統合と TEE サポートを特徴としています。
これは、高度に構成可能で、自律的で、永続的な記憶を持つ AI エージェントのネットワークを作成することを目的としています。
さまざまな業界のエージェントを接続することで、Anda はスーパー AGI システムを作成し、人工知能をより高いレベルに進化させることを目指しています。

![Anda Diagram](./anda_diagram.webp)

### ✨ 主な特徴

1. **構成可能性**:
   Anda エージェントは特定のドメイン固有の問題を解決することに特化しており、他のエージェントと柔軟に組み合わせて複雑なタスクに取り組むことができます。単一のエージェントが問題を単独で解決できない場合、他のエージェントと協力して強力な問題解決ネットワークを形成します。このモジュラー設計により、Anda は多様なニーズに適応できます。

2. **シンプルさ**:
   Anda はシンプルさと使いやすさを重視して設計されており、開発者が強力で効率的なエージェントを迅速に構築できるようにします。非開発者も簡単な設定で自分のエージェントを作成できるため、技術的な障壁が低くなり、エージェントの開発と応用に広く参加できるようになります。

3. **信頼性**:
   Anda エージェントは、Trusted Execution Environments (TEEs) に基づく分散型信頼実行環境 (dTEE) 内で動作し、セキュリティ、プライバシー、およびデータの完全性を確保します。このアーキテクチャは、エージェントの運用に高度に信頼できるインフラストラクチャを提供し、データと計算プロセスを保護します。

4. **自律性**:
   Anda エージェントは、ICP ブロックチェーンから永続的なアイデンティティと暗号化機能を取得し、大規模言語モデル (LLMs) の推論および意思決定能力と組み合わせます。これにより、エージェントは経験と知識に基づいて自律的かつ効率的に問題を解決し、動的な環境に適応し、複雑なシナリオで効果的な意思決定を行うことができます。

5. **永続的な記憶**:
   Anda エージェントの記憶状態は、ICP ブロックチェーンおよび dTEE の信頼できるストレージネットワークに保存され、継続的なアルゴリズムのアップグレード、知識の蓄積、および進化を保証します。この永続的な記憶メカニズムにより、エージェントは無期限に動作し、「不死」を実現することさえ可能になり、スーパー AGI システムの基盤を築きます。

### 🧠 ビジョンと目標

Anda の目標は、無数のエージェントを作成および接続し、オープンで安全、信頼性が高く、高度に協調的なエージェントネットワークを構築し、最終的にスーパー AGI システムを実現することです。私たちは、Anda がさまざまな業界に革命的な変革をもたらし、AI 技術の広範な応用を推進し、人類社会により大きな価値を創造すると信じています。

## 🐼 ICPanda DAO について

ICPanda DAO は、インターネットコンピュータプロトコル (ICP) ブロックチェーン上に設立された SNS DAO であり、`PANDA` トークンを発行しています。`Anda` フレームワークの創設者として、ICPanda DAO は Web3 と AI の統合の未来を探求しています。

- **ウェブサイト**: [https://panda.fans/](https://panda.fans/)
- **パーマリンク**: [https://dmsg.net/PANDA](https://dmsg.net/PANDA)
- **ICP SNS**: [https://dashboard.internetcomputer.org/sns/d7wvo-iiaaa-aaaaq-aacsq-cai](https://dashboard.internetcomputer.org/sns/d7wvo-iiaaa-aaaaq-aacsq-cai)
- **トークン**: ICP ネットワーク上の PANDA、[https://www.coingecko.com/en/coins/icpanda-dao](https://www.coingecko.com/en/coins/icpanda-dao)

## 🔎 プロジェクト

ドキュメント:
- [Anda アーキテクチャ](./docs/architecture.md)

### プロジェクト構造

```sh
anda/
├── anda_cli/  # Anda エンジンサーバーのコマンドラインインターフェース
├── anda_core/        # 基本的な型とインターフェースを含むコアライブラリ
├── anda_engine/      # エージェントのランタイムと管理エンジンの実装
├── anda_engine_server/ # 複数の Anda エンジンを提供する HTTP サーバー
├── anda_web3_client/ # 非 TEE 環境での Web3 統合のための Rust SDK
├── agents/           # さまざまな AI エージェントの実装
│ ├── anda_bot/       # 例: Anda ICP
│ └── .../            # 将来のリリースでさらに多くのエージェント
├── tools/            # ツールライブラリ
│ ├── anda_bnb/       # Anda エージェントツールは BNB チェーン (BNB) との統合を提供
│ ├── anda_icp/       # インターネットコンピュータ (ICP) との統合ツールを提供
│ └── .../            # 将来のリリースでさらに多くのツール
├── characters/       # キャラクターの例
└── examples/         # AI エージェントの例
```

### 使用方法と貢献方法

#### 非開発者向け:

`agents` ディレクトリ内のエージェントをフォローできます。例えば、[`anda_bot`](https://github.com/ldclabs/anda/tree/main/agents/anda_bot)。
現在、デプロイメントプロセスは複雑ですが、将来的にはワンクリックデプロイメントのためのクラウドサービスを開始する予定です。

#### 開発者向け:

- `tools` に外部サービスとの統合ツールを追加できます。
- `agents` にさらに多くのエージェントアプリケーションを作成できます。
- または、コアエンジン `anda_core` および `anda_engine` を強化できます。

### 関連プロジェクト

- [IC-TEE](https://github.com/ldclabs/ic-tee): 🔐 Trusted Execution Environments (TEEs) をインターネットコンピュータと連携させる。
- [IC-COSE](https://github.com/ldclabs/ic-cose): ⚙️ インターネットコンピュータ上の署名と暗号化を備えた分散型設定サービス。

## 📝 ライセンス

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` は MIT ライセンスの下でライセンスされています。完全なライセンステキストについては [LICENSE](./LICENSE-MIT) を参照してください。
