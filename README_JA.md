# `Anda`

> Rust で構築された、構成可能な AI エージェントランタイムフレームワーク。

## README の翻訳

[English readme](./README.md) | [中文说明](./README_CN.md) | [日本語の説明](./README_JA.md)

## 紹介

Anda は、モデル、ツール、メモリ、ほかのエージェントを 1 つのランタイムに組み合わせるための Rust 製 AI エージェントフレームワークです。構成可能性、型安全な拡張ポイント、非同期実行、実用的なランタイム制御を重視しています。

コアエンジンは、agents と tools の登録、能力ラベルによるモデルリクエストのルーティング、ローカルまたはリモート関数の呼び出し、コンテキスト状態の分離、必要に応じた永続化やメモリ層の追加をサポートします。

![Anda Diagram](./anda_diagram.webp)

## 主な特徴

1. **構成可能な agents と tools**
   Agents と tools は安定した traits と function definitions を通じて登録されるため、特化したコンポーネントを 1 つの固定アプリケーション形態に縛られずに大きなワークフローへ組み合わせられます。

2. **モデルルーティング**
   エンジンは `primary`、`pro`、`flash`、`lite` などの能力ラベルで completion リクエストをルーティングできます。provider 固有の adapter は共通のリクエスト/出力契約の背後に隠れます。

3. **ランタイムオーケストレーション**
   `CompletionRunner` は、モデルターンの反復、tool calls、agent calls、usage 集計、artifacts、steering messages、follow-up messages、cancellation、長時間セッション向けのコンパクトな continuation handoff を処理します。

4. **分離された実行コンテキスト**
   `BaseCtx` と `AgentCtx` は、各 agent または tool に対して、分離された state、cache、object storage、HTTP calls、signed calls、cancellation、child contexts を提供します。

5. **拡張可能なメモリとスキル**
   オプションの extensions は、conversation storage、KIP ベースの memory tools、filesystem、shell、fetch、notes、todos、ファイルベースの skills を提供します。

6. **発見フローに対応したツールバンドル**
   静的 tools、動的 providers、MCP servers は capability groups を公開できるため、agents は `tools_groups` で関連ツールのまとまりを確認し、schema が必要になった時だけ `tools_select` で group を展開できます。

## プロジェクト

ドキュメント:
- [Anda アーキテクチャ](./docs/architecture.md)

### プロジェクト構造

```sh
anda/
├── anda_cli/              # Anda エンジンサーバーのコマンドラインインターフェース
├── anda_core/             # コア traits、型、ランタイム契約
├── anda_engine/           # エージェントランタイム、オーケストレーション、コンテキスト、モデル、拡張
└── anda_engine_server/    # 1 つ以上の Anda エンジンを提供する HTTP サーバー
```

### 使用方法と貢献方法

#### アプリケーション構築者向け:

`anda_cli` と `anda_engine_server` を使用して、設定済みのエンジンを実行し操作できます。

#### 開発者向け:

- `anda_core` の traits を使用してカスタム agents と tools を構築できます。
- `anda_engine` に再利用可能なランタイム機能を追加できます。
- モデル adapters、コンテキスト機能、メモリ統合、サーバー API を改善できます。

### Anda 上に構築されたプロダクト

- [Anda Brain](https://github.com/ldclabs/anda-brain): Anda フレームワーク上に構築された永続メモリと認知のためのプロダクト。
- [Anda Bot](https://github.com/ldclabs/anda-bot): Anda フレームワーク上に構築された個人 AI アシスタントとアプリケーションランタイム。

### 関連プロジェクト

- [KIP](https://github.com/ldclabs/KIP): Anda memory tools が使用する Knowledge Interaction Protocol。

## ライセンス

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` は MIT ライセンスの下でライセンスされています。完全なライセンステキストについては [LICENSE](./LICENSE-MIT) を参照してください。
