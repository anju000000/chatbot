# プロジェクト定義書

## ローカル LLM チャットボット（将来のクラウド移行を想定）

### 1. 文書情報

| 項目   | 内容                                                         |
| ------ | ------------------------------------------------------------ |
| 文書名 | ローカル LLM チャットボット プロジェクト定義                 |
| 目的   | ローカルで動作するチャットボットを構築し、将来 AWS 等のクラウドへ移行しやすい構成にする |
| 想定読者 | 開発者・プロダクトオーナー                                 |

---

### 2. 背景・目的

- **現状**: クラウド LLM のコストやデータ持ち出しを避けつつ、PC 上で安全に LLM を実行したい。
- **将来**: 商用 LLM（例: AWS Bedrock の Claude 等）や本番インフラへ載せ替え可能にしたい。
- **方針**: モデル呼び出しは **抽象レイヤー（LangChain 等）経由** にし、**OpenAI 互換 API** や **Bedrock 用アダプタ** を差し替え可能にする。

---

### 3. スコープ

**含む**

- ローカルでのチャット UI と会話 API
- （任意）RAG 用のドキュメント取り込み・検索の土台
- ログ・トレース（Langfuse 等）の接続余地
- Docker による実行環境の固定

**含まない（初期フェーズ）**

- 本番用 React/TypeScript フロントの完成品（フェーズ後半で検討）
- Multi-Agent・MCP の本格実装（拡張フェーズとして位置づけ）
- Terraform による本番 AWS 一式（移行フェーズで詳細化）

---

### 4. 技術スタック（レイヤ別）

| レイヤ           | 技術                    | 役割                                                         |
| ---------------- | ----------------------- | ------------------------------------------------------------ |
| **ローカル LLM** | Ollama                  | ローカル実行基盤。API でモデル推論。                         |
|                  | オープンウェイト LLM    | Ollama で配布されるモデルを選択・固定。                      |
| **AI FW**        | LangChain（または LCEL） | プロンプト・チェーン・履歴・モデル抽象化。Bedrock / Ollama 切替の土台。 |
|                  | RAG                     | 社内文書等を検索して回答精度を上げる（ベクトル DB は下記 DB と連携）。 |
|                  | Langfuse                | 応答品質・レイテンシ・トークン等の観測・改善ループ。         |
|                  | MLflow（任意）          | 実験・モデル版管理が必要になった段階で検討。                 |
| **フロント**     | Streamlit               | 迅速な検証・モック UI（Python のみ）。                       |
|                  | React + TypeScript      | 本番想定のリッチ UI（後フェーズ）。                          |
| **バックエンド** | Python + FastAPI        | REST/SSE 等でチャット API、同時リクエスト処理。              |
| **データ**       | RDB / NoSQL             | 会話履歴、ユーザー設定等。                                   |
|                  | ベクトルストア          | RAG 用埋め込み検索（pgvector、Chroma、Qdrant 等は要件で選定）。 |
| **クラウド・Infra（将来）** | AWS Bedrock      | マネージド商用 LLM API。                                     |
|                  | Terraform               | IaC で環境の再現性。                                         |
|                  | Docker                  | 開発〜本番まで環境差の抑制。                                 |
| **拡張**         | Multi-Agent             | 複雑タスクの分担（後続）。                                   |
|                  | MCP                     | 外部ツール・コンテキスト連携の標準化（後続）。               |

---

### 5. アーキテクチャ方針（API 互換・移行）

1. **モデルアクセス**  
   - アプリ本体は **LangChain のチャットモデル抽象**（または同等の薄いラッパ）のみを参照する。  
   - ローカル: `ChatOllama` 等。  
   - クラウド: `ChatBedrock` / `ChatBedrockConverse` 等を **設定（環境変数）で切替**。  
   - 可能なら **OpenAI 互換エンドポイント**（Ollama の `/v1` 等）と **Bedrock** の両方を同じインターフェースで扱えるよう設計する。

2. **フロントとバックの境界**  
   - UI は **FastAPI のチャット API** にのみ依存する（モデル名はサーバ側設定）。  
   - 将来 React に替えても API 契約（JSON スキーマ・ストリーミング形式）を維持する。

3. **シークレット・設定**  
   - ローカルは `.env`、本番は AWS Secrets Manager 等を想定し、**キー名をプロジェクトで統一**する。

4. **観測**  
   - Langfuse は **プロバイダ非依存**でトレース可能なため、Ollama から Bedrock へ変えても同じダッシュボードで比較しやすい。

---

### 6. 開発フェーズ（提案）

| フェーズ | 内容                                                         | 主な成果物                         |
| -------- | ------------------------------------------------------------ | ---------------------------------- |
| **P0**   | FastAPI + Ollama + LangChain でチャット API、Streamlit で最小 UI | ローカル一発起動、会話の永続化は任意 |
| **P1**   | Docker 化、`.env` でモデル名・温度など切替                   | 環境の再現性                       |
| **P2**   | RAG（取り込みパイプライン + ベクトル検索）                   | ドキュメント根拠付き回答           |
| **P3**   | Langfuse 本接続                                              | メトリクス・プロンプト版管理       |
| **P4**   | Bedrock 接続の実装・切替テスト                               | クラウド LLM への移行実証          |
| **P5**   | React/TS フロントまたは Terraform（要件に応じて順序調整）    | 本番寄り UI / インフラ             |

---

### 7. 非機能要件（抜粋）

- **パフォーマンス**: ローカル GPU/CPU に依存。タイムアウト・ストリーミングで体感を改善。
- **セキュリティ**: 本番では IAM・VPC エンドポイント等を別紙で定義。ローカルでは API キーをリポジトリに含めない。
- **保守性**: モデルプロバイダ固有のコードは **アダプタ 1 箇所** に集約（OAOO）。

---

### 8. 成功基準（初期）

- 同一リポジトリ・同一 API で、**Ollama と Bedrock（またはスタブ）を設定だけで切り替え可能**であること。
- `docker compose up`（または同等）で **フロント＋API＋（任意）DB** がローカルで起動すること。
- 会話または RAG 応答が **Langfuse でトレース可能**であること（P3 以降）。

---

### 9. リスク・未決事項

- オープンモデルと Bedrock モデルで **出力品質・コンテキスト長が異なる**ため、プロンプトは環境別チューニングが必要になる可能性がある。
- RAG の **正確性・更新頻度**は業務要件次第で設計が変わる。
- MLflow をいつ必須にするかは **実験管理の必要性が出てから** でよい（YAGNI）。

---

### 10. ディレクトリ構成例

実装時はリポジトリ規模に合わせて調整する。以下は **モノレポ（バックエンド + Streamlit + 後続の frontend）** を想定した例。

```
Ollama/                              # リポジトリルート（例）
├── docs/
│   ├── PROJECT_DEFINITION.md        # 本書
│   └── aidlc/
│       └── bolts/                   # AI-DLC ボルト（必要に応じて）
│           └── bolt-NNN-<slug>/
│               ├── intent.md
│               ├── tasks.md
│               └── progress.md
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI エントリ
│   │   ├── api/
│   │   │   └── routes/
│   │   │       └── chat.py          # チャット REST/SSE
│   │   ├── core/
│   │   │   ├── config.py            # 設定（環境変数読み込み）
│   │   │   └── logging.py
│   │   └── services/
│   │       ├── llm_factory.py       # Ollama / Bedrock 切替（単一責任）
│   │       └── chat_service.py
│   ├── tests/
│   ├── pyproject.toml               # または requirements.txt
│   └── Dockerfile
├── frontend_streamlit/
│   ├── app.py                       # Streamlit エントリ
│   ├── components/                  # 必要に応じて分割
│   ├── requirements.txt
│   └── Dockerfile
├── frontend_react/                  # P5 以降で追加する場合
│   └── (create-react-app / Vite 等は別途決定)
├── infra/
│   ├── docker/
│   │   └── docker-compose.yml       # API + UI +（任意）DB + Langfuse
│   └── terraform/                   # 本番移行時
│       └── (環境ごとに分割)
├── scripts/                         # ワンショット用（マイグレーション等）
├── .env.example                     # リポジトリにコミット（実値は含めない）
├── .gitignore
└── docs/aidlc/bolts/bolt-001-p0-local-chat/setup.md  # P0 起動手順・前提
```

**役割の要点**

- `llm_factory.py` にプロバイダ切替を集約し、アプリ層は「チャット用 LLM インターフェース」のみ利用する。
- `docker-compose` はローカル開発の単一入口にし、本番は Terraform + コンテナレジストリ等へ展開する想定。

---

### 11. 環境変数一覧（テンプレート）

以下を `.env.example` としてリポジトリに置き、実運用では `.env`（非コミット）で上書きする。

```bash
# --- アプリ共通 ---
APP_ENV=development
# development | staging | production
LOG_LEVEL=INFO

# FastAPI / CORS（Streamlit や将来の React オリジン）
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501

# --- LLM プロバイダ切替 ---
# ollama | bedrock | openai_compatible（プロジェクトで採用する値に統一）
LLM_PROVIDER=ollama

# --- Ollama（ローカル）---
OLLAMA_BASE_URL=http://host.docker.internal:11434
# Linux の Docker 内からホストの Ollama へは host ネットワークやゲートウェイ IP の調整が必要な場合あり
OLLAMA_MODEL=llama3.2

# --- OpenAI 互換 API（任意・将来のプロキシや他サービス用）---
# OPENAI_COMPATIBLE_BASE_URL=https://api.example.com/v1
# OPENAI_COMPATIBLE_API_KEY=
# OPENAI_COMPATIBLE_MODEL=gpt-4o-mini

# --- AWS Bedrock（P4 以降・本番想定）---
# AWS_REGION=ap-northeast-1
# BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
# 認証は環境・デプロイ先に合わせる: IAM ロール推奨。ローカル検証のみなら:
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_SESSION_TOKEN=

# --- Langfuse（P3 以降・未設定ならノーオペで動くようにする）---
# LANGFUSE_PUBLIC_KEY=
# LANGFUSE_SECRET_KEY=
# LANGFUSE_HOST=https://cloud.langfuse.com

# --- データベース（会話履歴・RAG メタ、任意）---
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/chatbot
# または SQLite ローカル例:
# DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# --- ベクトルストア（RAG 利用時・プロバイダに応じて）---
# VECTOR_STORE_PROVIDER=chroma
# CHROMA_PERSIST_DIR=./data/chroma

# --- Streamlit ---
STREAMLIT_API_BASE_URL=http://localhost:8000
```

リポジトリルートに **同名の `.env.example`** を置いてある。コピーして `.env` を作成し、実値を設定する。

**命名のルール**

- プロバイダ固有の接頭辞（`OLLAMA_`, `BEDROCK_`, `LANGFUSE_`）を付け、衝突と誤設定を防ぐ。
- シークレットは **リポジトリに含めない**。CI ではシークレットストアを参照する。

---

### 12. 改訂履歴

| 日付       | 内容     |
| ---------- | -------- |
| 2026-04-03 | 初版作成（ディレクトリ構成・環境変数テンプレートを含む） |
