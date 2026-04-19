# bolt-001: P0 ローカル最小チャット

## Intent（目的）

FastAPI + LangChain + Ollama + Streamlit により、ローカルで動作する最小チャットボットを構築する。LLM 呼び出しは LangChain 経由に集約し、将来のプロバイダ切替に備える。

## 完了条件

- [x] Ollama を前提に、`POST /api/v1/chat` でチャット応答が返る（LangChain `ChatOllama` 使用）
- [x] Streamlit から上記 API を呼び、セッション内で会話が継続する
- [x] 設定は環境変数（`.env`）で `OLLAMA_BASE_URL` / `OLLAMA_MODEL` 等を変更可能
- [x] `llm_factory` にプロバイダ切替の入口を1箇所に置く（P0 は `ollama` のみ実装）
- [x] 起動手順が本ボルトの [setup.md](setup.md) に記載されている
- [x] API のユニットテスト（LLM をモック）が通る

## 対象範囲

- 対象テーブル: なし（DB なし）
- 対象ファイル: `backend/`、`frontend_streamlit/`、`.gitignore`、本ボルト配下の `setup.md` / `tasks.md` / `progress.md`
- 対象画面: Streamlit チャット UI（最小）

## 対象外（別ボルト）

- Docker / compose（P1）
- RAG、Langfuse、Bedrock 実装
