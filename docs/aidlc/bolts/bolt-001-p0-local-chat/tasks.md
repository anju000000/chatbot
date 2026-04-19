# bolt-001 タスク一覧

本リポジトリは Python / FastAPI のため、bolt-workflow のカテゴリを次のように対応付ける。

| テンプレート | 本ボルトでの意味 |
| ------------ | ---------------- |
| [SQL]        | データベース（P0 では未使用） |
| [PHP]        | → **[BACKEND]** アプリロジック |
| [VIEW]       | → **[FRONTEND]** Streamlit |
| [TEST]       | pytest |
| [DOC]        | ボルト文書（`setup.md` 等） |
| [CONFIG]     | `.gitignore`、`pyproject.toml`、`.env` 運用 |

## [SQL] データベース

- [x] （該当なし）

## [BACKEND] FastAPI / LangChain

- [x] `backend/app/core/config.py` 設定（pydantic-settings）
- [x] `backend/app/services/llm_factory.py`（Ollama、将来用分岐）
- [x] `backend/app/services/chat_service.py`（メッセージ変換・invoke）
- [x] `backend/app/api/routes/chat.py`（`POST /api/v1/chat`）
- [x] `backend/app/main.py`（CORS、ルータ登録）

## [FRONTEND] Streamlit

- [x] `frontend_streamlit/app.py`（チャット UI・API 呼び出し）

## [TEST] テスト

- [x] `backend/tests/test_chat.py`（TestClient + `run_chat` モック）

## [DOC] ドキュメント

- [x] `setup.md`（前提・起動手順。旧 README の内容を包含）
- [x] `progress.md` 更新

## [CONFIG] 設定

- [x] `.gitignore`（`.env`、`.venv` 等）
- [x] `backend/pyproject.toml`
- [x] `frontend_streamlit/requirements.txt`
