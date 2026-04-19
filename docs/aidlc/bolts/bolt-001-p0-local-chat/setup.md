# bolt-001: 起動手順（P0）

ルート `README.md` の代替。プロジェクト全体の方針は [PROJECT_DEFINITION.md](../../../PROJECT_DEFINITION.md) を参照。

## 前提

- Python 3.11+
- [Ollama](https://ollama.com/) が起動済みで、利用モデルを `pull` 済み（例: `ollama pull llama3.2`）。`OLLAMA_MODEL` は `ollama list` の名前と一致させる。

## セットアップ

リポジトリルートで `.env` を用意する（ルートの `.env.example` をコピーして編集）。

ローカル直実行時の例:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
LLM_PROVIDER=ollama
STREAMLIT_API_BASE_URL=http://127.0.0.1:8000
```

## バックエンド（API）

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- OpenAPI: http://127.0.0.1:8000/docs
- チャット: `POST /api/v1/chat`（本文は `{"messages":[{"role":"user","content":"..."}]}`）

## Streamlit（UI）

別ターミナル:

```powershell
cd frontend_streamlit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで表示された URL（既定では http://localhost:8501）を開く。

## テスト

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest
```

## 関連ドキュメント

- [intent.md](intent.md)
