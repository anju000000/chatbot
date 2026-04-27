import os
from pathlib import Path

import httpx
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

API_BASE = os.environ.get("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="ローカルチャット (P0)", page_icon="💬")
st.title("ローカルチャット (P0)")
st.caption(f"API: `{API_BASE}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("メッセージを入力"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("応答を生成しています…"):
        try:
            with httpx.Client(timeout=300.0) as client:
                r = client.post(
                    f"{API_BASE}/api/v1/chat",
                    json={"messages": st.session_state.messages},
                )
                r.raise_for_status()
                assistant = r.json()["content"]
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            assistant = f"HTTP エラー ({e.response.status_code}): {detail}"
        except httpx.RequestError as e:
            assistant = f"接続エラー: {e!s}"
        except (KeyError, ValueError) as e:
            assistant = f"応答の解析エラー: {e!s}"
    st.session_state.messages.append({"role": "assistant", "content": assistant})
    st.rerun()
