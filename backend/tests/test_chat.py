from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    yield
    app.dependency_overrides.clear()


def test_chat_success(client: TestClient) -> None:
    with patch("app.api.routes.chat.run_chat", return_value="mocked reply"):
        r = client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
    assert r.status_code == 200
    assert r.json() == {"content": "mocked reply"}


def test_chat_unknown_role(client: TestClient) -> None:
    r = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "narrator", "content": "x"}]},
    )
    assert r.status_code == 400


def test_chat_empty_messages(client: TestClient) -> None:
    r = client.post("/api/v1/chat", json={"messages": []})
    assert r.status_code == 422


def test_chat_unsupported_provider(client: TestClient) -> None:
    class FakeSettings:
        llm_provider = "bedrock"
        ollama_base_url = "http://127.0.0.1:11434"
        ollama_model = "x"
        ollama_temperature = 0.7

    app.dependency_overrides[get_settings] = lambda: FakeSettings()
    r = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 501
