"""Tests for sanitized dashboard API client POST requests."""

from typing import Any

import pytest
import requests

from app.dashboard.api_client import APIClientError, AgentSentinelAPIClient


class MockResponse:
    """Provide the requests response fields consumed by the client."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        json_error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.payload = payload
        self.json_error = json_error

    def json(self) -> Any:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def test_post_sends_json_and_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    payload = {"question": "What is 2 + 2?", "answer": "4"}

    def mock_post(url: str, **kwargs: Any) -> MockResponse:
        calls.append((url, kwargs))
        return MockResponse(payload={"created": True})

    monkeypatch.setattr(requests, "post", mock_post)
    client = AgentSentinelAPIClient("http://backend/", timeout=8.5)

    response = client.post("/api/v1/evaluations", json=payload)

    assert response == {"created": True}
    assert calls == [
        (
            "http://backend/api/v1/evaluations",
            {"timeout": 8.5, "json": payload},
        )
    ]


def test_get_preserves_query_parameters_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def mock_get(url: str, **kwargs: Any) -> MockResponse:
        calls.append((url, kwargs))
        return MockResponse(payload={"items": []})

    monkeypatch.setattr(requests, "get", mock_get)
    client = AgentSentinelAPIClient("http://backend/", timeout=3.25)

    response = client.get("/api/v1/evaluations", params={"limit": 10})

    assert response == {"items": []}
    assert calls == [
        (
            "http://backend/api/v1/evaluations",
            {"timeout": 3.25, "params": {"limit": 10}},
        )
    ]


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (requests.Timeout("sensitive timeout details"), "timed out"),
        (requests.ConnectionError("sensitive connection details"), "unavailable"),
    ],
)
def test_post_sanitizes_timeout_and_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: requests.RequestException,
    message: str,
) -> None:
    def mock_post(*_: Any, **__: Any) -> MockResponse:
        raise exception

    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(APIClientError) as error_info:
        AgentSentinelAPIClient().post("/api/v1/diagnostics", json={})

    assert message in str(error_info.value)
    assert "sensitive" not in str(error_info.value)


def test_post_sanitizes_http_error_without_exposing_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_post(*_: Any, **__: Any) -> MockResponse:
        return MockResponse(
            status_code=503,
            payload={"detail": "api_key=do-not-show"},
        )

    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(APIClientError) as error_info:
        AgentSentinelAPIClient().post("/api/v1/evaluations", json={})

    assert str(error_info.value) == "The backend returned an HTTP 503 response."
    assert error_info.value.status_code == 503
    assert "do-not-show" not in str(error_info.value)


def test_post_rejects_non_json_response_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: MockResponse(
            json_error=ValueError("sensitive response content")
        ),
    )

    with pytest.raises(APIClientError, match="could not be read") as error_info:
        AgentSentinelAPIClient().post("/api/v1/evaluations", json={})

    assert "sensitive" not in str(error_info.value)
