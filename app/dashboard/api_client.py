"""Reusable HTTP client for the AgentSentinel backend API."""

import os
from collections.abc import Callable
from typing import Any

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
"""Default base URL for the local AgentSentinel API."""


class APIClientError(Exception):
    """A safe, user-facing description of an API request failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Store a sanitized message and optional HTTP status for safe handling."""
        super().__init__(message)
        self.status_code = status_code


class AgentSentinelAPIClient:
    """Make safe GET and POST requests to the configured AgentSentinel API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> None:
        """Create a client using an explicit URL or the environment default."""
        configured_url = base_url or os.getenv(
            "AGENTSENTINEL_API_URL",
            DEFAULT_API_URL,
        )
        self._base_url = configured_url.rstrip("/")
        self._timeout = timeout

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Fetch and decode a JSON response, raising only safe errors."""
        return self._request(requests.get, path, params=params)

    def post(self, path: str, *, json: dict[str, Any]) -> Any:
        """Send a JSON request body and decode the response safely."""
        return self._request(requests.post, path, json_body=json)

    def _request(
        self,
        request: Callable[..., requests.Response],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Perform one request with shared timeout and safe error handling."""
        url = f"{self._base_url}/{path.lstrip('/')}"
        request_kwargs: dict[str, Any] = {"timeout": self._timeout}
        if params is not None:
            request_kwargs["params"] = params
        if json_body is not None:
            request_kwargs["json"] = json_body
        try:
            response = request(url, **request_kwargs)
        except requests.Timeout:
            raise APIClientError(
                "The backend request timed out. Please try again."
            ) from None
        except requests.ConnectionError:
            raise APIClientError(
                "The backend is unavailable. Check that the API server is running."
            ) from None
        except requests.RequestException:
            raise APIClientError(
                "The request to the backend could not be completed."
            ) from None

        if not response.ok:
            raise APIClientError(
                f"The backend returned an HTTP {response.status_code} response.",
                status_code=response.status_code,
            )

        try:
            return response.json()
        except ValueError:
            raise APIClientError(
                "The backend returned a response that could not be read."
            ) from None
