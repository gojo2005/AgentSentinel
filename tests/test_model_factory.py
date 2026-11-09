"""Tests for provider selection and tool binding without provider API calls."""

from __future__ import annotations

import pytest
from langchain_litellm import ChatLiteLLM

from app.agent import model_factory
from app.agent.graph import _TOOLS
from app.evaluation.llm_judge import LLMJudge


class FakeChatModel:
    """Capture model constructor arguments and bound tools."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.bound_tools: list[object] | None = None

    def bind_tools(self, tools: list[object]) -> FakeChatModel:
        self.bound_tools = tools
        return self


def test_groq_is_the_default_and_uses_chat_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTSENTINEL_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "configured-groq-model")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    tools = [object()]
    model = FakeChatModel()
    captured: list[dict[str, object]] = []

    def create_chat_groq(**kwargs: object) -> FakeChatModel:
        captured.append(kwargs)
        return model

    monkeypatch.setattr(model_factory, "ChatGroq", create_chat_groq)

    selection = model_factory.create_chat_model(tools)

    assert selection.model is model
    assert selection.provider == "Groq"
    assert selection.model_name == "configured-groq-model"
    assert captured == [
        {
            "model": "configured-groq-model",
            "api_key": "test-groq-key",
            "temperature": 0,
            "timeout": model_factory._LLM_REQUEST_TIMEOUT_SECONDS,
        }
    ]
    assert model.bound_tools == tools


@pytest.mark.parametrize(
    ("provider", "model_name", "credential_name", "provider_label"),
    [
        ("openai", "openai/gpt-4o-mini", "OPENAI_API_KEY", "OpenAI"),
        (
            "anthropic",
            "anthropic/claude-3-5-sonnet-20240620",
            "ANTHROPIC_API_KEY",
            "Anthropic",
        ),
        ("gemini", "gemini/gemini-1.5-flash", "GEMINI_API_KEY", "Gemini"),
    ],
)
def test_litellm_providers_use_chat_litellm_and_bind_tools(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model_name: str,
    credential_name: str,
    provider_label: str,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", provider)
    monkeypatch.setenv("LITELLM_MODEL", model_name)
    monkeypatch.setenv(credential_name, "test-provider-key")
    tools = [object(), object()]
    model = FakeChatModel()
    captured: list[dict[str, object]] = []

    def create_chat_litellm(**kwargs: object) -> FakeChatModel:
        captured.append(kwargs)
        return model

    monkeypatch.setattr(model_factory, "ChatLiteLLM", create_chat_litellm)

    selection = model_factory.create_chat_model(tools)

    assert selection.model is model
    assert selection.provider == provider_label
    assert selection.model_name == model_name
    assert captured == [
        {
            "model": model_name,
            "temperature": 0,
            "request_timeout": model_factory._LLM_REQUEST_TIMEOUT_SECONDS,
        }
    ]
    assert model.bound_tools == tools


def test_actual_chat_litellm_binds_agent_tools_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-used-for-network")
    for credential_name in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(credential_name, raising=False)

    selection = model_factory.create_chat_model(_TOOLS)

    assert selection.provider == "OpenAI"
    assert selection.model_name == "openai/gpt-4o-mini"
    assert isinstance(selection.model.bound, ChatLiteLLM)
    bound_tool_names = [
        tool["function"]["name"] for tool in selection.model.kwargs["tools"]
    ]
    assert bound_tool_names == [
        "calculator",
        "knowledge_base_search",
        "mock_weather",
    ]


@pytest.mark.parametrize(
    ("provider", "credential_names"),
    [
        ("groq", ("GROQ_API_KEY",)),
        ("openai", ("OPENAI_API_KEY",)),
        ("anthropic", ("ANTHROPIC_API_KEY",)),
        ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    ],
)
def test_missing_provider_credentials_raise_safe_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    credential_names: tuple[str, ...],
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", provider)
    monkeypatch.setenv("GROQ_MODEL", "configured-groq-model")
    monkeypatch.setenv("LITELLM_MODEL", f"{provider}/configured-model")
    for credential_name in credential_names:
        monkeypatch.delenv(credential_name, raising=False)
    monkeypatch.setattr(
        model_factory,
        "ChatGroq",
        lambda **_: pytest.fail("ChatGroq should not be constructed"),
    )
    monkeypatch.setattr(
        model_factory,
        "ChatLiteLLM",
        lambda **_: pytest.fail("ChatLiteLLM should not be constructed"),
    )

    with pytest.raises(ValueError) as error_info:
        model_factory.create_chat_model([])

    assert all(name in str(error_info.value) for name in credential_names)
    assert "test-provider-key" not in str(error_info.value)


def test_litellm_model_must_match_selected_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LITELLM_MODEL", "anthropic/configured-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-key")

    with pytest.raises(ValueError, match="openai/"):
        model_factory.create_chat_model([])


def test_unsupported_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-that-must-not-leak")

    with pytest.raises(
        ValueError,
        match="Unsupported AGENTSENTINEL_LLM_PROVIDER",
    ) as error_info:
        model_factory.create_chat_model([])
    assert "secret-that-must-not-leak" not in str(error_info.value)


def test_gemini_accepts_google_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LITELLM_MODEL", "gemini/gemini-1.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        model_factory,
        "ChatLiteLLM",
        lambda **kwargs: FakeChatModel(**kwargs),
    )

    selection = model_factory.create_chat_model([])

    assert selection.provider == "Gemini"
    assert selection.model_name == "gemini/gemini-1.5-flash"


def test_llm_judge_uses_configured_provider_without_binding_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeChatModel()
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        model_factory,
        "ChatLiteLLM",
        lambda **_: model,
    )

    judge = LLMJudge()

    assert judge._llm is model
    assert judge._provider == "OpenAI"
    assert judge._model_name == "openai/gpt-4o-mini"
    assert model.bound_tools is None


def test_actual_chat_litellm_supports_structured_judge_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-used-for-network")
    for credential_name in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(credential_name, raising=False)

    judge = LLMJudge()
    structured_model = judge._llm.with_structured_output(
        {"type": "object", "properties": {"score": {"type": "number"}}},
        method="json_mode",
    )

    assert structured_model is not None
