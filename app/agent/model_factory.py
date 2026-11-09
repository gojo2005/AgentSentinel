"""Provider-aware construction of tool-bound chat models."""

import os
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_litellm import ChatLiteLLM

from core.config import get_llm_provider


_LLM_REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ChatModelSelection:
    """A tool-bound chat runnable and its non-secret provider identity."""

    model: Runnable[Any, AIMessage]
    provider: str
    model_name: str


_LITELLM_PROVIDERS = {
    "openai": ("OpenAI", "openai/", "OPENAI_API_KEY"),
    "anthropic": ("Anthropic", "anthropic/", "ANTHROPIC_API_KEY"),
    "gemini": ("Gemini", "gemini/", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
}


def get_model_identity() -> tuple[str, str | None]:
    """Return the configured provider and model name without reading secrets."""
    provider = get_llm_provider()
    model_setting = "GROQ_MODEL" if provider == "groq" else "LITELLM_MODEL"
    model_name = os.getenv(model_setting)
    if model_name is not None:
        model_name = model_name.strip() or None

    if provider == "groq":
        return "Groq", model_name
    if provider in _LITELLM_PROVIDERS:
        return _LITELLM_PROVIDERS[provider][0], model_name
    return provider, model_name


def _required_setting(name: str, provider: str) -> str:
    """Read one required environment setting without exposing its value."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"{name} is required when the {provider} provider is selected."
        )
    return value


def _required_credential(names: str | tuple[str, ...], provider: str) -> None:
    """Require one provider credential without returning or logging its value."""
    credential_names = (names,) if isinstance(names, str) else names
    if any(os.getenv(name, "").strip() for name in credential_names):
        return
    required_names = " or ".join(credential_names)
    raise ValueError(
        f"{required_names} is required when the {provider} provider is selected."
    )


def create_unbound_chat_model() -> tuple[BaseChatModel, str, str]:
    """Build the configured provider model without binding tools."""
    provider = get_llm_provider()
    provider_name, model_name = get_model_identity()

    if provider == "groq":
        model_name = _required_setting("GROQ_MODEL", provider_name)
        api_key = _required_setting("GROQ_API_KEY", provider_name)
        chat_model = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0,
            timeout=_LLM_REQUEST_TIMEOUT_SECONDS,
        )
    elif provider in _LITELLM_PROVIDERS:
        provider_name, model_prefix, credential_name = _LITELLM_PROVIDERS[provider]
        model_name = _required_setting("LITELLM_MODEL", provider_name)
        if not model_name.startswith(model_prefix) or model_name == model_prefix:
            raise ValueError(
                f"LITELLM_MODEL must use the LiteLLM '{model_prefix}<model>' "
                f"identifier format for {provider_name}."
            )
        _required_credential(credential_name, provider_name)
        chat_model = ChatLiteLLM(
            model=model_name,
            temperature=0,
            request_timeout=_LLM_REQUEST_TIMEOUT_SECONDS,
        )
    else:
        raise ValueError(
            "Unsupported AGENTSENTINEL_LLM_PROVIDER. Choose groq, openai, "
            "anthropic, or gemini."
        )

    return chat_model, provider_name, model_name


def create_chat_model(tools: Sequence[Any]) -> ChatModelSelection:
    """Build the configured provider model, bind tools, and return its identity."""
    chat_model, provider_name, model_name = create_unbound_chat_model()
    return ChatModelSelection(
        model=chat_model.bind_tools(list(tools)),
        provider=provider_name,
        model_name=model_name,
    )
