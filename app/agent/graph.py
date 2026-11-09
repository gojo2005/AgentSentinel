"""LangGraph tool-calling agent for the AgentSentinel sample application."""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from dotenv import dotenv_values
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.model_factory import create_chat_model, get_model_identity
from app.agent.tools import calculator, knowledge_base_search, mock_weather
from app.monitoring.event_logger import EventLogger
from app.monitoring.pricing import estimate_model_cost


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_NAME = "AgentSentinel Sample AI Assistant"
_RUN_ID_CONFIG_KEY = "agentsentinel_run_id"
_RUN_ID_METADATA_KEY = "agentsentinel_run_id"
_EVENT_LOGGER_CONFIG_KEY = "agentsentinel_event_logger"
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret)\b\s*[:=]\s*)"
    r"([^\s,;]+)|\b(?:gsk_|sk-|sk_|xox[baprs]-)[A-Za-z0-9_-]{16,}\b"
)

_TOOLS = [calculator, knowledge_base_search, mock_weather]
_TOOL_NODE = ToolNode(_TOOLS)
_SYSTEM_PROMPT = """\
You are AgentSentinel's sample AI assistant.

You can:
- Perform arithmetic using the calculator tool.
- Search local project documents using the knowledge_base_search tool.
- Retrieve simulated weather using the mock_weather tool.

Use the appropriate tool when it is necessary to answer the user's question.
Never describe mock weather as live or real-time weather. Provide clear,
concise answers.
"""


def _run_id_from_config(config: RunnableConfig) -> UUID:
    """Get this agent invocation's identifier from LangGraph's run config."""
    configurable = config.get("configurable", {})
    run_id = configurable.get(_RUN_ID_CONFIG_KEY)
    if not isinstance(run_id, UUID):
        raise ValueError("The LangGraph run is missing its AgentSentinel run ID.")
    return run_id


def _event_logger_from_config(config: RunnableConfig) -> EventLogger:
    """Get the event logger provided for this LangGraph invocation."""
    configurable = config.get("configurable", {})
    logger = configurable.get(_EVENT_LOGGER_CONFIG_KEY)
    if not isinstance(logger, EventLogger):
        raise ValueError("The LangGraph run is missing its AgentSentinel event logger.")
    return logger


def _safe_error_description(error: Exception, operation: str) -> str:
    """Describe an error without storing potentially sensitive exception text."""
    return f"{operation} failed ({type(error).__name__})."


def _reported_token_count(value: Any) -> int | None:
    """Return a positive integer token count, excluding adapter zero defaults."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _extract_token_usage(response: Any | None) -> dict[str, Any]:
    """Normalize reported usage while treating missing/ambiguous counts as unknown."""
    usage_metadata = getattr(response, "usage_metadata", None)
    if not isinstance(usage_metadata, Mapping):
        usage_metadata = {}

    input_tokens = _reported_token_count(usage_metadata.get("input_tokens"))
    output_tokens = _reported_token_count(usage_metadata.get("output_tokens"))
    total_tokens = _reported_token_count(usage_metadata.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "available": all(
            count is not None
            for count in (input_tokens, output_tokens, total_tokens)
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _redact_sensitive_data(value: Any) -> Any:
    """Redact configured credentials and common secret formats from event data."""
    dotenv_settings = dotenv_values(_PROJECT_ROOT / ".env")
    secret_values = {
        secret.strip()
        for name, secret in {
            **dotenv_settings,
            **os.environ,
        }.items()
        if secret
        and any(
            marker in name.casefold()
            for marker in ("key", "secret", "token", "password", "credential")
        )
    }

    if isinstance(value, str):
        redacted_value = value
        for secret in secret_values:
            redacted_value = redacted_value.replace(secret, "[REDACTED]")
        return _CREDENTIAL_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}[REDACTED]"
                if match.group(1)
                else "[REDACTED]"
            ),
            redacted_value,
        )
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(
                    marker in str(key).casefold()
                    for marker in ("key", "secret", "token", "password", "credential")
                )
                else _redact_sensitive_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_data(item) for item in value)
    return value


def call_model(
    state: MessagesState,
    config: RunnableConfig,
) -> dict[str, list[AnyMessage]]:
    """Ask the configured provider model for the next assistant message."""
    run_id = _run_id_from_config(config)
    logger = _event_logger_from_config(config)
    start_time = perf_counter()
    provider, model_name = get_model_identity()
    token_usage = _extract_token_usage(None)
    try:
        selection = create_chat_model(_TOOLS)
        provider = selection.provider
        model_name = selection.model_name
        response = selection.model.invoke(state["messages"], config=config)
        token_usage = _extract_token_usage(response)
    except Exception as error:
        logger.log_event(
            run_id=run_id,
            event_type="llm_call",
            agent_name=_AGENT_NAME,
            provider=provider,
            model=model_name,
            status="failure",
            input={
                "message_count": len(state["messages"]),
                "message_types": [message.type for message in state["messages"]],
            },
            duration_ms=(perf_counter() - start_time) * 1000,
            error=_safe_error_description(error, "Model invocation"),
            metadata={
                "token_usage": token_usage,
                "estimated_cost": estimate_model_cost(
                    provider,
                    model_name,
                    token_usage,
                ),
            },
        )
        raise

    logger.log_event(
        run_id=run_id,
        event_type="llm_call",
        agent_name=_AGENT_NAME,
        provider=provider,
        model=model_name,
        status="success",
        input={
            "message_count": len(state["messages"]),
            "message_types": [message.type for message in state["messages"]],
        },
        output=_redact_sensitive_data(response.content),
        duration_ms=(perf_counter() - start_time) * 1000,
        metadata={
            "token_usage": token_usage,
            "estimated_cost": estimate_model_cost(
                provider,
                model_name,
                token_usage,
            ),
        },
    )
    return {"messages": [response]}


def call_tools(
    state: MessagesState,
    config: RunnableConfig,
) -> dict[str, list[AnyMessage]]:
    """Run requested tools and log their results for the current agent run."""
    run_id = _run_id_from_config(config)
    logger = _event_logger_from_config(config)
    start_time = perf_counter()
    tool_calls = (
        state["messages"][-1].tool_calls
        if state["messages"] and isinstance(state["messages"][-1], AIMessage)
        else []
    )

    try:
        tool_updates = _TOOL_NODE.invoke(state, config=config)
    except Exception as error:
        duration_ms = (perf_counter() - start_time) * 1000
        logger.log_event(
            run_id=run_id,
            event_type="tool_call",
            agent_name=_AGENT_NAME,
            tool_name="tool_batch",
            status="failure",
            duration_ms=duration_ms,
            error=_safe_error_description(error, "Tool execution"),
            metadata={
                "duration_scope": "tool_batch",
                "requested_tool_count": len(tool_calls),
                "requested_tool_names": [call["name"] for call in tool_calls],
            },
        )
        raise

    duration_ms = (perf_counter() - start_time) * 1000
    tool_messages = [
        message
        for message in tool_updates.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    results_by_call_id = {
        message.tool_call_id: message for message in tool_messages
    }

    for tool_call in tool_calls:
        tool_result = results_by_call_id.get(tool_call["id"])
        failed = tool_result is not None and tool_result.status == "error"
        logger.log_event(
            run_id=run_id,
            event_type="tool_call",
            agent_name=_AGENT_NAME,
            tool_name=tool_call["name"],
            status="failure" if failed else "success",
            input=_redact_sensitive_data(tool_call.get("args")),
            output=(
                _redact_sensitive_data(tool_result.content)
                if tool_result is not None
                else None
            ),
            duration_ms=duration_ms,
            error="Tool execution returned an error." if failed else None,
            metadata={
                "duration_scope": "tool_batch",
                "requested_tool_count": len(tool_calls),
            },
        )

    return tool_updates


def _build_agent():
    """Create and compile the model-and-tools execution graph."""
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("tools", call_tools)

    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges(
        "call_model",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "call_model")

    return graph_builder.compile()


agent = _build_agent()


def run_agent(question: str, logger: EventLogger | None = None) -> str:
    """Run the agent for one question and return its final answer."""
    if not question.strip():
        raise ValueError("Please provide a question for the agent.")

    selected_logger = logger if logger is not None else EventLogger()
    run_id = uuid4()
    start_time = perf_counter()
    selected_logger.log_event(
        run_id=run_id,
        event_type="run_started",
        agent_name=_AGENT_NAME,
        status="started",
        input={"question": _redact_sensitive_data(question)},
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=question),
                ]
            },
            config={
                "configurable": {
                    _RUN_ID_CONFIG_KEY: run_id,
                    _EVENT_LOGGER_CONFIG_KEY: selected_logger,
                },
                "metadata": {_RUN_ID_METADATA_KEY: str(run_id)},
            },
        )
        final_message = result["messages"][-1]
        if isinstance(final_message.content, str):
            answer = final_message.content
        else:
            text_blocks = [
                block["text"]
                for block in final_message.content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            answer = "\n".join(text_blocks)
    except Exception as error:
        selected_logger.log_event(
            run_id=run_id,
            event_type="run_failed",
            agent_name=_AGENT_NAME,
            status="failure",
            duration_ms=(perf_counter() - start_time) * 1000,
            error=_safe_error_description(error, "Agent execution"),
        )
        raise

    selected_logger.log_event(
        run_id=run_id,
        event_type="run_completed",
        agent_name=_AGENT_NAME,
        status="success",
        output=_redact_sensitive_data(answer),
        duration_ms=(perf_counter() - start_time) * 1000,
    )
    return answer
