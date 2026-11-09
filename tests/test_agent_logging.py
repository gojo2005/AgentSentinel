"""Tests for logger injection into the LangGraph agent invocation."""

from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.agent import graph
from app.agent.graph import run_agent
from app.agent.model_factory import ChatModelSelection
from app.monitoring.event_logger import EventLogger
from app.monitoring.schemas import ExecutionEvent


class FakeEventSink:
    """Collect events forwarded by an EventLogger."""

    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    def write_event(self, event: ExecutionEvent) -> None:
        """Store a forwarded event."""
        self.events.append(event)


@pytest.fixture(autouse=True)
def disable_langsmith_tracing_for_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep graph tests offline and independent of local credentials."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(graph, "dotenv_values", lambda _: {})


def install_successful_graph_invoke(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured_configs: list[dict[str, Any]] | None = None,
    invocation_count: list[int] | None = None,
) -> None:
    """Replace the compiled graph invocation with a deterministic answer."""
    def invoke(
        input_data: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, list[AIMessage]]:
        if captured_configs is not None:
            captured_configs.append(config)
        if invocation_count is not None:
            invocation_count.append(1)
        return {"messages": [AIMessage(content="A deterministic answer.")]}

    monkeypatch.setattr(graph.agent, "invoke", invoke)


def test_run_agent_without_logger_uses_in_memory_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[dict[str, Any]] = []
    install_successful_graph_invoke(
        monkeypatch,
        captured_configs=captured_configs,
    )

    answer = run_agent("Test question")

    assert answer == "A deterministic answer."
    assert len(captured_configs) == 1
    configurable = captured_configs[0]["configurable"]
    logger = configurable[graph._EVENT_LOGGER_CONFIG_KEY]
    assert isinstance(logger, EventLogger)
    assert logger.get_events()[0].event_type == "run_started"
    assert logger.get_events()[1].event_type == "run_completed"
    assert [event.event_type for event in logger.get_events()] == [
        "run_started",
        "run_completed",
    ]


def test_run_agent_uses_supplied_logger_and_records_run_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_successful_graph_invoke(monkeypatch)
    sink = FakeEventSink()
    custom_logger = EventLogger(sink=sink)

    answer = run_agent("Test question", logger=custom_logger)

    assert answer == "A deterministic answer."
    events = custom_logger.get_events()
    assert [event.event_type for event in events] == [
        "run_started",
        "run_completed",
    ]
    assert sink.events == events
    assert events[0].status == "started"
    assert events[1].status == "success"
    assert events[0].run_id == events[1].run_id


def test_run_agent_passes_supplied_logger_and_run_id_in_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[dict[str, Any]] = []
    install_successful_graph_invoke(
        monkeypatch,
        captured_configs=captured_configs,
    )
    custom_logger = EventLogger()

    run_agent("Test question", logger=custom_logger)

    configurable = captured_configs[0]["configurable"]
    assert configurable[graph._EVENT_LOGGER_CONFIG_KEY] is custom_logger
    run_id = configurable[graph._RUN_ID_CONFIG_KEY]
    assert isinstance(run_id, UUID)
    assert captured_configs[0]["metadata"][graph._RUN_ID_METADATA_KEY] == str(run_id)
    assert all(event.run_id == run_id for event in custom_logger.get_events())


def test_empty_question_raises_before_graph_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation_count: list[int] = []
    install_successful_graph_invoke(
        monkeypatch,
        invocation_count=invocation_count,
    )
    logger = EventLogger()

    with pytest.raises(ValueError, match="Please provide a question"):
        run_agent(" \t ", logger=logger)

    assert invocation_count == []
    assert logger.get_events() == []


def test_graph_failure_logs_run_failed_and_reraises_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = RuntimeError("graph invocation failed")

    def failing_invoke(
        input_data: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, list[AIMessage]]:
        raise original_error

    monkeypatch.setattr(graph.agent, "invoke", failing_invoke)
    logger = EventLogger()

    with pytest.raises(RuntimeError) as error_info:
        run_agent("Test question", logger=logger)

    assert error_info.value is original_error
    events = logger.get_events()
    assert [event.event_type for event in events] == ["run_started", "run_failed"]
    assert events[0].status == "started"
    assert events[1].status == "failure"
    assert events[0].run_id == events[1].run_id


def test_call_model_logs_selected_provider_and_model_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = AIMessage(
        content="A deterministic answer.",
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        },
    )
    selection = ChatModelSelection(
        model=RunnableLambda(lambda messages: response),
        provider="OpenAI",
        model_name="openai/configured-model",
    )
    monkeypatch.setattr(graph, "create_chat_model", lambda tools: selection)
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: run_id,
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    result = graph.call_model(
        {"messages": [HumanMessage(content="Test question")]},
        config,
    )

    event = logger.get_events()[0]
    assert result["messages"] == [response]
    assert event.event_type == "llm_call"
    assert event.provider == "OpenAI"
    assert event.model == "openai/configured-model"
    assert event.status == "success"
    assert event.duration_ms is not None
    assert event.error is None
    assert event.metadata["token_usage"] == {
        "available": True,
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }


@pytest.mark.parametrize(
    ("usage_metadata", "expected"),
    [
        (
            None,
            {
                "available": False,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        ),
        (
            {"input_tokens": 11, "output_tokens": 7},
            {
                "available": True,
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
        ),
        (
            {"input_tokens": 0, "output_tokens": 7, "total_tokens": 7},
            {
                "available": False,
                "input_tokens": None,
                "output_tokens": 7,
                "total_tokens": 7,
            },
        ),
        (
            {"input_tokens": 11, "total_tokens": 11},
            {
                "available": False,
                "input_tokens": 11,
                "output_tokens": None,
                "total_tokens": 11,
            },
        ),
        (
            {"input_tokens": 11, "output_tokens": 7, "total_tokens": 0},
            {
                "available": True,
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
        ),
    ],
)
def test_token_usage_extraction_handles_missing_and_ambiguous_fields(
    usage_metadata: dict[str, int] | None,
    expected: dict[str, int | bool | None],
) -> None:
    response = SimpleNamespace(usage_metadata=usage_metadata)

    assert graph._extract_token_usage(response) == expected


def test_call_model_logs_usage_unavailable_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = ChatModelSelection(
        model=RunnableLambda(lambda messages: AIMessage(content="No usage")),
        provider="Groq",
        model_name="configured-model",
    )
    monkeypatch.setattr(graph, "create_chat_model", lambda tools: selection)
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: UUID("12345678-1234-5678-1234-567812345678"),
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    graph.call_model({"messages": [HumanMessage(content="Question")]}, config)

    assert logger.get_events()[0].metadata["token_usage"] == {
        "available": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_call_model_records_versioned_estimated_cost_for_priced_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = AIMessage(
        content="A deterministic answer.",
        usage_metadata={
            "input_tokens": 1_000,
            "output_tokens": 500,
            "total_tokens": 1_500,
        },
    )
    selection = ChatModelSelection(
        model=RunnableLambda(lambda messages: response),
        provider="Anthropic",
        model_name="anthropic/claude-sonnet-5",
    )
    monkeypatch.setattr(graph, "create_chat_model", lambda tools: selection)
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: UUID("12345678-1234-5678-1234-567812345678"),
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    graph.call_model({"messages": [HumanMessage(content="Question")]}, config)

    estimate = logger.get_events()[0].metadata["estimated_cost"]
    assert estimate["available"] is True
    assert estimate["amount_usd"] == "0.007000000000"
    assert estimate["pricing_version"] == "2026-11-09-v1"
    assert estimate["verified_on"] == "2026-11-09"
    assert estimate["source"].startswith("https://")


def test_call_model_logs_sanitized_failure_for_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_detail = "provider failure includes sensitive-key-value"

    def fail_invocation(messages: list[HumanMessage]) -> AIMessage:
        raise RuntimeError(secret_detail)

    selection = ChatModelSelection(
        model=RunnableLambda(fail_invocation),
        provider="Gemini",
        model_name="gemini/configured-model",
    )
    monkeypatch.setattr(graph, "create_chat_model", lambda tools: selection)
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: run_id,
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    with pytest.raises(RuntimeError, match="provider failure"):
        graph.call_model(
            {"messages": [HumanMessage(content="Test question")]},
            config,
        )

    event = logger.get_events()[0]
    assert event.event_type == "llm_call"
    assert event.provider == "Gemini"
    assert event.model == "gemini/configured-model"
    assert event.status == "failure"
    assert event.duration_ms is not None
    assert event.error == "Model invocation failed (RuntimeError)."
    assert secret_detail not in str(event.model) + str(event.error)


def test_call_model_logs_timeout_without_sensitive_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout failures retain safe type metadata without raw provider errors."""
    secret_detail = "timeout response contained sensitive-api-key-value"

    def timeout_invocation(messages: list[HumanMessage]) -> AIMessage:
        raise TimeoutError(secret_detail)

    selection = ChatModelSelection(
        model=RunnableLambda(timeout_invocation),
        provider="Groq",
        model_name="configured-groq-model",
    )
    monkeypatch.setattr(graph, "create_chat_model", lambda tools: selection)
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: run_id,
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    with pytest.raises(TimeoutError, match="sensitive-api-key-value"):
        graph.call_model(
            {"messages": [HumanMessage(content="Test question")]},
            config,
        )

    event = logger.get_events()[0]
    assert event.event_type == "llm_call"
    assert event.status == "failure"
    assert event.error == "Model invocation failed (TimeoutError)."
    assert secret_detail not in str(event.model) + str(event.error)


def test_actual_graph_routes_tool_call_and_returns_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_inputs: list[list[Any]] = []
    model_configs: list[RunnableConfig] = []
    tool_configs: list[RunnableConfig] = []

    def respond(messages: list[Any], config: RunnableConfig) -> AIMessage:
        model_inputs.append(messages)
        model_configs.append(config)
        if len(model_inputs) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "2 + 3"},
                        "id": "calculator-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "2 + 3 = 5"
        return AIMessage(content="The answer is 5.")

    selection = ChatModelSelection(
        model=RunnableLambda(respond),
        provider="Groq",
        model_name="configured-groq-model",
    )
    monkeypatch.setattr(graph, "create_chat_model", lambda tools: selection)
    actual_tool_node = graph._TOOL_NODE

    class CapturingToolNode:
        def invoke(
            self,
            state: dict[str, Any],
            config: RunnableConfig,
        ) -> dict[str, Any]:
            tool_configs.append(config)
            return actual_tool_node.invoke(state, config=config)

    monkeypatch.setattr(graph, "_TOOL_NODE", CapturingToolNode())
    logger = EventLogger()

    answer = run_agent("What is 2 + 3?", logger=logger)

    assert answer == "The answer is 5."
    assert len(model_inputs) == 2
    events = logger.get_events()
    assert [event.event_type for event in events] == [
        "run_started",
        "llm_call",
        "tool_call",
        "llm_call",
        "run_completed",
    ]
    tool_event = events[2]
    assert tool_event.tool_name == "calculator"
    assert tool_event.status == "success"
    assert tool_event.output == "2 + 3 = 5"
    run_id = events[0].run_id
    assert len(model_configs) == 2
    assert len(tool_configs) == 1
    assert all(
        config["metadata"][graph._RUN_ID_METADATA_KEY] == str(run_id)
        for config in model_configs
    )
    assert tool_configs[0]["metadata"][graph._RUN_ID_METADATA_KEY] == str(run_id)
    assert model_configs[0]["configurable"][graph._EVENT_LOGGER_CONFIG_KEY] is logger


def test_model_construction_failure_logs_available_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: run_id,
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        graph.call_model(
            {"messages": [HumanMessage(content="Test question")]},
            config,
        )

    event = logger.get_events()[0]
    assert event.provider == "OpenAI"
    assert event.model == "openai/gpt-4o-mini"
    assert event.status == "failure"
    assert event.error == "Model invocation failed (ValueError)."


def test_model_constructor_exception_text_is_not_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_detail = "constructor received a sensitive-key-value"
    monkeypatch.setenv("AGENTSENTINEL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv(
        "LITELLM_MODEL",
        "anthropic/claude-3-5-sonnet-20240620",
    )
    monkeypatch.setattr(
        graph,
        "create_chat_model",
        lambda tools: (_ for _ in ()).throw(RuntimeError(secret_detail)),
    )
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    logger = EventLogger()
    config = {
        "configurable": {
            graph._RUN_ID_CONFIG_KEY: run_id,
            graph._EVENT_LOGGER_CONFIG_KEY: logger,
        }
    }

    with pytest.raises(RuntimeError, match="sensitive-key-value"):
        graph.call_model(
            {"messages": [HumanMessage(content="Test question")]},
            config,
        )

    event = logger.get_events()[0]
    assert event.provider == "Anthropic"
    assert event.model == "anthropic/claude-3-5-sonnet-20240620"
    assert event.status == "failure"
    assert event.error == "Model invocation failed (RuntimeError)."
    assert secret_detail not in str(event.error)
