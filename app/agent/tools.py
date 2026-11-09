"""Small, self-contained tools for the AgentSentinel sample agent."""

import ast
import math
import re
from pathlib import Path

from langchain_core.tools import tool


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_KNOWLEDGE_BASE_DIRECTORY = _PROJECT_ROOT / "data" / "knowledge_base"
_MAX_EXPRESSION_LENGTH = 256
_MAX_EXPONENT_MAGNITUDE = 1_000
_MAX_INTEGER_BITS = 4_096
_MAX_SEARCH_RESULTS = 5
_MAX_SNIPPETS_PER_FILE = 3


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    """Evaluate one supported arithmetic expression node."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only integer and decimal numbers are supported.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("The expression must use finite numbers.")
        return value

    if isinstance(node, ast.UnaryOp):
        value = _evaluate_arithmetic(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("Only unary plus and minus are supported.")

    if isinstance(node, ast.BinOp):
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT_MAGNITUDE:
                raise ValueError(
                    f"Exponents must be between -{_MAX_EXPONENT_MAGNITUDE} "
                    f"and {_MAX_EXPONENT_MAGNITUDE}."
                )
            result = left**right
        else:
            raise ValueError("That arithmetic operation is not supported.")

        if isinstance(result, complex):
            raise ValueError("Complex-number results are not supported.")
        if isinstance(result, int) and result.bit_length() > _MAX_INTEGER_BITS:
            raise ValueError("The result is too large.")
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("The result is too large or not a finite number.")
        return result

    raise ValueError("Only numbers and basic arithmetic operations are supported.")


@tool
def calculator(expression: str) -> str:
    """Safely calculate an expression using +, -, *, /, **, and parentheses.

    The expression may contain numeric literals and arithmetic operators only.
    Variables, function calls, and other Python syntax are rejected.
    """
    expression = expression.strip()
    if not expression:
        return "Error: Please provide a mathematical expression."
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return f"Error: Expressions must be {_MAX_EXPRESSION_LENGTH} characters or fewer."

    try:
        syntax_tree = ast.parse(expression, mode="eval")
        result = _evaluate_arithmetic(syntax_tree.body)
    except SyntaxError:
        return "Error: The expression has invalid syntax."
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except RecursionError:
        return "Error: The expression is nested too deeply."
    except (OverflowError, ValueError) as error:
        return f"Error: {error}"

    return f"{expression} = {result}"


@tool
def knowledge_base_search(query: str) -> str:
    """Find keyword matches in Markdown and text files under data/knowledge_base.

    Matching is case-insensitive. The result includes matching filenames and
    a few lines of context; this tool does not use embeddings or external APIs.
    """
    keywords = set(re.findall(r"\w+", query.casefold()))
    if not keywords:
        return "Please provide one or more keywords to search for."

    if not _KNOWLEDGE_BASE_DIRECTORY.is_dir():
        return (
            "Knowledge base directory not found: "
            f"{_KNOWLEDGE_BASE_DIRECTORY}"
        )

    matches: list[tuple[int, str, list[str]]] = []
    read_errors: list[str] = []

    try:
        documents = sorted(
            path
            for path in _KNOWLEDGE_BASE_DIRECTORY.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".txt", ".md"}
        )
    except OSError as error:
        return f"Could not list knowledge base documents: {error}"

    for document_path in documents:
        try:
            document_text = document_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            read_errors.append(f"{document_path.name}: {error}")
            continue

        matching_lines = [
            line.strip()
            for line in document_text.splitlines()
            if keywords.intersection(re.findall(r"\w+", line.casefold()))
        ]
        filename_words = set(re.findall(r"\w+", document_path.stem.casefold()))
        matched_keywords = keywords.intersection(filename_words)
        for line in matching_lines:
            matched_keywords.update(
                keywords.intersection(re.findall(r"\w+", line.casefold()))
            )

        if matched_keywords:
            relative_name = document_path.relative_to(
                _KNOWLEDGE_BASE_DIRECTORY
            ).as_posix()
            matches.append(
                (
                    len(matched_keywords),
                    relative_name,
                    matching_lines[:_MAX_SNIPPETS_PER_FILE],
                )
            )

    if not matches:
        message = "No matching documents found."
        if not documents:
            message = "No .txt or .md documents found in the knowledge base."
        if read_errors:
            message += "\nCould not read: " + "; ".join(read_errors)
        return message

    matches.sort(key=lambda match: (-match[0], match[1].casefold()))
    result_lines = ["Knowledge base matches:"]
    for _, filename, snippets in matches[:_MAX_SEARCH_RESULTS]:
        result_lines.append(f"- {filename}")
        if snippets:
            result_lines.extend(f"  {snippet}" for snippet in snippets)
        else:
            result_lines.append("  Filename matched; no matching text line found.")

    if read_errors:
        result_lines.append("Some documents could not be read: " + "; ".join(read_errors))
    return "\n".join(result_lines)


_SIMULATED_WEATHER: dict[str, tuple[int, int, str]] = {
    "jaipur": (34, 28, "Sunny"),
    "delhi": (31, 42, "Partly cloudy"),
    "mumbai": (29, 76, "Humid with light clouds"),
    "bangalore": (26, 54, "Mild and cloudy"),
}


@tool
def mock_weather(city: str) -> str:
    """Return simulated weather for Jaipur, Delhi, Mumbai, or Bangalore.

    The temperature, humidity, and conditions are fixed sample values, not a
    live forecast or a real-time observation.
    """
    normalized_city = city.strip().casefold()
    weather = _SIMULATED_WEATHER.get(normalized_city)
    if weather is None:
        supported_cities = ", ".join(
            city_name.title() for city_name in _SIMULATED_WEATHER
        )
        return (
            f"Unsupported city: {city.strip() or '(empty)'}. "
            f"Supported cities are: {supported_cities}."
        )

    temperature, humidity, conditions = weather
    return (
        f"Simulated weather for {city.strip().title()}: "
        f"{temperature} degrees C, {humidity}% humidity, {conditions}. "
        "This is simulated data, not real-time weather."
    )
