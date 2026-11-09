"""Project configuration loaded from the AgentSentinel root ``.env`` file."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Absolute path to the AgentSentinel project root."""

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/agentsentinel.db")
"""Database connection URL; defaults to the local SQLite database."""


def get_llm_provider() -> str:
    """Return the selected LLM provider, defaulting to Groq."""
    return os.getenv("AGENTSENTINEL_LLM_PROVIDER", "groq").strip().lower() or "groq"
