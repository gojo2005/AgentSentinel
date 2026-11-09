"""Health check endpoint for the AgentSentinel API."""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def get_health() -> dict[str, str]:
    """Return the service status and application name."""
    return {"status": "healthy", "application": "AgentSentinel"}
