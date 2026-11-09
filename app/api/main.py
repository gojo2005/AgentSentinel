"""FastAPI application entry point for AgentSentinel."""

from fastapi import FastAPI

from app.api.routes.agent import router as agent_router
from app.api.routes.diagnostics import router as diagnostics_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.executions import router as executions_router
from app.api.routes.health import router as health_router

app = FastAPI(
    title="AgentSentinel API",
    description="API for monitoring, evaluating, and diagnosing AI agent executions.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(agent_router, prefix="/api/v1")
app.include_router(executions_router, prefix="/api/v1")
app.include_router(evaluations_router, prefix="/api/v1")
app.include_router(diagnostics_router, prefix="/api/v1")
