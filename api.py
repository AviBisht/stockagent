import os
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

# Ensure current stockagent package is in python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Ensure root workspace directory is also in sys.path
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from state import APP_NAME, USER_ID, SESSION, trading_state, MEMORY_BANK, INTERNAL_AUDIT_LOG
from trading_tools import get_portfolio_status
from Stock_agent import runner, session_service
from google.genai import types

# Initialize FastAPI application
app = FastAPI(
    title="StockAgent API",
    description="FastAPI backend for Autonomous StockAgent with Model Reliability Guardian (MRG)",
    version="1.0.0"
)

# CORS Middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development (e.g. localhost:5173, localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Trading command or market analysis prompt")

class ChatResponse(BaseModel):
    response: str
    status: str
    risk: Optional[Any] = None
    confidence: Optional[Any] = None

class SystemStatusResponse(BaseModel):
    app_name: str
    trading_active: bool
    status: str
    pause_reason: Optional[str] = None
    positions_count: int
    pending_orders_count: int
    api_key_configured: bool
    system: str

class PortfolioResponse(BaseModel):
    trading_active: bool
    pause_reason: Optional[str] = None
    positions: dict[str, Any]
    pending_orders: list[Any]
    status_summary: str

class MemoryResponse(BaseModel):
    risk_scores: dict[str, Any]
    strategies: dict[str, Any]

class AuditLogsResponse(BaseModel):
    audit_logs: dict[str, Any]


@app.get("/", tags=["Info"])
async def root():
    """Basic root health and info endpoint."""
    return {
        "service": "StockAgent API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "GET /api/status",
            "GET /api/portfolio",
            "GET /api/memory",
            "GET /api/audit-logs",
            "POST /api/chat"
        ]
    }


@app.get("/api/status", response_model=SystemStatusResponse, tags=["Monitoring"])
async def get_status():
    """Return whether the trading system is active/paused, pause reason, and basic system info."""
    return SystemStatusResponse(
        app_name=APP_NAME,
        trading_active=trading_state.is_trading_allowed(),
        status="active" if trading_state.is_trading_allowed() else "paused",
        pause_reason=trading_state.pause_reason,
        positions_count=len(trading_state.current_positions),
        pending_orders_count=len(trading_state.pending_orders),
        api_key_configured=bool(os.getenv("GOOGLE_API_KEY")),
        system="MRG_System (ADK Multi-Agent)"
    )


@app.get("/api/portfolio", response_model=PortfolioResponse, tags=["Trading"])
async def get_portfolio():
    """Return the current portfolio/positions using existing state and portfolio functions."""
    formatted_positions = {}
    for ticker, pos in trading_state.current_positions.items():
        formatted_positions[ticker] = {
            "quantity": pos.get("quantity"),
            "entry_price": pos.get("entry_price"),
            "timestamp": str(pos.get("timestamp")) if pos.get("timestamp") else None
        }

    return PortfolioResponse(
        trading_active=trading_state.is_trading_allowed(),
        pause_reason=trading_state.pause_reason,
        positions=formatted_positions,
        pending_orders=list(trading_state.pending_orders),
        status_summary=get_portfolio_status()
    )


@app.get("/api/memory", response_model=MemoryResponse, tags=["Memory & Audit"])
async def get_memory():
    """Return existing archived memory in a safe structured format."""
    return MemoryResponse(
        risk_scores=MEMORY_BANK.get("risk_scores", {}),
        strategies=MEMORY_BANK.get("strategies", {})
    )


@app.get("/api/audit-logs", response_model=AuditLogsResponse, tags=["Memory & Audit"])
async def get_audit_logs():
    """Return existing audit information in a safe structured format."""
    return AuditLogsResponse(
        audit_logs=INTERNAL_AUDIT_LOG
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Agent Chat"])
async def chat(request: ChatRequest):
    """
    Invoke the existing ADK Runner/session mechanism to process a user query.
    Returns agent response, system status, evaluated risk, and confidence score.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GOOGLE_API_KEY is not configured in environment or .env. Please configure your Gemini API key to invoke the StockAgent."
        )

    session_id = "WEB_SESSION"

    # Get or create ADK session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except Exception:
        try:
            session = await session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
        except Exception:
            session = None

    if not session:
        try:
            session = await session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize session: {type(e).__name__}"
            )

    query_content = types.Content(role="user", parts=[types.Part(text=request.query)])
    response_parts = []

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text and part.text != "None":
                        response_parts.append(part.text)
    except Exception as e:
        error_msg = str(e)
        if api_key in error_msg:
            error_msg = error_msg.replace(api_key, "[REDACTED_API_KEY]")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"StockAgent ADK execution error: {error_msg}"
        )

    final_text = "\n".join(response_parts).strip() if response_parts else "Operation completed with no textual output."

    # Retrieve latest risk and confidence metrics
    latest_risk = None
    latest_confidence = None

    if MEMORY_BANK.get("risk_scores"):
        last_item = list(MEMORY_BANK["risk_scores"].values())[-1]
        latest_risk = last_item.get("score")
        latest_confidence = last_item.get("confidence")

    if latest_confidence is None and INTERNAL_AUDIT_LOG.get("confidence"):
        last_log = INTERNAL_AUDIT_LOG["confidence"][-1]
        latest_confidence = last_log.get("value")

    status_str = "active" if trading_state.is_trading_allowed() else "paused"

    return ChatResponse(
        response=final_text,
        status=status_str,
        risk=latest_risk,
        confidence=latest_confidence
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting StockAgent API server on http://{host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=True)
