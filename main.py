"""FastAPI backend for the Haqiba Bags AI support agent.

Run locally from the project root::

    uvicorn app.api.main:app --reload --port 8000

Then open http://localhost:8000 for the demo UI, or http://localhost:8000/docs
for the interactive API docs.
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, Ticket, TicketsResponse
from app.db.database import DB_PATH, get_db_connection, init_db

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("app.api")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
# Per-IP limit on /api/chat. A public demo shares one Groq key, so this keeps a
# single visitor from burning through the free-tier quota. 0 disables it.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))


# --------------------------------------------------------------------------- #
# Startup: self-healing data layer
# --------------------------------------------------------------------------- #
def seed_database() -> None:
    from app.db.seed import seed

    seed()


def ingest_policies() -> None:
    from app.rag.ingest import ingest

    ingest()


def _database_needs_seed() -> bool:
    if not DB_PATH.exists():
        return True
    init_db()  # make sure all tables exist
    with get_db_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0


def _vector_store_needs_ingest() -> bool:
    from app.rag.config import CHROMA_DIR
    from app.rag.retriever import VectorStoreNotReadyError, get_vector_store

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return True
    try:
        get_vector_store()
        return False
    except VectorStoreNotReadyError:
        return True


def prepare_data() -> None:
    """Create the SQLite DB and the Chroma index if they are missing.

    Ephemeral hosts (Render free tier, Hugging Face Spaces) start from the
    image on every restart, so the app rebuilds whatever is missing. Existing
    data is never overwritten: tickets and orders survive normal restarts.
    """
    if _database_needs_seed():
        logger.info("Database missing or empty: seeding %s", DB_PATH)
        seed_database()
    else:
        logger.info("Database OK: %s", DB_PATH)

    if _vector_store_needs_ingest():
        logger.info("Vector store missing or empty: ingesting store policies")
        ingest_policies()
        from app.rag.retriever import get_vector_store

        get_vector_store.cache_clear()  # drop any handle cached before ingestion
    else:
        logger.info("Vector store OK")


@asynccontextmanager
async def lifespan(app: FastAPI):
    prepare_data()

    # Warm up: load the embedding model and build the graph now, so the first
    # visitor doesn't wait for it. A missing GROQ_API_KEY must not stop the
    # server: the UI, /health and /api/tickets still work, and /api/chat
    # answers 503 with a clear message.
    from app.rag.retriever import get_vector_store

    get_vector_store()
    try:
        from app.agent.graph import get_graph

        get_graph()
        app.state.agent_ready = True
        logger.info("Agent ready")
    except RuntimeError as exc:
        app.state.agent_ready = False
        logger.error("Agent not ready: %s", exc)
    yield


app = FastAPI(
    title="Haqiba Bags - AI Customer Support Agent",
    description="LangGraph ReAct agent with input guardrails, RAG over store policies, "
                "SQLite tools and human handoff.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Rate limiting (in-memory sliding window, per client IP)
# --------------------------------------------------------------------------- #
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")  # set by Render / HF proxies
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    now = time.monotonic()
    key = _client_ip(request)
    with _hits_lock:
        window = _hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Too many messages. Please wait a minute and try again.",
            )
        window.append(now)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"], dependencies=[Depends(rate_limit)])
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Send one message to the agent. Reuse ``thread_id`` to continue a conversation.

    Declared with ``def`` (not ``async def``) on purpose: the agent makes
    blocking network calls, so FastAPI runs it in a worker thread and the
    event loop stays free for other requests.
    """
    if not getattr(request.app.state, "agent_ready", False):
        raise HTTPException(
            status_code=503,
            detail="The agent is not configured on this server (GROQ_API_KEY is missing).",
        )

    from app.agent.graph import chat_with_agent

    start = time.perf_counter()
    try:
        result = chat_with_agent(req.message, req.thread_id)
    except Exception:
        logger.exception("Unhandled error in /api/chat (thread=%s)", req.thread_id)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your message. Please try again.",
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info(
        "chat thread=%s blocked=%s tools=%s latency_ms=%.0f",
        req.thread_id, result["guardrail_blocked"], result["tools_used"], latency_ms,
    )
    return ChatResponse(
        reply=result["reply"],
        tools_used=result["tools_used"],
        guardrail_blocked=result["guardrail_blocked"],
        latency_ms=latency_ms,
    )


@app.get("/api/tickets", response_model=TicketsResponse, tags=["support"])
def list_tickets() -> TicketsResponse:
    """All support tickets, newest first. Lets reviewers verify human-handoff tickets."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT ticket_id, customer_name, order_id, issue_summary,
                       priority, status, created_at
                FROM support_tickets
                ORDER BY created_at DESC, rowid DESC
                """
            ).fetchall()
    except Exception:
        logger.exception("Failed to list tickets")
        raise HTTPException(status_code=500, detail="Could not load support tickets.")

    tickets = [Ticket(**dict(r)) for r in rows]
    return TicketsResponse(count=len(tickets), tickets=tickets)
