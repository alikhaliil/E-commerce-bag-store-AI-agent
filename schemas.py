"""Request/response models for the HTTP API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(..., min_length=1, max_length=1000, description="The customer's message.")
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Conversation id. Reuse it to keep memory; send a new one to start fresh.",
        examples=["web-3f2a9c1e"],
    )


class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str]
    guardrail_blocked: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    order_id: Optional[str]
    issue_summary: str
    priority: str
    status: str
    created_at: str


class TicketsResponse(BaseModel):
    count: int
    tickets: list[Ticket]
