"""Graph state for the support agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation state, persisted per thread by the checkpointer.

    ``add_messages`` appends new messages to the history (and replaces a
    message in place when one with the same id is returned), so each node
    only returns the messages it produced.

    ``guardrail_blocked`` / ``guardrail_category`` are set by the guardrail
    node on EVERY turn, so they always describe the latest user message.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    guardrail_blocked: bool
    guardrail_category: str
