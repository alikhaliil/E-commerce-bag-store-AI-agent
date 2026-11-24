"""LangGraph ReAct agent with an input guardrail and per-thread memory.

    START -> guardrail --(blocked)--> END
                 |
                 +--(allowed)--> agent --(tool calls?)--> tools -> agent -> ... -> END

Interactive test from the project root (after seeding + ingesting)::

    python -m app.agent.graph
"""

import logging
import os
import uuid
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.guardrails import InputGuardrail
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS

load_dotenv()
logger = logging.getLogger(__name__)

# llama-3.3-70b-versatile was shut down by Groq on 2026-08-16 (free/dev tiers).
# gpt-oss-120b is Groq's recommended replacement. Override with GROQ_MODEL in .env.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

# Only the most recent messages are sent to the LLM each turn. The full
# history stays in the checkpointer. This keeps latency and Groq TPM usage
# bounded on long chats.
MAX_HISTORY_MESSAGES = int(os.getenv("AGENT_MAX_HISTORY_MESSAGES", "24"))

# Max graph steps per user turn. Stops a model that keeps calling tools in a loop.
RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))

# What a blocked user message is replaced with in the stored history, so an
# injection attempt never reaches the main agent on later turns.
BLOCKED_PLACEHOLDER = "[Message withheld by the safety filter: {category}]"

FALLBACK_REPLY = (
    "Sorry, I'm having trouble processing your request right now. Please try again "
    "in a moment, or contact our support team on WhatsApp during working hours. "
    "/ عذرًا، حصلت مشكلة مؤقتة. من فضلك حاول تاني بعد شوية أو كلّم خدمة العملاء على واتساب."
)


def create_llm() -> BaseChatModel:
    """Create the Groq chat model. Fails fast with a clear message if no API key."""
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, max_retries=2)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(parts).strip()


def _last_human_and_previous_ai(
    messages: list[BaseMessage],
) -> tuple[Optional[HumanMessage], str]:
    """Latest user message, and the assistant's last text reply before it."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            prev_ai = next(
                (
                    _message_text(m)
                    for m in reversed(messages[:i])
                    if isinstance(m, AIMessage) and not m.tool_calls and _message_text(m)
                ),
                "",
            )
            return messages[i], prev_ai
    return None, ""


def build_graph(
    llm: BaseChatModel,
    checkpointer: Any | None = None,
    guardrail: InputGuardrail | None = None,
) -> CompiledStateGraph:
    """Build and compile the guarded ReAct graph.

    ``llm`` and ``guardrail`` are parameters so tests and evals can inject
    fakes or other models without touching the wiring.
    """
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    guardrail = guardrail or InputGuardrail()

    def guardrail_node(state: AgentState) -> dict:
        human, previous_ai = _last_human_and_previous_ai(state["messages"])
        if human is None:
            return {"guardrail_blocked": False, "guardrail_category": "safe"}

        decision = guardrail.check(_message_text(human), previous_ai)
        if not decision.blocked:
            return {"guardrail_blocked": False, "guardrail_category": decision.category}

        return {
            "guardrail_blocked": True,
            "guardrail_category": decision.category,
            "messages": [
                # Same id -> add_messages replaces the original text in history.
                HumanMessage(
                    id=human.id,
                    content=BLOCKED_PLACEHOLDER.format(category=decision.category),
                ),
                AIMessage(content=decision.refusal or ""),
            ],
        }

    def route_after_guardrail(state: AgentState) -> str:
        return END if state.get("guardrail_blocked") else "agent"

    def agent_node(state: AgentState) -> dict:
        history = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,  # count messages, not tokens
            max_tokens=MAX_HISTORY_MESSAGES,
            start_on="human",  # never start mid tool-call / tool-result pair
            include_system=False,
        )
        response = llm_with_tools.invoke([SystemMessage(content=build_system_prompt()), *history])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", route_after_guardrail, {"agent": "agent", END: END})
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer or MemorySaver())


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """The app-wide graph (one LLM client and one in-memory checkpointer per process)."""
    return build_graph(create_llm())


def run_agent_turn(
    user_message: str,
    thread_id: str = "default",
    graph: CompiledStateGraph | None = None,
) -> dict:
    """Run one turn and return full details (used by evals and debugging).

    Returns ``reply``, ``tools_used``, ``guardrail_blocked``,
    ``guardrail_category``, ``tool_results`` (list of {name, content}),
    ``error`` (str or None) and ``thread_id``.
    """
    graph = graph or get_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    details: dict[str, Any] = {
        "reply": FALLBACK_REPLY,
        "tools_used": [],
        "guardrail_blocked": False,
        "guardrail_category": "safe",
        "tool_results": [],
        "error": None,
        "thread_id": thread_id,
    }

    user_message = user_message.strip()
    if not user_message:
        details["reply"] = "How can I help you today?"
        return details

    turn_id = str(uuid.uuid4())
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message, id=turn_id)]}, config=config
        )
    except GraphRecursionError:
        logger.warning("Recursion limit hit on thread %s", thread_id)
        details["error"] = "recursion_limit"
        return details
    except Exception as exc:
        # Groq outages, rate limits after retries, malformed tool calls, etc.
        logger.exception("Agent turn failed on thread %s", thread_id)
        details["error"] = f"{type(exc).__name__}: {exc}"
        return details

    messages = result["messages"]
    start = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].id == turn_id), 0)
    turn_messages = messages[start + 1 :]

    details["guardrail_blocked"] = bool(result.get("guardrail_blocked", False))
    details["guardrail_category"] = result.get("guardrail_category", "safe")
    details["tools_used"] = [
        call["name"]
        for msg in turn_messages
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
    ]
    details["tool_results"] = [
        {"name": msg.name, "content": _message_text(msg)}
        for msg in turn_messages
        if isinstance(msg, ToolMessage)
    ]

    final = messages[-1]
    reply = _message_text(final) if isinstance(final, AIMessage) else ""
    details["reply"] = reply or FALLBACK_REPLY
    return details


def chat_with_agent(
    user_message: str,
    thread_id: str = "default",
    graph: CompiledStateGraph | None = None,
) -> dict:
    """Send one user message and return ``{"reply", "tools_used", "guardrail_blocked"}``.

    ``tools_used`` lists tool calls made during THIS turn only, in order.
    """
    d = run_agent_turn(user_message, thread_id=thread_id, graph=graph)
    return {
        "reply": d["reply"],
        "tools_used": d["tools_used"],
        "guardrail_blocked": d["guardrail_blocked"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    thread_id = "cli-session"
    print("=" * 70)
    print(f" Haqiba Bags Support Agent | model: {GROQ_MODEL} | thread: {thread_id}")
    print(" Type 'exit' to quit, 'new' to start a fresh conversation.")
    print("=" * 70)

    get_graph()  # fail fast on a missing API key before the first prompt

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "new":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            print(f"[started new conversation: {thread_id}]")
            continue

        result = chat_with_agent(user_input, thread_id=thread_id)
        tools = ", ".join(result["tools_used"]) or "none"
        shield = "BLOCKED" if result["guardrail_blocked"] else "passed"
        print(f"\nAgent: {result['reply']}")
        print(f"   [guardrail: {shield} | tools used: {tools}]")

    print("Goodbye!")
