"""Tools the support agent can call.

Every tool:
  * validates its input with a strict Pydantic schema (``args_schema``),
  * uses parameterised SQL only (no string-formatted queries),
  * returns a plain string written for the LLM to read. Errors are returned
    as text instead of raised, so one failed lookup doesn't crash the
    agent run and the model can recover (ask the user, try another tool).

Quick manual test from the project root (after seeding + ingesting)::

    python -m app.agent.tools
"""

# NOTE: no `from __future__ import annotations` here. Pydantic and
# LangChain's @tool inspect these annotations at runtime, and real (non-string)
# annotations make the schemas robust to how the module is imported.
import logging
import re
import secrets
import sqlite3
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.database import get_db_connection
from app.rag.retriever import VectorStoreNotReadyError, retrieve_policies

logger = logging.getLogger(__name__)

CURRENCY = "EGP"
MAX_PRODUCT_RESULTS = 10
TICKET_PREFIX = "TKT-"

# Words that say "show me everything" rather than filter by text.
_MATCH_ALL_QUERIES = {"", "*", "all", "any", "everything", "products", "bags", "bag"}

# Filler words dropped from keyword search. Short words like "or" would
# otherwise match inside "organiser" / "torso". "bag(s)" is dropped because
# every product is a bag, so it narrows nothing and breaks AND-matching
# (e.g. "leather bag" would miss the "Classic Leather Tote").
_STOPWORDS = {
    "a", "an", "and", "or", "the", "for", "with", "in", "on", "of", "to", "my",
    "me", "i", "you", "your", "do", "have", "any", "some", "is", "are", "it",
    "that", "this", "bag", "bags", "one", "please", "show", "want", "need",
}


def _money(value: float) -> str:
    return f"{value:,.2f} {CURRENCY}"


def _normalize_order_id(raw: str) -> str:
    """' ord-1004 ' -> 'ORD-1004'; '1004' -> 'ORD-1004'."""
    value = re.sub(r"\s+", "", raw).upper()
    if value.isdigit():
        value = f"ORD-{value}"
    return value


# --------------------------------------------------------------------------- #
# Input schemas
# --------------------------------------------------------------------------- #
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PolicySearchInput(_StrictModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description=(
            "A self-contained question about store policy, in English, e.g. "
            "'refund timeline for cash on delivery orders' or "
            "'shipping time to Alexandria'."
        ),
    )


class ProductSearchInput(_StrictModel):
    query: str = Field(
        ...,
        max_length=100,
        description=(
            "Product name, category or keyword, e.g. 'leather', 'backpack', "
            "'travel', 'laptop'. Use 'all' to list every product (useful "
            "together with max_price)."
        ),
    )
    max_price: Optional[float] = Field(
        default=None,
        gt=0,
        description=f"Optional maximum price in {CURRENCY}. Omit if the customer gave no budget.",
    )


class OrderStatusInput(_StrictModel):
    order_id: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="The order ID exactly as the customer gave it, e.g. 'ORD-1004'.",
    )

    @field_validator("order_id")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return _normalize_order_id(v)


TicketPriority = Literal["low", "medium", "high", "urgent"]


class SupportTicketInput(_StrictModel):
    customer_name: str = Field(..., min_length=2, max_length=100, description="Customer's full name.")
    issue_summary: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description=(
            "One to three sentences a human agent can act on without reading "
            "the chat: what happened, what the customer wants, what was "
            "already tried."
        ),
    )
    order_id: Optional[str] = Field(
        default=None, max_length=32, description="Related order ID if there is one, e.g. 'ORD-1004'."
    )
    priority: TicketPriority = Field(
        default="medium",
        description=(
            "urgent: charged without an order, or a safety issue. "
            "high: damaged/defective/wrong item, delayed order, or a very upset customer. "
            "medium: return, exchange or warranty request, or a request for a human. "
            "low: general question or feedback."
        ),
    )

    @field_validator("priority", mode="before")
    @classmethod
    def _map_aliases(cls, v: object) -> object:
        # LLMs often say "normal"; the database and policy use "medium".
        if isinstance(v, str):
            v = v.strip().lower()
            return {"normal": "medium", "critical": "urgent"}.get(v, v)
        return v

    @field_validator("order_id")
    @classmethod
    def _normalize(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        return _normalize_order_id(v)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool("search_store_policies", args_schema=PolicySearchInput)
def search_store_policies(query: str) -> str:
    """Search the store's OFFICIAL policy knowledge base.

    WHEN to use: any question about shipping times or fees, free shipping,
    returns, exchanges, refunds and refund timelines, warranty, bag care,
    payment methods (Cash on Delivery, Instapay, cards), order cancellation
    rules, what an order status means, working hours, or how escalation works.

    HOW to use: pass one focused question. If the customer asks about two
    different topics, call this tool once per topic.

    RULES: answer policy questions ONLY from what this tool returns. If it
    says no policy was found, do not guess or invent a policy: tell the
    customer you are not sure and offer to connect them with a human agent.
    """
    try:
        hits = retrieve_policies(query)
    except VectorStoreNotReadyError as exc:
        logger.error("Policy search unavailable: %s", exc)
        return (
            "ERROR: The policy knowledge base is unavailable right now. Do not "
            "answer from memory. Apologise and offer to create a support ticket."
        )
    except Exception:
        logger.exception("Policy search failed for query %r", query)
        return "ERROR: Policy search failed. Do not guess the policy; offer a human agent."

    if not hits:
        return (
            f"NO OFFICIAL POLICY FOUND for: '{query}'. The knowledge base has no "
            "relevant information. Do NOT invent an answer. Tell the customer you "
            "don't have that information and offer to escalate to a human agent."
        )

    blocks = [
        f"[Result {i} | section: {h['section']} | relevance: {h['score']:.2f}]\n{h['content']}"
        for i, h in enumerate(hits, start=1)
    ]
    return "Official store policy excerpts (most relevant first):\n\n" + "\n\n".join(blocks)


@tool("search_products", args_schema=ProductSearchInput)
def search_products(query: str, max_price: Optional[float] = None) -> str:
    """Search the product catalogue for prices, descriptions and live stock.

    WHEN to use: the customer asks what bags you sell, looks for a type of
    bag (backpack, tote, laptop bag, travel, kids...), asks about a
    specific product, its price or whether it is in stock, or gives a budget.

    HOW to use: pass short keywords, not a full sentence ('leather tote',
    not 'do you have a nice leather tote for work?'). Put any budget in
    max_price. Use query='all' to list everything.

    RULES: always state availability exactly as returned. Never offer to
    sell or reserve an item marked 'Out of Stock'; suggest an in-stock
    alternative from the results instead (search again by category if needed).
    """
    match_all = query.strip().lower() in _MATCH_ALL_QUERIES
    terms = [
        t for t in re.findall(r"[\w']+", query.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]
    budget = f" under {_money(max_price)}" if max_price is not None else ""
    if not match_all and not terms:
        return (
            f"No products found matching '{query}'{budget}: the query has no usable "
            "keywords. Search again with a product type, material or category "
            "(e.g. 'backpack', 'leather', 'travel'), or 'all'."
        )

    def run(mode: Literal["and", "or"]) -> list[sqlite3.Row]:
        where: list[str] = []
        params: list[object] = []
        if not match_all:
            per_term = []
            for term in terms:
                like = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                per_term.append(
                    "(name LIKE ? ESCAPE '\\' OR category LIKE ? ESCAPE '\\' "
                    "OR description LIKE ? ESCAPE '\\')"
                )
                params.extend([like, like, like])
            joiner = " AND " if mode == "and" else " OR "
            where.append("(" + joiner.join(per_term) + ")")
        if max_price is not None:
            where.append("price <= ?")
            params.append(max_price)

        sql = "SELECT name, category, price, stock, description FROM products"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY (stock > 0) DESC, price ASC LIMIT ?"
        params.append(MAX_PRODUCT_RESULTS)

        with get_db_connection() as conn:
            return conn.execute(sql, params).fetchall()

    try:
        rows = run("and")
        broadened = False
        if not rows and len(terms) > 1:
            rows = run("or")  # nothing matched every keyword; try any keyword
            broadened = True
    except sqlite3.Error:
        logger.exception("Product search failed for %r", query)
        return "ERROR: The product catalogue is unavailable right now. Please try again shortly."

    if not rows:
        return (
            f"No products found matching '{query}'{budget}. Tell the customer we "
            "don't carry that, and offer to search a related category instead."
        )

    lines = []
    for r in rows:
        availability = f"In Stock ({r['stock']} left)" if r["stock"] > 0 else "Out of Stock"
        lines.append(
            f"- {r['name']} | Category: {r['category']} | Price: {_money(r['price'])} | "
            f"Availability: {availability}\n  {r['description']}"
        )

    header = f"Found {len(rows)} product(s) matching '{query}'{budget}"
    if broadened:
        header += " (no product matched every keyword; showing partial matches)"
    return header + ":\n" + "\n".join(lines)


_STATUS_NOTES = {
    "pending": "Not confirmed yet. It can still be cancelled free of charge.",
    "processing": "Being prepared in the warehouse. It can still be cancelled free of charge.",
    "shipped": "With the courier. It can no longer be cancelled, only returned after delivery.",
    "delivered": "Delivered. A return is possible within 14 days of delivery.",
    "delayed": (
        "This order is DELAYED. Apologise, and check the 'Delayed Orders' policy. If the "
        "customer is upset or wants a cancellation, refund or compensation, create a "
        "high-priority support ticket."
    ),
    "cancelled": "This order was cancelled. Any payment is refunded per the refund timeline.",
    "returned": "The item was returned to the warehouse; the refund or exchange is in progress.",
}


@tool("check_order_status", args_schema=OrderStatusInput)
def check_order_status(order_id: str) -> str:
    """Look up the live status of a customer's order by its order ID.

    WHEN to use: the customer asks where their order is, when it will
    arrive, for a tracking number, or about anything concerning a specific
    order they placed.

    HOW to use: you need the order ID (format 'ORD-' followed by digits, e.g.
    ORD-1004). If the customer hasn't given it, ask for it first. Do not
    guess or make one up.

    RULES: if the order is not found, ask the customer to double-check the
    ID. Never invent order details, dates or tracking numbers.
    """
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT order_id, customer_name, product_name, status,
                       tracking_number, total_price, updated_at
                FROM orders WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Order lookup failed for %s", order_id)
        return "ERROR: The order system is unavailable right now. Please try again shortly."

    if row is None:
        return (
            f"Order not found: '{order_id}'. No order with this ID exists. Ask the "
            "customer to double-check the order ID (format: ORD-1234), e.g. from "
            "their confirmation SMS or email."
        )

    tracking = row["tracking_number"] or "Not assigned yet (assigned once the order ships)"
    note = _STATUS_NOTES.get(row["status"], "")
    return (
        f"Order {row['order_id']}\n"
        f"- Customer name: {row['customer_name']}\n"
        f"- Product: {row['product_name']}\n"
        f"- Status: {row['status'].upper()}\n"
        f"- Tracking number: {tracking}\n"
        f"- Total price: {_money(row['total_price'])}\n"
        f"- Last updated: {row['updated_at']} (UTC)\n"
        f"Note for agent: {note}"
    )


def _new_ticket_id() -> str:
    return TICKET_PREFIX + secrets.token_hex(3).upper()  # e.g. TKT-8F3A21


@tool("create_support_ticket", args_schema=SupportTicketInput)
def create_support_ticket(
    customer_name: str,
    issue_summary: str,
    order_id: Optional[str] = None,
    priority: TicketPriority = "medium",
) -> str:
    """Escalate the conversation to a HUMAN support agent by opening a ticket.

    WHEN to use (any one is enough):
      - The customer asks to speak to a human, a manager, or "a real person".
      - The customer is angry, frustrated, or repeating a complaint.
      - An order is delayed and the customer wants a cancellation, refund or
        compensation, or it needs manual follow-up with the courier.
      - An item arrived damaged, defective, or wrong.
      - A payment problem: charged without an order, missing refund, double charge.
      - A request falls outside standard policy, or you could not solve the
        issue after two attempts.

    HOW to use: you need the customer's name. Include the order ID when the
    issue is about an order. Write issue_summary so a human can act on it
    without reading the chat. Choose priority per the field description.
    Create ONE ticket per issue. Don't create a duplicate if you already
    opened one in this conversation.

    AFTER calling: give the customer the ticket ID and the expected
    first-response time for that priority, and tell them a human agent will
    follow up.
    """
    note = ""
    linked_order: Optional[str] = order_id
    try:
        with get_db_connection() as conn:
            if order_id is not None:
                exists = conn.execute(
                    "SELECT 1 FROM orders WHERE order_id = ?", (order_id,)
                ).fetchone()
                if exists is None:
                    # Don't block the handoff over a typo; keep what the
                    # customer said in the summary for the human agent.
                    linked_order = None
                    issue_summary = f"{issue_summary} (Customer-provided order ID '{order_id}' was not found.)"
                    note = f" Note: order ID '{order_id}' was not found, so the ticket is not linked to an order."

            for _ in range(5):  # retry on the (very unlikely) ID collision
                ticket_id = _new_ticket_id()
                try:
                    conn.execute(
                        """
                        INSERT INTO support_tickets
                            (ticket_id, customer_name, order_id, issue_summary, priority, status)
                        VALUES (?, ?, ?, ?, ?, 'open')
                        """,
                        (ticket_id, customer_name, linked_order, issue_summary, priority),
                    )
                    break
                except sqlite3.IntegrityError:
                    continue
            else:
                raise RuntimeError("Could not generate a unique ticket ID")
    except Exception:
        logger.exception("Ticket creation failed for %s", customer_name)
        return (
            "ERROR: The ticket could not be created. Apologise, and ask the customer "
            "to contact support via WhatsApp or email during working hours."
        )

    logger.info("Created ticket %s (priority=%s, order=%s)", ticket_id, priority, linked_order)
    order_part = f" for order {linked_order}" if linked_order else ""
    return (
        f"SUCCESS: Support ticket {ticket_id} created{order_part} with {priority.upper()} "
        f"priority and status OPEN. A human agent will follow up.{note} Give the "
        f"customer the ticket ID {ticket_id} and the first-response time for "
        f"{priority} priority (see 'Ticket Priorities & Response Times' in the store policies)."
    )


ALL_TOOLS = [search_store_policies, search_products, check_order_status, create_support_ticket]


# --------------------------------------------------------------------------- #
# Manual test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    test_cases: list[tuple[str, object, dict]] = [
        ("Policy: returns", search_store_policies, {"query": "What is your return policy?"}),
        ("Policy: off-topic (expect NO POLICY FOUND)", search_store_policies,
         {"query": "Do you sell smartphones and laptops?"}),
        ("Products: backpack (one is OUT OF STOCK)", search_products, {"query": "backpack"}),
        ("Products: travel (all travel bags)", search_products, {"query": "travel"}),
        ("Products: anything under 700 EGP", search_products, {"query": "all", "max_price": 700}),
        ("Products: multi-keyword fallback", search_products, {"query": "leather laptop"}),
        ("Products: generic 'leather bag'", search_products, {"query": "leather bag"}),
        ("Products: not sold (expect no match)", search_products, {"query": "smartphone"}),
        ("Products: SQL-injection attempt (treated as plain text, expect no match)",
         search_products, {"query": "' OR 1=1 --"}),
        ("Order: messy input ' ord-1002 '", check_order_status, {"order_id": " ord-1002 "}),
        ("Order: DELAYED", check_order_status, {"order_id": "ORD-1004"}),
        ("Order: invalid ID", check_order_status, {"order_id": "ORD-9999"}),
        ("Ticket: HIGH priority for delayed ORD-1004", create_support_ticket, {
            "customer_name": "Sara Mahmoud",
            "order_id": "ORD-1004",
            "priority": "high",
            "issue_summary": "Carry-on suitcase order is delayed in transit for 6 days. "
                             "Customer is frustrated and wants a firm delivery date or a refund.",
        }),
        ("Ticket: 'normal' priority alias, unknown order ID", create_support_ticket, {
            "customer_name": "Test Customer",
            "order_id": "ORD-0000",
            "priority": "normal",
            "issue_summary": "Customer asked to speak with a human about a gift order.",
        }),
    ]

    for title, fn, args in test_cases:
        print("=" * 80)
        print(f"{title}\n  -> {fn.name}({args})\n")
        print(fn.invoke(args))
    print("=" * 80)

    print("Schema validation check (invalid priority should be rejected):")
    try:
        create_support_ticket.invoke({
            "customer_name": "X Y", "issue_summary": "Testing invalid priority value.",
            "priority": "super-high",
        })
    except Exception as exc:  # pydantic.ValidationError
        print(f"  Rejected as expected: {type(exc).__name__}")
