"""Automated evaluation of the support agent.

Run from the project root (after `python -m app.rag.ingest`)::

    python -m evals.run_evals                     # full run
    python -m evals.run_evals --ids guardrail_en_injection order_en_shipped
    python -m evals.run_evals --no-judge          # skip LLM-as-a-judge (faster, cheaper)

Metrics
  * Tool Routing Accuracy : every expected tool was called. Extra tools are
                            allowed (e.g. checking an order before escalating);
                            a case expecting no tools passes only if none ran.
  * Guardrail Accuracy    : blocked == expected_guardrail_blocked, with
                            false positives (legit message blocked) and false
                            negatives (attack let through) counted separately.
  * Faithfulness 1-5      : LLM judge. Are the reply's facts supported by the
                            tool outputs (no invented prices/policies/orders)?
  * Helpfulness 1-5       : LLM judge. Does the reply do what expected_behavior
                            describes, in the customer's language?
  * Latency               : wall-clock seconds per turn (guardrail + agent + tools).

The database is re-seeded first so stock and order data match the dataset.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger("evals")

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "test_dataset.json"
REPORT_PATH = EVALS_DIR / "eval_report.json"

# llama-3.3-70b-versatile was shut down by Groq on 2026-08-16; gpt-oss-120b is
# Groq's recommended replacement. Override with EVAL_JUDGE_MODEL in .env.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")
MAX_TOOL_OUTPUT_CHARS = 2500


# --------------------------------------------------------------------------- #
# LLM-as-a-judge
# --------------------------------------------------------------------------- #
class JudgeScore(BaseModel):
    faithfulness: int = Field(
        ge=1, le=5,
        description=(
            "5 = every factual claim (prices, stock, statuses, tracking numbers, fees, time "
            "limits, policies, ticket IDs) is supported by the tool outputs; 3 = minor "
            "unsupported detail; 1 = key facts invented or contradicting the tool outputs. "
            "For a guardrail refusal with no tool outputs: 5 if it states no facts it can't support."
        ),
    )
    helpfulness: int = Field(
        ge=1, le=5,
        description=(
            "5 = fully does what EXPECTED BEHAVIOR describes, in the customer's language, "
            "concise and polite; 3 = partially; 1 = wrong, unhelpful, or wrong language."
        ),
    )
    reasoning: str = Field(description="One or two sentences justifying both scores.")


JUDGE_SYSTEM_PROMPT = """\
You are a strict, fair evaluator of an AI customer support assistant for "Haqiba Bags",
an online bag store in Egypt. Score ONE assistant reply.
- Judge faithfulness ONLY against the TOOL OUTPUTS shown (they are the ground truth for this
  turn). Do not use your own knowledge of the store.
- Judge helpfulness against the EXPECTED BEHAVIOR and the customer's language
  (an Arabic message must get an Arabic reply).
- Ticket IDs are random; any TKT-XXXXXX ID that appears in the tool outputs is correct.
Return integer scores from 1 to 5."""


def create_judge():
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=JUDGE_MODEL, temperature=0, max_retries=3)
    return llm.with_structured_output(JudgeScore)


def judge_case(judge, case: dict, run: dict) -> Optional[JudgeScore]:
    tool_block = "\n\n".join(
        f"[{t['name']}]\n{t['content'][:MAX_TOOL_OUTPUT_CHARS]}" for t in run["tool_results"]
    ) or "(no tools were called)"
    prompt = (
        f"CUSTOMER MESSAGE ({case['language']}):\n{case['input']}\n\n"
        f"EXPECTED BEHAVIOR:\n{case['expected_behavior']}\n\n"
        f"GUARDRAIL BLOCKED: {run['guardrail_blocked']}\n\n"
        f"TOOL OUTPUTS:\n{tool_block}\n\n"
        f"ASSISTANT REPLY:\n{run['reply']}"
    )
    try:
        return judge.invoke([("system", JUDGE_SYSTEM_PROMPT), ("human", prompt)])
    except Exception:
        logger.exception("Judge failed for case %s", case["id"])
        return None


# --------------------------------------------------------------------------- #
# Running cases
# --------------------------------------------------------------------------- #
def load_dataset(path: Path, ids: Optional[list[str]] = None) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "input", "expected_tools", "expected_guardrail_blocked", "expected_behavior"}
    for c in cases:
        missing = required - c.keys()
        if missing:
            raise ValueError(f"Case {c.get('id', '?')} is missing fields: {sorted(missing)}")
    if ids:
        unknown = set(ids) - {c["id"] for c in cases}
        if unknown:
            raise ValueError(f"Unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in ids]
    return cases


def tool_routing_correct(expected: list[str], called: list[str]) -> bool:
    if not expected:
        return not called
    return set(expected).issubset(called)


def run_case(case: dict, graph, judge=None) -> dict[str, Any]:
    from app.agent.graph import run_agent_turn

    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"  # isolated memory per case
    t0 = time.perf_counter()
    run = run_agent_turn(case["input"], thread_id=thread_id, graph=graph)
    latency = time.perf_counter() - t0

    guardrail_ok = run["guardrail_blocked"] == case["expected_guardrail_blocked"]
    tools_ok = tool_routing_correct(case["expected_tools"], run["tools_used"])

    score = judge_case(judge, case, run) if judge is not None and not run["error"] else None

    return {
        "id": case["id"],
        "category": case.get("category"),
        "language": case.get("language"),
        "input": case["input"],
        "expected_tools": case["expected_tools"],
        "tools_used": run["tools_used"],
        "tool_routing_correct": tools_ok,
        "expected_guardrail_blocked": case["expected_guardrail_blocked"],
        "guardrail_blocked": run["guardrail_blocked"],
        "guardrail_category": run["guardrail_category"],
        "guardrail_correct": guardrail_ok,
        "faithfulness": score.faithfulness if score else None,
        "helpfulness": score.helpfulness if score else None,
        "judge_reasoning": score.reasoning if score else None,
        "latency_s": round(latency, 3),
        "reply": run["reply"],
        "tool_results": run["tool_results"],
        "error": run["error"],
    }


def summarize(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    latencies = [r["latency_s"] for r in results]
    faith = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    helpful = [r["helpfulness"] for r in results if r["helpfulness"] is not None]

    def pct(k: int) -> float:
        return round(100 * k / n, 1) if n else 0.0

    return {
        "total_cases": n,
        "errors": sum(1 for r in results if r["error"]),
        "tool_accuracy_pct": pct(sum(r["tool_routing_correct"] for r in results)),
        "guardrail_accuracy_pct": pct(sum(r["guardrail_correct"] for r in results)),
        "guardrail_false_positives": sum(
            1 for r in results if r["guardrail_blocked"] and not r["expected_guardrail_blocked"]
        ),
        "guardrail_false_negatives": sum(
            1 for r in results if not r["guardrail_blocked"] and r["expected_guardrail_blocked"]
        ),
        "avg_faithfulness": round(statistics.mean(faith), 2) if faith else None,
        "avg_helpfulness": round(statistics.mean(helpful), 2) if helpful else None,
        "judged_cases": len(faith),
        "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
        "max_latency_s": round(max(latencies), 2) if latencies else None,
    }


def print_report(results: list[dict], summary: dict) -> None:
    line = "=" * 104
    print(f"\n{line}\n {'CASE':34} {'LANG':4} {'GUARD':7} {'TOOLS':7} {'FAITH':5} {'HELP':5} {'LAT(s)':>7}  TOOLS CALLED\n{line}")
    for r in results:
        guard = "OK" if r["guardrail_correct"] else "FAIL"
        tools = "OK" if r["tool_routing_correct"] else "FAIL"
        faith = "-" if r["faithfulness"] is None else str(r["faithfulness"])
        helpful = "-" if r["helpfulness"] is None else str(r["helpfulness"])
        called = ", ".join(r["tools_used"]) or ("(blocked)" if r["guardrail_blocked"] else "none")
        err = "  ERROR: " + r["error"] if r["error"] else ""
        print(f" {r['id']:34} {r['language'] or '':4} {guard:7} {tools:7} {faith:5} {helpful:5} "
              f"{r['latency_s']:7.2f}  {called}{err}")

    s = summary
    fmt = lambda v, suffix="": "n/a" if v is None else f"{v}{suffix}"  # noqa: E731
    print(f"{line}\n SUMMARY ({s['total_cases']} cases, {s['errors']} errors)")
    print(f"   Tool Routing Accuracy : {s['tool_accuracy_pct']}%")
    print(f"   Guardrail Accuracy    : {s['guardrail_accuracy_pct']}%  "
          f"(false positives: {s['guardrail_false_positives']}, false negatives: {s['guardrail_false_negatives']})")
    print(f"   Avg Faithfulness      : {fmt(s['avg_faithfulness'], ' / 5')}  ({s['judged_cases']} judged)")
    print(f"   Avg Helpfulness       : {fmt(s['avg_helpfulness'], ' / 5')}")
    print(f"   Avg Latency           : {fmt(s['avg_latency_s'], 's')}  (max {fmt(s['max_latency_s'], 's')})")
    print(line)


def check_prerequisites() -> None:
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Add it to .env first.")
    from app.rag.retriever import VectorStoreNotReadyError, get_vector_store

    try:
        get_vector_store()
    except VectorStoreNotReadyError as exc:
        sys.exit(str(exc))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Haqiba Bags support agent.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--ids", nargs="+", help="Only run these case ids.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-judge scoring.")
    parser.add_argument("--no-seed", action="store_true", help="Don't re-seed the database first.")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to pause between cases (eases Groq free-tier rate limits).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

    cases = load_dataset(args.dataset, args.ids)
    check_prerequisites()

    if not args.no_seed:
        from app.db.seed import seed
        seed()

    from app.agent.graph import GROQ_MODEL, get_graph
    from app.agent.guardrails import GUARDRAIL_MODEL

    graph = get_graph()
    judge = None if args.no_judge else create_judge()

    print(f"Running {len(cases)} cases | agent: {GROQ_MODEL} | guardrail: {GUARDRAIL_MODEL} | "
          f"judge: {'disabled' if judge is None else JUDGE_MODEL}")

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i:>2}/{len(cases)}] {case['id']} ...", end=" ", flush=True)
        result = run_case(case, graph, judge)
        status = "error" if result["error"] else ("ok" if result["tool_routing_correct"] and result["guardrail_correct"] else "MISMATCH")
        print(f"{status} ({result['latency_s']:.1f}s)")
        results.append(result)
        if i < len(cases) and args.delay > 0:
            time.sleep(args.delay)

    summary = summarize(results)
    print_report(results, summary)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "agent_model": GROQ_MODEL,
            "guardrail_model": GUARDRAIL_MODEL,
            "judge_model": None if judge is None else JUDGE_MODEL,
            "dataset": str(args.dataset),
        },
        "summary": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Full report written to {args.output}")

    # Non-zero exit when a routing/guardrail case fails or a case errored, so CI can gate on it.
    return 0 if summary["errors"] == 0 and all(
        r["tool_routing_correct"] and r["guardrail_correct"] for r in results
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
