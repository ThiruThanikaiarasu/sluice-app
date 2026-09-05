"""Neatlogs tracing for the Sluice agent loop.

A run is: solve -> (diagnose, only if infeasible) -> write_memo -> optional
record_override. Each step gets a named span carrying the numbers a judge
needs to see without reading code: scenario, plan status, cost, violation
count, solve time.

Neatlogs instruments OpenAI by patching it at *import* time. Anything that
imports `openai` before `neatlogs.init()` runs gets an unpatched client, and
the failure mode is silent -- the code runs fine, the dashboard is just
empty. `init()` in this module is the only supported entry point, and it must
run before the first call to `src.llm.complete()`. `src/llm.py` defers its
`from openai import OpenAI` into `client()` (an `lru_cache`d function, so the
import cost is paid once) specifically so no caller has to get import order
right -- as long as `tracing.init()` runs before the first `llm.complete()`
call anywhere in the process, every OpenAI call is captured.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Callable, TypeVar

import neatlogs
from neatlogs import span

from . import solver

T = TypeVar("T")

_initialized = False


def init() -> None:
    """Start the Neatlogs client. Idempotent -- safe to call from every entry
    point (CLI, Streamlit app, tests) without worrying about double-init."""
    global _initialized
    if _initialized:
        return
    api_key = os.environ.get("NEATLOGS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEATLOGS_API_KEY is unset. Check .env is loaded and symlinked."
        )
    neatlogs.init(
        api_key=api_key,
        endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://ingest.neatlogs.com"),
        workflow_name="sluice",
        instrumentations=["openai"],
    )
    _initialized = True


def flush() -> None:
    """Flush buffered spans. Call before process exit -- a batched exporter
    that never flushes looks identical to one that's silently broken."""
    neatlogs.flush()


def _plan_attrs(plan: solver.Plan) -> dict:
    return {
        "scenario": plan.scenario,
        "status": plan.status,
        "cost_minor": plan.total_cost_minor,
        "solve_seconds": round(plan.solve_seconds, 4),
        "transfer_count": len(plan.transfers),
        "binding_constraint_count": len(plan.binding_constraints),
    }


@span(kind="TOOL", tool_name="solve", capture_input=False, capture_output=False)
def traced_solve(conn: sqlite3.Connection, scenario: str) -> solver.Plan:
    """Run the solver inside a `solve` span, with the resulting plan's
    headline numbers attached so the trace shows what the solver decided
    without anyone having to open the plan object."""
    plan = solver.solve(conn, scenario)
    neatlogs.log("solve finished: {status}", **_plan_attrs(plan))
    return plan


@span(kind="TOOL", tool_name="verify", capture_input=False, capture_output=False)
def traced_verify(conn: sqlite3.Connection, plan: solver.Plan) -> list[str]:
    """Run the independent constraint check inside its own span so a judge
    can see, per run, that violations == 0 without re-deriving it."""
    violations = solver.verify(conn, plan)
    neatlogs.log(
        "verify finished: {violation_count} violations",
        scenario=plan.scenario,
        violation_count=len(violations),
    )
    return violations


@span(kind="AGENT", role="diagnose", capture_input=False, capture_output=False)
def traced_diagnose(fn: Callable[[], T], *, scenario: str, plan: solver.Plan) -> T:
    """Wrap whatever produces the infeasibility diagnosis (an `llm.complete`
    call fed `plan.binding_constraints`) so the span carries the plan it was
    diagnosing. The OpenAI call itself is auto-captured as a child span by
    the `instrumentations=["openai"]` patch, as long as `init()` already ran."""
    result = fn()
    neatlogs.log(
        "diagnosis produced",
        scenario=scenario,
        plan_status=plan.status,
        binding_constraint_count=len(plan.binding_constraints),
        output_chars=len(result) if isinstance(result, str) else None,
    )
    return result


@span(kind="AGENT", role="write_memo", capture_input=False, capture_output=False)
def traced_write_memo(fn: Callable[[], T], *, scenario: str, plan: solver.Plan) -> T:
    """Wrap the memo-writing LLM call with the plan it is writing about."""
    result = fn()
    neatlogs.log(
        "memo written",
        scenario=scenario,
        plan_status=plan.status,
        cost_minor=plan.total_cost_minor,
        output_chars=len(result) if isinstance(result, str) else None,
    )
    return result


@span(kind="TOOL", tool_name="record_override", capture_input=False, capture_output=False)
def traced_record_override(
    fn: Callable[[], T], *, scenario: str, rule_text: str
) -> T:
    """Wrap a treasurer override being written to `learned_rule` so the
    outcome (override recurrence trending to zero, or not) is traceable."""
    result = fn()
    neatlogs.log("override recorded", scenario=scenario, rule_text=rule_text)
    return result


def _demo() -> None:
    """Run one full traced loop against the base scenario and print the
    numbers, so a human (or `python -m src.tracing`) can confirm end to end
    that llm.py's deferred import actually closes the ordering trap: this
    module imports nothing from openai itself, calls init() first, and only
    then calls llm.complete(), which imports openai for the first time here."""
    from . import baseline, llm
    from .positions import summarise
    from .seed import seed

    init()

    conn = seed("base", db_path="/tmp/sluice_tracing_demo.db")
    plan = traced_solve(conn, "base")
    violations = traced_verify(conn, plan)
    print(f"solve: status={plan.status} cost_minor={plan.total_cost_minor} "
          f"transfers={len(plan.transfers)} solve_seconds={plan.solve_seconds:.3f}")
    print(f"verify: {len(violations)} violations")

    naive = baseline.naive_plan(conn, "base")
    print(f"naive: status={naive.status} cost_minor={naive.total_cost_minor}")

    summary = summarise(conn)
    memo_input = "\n".join(
        f"{eid}: floor={s.floor} closing={s.closing_balance} peak_shortfall={s.peak_shortfall}"
        for eid, s in summary.items()
    )
    memo = traced_write_memo(
        lambda: llm.complete(
            "You are a treasury analyst. Write a two-sentence summary of the "
            "cash position below for a CFO.",
            memo_input,
            max_tokens=4096,
        ),
        scenario="base",
        plan=plan,
    )
    print(f"memo ({len(memo)} chars): {memo[:200]}")

    flush()
    print("flushed -- check the Neatlogs dashboard for workflow 'sluice'")


if __name__ == "__main__":
    _demo()
