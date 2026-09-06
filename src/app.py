"""Streamlit review UI -- the surface the demo is driven from.

Presentation only: no solving, no constraint logic, no arithmetic on money
beyond formatting minor units into major for display. Every figure on screen
comes from `positions.summarise()`, a `Plan` returned by the solver or the
naive baseline, or Worker A/B's modules.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run src/app.py` executes this file as a script with no package
# context, so relative imports fail. Put the repo root on sys.path and import
# `src` as a package regardless of how this file was invoked.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import baseline, metrics, positions as positions_mod, seed, solver, theme, tracing
from src.db import REPO_ROOT
from src.diagnosis import diagnose, record_override
from src.memo import write_escalation, write_memo
from src.models import format_money
from src.seed import SCENARIOS

# Neatlogs traces every solve/diagnose/memo call when a NEATLOGS_API_KEY is
# configured; without one -- or if Neatlogs itself is unreachable/misconfigured
# -- this degrades to plain untraced calls rather than crashing the app, since
# a trace key is an operator convenience, not a correctness requirement.
try:
    tracing.init()
    HAS_TRACING = True
except Exception:
    HAS_TRACING = False

DB_DIR = REPO_ROOT / "data" / "app"


def _db_path(scenario: str) -> Path:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR / f"{scenario}.db"


def _connect(scenario: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path(scenario)), timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_seeded(scenario: str) -> None:
    """Seed this scenario's db file if it doesn't exist yet.

    Seeding wipes `learned_rule`, so once a scenario has been seeded this run
    the override loop (Reject -> record_override -> re-diagnose) keeps
    accumulating on the same file instead of losing its state every rerun.
    """
    if _db_path(scenario).exists():
        return
    seed.seed(scenario, db_path=_db_path(scenario)).close()


@st.cache_data(show_spinner="Solving...")
def solve_scenario(scenario: str):
    _ensure_seeded(scenario)
    conn = _connect(scenario)
    conn.row_factory = sqlite3.Row
    try:
        if HAS_TRACING:
            plan = tracing.traced_solve(conn, scenario)
            violations = tracing.traced_verify(conn, plan)
        else:
            plan = solver.solve(conn, scenario)
            violations = solver.verify(conn, plan)
        naive = baseline.naive_plan(conn, scenario)
        summary = positions_mod.summarise(conn)
        opening = positions_mod.opening_balances(conn)
        run_metrics = metrics.measure(conn, plan, naive)
        metrics.persist(conn, run_metrics)
    finally:
        conn.close()
    return {
        "plan": plan,
        "violations": violations,
        "naive": naive,
        "summary": summary,
        "opening": opening,
        "metrics": run_metrics,
    }


@st.cache_data(show_spinner="Diagnosing...")
def get_diagnosis(scenario: str, version: int):
    _ensure_seeded(scenario)
    conn = _connect(scenario)
    conn.row_factory = sqlite3.Row
    try:
        plan = solver.solve(conn, scenario)
        if HAS_TRACING:
            return tracing.traced_diagnose(
                lambda: diagnose(conn, plan), scenario=scenario, plan=plan
            )
        return diagnose(conn, plan)
    finally:
        conn.close()


@st.cache_data(show_spinner="Writing memo...")
def get_memo(scenario: str, kind: str, version: int = 0):
    _ensure_seeded(scenario)
    conn = _connect(scenario)
    conn.row_factory = sqlite3.Row
    try:
        if kind == "memo":
            plan = solver.solve(conn, scenario)
            naive = baseline.naive_plan(conn, scenario)
            if HAS_TRACING:
                return tracing.traced_write_memo(
                    lambda: write_memo(conn, plan, naive), scenario=scenario, plan=plan
                )
            return write_memo(conn, plan, naive)
        diag = get_diagnosis(scenario, version)
        if diag is None:
            return None
        return write_escalation(diag)
    finally:
        conn.close()


def money(amount_minor: int, currency: str) -> str:
    return format_money(amount_minor, currency)


def reject_remedy(scenario: str, remedy, reason: str, run_id: str) -> None:
    conn = _connect(scenario)
    conn.row_factory = sqlite3.Row
    try:
        if HAS_TRACING:
            tracing.traced_record_override(
                lambda: record_override(conn, remedy, reason, run_id),
                scenario=scenario, rule_text=remedy.action,
            )
        else:
            record_override(conn, remedy, reason, run_id)
        conn.commit()
    finally:
        conn.close()
    st.session_state[f"version:{scenario}"] = st.session_state.get(f"version:{scenario}", 0) + 1


def render_positions(data: dict) -> None:
    st.subheader("Positions")
    summary = data["summary"]
    opening = data["opening"]
    rows = []
    for entity_id, s in sorted(summary.items()):
        rows.append({
            "Entity": entity_id,
            "Currency": s.currency,
            "Opening balance": money(opening.get(entity_id, 0), s.currency),
            "Floor": money(s.floor, s.currency),
            "Worst day": s.worst_day,
            "Min balance": money(s.min_balance, s.currency),
            "Shortfall": money(s.peak_shortfall, s.currency),
            "Free cash": money(s.free_cash, s.currency),
            "Breaches floor": s.peak_shortfall > 0,
        })

    def highlight(row):
        color = "background-color: #301917" if row["Breaches floor"] else ""
        return [color] * len(row)

    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.apply(highlight, axis=1),
        hide_index=True,
        use_container_width=True,
    )


def render_plan(data: dict) -> None:
    st.subheader("Plan")
    plan = data["plan"]
    naive = data["naive"]

    if not plan.feasible:
        st.info("No plan to show -- this scenario is infeasible. See Escalation below.")
        return

    rows = [{
        "From": t.from_entity,
        "To": t.to_entity,
        "Amount": money(t.amount_minor, t.amount_currency),
        "Send day": t.send_day,
        "Land day": t.land_day,
        "Lands as": money(t.landed_minor, t.landed_currency),
        "Rationale": t.rationale,
    } for t in plan.transfers]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fx_cost = sum(t.fx_cost_minor for t in plan.transfers)
    fee_cost = sum(t.fee_minor for t in plan.transfers)
    interest_cost = sum(t.interest_minor for t in plan.transfers)

    st.markdown("**Cost**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FX spread", money(fx_cost, "USD"))
    c2.metric("Wire fees", money(fee_cost, "USD"))
    c3.metric("IC interest", money(interest_cost, "USD"))
    c4.metric("Total", money(plan.total_cost_minor, "USD"))

    if naive.feasible:
        delta = naive.total_cost_minor - plan.total_cost_minor
        pct = (delta / naive.total_cost_minor * 100) if naive.total_cost_minor else 0.0
        st.markdown("**Cost vs. naive fund-from-HQ baseline**")
        c1, c2 = st.columns(2)
        c1.metric("Saved", money(delta, "USD"))
        c2.metric("Saved %", f"{pct:.1f}%")
    else:
        st.caption("Naive baseline is itself infeasible for this scenario -- no comparable cost.")


def render_memo(scenario: str, data: dict) -> None:
    st.subheader("Memo")
    plan = data["plan"]
    if not plan.feasible:
        st.caption("No approval memo for an infeasible plan -- see Escalation.")
        return
    try:
        text = get_memo(scenario, "memo")
    except Exception as exc:  # LLM call can fail; never crash the demo screen
        st.error(f"Memo generation failed: {exc}")
        return
    st.markdown(text)


def render_escalation(scenario: str, data: dict) -> None:
    plan = data["plan"]
    if plan.feasible:
        return

    st.subheader("Escalation")

    version = st.session_state.get(f"version:{scenario}", 0)
    run_id = f"{scenario}:v{version}"
    try:
        diag = get_diagnosis(scenario, version)
    except Exception as exc:
        st.error(f"Diagnosis failed: {exc}")
        return

    st.markdown(f"**Binding constraint:** {diag.binding_constraint}")
    st.write(diag.explanation)
    st.metric("Shortfall", money(diag.shortfall_minor, "USD"))

    st.markdown("**Ranked remedies**")
    for remedy in diag.remedies:
        with st.container(border=True):
            st.markdown(f"**#{remedy.rank} -- {remedy.action}**")
            st.write(f"{remedy.entity_id} · {money(remedy.amount_minor, remedy.currency)} · "
                     f"{remedy.kind} · {'reversible' if remedy.reversible else 'not reversible'}")
            st.write(remedy.business_cost)
            st.caption(remedy.rationale)
            if remedy.requires_signoff:
                st.caption(f"Requires sign-off: {remedy.requires_signoff}")

            key_base = f"{scenario}:{remedy.rank}:{remedy.action}"
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"approve:{key_base}"):
                st.success(f"Approved: {remedy.action}")
            if c2.button("Reject", key=f"reject:{key_base}"):
                reject_remedy(scenario, remedy, f"rejected via UI: {remedy.action}", run_id)
                st.rerun()

    st.markdown(f"**Recommendation:** {diag.recommendation}")
    st.caption(f"Escalate to: {diag.escalate_to}")

    st.markdown("---")
    st.markdown("**Escalation memo**")
    try:
        text = get_memo(scenario, "escalation", version)
    except Exception as exc:
        st.error(f"Escalation memo generation failed: {exc}")
    else:
        if text:
            st.markdown(text)


def render_metrics(data: dict) -> None:
    st.subheader("Metrics")
    plan = data["plan"]
    naive = data["naive"]
    violations = data["violations"]
    m = data["metrics"]

    c1, c2, c3, c4 = st.columns(4)

    if violations:
        c1.metric("Constraint violations", len(violations))
        c1.error("\n".join(violations))
    else:
        c1.metric("Constraint violations", m.violations)

    if plan.feasible and naive.feasible:
        c2.metric("Cost saved vs. baseline", money(m.saved_minor, "USD"))
    else:
        c2.metric("Cost saved vs. baseline", "N/A")

    c3.metric("Time to plan", f"{m.solve_seconds:.2f}s", help="Manual process: 1-2 hours")
    c4.metric("Autonomy", "Yes" if not m.escalated else "Escalated")


def main() -> None:
    st.set_page_config(page_title="Sluice", layout="wide")
    theme.apply()
    st.title("Sluice -- multi-entity cash positioning")

    scenario = st.selectbox("Scenario", SCENARIOS, key="scenario")

    data = solve_scenario(scenario)

    render_metrics(data)
    st.divider()
    render_positions(data)
    st.divider()
    render_plan(data)
    st.divider()
    render_memo(scenario, data)
    st.divider()
    render_escalation(scenario, data)

    if HAS_TRACING:
        # Flush at the end of each script run, not just at process exit --
        # a batched exporter that only flushes on interpreter shutdown never
        # actually flushes in a long-lived Streamlit server process.
        tracing.flush()


if __name__ == "__main__":
    main()
