"""Run metrics: the numbers that go on screen and in the README's
"what improved across iterations" section.

Every field is computed from the solver, the database or the clock. None is
estimated, and none comes from the LLM -- an agent narrating its own savings
percentage is exactly the kind of unverifiable number this project exists to
avoid.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import solver

_TABLE = "run_metrics"


@dataclass(frozen=True)
class RunMetrics:
    scenario: str
    violations: int
    cost_minor: int
    baseline_cost_minor: int
    saved_minor: int
    saved_pct: float
    solve_seconds: float
    escalated: bool
    trace_url: str | None


def measure(
    conn: sqlite3.Connection,
    plan: solver.Plan,
    baseline: solver.Plan,
    *,
    trace_url: str | None = None,
) -> RunMetrics:
    """Compare a solved plan against the naive baseline plan for the same
    scenario. `verify()` is re-run here rather than trusted from the caller,
    since a stale violation count defeats the point of the metric."""
    violations = len(solver.verify(conn, plan))

    cost = plan.total_cost_minor
    baseline_cost = baseline.total_cost_minor
    both_feasible = plan.feasible and baseline.feasible
    saved = (baseline_cost - cost) if both_feasible else 0
    saved_pct = (saved / baseline_cost * 100.0) if both_feasible and baseline_cost else 0.0

    return RunMetrics(
        scenario=plan.scenario,
        violations=violations,
        cost_minor=cost,
        baseline_cost_minor=baseline_cost,
        saved_minor=saved,
        saved_pct=saved_pct,
        solve_seconds=plan.solve_seconds,
        escalated=not plan.feasible,
        trace_url=trace_url,
    )


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario TEXT NOT NULL,
            violations INTEGER NOT NULL,
            cost_minor INTEGER NOT NULL,
            baseline_cost_minor INTEGER NOT NULL,
            saved_minor INTEGER NOT NULL,
            saved_pct REAL NOT NULL,
            solve_seconds REAL NOT NULL,
            escalated INTEGER NOT NULL,
            trace_url TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def persist(conn: sqlite3.Connection, m: RunMetrics) -> None:
    """Append one run's metrics. Runs accumulate so override recurrence and
    autonomy rate are measured across runs, not asserted from a single one."""
    _ensure_table(conn)
    conn.execute(
        f"""
        INSERT INTO {_TABLE} (
            scenario, violations, cost_minor, baseline_cost_minor,
            saved_minor, saved_pct, solve_seconds, escalated, trace_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            m.scenario,
            m.violations,
            m.cost_minor,
            m.baseline_cost_minor,
            m.saved_minor,
            m.saved_pct,
            m.solve_seconds,
            int(m.escalated),
            m.trace_url,
        ),
    )
    conn.commit()


def history(conn: sqlite3.Connection, scenario: str | None = None) -> list[dict]:
    """All persisted runs, oldest first, for trend charts."""
    _ensure_table(conn)
    if scenario is None:
        rows = conn.execute(f"SELECT * FROM {_TABLE} ORDER BY id")
    else:
        rows = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE scenario = ? ORDER BY id", (scenario,)
        )
    return [dict(r) for r in rows]


def _demo() -> None:
    from . import baseline
    from .seed import seed

    for scenario in ("base", "covenant_shock", "infeasible"):
        conn = seed(scenario, db_path=f"/tmp/sluice_metrics_demo_{scenario}.db")
        plan = solver.solve(conn, scenario)
        naive = baseline.naive_plan(conn, scenario)
        m = measure(conn, plan, naive)
        persist(conn, m)
        print(
            f"{scenario}: violations={m.violations} status={plan.status} "
            f"cost_minor={m.cost_minor} baseline_cost_minor={m.baseline_cost_minor} "
            f"saved_minor={m.saved_minor} saved_pct={m.saved_pct:.2f}% "
            f"solve_seconds={m.solve_seconds:.3f} escalated={m.escalated}"
        )


if __name__ == "__main__":
    _demo()
