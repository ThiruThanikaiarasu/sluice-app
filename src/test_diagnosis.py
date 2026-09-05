"""Tests for infeasibility diagnosis and ranked remedies.

Exercises the live TensorMux key -- no mocking of `complete()` -- since the
whole point of `diagnose()` is that it runs unmodified against the real
model. `.env` is symlinked into the worktree but not auto-loaded by the app,
so load it explicitly here.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import pytest

from . import diagnosis, solver
from .models import HARD
from .seed import seed

REJECTED_ACTION = "Delay MER-CA's payment: Office lease prepayment"


def _conn(scenario, name):
    return seed(scenario, db_path=f"/tmp/sluice_test_diag_{name}.db")


def _infeasible_plan():
    conn = _conn("infeasible", "diag")
    plan = solver.solve(conn, "infeasible")
    assert plan.status == "INFEASIBLE"
    return conn, plan


def test_diagnose_names_a_specific_binding_constraint():
    conn, plan = _infeasible_plan()
    d = diagnosis.diagnose(conn, plan)
    assert d.binding_constraint in plan.binding_constraints
    assert "over-constrained" not in d.binding_constraint.lower()
    assert len(d.binding_constraint) > 20


def test_diagnose_produces_three_or_more_ranked_remedies_with_amounts_and_costs():
    conn, plan = _infeasible_plan()
    d = diagnosis.diagnose(conn, plan)
    assert len(d.remedies) >= 3
    ranks = [r.rank for r in d.remedies]
    assert ranks == list(range(1, len(d.remedies) + 1))
    for r in d.remedies:
        assert r.amount_minor > 0
        assert r.business_cost.strip()
        assert r.action.strip()


def test_no_hard_covenant_ever_appears_in_remedies():
    conn, plan = _infeasible_plan()
    d = diagnosis.diagnose(conn, plan)
    for r in d.remedies:
        covenant = diagnosis._entity_covenant(conn, r.entity_id)
        if covenant and covenant["hardness"] == HARD:
            assert r.kind != diagnosis.SOFT_COVENANT, (
                f"remedy proposes softening a hard covenant at {r.entity_id}"
            )
    hard_ids = [
        eid for eid in {r.entity_id for r in d.remedies}
        if (c := diagnosis._entity_covenant(conn, eid)) and c["hardness"] == HARD
    ]
    assert set(hard_ids) == {"MER-IE", "MER-UK"}
    for eid in hard_ids:
        assert all(r.kind != diagnosis.SOFT_COVENANT for r in d.remedies if r.entity_id == eid)


def test_override_removes_rejected_remedy_from_next_run():
    conn, plan = _infeasible_plan()
    first = diagnosis.diagnose(conn, plan)
    target = next(r for r in first.remedies if r.action == REJECTED_ACTION)

    diagnosis.record_override(conn, target, "lease landlord relationship at risk", "run-test")
    rules = diagnosis.learned_rules(conn)
    assert any(r["rule_text"].startswith("Do not recommend") for r in rules)

    second = diagnosis.diagnose(conn, plan)
    assert not any(r.action == REJECTED_ACTION for r in second.remedies)


def test_diagnose_raises_on_feasible_plan():
    conn = _conn("base", "diag_base")
    plan = solver.solve(conn, "base")
    assert plan.status == "OPTIMAL"
    with pytest.raises(ValueError):
        diagnosis.diagnose(conn, plan)
