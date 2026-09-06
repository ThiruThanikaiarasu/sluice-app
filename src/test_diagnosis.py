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

REJECTED_ACTION = "Delay payables at MER-IE, MER-SG, MER-CA"


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


def test_diagnose_produces_three_or_four_consolidated_remedies():
    conn, plan = _infeasible_plan()
    d = diagnosis.diagnose(conn, plan)
    assert 3 <= len(d.remedies) <= 4
    ranks = [r.rank for r in d.remedies]
    assert ranks == list(range(1, len(d.remedies) + 1))
    for r in d.remedies:
        assert r.amount_minor > 0
        assert r.business_cost.strip()
        assert r.action.strip()


def test_no_hard_covenant_ever_appears_in_remedies():
    conn, plan = _infeasible_plan()
    d = diagnosis.diagnose(conn, plan)

    hard_ids = {
        sf_entity_id for sf_entity_id in
        (eid for r in d.remedies for eid in r.entity_id.split(", "))
        if (c := diagnosis._entity_covenant(conn, sf_entity_id)) and c["hardness"] == HARD
    }
    assert hard_ids == {"MER-IE", "MER-UK"}

    for r in d.remedies:
        entities = r.entity_id.split(", ")
        if r.kind == diagnosis.SOFT_COVENANT:
            assert not hard_ids.intersection(entities), (
                f"remedy proposes softening a hard covenant: {entities}"
            )


def test_override_removes_rejected_remedy_from_next_run():
    conn, plan = _infeasible_plan()
    first = diagnosis.diagnose(conn, plan)
    target = next(r for r in first.remedies if r.action == REJECTED_ACTION)

    diagnosis.record_override(conn, target, "lease landlord relationship at risk", "run-test")
    rules = diagnosis.learned_rules(conn)
    assert any(r["rule_text"].startswith("Do not recommend") for r in rules)

    second = diagnosis.diagnose(conn, plan)
    assert not any(r.action == REJECTED_ACTION for r in second.remedies)


def test_override_survives_a_changed_shortfall_set():
    """A rejection is keyed on (kind, entity), not on the exact set of
    entities that happened to be short in the run it was recorded in. If
    MER-SG stops qualifying for delay_payable, the earlier rejection of
    MER-IE and MER-CA must still hold even though the consolidated action
    string for that kind is now different from REJECTED_ACTION."""
    conn, plan = _infeasible_plan()
    first = diagnosis.diagnose(conn, plan)
    target = next(r for r in first.remedies if r.action == REJECTED_ACTION)
    diagnosis.record_override(conn, target, "lease landlord relationship at risk", "run-test")

    # MER-SG's payable no longer reads as delayable (e.g. reclassified as
    # a statutory obligation) -- the delay_payable candidate set for this
    # kind now covers only MER-IE and MER-CA, a different entity set than
    # the one the rejection above was recorded against.
    conn.execute(
        "UPDATE cash_forecast SET note = 'tax settlement' "
        "WHERE entity_id = 'MER-SG' AND note = 'APAC contractor settlement'"
    )
    conn.commit()

    second = diagnosis.diagnose(conn, plan)
    delay_remedies = [r for r in second.remedies if r.kind == diagnosis.DELAY_PAYABLE]
    assert not delay_remedies, (
        "MER-IE and MER-CA were already rejected for delay_payable; "
        f"they should not resurface just because the entity set changed: {delay_remedies}"
    )


def test_diagnose_raises_on_feasible_plan():
    conn = _conn("base", "diag_base")
    plan = solver.solve(conn, "base")
    assert plan.status == "OPTIMAL"
    with pytest.raises(ValueError):
        diagnosis.diagnose(conn, plan)
