"""Tests for infeasibility diagnosis and ranked remedies.

Exercises the live TensorMux key -- no mocking of `complete()` -- since the
whole point of `diagnose()` is that it runs unmodified against the real
model. `.env` is symlinked into the worktree but not auto-loaded by the app,
so load it explicitly here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import pytest

from . import diagnosis, solver
from .models import HARD
from .seed import seed

# These tests exercise the live TensorMux key -- no mocking of `complete()` --
# since the whole point of `diagnose()` is that it runs unmodified against
# the real model. Without a key configured, skip rather than fail: a missing
# credential in a dev/CI environment is not the same signal as a real defect.
# Applied per-test rather than as a module-level pytestmark so that
# test_diagnose_raises_on_feasible_plan -- which raises before diagnose()
# ever reaches the LLM call -- still runs without a key.
requires_llm = pytest.mark.skipif(
    not os.environ.get("SLUICE_LLM_API_KEY"),
    reason="requires a live SLUICE_LLM_API_KEY",
)

REJECTED_ACTION = "Delay payables at MER-IE, MER-SG, MER-CA"


def _conn(scenario, name, tmp_path):
    return seed(scenario, db_path=tmp_path / f"sluice_test_diag_{name}.db")


def test_revolver_remedy_caps_the_draw_at_the_facility_limit(tmp_path):
    """Pure-Python remedy construction, no LLM involved. MER-CA's seeded
    revolver_facility limit is USD 500,000 (RBC) -- a shortfall larger than
    that must be capped, not offered in full against a facility that doesn't
    have the room, and the business_cost must say so rather than silently
    understating what the lever actually closes."""
    conn = _conn("base", "revolver_cap", tmp_path)
    shortfall = solver.FloorShortfall(
        entity_id="MER-CA", currency="USD", amount_minor=70_000_000,
        day=5, line="MER-CA short 700000.00 USD of its floor on day 5",
    )
    remedy = diagnosis._build_revolver_remedy(conn, [shortfall], set())
    assert remedy is not None
    assert remedy.entity_id == "MER-CA"
    assert remedy.amount_minor == 50_000_000  # capped at the 500,000 USD limit
    assert "covers" in remedy.business_cost


def test_revolver_remedy_omits_an_entity_with_no_seeded_facility(tmp_path):
    conn = _conn("base", "revolver_missing", tmp_path)
    shortfall = solver.FloorShortfall(
        entity_id="MER-CA", currency="USD", amount_minor=10_000,
        day=5, line="MER-CA short 100.00 USD of its floor on day 5",
    )
    conn.execute("DELETE FROM revolver_facility WHERE entity_id = 'MER-CA'")
    conn.commit()
    remedy = diagnosis._build_revolver_remedy(conn, [shortfall], set())
    assert remedy is None


def _infeasible_plan(tmp_path):
    conn = _conn("infeasible", "diag", tmp_path)
    plan = solver.solve(conn, "infeasible")
    assert plan.status == "INFEASIBLE"
    return conn, plan


@requires_llm
def test_diagnose_names_a_specific_binding_constraint(tmp_path):
    conn, plan = _infeasible_plan(tmp_path)
    d = diagnosis.diagnose(conn, plan)
    assert d.binding_constraint in plan.binding_constraints
    assert "over-constrained" not in d.binding_constraint.lower()
    assert len(d.binding_constraint) > 20


@requires_llm
def test_diagnose_produces_three_or_four_consolidated_remedies(tmp_path):
    conn, plan = _infeasible_plan(tmp_path)
    d = diagnosis.diagnose(conn, plan)
    assert 3 <= len(d.remedies) <= 4
    ranks = [r.rank for r in d.remedies]
    assert ranks == list(range(1, len(d.remedies) + 1))
    for r in d.remedies:
        assert r.amount_minor > 0
        assert r.business_cost.strip()
        assert r.action.strip()


@requires_llm
def test_no_hard_covenant_ever_appears_in_remedies(tmp_path):
    conn, plan = _infeasible_plan(tmp_path)
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


@requires_llm
def test_override_removes_rejected_remedy_from_next_run(tmp_path):
    conn, plan = _infeasible_plan(tmp_path)
    first = diagnosis.diagnose(conn, plan)
    target = next(r for r in first.remedies if r.action == REJECTED_ACTION)

    diagnosis.record_override(conn, target, "lease landlord relationship at risk", "run-test")
    rules = diagnosis.learned_rules(conn)
    assert any(r["rule_text"].startswith("Do not recommend") for r in rules)

    second = diagnosis.diagnose(conn, plan)
    assert not any(r.action == REJECTED_ACTION for r in second.remedies)


@requires_llm
def test_override_survives_a_changed_shortfall_set(tmp_path):
    """A rejection is keyed on (kind, entity), not on the exact set of
    entities that happened to be short in the run it was recorded in. If
    MER-SG stops qualifying for delay_payable, the earlier rejection of
    MER-IE and MER-CA must still hold even though the consolidated action
    string for that kind is now different from REJECTED_ACTION."""
    conn, plan = _infeasible_plan(tmp_path)
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


@requires_llm
def test_override_is_per_entity_not_per_kind(tmp_path):
    """A rejection excludes only the entities named in the rejected remedy,
    not every entity that could ever qualify for that kind -- otherwise
    rejecting one course would silently blind the whole lever for entities
    the treasurer never objected to."""
    conn, plan = _infeasible_plan(tmp_path)
    first = diagnosis.diagnose(conn, plan)
    delay_first = next(r for r in first.remedies if r.kind == diagnosis.DELAY_PAYABLE)
    assert set(delay_first.entity_id.split(", ")) == {"MER-IE", "MER-SG", "MER-CA"}

    only_mer_ca = diagnosis.Remedy(
        action="Delay payables at MER-CA", kind=diagnosis.DELAY_PAYABLE,
        entity_id="MER-CA", amount_minor=1, currency="USD",
        business_cost="test", reversible=False, requires_signoff=None,
        rank=1, rationale="test",
    )
    diagnosis.record_override(conn, only_mer_ca, "vendor relationship must be preserved", "run-test")

    second = diagnosis.diagnose(conn, plan)
    delay_second = next(r for r in second.remedies if r.kind == diagnosis.DELAY_PAYABLE)
    entities = set(delay_second.entity_id.split(", "))
    assert "MER-CA" not in entities
    assert entities == {"MER-IE", "MER-SG"}


@requires_llm
def test_diagnose_does_not_crash_when_every_remedy_is_rejected(tmp_path):
    """Rejecting a consolidated course excludes every entity it names for
    that kind, so rejecting all three courses is enough to leave zero
    candidates -- a real outcome that must escalate cleanly, not crash on
    an empty remedies[0] recommendation fallback."""
    conn, plan = _infeasible_plan(tmp_path)
    first = diagnosis.diagnose(conn, plan)
    for remedy in first.remedies:
        diagnosis.record_override(conn, remedy, "rejecting every lever for this test", "run-test")

    second = diagnosis.diagnose(conn, plan)
    assert second.remedies == ()
    assert second.recommendation.strip()
    assert second.escalate_to.strip()


def test_diagnose_raises_on_feasible_plan(tmp_path):
    conn = _conn("base", "diag_base", tmp_path)
    plan = solver.solve(conn, "base")
    assert plan.status == "OPTIMAL"
    with pytest.raises(ValueError):
        diagnosis.diagnose(conn, plan)
