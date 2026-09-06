"""Regression tests for the devil's-advocate review findings that don't need
a live LLM call: FX determinism/rounding, currency-mismatch guards in
positions.py, the memo's infeasible-baseline savings guard, and diagnosis's
deterministic ranking/covenant-preference fixes."""

from __future__ import annotations

import sqlite3

import pytest

from . import diagnosis, memo, positions, solver
from .fx import FXTable
from .seed import seed


def _fx_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE fx_rate (base TEXT, quote TEXT, mid REAL, spread_bps INTEGER)"
    )
    conn.executemany("INSERT INTO fx_rate VALUES (?, ?, ?, ?)", rows)
    return conn


def test_cross_rate_is_deterministic_regardless_of_via_currency_order():
    # Two candidate `via` currencies (B and C) both bridge A -> D. Repeated
    # FXTable construction from the same rows must always pick the same one.
    rows = [
        ("A", "B", 2.0, 10),
        ("B", "D", 3.0, 10),
        ("A", "C", 5.0, 10),
        ("C", "D", 7.0, 10),
    ]
    quotes = {FXTable(_fx_conn(rows)).rate("A", "D").mid for _ in range(20)}
    assert len(quotes) == 1


def test_convert_rounds_instead_of_truncating():
    conn = _fx_conn([("A", "B", 1.0005, 0)])
    fx = FXTable(conn)
    # 1000 * 1.0005 = 1000.5 -> rounds to 1000 or 1001, never truncates to 1000
    # in a way that disagrees with round().
    assert fx.convert(1000, "A", "B") == round(1000 * 1.0005)


def test_convert_same_currency_is_identity():
    conn = _fx_conn([("A", "B", 1.0, 0)])
    fx = FXTable(conn)
    assert fx.convert(-500, "A", "A") == -500


def test_binding_floors_raises_on_covenant_currency_mismatch(tmp_path):
    conn = seed("base", db_path=tmp_path / "mismatch.db")
    conn.execute(
        "UPDATE covenant SET currency = 'GBP' WHERE entity_id = 'MER-DE'"
    )
    with pytest.raises(ValueError):
        positions.binding_floors(conn)


def test_opening_balances_raises_on_account_currency_mismatch(tmp_path):
    conn = seed("base", db_path=tmp_path / "mismatch2.db")
    conn.execute(
        "UPDATE bank_account SET currency = 'GBP' WHERE entity_id = 'MER-DE'"
    )
    with pytest.raises(ValueError):
        positions.opening_balances(conn)


def test_summarise_includes_entity_with_no_forecast_rows(tmp_path):
    conn = seed("base", db_path=tmp_path / "noforecast.db")
    conn.execute("DELETE FROM cash_forecast WHERE entity_id = 'MER-SG'")
    summary = positions.summarise(conn)
    assert "MER-SG" in summary
    assert summary["MER-SG"].currency == "USD"


def test_memo_facts_show_na_when_baseline_infeasible(tmp_path):
    conn = seed("base", db_path=tmp_path / "memo.db")
    plan = solver.Plan(
        status="OPTIMAL", scenario="base", transfers=(), total_cost_minor=500,
        solve_seconds=0.0, binding_constraints=(), closing_balances={},
    )
    infeasible_baseline = solver.Plan(
        status="INFEASIBLE", scenario="base", transfers=(), total_cost_minor=0,
        solve_seconds=0.0, binding_constraints=("x short 1.00 USD of its floor",),
        closing_balances={},
    )
    facts = memo._facts_for_memo(conn, plan, infeasible_baseline)
    assert facts["delta"] == "N/A"
    assert facts["delta_pct"] is None
    assert "infeasible" in facts["baseline_cost"].lower()


def test_memo_facts_compute_delta_when_baseline_feasible(tmp_path):
    conn = seed("base", db_path=tmp_path / "memo2.db")
    plan = solver.Plan(
        status="OPTIMAL", scenario="base", transfers=(), total_cost_minor=500,
        solve_seconds=0.0, binding_constraints=(), closing_balances={},
    )
    feasible_baseline = solver.Plan(
        status="OPTIMAL", scenario="base", transfers=(), total_cost_minor=1500,
        solve_seconds=0.0, binding_constraints=(), closing_balances={},
    )
    facts = memo._facts_for_memo(conn, plan, feasible_baseline)
    assert facts["delta_minor"] == 1000
    assert facts["delta_pct"] is not None


def test_entity_covenant_prefers_hard_over_soft(tmp_path):
    conn = seed("base", db_path=tmp_path / "covenant.db")
    # MER-SG normally has only a soft covenant; add a hard one too.
    conn.execute(
        "INSERT INTO covenant (entity_id, kind, threshold, currency, hardness) "
        "VALUES ('MER-SG', 'test_hard', 100000, 'USD', 'hard')"
    )
    covenant = diagnosis._entity_covenant(conn, "MER-SG")
    assert covenant["hardness"] == "hard"


def _remedy(kind, entity_id, amount_minor):
    return diagnosis.Remedy(
        action=f"{kind}:{entity_id}", kind=kind, entity_id=entity_id,
        amount_minor=amount_minor, currency="USD", business_cost="cost",
        reversible=True, requires_signoff=None, rank=0, rationale="r",
    )


def test_deterministic_order_respects_priority_classes():
    candidates = [
        _remedy(diagnosis.DELAY_PAYABLE, "E1", 100),
        _remedy(diagnosis.REVOLVER, "E2", 50),
        _remedy(diagnosis.SOFT_COVENANT, "E3", 75),
    ]
    order = diagnosis._deterministic_order(candidates)
    kinds = [candidates[i].kind for i in order]
    assert kinds == [diagnosis.REVOLVER, diagnosis.SOFT_COVENANT, diagnosis.DELAY_PAYABLE]


def test_apply_llm_result_rejects_ranking_that_violates_priority_classes():
    candidates = [
        _remedy(diagnosis.REVOLVER, "E1", 100),
        _remedy(diagnosis.DELAY_PAYABLE, "E2", 50),
    ]
    # Puts the delay-payable ahead of the revolver draw -- violates the
    # "revolver before soft covenant before delayed payable" guarantee.
    bad_ranking = '{"ranked_ids": [1, 0]}'
    remedies, _ = diagnosis._apply_llm_result(candidates, bad_ranking)
    assert remedies[0].kind == diagnosis.REVOLVER


def test_apply_llm_result_handles_duplicate_candidates_without_dropping_one():
    # Two remedies that compare equal by value -- exercises the .index()
    # ambiguity that used to silently drop one of them.
    candidates = [
        _remedy(diagnosis.REVOLVER, "E1", 100),
        _remedy(diagnosis.REVOLVER, "E1", 100),
    ]
    remedies, _ = diagnosis._apply_llm_result(candidates, "not json")
    assert len(remedies) == 2
    assert {r.rank for r in remedies} == {1, 2}
