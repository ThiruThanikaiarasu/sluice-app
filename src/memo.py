"""The approval memo: a one-page document a CFO signs.

`write_memo` renders the OPTIMAL branch: what moves, what it costs, what
cheaper-but-illegal route was rejected, and which floors have the least
headroom. `write_escalation` renders the INFEASIBLE branch and consumes
Worker A's `Diagnosis`.

Every number in either document comes from `plan`, `baseline`, or the
database. The LLM is asked to write prose around figures we computed; if it
emits a number of its own, we discard it rather than trust it.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import positions
from .llm import complete
from .models import format_money, to_major
from .solver import Plan, _ic_agreements, repayment_for_transfer

MEMO_MAX_TOKENS = 8192
ESCALATION_MAX_TOKENS = 8192

_fmt_money = format_money


def _fmt_signed_money(amount_minor: int, currency: str) -> str:
    major = to_major(amount_minor)
    sign = "-" if major < 0 else ""
    return f"{sign}{currency} {abs(major):,.2f}"


def _entity_names(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM entity")}


def _rejected_routes(conn: sqlite3.Connection) -> list[dict]:
    """Prohibited intercompany pairs between an entity with spare cash and an
    entity that actually needed funding this horizon -- the cheap-but-illegal
    alternative a route optimiser would have picked were it legal."""
    ic = _ic_agreements(conn)
    have_cash = set(positions.lenders(conn))
    need_cash = set(positions.shortfalls(conn))
    rejected = []
    for (lender, borrower), agreement in ic.items():
        if agreement["permitted"]:
            continue
        if lender in have_cash and borrower in need_cash:
            rejected.append({
                "lender": lender,
                "borrower": borrower,
                "reason": agreement["reason"],
            })
    return rejected


def _headroom(conn: sqlite3.Connection, plan: Plan) -> list[dict]:
    """Per-entity closing balance vs. floor at the plan's worst day for that
    entity -- how close each floor came to binding, tightest first."""
    floors = positions.binding_floors(conn)
    names = _entity_names(conn)
    rows = []
    for entity_id, floor in floors.items():
        days = plan.closing_balances.get(entity_id, {})
        if not days:
            continue
        worst_day, worst_balance = min(days.items(), key=lambda kv: kv[1])
        headroom_minor = worst_balance - floor
        rows.append({
            "entity_id": entity_id,
            "entity_name": names.get(entity_id, entity_id),
            "floor": floor,
            "worst_day": worst_day,
            "worst_balance": worst_balance,
            "headroom_minor": headroom_minor,
        })
    rows.sort(key=lambda r: r["headroom_minor"])
    return rows


def _ic_outstanding(conn: sqlite3.Connection, plan: Plan) -> list[dict]:
    """Amount lent per (lender, borrower) pair by this plan that is still
    outstanding at the end of the horizon -- loans this plan also scheduled
    a repayment for (because their maturity fit inside the horizon) are
    excluded here and appear in `_ic_repaid` instead.

    A loan that never matures within the horizon has interest charged
    through the horizon's last day as if it stays open that whole time,
    and nothing schedules the money back -- that is a real intercompany
    receivable/payable a CFO needs to see named, not a number silently
    absorbed into "the plan cost X".
    """
    names = _entity_names(conn)
    currencies = {r["id"]: r["functional_currency"] for r in conn.execute(
        "SELECT id, functional_currency FROM entity"
    )}
    unclaimed = list(plan.repayments)
    totals: dict[tuple[str, str], int] = {}
    for t in plan.transfers:
        match = repayment_for_transfer(t, tuple(unclaimed))
        if match is not None:
            unclaimed.remove(match)
            continue
        key = (t.from_entity, t.to_entity)
        totals[key] = totals.get(key, 0) + t.amount_minor
    rows = []
    for (lender, borrower), amount in sorted(totals.items()):
        rows.append({
            "lender_id": lender,
            "lender_name": names.get(lender, lender),
            "borrower_id": borrower,
            "borrower_name": names.get(borrower, borrower),
            "amount": _fmt_money(amount, currencies.get(lender, "USD")),
        })
    return rows


def _ic_repaid(conn: sqlite3.Connection, plan: Plan) -> list[dict]:
    """Loans this plan both drew and repaid within the horizon -- the
    other half of the intercompany picture from `_ic_outstanding`."""
    names = _entity_names(conn)
    rows = []
    for r in sorted(plan.repayments, key=lambda r: (r.pay_day, r.lender_id, r.borrower_id)):
        rows.append({
            "lender_id": r.lender_id,
            "lender_name": names.get(r.lender_id, r.lender_id),
            "borrower_id": r.borrower_id,
            "borrower_name": names.get(r.borrower_id, r.borrower_id),
            "pay_day": r.pay_day,
            "principal": _fmt_money(r.principal_minor, r.currency),
            "interest": _fmt_money(r.interest_minor, r.currency),
            "total": _fmt_money(r.total_minor, r.currency),
        })
    return rows


def _cost_breakdown(plan: Plan) -> dict[str, int]:
    return {
        "fx_cost_minor": sum(t.fx_cost_minor for t in plan.transfers),
        "fee_minor": sum(t.fee_minor for t in plan.transfers),
        "interest_minor": sum(t.interest_minor for t in plan.transfers),
        "total_minor": plan.total_cost_minor,
    }


def _facts_for_memo(conn: sqlite3.Connection, plan: Plan, baseline: Plan) -> dict[str, Any]:
    names = _entity_names(conn)
    currencies = {r["id"]: r["functional_currency"] for r in conn.execute(
        "SELECT id, functional_currency FROM entity"
    )}

    transfers = [
        {
            "from_entity": t.from_entity,
            "from_name": names.get(t.from_entity, t.from_entity),
            "to_entity": t.to_entity,
            "to_name": names.get(t.to_entity, t.to_entity),
            "amount": _fmt_money(t.amount_minor, t.amount_currency),
            "landed": _fmt_money(t.landed_minor, t.landed_currency),
            "send_day": t.send_day,
            "land_day": t.land_day,
            "rationale": t.rationale,
        }
        for t in plan.transfers
    ]

    cost = _cost_breakdown(plan)
    # An INFEASIBLE baseline reports total_cost_minor == 0 (see solver.solve),
    # not "free" -- computing a delta against it would print a savings figure
    # against a plan that was never actually costed.
    if baseline.feasible:
        delta_minor = baseline.total_cost_minor - plan.total_cost_minor
        pct = (delta_minor / baseline.total_cost_minor * 100) if baseline.total_cost_minor else 0.0
        baseline_cost = _fmt_money(baseline.total_cost_minor, "USD")
        delta = _fmt_signed_money(delta_minor, "USD")
        delta_pct = round(pct, 1)
    else:
        delta_minor = None
        baseline_cost = "N/A (naive baseline is itself infeasible for this scenario)"
        delta = "N/A"
        delta_pct = None

    return {
        "transfers": transfers,
        "cost": {k: _fmt_money(v, "USD") for k, v in cost.items()},
        "baseline_cost": baseline_cost,
        "delta_minor": delta_minor,
        "delta": delta,
        "delta_pct": delta_pct,
        "rejected_routes": _rejected_routes(conn),
        "headroom": [
            {
                "entity_id": r["entity_id"],
                "entity_name": r["entity_name"],
                "floor": _fmt_money(r["floor"], currencies.get(r["entity_id"], "USD")),
                "worst_day": r["worst_day"],
                "headroom": _fmt_money(r["headroom_minor"], currencies.get(r["entity_id"], "USD")),
            }
            for r in _headroom(conn, plan)
        ],
        "binding_constraints": list(plan.binding_constraints),
        "ic_outstanding": _ic_outstanding(conn, plan),
        "ic_repaid": _ic_repaid(conn, plan),
    }


_MEMO_SYSTEM = """You write one-page treasury approval memos for a CFO. You \
are given exact, pre-computed figures as JSON -- transfers, costs, a cost \
delta against a baseline, rejected routes with their legal reasons, and \
covenant headroom. Write markdown prose around these figures. Do not invent, \
recompute, or alter any number: copy the figures exactly as given, including \
currency codes and signs. If a figure is not in the JSON, do not state it. \
Write for a reader who already understands cash positioning -- explain the \
decision, not the input data. Use the section headings given in the prompt. \
Quote rejection reasons verbatim; do not paraphrase them."""


def _memo_prompt(facts: dict[str, Any]) -> str:
    transfer_lines = "\n".join(
        f"- {t['from_name']} ({t['from_entity']}) -> {t['to_name']} ({t['to_entity']}): "
        f"sends {t['amount']} on day {t['send_day']}, lands {t['landed']} on day {t['land_day']}. "
        f"{t['rationale']}"
        for t in facts["transfers"]
    ) or "- No transfers required this horizon."

    rejected_lines = "\n".join(
        f"- {r['lender']} -> {r['borrower']}: {r['reason']}"
        for r in facts["rejected_routes"]
    ) or "- None: no prohibited route touched an entity in this plan."

    headroom_lines = "\n".join(
        f"- {h['entity_name']} ({h['entity_id']}): floor {h['floor']}, worst day {h['worst_day']}, "
        f"headroom {h['headroom']}"
        for h in facts["headroom"]
    )

    binding_lines = "\n".join(f"- {b}" for b in facts["binding_constraints"]) or "- None."

    ic_lines = "\n".join(
        f"- {r['lender_name']} ({r['lender_id']}) is owed {r['amount']} by "
        f"{r['borrower_name']} ({r['borrower_id']}), no repayment date modelled for it"
        for r in facts["ic_outstanding"]
    ) or "- None: every loan this plan drew is either repaid within the horizon or none was drawn."

    ic_repaid_lines = "\n".join(
        f"- {r['borrower_name']} ({r['borrower_id']}) repays {r['lender_name']} "
        f"({r['lender_id']}) on day {r['pay_day']}: principal {r['principal']} + "
        f"interest {r['interest']} = {r['total']}"
        for r in facts["ic_repaid"]
    ) or "- None: no loan this plan drew matures within the horizon."

    return f"""Write the approval memo with exactly these sections, in this \
order: "## Decision", "## What moves", "## What it costs", \
"## What was rejected and why", "## Constraints that came close to binding", \
"## Intercompany balances outstanding at day 14".

Facts (all numbers final, copy exactly):

DECISION: approve the transfer plan below for this horizon.

TRANSFERS:
{transfer_lines}

COST:
- FX spread: {facts['cost']['fx_cost_minor']}
- Wire fees: {facts['cost']['fee_minor']}
- Intercompany interest: {facts['cost']['interest_minor']}
- Total: {facts['cost']['total_minor']}
- Naive fund-from-HQ baseline total: {facts['baseline_cost']}
- Savings versus baseline: {facts['delta']}{f" ({facts['delta_pct']}%)" if facts['delta_pct'] is not None else ""}

REJECTED ROUTES (cheaper but illegal or prohibited):
{rejected_lines}

BINDING CONSTRAINTS THIS PLAN SATISFIES EXACTLY (its tightest points):
{binding_lines}

HEADROOM AGAINST EACH ENTITY'S FLOOR, TIGHTEST FIRST:
{headroom_lines}

LOANS THIS PLAN DREW AND ALSO REPAYS WITHIN THE HORIZON:
{ic_repaid_lines}

LOANS THIS PLAN DREW THAT REMAIN OUTSTANDING AT DAY 14 (mature beyond what \
the horizon and settlement calendar allow to repay in time):
{ic_lines}

For "## What moves" render a markdown table with columns: From, To, Amount, \
Currency, Send day, Land day. For "## What it costs" state the four cost \
lines then the savings-versus-baseline line as its own sentence, verbatim \
numbers. For "## What was rejected and why" explain each rejected route \
using its quoted reason; if there are none, say so in one sentence. For \
"## Constraints that came close to binding" list the tightest 1-3 floors and \
what happens if forecast flows worsen slightly. For "## Intercompany \
balances outstanding at day 14" first list any loans that repay within the \
horizon with their pay day and amounts, then separately list any that \
remain outstanding beyond day 14 with no repayment date modelled -- keep \
the two groups distinct and do not invent a repayment date for the second \
group."""


def write_memo(conn: sqlite3.Connection, plan: Plan, baseline: Plan) -> str:
    if plan.status != "OPTIMAL":
        raise ValueError(f"write_memo requires an OPTIMAL plan, got {plan.status!r}")

    facts = _facts_for_memo(conn, plan, baseline)
    return complete(_MEMO_SYSTEM, _memo_prompt(facts), max_tokens=MEMO_MAX_TOKENS)


# --------------------------------------------------------------------------
# Escalation (INFEASIBLE branch)
# --------------------------------------------------------------------------

_ESCALATION_SYSTEM = """You write treasury escalation documents for a CFO \
when no legal transfer plan can cover the group's cash shortfall. You are \
given exact, pre-computed figures and a ranked remedy list as JSON. Write \
markdown prose around these figures. Do not invent, recompute, or alter any \
number, amount, or entity name: copy them exactly as given. Never suggest, \
imply, or rank a remedy that breaches a hard covenant -- if one was excluded, \
say so and explain why it is off the table. Write for a reader who already \
understands cash positioning."""


def _facts_for_escalation(diagnosis: Any) -> dict[str, Any]:
    remedies = sorted(diagnosis.remedies, key=lambda r: r.rank)
    return {
        "binding_constraint": diagnosis.binding_constraint,
        "explanation": diagnosis.explanation,
        "shortfall": _fmt_money(diagnosis.shortfall_minor, "USD"),
        "remedies": [
            {
                "rank": r.rank,
                "action": r.action,
                "kind": r.kind,
                "entity_id": r.entity_id,
                "amount": _fmt_money(r.amount_minor, r.currency),
                "business_cost": r.business_cost,
                "reversible": r.reversible,
                "requires_signoff": r.requires_signoff,
                "rationale": r.rationale,
            }
            for r in remedies
        ],
        "recommendation": diagnosis.recommendation,
        "escalate_to": diagnosis.escalate_to,
    }


def _escalation_prompt(facts: dict[str, Any]) -> str:
    remedy_lines = "\n".join(
        f"{r['rank']}. {r['action']} ({r['kind']}, {r['entity_id']}, {r['amount']}). "
        f"Business cost: {r['business_cost']}. Reversible: {r['reversible']}. "
        f"Requires sign-off: {r['requires_signoff'] or 'none'}. "
        f"Rationale: {r['rationale']}"
        for r in facts["remedies"]
    ) or "No remedies available."

    return f"""Write the escalation document with exactly these sections, in \
this order: "## Decision", "## Why this cannot be solved by transfer alone", \
"## Ranked remedies", "## Recommendation", "## Escalate to".

Facts (all numbers and names final, copy exactly):

DECISION: no legal set of transfers covers the group's cash shortfall this \
horizon; a remedy outside ordinary transfers is required.

BINDING CONSTRAINT: {facts['binding_constraint']}

EXPLANATION: {facts['explanation']}

SHORTFALL THAT CANNOT BE COVERED BY TRANSFER: {facts['shortfall']}

RANKED REMEDIES (best first; a hard covenant breach must never appear here \
and none does):
{remedy_lines}

RECOMMENDATION: {facts['recommendation']}

ESCALATE TO: {facts['escalate_to']}

For "## Ranked remedies" render a markdown table with columns: Rank, Action, \
Amount, Business cost, Reversible, Sign-off. For "## Why this cannot be \
solved by transfer alone" use the binding constraint and explanation \
verbatim in meaning, in your own words for readability, but do not change \
the shortfall figure or entity names. For "## Recommendation" state the \
recommended remedy and one sentence on why it outranks the others. For \
"## Escalate to" name who must decide and by when (this horizon)."""


def write_escalation(diagnosis: Any) -> str:
    facts = _facts_for_escalation(diagnosis)
    return complete(_ESCALATION_SYSTEM, _escalation_prompt(facts), max_tokens=ESCALATION_MAX_TOKENS)
