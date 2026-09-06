"""Infeasibility diagnosis and ranked remedies.

`solver.solve()` returns a `Plan` with `status == "INFEASIBLE"` when no lawful
set of transfers covers every entity's floor. That is not a dead end -- it is
the moment a solver has no model for and a human needs a ranked, explained set
of options. This module builds that: which constraint actually binds, what it
costs to fix, and who has to sign off.

Design choice that matters: the remedy *candidates* (kind, entity, amount) and
the priority class they fall into (revolver < soft covenant < delay payable)
are computed in Python, not asked of the model. A hard covenant breach is
never constructed as a candidate in the first place, so no LLM formatting
quirk or prompt-injection-shaped constraint text can smuggle one into
`remedies` -- the "never" rule in the brief is a structural guarantee, not a
prompt instruction. The model only ranks within the pre-filtered candidate set
and writes the prose (business_cost detail, explanation, recommendation,
escalate_to). If its ranking or prose is missing or malformed we fall back to
the deterministic order and a rule-based business_cost -- `complete()` still
raises on a truly empty/truncated response, we do not paper over that.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .fx import FXTable
from .llm import complete
from .models import HARD, SOFT, to_major, to_minor
from .positions import summarise
from .solver import Plan

# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

REVOLVER = "revolver"
SOFT_COVENANT = "soft_covenant"
DELAY_PAYABLE = "delay_payable"

_KIND_PRIORITY = {REVOLVER: 0, SOFT_COVENANT: 1, DELAY_PAYABLE: 2}


@dataclass(frozen=True)
class Remedy:
    action: str
    kind: str
    entity_id: str
    amount_minor: int
    currency: str
    business_cost: str
    reversible: bool
    requires_signoff: str | None
    rank: int
    rationale: str


@dataclass(frozen=True)
class Diagnosis:
    binding_constraint: str
    explanation: str
    shortfall_minor: int
    remedies: tuple[Remedy, ...]
    recommendation: str
    escalate_to: str


# --------------------------------------------------------------------------
# Parsing the solver's binding_constraints into computed numbers
# --------------------------------------------------------------------------

_FLOOR_SHORT_RE = re.compile(
    r"^(?P<entity>\S+) short (?P<amount>[\d,]+(?:\.\d+)?) (?P<currency>[A-Z]{3}) "
    r"of its [\d,]+(?:\.\d+)? [A-Z]{3} floor on day (?P<day>\d+)"
)


@dataclass(frozen=True)
class Shortfall:
    entity_id: str
    currency: str
    amount_minor: int   # what could not be covered, entity's own currency
    day: int
    line: str            # the original binding_constraints string, verbatim


def _parse_shortfalls(plan: Plan) -> list[Shortfall]:
    """Pull the entity-level floor shortfalls out of `plan.binding_constraints`.

    These strings are the solver's elastic-relaxation diagnosis -- the
    authoritative source for "what is short and by how much". We do not
    recompute this independently; that would risk a second number that
    disagrees with the one the solver already proved.
    """
    out = []
    for line in plan.binding_constraints:
        m = _FLOOR_SHORT_RE.match(line)
        if not m:
            continue
        amount_major = m.group("amount").replace(",", "")
        out.append(Shortfall(
            entity_id=m.group("entity"),
            currency=m.group("currency"),
            amount_minor=to_minor(amount_major),
            day=int(m.group("day")),
            line=line,
        ))
    return out


def _entity_covenant(conn: sqlite3.Connection, entity_id: str) -> dict | None:
    row = conn.execute(
        "SELECT entity_id, kind, threshold, currency, hardness, source_doc, "
        "source_quote FROM covenant WHERE entity_id = ?",
        (entity_id,),
    ).fetchone()
    return dict(row) if row else None


def _payable_candidate(conn: sqlite3.Connection, entity_id: str) -> dict | None:
    """A negative one-off cash_forecast entry for this entity that reads as a
    genuine, delayable commercial payable -- not a tax, statutory or payroll
    obligation, which no responsible treasurer would recommend delaying."""
    rows = conn.execute(
        "SELECT day, net_flow, note FROM cash_forecast "
        "WHERE entity_id = ? AND note IS NOT NULL ORDER BY day",
        (entity_id,),
    ).fetchall()
    blocked = ("vat", "tax", "payroll")
    for r in rows:
        note = r["note"] or ""
        if r["net_flow"] >= 0:
            continue
        if any(b in note.lower() for b in blocked):
            continue
        return {"day": r["day"], "amount_minor": -r["net_flow"], "note": note}
    return None


# --------------------------------------------------------------------------
# Candidate remedy construction (deterministic -- no LLM in this path)
#
# Each shortfall entity can independently qualify for up to three remedy
# kinds. Emitting one Remedy per (entity, kind) is an enumeration, not a
# judgement -- a treasurer reading a dozen line items gets no more help than
# a spreadsheet. Instead we consolidate by kind: one Remedy per kind, naming
# every entity it covers, with amounts rolled up to USD so a multi-currency
# course of action is still a single comparable number. This still produces
# at most three or four candidates (revolver / soft-covenant / delay-payable,
# optionally a combination), never per-entity duplicates.
# --------------------------------------------------------------------------

def _build_revolver_remedy(conn: sqlite3.Connection, shortfalls: list[Shortfall]) -> Remedy | None:
    """Always offered, cheap and reversible, never touches the covenant
    itself -- so every shortfall entity qualifies."""
    if not shortfalls:
        return None

    per_entity = []
    total_usd_minor = 0
    for sf in shortfalls:
        covenant = _entity_covenant(conn, sf.entity_id)
        facility = "committed revolving facility"
        if covenant and covenant["kind"] == "overdraft_facility_minimum":
            facility = "group revolving credit facility (not the overdraft " \
                       "line the covenant itself measures)"
        per_entity.append(
            f"{sf.entity_id} draws {to_major(sf.amount_minor)} {sf.currency} "
            f"on its {facility}"
        )
        total_usd_minor += _usd_minor(conn, sf.amount_minor, sf.currency)

    entity_id = ", ".join(sf.entity_id for sf in shortfalls)
    lines = "; ".join(sf.line for sf in shortfalls)
    return Remedy(
        action=f"Draw on revolving facilities at {entity_id}",
        kind=REVOLVER,
        entity_id=entity_id,
        amount_minor=total_usd_minor,
        currency="USD",
        business_cost=(
            "Facility interest while drawn; no vendor or covenant "
            f"relationship affected. {'; '.join(per_entity)}."
        ),
        reversible=True,
        requires_signoff=None,
        rank=0,
        rationale=(
            "Cheapest and fully reversible option; covers every shortfall "
            f"in full. Against: {lines}"
        ),
    )


def _build_soft_covenant_remedy(conn: sqlite3.Connection, shortfalls: list[Shortfall]) -> Remedy | None:
    """Only entities whose OWN floor is soft qualify. Never constructed for a
    hard covenant -- this is the structural guarantee that a hard breach can
    never appear in `remedies`, enforced here rather than left to the model."""
    entries = []
    for sf in shortfalls:
        covenant = _entity_covenant(conn, sf.entity_id)
        if covenant and covenant["hardness"] == SOFT:
            entries.append((sf, covenant))
    if not entries:
        return None

    per_entity = [f"{sf.entity_id} ({covenant['source_doc']})" for sf, covenant in entries]
    entity_id = ", ".join(sf.entity_id for sf, _ in entries)
    total_usd_minor = sum(_usd_minor(conn, sf.amount_minor, sf.currency) for sf, _ in entries)
    lines = "; ".join(sf.line for sf, _ in entries)
    return Remedy(
        action=f"Accept soft covenant breaches at {entity_id}",
        kind=SOFT_COVENANT,
        entity_id=entity_id,
        amount_minor=total_usd_minor,
        currency="USD",
        business_cost=(
            f"Zero cash cost. Breaches internal policy at {', '.join(per_entity)}, "
            "requires Group Treasurer sign-off, and is recoverable as soon "
            "as cash recovers."
        ),
        reversible=True,
        requires_signoff="Group Treasurer",
        rank=0,
        rationale=(
            "These entities' floors are internal policy (soft), not "
            f"contractual. Against: {lines}"
        ),
    )


def _build_delay_payable_remedy(conn: sqlite3.Connection, shortfalls: list[Shortfall]) -> Remedy | None:
    """Only entities with a genuine, non-statutory payable to delay qualify."""
    entries = []
    for sf in shortfalls:
        payable = _payable_candidate(conn, sf.entity_id)
        if payable is not None:
            amount = min(sf.amount_minor, payable["amount_minor"])
            entries.append((sf, payable, amount))
    if not entries:
        return None

    per_entity = [
        f"\"{payable['note']}\" at {sf.entity_id} (day {payable['day']})"
        for sf, payable, _ in entries
    ]
    entity_id = ", ".join(sf.entity_id for sf, _, _ in entries)
    total_usd_minor = sum(_usd_minor(conn, amount, sf.currency) for sf, _, amount in entries)
    lines = "; ".join(sf.line for sf, _, _ in entries)
    return Remedy(
        action=f"Delay payables at {entity_id}",
        kind=DELAY_PAYABLE,
        entity_id=entity_id,
        amount_minor=total_usd_minor,
        currency="USD",
        business_cost=(
            f"Frees cash by holding back {', '.join(per_entity)}. Damages "
            "each counterparty relationship and is not recovered by paying "
            "later."
        ),
        reversible=False,
        requires_signoff=None,
        rank=0,
        rationale=f"Only non-statutory payables available. Against: {lines}",
    )


def _build_candidates(conn: sqlite3.Connection, shortfalls: list[Shortfall]) -> list[Remedy]:
    candidates = [
        _build_revolver_remedy(conn, shortfalls),
        _build_soft_covenant_remedy(conn, shortfalls),
        _build_delay_payable_remedy(conn, shortfalls),
    ]
    return [c for c in candidates if c is not None]


def _deterministic_order(candidates: list[Remedy]) -> list[Remedy]:
    return sorted(
        candidates,
        key=lambda r: (_KIND_PRIORITY[r.kind], -r.amount_minor, r.entity_id),
    )


# --------------------------------------------------------------------------
# LLM: rank within the pre-filtered set, write the prose
# --------------------------------------------------------------------------

def _usd_minor(conn: sqlite3.Connection, amount_minor: int, currency: str) -> int:
    fx = FXTable(conn)
    return fx.convert(amount_minor, currency, "USD")


def _primary_binding_constraint(shortfalls: list[Shortfall]) -> Shortfall:
    """The single most severe shortfall, in USD terms, drives the headline
    binding_constraint string -- 'most severe' is decided here in Python from
    real numbers, not asked of the model."""
    return shortfalls[0]


def _llm_prompt(conn: sqlite3.Connection, plan: Plan, shortfalls: list[Shortfall],
                 candidates: list[Remedy]) -> tuple[str, str]:
    summary = summarise(conn)
    entities_in_play = {sf.entity_id for sf in shortfalls}

    covenants = []
    for eid in sorted(entities_in_play):
        c = _entity_covenant(conn, eid)
        if c:
            covenants.append({
                "entity_id": eid,
                "hardness": c["hardness"],
                "threshold_major": str(to_major(c["threshold"])),
                "currency": c["currency"],
                "source_quote": c["source_quote"],
            })

    positions_payload = [
        {
            "entity_id": eid,
            "currency": s.currency,
            "floor_major": str(to_major(s.floor)),
            "min_balance_major": str(to_major(s.min_balance)),
            "worst_day": s.worst_day,
        }
        for eid, s in summary.items() if eid in entities_in_play
    ]

    candidates_payload = [
        {
            "id": i,
            "action": c.action,
            "kind": c.kind,
            "entity_id": c.entity_id,
            "amount_major": str(to_major(c.amount_minor)),
            "currency": c.currency,
        }
        for i, c in enumerate(candidates)
    ]

    system = (
        "You are a treasury remedy analyst. The solver has already proven the "
        "plan is infeasible and computed exactly which entities are short and "
        "by how much, and has already priced each remedy candidate -- you do "
        "not question, recompute, or restate those numbers. Your job: pick "
        "the recommended order to work through the given remedy candidates "
        "(already screened -- none of them breach a hard covenant), and "
        "write CFO-readable prose. Respond with JSON only, no markdown "
        "fences, matching this shape exactly:\n"
        '{"binding_constraint": str, "explanation": str, "recommendation": str, '
        '"escalate_to": str, "ranked_ids": [int, ...]}\n'
        "binding_constraint must be copied verbatim, character for character, "
        "from one entry of binding_constraints -- pick the single most severe "
        "one. Do not paraphrase or combine entries into a new sentence. "
        "ranked_ids must be a permutation of every candidate id, best first: "
        "revolver draws before soft-covenant breaches before delayed "
        "payables. Each candidate is already a consolidated course of "
        "action covering every entity it names -- do not ask for it to be "
        "split back out by entity. explanation is 2-4 sentences. If any entity in play has "
        "a hard covenant, say explicitly in explanation that breaching it "
        "was never considered and why."
    )
    user = json.dumps({
        "positions": positions_payload,
        "binding_constraints": list(plan.binding_constraints),
        "covenants": covenants,
        "remedy_candidates": candidates_payload,
    })
    return system, user


def _apply_llm_result(candidates: list[Remedy], raw: str) -> tuple[list[Remedy], dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _deterministic_order(candidates), {}

    n = len(candidates)

    ranked_ids = data.get("ranked_ids")
    if (isinstance(ranked_ids, list) and len(ranked_ids) == n
            and set(ranked_ids) == set(range(n))):
        order = ranked_ids
    else:
        order = [candidates.index(c) for c in _deterministic_order(candidates)]

    final = []
    for rank, idx in enumerate(order, start=1):
        c = candidates[idx]
        final.append(Remedy(
            action=c.action, kind=c.kind, entity_id=c.entity_id,
            amount_minor=c.amount_minor, currency=c.currency,
            business_cost=c.business_cost, reversible=c.reversible,
            requires_signoff=c.requires_signoff, rank=rank,
            rationale=c.rationale,
        ))
    return final, data


# --------------------------------------------------------------------------
# Learned rules -- the override loop
# --------------------------------------------------------------------------

def record_override(conn: sqlite3.Connection, remedy: Remedy, reason: str, run_id: str) -> None:
    constraint = {
        "kind": remedy.kind,
        "entity_id": remedy.entity_id,
        "action": remedy.action,
    }
    conn.execute(
        "INSERT INTO learned_rule (origin_run_id, rule_text, constraint_json, "
        "created_at) VALUES (?, ?, ?, ?)",
        (
            run_id,
            f"Do not recommend '{remedy.action}' for {remedy.entity_id} "
            f"({remedy.kind}). Rejected: {reason}",
            json.dumps(constraint),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def learned_rules(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, origin_run_id, rule_text, constraint_json, created_at "
        "FROM learned_rule ORDER BY created_at"
    ).fetchall()
    return [dict(r) for r in rows]


def _excluded_by_learned_rules(candidate: Remedy, rules: list[dict]) -> bool:
    for rule in rules:
        try:
            constraint = json.loads(rule["constraint_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if (constraint.get("kind") == candidate.kind
                and constraint.get("entity_id") == candidate.entity_id
                and constraint.get("action") == candidate.action):
            return True
    return False


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def diagnose(conn: sqlite3.Connection, plan: Plan) -> Diagnosis:
    if plan.status != "INFEASIBLE":
        raise ValueError(f"diagnose() called on a {plan.status} plan; nothing to diagnose")

    shortfalls = _parse_shortfalls(plan)
    if not shortfalls:
        raise ValueError(
            "no parseable floor shortfalls in plan.binding_constraints -- "
            "cannot diagnose without at least one"
        )
    shortfalls.sort(
        key=lambda s: _usd_minor(conn, s.amount_minor, s.currency), reverse=True
    )

    candidates = _build_candidates(conn, shortfalls)
    rules = learned_rules(conn)
    candidates = [c for c in candidates if not _excluded_by_learned_rules(c, rules)]

    total_shortfall_minor = sum(
        _usd_minor(conn, s.amount_minor, s.currency) for s in shortfalls
    )

    system, user = _llm_prompt(conn, plan, shortfalls, candidates)
    raw = complete(system, user, max_tokens=8192)
    remedies, data = _apply_llm_result(candidates, raw)

    primary = _primary_binding_constraint(shortfalls)
    hard_entities = [
        s.entity_id for s in shortfalls
        if (c := _entity_covenant(conn, s.entity_id)) and c["hardness"] == HARD
    ]

    # Only ever accept the model's binding_constraint if it is a verbatim copy
    # of one of the solver's own strings -- "quote the actual string" is not
    # negotiable, and a paraphrase/summary is not a quote.
    binding_constraint = data.get("binding_constraint")
    if binding_constraint not in plan.binding_constraints:
        binding_constraint = primary.line

    explanation = data.get("explanation") if isinstance(data.get("explanation"), str) else None
    if not explanation or not explanation.strip():
        hard_note = (
            f" {', '.join(hard_entities)} carries a hard, contractual floor "
            "and no remedy here proposes breaching it."
            if hard_entities else ""
        )
        explanation = (
            f"{primary.entity_id} is short {to_major(primary.amount_minor)} "
            f"{primary.currency} against its floor even after routing all "
            f"available, permitted intercompany capacity ({len(shortfalls)} "
            f"entities are short in total).{hard_note}"
        )

    recommendation = data.get("recommendation") if isinstance(data.get("recommendation"), str) else None
    if not recommendation or not recommendation.strip():
        top = remedies[0]
        recommendation = (
            f"Start with: {top.action} ({to_major(top.amount_minor)} "
            f"{top.currency}) -- {top.business_cost}"
        )

    escalate_to = data.get("escalate_to") if isinstance(data.get("escalate_to"), str) else None
    if not escalate_to or not escalate_to.strip():
        escalate_to = "Group Treasurer" if any(r.requires_signoff for r in remedies) else "Treasury Director"

    return Diagnosis(
        binding_constraint=binding_constraint.strip(),
        explanation=explanation.strip(),
        shortfall_minor=total_shortfall_minor,
        remedies=tuple(remedies),
        recommendation=recommendation.strip(),
        escalate_to=escalate_to.strip(),
    )
