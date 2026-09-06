"""Minimum-cost flow solver for multi-entity cash positioning.

Decision variables are transfer amounts between entities' operating bank
accounts, on each day of the horizon. The objective minimises FX spread, wire
fees and intercompany interest, all normalised to USD minor units so costs are
comparable across scenarios. Constraints: closing balance >= binding floor per
entity per day; cumulative intercompany exposure <= ic_agreement.max_limit per
lender/borrower pair; no flow across `permitted = 0` pairs; settlement lag
(sent day d lands day d + settlement_days); no negative balances.

When the model is infeasible we solve a second, "elastic" version of the same
LP where floor and intercompany-limit constraints are relaxed by a slack
variable carrying a heavy penalty. That elastic solve is always feasible (the
slacks can absorb any shortfall), and minimising it isolates exactly which
constraints could not be satisfied and by how much -- the diagnosis Worker A
consumes, not a solver internals dump.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import pulp

from .db import horizon_start
from .fx import FXTable
from .models import (
    HORIZON_DAYS,
    is_settlement_day,
    next_settlement_day,
    repayment_owed_minor,
    to_major,
)
from .positions import binding_floors, opening_balances

# Safe upper bound on any single transfer leg, in minor units of the sending
# currency. Generous relative to the seeded company's balances (low millions).
BIG_M = 2_000_000_000

# Weight that forces the elastic solve to eliminate shortfall before it cares
# about real transfer cost at all.
PENALTY = 10**9

# Target headroom above each covenant/policy floor, and the small notional
# cost charged once per entity (not once per entity-day -- a single shared
# slack variable covers that entity's worst day across the whole horizon,
# and only that one charge enters the objective) when a plan doesn't reach
# it anywhere. At BUFFER_PENALTY_BPS/10_000 = 0.05%, this is deliberately
# the same order of magnitude as a single leg's one-time FX spread charge
# (8-12bps seeded); charging it per day instead would compound to ~10x a
# real leg's spread over a 14-day horizon and stop being a tie-break -- the
# solver would pay genuine FX/fee cost chasing a notional buffer target.
# Soft either way: a plan can still park exactly on the floor when cash is
# genuinely too scarce to do better, but it now costs something in the
# objective rather than being free, so the solver stops choosing zero
# headroom by indifference.
BUFFER_BPS = 500
BUFFER_PENALTY_BPS = 5


@dataclass(frozen=True)
class FloorShortfall:
    """One entity's worst unmet floor, from the elastic diagnosis solve.

    Carries both the structured numbers and the human-readable line so
    downstream consumers (diagnosis.py) never have to re-derive one from the
    other via string parsing.
    """
    entity_id: str
    currency: str
    amount_minor: int
    day: int
    line: str


@dataclass(frozen=True)
class ICShortfall:
    lender_id: str
    borrower_id: str
    currency: str
    amount_minor: int
    line: str


@dataclass(frozen=True)
class Transfer:
    from_entity: str
    from_account: str
    to_entity: str
    to_account: str
    send_day: int
    land_day: int
    amount_minor: int
    amount_currency: str
    landed_minor: int
    landed_currency: str
    fx_cost_minor: int
    fee_minor: int
    interest_minor: int
    rationale: str


@dataclass(frozen=True)
class Repayment:
    """The mandatory return leg of an intercompany loan at maturity:
    borrower pays principal + accrued interest back to the lender.

    Not a `Transfer`: it is not a free decision the solver chooses (its
    amount is a fixed function of the loan that created it, not its own
    variable), it does not draw against the reverse pair's own lending
    limit, and its `total_minor` already includes interest -- `leg_cost`'s
    interest formula does not apply to it and must not be run against it.
    """
    lender_id: str
    lender_account: str
    borrower_id: str
    borrower_account: str
    pay_day: int
    land_day: int
    currency: str
    principal_minor: int
    interest_minor: int
    total_minor: int
    fx_cost_minor: int
    fee_minor: int
    paid_currency: str
    paid_minor: int
    rationale: str


@dataclass(frozen=True)
class Plan:
    status: str
    scenario: str
    transfers: tuple[Transfer, ...]
    total_cost_minor: int
    solve_seconds: float
    binding_constraints: tuple[str, ...]
    closing_balances: dict[str, dict[int, int]]
    floor_shortfalls: tuple[FloorShortfall, ...] = ()
    ic_shortfalls: tuple[ICShortfall, ...] = ()
    repayments: tuple[Repayment, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.status == "OPTIMAL"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _entities(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        r["id"]: r["functional_currency"]
        for r in conn.execute("SELECT id, functional_currency FROM entity")
    }


def _countries(conn: sqlite3.Connection) -> dict[str, str]:
    """entity_id -> ISO country code, for holiday-calendar lookups. A wire's
    settlement calendar depends on where its two ends actually bank, not on
    the currency it's denominated in (MER-SG and MER-CA are both USD but sit
    on Singapore and Canadian bank holidays respectively)."""
    return {
        r["id"]: r["country"]
        for r in conn.execute("SELECT id, country FROM entity")
    }


def operating_accounts(conn: sqlite3.Connection) -> dict[str, tuple[str, str, str]]:
    """entity_id -> (account_id, bank, currency) for its operating account.

    Requires exactly one per entity: silently keeping the last of several
    (or omitting an entity with none) would route transfers through whichever
    account happened to be picked, not the one anyone intended.
    """
    rows = conn.execute(
        "SELECT id, entity_id, bank, currency FROM bank_account "
        "WHERE purpose = 'operating'"
    ).fetchall()
    accounts: dict[str, tuple[str, str, str]] = {}
    for r in rows:
        if r["entity_id"] in accounts:
            raise ValueError(
                f"{r['entity_id']} has more than one 'operating' bank_account"
            )
        accounts[r["entity_id"]] = (r["id"], r["bank"], r["currency"])
    missing = {
        r["id"] for r in conn.execute("SELECT id FROM entity")
    } - accounts.keys()
    if missing:
        raise ValueError(f"entities with no 'operating' bank_account: {sorted(missing)}")
    return accounts


def _ic_agreements(conn: sqlite3.Connection) -> dict[tuple[str, str], dict]:
    rows = conn.execute(
        "SELECT lender_id, borrower_id, max_limit, rate_bps, term_days, permitted, reason "
        "FROM ic_agreement"
    )
    return {(r["lender_id"], r["borrower_id"]): dict(r) for r in rows}


def _transfer_costs(conn: sqlite3.Connection) -> dict[tuple[str, str], tuple[int, int]]:
    rows = conn.execute(
        "SELECT from_bank, to_bank, fixed_fee, settlement_days FROM transfer_cost"
    )
    return {(r["from_bank"], r["to_bank"]): (r["fixed_fee"], r["settlement_days"])
            for r in rows}


def _net_flows(conn: sqlite3.Connection) -> dict[str, dict[int, int]]:
    rows = conn.execute(
        "SELECT entity_id, day, net_flow FROM cash_forecast ORDER BY entity_id, day"
    )
    flows: dict[str, dict[int, int]] = {}
    for r in rows:
        flows.setdefault(r["entity_id"], {})[r["day"]] = r["net_flow"]
    return flows


def _transfer_cost_for(
    costs: dict[tuple[str, str], tuple[int, int]], from_bank: str, to_bank: str
) -> tuple[int, int]:
    """Look up (fixed_fee, settlement_days) for a bank pair, failing loudly.

    A missing row here means the seed data (or a typo'd bank name) has a gap,
    not that the transfer is free and instant -- silently defaulting to (0, 0)
    would let the solver optimise against a route that cannot actually be
    executed.
    """
    pair = costs.get((from_bank, to_bank))
    if pair is None:
        raise KeyError(
            f"no transfer_cost row for {from_bank!r} -> {to_bank!r}; "
            "seed data is missing this pair or a bank name is wrong"
        )
    return pair


def earliest_actionable_day(
    entity_id: str,
    entity_ids: list[str],
    accounts: dict[str, tuple[str, str, str]],
    ic: dict[tuple[str, str], dict],
    costs: dict[tuple[str, str], tuple[int, int]],
) -> int:
    """The first horizon day a transfer sent today could possibly land at
    `entity_id`, given the fastest permitted lender and settlement lag.

    A plan built today cannot un-happen days before this one -- whatever an
    entity's balance already is on those days is fixed reality, not something
    any transfer decision can influence. Floor testing (and the diagnosis of
    a shortfall) is only meaningful from this day forward.

    If `entity_id` has no permitted lender at all, this returns 0 rather than
    some later day: nothing will ever arrive to help it, so its own balance
    from day 0 onward (net of anything it chooses to lend out) is exactly as
    real as the floor requires, and any shortfall is genuinely unfixable
    from day 0 -- that is the correct day to start reporting it, not a grace
    window it does not have.
    """
    lags = []
    for lender in entity_ids:
        if lender == entity_id:
            continue
        agree = ic.get((lender, entity_id))
        if not agree or not agree["permitted"]:
            continue
        from_bank = accounts[lender][1]
        to_bank = accounts[entity_id][1]
        _, settlement_days = _transfer_cost_for(costs, from_bank, to_bank)
        lags.append(settlement_days)
    return min(lags) if lags else 0


def _baseline_balances(
    entity_ids: list[str],
    opening: dict[str, int],
    flows: dict[str, dict[int, int]],
) -> dict[str, dict[int, int]]:
    """Closing balance per entity/day with zero transfers -- what reality
    would have produced on its own.

    Used to bound how much of a pre-`earliest_actionable_day` shortfall a
    plan is excused for: a plan cannot be blamed for a dip baseline reality
    already had (nothing could have landed there in time to prevent it), but
    it is always fully in control of its own outbound sends, even before
    that day, and must never be allowed to use the exemption to push an
    entity below where it would have sat on its own.
    """
    balances: dict[str, dict[int, int]] = {e: {} for e in entity_ids}
    for e in entity_ids:
        running = opening.get(e, 0)
        for d in range(HORIZON_DAYS):
            running += flows.get(e, {}).get(d, 0)
            balances[e][d] = running
    return balances


def _permitted_pairs(entity_ids: list[str], ic: dict[tuple[str, str], dict]) -> list[tuple[str, str]]:
    pairs = []
    for i in entity_ids:
        for j in entity_ids:
            if i == j:
                continue
            agree = ic.get((i, j))
            if agree and agree["permitted"]:
                pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------
# Shared cost math -- one place both the solver and the naive baseline use so
# a Transfer's numbers always mean the same thing regardless of who built it.
# --------------------------------------------------------------------------

def leg_cost(amount_minor: int, ci: str, cj: str, rate_bps: int, days_out: int,
             fee_usd: int, fx: FXTable) -> tuple[int, int, int, int]:
    """Given a transfer amount (minor units of ci) return
    (landed_minor, fx_cost_minor_usd, fee_minor_usd, interest_minor_usd).

    `days_out` is the number of days interest is charged for: the loan's
    real contractual term if it matures and repays within the horizon, or
    the number of days remaining in the horizon if it doesn't (in which
    case this is only an as-of-day-13 snapshot, not the loan's true cost --
    the caller is responsible for picking the right one, not this function).
    """
    quote = fx.rate(ci, cj)
    # round(), not int(): truncation systematically shorts the receiving
    # entity by up to a minor unit, which can trip a floor the LP considered
    # exactly satisfied at full float precision.
    landed = round(amount_minor * quote.effective)
    fx_cost_ci = round(amount_minor * quote.spread_bps / 20_000)
    usd_rate = fx.rate(ci, "USD").effective
    fx_cost_usd = round(fx_cost_ci * usd_rate)
    interest_ci = round(amount_minor * rate_bps / 10_000 * max(0, days_out) / 365)
    interest_usd = round(interest_ci * usd_rate)
    return landed, fx_cost_usd, fee_usd, interest_usd


def project_balances(
    entity_ids: list[str],
    opening: dict[str, int],
    flows: dict[str, dict[int, int]],
    transfers: tuple[Transfer, ...],
    repayments: tuple[Repayment, ...] = (),
) -> dict[str, dict[int, int]]:
    """Closing balance per entity per day given a fixed set of transfers
    and (loan) repayments.

    Pure re-derivation from the ledger + transfer legs -- used both to build
    Plan.closing_balances and, independently, by verify() to check it.
    """
    balances: dict[str, dict[int, int]] = {e: {} for e in entity_ids}
    for e in entity_ids:
        running = opening.get(e, 0)
        for d in range(HORIZON_DAYS):
            running += flows.get(e, {}).get(d, 0)
            for t in transfers:
                if t.from_entity == e and t.send_day == d:
                    running -= t.amount_minor
                if t.to_entity == e and t.land_day == d:
                    running += t.landed_minor
            for r in repayments:
                if r.borrower_id == e and r.pay_day == d:
                    running -= r.paid_minor
                if r.lender_id == e and r.land_day == d:
                    running += r.total_minor
            balances[e][d] = running
    return balances


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------

def _build_legs(entity_ids, entities, accounts, ic, costs, fx, pairs, start, countries):
    """Static per-(i,j,d) metadata: land_day, fee, FX quote, ic rate, and
    (if it fits inside the horizon) the mandatory repayment leg at maturity.

    `land_day` is settlement-calendar-adjusted: a wire never actually clears
    on a weekend, or on a public holiday in either the sending or receiving
    country, whatever the raw settlement_days lag arithmetic says.

    Each loan is denominated in the lender's currency: the borrower's legal
    obligation at maturity is `principal * (1 + rate * term/365)` in the
    lender's currency, and the borrower -- who holds its own currency, not
    the lender's -- is the one converting to pay it, so it (not the lender)
    bears that conversion's spread. That mirrors who bears the spread on
    the forward draw: whoever is doing the converting.
    """
    legs = {}
    for (i, j) in pairs:
        from_bank = accounts[i][1]
        to_bank = accounts[j][1]
        fee_usd, settlement_days = _transfer_cost_for(costs, from_bank, to_bank)
        rate = fx.rate(entities[i], entities[j])
        rate_bps = ic[(i, j)]["rate_bps"]
        term_days = ic[(i, j)]["term_days"]
        pair_countries = frozenset({countries[i], countries[j]})
        repay_fee_usd, repay_settlement_days = _transfer_cost_for(costs, to_bank, from_bank)
        repay_quote = fx.rate(entities[j], entities[i])  # borrower (cj) -> lender (ci)
        repay_coef = 1 + rate_bps / 10_000 * term_days / 365  # owed, in ci, per unit lent
        for d in range(HORIZON_DAYS):
            if not is_settlement_day(d, start, pair_countries):
                continue  # wires are not initiated on a non-settlement day
            land_day = next_settlement_day(d + settlement_days, start, pair_countries)
            if land_day > HORIZON_DAYS - 1:
                continue

            maturity_day = land_day + term_days
            repay = None
            if maturity_day <= HORIZON_DAYS - 1:
                pay_day = next_settlement_day(maturity_day, start, pair_countries)
                repay_land_day = next_settlement_day(
                    pay_day + repay_settlement_days, start, pair_countries
                )
                if pay_day <= HORIZON_DAYS - 1 and repay_land_day <= HORIZON_DAYS - 1:
                    repay = {
                        "pay_day": pay_day,
                        "land_day": repay_land_day,
                        "coef_ci": repay_coef,
                        "send_coef": repay_coef / repay_quote.effective,
                        "quote": repay_quote,
                        "fee_usd": repay_fee_usd,
                    }

            legs[(i, j, d)] = {
                "land_day": land_day,
                "fee_usd": fee_usd,
                "rate": rate,
                "rate_bps": rate_bps,
                "term_days": term_days,
                "maturity_day": maturity_day,
                "repay": repay,
            }
    return legs


def _extract_transfers(entities, accounts, legs, x, fx) -> list[Transfer]:
    transfers = []
    for (i, j, d), var in x.items():
        amt = round(var.value() or 0)
        if amt <= 0:
            continue
        meta = legs[(i, j, d)]
        ci, cj = entities[i], entities[j]
        # Real contractual term if this loan matures and repays inside the
        # horizon; otherwise an as-of-day-13 snapshot, since nothing beyond
        # the horizon is visible to price against.
        days_out = meta["term_days"] if meta["repay"] else max(0, HORIZON_DAYS - meta["land_day"])
        landed, fx_cost, fee, interest = leg_cost(
            amt, ci, cj, meta["rate_bps"], days_out, meta["fee_usd"], fx
        )
        transfers.append(Transfer(
            from_entity=i,
            from_account=accounts[i][0],
            to_entity=j,
            to_account=accounts[j][0],
            send_day=d,
            land_day=meta["land_day"],
            amount_minor=amt,
            amount_currency=ci,
            landed_minor=landed,
            landed_currency=cj,
            fx_cost_minor=fx_cost,
            fee_minor=fee,
            interest_minor=interest,
            rationale=(
                f"{i} sends {ci} {to_major(amt)} to {j} on day {d}, "
                f"lands day {meta['land_day']} as {cj} {to_major(landed)}"
            ),
        ))
    transfers.sort(key=lambda t: (t.send_day, t.from_entity, t.to_entity))
    return transfers


def _extract_repayments(entities, accounts, legs, x, fx) -> list[Repayment]:
    """Realised repayment legs: one per drawn loan whose maturity fits
    inside the horizon. Rounds at extraction time, same as
    `_extract_transfers` -- the LP itself works in continuous floats, and
    only the realised, auditable numbers get rounded."""
    repayments = []
    for (i, j, d), var in x.items():
        amt = round(var.value() or 0)
        meta = legs[(i, j, d)]
        repay = meta["repay"]
        if amt <= 0 or repay is None:
            continue
        ci, cj = entities[i], entities[j]
        principal = amt
        total_ci = repayment_owed_minor(principal, meta["rate_bps"], meta["term_days"])
        interest_ci = total_ci - principal
        quote = repay["quote"]  # cj -> ci, borrower converting
        paid_cj = round(total_ci / quote.effective)
        fx_cost_cj = round(paid_cj * quote.spread_bps / 20_000)
        fx_cost_usd = round(fx_cost_cj * fx.rate(cj, "USD").effective)
        repayments.append(Repayment(
            lender_id=i,
            lender_account=accounts[i][0],
            borrower_id=j,
            borrower_account=accounts[j][0],
            pay_day=repay["pay_day"],
            land_day=repay["land_day"],
            currency=ci,
            principal_minor=principal,
            interest_minor=interest_ci,
            total_minor=total_ci,
            fx_cost_minor=fx_cost_usd,
            fee_minor=repay["fee_usd"],
            paid_currency=cj,
            paid_minor=paid_cj,
            rationale=(
                f"{j} repays {i} principal {ci} {to_major(principal)} + "
                f"interest {ci} {to_major(interest_ci)} on day {repay['pay_day']}, "
                f"lands day {repay['land_day']} as {ci} {to_major(total_ci)} "
                f"(paid as {cj} {to_major(paid_cj)})"
            ),
        ))
    repayments.sort(key=lambda r: (r.pay_day, r.lender_id, r.borrower_id))
    return repayments


def repayment_for_transfer(t: Transfer, repayments: tuple[Repayment, ...]) -> Repayment | None:
    """Which (if any) of `plan.repayments` closes out loan `t`, for display
    grouping only -- not a substitute for `verify()`'s from-scratch
    independent recomputation of whether a loan should have repaid.

    Matches by (lender, borrower, maturity no earlier than the loan
    landed), taking the earliest unclaimed candidate: with a single global
    `term_days` this is normally unambiguous, and the greedy earliest-match
    rule degrades sensibly if two loans between the same pair ever did
    share a maturity day.
    """
    for r in sorted(repayments, key=lambda r: r.pay_day):
        if r.lender_id == t.from_entity and r.borrower_id == t.to_entity and r.pay_day >= t.land_day:
            return r
    return None


def _build_model(entity_ids, entities, floors, opening, flows, lag_min, legs, usd_rate,
                  pairs, ic):
    """Variables, constraints, and the two objective expressions, built
    once and shared across both solve phases (see `solve`'s docstring on
    why there are two).

    `real_cost_expr` is every dollar that actually leaves the group: FX
    spread and wire fees on both the draw and (when it fits in the horizon)
    the repayment. `interest_expr` is intercompany interest -- real to the
    paying entity, but an internal transfer that nets to zero on group
    consolidation, so it is never allowed to compete against real_cost_expr
    inside one blended number; the two-phase solve in `solve()` is what
    keeps them separate.
    """
    x: dict[tuple[str, str, int], pulp.LpVariable] = {}
    y: dict[tuple[str, str, int], pulp.LpVariable] = {}
    constraints = []
    for key in legs:
        i, j, d = key
        x[key] = pulp.LpVariable(f"x_{i}_{j}_{d}", lowBound=0, upBound=BIG_M, cat="Integer")
        y[key] = pulp.LpVariable(f"y_{i}_{j}_{d}", cat="Binary")
        constraints.append((x[key] <= BIG_M * y[key], f"link_{i}_{j}_{d}"))

    real_cost_terms = []
    interest_terms = []
    for key, meta in legs.items():
        i, j, d = key
        ci, cj = entities[i], entities[j]
        fx_coef = (meta["rate"].spread_bps / 20_000) * usd_rate[ci]
        repay = meta["repay"]
        days_out = meta["term_days"] if repay else max(0, HORIZON_DAYS - meta["land_day"])
        interest_coef = (meta["rate_bps"] / 10_000) * (days_out / 365) * usd_rate[ci]
        real_cost_terms.append(x[key] * fx_coef)
        interest_terms.append(x[key] * interest_coef)
        real_cost_terms.append(y[key] * meta["fee_usd"])
        if repay:
            # The repayment leg's own real cost: the borrower converting
            # back to the lender's currency pays its own spread (whoever
            # converts bears it, same rule as the forward draw), plus a
            # second wire fee for the return leg. This is real cash cost,
            # not interest, so it belongs in real_cost_expr.
            repay_fx_coef = (
                repay["send_coef"] * (repay["quote"].spread_bps / 20_000) * usd_rate[cj]
            )
            real_cost_terms.append(x[key] * repay_fx_coef)
            real_cost_terms.append(y[key] * repay["fee_usd"])

    balance_expr = {e: {} for e in entity_ids}
    for e in entity_ids:
        ce = entities[e]
        buffer_target = floors.get(e, 0) + round(floors.get(e, 0) * BUFFER_BPS / 10_000)
        # Soft preference, not a constraint: a plan that parks a balance
        # exactly on its floor with zero headroom is legal but not something
        # a treasurer should have to sign off on unless cash is genuinely too
        # scarce to do better. One slack variable per entity -- not one per
        # entity-day -- covers that entity's worst day across the whole
        # horizon (every day's constraint below shares it, so the LP must
        # size it to the largest gap), but it is only charged once in the
        # objective. Charging it per day instead would compound a persisting
        # gap into a cost many times larger than the real FX
        # spread of actually moving cash to close it, which stops being a tie-break.
        buffer_slack = pulp.LpVariable(f"buffer_slack_{e}", lowBound=0)
        cum_flow = 0
        for d in range(HORIZON_DAYS):
            cum_flow += flows.get(e, {}).get(d, 0)
            baseline_bal = opening.get(e, 0) + cum_flow
            expr = baseline_bal
            for key, var in x.items():
                i, j, dd = key
                meta = legs[key]
                if i == e and dd <= d:
                    expr = expr - var
                if j == e and meta["land_day"] <= d:
                    expr = expr + var * meta["rate"].effective
                repay = meta["repay"]
                if repay:
                    if j == e and repay["pay_day"] <= d:
                        expr = expr - var * repay["send_coef"]
                    if i == e and repay["land_day"] <= d:
                        expr = expr + var * repay["coef_ci"]
            balance_expr[e][d] = expr
            if d >= lag_min[e]:
                constraints.append((expr >= floors.get(e, 0), f"floor_{e}_{d}"))
                constraints.append((expr >= 0, f"nonneg_{e}_{d}"))
                constraints.append((expr + buffer_slack >= buffer_target, f"buffer_{e}_{d}"))
            else:
                # No inbound transfer can land here by definition of
                # lag_min[e] -- so a floor/negative shortfall baseline
                # reality already has on its own is unfixable and stays
                # exempt. But this entity's own outbound sends are fully
                # its choice even this early: it may never use the
                # exemption to end up worse off than doing nothing at all.
                constraints.append(
                    (expr >= min(floors.get(e, 0), baseline_bal), f"floor_{e}_{d}")
                )
                constraints.append((expr >= min(0, baseline_bal), f"nonneg_{e}_{d}"))
        # Charged once per entity here, after the day loop, regardless of how
        # many (or zero) days actually constrained buffer_slack above. A
        # preference about headroom, not a real cost -- bucketed into
        # real_cost_expr (not interest_expr) purely so phase 2 can't spend
        # it away chasing a lower interest number.
        real_cost_terms.append(
            buffer_slack * (BUFFER_PENALTY_BPS / 10_000) * usd_rate[ce]
        )

    for (i, j) in pairs:
        total_lent = pulp.lpSum(x[key] for key in x if key[0] == i and key[1] == j)
        constraints.append((total_lent <= ic[(i, j)]["max_limit"], f"iclimit_{i}_{j}"))

    return {
        "x": x,
        "y": y,
        "balance_expr": balance_expr,
        "real_cost_expr": pulp.lpSum(real_cost_terms),
        "interest_expr": pulp.lpSum(interest_terms),
        "constraints": constraints,
    }


def _add_constraints(prob: pulp.LpProblem, constraints) -> None:
    for constraint, name in constraints:
        prob += constraint, name


def solve(conn: sqlite3.Connection, scenario: str) -> Plan:
    start = time.monotonic()

    entities = _entities(conn)
    entity_ids = list(entities.keys())
    accounts = operating_accounts(conn)
    floors = binding_floors(conn)
    opening = opening_balances(conn)
    flows = _net_flows(conn)
    ic = _ic_agreements(conn)
    costs = _transfer_costs(conn)
    fx = FXTable(conn)
    hstart = horizon_start(conn)
    countries = _countries(conn)

    pairs = _permitted_pairs(entity_ids, ic)
    legs = _build_legs(entity_ids, entities, accounts, ic, costs, fx, pairs, hstart, countries)
    lag_min = {e: earliest_actionable_day(e, entity_ids, accounts, ic, costs) for e in entity_ids}

    usd_rate = {c: fx.rate(c, "USD").effective for c in set(entities.values())}

    model = _build_model(entity_ids, entities, floors, opening, flows, lag_min, legs, usd_rate,
                          pairs, ic)
    x = model["x"]

    # Lexicographic ordering via two solves, phase 1 minimising real cost
    # then phase 2 minimising interest subject to real cost held at phase
    # 1's optimum. (A single MILP minimising REAL_COST_WEIGHT*real_cost +
    # interest was tried first, to avoid a second solve -- it produces the
    # right answer but a weight large enough to guarantee lexicographic
    # ordering makes the objective's coefficients span ~9 orders of
    # magnitude, which wrecks CBC's branch-and-bound performance: solve
    # time went from ~6s to over a minute on this same scenario. Two solves
    # of well-conditioned single-scale objectives is faster in practice
    # than one solve of a badly-conditioned combined one.)
    prob1 = pulp.LpProblem("sluice_phase1_real_cost", pulp.LpMinimize)
    prob1 += model["real_cost_expr"]
    _add_constraints(prob1, model["constraints"])
    prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob1.status]
    solve_seconds = time.monotonic() - start

    if status != "Optimal":
        binding, floor_shortfalls, ic_shortfalls = _diagnose_infeasibility(
            entity_ids, entities, accounts, floors, opening, flows, ic, pairs,
            legs, usd_rate, lag_min,
        )
        return Plan(
            status="INFEASIBLE",
            scenario=scenario,
            transfers=(),
            total_cost_minor=0,
            solve_seconds=solve_seconds,
            binding_constraints=tuple(binding),
            closing_balances={},
            floor_shortfalls=tuple(floor_shortfalls),
            ic_shortfalls=tuple(ic_shortfalls),
        )

    # Rounded to the nearest minor unit before locking, not used as the raw
    # float CBC reports: real_cost_expr's true value is not integral (its
    # coefficients are FX/interest rate fractions), so the raw value carries
    # sub-cent noise that -- confirmed empirically -- is large enough to
    # make phase 2's lock constraint spuriously Infeasible on some
    # scenarios even though phase 1's own solution trivially satisfies it.
    # Locking to the rounded value (the same rounding every reported cost
    # already goes through) plus 1 minor unit of slack is exact enough for
    # money and immune to that noise.
    real_cost_optimum = round(pulp.value(model["real_cost_expr"]))

    prob2 = pulp.LpProblem("sluice_phase2_interest", pulp.LpMinimize)
    prob2 += model["interest_expr"]
    _add_constraints(prob2, model["constraints"])
    prob2 += model["real_cost_expr"] <= real_cost_optimum + 1, "lock_real_cost"
    prob2.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_seconds = time.monotonic() - start
    # phase 2 starts from phase 1's already-Optimal, now-further-constrained
    # feasible region, so it cannot come back Infeasible -- if it ever did,
    # that would mean the lock constraint above is wrong, not that this
    # scenario is unsolvable.
    assert pulp.LpStatus[prob2.status] == "Optimal", (
        f"phase 2 (interest tie-break) failed with phase 1 already Optimal: "
        f"{pulp.LpStatus[prob2.status]!r} -- lock_real_cost constraint is "
        "almost certainly wrong"
    )

    transfers = tuple(_extract_transfers(entities, accounts, legs, x, fx))
    repayments = tuple(_extract_repayments(entities, accounts, legs, x, fx))
    total_cost = sum(t.fx_cost_minor + t.fee_minor + t.interest_minor for t in transfers)
    total_cost += sum(r.fx_cost_minor + r.fee_minor for r in repayments)
    closing_balances = project_balances(entity_ids, opening, flows, transfers, repayments)

    binding = []
    for e in entity_ids:
        floor = floors.get(e, 0)
        for d in range(lag_min[e], HORIZON_DAYS):
            # Within 1 minor unit, not exact equality: closing_balances is
            # re-derived through leg_cost's round(), which can differ from
            # the LP's full-float value by a unit even when the LP treated
            # the floor as exactly binding.
            if abs(closing_balances[e][d] - floor) <= 1:
                binding.append(f"{e} floor tight (={to_major(floor)} {entities[e]}) on day {d}")
    for (i, j) in pairs:
        total_lent = sum(t.amount_minor for t in transfers if t.from_entity == i and t.to_entity == j)
        limit = ic[(i, j)]["max_limit"]
        if total_lent == limit and limit > 0:
            binding.append(f"{i}->{j} intercompany limit fully drawn ({to_major(limit)} {entities[i]})")

    return Plan(
        status="OPTIMAL",
        scenario=scenario,
        transfers=transfers,
        total_cost_minor=total_cost,
        solve_seconds=solve_seconds,
        binding_constraints=tuple(binding),
        closing_balances=closing_balances,
        repayments=repayments,
    )


def _diagnose_infeasibility(
    entity_ids, entities, accounts, floors, opening, flows, ic, pairs, legs,
    usd_rate, lag_min,
) -> tuple[list[str], list[FloorShortfall], list[ICShortfall]]:
    """Solve the elastic relaxation and report exactly which constraints, and
    by how much, could not be satisfied."""
    prob = pulp.LpProblem("sluice_diagnosis", pulp.LpMinimize)

    x: dict[tuple[str, str, int], pulp.LpVariable] = {}
    for key in legs:
        i, j, d = key
        x[key] = pulp.LpVariable(f"x_{i}_{j}_{d}", lowBound=0, upBound=BIG_M, cat="Continuous")

    floor_slack: dict[tuple[str, int], pulp.LpVariable] = {}
    ic_slack: dict[tuple[str, str], pulp.LpVariable] = {}

    objective_terms = []
    for key, meta in legs.items():
        i, j, d = key
        ci = entities[i]
        fx_coef = (meta["rate"].spread_bps / 20_000) * usd_rate[ci]
        days_out = max(0, HORIZON_DAYS - meta["land_day"])
        interest_coef = (meta["rate_bps"] / 10_000) * (days_out / 365) * usd_rate[ci]
        # Fee is per-leg, not per-unit, and there's no binary "used this leg"
        # indicator in this relaxation to attach it to. Scale it down by
        # BIG_M rather than an arbitrary small constant so it stays a
        # negligible tie-break against real fx/interest coefficients
        # (~1e-4 per unit) instead of dwarfing them.
        objective_terms.append(x[key] * (fx_coef + interest_coef + meta["fee_usd"] / BIG_M))

    balance_expr = {e: {} for e in entity_ids}
    for e in entity_ids:
        cum_flow = 0
        for d in range(HORIZON_DAYS):
            cum_flow += flows.get(e, {}).get(d, 0)
            expr = opening.get(e, 0) + cum_flow
            for key, var in x.items():
                i, j, dd = key
                if i == e and dd <= d:
                    expr = expr - var
                if j == e and legs[key]["land_day"] <= d:
                    expr = expr + var * legs[key]["rate"].effective
            balance_expr[e][d] = expr
            if d >= lag_min[e]:
                slack = pulp.LpVariable(f"floor_slack_{e}_{d}", lowBound=0)
                floor_slack[(e, d)] = slack
                prob += expr + slack >= floors.get(e, 0), f"floor_{e}_{d}"
                objective_terms.append(slack * PENALTY)

    for (i, j) in pairs:
        total_lent = pulp.lpSum(x[key] for key in x if key[0] == i and key[1] == j)
        slack = pulp.LpVariable(f"ic_slack_{i}_{j}", lowBound=0)
        ic_slack[(i, j)] = slack
        prob += total_lent - slack <= ic[(i, j)]["max_limit"], f"iclimit_{i}_{j}"
        objective_terms.append(slack * PENALTY)

    prob += pulp.lpSum(objective_terms)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # One line per entity, its single worst (largest-shortfall) day, rather
    # than every day it stays underwater once the group runs out of capacity
    # -- the diagnosis wants the binding constraint, not a daily ledger.
    worst_by_entity: dict[str, tuple[int, float]] = {}
    for (e, d), slack in floor_slack.items():
        val = slack.value() or 0
        if val > 0.5 and val > worst_by_entity.get(e, (None, 0))[1]:
            worst_by_entity[e] = (d, val)

    binding: list[str] = []
    floor_shortfalls: list[FloorShortfall] = []
    for e, (d, val) in sorted(worst_by_entity.items()):
        amount = round(val)
        line = (
            f"{e} short {to_major(amount)} {entities[e]} of its "
            f"{to_major(floors.get(e, 0))} {entities[e]} floor on day {d} "
            "even after routing all available, permitted intercompany capacity"
        )
        binding.append(line)
        floor_shortfalls.append(FloorShortfall(
            entity_id=e, currency=entities[e], amount_minor=amount, day=d, line=line,
        ))

    ic_shortfalls: list[ICShortfall] = []
    for (i, j), slack in sorted(ic_slack.items()):
        val = slack.value() or 0
        if val > 0.5:
            amount = round(val)
            line = (
                f"{i}->{j} intercompany limit of {to_major(ic[(i, j)]['max_limit'])} "
                f"{entities[i]} is {to_major(amount)} short of what the plan needed to route"
            )
            binding.append(line)
            ic_shortfalls.append(ICShortfall(
                lender_id=i, borrower_id=j, currency=entities[i], amount_minor=amount, line=line,
            ))

    if not binding:
        binding.append(
            "solve failed with no isolable shortfall found by the elastic "
            "relaxation -- check for a modelling bug"
        )
    return binding, floor_shortfalls, ic_shortfalls


# --------------------------------------------------------------------------
# Independent verification
# --------------------------------------------------------------------------

def verify(conn: sqlite3.Connection, plan: Plan) -> list[str]:
    """Re-check every constraint against `plan`, independently of the solver.

    Trusts nothing the solver claims: recomputes balances, exposure and costs
    straight from the database and the transfer legs.
    """
    violations: list[str] = []

    if plan.status not in ("OPTIMAL", "INFEASIBLE"):
        violations.append(f"invalid status {plan.status!r}")
        return violations

    if plan.status == "INFEASIBLE":
        if plan.transfers:
            violations.append("INFEASIBLE plan has non-empty transfers")
        if plan.total_cost_minor != 0:
            violations.append("INFEASIBLE plan has non-zero total_cost_minor")
        if not plan.binding_constraints:
            violations.append("INFEASIBLE plan has empty binding_constraints")
        return violations

    entities = _entities(conn)
    entity_ids = list(entities.keys())
    account_rows = conn.execute(
        "SELECT id, entity_id, bank, currency FROM bank_account"
    )
    account_by_id = {r["id"]: (r["entity_id"], r["bank"], r["currency"]) for r in account_rows}
    floors = binding_floors(conn)
    opening = opening_balances(conn)
    flows = _net_flows(conn)
    ic = _ic_agreements(conn)
    costs = _transfer_costs(conn)
    accounts = operating_accounts(conn)
    lag_min = {e: earliest_actionable_day(e, entity_ids, accounts, ic, costs) for e in entity_ids}
    start = horizon_start(conn)
    fx = FXTable(conn)
    countries = _countries(conn)

    ic_cumulative: dict[tuple[str, str], int] = {}
    recomputed_total = 0
    expected_repayments: list[tuple] = []
    claimable_repayments = list(plan.repayments)

    for t in plan.transfers:
        pair_countries = frozenset({
            countries.get(t.from_entity, ""), countries.get(t.to_entity, ""),
        }) - {""}
        if t.amount_minor <= 0:
            violations.append(
                f"transfer {t.from_entity}->{t.to_entity} day {t.send_day}: "
                f"non-positive amount_minor {t.amount_minor}"
            )
        if not (0 <= t.send_day < HORIZON_DAYS):
            violations.append(
                f"transfer {t.from_entity}->{t.to_entity}: send_day {t.send_day} out of horizon"
            )
        elif not is_settlement_day(t.send_day, start, pair_countries):
            violations.append(
                f"transfer {t.from_entity}->{t.to_entity}: send_day {t.send_day} "
                "is not a settlement day (weekend or a bank holiday in "
                f"{sorted(pair_countries)})"
            )

        from_acc = account_by_id.get(t.from_account)
        if from_acc is None or from_acc[0] != t.from_entity:
            violations.append(f"from_account {t.from_account} does not belong to {t.from_entity}")
        to_acc = account_by_id.get(t.to_account)
        if to_acc is None or to_acc[0] != t.to_entity:
            violations.append(f"to_account {t.to_account} does not belong to {t.to_entity}")

        agree = ic.get((t.from_entity, t.to_entity))
        if not agree or not agree["permitted"]:
            violations.append(
                f"transfer {t.from_entity}->{t.to_entity} routes across a "
                "prohibited (permitted=0) ic_agreement pair"
            )

        if from_acc is not None and to_acc is not None:
            fee, settle = costs.get((from_acc[1], to_acc[1]), (None, None))
            if settle is None:
                violations.append(f"no transfer_cost row for {from_acc[1]}->{to_acc[1]}")
            else:
                expected_land = next_settlement_day(t.send_day + settle, start, pair_countries)
                if t.land_day != expected_land:
                    violations.append(
                        f"transfer {t.from_entity}->{t.to_entity} day {t.send_day}: "
                        f"land_day {t.land_day} != settlement-calendar-adjusted "
                        f"send_day + settlement_days ({expected_land})"
                    )

        if from_acc is not None and to_acc is not None:
            agreement = ic.get((t.from_entity, t.to_entity), {})
            rate_bps = agreement.get("rate_bps")
            term_days = agreement.get("term_days")
            fee_usd, _ = costs.get((from_acc[1], to_acc[1]), (0, 0))
            if rate_bps is not None and term_days is not None:
                # Whether this loan repays within the horizon is a planning
                # choice, not a fact derivable from the transfer alone: the
                # naive baseline deliberately never repays even when it
                # could, and that is a legitimate different choice, not a
                # defect. So take "claimed by a repayment in plan.repayments"
                # as the plan's own choice, and check that choice is
                # internally consistent (correct amounts, correct days,
                # inside the horizon) -- not that it made the same choice
                # the primary solver would have.
                claimed = next(
                    (r for r in claimable_repayments
                     if r.lender_id == t.from_entity and r.borrower_id == t.to_entity
                     and r.pay_day >= t.land_day),
                    None,
                )
                if claimed is not None:
                    claimable_repayments.remove(claimed)
                days_out = term_days if claimed is not None else max(0, HORIZON_DAYS - t.land_day)
                landed, fx_cost, fee, interest = leg_cost(
                    t.amount_minor, entities[t.from_entity], entities[t.to_entity],
                    rate_bps, days_out, fee_usd, fx,
                )

                if claimed is not None:
                    ci, cj = entities[t.from_entity], entities[t.to_entity]
                    pay_day = next_settlement_day(t.land_day + term_days, start, pair_countries)
                    _, repay_settle = costs.get((to_acc[1], from_acc[1]), (0, 0))
                    repay_land_day = next_settlement_day(
                        pay_day + repay_settle, start, pair_countries
                    )
                    total_ci = repayment_owed_minor(t.amount_minor, rate_bps, term_days)
                    quote = fx.rate(cj, ci)
                    paid_cj = round(total_ci / quote.effective)
                    repay_fee_usd, _ = costs.get((to_acc[1], from_acc[1]), (0, 0))
                    fx_cost_cj = round(paid_cj * quote.spread_bps / 20_000)
                    repay_fx_cost_usd = round(fx_cost_cj * fx.rate(cj, "USD").effective)
                    recomputed_total += repay_fx_cost_usd + repay_fee_usd
                    if (pay_day > HORIZON_DAYS - 1 or repay_land_day > HORIZON_DAYS - 1):
                        violations.append(
                            f"repayment {t.to_entity}->{t.from_entity}: claimed to close "
                            f"the loan sent day {t.send_day}, but its own maturity/landing "
                            "falls outside the horizon"
                        )
                    expected_repayments.append((
                        t.from_entity, t.to_entity, pay_day, repay_land_day,
                        total_ci, paid_cj, repay_fx_cost_usd, repay_fee_usd,
                    ))
                if landed != t.landed_minor:
                    violations.append(
                        f"transfer {t.from_entity}->{t.to_entity} day {t.send_day}: "
                        f"landed_minor {t.landed_minor} != recomputed {landed}"
                    )
                if (fx_cost, fee, interest) != (t.fx_cost_minor, t.fee_minor, t.interest_minor):
                    violations.append(
                        f"transfer {t.from_entity}->{t.to_entity} day {t.send_day}: "
                        f"cost (fx={t.fx_cost_minor}, fee={t.fee_minor}, interest={t.interest_minor}) "
                        f"!= recomputed (fx={fx_cost}, fee={fee}, interest={interest})"
                    )
                recomputed_total += fx_cost + fee + interest

        key = (t.from_entity, t.to_entity)
        ic_cumulative[key] = ic_cumulative.get(key, 0) + t.amount_minor

    for (lender, borrower), total in ic_cumulative.items():
        limit = ic.get((lender, borrower), {}).get("max_limit")
        if limit is not None and total > limit:
            violations.append(
                f"IC exposure {lender}->{borrower} totals {total} minor units, "
                f"exceeds max_limit {limit}"
            )

    if recomputed_total != plan.total_cost_minor:
        violations.append(
            f"plan.total_cost_minor {plan.total_cost_minor} != recomputed "
            f"{recomputed_total} (sum of independently-recomputed per-leg fx+fee+interest, "
            "including repayment legs)"
        )

    # Every repayment the plan claimed (matched above, per-transfer) is
    # re-verified here against its own specific fields -- exact day and
    # amount, not just "some repayment exists" -- and any repayment left
    # over that matched no transfer at all is a defect regardless of
    # planning policy.
    unmatched_actual = list(plan.repayments)
    for (lender, borrower, pay_day, land_day, total_ci, paid_cj,
         fx_cost, fee) in expected_repayments:
        match = next(
            (r for r in unmatched_actual
             if r.lender_id == lender and r.borrower_id == borrower
             and r.pay_day == pay_day and r.land_day == land_day
             and abs(r.total_minor - total_ci) <= 1 and abs(r.paid_minor - paid_cj) <= 1),
            None,
        )
        if match is None:
            violations.append(
                f"loan {lender}->{borrower} matures day {pay_day} lands {land_day}: "
                f"expected a repayment of {total_ci} minor units ({lender}'s currency), "
                "none found in plan.repayments"
            )
        else:
            unmatched_actual.remove(match)
            if (match.fx_cost_minor, match.fee_minor) != (fx_cost, fee):
                violations.append(
                    f"repayment {borrower}->{lender} day {pay_day}: cost "
                    f"(fx={match.fx_cost_minor}, fee={match.fee_minor}) != recomputed "
                    f"(fx={fx_cost}, fee={fee})"
                )
    for r in unmatched_actual:
        violations.append(
            f"repayment {r.borrower_id}->{r.lender_id} day {r.pay_day}: does not match any "
            "transfer's loan (wrong lender/borrower/day, or claims a loan twice)"
        )

    balances = project_balances(entity_ids, opening, flows, plan.transfers, plan.repayments)
    baseline = _baseline_balances(entity_ids, opening, flows)
    for e in entity_ids:
        for d in range(HORIZON_DAYS):
            bal = balances[e][d]
            floor = floors.get(e, 0)
            if d >= lag_min[e]:
                if bal < floor:
                    violations.append(f"{e} closing balance on day {d} is {bal}, below floor {floor}")
                if bal < 0:
                    violations.append(f"{e} closing balance on day {d} is negative: {bal}")
            else:
                # Before lag_min[e] no inbound transfer can possibly have
                # landed, so a shortfall baseline reality already had here is
                # unfixable and exempt -- but this entity's own outbound
                # sends are always within the plan's control, and must never
                # be allowed to leave it worse off than doing nothing.
                base = baseline[e][d]
                floor_bound = min(floor, base)
                nonneg_bound = min(0, base)
                if bal < floor_bound:
                    violations.append(
                        f"{e} closing balance on day {d} is {bal}, below "
                        f"floor {floor_bound} (day is before earliest_actionable_day "
                        f"{lag_min[e]}, but the plan's own outbound transfers made "
                        f"it worse than the baseline {base})"
                    )
                if bal < nonneg_bound:
                    violations.append(
                        f"{e} closing balance on day {d} is negative: {bal} "
                        f"(day is before earliest_actionable_day {lag_min[e]}, "
                        f"but the plan's own outbound transfers made it worse "
                        f"than the baseline {base})"
                    )
            plan_bal = plan.closing_balances.get(e, {}).get(d)
            if plan_bal is not None and plan_bal != bal:
                violations.append(
                    f"{e} day {d}: plan.closing_balances reports {plan_bal}, "
                    f"recomputed {bal}"
                )

    return violations
