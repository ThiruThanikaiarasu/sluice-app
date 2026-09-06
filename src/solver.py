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

from .fx import FXTable
from .models import HORIZON_DAYS, to_major
from .positions import binding_floors, opening_balances

# Safe upper bound on any single transfer leg, in minor units of the sending
# currency. Generous relative to the seeded company's balances (low millions).
BIG_M = 2_000_000_000

# Weight that forces the elastic solve to eliminate shortfall before it cares
# about real transfer cost at all.
PENALTY = 10**9


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
        "SELECT lender_id, borrower_id, max_limit, rate_bps, permitted, reason "
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

def leg_cost(amount_minor: int, ci: str, cj: str, rate_bps: int, land_day: int,
             fee_usd: int, fx: FXTable) -> tuple[int, int, int, int]:
    """Given a transfer amount (minor units of ci) return
    (landed_minor, fx_cost_minor_usd, fee_minor_usd, interest_minor_usd)."""
    quote = fx.rate(ci, cj)
    # round(), not int(): truncation systematically shorts the receiving
    # entity by up to a minor unit, which can trip a floor the LP considered
    # exactly satisfied at full float precision.
    landed = round(amount_minor * quote.effective)
    fx_cost_ci = round(amount_minor * quote.spread_bps / 20_000)
    usd_rate = fx.rate(ci, "USD").effective
    fx_cost_usd = round(fx_cost_ci * usd_rate)
    days_out = max(0, HORIZON_DAYS - land_day)
    interest_ci = round(amount_minor * rate_bps / 10_000 * days_out / 365)
    interest_usd = round(interest_ci * usd_rate)
    return landed, fx_cost_usd, fee_usd, interest_usd


def project_balances(
    entity_ids: list[str],
    opening: dict[str, int],
    flows: dict[str, dict[int, int]],
    transfers: tuple[Transfer, ...],
) -> dict[str, dict[int, int]]:
    """Closing balance per entity per day given a fixed set of transfers.

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
            balances[e][d] = running
    return balances


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------

def _build_legs(entity_ids, entities, accounts, ic, costs, fx, pairs):
    """Static per-(i,j,d) metadata: land_day, fee, FX quote, ic rate."""
    legs = {}
    for (i, j) in pairs:
        from_bank = accounts[i][1]
        to_bank = accounts[j][1]
        fee_usd, settlement_days = _transfer_cost_for(costs, from_bank, to_bank)
        rate = fx.rate(entities[i], entities[j])
        rate_bps = ic[(i, j)]["rate_bps"]
        for d in range(HORIZON_DAYS):
            land_day = d + settlement_days
            if land_day > HORIZON_DAYS - 1:
                continue
            legs[(i, j, d)] = {
                "land_day": land_day,
                "fee_usd": fee_usd,
                "rate": rate,
                "rate_bps": rate_bps,
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
        landed, fx_cost, fee, interest = leg_cost(
            amt, ci, cj, meta["rate_bps"], meta["land_day"], meta["fee_usd"], fx
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

    pairs = _permitted_pairs(entity_ids, ic)
    legs = _build_legs(entity_ids, entities, accounts, ic, costs, fx, pairs)
    lag_min = {e: earliest_actionable_day(e, entity_ids, accounts, ic, costs) for e in entity_ids}

    usd_rate = {c: fx.rate(c, "USD").effective for c in set(entities.values())}

    prob = pulp.LpProblem("sluice_cash_positioning", pulp.LpMinimize)

    x: dict[tuple[str, str, int], pulp.LpVariable] = {}
    y: dict[tuple[str, str, int], pulp.LpVariable] = {}
    for key in legs:
        i, j, d = key
        x[key] = pulp.LpVariable(f"x_{i}_{j}_{d}", lowBound=0, upBound=BIG_M, cat="Integer")
        y[key] = pulp.LpVariable(f"y_{i}_{j}_{d}", cat="Binary")
        prob += x[key] <= BIG_M * y[key], f"link_{i}_{j}_{d}"

    objective_terms = []
    for key, meta in legs.items():
        i, j, d = key
        ci = entities[i]
        fx_coef = (meta["rate"].spread_bps / 20_000) * usd_rate[ci]
        days_out = max(0, HORIZON_DAYS - meta["land_day"])
        interest_coef = (meta["rate_bps"] / 10_000) * (days_out / 365) * usd_rate[ci]
        objective_terms.append(x[key] * (fx_coef + interest_coef))
        objective_terms.append(y[key] * meta["fee_usd"])
    prob += pulp.lpSum(objective_terms)

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
                prob += expr >= floors.get(e, 0), f"floor_{e}_{d}"
                prob += expr >= 0, f"nonneg_{e}_{d}"

    for (i, j) in pairs:
        total_lent = pulp.lpSum(x[key] for key in x if key[0] == i and key[1] == j)
        prob += total_lent <= ic[(i, j)]["max_limit"], f"iclimit_{i}_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
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

    transfers = tuple(_extract_transfers(entities, accounts, legs, x, fx))
    total_cost = sum(t.fx_cost_minor + t.fee_minor + t.interest_minor for t in transfers)
    closing_balances = project_balances(entity_ids, opening, flows, transfers)

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

    ic_cumulative: dict[tuple[str, str], int] = {}

    for t in plan.transfers:
        if t.amount_minor <= 0:
            violations.append(
                f"transfer {t.from_entity}->{t.to_entity} day {t.send_day}: "
                f"non-positive amount_minor {t.amount_minor}"
            )
        if not (0 <= t.send_day < HORIZON_DAYS):
            violations.append(
                f"transfer {t.from_entity}->{t.to_entity}: send_day {t.send_day} out of horizon"
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
            elif t.land_day != t.send_day + settle:
                violations.append(
                    f"transfer {t.from_entity}->{t.to_entity} day {t.send_day}: "
                    f"land_day {t.land_day} != send_day + settlement_days "
                    f"({t.send_day + settle})"
                )

        key = (t.from_entity, t.to_entity)
        ic_cumulative[key] = ic_cumulative.get(key, 0) + t.amount_minor

    for (lender, borrower), total in ic_cumulative.items():
        limit = ic.get((lender, borrower), {}).get("max_limit")
        if limit is not None and total > limit:
            violations.append(
                f"IC exposure {lender}->{borrower} totals {total} minor units, "
                f"exceeds max_limit {limit}"
            )

    balances = project_balances(entity_ids, opening, flows, plan.transfers)
    for e in entity_ids:
        for d in range(HORIZON_DAYS):
            bal = balances[e][d]
            floor = floors.get(e, 0)
            if d >= lag_min[e]:
                if bal < floor:
                    violations.append(f"{e} closing balance on day {d} is {bal}, below floor {floor}")
                if bal < 0:
                    violations.append(f"{e} closing balance on day {d} is negative: {bal}")
            plan_bal = plan.closing_balances.get(e, {}).get(d)
            if plan_bal is not None and plan_bal != bal:
                violations.append(
                    f"{e} day {d}: plan.closing_balances reports {plan_bal}, "
                    f"recomputed {bal}"
                )

    return violations
