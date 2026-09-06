"""Naive baseline: fund every shortfall from the US parent.

Ignores FX cost and routing efficiency -- it always routes through MER-US
regardless of whether a cheaper intercompany path exists -- but it still
respects prohibited pairs and settlement lag, and still uses the real cost
math in `solver.leg_cost`, so its `total_cost_minor` is genuinely comparable
to the solved plan's. The delta between the two is the headline savings
number.
"""

from __future__ import annotations

import sqlite3
import time

from .fx import FXTable
from .models import HORIZON_DAYS
from .positions import binding_floors, opening_balances, shortfalls
from .solver import (
    Plan,
    Transfer,
    _entities,
    _ic_agreements,
    _net_flows,
    _transfer_cost_for,
    _transfer_costs,
    earliest_actionable_day,
    leg_cost,
    operating_accounts,
    project_balances,
)

FUNDER = "MER-US"


def naive_plan(conn: sqlite3.Connection, scenario: str) -> Plan:
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

    from_bank = accounts[FUNDER][1]
    ci = entities[FUNDER]

    transfers: list[Transfer] = []
    for entity_id, summary in shortfalls(conn).items():
        if entity_id == FUNDER:
            continue
        agreement = ic.get((FUNDER, entity_id))
        if not agreement or not agreement["permitted"]:
            # Naive still refuses to break the law; it just has no fallback.
            continue

        to_bank = accounts[entity_id][1]
        fee_usd, settlement_days = _transfer_cost_for(costs, from_bank, to_bank)
        cj = entities[entity_id]
        rate_bps = agreement["rate_bps"]

        quote = fx.rate(ci, cj)
        # Fund the full peak shortfall with a margin against rounding, sent
        # as early as possible so it lands well before the need materialises.
        needed_minor = summary.peak_shortfall
        amount_minor = int(needed_minor / quote.effective) + 1
        # Naive still has to obey the intercompany limit -- funding past it
        # would make the "naive" plan itself illegal, not just expensive.
        amount_minor = min(amount_minor, agreement["max_limit"])
        if amount_minor <= 0:
            continue

        send_day = 0
        land_day = send_day + settlement_days
        if land_day > HORIZON_DAYS - 1:
            continue

        landed, fx_cost, fee, interest = leg_cost(
            amount_minor, ci, cj, rate_bps, land_day, fee_usd, fx
        )
        transfers.append(Transfer(
            from_entity=FUNDER,
            from_account=accounts[FUNDER][0],
            to_entity=entity_id,
            to_account=accounts[entity_id][0],
            send_day=send_day,
            land_day=land_day,
            amount_minor=amount_minor,
            amount_currency=ci,
            landed_minor=landed,
            landed_currency=cj,
            fx_cost_minor=fx_cost,
            fee_minor=fee,
            interest_minor=interest,
            rationale=(
                f"naive: fund {entity_id}'s entire peak shortfall from "
                f"{FUNDER} on day {send_day}, ignoring routing cost"
            ),
        ))

    transfers.sort(key=lambda t: (t.send_day, t.from_entity, t.to_entity))
    closing_balances = project_balances(entity_ids, opening, flows, tuple(transfers))

    lag_min = {e: earliest_actionable_day(e, entity_ids, accounts, ic, costs) for e in entity_ids}
    breach = False
    binding: list[str] = []
    for e in entity_ids:
        floor = floors.get(e, 0)
        for d in range(lag_min[e], HORIZON_DAYS):
            if closing_balances[e][d] < floor:
                breach = True
                binding.append(f"{e} breaches its floor on day {d} under the naive plan")

    solve_seconds = time.monotonic() - start

    if breach:
        return Plan(
            status="INFEASIBLE",
            scenario=scenario,
            transfers=(),
            total_cost_minor=0,
            solve_seconds=solve_seconds,
            binding_constraints=tuple(binding),
            closing_balances={},
        )

    total_cost = sum(t.fx_cost_minor + t.fee_minor + t.interest_minor for t in transfers)
    return Plan(
        status="OPTIMAL",
        scenario=scenario,
        transfers=tuple(transfers),
        total_cost_minor=total_cost,
        solve_seconds=solve_seconds,
        binding_constraints=(),
        closing_balances=closing_balances,
    )
