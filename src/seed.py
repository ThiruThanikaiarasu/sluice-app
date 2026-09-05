"""Seeds Meridian Systems, a fictional six-entity group.

Three scenarios, all cut from the same company:

  base            The UK and Germany both run short. Solvable, and the cheap
                  route runs through Ireland, which is sitting on idle cash.
  covenant_shock  Ireland's term loan floor is amended upward to EUR 6.5m. The
                  group's largest lender becomes a borrower and the plan has to
                  reroute through the US at higher FX cost.
  infeasible      Covenant shock, plus a far larger German VAT assessment and a
                  US collection that slips past the horizon. No lawful set of
                  transfers covers every shortfall.

    python -m sluice.seed --scenario base
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .db import DEFAULT_DB_PATH, init_db
from .models import HARD, HORIZON_DAYS, SOFT, to_minor

SCENARIOS = ("base", "covenant_shock", "infeasible")
HORIZON_START = date(2026, 9, 7)

POLICY = "Group Treasury Policy v4.1"
BOI_FACILITY = "BOI Facility Agreement dated 2024-03-15, Clause 21.2"
BOI_AMENDED = ("BOI Facility Agreement dated 2024-03-15, as amended by "
               "Amendment No.2 dated 2026-09-01, Clause 21.2")

ENTITIES = [
    ("MER-US", "Meridian Systems Inc.", "US", "USD"),
    ("MER-UK", "Meridian Systems Ltd.", "GB", "GBP"),
    ("MER-DE", "Meridian Systems GmbH", "DE", "EUR"),
    ("MER-IE", "Meridian Technologies Ireland Ltd.", "IE", "EUR"),
    ("MER-SG", "Meridian APAC Pte. Ltd.", "SG", "USD"),
    ("MER-CA", "Meridian Canada Inc.", "CA", "USD"),
]

ACCOUNTS = [
    ("ACC-US-01", "MER-US", "JPMorgan", "USD", 3_200_000, "operating"),
    ("ACC-US-02", "MER-US", "JPMorgan", "USD", 900_000, "payroll"),
    ("ACC-UK-01", "MER-UK", "Barclays", "GBP", 1_950_000, "operating"),
    ("ACC-UK-02", "MER-UK", "HSBC", "GBP", 320_000, "collections"),
    ("ACC-DE-01", "MER-DE", "Deutsche Bank", "EUR", 2_800_000, "operating"),
    ("ACC-DE-02", "MER-DE", "Deutsche Bank", "EUR", 450_000, "tax reserve"),
    ("ACC-IE-01", "MER-IE", "Bank of Ireland", "EUR", 6_300_000, "operating"),
    ("ACC-SG-01", "MER-SG", "DBS", "USD", 1_750_000, "operating"),
    ("ACC-SG-02", "MER-SG", "DBS", "USD", 640_000, "collections"),
    ("ACC-CA-01", "MER-CA", "RBC", "USD", 890_000, "operating"),
]

BANKS = ["JPMorgan", "Barclays", "HSBC", "Deutsche Bank",
         "Bank of Ireland", "DBS", "RBC"]
SAME_COUNTRY = [{"JPMorgan", "RBC"}, {"Barclays", "HSBC"}]

COVENANTS = [
    ("MER-IE", "term_loan_minimum", 2_000_000, "EUR", HARD, BOI_FACILITY,
     "The Borrower shall procure that Unrestricted Cash is at no time less "
     "than EUR 2,000,000."),
    ("MER-UK", "overdraft_facility_minimum", 250_000, "GBP", HARD,
     "Barclays Overdraft Facility Letter dated 2025-01-08, Schedule 2",
     "The Customer shall maintain a cleared credit balance of not less than "
     "GBP 250,000 across all Accounts."),
    ("MER-US", "internal_policy_minimum", 1_000_000, "USD", SOFT, POLICY,
     "Operating entities shall hold not less than 30 days of forecast opex."),
    ("MER-DE", "internal_policy_minimum", 300_000, "EUR", SOFT, POLICY, None),
    ("MER-SG", "internal_policy_minimum", 200_000, "USD", SOFT, POLICY, None),
    ("MER-CA", "internal_policy_minimum", 150_000, "USD", SOFT, POLICY, None),
]

# Everything not listed here is a permitted lending pair.
PROHIBITED = {
    ("MER-IE", "MER-US"): "Upstream loan from a CFC to its US parent would be "
                          "a deemed dividend under IRC s.956.",
    ("MER-SG", "MER-DE"): "Intercompany lending suspended pending IRAS review "
                          "of the APAC transfer pricing arrangement.",
    ("MER-CA", "MER-UK"): "No intercompany loan agreement in place.",
    ("MER-CA", "MER-DE"): "No intercompany loan agreement in place.",
    ("MER-CA", "MER-IE"): "No intercompany loan agreement in place.",
    ("MER-CA", "MER-SG"): "No intercompany loan agreement in place.",
}

# lender -> (limit in lender currency, annual rate bps)
LENDING_CAPACITY = {
    "MER-US": (10_000_000, 500),
    "MER-IE": (5_000_000, 450),
    "MER-DE": (3_000_000, 425),
    "MER-UK": (2_000_000, 475),
    "MER-SG": (2_000_000, 500),
    "MER-CA": (1_000_000, 500),
}

FX_RATES = [
    ("EUR", "USD", 1.0850, 8),
    ("GBP", "USD", 1.2650, 10),
    ("EUR", "GBP", 0.8577, 15),
]

# entity -> (daily baseline flow, {day: (one-off amount, description)})
FLOWS = {
    "MER-US": (-120_000, {
        3: (2_500_000, "Enterprise renewal collection - Northwind"),
        10: (-1_200_000, "Federal estimated tax payment"),
    }),
    "MER-UK": (-45_000, {
        5: (-2_100_000, "Monthly payroll incl. H1 bonus accrual"),
    }),
    "MER-DE": (-60_000, {
        8: (-3_200_000, "Quarterly VAT settlement"),
        12: (400_000, "Distributor receipt - Kellner GmbH"),
    }),
    "MER-IE": (95_000, {
        6: (-800_000, "IP amortisation true-up payment"),
    }),
    "MER-SG": (-38_000, {
        9: (-600_000, "APAC contractor settlement"),
    }),
    "MER-CA": (-22_000, {
        4: (-350_000, "Office lease prepayment"),
    }),
}


def _covenants(scenario: str) -> list[tuple]:
    rows = [list(c) for c in COVENANTS]
    if scenario in ("covenant_shock", "infeasible"):
        for row in rows:
            if row[0] == "MER-IE":
                row[2] = 6_500_000
                row[5] = BOI_AMENDED
                row[6] = ("The Borrower shall procure that Unrestricted Cash "
                          "is at no time less than EUR 6,500,000.")
    return [tuple(r) for r in rows]


def _flows(scenario: str) -> dict:
    flows = {e: (base, dict(spikes)) for e, (base, spikes) in FLOWS.items()}
    if scenario == "infeasible":
        flows["MER-DE"][1][8] = (
            -5_800_000,
            "Quarterly VAT settlement incl. prior-period assessment",
        )
        del flows["MER-US"][1][3]
    return flows


def _transfer_costs() -> list[tuple]:
    rows = []
    for src in BANKS:
        for dst in BANKS:
            if src == dst:
                fee, days = 0, 0
            elif any({src, dst} == pair for pair in SAME_COUNTRY):
                fee, days = 15, 1
            else:
                fee, days = 35, 2
            rows.append((src, dst, to_minor(fee), days))
    return rows


def _ic_agreements() -> list[tuple]:
    rows = []
    for lender, (limit, rate) in LENDING_CAPACITY.items():
        for borrower, *_ in ENTITIES:
            if lender == borrower:
                continue
            reason = PROHIBITED.get((lender, borrower))
            rows.append((lender, borrower, to_minor(limit), rate,
                         0 if reason else 1, reason))
    return rows


def _forecast(scenario: str) -> list[tuple]:
    rows = []
    for entity_id, (baseline, spikes) in _flows(scenario).items():
        for day in range(HORIZON_DAYS):
            amount, note = spikes.get(day, (0, None))
            rows.append((
                entity_id,
                day,
                (HORIZON_START + timedelta(days=day)).isoformat(),
                to_minor(baseline + amount),
                note,
            ))
    return rows


def seed(scenario: str = "base",
         db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected {SCENARIOS}")

    conn = init_db(db_path)
    conn.executemany("INSERT INTO entity VALUES (?, ?, ?, ?)", ENTITIES)
    conn.executemany(
        "INSERT INTO bank_account VALUES (?, ?, ?, ?, ?, ?)",
        [(i, e, b, c, to_minor(bal), p) for i, e, b, c, bal, p in ACCOUNTS],
    )
    conn.executemany(
        "INSERT INTO covenant (entity_id, kind, threshold, currency, hardness,"
        " source_doc, source_quote) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(e, k, to_minor(t), c, h, d, q)
         for e, k, t, c, h, d, q in _covenants(scenario)],
    )
    conn.executemany("INSERT INTO ic_agreement VALUES (?, ?, ?, ?, ?, ?)",
                     _ic_agreements())
    conn.executemany("INSERT INTO fx_rate VALUES (?, ?, ?, ?)", FX_RATES)
    conn.executemany("INSERT INTO transfer_cost VALUES (?, ?, ?, ?)",
                     _transfer_costs())
    conn.executemany("INSERT INTO cash_forecast VALUES (?, ?, ?, ?, ?)",
                     _forecast(scenario))
    conn.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [("scenario", scenario),
         ("horizon_start", HORIZON_START.isoformat()),
         ("horizon_days", str(HORIZON_DAYS))],
    )
    conn.commit()
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the Meridian Systems group.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="base")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    conn = seed(args.scenario, args.db)
    print(f"Seeded {args.scenario!r} -> {args.db}")
    for table in ("entity", "bank_account", "covenant", "ic_agreement",
                  "cash_forecast", "fx_rate", "transfer_cost"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<16}{n:>5}")


if __name__ == "__main__":
    main()
