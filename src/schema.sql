-- Sluice: multi-entity treasury cash positioning.
--
-- Monetary amounts are integer minor units (cents) throughout. A constraint
-- solver working in floats produces plans that are wrong by a cent, and a plan
-- that is wrong by a cent is a plan a bank rejects.

PRAGMA foreign_keys = ON;

CREATE TABLE entity (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    country             TEXT NOT NULL,
    functional_currency TEXT NOT NULL
);

CREATE TABLE bank_account (
    id        TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entity(id),
    bank      TEXT NOT NULL,
    currency  TEXT NOT NULL,
    balance   INTEGER NOT NULL,           -- minor units, at horizon day 0
    purpose   TEXT
);

-- A floor on an entity's total cash.
--
-- `hardness` is what the remedy ranker sorts on when the solver comes back
-- infeasible. A 'hard' floor is contractual and may never be breached at any
-- price; a 'soft' floor is internal policy and may be relaxed with treasurer
-- sign-off. Collapsing these two into one number is what makes naive tooling
-- dangerous: it either treats policy as inviolable and declares false
-- emergencies, or treats covenants as advisory and recommends a default.
CREATE TABLE covenant (
    id           INTEGER PRIMARY KEY,
    entity_id    TEXT NOT NULL REFERENCES entity(id),
    kind         TEXT NOT NULL,
    threshold    INTEGER NOT NULL,
    currency     TEXT NOT NULL,
    hardness     TEXT NOT NULL CHECK (hardness IN ('hard', 'soft')),
    source_doc   TEXT,
    source_quote TEXT
);

CREATE TABLE cash_forecast (
    entity_id TEXT NOT NULL REFERENCES entity(id),
    day       INTEGER NOT NULL,           -- 0-indexed from horizon start
    date      TEXT NOT NULL,
    net_flow  INTEGER NOT NULL,           -- minor units, functional currency
    note      TEXT,
    PRIMARY KEY (entity_id, day)
);

-- Who may lend to whom, and how much.
--
-- Rows with permitted = 0 carry the legal reason in `reason`. These are the
-- rows that make "just fund everything from the parent" wrong, and the reason
-- text is what the approval memo quotes back to the treasurer.
CREATE TABLE ic_agreement (
    lender_id   TEXT NOT NULL REFERENCES entity(id),
    borrower_id TEXT NOT NULL REFERENCES entity(id),
    max_limit   INTEGER NOT NULL,         -- minor units, lender currency
    rate_bps    INTEGER NOT NULL,         -- annual, basis points
    permitted   INTEGER NOT NULL CHECK (permitted IN (0, 1)),
    reason      TEXT,
    PRIMARY KEY (lender_id, borrower_id)
);

CREATE TABLE fx_rate (
    base       TEXT NOT NULL,
    quote      TEXT NOT NULL,
    mid        REAL NOT NULL,
    spread_bps INTEGER NOT NULL,
    PRIMARY KEY (base, quote)
);

CREATE TABLE transfer_cost (
    from_bank       TEXT NOT NULL,
    to_bank         TEXT NOT NULL,
    fixed_fee       INTEGER NOT NULL,     -- minor units USD
    settlement_days INTEGER NOT NULL,
    PRIMARY KEY (from_bank, to_bank)
);

-- Constraints derived from human overrides, injected into later solves so a
-- rejected remedy never resurfaces.
CREATE TABLE learned_rule (
    id              INTEGER PRIMARY KEY,
    origin_run_id   TEXT,
    rule_text       TEXT NOT NULL,
    constraint_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
