# Sluice

**Multi-entity cash positioning agent**
**Syndicate by Maximor — Track 2: Autonomous Office of the CFO**

## What it does

Every morning, a treasury analyst at a multi-entity company opens ~10 bank
portals, writes down balances, checks which subsidiaries are about to dip
below their required minimums, and decides which entity should send cash to
which — subject to loan covenants, intercompany lending limits, FX cost, and
settlement timing. It takes 1–2 hours daily, it's error-prone, and a mistake
means either a covenant breach or an overdraft.

Sluice automates that run end to end, including the cases where there is no
clean answer.

Remove the LLM and there is still a system: a ledger, a constraint model, and
a min-cost-flow solver that produces a verifiable optimal transfer plan. The
LLM does only what the solver cannot — diagnose *why* a plan is infeasible,
judge which remedies are actually worth putting in front of a human, write
the memo a CFO signs, and absorb human overrides as durable constraints.

`verify()` independently re-derives every balance, exposure and settlement
date from the database and returns a list of constraint violations. This runs
programmatically on every plan, not eyeballed — it returns `[]` on both
feasible scenarios below.

**One honest caveat:** covenant floors are enforced from each entity's
`earliest_actionable_day` — the first day a transfer sent today could
physically land, given the fastest permitted lender's settlement lag.
Balances before that day cannot be influenced by any plan built today, so
"zero violations" means zero violations from the first day a plan could act,
not from day zero.

## Verified results

| Scenario | Status | Transfers | Constraint violations | Solver cost | Naive baseline | Saved |
|---|---|---|---|---|---|---|
| `base` | OPTIMAL | 4 | **0** | 226,472 | 414,835 | **45.4%** |
| `covenant_shock` | OPTIMAL | 7 | **0** | 328,754 | 492,653 | **33.3%** |
| `infeasible` | INFEASIBLE | 0 | n/a | — | — | escalates |

Costs are USD minor units. Solve time: ~9s (`base`), ~20s (`covenant_shock`),
~0.2s (`infeasible`). The full infeasible chain, including LLM diagnosis and
the escalation memo, is ~75s — against a 1–2 hour manual process.

## How to run

```
pip install -r requirements.txt
cp .env.example .env        # add TensorMux + Neatlogs keys
python -m src.seed --scenario base
streamlit run src/app.py
```

Run from the repo root (cloning `sluice-app` already puts you there —
`requirements.txt` and `src/` are at the top level, there is no `code-base`
subdirectory to `cd` into). Modules sit flat in `src/` and import as
`src.<module>`. Scenarios are `base`, `covenant_shock`, and `infeasible`.

## The agent workflow, and how AO was used

Built through AO (Agent Orchestrator) from the start: 11 sessions on the
`code-base` project — 2 orchestrator, 9 worker — producing 5 merged PRs.

The build was deliberately shaped as a serial spine then a parallel fan-out:

- **Spine (sequential):** schema and data model → solver + naive baseline.
  Everything downstream consumes a `Plan` object, so fanning out before it
  existed would have meant four workers guessing at one interface. A frozen
  `PLAN_CONTRACT.md` was written before any parallel work started.
- **Fan-out (4 workers, simultaneous, isolated git worktrees):** A —
  diagnosis + ranked remedies; B — memo generation; C — Streamlit UI; D —
  Neatlogs tracing + metrics. Owned files were disjoint by design; the only
  file two workers touched was `requirements.txt`.

Workers ran in isolated worktrees and opened PRs. PR auto-review did not run:
AO could not resolve the repo because the git remote used an SSH host alias,
so no PR attached to a session until that was corrected. We did not run an
automated review loop on this project.

## What improved across iterations

1. **Silent LLM truncation.** GLM-4.7-Flash is a reasoning model: it fills a
   `reasoning` field and leaves `content` null until reasoning completes,
   burning ~450–500 tokens first. `llm.complete()` originally returned
   `content or ""`, so a truncated response was indistinguishable from a
   real empty answer — an empty string would have been rendered straight
   into a treasurer's memo. It now raises on `finish_reason == "length"`,
   and the default `max_tokens` went 2048 → 4096.
2. **Determinism needs more than temperature.** Temperature is pinned at 0,
   but two calls differing only in `max_tokens` returned different answers.
   Auditability requires pinning both.
3. **Context discipline.** The 32k window forced passing
   `positions.summarise()` (six rows) and only binding constraints rather
   than the raw 84-row projection. This measurably improved output: the
   model stopped narrating data and started explaining the decision.
4. **Remedy consolidation.** `diagnosis.diagnose()` originally returned 13
   remedies on the `infeasible` scenario — one row per (entity, remedy kind):
   6 revolver draws, 4 soft-covenant breaches, 3 delayed payables. That is an
   enumeration, not a judgement — a treasurer reading 13 options gets no more
   help than a spreadsheet gives them. It now returns 3 consolidated courses
   of action (one revolver remedy, one soft-covenant remedy, one
   delayed-payable remedy, each naming every entity it covers and rolling
   amounts up to USD), with a combination remedy left as an option for a
   scenario where no single lever closes the shortfall alone.

## Metrics

Cost saved vs. naive baseline · constraint violations (must be 0) ·
autonomy rate (runs completed without escalation) · time to plan vs. the
manual process · override recurrence (rejected remedies must never
resurface). `metrics.py` computes and persists these per run; `history()`
reads them back.

## Stack

Python 3.11 · SQLite · PuLP/CBC · Streamlit · TensorMux (GLM-4.7-Flash) ·
Neatlogs (tracing) · AO (build orchestration).
