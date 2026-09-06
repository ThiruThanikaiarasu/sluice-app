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

`verify()` independently re-derives every balance, exposure, settlement date
*and* per-leg FX/fee/interest cost from the database and the transfer legs,
and returns a list of constraint violations. This runs programmatically on
every plan, not eyeballed — it returns `[]` on both feasible scenarios below.
It does not re-run the solver itself, so it cannot prove the plan is
*optimal* — only that it is legal and that every number on it is real.

**Caveats, stated rather than hidden:**

- Covenant floors are enforced from each entity's `earliest_actionable_day`
  — the first day a transfer sent today could physically land, given the
  fastest permitted lender's settlement lag. Balances before that day cannot
  be influenced by any plan built today, so "zero violations" means zero
  violations from the first day a plan could act, not from day zero.
- Settlement now respects a business-day calendar: a wire is never initiated
  or landed on a weekend, and the horizon's calendar is checked against real
  dates rather than assumed to always fall on a business day (an earlier
  version scheduled UK payroll and two other cash events on a Saturday or
  Sunday — that was a data bug, now fixed). No holiday calendar is modelled
  yet, only weekends.
- The objective minimises total cost *including* intercompany interest, which
  is real to the paying entity but nets to zero on group consolidation. In
  `base`, most of the saving is real: FX + fees alone are down USD 325.10
  versus the naive baseline. In `covenant_shock`, they are not: the solved
  plan pays USD 52.90 *more* in FX + fees than the naive baseline once
  interest is stripped out, trading it for a much larger interest saving that
  is invisible to the consolidated group. The app now shows both numbers
  side by side rather than only the blended total.
- The solver drives each entity toward its floor only as far as a small
  soft-preference penalty allows; it is not free to leave zero headroom the
  way an earlier version was. It can still legally land exactly on a floor
  when cash is genuinely too scarce to do better — that is disclosed in
  `binding_constraints`, not hidden. The penalty is charged once per entity
  against a single shared slack variable sized to that entity's worst day,
  not once per entity per day — charging it daily would compound a
  persisting gap into a cost an order of magnitude larger than the real FX
  spread of actually moving cash to close it, which stops being a tie-break
  and starts being a second objective the solver optimises against.

## Verified results

| Scenario | Status | Transfers | Constraint violations | Solver cost | Naive baseline | Saved (total) | Saved (FX+fees only) |
|---|---|---|---|---|---|---|---|
| `base` | OPTIMAL | 4 | **0** | 2,414.55 | 4,015.24 | **39.9%** | USD 325.10 |
| `covenant_shock` | OPTIMAL | 6 | **0** | 3,886.84 | 5,214.92 | **25.5%** | −USD 52.90 |
| `infeasible` | INFEASIBLE | 0 | n/a | — | — | escalates | — |

Costs are USD. Solve time: ~6s (`base`), ~4s (`covenant_shock`), ~0.1s
(`infeasible`). The full infeasible chain, including LLM diagnosis and the
escalation memo, is ~75s — against a 1–2 hour manual process. Numbers above
differ from an earlier revision for two reasons: the business-day fix pushes
some transfers to a later, more expensive settlement leg, and an earlier cut
of the covenant buffer charged its notional penalty once per entity *per
day* rather than once per entity — both are the plan getting more honest,
not less optimal for the same problem.

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
   help than a spreadsheet gives them. It now returns at most 3 consolidated
   courses of action (one revolver remedy, one soft-covenant remedy, one
   delayed-payable remedy), each naming every entity it covers and rolling
   amounts up to USD. Rejecting a course excludes every entity it names from
   that remedy kind on the next diagnosis — not the exact consolidated
   string, which would stop applying the moment any other entity's shortfall
   changed — and every Approve/Reject click is now persisted to
   `decision_log` instead of only producing a toast that disappears on
   rerun.
5. **Revolver draws now point at a real facility.** Earlier, "draw on the
   revolver" named no lender, no limit, no rate — a placeholder string a bank
   would never accept. `revolver_facility` now seeds one named, priced,
   committed facility per entity, and the remedy only offers a facility whose
   currency matches the shortfall and caps the draw at that facility's
   committed limit, rather than assuming an infinite, undocumented backstop.
6. **`.env` was never actually loaded outside the test suite.** `llm.py` and
   `tracing.py` read `os.environ` directly; only `test_diagnosis.py` called
   `load_dotenv()` first. Every real run of the Streamlit app (not the tests)
   failed with `SLUICE_LLM_API_KEY is unset` the moment it needed the LLM —
   the escalation and memo screens, the actual point of the demo, never ran.
   `app.py` now loads `.env` before anything else.

## Metrics

Cost saved vs. naive baseline · constraint violations (must be 0, or N/A when
the plan escalated instead of solving) · autonomy rate (runs completed
without escalation) · time to plan vs. the manual process · override
recurrence (rejected remedies must never resurface). `metrics.py` computes
and persists these per run; `app.py` now calls `measure()`/`persist()` on
every solve and shows `history()` in an expander, instead of only being
reachable via `python -m src.metrics`. Neatlogs tracing (`tracing.py`) is
initialised the same way — on app start, if `NEATLOGS_API_KEY` is set — and
is skipped without crashing the demo if it isn't; the Metrics panel says
which happened.

## Stack

Python 3.11 · SQLite · PuLP/CBC · Streamlit · TensorMux (GLM-4.7-Flash) ·
Neatlogs (tracing) · AO (build orchestration).
