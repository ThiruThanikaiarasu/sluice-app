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
  be *improved* by any plan built today, so "zero violations" means zero
  violations from the first day a plan could act, not from day zero. That
  exemption only ever covers what a plan cannot influence: an entity's own
  outbound sends are within the plan's control from day 0, and both the
  solver and `verify()` now hold a plan to its baseline (zero-transfer)
  balance on those early days, not just to the floor — a plan may never use
  the exemption to leave an entity worse off than doing nothing at all. The
  UI's headline "Constraint violations" and "Cost saved vs. baseline" tiles
  now carry this same qualifier and a pointer to the FX-only figure, instead
  of only being explained here.
- Settlement now respects a business-day *and* public-holiday calendar (via
  the `holidays` package, keyed off each entity's `country`): a wire is never
  initiated or landed on a weekend or a bank holiday in either the sending or
  receiving country, and the horizon's calendar is checked against real
  dates rather than assumed to always fall on a business day (an earlier
  version scheduled UK payroll and two other cash events on a Saturday or
  Sunday — that was a data bug, now fixed). Weekends-only was itself a gap:
  a wire between Ireland and Germany scheduled to land on 26 December would
  not actually land — now fixed the same way the weekend bug was.
- Covenants here mean minimum-cash floors only, and they are tested
  continuously (daily), which is a real, common covenant type and matches
  the actual facility text seeded for each entity (e.g. "Unrestricted Cash
  at no time less than €2,000,000"). Leverage, DSCR, interest-cover, and
  other ratio covenants — typically tested at period end against
  consolidated financials, not daily — are not represented at all. This is
  a scope limit, not a hidden one: the UI now says so next to the floor
  table it applies to.
- The solve is now lexicographic, not one blended objective: phase 1
  minimises real group cost (FX spread + wire fees, on both the draw and any
  repayment) and phase 2, holding that real cost at its optimum, minimises
  intercompany interest as a tie-break only. Interest is real to the paying
  entity but nets to zero on group consolidation, so it can no longer buy
  its way into "optimal" by trading away real FX/fee cost the way a single
  blended objective allowed. `total_cost_minor` (what's reported and what
  the headline "Cost saved vs. baseline" tile shows) still adds interest
  back in for disclosure, so it is not the same number phase 1 optimised —
  the app's "Group-consolidated saving" figure, not the headline tile, is
  the real cash the group saves. In `base` that's USD 395.10 of the total
  saving; in `covenant_shock` the solved plan actually pays USD 26.26 *more*
  in FX + fees than the naive baseline once interest is stripped out, buying
  a much larger interest saving that is invisible to the consolidated group.
  (An earlier build tried a single MILP with a heavily-weighted combined
  objective instead of two solves, to avoid solving twice -- reverted: a
  weight large enough to guarantee the ordering spans ~9 orders of magnitude
  against the interest terms, and that coefficient spread pushed CBC's
  solve time on `covenant_shock` from ~2.5s past a minute. Two
  well-conditioned solves is faster in practice than one badly-conditioned
  one.)
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
- The solver is a MILP with one binary leg-activation variable per
  (entity-pair, day): ~300 binaries at 6 entities over the 14-day horizon,
  solved by CBC in ~6s. That does not linearly extend to the ~10 banks the
  pitch describes — 20 entities is roughly 5,300 binaries, well past where
  CBC solves comfortably in a demo. Scaling past this would mean decomposing
  by entity cluster, warm-starting from the previous day's plan, or moving
  to a commercial MILP solver; none of that is implemented yet.
- Every intercompany loan now has a real maturity (`ic_agreement.term_days`,
  a single disclosed 7-day assumption for every pair -- there is no
  per-pair contractual term in the seed data to draw a more granular number
  from) and, when that maturity and its own settlement lag both land inside
  the 14-day horizon, a mandatory repayment leg: the borrower pays principal
  + accrued interest back to the lender, converting at its own cost the same
  way the forward draw does. `verify()` independently re-derives whether
  each loan should have repaid and checks the repayment's own day and
  amount, not just that one exists. The app and memo show which loans repaid
  within the horizon and which remain outstanding beyond it, so this isn't
  only visible by reading `interest_minor` and inferring it.
  Two things are still simplified, on purpose: the naive baseline does not
  model repayment at all (see the baseline note below), and neither seeded
  scenario's chosen routes happen to mature in time for a repayment to
  actually appear on screen -- the mechanism is exercised by dedicated
  tests instead. A shorter term was tried specifically to make it visible
  in `base`/`covenant_shock` and reverted (see below).
- The naive baseline still prices every loan as if it stays open through
  the whole horizon, even though it carries the same real maturity a
  solver-drawn loan would. This was tried the honest way -- pricing naive's
  loans against their real term and scheduling their repayment too -- and
  reverted: naive sizes each draw to cover only the *peak* shortfall, with
  no mechanism to also plan for paying that draw back, and once repayment
  was priced in, naive's own repayment pushed at least one entity below its
  floor and made the naive baseline itself INFEASIBLE in a seeded scenario.
  That is a real result (naive treasury practice really can create a
  repayment crisis it never saw coming) but a materially bigger change than
  a repayment-scheduling fix should carry — it needs either a naive baseline
  that also prices its own repayment risk or a loan-rollover mechanism,
  and neither is implemented. Until then, naive's cost is a real
  understatement, and the solved plan's advantage over it is, if anything,
  larger than the numbers below show.
- Explicitly out of scope, not modelled at all: withholding tax and
  transfer-pricing/arm's-length documentation on cross-border intercompany
  interest; thin-capitalisation limits; bank cut-off times (a transfer
  landing on "day 2" assumes it clears any time that day, not before a
  specific cutoff); and forecast uncertainty (`cash_forecast` is a single
  deterministic point estimate, so a plan reporting "USD 0.00 headroom" on
  some day states a precision no cash forecast actually has).
- The design moves cash via discrete intercompany wires rather than a
  notional pool or a physical header-account sweep, which is the more
  common mechanism at this group size in practice. Wires were chosen here
  because each one is a fully auditable, individually-costed, individually
  approvable transaction — the same property that makes `verify()` and the
  approval memo possible per-transfer. A pooling structure would need a
  different model entirely (participation agreements, an interest
  set-off formula, daily sweep timing) and was out of scope for this build.

## Verified results

| Scenario | Status | Transfers | Constraint violations | Solver cost | Naive baseline | Saved (total) | Saved (FX+fees only) |
|---|---|---|---|---|---|---|---|
| `base` | OPTIMAL | 2 | **0** | 2,616.04 | 3,756.83 | **30.4%** | USD 395.10 |
| `covenant_shock` | OPTIMAL | 4 | **0** | 4,259.47 | 4,878.45 | **12.7%** | −USD 26.26 |
| `infeasible` | INFEASIBLE | 0 | n/a | — | — | escalates | — |

Costs are USD. Solve time: ~0.4s (`base`), ~2.5s (`covenant_shock`), ~0.1s
(`infeasible`). The full infeasible chain, including LLM diagnosis and the
escalation memo, is ~75s — against a 1–2 hour manual process. Numbers above
differ from an earlier revision for several reasons, all the plan getting
more honest rather than less optimal for the same problem: the business-day
fix pushed some transfers to a later, more expensive settlement leg; an
earlier cut of the covenant buffer charged its notional penalty once per
entity *per day* rather than once per entity; and the objective is now
lexicographic (real FX+fee cost first, intercompany interest only as a
tie-break) with a real per-loan holiday-aware settlement calendar and a
real loan maturity, all of which changes which combination of legs is
actually cheapest.

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
