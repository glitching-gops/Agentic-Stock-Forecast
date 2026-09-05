# ZeRO Agentic Stock Forecast

> A forecasting system for NSE equities built around its own evaluation harness:
> purged walk-forward validation, pre-registered success criteria, cost-aware
> portfolio simulation, and conformal prediction intervals. It publishes a
> 30-session forecast for every name in a frozen 84-stock universe, each carrying
> the held-out evidence behind it.
>
> Six phases of measurement say the point forecast has no edge. That finding is
> the headline result, and the apparatus that produced it is the project.

[![API](https://img.shields.io/badge/API-Render-46E3B7?style=flat&logo=render)](https://agentic-stock-forecast.onrender.com/api/health)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python)](https://python.org)

---

## What Was Built

The forecasting model is ordinary. The measurement apparatus around it is not,
and it is what caught every result this project has had to retire.

| Component | What it is |
|---|---|
| **Purged panel walk-forward** | Splits the shared **date** grid, not row positions. Training rows whose h-session label reaches into the test window are purged; a further h sessions are embargoed. `PurgedWalkForward` (one series) and `PurgedPanelWalkForward` (a panel) are separate types, because a positional split on a panel lands on a different calendar date for every ticker |
| **Two floors, not zero** | 57.67% of 30-session returns on this universe are positive and ~33% of their variance is common across names, so a zero forecast is beaten by drift before a model opens its eyes. Every comparator is scored against `market` (MAE) and `beta_market` (rank IC) — the latter sorts by beta and holds no company-specific view |
| **Cost-aware simulator** | The measured Zerodha/NSE delivery stack — STT 0.1% **on both sides**, stamp duty, exchange, SEBI, GST — at **0.2225% round trip**. Turnover is computed from the holdings that actually changed, name by name |
| **Deflated Sharpe** | Bailey & López de Prado. The expected maximum Sharpe under the null grows with the number of configurations tried, and this project has tried 103 |
| **Conformal intervals** | Split-conformal, calibrated on out-of-sample residuals. Realised coverage is measured against the claim, not assumed |
| **Experiment tracking** | `config_hash` and `data_hash` kept deliberately separate, so a metric that moves while one is constant says whether code or data caused it |
| **Validation gate** | Eight checks between the last write and the first publish. FAIL aborts the run, WARN records — a gate that fails on everything gets switched off, one that fails on nothing is decoration |
| **Mutation-tested guards** | Every new guard is verified by restoring the defect it prevents and confirming the suite goes red. Several tests in this repo were found toothless that way |

Two independent compute cadences run on GitHub Actions, never on the web
backend — an in-process scheduler on the serving instance is what OOM-killed the
first production deployment.

---

## The Result

**The point forecast does not beat its baselines, at any stage of six phases of
attack.** The pre-registered bar was fixed in advance: *a comparator succeeds if
its rebalance rank IC is positive with t > 2.*

Panel comparators on the absolute 30-session return. One run, one panel — 116
tickers, 64 non-overlapping rebalances, 5 purged folds — because mixing figures
from runs over different universes is one of the errors this project has already
made and now guards against:

| comparator | reb IC | reb t | MAE | clears floor |
|---|---|---|---|---|
| **`beta_market`** (floor) | **+0.0464** | +1.51 | 0.08970 | — |
| `pooled_xgb` | +0.0407 | **+2.55** | 0.09376 | **no** |
| `regime_factor` | +0.0277 | +1.15 | 0.08974 | **no** |
| `linear_factor` | +0.0246 | +1.25 | 0.08966 | **no** |
| `news_factor` | +0.0077 | +0.43 | 0.08986 | **no** |
| **`market`** (floor) | — | — | **0.08944** | — |

`pooled_xgb` reaches t +2.55 and still clears nothing: it ranks *below* a beta
sort and is 4.8% worse than a constant forecast on MAE. It also fails the
standing checks — its edge lives in the earliest fold and falls below the bar
at four of six `min_train` settings.

`beta_market` — which sorts by beta and holds no view about any company —
out-ranks everything. Also scored and closed: six foundation-model
configurations zero-shot (Chronos-2, TimesFM-2.5, Kronos), a linear probe on
frozen embeddings, LoRA fine-tuning nested inside each fold, point-in-time
valuation, and a 44,461-article dated news archive.

**Three results cleared the bar and all three were retired**, each by a check
the project ran on itself: valuation at t +3.32 (a maximum over an unstable
`min_train` grid — +1.00 at the default), LoRA at t +2.37 (entirely in the two
earliest folds, negative in the most recent), and a regime-conditional split
that reached **t +5.21** at one cell whose immediate neighbours on the same rows
read +1.78 and +1.21.

### What holds up

- **Conformal coverage: 80.1% against a nominal 80%**, measured rather than
  assumed. The uncertainty quantification works even though the point forecast
  does not.
- **Break-even rank IC**, the most reusable output: an ordering needs **0.0051**
  at zero market impact, rising to **0.0282** at 50bp, just to pay for itself.
  This converts every statistic here into an economic statement.
- **The null is not beta.** The obvious reading — that these orderings are a
  disguised beta tilt — was tested directly by removing the beta channel from
  the target within each date. `beta_market` loses all of its ordering
  (+0.0464 → −0.0045) while `pooled_xgb` keeps essentially all of its own —
  105% of its raw rank IC survives. They are close to independent, and the beta
  channel explains only 6.5% of within-date variance. The models were never tracking beta. They are weak on their own
  account, which is the worse of the two explanations.

Held beta-neutral and net of measured costs, the best book reaches a Sharpe of
0.79 and deflates to **−1.03** against a 1.96 threshold once all 103 trials this
panel has been asked are counted — below the maximum expected from luck alone.

Full detail, with every table and the reasoning: **[/research](https://agentic-stock-forecast.vercel.app/research)**.

---

## Why the Earlier Numbers Were Wrong

The first version of this README reported **~4.3% MAPE and ~85% directional
accuracy**. Those were artifacts. An audit (findings F1–F15) found three
compounding defects:

1. **The reported metric was fitted and scored on the same rows.** A Ridge
   meta-learner was trained on the validation set, then scored on it.
2. **Hyperparameters were tuned on the test slice.** Optuna received the full
   labelled set, including the 15% reported as held out.
3. **The universe was selected on those metrics** — top 5 per sector by a
   composite score largely composed of the leaked figures.

Underneath sat three silent functional bugs: the LSTM never wrote a checkpoint
(so the advertised ensemble never ran), the warm-path forecast raised `KeyError`
and persisted today's price as the forecast, and training labels were never
backfilled.

All of it is fixed, and `tests/test_leakage.py` is the suite whose absence let it
ship. The universe is now **frozen at 84 tickers selected on data quality
only** — never on measured skill — and the ranking layer was deleted outright:
ranking 84 names needs 84 comparable numbers, and the evidence gate produces 3,
which is what chance produces.

---

## Architecture

```text
              DAILY (GitHub Actions, weekdays 18:30 IST)
frozen universe ──► OHLCV ──► 24 signals ──► per-ticker forecast
                                             XGBoost, CACHED hyperparameters
                                             conformal interval + P(up)
                                             LLM narrative (no numbers)
                                                  │
                                                  ▼
                                          Critic Agent
                          deterministic evidence gate (STRONG/WEAK/INSUFFICIENT)
                                                  │
                                                  ▼
                                       Supabase PostgreSQL
                                                  ▲
                                                  │ reads evaluation + calibration
              WEEKLY (GitHub Actions, Saturday 08:30 IST)
              purged walk-forward · nested Optuna · conformal calibration
              news scoring · panel comparator table
                                                  │
                                                  ▼
              Supabase ──► FastAPI (Render, reads only) ──► Next.js (Vercel, ISR)
```

Render serves reads and never runs scheduled compute. An evidence grade can be
up to a week older than the price attached to it; that gap is published as
`evaluated_at` rather than hidden.

**The critic's LLM review was retired on measurement.** Audited over all 1,152
forecast rows ever written, it ran on 38, raised flags on 172, and changed
**zero**. It was not unhelpful — it was structurally incapable of helping, since
its only channel to a published row required a grade no row has ever received.

---

## Evaluation Methodology

| Control | What it does |
|---|---|
| **Purging** | Training rows whose 30-session label reaches into the test window are removed |
| **Embargo** | A further 30 sessions are dropped, widening the gap |
| **Nested tuning** | Hyperparameters are searched inside each training fold only |
| **Non-overlapping rebalances** | Consecutive dates share 29 of 30 forward sessions, so ~1,900 dates hold ~64 independent windows. `reb_t` is computed on those alone; a naive t is inflated ~5× |
| **`daily_IC` vs `reb_t` kept apart** | They describe different samples and can carry opposite signs. Only the non-overlapping one supports inference |
| **Floors, not zero** | Every comparator is graded `clears_floor` against `market` on MAE **and** `beta_market` on rank IC |
| **Ties earn nothing** | A prediction with no ordering forms no portfolio. Fully-tied dates are skipped and counted; stable sorts once turned three constant baselines into a real-looking alpha that was just the alphabetically-first fifth of the universe |
| **Pre-registration** | Success criteria are fixed in source before the measurement runs |
| **Trial accounting** | Every configuration tried is counted and carried into the deflated Sharpe |

```bash
pytest tests/          # 482 tests: leakage, regression, phase suites
```

---

## Known Limitations

- **There is no demonstrated edge.** As measured, no comparator clears both
  floors, and nothing survives a `min_train` sweep. That is the finding.
- **Survivorship bias.** Point-in-time index membership is recorded only from
  the first universe sync onward; earlier windows use present-day membership.
- **Market impact is swept, not known.** The fee stack is measured exactly;
  impact is the term no schedule contains, so it is reported across 0–50bp
  rather than assumed at one value.
- **T+1 settlement is not modelled.** A signal computed at the 15:30 close is
  actionable at the next open at the earliest.
- **Restatement bias in fundamentals.** yfinance serves statements as restated,
  not as originally filed. Revisions are now logged to an append-only table, so
  the bias is measurable going forward — but not backwards.
- **News retrieval is imperfect.** The alias filter measured precision 0.68 /
  recall 0.72 before a per-ticker fix; the six ambiguous names were corrected and
  the null did not move.
- **One horizon.** Everything here is 30 sessions. Whether the null is
  horizon-specific is scoped and pre-registered, not yet run.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM routing | Task → ordered `(provider, model)` table. **OpenRouter** for few-call/long-prompt work (narrative, regime), **Groq** for many-call/short-prompt work. They cap different resources — Groq caps tokens (8k/min), OpenRouter caps requests — and every chain ends on Groq |
| Models | XGBoost + Optuna (seeded, nested); ridge factor comparators |
| Foundation models | Chronos-2, TimesFM-2.5, Kronos — all scored, all closed |
| Sentiment | ProsusAI/FinBERT, pinned to a resolved commit |
| Uncertainty | Split-conformal prediction |
| Backend | FastAPI + Uvicorn on Render (reads only) |
| Frontend | Next.js (App Router, ISR) + Recharts, on Vercel |
| Database | Supabase PostgreSQL |
| Compute | GitHub Actions — daily forecast + weekly evaluation |
| Tests | pytest, with mutation verification |

---

## Signal Library

**Momentum (6):** RSI-14, Stochastic %K, Williams %R, ROC-10, lag-1/5 returns
**Trend (6):** SMA-20, EMA-9/21/50, SMA-50 deviation, 52-week proximity
**Volatility (4):** Bollinger width/upper/lower, ATR-14
**Volume (2):** OBV, volume ROC
**Regime (1):** Hurst exponent over log prices
**Relative (3):** sector-relative momentum, 5/10/20 sessions
**Fundamental:** point-in-time valuation, lagged 60 days to the SEBI filing deadline
**News (4):** count excess, sentiment mean/dispersion/momentum, over a 30-session window
**Macro (4):** USD/INR, India VIX, NIFTY 5d/20d returns

Nine of the technical features are denominated in price or volume, so pooled
across tickers their level *is* ticker identity. They are listed as
`PRICE_SCALED` and excluded from the cross-sectional factor set — the module
raises `ImportError` at load if that is ever violated. The market-wide macro
columns are identically zero after within-date standardisation and reach the
panel only as interactions with a trailing beta.

---

## Project Structure

```text
├── .github/workflows/     daily-pipeline.yml · weekly-evaluation.yml
├── agents/                LangGraph nodes, LLM provider router, shared state
├── api/                   FastAPI routers and schemas — reads only
├── data/
│   ├── frozen_universe.py the 84 names, with the measurement that admitted each
│   ├── universe.py        the screening rule, still executable as an audit
│   ├── tickers.py         ticker metadata and benchmark mapping
│   └── db.py              schema and migrations
├── pipeline/
│   ├── signals.py         indicators and both labels
│   ├── panel.py           the cross-sectional panel, z-scoring, retargeting
│   ├── evaluation.py      purged walk-forward, rebalance books, deflated Sharpe
│   ├── baselines.py       every comparator, and the floors
│   ├── portfolio.py       cost-aware simulator (P4)
│   ├── neutralise.py      beta-neutralised residuals (P5)
│   ├── news*.py           dated archive, FinBERT scoring, features (P3)
│   ├── regime.py          market state, trailing beta, interactions
│   ├── series.py          adapter for time-series foundation models
│   ├── conformal.py       intervals and calibrated probabilities
│   └── vendor/kronos/     vendored at a pinned commit, with its licence
├── web/                   Next.js frontend — reads the API only
├── tests/                 482 tests
└── tools/                 measurement CLIs and audits
```

---

## Running Locally

```bash
git clone https://github.com/glitching-gops/Agentic-Stock-Forecast.git
cd Agentic-Stock-Forecast

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # DATABASE_URL, GROQ_API_KEY,
                                   # OPENROUTER_API_KEY, ADMIN_API_KEY
python main.py
```

Backend and frontend separately:

```bash
uvicorn api.main:app --reload --port 8000
cd web && npm run dev
```

Reproduce the measurements:

```bash
python tools/run_baselines.py       # the comparator table
python tools/run_portfolio.py       # cost-aware books, break-even IC
python tools/run_neutralised.py     # beta-neutralised residuals
```

`torch` is deliberately absent from `requirements.txt` — it lives in
`requirements-series.txt`, installed only by the foundation-model workflow. A
test asserts it is not in `sys.modules` after importing the pipeline, because it
was the largest contributor to memory pressure on an instance that had already
been OOM-killed.

---

## Production Deployment

- **Backend (Render)** — reads and on-demand admin forecasts only. Needs
  `DATABASE_URL`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `ADMIN_API_KEY`,
  `ALLOWED_ORIGINS`. Runs no scheduled compute. Use the **transaction pooler
  (port 6543)**, not a direct connection.
- **Compute (GitHub Actions)** — both workflows run directly against Supabase,
  independent of whether Render is awake. Same secrets, added under
  Settings → Secrets and variables → Actions. Both accept `workflow_dispatch`.

---

## Roadmap

Phases 0–5 are complete. Each closed on a measurement, and every result that
cleared the bar was retired by a check the project ran on itself.

| Phase | Status |
|---|---|
| **0 — Correctness** | Complete. F1–F15, all fixed and tested |
| **1 — Infrastructure** | Complete. Validation gate, experiment tracking, benchmark audit |
| **2 — Forecasting** | Complete. Pooled model, factor comparators, foundation models, LoRA, valuation — all null |
| **3 — Agents** | Complete. Dated news archive built and measured; the critic's LLM review retired on evidence |
| **4 — Evaluation** | Complete. Cost-aware simulator, validated on planted edges before use. Break-even rank IC is the reusable output |
| **5 — Research** | Complete. The null is not beta — the models are weak independently of it |
| **6 — Horizon sweep** | Scoped and pre-registered, not run. Is the 30-session null horizon-specific? |

One pre-registered test is still pending by calendar: whether the evidence
*grade* predicts realised direction. It was written with the outcomes table
empty, so the method and the bar were fixed with the answer genuinely
unavailable, and it becomes runnable from **mid-October 2026**.

---

## Author

**Venu Gopal Battula** — [github.com/glitching-gops](https://github.com/glitching-gops)

---

## Disclaimer

A research and portfolio project. Forecasts come from statistical models and are
not financial advice. Measured out-of-sample performance shows no edge, and the
cost-aware simulation is a historical measurement, not a track record. Past
performance does not guarantee future results.

---

## Acknowledgements

- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) — financial sentiment
- [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) ·
  [Google TimesFM](https://github.com/google-research/timesfm) ·
  [Kronos](https://github.com/shiyu-coder/Kronos) — time-series foundation models
- [Groq](https://groq.com) · [OpenRouter](https://openrouter.ai) — LLM inference
- [Supabase](https://supabase.com) · [Render](https://render.com) ·
  [Vercel](https://vercel.com) — hosting
- Purging, embargo and the deflated Sharpe follow López de Prado,
  *Advances in Financial Machine Learning*
