# ZeRO Agentic Stock Forecast

> Ranks NSE stocks by predicted 30-session return **relative to their sector benchmark**, using a LangGraph pipeline, a purged walk-forward evaluation harness, and conformal prediction intervals. Every performance figure is out-of-sample and reported next to the baseline it has to beat.
>
> **Current status: the model does not beat those baselines.** That finding is
> reported here rather than hidden, and the measurement apparatus that produces
> it is the point of the project. See [Measured Performance](#measured-performance).

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://glitching-gops-zer0.streamlit.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?style=flat&logo=render)](https://agentic-stock-forecast.onrender.com/api/health)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](https://python.org)

---

## What It Does

For each stock in a rule-defined NSE universe, ZeRO:

- Ingests 10 years of daily OHLCV, storing raw and adjusted prices separately
  so corporate actions cannot corrupt the series
- Computes 24 technical, sector-relative and fundamental signals
- Predicts the **30-session log return in excess of the stock's sector index**
- Attaches an 80% conformal prediction interval and a calibrated probability of
  outperformance
- Grades the forecast on held-out evidence — rank IC, hit rate vs the
  majority-class baseline — before it reaches the dashboard
- Ranks the universe by a composite of signal and conviction, gated by that
  evidence grade

---

## Measured Performance

<!-- PERFORMANCE_BLOCK -->

Purged walk-forward validation, 4 folds, 30-session embargo, 8 Optuna trials
tuned inside each training fold. 39 NIFTY 100 stocks, 10 years of daily data,
measured 15 August 2026. Every figure is out-of-sample and stated **before
transaction costs**.

| Metric | Model | Baseline | Model wins |
|---|---|---|---|
| Mean rank IC | **−0.065** | 0.000 (no skill) | 10 / 39 stocks |
| Rank IC \|t\| ≥ 2 | — | — | 1 / 39 stocks |
| Directional accuracy | **48.7%** | 54.4% (majority class) | 3 / 39 stocks |
| Mean absolute error | **0.0751** | 0.0706 (zero excess return) | 3 / 39 stocks |
| 80% interval coverage | **80.1%** | 80.0% (nominal) | ✅ calibrated |
| Brier score | 0.257 | 0.250 (uninformative) | ✗ |

Cross-sectional, rebalanced every 30 sessions (63 rebalances) — the way a
leaderboard is actually used:

| Metric | Value | Significance |
|---|---|---|
| Mean rank IC | −0.015 | *p* = 0.62 |
| Top quintile 30-session return | +1.30% | — |
| Equal-weight universe | +0.86% | — |
| Alpha vs equal weight | +0.45% | *t* = 0.72, *p* = 0.48 |
| Long–short spread | −0.21% | *t* = −0.22, *p* = 0.83 |

### Read this plainly

**The model does not currently beat its baselines.** Directional accuracy is
below the majority-class rate, mean rank IC is slightly negative, and the
long–short spread has the wrong sign. Nothing here is significant in either
direction — this is a null result, not a working strategy.

That is the point of the rewrite. The previous version of this README reported
**~4.3% MAPE and ~85% directional accuracy**; those numbers came from fitting a
meta-learner on the validation set and scoring it on that same set, over a
universe that had been selected on those very metrics. The honest numbers are
above, and they are worse than the honest numbers in an earlier draft of this
audit (rank IC ≈ +0.05) for three identifiable reasons: the universe is no
longer performance-selected, the target is now excess return rather than raw
return (which strips out the market and sector beta that inflated the earlier
correlation), and folds are properly purged.

**One result does hold up.** Conformal interval coverage came in at 80.1%
against a nominal 80%, and it was measured rather than assumed. The uncertainty
quantification works even though the point forecast does not.

What this system currently is: a correct, tested, reproducible measuring
instrument for NSE forecasting research, which reports that its own signal is
absent. Making the signal real is Phase 2 onward — a pooled cross-sectional
model, a linear factor comparator, and a foundation-model baseline, each of
which now has an honest yardstick to be judged against.

---

## What Changed, and Why

An audit of the previous version found that its reported metrics —
**~4.3% MAPE and ~85% directional accuracy** — were artifacts rather than
results. Three defects compounded:

1. **The reported metric was fitted and scored on the same rows.** A Ridge
   meta-learner was trained on the validation set, then scored by predicting on
   that same validation set. Re-running that exact procedure on live NSE data
   reproduces the headline figures from pure in-sample fit.
2. **Hyperparameters were tuned on the test slice.** Optuna received the full
   labelled set, including the 15% later reported as held out.
3. **The universe was selected on those metrics.** The 53-stock list came from
   ranking stocks by composite score — a score largely composed of the leaked
   figures — and keeping the top 5 per sector.

Underneath sat three silent functional bugs, each confirmed by execution: the
LSTM never wrote a checkpoint (so the advertised ensemble was never running),
the warm-path forecast function raised `KeyError` and persisted today's price as
the forecast, and training labels were never backfilled — freezing the labelled
training set at the day the database was first populated.

Phase 0 of the remediation fixed all of it. The numbers above are what the
system actually achieves.

---

## Architecture

Compute runs on two independent cadences, on GitHub Actions — not on the
backend. The first production run tried to do all of this in one process on
Render's free tier; it starved the instance's single CPU core for over an
hour and then OOM-killed it. Splitting by cadence and moving the compute off
the serving process fixed both problems at once.

```text
              DAILY (GitHub Actions, weekdays 18:30 IST)
NSE constituent list ──► point-in-time universe (index + liquidity + listing rule)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          Trading Data Agent                External Data Agent
       OHLCV → 24 signals                 macro · sentiment (display only)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          Forecasting Agent
              XGBoost, CACHED hyperparameters — one fit, no search
              conformal interval + P(outperform), read from last weekly run
              LLM narrative (no numbers)
                                    │
                                    ▼
                            Critic Agent
              deterministic evidence gate (STRONG/WEAK/INSUFFICIENT)
              LLM signal review — may only downgrade
                                    │
                                    ▼
                          Supabase PostgreSQL
                                    ▲
                                    │  reads evaluation + calibration
              WEEKLY (GitHub Actions, Saturday 08:30 IST)
              purged walk-forward evaluation, nested Optuna tuning
              conformal calibration on out-of-sample residuals
              refreshes cached production hyperparameters
                                    │
                                    ▼
              Supabase PostgreSQL ──► FastAPI (Render) ──► Streamlit dashboard
```

Render only ever serves reads and on-demand admin-triggered single-ticker
forecasts (still the cheap daily path) — it runs no scheduled compute and no
Optuna search. An evidence grade can therefore be up to a week older than the
price it's attached to; that gap is reported as `evaluated_at`, not hidden.

---

## Evaluation Methodology

This is the part that matters most, so it is stated explicitly.

| Control | What it does |
|---|---|
| **Purging** | Training rows whose 30-session label reaches into the test window are removed |
| **Embargo** | A further 30 sessions are dropped, widening the gap |
| **Nested tuning** | Hyperparameters are searched inside each training fold only; the tuner cannot see test rows |
| **Seeded search** | Optuna studies use a fixed seed, so tuning is reproducible |
| **Baselines** | Directional accuracy is always shown beside the majority-class rate; error beside the naive zero-excess forecast |
| **Overlap correction** | Consecutive 30-session labels are ~97% overlapping, so effective sample size is `n / 30`. Skipping this inflates every t-statistic by ~5.5× |
| **Coverage check** | Conformal intervals claim 80%; realised coverage is measured against that claim |
| **Cadence separation** | The purged walk-forward evaluation runs weekly, not daily — it measures a slow-moving property ("does this model form have skill"), and re-running it daily was the direct cause of an OOM crash. Daily runs fit fresh with cached hyperparameters and read the last weekly result; `evaluated_at` on every forecast states exactly how old that evidence is |

Run it yourself:

```bash
pytest tests/          # leakage, regression, and daily/weekly split tests
```

The suite fails if any audited defect is reintroduced — verified by mutation
testing (deliberately restoring each defect and confirming the tests catch it).

---

## Known Limitations

Stated plainly, because a forecasting system that hides these is not worth trusting.

- **No transaction costs.** Indian delivery round trips run roughly 30–60 bps
  before market impact. On a monthly rebalance that is comparable to the entire
  measured edge.
- **No portfolio backtest.** Forecast accuracy and investment profitability are
  different questions; only the first is currently measured. No Sharpe, no
  drawdown, no NIFTY-relative return.
- **Survivorship bias.** Point-in-time index membership is recorded only from
  the first universe sync onward. Earlier windows use present-day membership.
- **News sentiment is not a model feature.** It only ever existed for the
  current date — zero across the training set, non-zero only for the row being
  predicted. It returns when a dated news archive exists.
- **T+1 settlement is not modelled.** A signal computed at the 15:30 close is
  actionable at the next open at the earliest.
- **There is no demonstrated edge.** As measured, the model loses to a
  majority-class baseline on direction and to a zero-excess forecast on
  magnitude. That is the honest finding, not a failure of the writeup.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq API — `openai/gpt-oss-20b` (narrative and signal review only) |
| Model | XGBoost, Optuna (seeded, nested tuning) |
| Uncertainty | Split-conformal prediction |
| Signals | `ta`, pandas, numpy |
| Universe | NSE archives constituent lists, liquidity screen |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Database | Supabase PostgreSQL |
| Compute | GitHub Actions — daily forecast (weekdays 18:30 IST) + weekly evaluation (Saturday 08:30 IST), both writing directly to Supabase |
| Tests | pytest |

---

## Signal Library

**Momentum (6):** RSI-14, Stochastic %K, Williams %R, ROC-10, lag-1/5 returns
**Trend (6):** SMA-20, EMA-9/21/50, SMA-50 deviation, 52-week proximity
**Volatility (4):** Bollinger width/upper/lower, ATR-14
**Volume (2):** OBV, volume ROC
**Regime (1):** Hurst exponent over log prices
**Relative (3):** sector-relative momentum, 5/10/20 sessions
**Fundamental (1):** quarterly EPS surprise, lagged to the first tradable session
**Macro (6):** USD/INR, India VIX, NIFTY 5d/20d returns, FII/DII net flows

---

## Project Structure

```text
├── .github/workflows/
│   ├── daily-pipeline.yml      light daily job, runs against Supabase directly
│   └── weekly-evaluation.yml   expensive purged walk-forward evaluation
├── agents/            LangGraph nodes and shared state
├── api/               FastAPI routers and schemas — reads + on-demand admin only
├── app/               Streamlit dashboard
├── data/
│   ├── universe.py    point-in-time universe rule and membership history
│   ├── tickers.py     ticker metadata and benchmark mapping
│   └── db.py          schema and migrations
├── pipeline/
│   ├── fetch.py       OHLCV ingestion (raw + adjusted)
│   ├── signals.py     indicators and the excess-return target
│   ├── model.py       forecast_ticker_daily (cheap) / evaluate_and_persist_ticker (weekly)
│   ├── evaluation.py  purged walk-forward harness, metrics, baselines
│   ├── conformal.py   prediction intervals and calibrated probabilities
│   ├── tuning.py      nested, seeded hyperparameter search
│   └── archived/      LSTM and meta-learner, with the reasons they were removed
├── tests/             leakage, regression, and daily/weekly split suite
├── tools/             maintenance scripts
├── main.py            local entry point
└── scheduler.py       run_pipeline_job (daily) / run_weekly_evaluation_job (weekly);
                        start_scheduler() is for local dev only, not called on Render
```

---

## Running Locally

```bash
git clone https://github.com/glitching-gops/Agentic-Stock-Forecast.git
cd Agentic-Stock-Forecast

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # fill in DATABASE_URL, GROQ_API_KEY, ADMIN_API_KEY

python main.py                     # sync universe, ingest, forecast, launch dashboard
```

Backend and dashboard separately:

```bash
uvicorn api.main:app --reload --port 8000
streamlit run app/main.py
```

Recover historical index membership when archive.org is responsive:

```bash
python -c "from data.universe import backfill_membership_from_wayback as b; print(b())"
```

---

## Production Deployment

Two independent pieces, deployed separately:

- **Backend (Render)** — serves reads and on-demand admin-triggered forecasts
  only. Needs `DATABASE_URL`, `GROQ_API_KEY`, `ADMIN_API_KEY`, `ALLOWED_ORIGINS`
  as environment variables. Runs no scheduled compute.
- **Compute (GitHub Actions)** — `.github/workflows/daily-pipeline.yml` and
  `weekly-evaluation.yml` run the actual pipeline directly against Supabase,
  independent of whether Render is awake. Needs `DATABASE_URL` and
  `GROQ_API_KEY` added as **repository secrets** (Settings → Secrets and
  variables → Actions) — the same values as Render's env vars. Both workflows
  also accept `workflow_dispatch` for a manual run from the Actions tab.

Render's admin routes remain as a fallback: `POST /api/admin/run-daily-pipeline`
for the light job, `POST /api/admin/run-weekly-evaluation` for the expensive
one (prefer re-running the GitHub Actions workflow over calling this directly
— it's the workload that OOM-killed the first production deployment when it
ran daily, in-process, on Render's single free-tier core).

---

## Roadmap

Phase 0 (correctness) is complete. What follows:

- **Phase 1 — Infrastructure.** Validation gate, experiment tracking with config
  and data hashes, corporate-actions table.
- **Phase 2 — Forecasting.** Pooled cross-sectional model, linear factor
  comparator, zero-shot time-series foundation model baseline.
- **Phase 3 — Agents.** Re-scope the LLM to dated, attributed evidence
  extraction; ablate whether the critic improves realised hit rate at all.
- **Phase 4 — Evaluation.** Cost-aware portfolio simulator, Sharpe / Sortino /
  max drawdown / Calmar against NIFTY 50 TR, deflated Sharpe.
- **Phase 5 — Research.** Entity-masked contamination measurement,
  regime-conditional models, risk-aware sizing.

---

## Author

**Venu Gopal Battula** — [github.com/glitching-gops](https://github.com/glitching-gops)

---

## Disclaimer

A research and portfolio project. Forecasts come from statistical models and are
not financial advice. Measured out-of-sample performance is weak and is reported
before transaction costs. Past performance does not guarantee future results.

---

## Acknowledgements

- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) — financial sentiment
- [Groq](https://groq.com) — LLM inference
- [Supabase](https://supabase.com) — PostgreSQL hosting
- [Render](https://render.com) · [Streamlit](https://streamlit.io) — hosting
- Purging, embargo and deflated Sharpe follow López de Prado,
  *Advances in Financial Machine Learning*
