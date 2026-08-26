# db.py — SQLite database setup using SQLAlchemy
# Creates tables: ohlcv, signals, sentiment, and macro

import os
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import NoSuchTableError
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SQLITE_PATH = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'stock_forecast.db')}"

# Use DATABASE_URL from environment if present, otherwise fall back to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", _SQLITE_PATH)

# SQLAlchemy requires postgresql:// not postgres:// for newer versions
# Normalize postgres:// → postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Force psycopg2 driver — psycopg (v3) is not installed on Render
# Converts: postgresql://... → postgresql+psycopg2://...
# Converts: postgresql+psycopg://... → postgresql+psycopg2://...
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)

_ENGINE = None


def to_native(value):
    """
    Converts a numpy scalar to its plain Python equivalent, leaving anything
    else untouched.

    This is not cosmetic. np.float64 subclasses Python float, so psycopg2
    happily accepts it and adapts it with the float adapter — which renders
    the value with repr(). Under numpy 1.x that produced "42.75"; numpy 2.x
    changed the repr to "np.float64(42.75)", which Postgres parses as a
    schema-qualified name and rejects with:

        InvalidSchemaName: schema "np" does not exist

    Two things kept this hidden. SQLite accepts a float subclass without ever
    calling repr(), so it only breaks against Postgres; and the numpy values
    reach the write only for tickers that already have a persisted weekly
    evaluation, so any ticker never yet evaluated inserts cleanly. The daily
    run that exposed it failed on exactly the 33 tickers the weekly job had
    reached (alphabetically ABB->GAIL) and succeeded on the other 62.
    """
    if isinstance(value, np.generic):       # np.float64, np.int64, np.bool_, ...
        return value.item()
    return value


def to_native_params(params: dict) -> dict:
    """Applies to_native() across a bind-parameter dict."""
    return {k: to_native(v) for k, v in params.items()}


# ── Telling "the schema is behind" apart from "the database is gone" ─────────
#
# Several read paths fail soft to an empty result on purpose: a fresh
# production database has no index_membership table until the first sync, and
# data/db.py adds some leaderboard columns lazily, so a query naming one of
# them must degrade rather than 500.
#
# That intent is narrow and the guard was not. A bare `except Exception` around
# those queries also catches "cannot connect to the database", and an outage
# was therefore reported to callers as HTTP 200 with an EMPTY UNIVERSE. That is
# worse than a 500: the Next.js frontend caches read-through, so a revalidation
# during an outage silently replaced a good page with an empty board, and
# nothing anywhere said the database was down.
#
# So the soft path is now keyed on the specific SQLSTATE it was written for.
# Anything else propagates.
_UNDEFINED_TABLE = "42P01"
_UNDEFINED_COLUMN = "42703"


def _chain(exc: BaseException):
    """
    `exc` and everything it wraps: SQLAlchemy's `orig`, and `__cause__`.

    Both links are needed and neither is enough. pandas raises its OWN
    `DatabaseError` from a driver failure inside `read_sql`, so the SQLSTATE
    sits two levels down — under `__cause__` and then under `orig`. Reading
    only the outermost exception's message would miss it, and against Postgres
    that message says `relation "x" does not exist` rather than anything the
    SQLite text match below looks for.
    """
    seen: set[int] = set()
    queue = [exc]
    while queue:
        link = queue.pop(0)
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        queue.append(getattr(link, "orig", None))
        queue.append(link.__cause__)


def is_missing_relation(exc: BaseException) -> bool:
    """
    True when a query failed because the table or column is not there yet.

    False for every other database failure, connection loss above all.

    Postgres reports this through SQLSTATE, which is unambiguous, so a code is
    trusted the moment one is found. SQLite has no SQLSTATE and raises
    `OperationalError` for a missing table — the SAME class SQLAlchemy uses for
    a failed connection against Postgres — so on that path the message is the
    only discriminator and it is matched narrowly. Tests run on SQLite, so that
    branch is the one under test and the SQLSTATE branch is pinned with a stub.
    """
    for link in _chain(exc):
        # The inspector does not go through the driver at all. It reports an
        # absent table as NoSuchTableError, whose str() is the bare table name
        # and matches no message test.
        if isinstance(link, NoSuchTableError):
            return True
        pgcode = getattr(link, "pgcode", None)
        if pgcode is not None:
            return pgcode in (_UNDEFINED_TABLE, _UNDEFINED_COLUMN)

    return "no such table" in str(exc).lower() or "no such column" in str(exc).lower()


def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
        
    if DATABASE_URL.startswith("postgresql"):
        _ENGINE = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True,
            connect_args={"prepare_threshold": None}
        )
    else:
        _ENGINE = create_engine(DATABASE_URL, echo=False)
        
    return _ENGINE

def init_db():
    engine = get_engine()
    with engine.connect() as conn:

        # Raw OHLCV table.
        # `close` is the unadjusted traded price; `adj_close` is the
        # corporate-action-adjusted series. Storing both is what lets the
        # pipeline detect adjustment breaks instead of silently splicing two
        # adjustment bases together (audit finding F11).
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                date        TEXT,
                ticker      TEXT,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                adj_close   REAL,
                volume      REAL,
                PRIMARY KEY (date, ticker)
            )
        """))

        for col in ["adj_close"]:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE ohlcv ADD COLUMN {col} REAL"))
            except Exception:
                pass  # Column already exists

        # Computed signals table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS signals (
                date        TEXT,
                ticker      TEXT,
                close       REAL,
                rsi         REAL,
                macd_hist   REAL,
                bb_width    REAL,
                obv         REAL,
                sma_20      REAL,
                ema_9       REAL,
                ema_21      REAL,
                atr_14      REAL,
                stoch_k     REAL,
                williams_r  REAL,
                roc_10      REAL,
                vroc_10     REAL,
                prox_52w    REAL,
                lag1_ret    REAL,
                lag5_ret    REAL,
                dev_sma50   REAL,
                ema_50      REAL,
                bb_upper    REAL,
                bb_lower    REAL,
                hurst       REAL,
                target      REAL,
                PRIMARY KEY (date, ticker)
            )
        """))

        # Add new columns if they do not already exist (safe migration)
        new_columns = [
            "ema_50", "bb_upper", "bb_lower", "hurst",
            "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
            "earnings_surprise",
            # Phase 0: the target is now a forward EXCESS return rather than an
            # absolute price level. Predicting a price level made MAPE look
            # flattering (a random walk beats the model on it) and capped every
            # forecast at the training maximum, because trees cannot
            # extrapolate (audit findings F8, and the target half of F1).
            "target_return",          # forward log return of the stock
            "target_excess_return",   # forward log return minus benchmark
            "benchmark_return",       # forward log return of the benchmark
            # Phase 2: the benchmark's LEVEL, not just its forward return.
            # benchmark_return is a label - it looks 30 sessions ahead - so
            # nothing may read it as an input. Without the level there is no way
            # to construct the relative price series close/benchmark_close,
            # whose forward log return IS target_excess_return by construction.
            # That series is what lets a time-series model predict the excess
            # return directly instead of forecasting the stock and the index
            # separately and differencing two independent errors.
            "benchmark_close",
        ]
        for col in new_columns:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN {col} REAL"))
            except Exception:
                pass  # Column already exists, skip

        # Non-numeric signal columns recording which benchmark was used.
        for col, coltype in [("benchmark_ticker", "TEXT"),
                             ("benchmark_sector_specific", "INTEGER")]:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN {col} {coltype}"))
            except Exception:
                pass  # Column already exists, skip

        # Sentiment table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment (
                date            TEXT,
                ticker          TEXT,
                headline        TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                PRIMARY KEY (date, ticker, headline)
            )
        """))

        # Macro table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS macro (
                date                TEXT PRIMARY KEY,
                usdinr              REAL,
                india_vix           REAL,
                nifty_5d_return     REAL,
                nifty_20d_return    REAL,
                fii_net_flow        REAL,
                dii_net_flow        REAL
            )
        """))

        # Add new macro columns (safe migration)
        macro_new_columns = ["fii_net_flow", "dii_net_flow"]
        for col in macro_new_columns:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE macro ADD COLUMN {col} REAL"))
            except Exception:
                pass  # Column already exists, skip

        # Add indexes for faster querying by ticker and date

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON ohlcv (ticker, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals (ticker, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment (ticker, date)"))

        # Step 6: Create the forecasts and leaderboard tables
        # Use SERIAL PRIMARY KEY for PostgreSQL, INTEGER PRIMARY KEY AUTOINCREMENT for SQLite
        id_col_type = "INTEGER PRIMARY KEY AUTOINCREMENT" if _SQLITE_PATH in DATABASE_URL or DATABASE_URL.startswith("sqlite") else "SERIAL PRIMARY KEY"
        
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS forecasts (
                id                       {id_col_type},
                ticker                   TEXT NOT NULL,
                company                  TEXT,
                sector                   TEXT,
                current_price            REAL,
                forecast_price           REAL,
                direction                TEXT,
                change_pct               REAL,
                mape                     REAL,
                directional_accuracy     REAL,
                forecast_confidence      TEXT,
                signal_narrative         TEXT,
                critic_verdict           TEXT,
                critic_reasoning         TEXT,
                critic_flags             TEXT,
                critic_confidence_adjustment TEXT,
                last_updated             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                ticker                   TEXT PRIMARY KEY,
                company                  TEXT,
                sector                   TEXT,
                current_price            REAL,
                forecast_price           REAL,
                upside_pct               REAL,
                composite_score          REAL,
                critic_verdict           TEXT,
                forecast_confidence      TEXT,
                mape                     REAL,
                directional_accuracy     REAL,
                last_updated             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Phase 0 forecast record. The model predicts an excess return; the
        # displayed rupee target is derived from it, and is shown alongside a
        # conformal interval, a calibrated probability, and the random-walk
        # reference so a reader can see what the forecast is being compared to.
        forecast_extra = [
            ("pred_excess_return",   "REAL"),    # model output, log excess return
            ("pred_return",          "REAL"),    # implied total return
            ("interval_low",         "REAL"),    # conformal lower bound, price
            ("interval_high",        "REAL"),    # conformal upper bound, price
            ("interval_coverage",    "REAL"),    # nominal coverage, e.g. 0.80
            ("prob_outperform",      "REAL"),    # calibrated P(excess return > 0)
            ("random_walk_price",    "REAL"),    # baseline: today's price
            ("benchmark_ticker",     "TEXT"),
            ("benchmark_name",       "TEXT"),
            ("benchmark_sector_specific", "INTEGER"),
            ("eval_rank_ic",         "REAL"),    # honest walk-forward metrics
            ("eval_hit_rate",        "REAL"),
            ("eval_baseline_hit_rate", "REAL"),
            ("eval_beats_random_walk", "INTEGER"),
            ("model_version",        "TEXT"),
            ("universe_rule",        "TEXT"),
            # When the held-out evaluation behind this forecast was last
            # measured. Distinct from last_updated (the forecast itself is
            # regenerated daily; the evidence backing it is only re-measured
            # weekly) — surfaced so staleness is visible, not implied away.
            ("evaluated_at",         "TEXT"),
            # Why composite_score is what it is — chiefly, why it is zero.
            # Most of the leaderboard scores 0.0, and that single value covers
            # "never evaluated", "predicted to underperform" and "flagged out",
            # which are very different statements. See
            # agents.graph.classify_score_basis.
            ("score_basis",          "TEXT"),
        ]
        for table in ["forecasts", "leaderboard"]:
            for col, coltype in forecast_extra:
                try:
                    with conn.begin_nested():
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"))
                except Exception:
                    pass  # Column already exists, skip

        # Realised outcomes, written back at T+30. This is what converts a
        # claimed accuracy into an observed one.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS forecast_outcomes (
                forecast_id          INTEGER,
                ticker               TEXT NOT NULL,
                forecast_date        TEXT NOT NULL,
                resolution_date      TEXT,
                pred_excess_return   REAL,
                realised_excess_return REAL,
                realised_return      REAL,
                benchmark_return     REAL,
                direction_correct    INTEGER,
                inside_interval      INTEGER,
                PRIMARY KEY (ticker, forecast_date)
            )
        """))

        # Splits and dividends. F11 fixed the SYMPTOM of a spliced adjustment
        # basis by rewriting the whole OHLCV series each run; this records the
        # CAUSE, so a price break is either explained by a row here or it is a
        # data-quality finding. See pipeline/corporate_actions.py.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                ticker       TEXT NOT NULL,
                date         TEXT NOT NULL,
                action_type  TEXT NOT NULL,      -- SPLIT | DIVIDEND
                ratio        REAL,               -- SPLIT only, e.g. 2.0 for 1:2
                amount       REAL,               -- DIVIDEND only, per share
                implausible  INTEGER DEFAULT 0,  -- outside a believable range
                PRIMARY KEY (ticker, date, action_type)
            )
        """))

        # Valuation fundamentals, stored point-in-time.
        #
        # `effective_date` is the load-bearing column and the reason this is a
        # table rather than a join against a vendor call. A fundamental is
        # known to the market when it is FILED, not when the fiscal period
        # ended; attaching FY2025 earnings to 31 March 2025 hands the model two
        # months nobody had. SEBI (LODR) Reg 33 allows 60 days for audited
        # annual results, so effective_date is period_end + 60.
        #
        # Keyed on period rather than date: one row per fiscal period per
        # ticker, expanded onto the daily grid by an as-of join at read time.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker               TEXT NOT NULL,
                period_end           TEXT NOT NULL,  -- fiscal period end
                effective_date       TEXT NOT NULL,  -- first date it was knowable
                eps                  REAL,           -- diluted preferred, annual
                book_value_per_share REAL,
                shares               REAL,
                source               TEXT,
                -- Vintage. `first_seen` is when this period first entered the
                -- table; `fetched_at` is the last sync that confirmed or
                -- changed it. Rows written before vintage tracking existed
                -- carry NULL, which honestly means "unknown" rather than a
                -- backfilled guess.
                first_seen           TEXT,
                fetched_at           TEXT,
                PRIMARY KEY (ticker, period_end)
            )
        """))

        for col in ["first_seen", "fetched_at"]:
            try:
                with conn.begin_nested():
                    conn.execute(text(
                        f"ALTER TABLE fundamentals ADD COLUMN {col} TEXT"))
            except Exception:
                pass  # Column already exists, skip

        # Restatement log. APPEND-ONLY, and the reason the table above can stay
        # a simple current-view upsert.
        #
        # yfinance serves financial statements AS RESTATED, not as originally
        # filed. If a company restates FY2024 during FY2025 we see the restated
        # figure and attach it to 2024 — information nobody had at the time,
        # and the direction of that error flatters the model. The vendor cannot
        # tell us about restatements that happened before we started looking,
        # but it cannot hide one that happens while we are watching: a plain
        # upsert would simply overwrite the old figure and leave no trace that
        # anything moved.
        #
        # So every observed change to a figure already on file is recorded
        # here, with both values and the date we saw it move. That converts
        # "how bad is the restatement bias?" from unanswerable into a query,
        # accumulating from the day this ships. Nothing ever updates or deletes
        # a row here, for the same reason forecast_outcomes never updates one:
        # a record that can be revised is not evidence.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fundamental_revisions (
                ticker        TEXT NOT NULL,
                period_end    TEXT NOT NULL,   -- fiscal period that moved
                observed_at   TEXT NOT NULL,   -- when WE saw it move
                field         TEXT NOT NULL,   -- eps | book_value_per_share
                old_value     REAL,
                new_value     REAL,
                first_seen    TEXT,            -- when the old value was recorded
                source        TEXT,
                PRIMARY KEY (ticker, period_end, field, observed_at)
            )
        """))

        # One row per pipeline run: what code, what configuration, what data.
        # Without it, "the metrics moved" has no answer — the model, the
        # feature list, the universe rule and the labelled set can all change
        # between two runs and nothing recorded which of them did.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id         TEXT PRIMARY KEY,
                job            TEXT NOT NULL,     -- daily | weekly | manual
                started_at     TEXT NOT NULL,
                finished_at    TEXT,
                status         TEXT,              -- RUNNING | OK | ABORTED | FAILED
                git_sha        TEXT,
                model_version  TEXT,
                config_hash    TEXT,              -- features, target, horizon, eval params
                data_hash      TEXT,              -- universe x coverage x label counts
                universe_rule  TEXT,
                n_tickers      INTEGER,
                labelled_rows  INTEGER,
                gate_status    TEXT,              -- PASS | WARN | FAIL
                gate_report    TEXT,              -- JSON, one entry per check
                metrics        TEXT,              -- JSON
                notes          TEXT
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                ticker              TEXT PRIMARY KEY,
                xgb_mape            REAL,
                xgb_dir_acc         REAL,
                lstm_val_mape       REAL,
                ensemble_mape       REAL,
                ensemble_dir_acc    REAL,
                lstm_epochs_trained INTEGER,
                meta_xgb_coef       REAL,
                meta_lstm_coef      REAL,
                meta_hurst_coef     REAL,
                ensemble_in_use     INTEGER DEFAULT 1,
                last_trained        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Add new model_metadata columns (safe migration).
        # The eval_* columns replace xgb_mape / xgb_dir_acc / ensemble_*, which
        # are retained only so old rows still read. Writing an excess-return MAE
        # into a column named "mape" is exactly the sort of quiet mislabelling
        # that made the previous metrics unreadable.
        meta_new_columns = [
            "ensemble_mape", "ensemble_dir_acc",
            "eval_rank_ic", "eval_rank_ic_t", "eval_hit_rate",
            "eval_baseline_hit_rate", "eval_mae", "eval_mae_naive",
            "eval_n_oos", "eval_n_effective",
            # Conformal calibration, persisted so the daily job can build a
            # price view without recomputing the purged walk-forward run that
            # produced it. conformal_residuals stores the calibration pool as
            # a JSON list — needed (not just the quantile) because
            # ConformalCalibration.prob_positive() is a distribution-free
            # empirical estimate over those residuals, not a parametric one.
            "conformal_quantile", "conformal_coverage", "conformal_n",
        ]
        for col in meta_new_columns:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE model_metadata ADD COLUMN {col} REAL"))
            except Exception:
                pass  # Column already exists, skip

        for col in ["model_version", "eval_protocol", "conformal_residuals"]:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE model_metadata ADD COLUMN {col} TEXT"))
            except Exception:
                pass  # Column already exists, skip

        # evaluated_at is separate from last_trained: last_trained updates
        # every DAY (the production model is refit daily with cached
        # hyperparameters); evaluated_at only updates when the WEEKLY
        # purged walk-forward evaluation actually reruns. The gap between
        # them is exactly how stale the evidence grade is, and that gap is
        # reported rather than hidden.
        try:
            with conn.begin_nested():
                conn.execute(text("ALTER TABLE model_metadata ADD COLUMN evaluated_at TIMESTAMP"))
        except Exception:
            pass  # Column already exists, skip

        conn.commit()
    print("Database initialised.")
