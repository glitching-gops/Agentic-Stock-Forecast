# db.py — SQLite database setup using SQLAlchemy
# Creates tables: ohlcv, signals, sentiment, and macro

import os
import numpy as np
from sqlalchemy import create_engine, text
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
