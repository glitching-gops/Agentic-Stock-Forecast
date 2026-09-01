"""
data/frozen_universe.py — the fixed forecasting universe.

The project no longer ranks stocks; it forecasts every name in a FIXED
universe. That changes what a universe is for. ``get_universe()`` used to
recompute membership, a liquidity floor and a listing-history floor on every
call, which meant the set of stocks being forecast could change between the
daily job and the weekly evaluation, between two API requests, or between a
measured result and the run that quotes it. A universe recomputed at run time
is a universe you cannot compare across time.

So the list is frozen here, as data, with the measurement that produced each
row beside it. ``data.universe.screen_universe()`` still holds the rule, and
``tools/audit_universe.py`` re-runs it and reports every disagreement — the
same arrangement as ``tools/audit_benchmarks.py --apply-check``: the rule stays
executable so drift is visible, but nothing acts on it automatically.

SELECTION IS ON DATA QUALITY ONLY, NEVER ON MEASURED SKILL. This is the F4
lesson restated. The previous universe was built by ``tools/select_top_50.py``,
which ranked by composite score — a function of the model's own reported
accuracy — so every metric averaged over the survivors was biased upward by
construction. Selecting on forecast accuracy here would do the same thing in a
new place and would make every forward number uninterpretable: the surviving
universe would look skilled because it was chosen for looking skilled.

Note what is DELIBERATELY absent as a consequence. DMART, PNB and UNIONBANK are
the only three tickers that currently clear the evidence gate; DMART is not in
this list, because it carries 2,337 sessions. Keeping it would be selecting on
the outcome.

── The criteria, measured 2026-09-01 against the live database ────────────────

  Sessions of history   >= 2,400        keeps 84 of 120
  Index membership      NIFTY 100 today keeps 84 of 84 (non-binding)
  Median daily value    >= Rs 25 cr     keeps 84 of 84 (non-binding)

Only the history criterion binds. All 120 tickers in ``ohlcv`` clear the
liquidity floor — the thinnest name in the whole table is at Rs 12.9 cr and the
thinnest in this universe is BAJAJHLDNG at Rs 43.4 cr — so a liquidity screen
removes nobody today. It is recorded anyway, so a future thin listing is
refused by a rule rather than by whoever notices.

THE THRESHOLD IS NOT A TUNED KNOB, and that matters given this project's
history of results that lived at one arbitrary setting. The row-count
distribution has a natural break: 84 tickers hold 2,428-2,483 sessions and the
next name down holds 2,337. Anything from 2,338 to 2,428 selects exactly the
same 84 stocks, so the choice sits in a 91-session dead zone rather than on a
slope. It is also far above the hard floor — ``MIN_ROWS_FOR_EVALUATION`` is
505 — because five purged folds with a 30-session embargo want real
out-of-sample room, not the bare minimum that produces one fold.

CONTIGUITY IS RECORDED, NOT ENFORCED. Five of the 84 are missing exactly one
interior session from the ~2,483-session NSE calendar (BOSCHLTD, DIVISLAB,
INDIGO, SIEMENS, SOLARINDS). A hole does matter — a row-stepped 30-session
horizon silently measures 31 across one, which is the vroc_10 defect recorded
in CLAUDE.md's landmines — but it is a REPAIRABLE fault in the fetch, not a
structural fact about the company. The repair is to refetch the bar; dropping a
large-cap over one missing print in 2,483 would be the wrong remedy applied to
the wrong layer. ``pipeline.validation.check_sessions_are_contiguous`` is the
guard that acts on it.
"""

from __future__ import annotations

# (ticker, sessions in ohlcv, first session, missing interior sessions,
#  median daily traded value over the last 90 sessions, in Rs crore)
#
# Measured 2026-09-01. These figures are a RECORD of why each name is here,
# not a live reading — they will drift as data accumulates, and
# tools/audit_universe.py is what re-measures them.
FROZEN_MEASUREMENTS: tuple[tuple[str, int, str, int, float], ...] = (
    ("ABB.NS",        2482, '2016-08-16', 0,    200.7),
    ("ADANIENSOL.NS", 2483, '2016-08-16', 0,    402.6),
    ("ADANIENT.NS",   2483, '2016-08-16', 0,    471.0),
    ("ADANIPORTS.NS", 2483, '2016-08-16', 0,    343.4),
    ("ADANIPOWER.NS", 2483, '2016-08-16', 0,    580.8),
    ("AMBUJACEM.NS",  2483, '2016-08-16', 0,    119.2),
    ("APOLLOHOSP.NS", 2482, '2016-08-16', 0,    258.4),
    ("ASIANPAINT.NS", 2483, '2016-08-16', 0,    229.3),
    ("AXISBANK.NS",   2483, '2016-08-16', 0,    771.3),
    ("BAJAJ-AUTO.NS", 2483, '2016-08-16', 0,    326.2),
    ("BAJAJFINSV.NS", 2483, '2016-08-16', 0,    183.5),
    ("BAJAJHLDNG.NS", 2482, '2016-08-16', 0,     43.4),
    ("BAJFINANCE.NS", 2483, '2016-08-16', 0,    714.9),
    ("BANKBARODA.NS", 2483, '2016-08-16', 0,    269.9),
    ("BEL.NS",        2483, '2016-08-16', 0,    434.6),
    ("BHARTIARTL.NS", 2483, '2016-08-16', 0,   1153.4),
    ("BOSCHLTD.NS",   2482, '2016-08-16', 1,    126.6),
    ("BPCL.NS",       2483, '2016-08-16', 0,    206.2),
    ("BRITANNIA.NS",  2483, '2016-08-16', 0,    146.4),
    ("CANBK.NS",      2483, '2016-08-16', 0,    270.6),
    ("CGPOWER.NS",    2483, '2016-08-16', 0,    285.0),
    ("CHOLAFIN.NS",   2483, '2016-08-16', 0,    209.4),
    ("CIPLA.NS",      2483, '2016-08-16', 0,    146.3),
    ("COALINDIA.NS",  2483, '2016-08-16', 0,    283.6),
    ("CUMMINSIND.NS", 2482, '2016-08-16', 0,    212.1),
    ("DIVISLAB.NS",   2482, '2016-08-16', 1,    242.4),
    ("DLF.NS",        2483, '2016-08-16', 0,    194.8),
    ("DRREDDY.NS",    2483, '2016-08-16', 0,    213.9),
    ("EICHERMOT.NS",  2483, '2016-08-16', 0,    313.9),
    ("GAIL.NS",       2483, '2016-08-16', 0,    134.8),
    ("GODREJCP.NS",   2483, '2016-08-16', 0,    106.1),
    ("GRASIM.NS",     2482, '2016-08-16', 0,    212.5),
    ("HCLTECH.NS",    2483, '2016-08-16', 0,    380.7),
    ("HDFCBANK.NS",   2483, '2016-08-16', 0,   2277.0),
    ("HINDALCO.NS",   2483, '2016-08-16', 0,    494.0),
    ("HINDUNILVR.NS", 2483, '2016-08-16', 0,    293.5),
    ("HINDZINC.NS",   2483, '2016-08-16', 0,    243.1),
    ("ICICIBANK.NS",  2483, '2016-08-16', 0,   1749.9),
    ("INDHOTEL.NS",   2483, '2016-08-16', 0,    137.0),
    ("INDIGO.NS",     2482, '2016-08-16', 1,    362.9),
    ("INFY.NS",       2483, '2016-08-16', 0,   1092.6),
    ("IOC.NS",        2483, '2016-08-16', 0,    158.4),
    ("ITC.NS",        2483, '2016-08-16', 0,    383.0),
    ("JINDALSTEL.NS", 2483, '2016-08-16', 0,    111.8),
    ("JSWSTEEL.NS",   2483, '2016-08-16', 0,    184.5),
    ("KOTAKBANK.NS",  2483, '2016-08-16', 0,    531.9),
    ("LT.NS",         2483, '2016-08-16', 0,    668.9),
    ("LTM.NS",        2482, '2016-08-16', 0,    151.1),
    ("M&M.NS",        2483, '2016-08-16', 0,    748.6),
    ("MARUTI.NS",     2483, '2016-08-16', 0,    466.2),
    ("MOTHERSON.NS",  2483, '2016-08-16', 0,    217.3),
    ("MUTHOOTFIN.NS", 2482, '2016-08-16', 0,    235.6),
    ("NESTLEIND.NS",  2483, '2016-08-16', 0,    257.0),
    ("NTPC.NS",       2483, '2016-08-16', 0,    337.6),
    ("ONGC.NS",       2483, '2016-08-16', 0,    312.2),
    ("PFC.NS",        2483, '2016-08-16', 0,    243.2),
    ("PIDILITIND.NS", 2483, '2016-08-16', 0,    120.5),
    ("PNB.NS",        2483, '2016-08-16', 0,    151.1),
    ("POWERGRID.NS",  2483, '2016-08-16', 0,    240.6),
    ("RECLTD.NS",     2483, '2016-08-16', 0,    169.8),
    ("RELIANCE.NS",   2483, '2016-08-16', 0,   1727.5),
    ("SBIN.NS",       2483, '2016-08-16', 0,   1130.7),
    ("SHREECEM.NS",   2483, '2016-08-16', 0,     47.1),
    ("SHRIRAMFIN.NS", 2483, '2016-08-16', 0,    468.5),
    ("SIEMENS.NS",    2482, '2016-08-16', 1,    123.9),
    ("SOLARINDS.NS",  2482, '2016-08-16', 1,    187.0),
    ("SUNPHARMA.NS",  2483, '2016-08-16', 0,    393.8),
    ("TATACONSUM.NS", 2483, '2016-08-16', 0,    160.2),
    ("TATAPOWER.NS",  2483, '2016-08-16', 0,    180.0),
    ("TATASTEEL.NS",  2483, '2016-08-16', 0,    483.3),
    ("TCS.NS",        2483, '2016-08-16', 0,    774.6),
    ("TECHM.NS",      2483, '2016-08-16', 0,    317.5),
    ("TITAN.NS",      2483, '2016-08-16', 0,    312.7),
    ("TMPV.NS",       2483, '2016-08-16', 0,    292.4),
    ("TORNTPHARM.NS", 2483, '2016-08-16', 0,    161.8),
    ("TRENT.NS",      2483, '2016-08-16', 0,    266.8),
    ("TVSMOTOR.NS",   2483, '2016-08-16', 0,    318.0),
    ("ULTRACEMCO.NS", 2483, '2016-08-16', 0,    286.6),
    ("UNIONBANK.NS",  2483, '2016-08-16', 0,    213.3),
    ("UNITDSPR.NS",   2483, '2016-08-16', 0,     92.6),
    ("VBL.NS",        2428, '2016-11-08', 0,    237.7),
    ("VEDL.NS",       2483, '2016-08-16', 0,    471.6),
    ("WIPRO.NS",      2483, '2016-08-16', 0,    298.2),
    ("ZYDUSLIFE.NS",  2483, '2016-08-16', 0,    122.9),
)

#: The forecasting universe. Frozen 2026-09-01; changing it is a deliberate
#: act that must bump MODEL_VERSION, because every panel-level result is
#: measured over these rows and a different set of rows is a different
#: measurement.
FROZEN_UNIVERSE: tuple[str, ...] = tuple(t for t, *_ in FROZEN_MEASUREMENTS)

#: Recorded alongside results, the way UniverseRule.fingerprint() was.
FROZEN_AS_OF = "2026-09-01"
MIN_SESSIONS = 2_400


def frozen_universe() -> list[str]:
    """The fixed forecasting universe, as a fresh mutable list."""
    return list(FROZEN_UNIVERSE)


def frozen_fingerprint() -> str:
    """
    Stable identifier for THIS universe, for run metadata.

    Includes the member count and a hash of the sorted membership, so a run
    before and after a universe change is distinguishable in experiment_runs.
    A fingerprint naming only the rule would not be: the rule is unchanged
    when the list is edited by hand, which is now the only way it changes.
    """
    import hashlib

    digest = hashlib.sha256(
        "\n".join(sorted(FROZEN_UNIVERSE)).encode()
    ).hexdigest()[:12]
    return f"frozen:{FROZEN_AS_OF}:n={len(FROZEN_UNIVERSE)}:{digest}"
