"""
pipeline/delivery.py — NSE security-wise delivery position, point-in-time.

Stage 1, Pilot 2. The method and every decision rule are fixed in
`docs/stage1b-preregistration.md`, written before any model saw these columns.

THE SOURCE IS THE DAILY MTO FILE, NOT THE BHAVCOPY
--------------------------------------------------
Measured 2026-09-13 against nsearchives.nseindia.com:

* The UDiFF CM bhavcopy that replaced the old one in July 2024 (NSE circular
  62424, `BhavCopy_NSE_CM_0_0_0_<yyyymmdd>_F_0000.csv.zip`) carries NO
  delivery column at all.
* `sec_bhavdata_full_<ddmmyyyy>.csv` does carry DELIV_QTY / DELIV_PER. It is
  what both maintained libraries fetch (jugaad-data 0.35.5's
  `full_bhavcopy`, and `nse` 4.0.1's `deliveryBhavcopy`), but the archive
  answers 404 for it in 2016 and 2019. It only reaches back to mid-2024, and
  this panel starts 2016-10-28.
* `MTO_<ddmmyyyy>.DAT`, the Security-Wise Delivery Position, is served for
  every session tested from 2016-10-28 to 2026-09-11, in ONE unchanged
  format: a `10,MTO,<ddmmyyyy>,...` header, then
  `20,<sr>,<symbol>,<series>,<qty traded>,<deliverable qty>,<% deliverable>`.
  On both post-UDiFF dates cross-checked (2024-07-08 and 2026-09-11), its
  deliverable quantity and percentage equal `sec_bhavdata_full`'s for every
  EQ symbol, to the digit.

So this reads MTO files directly, with `requests`, which is already a
dependency. The archive served them WITHOUT cookies from this machine, while
www.nseindia.com's landing page answered a scripted client with 403. A 403 on
the archive therefore triggers ONE landing-page visit (for cookies, as
jugaad-data's own fallback does) and one retry, and is otherwise recorded as
BLOCKED.

WHAT A MISSING FILE MEANS
-------------------------
NSE answers 404 for a date with no session (measured on the 2025-10-02 and
2026-08-15 holidays). Requests here are made only for the panel's own
trading dates, though, so a 404 on one of them is recorded as `not_found`,
not as a holiday, and the coverage report lists every one. `blocked`,
`error` and `malformed` are never read as "no delivery that day". A session
with no EQ row for a ticker (halted, suspended, trading only in BE) is
missing, and stays missing.

POINT-IN-TIME
-------------
The file for session T is published after that session's close (roughly
4-6 pm IST). So it is usable from session T+1: every feature at t is built
from delivery of t-1 and earlier, on the panel's OWN session grid. A missing
session stays NaN. Nothing is forward-filled, because a stale delivery
figure presented as today's would be the FII/DII failure again: a failed
measurement stored as a valid value.

SYMBOLS CHANGE
--------------
The file names a security by its symbol ON THAT DATE. Nine universe names
were renamed inside the panel's window (CROMPGREAV->CGPOWER, LTI->LTIM->LTM,
TATAMOTORS->TMPV, ...). `symbol_on` walks NSE's own symbolchange.csv
backwards from the current symbol, so a ticker's early history is not
silently empty.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ARCHIVE = "https://nsearchives.nseindia.com"
MTO_PATH = "/archives/equities/mto/MTO_{ddmmyyyy}.DAT"
SYMBOL_CHANGE_URL = ARCHIVE + "/content/equities/symbolchange.csv"
HOMEPAGE = "https://www.nseindia.com/"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

#: At most three requests a second, whatever the network does.
MIN_REQUEST_INTERVAL = 1.0 / 3.0
REQUEST_TIMEOUT = 30
RETRIES = 3

SERIES = "EQ"

#: The trailing baseline an abnormal-delivery figure is measured against, and
#: the observations it needs before it is defined.
TRAILING_WINDOW = 60
TRAILING_MIN = 40
SHORT_WINDOW = 5

LEVEL_COL = "deliv_pct_l1"      # yesterday's delivery %, raw
ABN_COL = "deliv_abn_l1"        # yesterday's, minus its own trailing 60-session mean
ABN5_COL = "deliv_abn5_l1"      # the last 5 sessions' mean, minus the same baseline
ABNORMAL_COLS = [ABN_COL, ABN5_COL]
DELIVERY_COLS = [LEVEL_COL] + ABNORMAL_COLS

RECORD_COLUMNS = ["record", "sr", "symbol", "series", "qty_traded",
                  "deliv_qty", "deliv_pct"]


class MTOFormatError(ValueError):
    """The file is not the MTO format this parser was validated against."""


# ── parsing ───────────────────────────────────────────────────────────────────


def mto_url(day: pd.Timestamp) -> str:
    return ARCHIVE + MTO_PATH.format(ddmmyyyy=pd.Timestamp(day).strftime("%d%m%Y"))


def parse_mto(text: str, expected: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    One MTO file -> (symbol, series, qty_traded, deliv_qty, deliv_pct).

    REFUSES rather than guesses. A file whose header names a different date
    raises, as does a record with other than seven fields: one comma in a
    field would shift every column after it. `pipeline.macro` learned the
    date half of this when NSE was found to accept `?date=` and ignore it.
    """
    lines = text.splitlines()
    header = next((ln for ln in lines if ln.startswith("10,MTO,")), None)
    if header is None:
        raise MTOFormatError("no '10,MTO,<date>' header record")
    file_day = pd.to_datetime(header.split(",")[2].strip(), format="%d%m%Y")
    if expected is not None and file_day != pd.Timestamp(expected).normalize():
        raise MTOFormatError(f"file is for {file_day.date()}, asked for "
                             f"{pd.Timestamp(expected).date()}")

    rows = []
    for ln in lines:
        if not ln.startswith("20,"):
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(RECORD_COLUMNS):
            raise MTOFormatError(f"record with {len(parts)} fields: {ln[:80]!r}")
        rows.append(parts)
    if not rows:
        raise MTOFormatError("header present but no '20,' records")

    df = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    for col in ("qty_traded", "deliv_qty", "deliv_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["file_date"] = file_day
    return df[["symbol", "series", "qty_traded", "deliv_qty", "deliv_pct"]]


def parse_symbol_changes(text: str) -> pd.DataFrame:
    """NSE's symbolchange.csv -> (old, new, date). The company name may itself
    contain commas, so each line is split from the RIGHT."""
    rows = []
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.rsplit(",", 3)]
        if len(parts) == 4:
            rows.append(parts[1:])
    df = pd.DataFrame(rows, columns=["old", "new", "date"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)


def rename_chain(current: str, changes: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    """
    The renames that led to `current`, newest first: [(effective date, old
    symbol), ...].

    A rename old -> new effective on D means every session before D used
    `old`. Each step takes the latest rename into the current symbol that
    predates the step before, so a symbol later reused by some other company
    cannot splice a stranger's history in.
    """
    chain: list[tuple[pd.Timestamp, str]] = []
    sym, bound, seen = current, pd.Timestamp.max, {current}
    while True:
        into = changes[(changes["new"] == sym) & (changes["date"] < bound)]
        if into.empty:
            return chain
        row = into.sort_values("date").iloc[-1]
        if row["old"] in seen:              # a cycle in the file: stop, do not loop
            return chain
        chain.append((row["date"], row["old"]))
        sym, bound = row["old"], row["date"]
        seen.add(sym)


def symbol_on(current: str, day: pd.Timestamp, changes: pd.DataFrame,
              chain: list[tuple[pd.Timestamp, str]] | None = None) -> str:
    """The symbol `current` traded under on `day`."""
    day = pd.Timestamp(day)
    sym = current
    for effective, old in (rename_chain(current, changes) if chain is None else chain):
        if day < effective:
            sym = old
        else:
            break
    return sym


def symbol_history(current: str, changes: pd.DataFrame) -> set[str]:
    """Every symbol `current` has traded under, as far back as the file goes."""
    return {current} | {old for _, old in rename_chain(current, changes)}


# ── fetching ──────────────────────────────────────────────────────────────────


@dataclass
class FetchResult:
    day: pd.Timestamp
    status: str                      # ok | not_found | blocked | error | malformed
    http: int | None = None
    sha256: str | None = None
    frame: pd.DataFrame | None = None
    detail: str = ""


@dataclass
class ArchiveClient:
    """
    A throttled session for nsearchives.nseindia.com.

    ``sleep`` and ``clock`` are injectable so the throttle is testable
    without a network or a wall clock.
    """
    session: object = None
    min_interval: float = MIN_REQUEST_INTERVAL
    sleep: object = time.sleep
    clock: object = time.monotonic
    _last: float = field(default=-1e9, repr=False)
    _warmed: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.session is None:
            import requests

            self.session = requests.Session()
            self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*",
                                         "Referer": HOMEPAGE})

    def _throttle(self) -> None:
        wait = self.min_interval - (self.clock() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = self.clock()

    def get(self, url: str):
        self._throttle()
        r = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 403 and not self._warmed:
            # One landing-page visit for cookies, then one retry. Its own
            # status is not trusted: it answered 403 on 2026-09-13.
            self._warmed = True
            self._throttle()
            try:
                self.session.get(HOMEPAGE, timeout=REQUEST_TIMEOUT)
            except Exception:                                   # noqa: BLE001
                pass
            self._throttle()
            r = self.session.get(url, timeout=REQUEST_TIMEOUT)
        return r

    def fetch_mto(self, day: pd.Timestamp) -> FetchResult:
        day = pd.Timestamp(day).normalize()
        last_exc = ""
        for attempt in range(RETRIES):
            try:
                r = self.get(mto_url(day))
            except Exception as exc:                            # noqa: BLE001
                last_exc = f"{type(exc).__name__}: {exc}"[:200]
                self.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                digest = hashlib.sha256(r.content).hexdigest()
                try:
                    frame = parse_mto(r.content.decode("utf-8", "replace"), day)
                except MTOFormatError as exc:
                    return FetchResult(day, "malformed", 200, digest, None, str(exc))
                return FetchResult(day, "ok", 200, digest, frame)
            if r.status_code == 404:
                return FetchResult(day, "not_found", 404)
            if r.status_code == 403:
                return FetchResult(day, "blocked", 403)
            last_exc = f"HTTP {r.status_code}"
            self.sleep(2.0 * (attempt + 1))
        return FetchResult(day, "error", None, detail=last_exc)

    def fetch_symbol_changes(self) -> str:
        r = self.get(SYMBOL_CHANGE_URL)
        r.raise_for_status()
        return r.content.decode("utf-8", "replace")


# ── from files to the panel ───────────────────────────────────────────────────


def to_panel_tickers(records: pd.DataFrame, tickers: list[str],
                     changes: pd.DataFrame) -> pd.DataFrame:
    """
    (date, symbol, series, ...) as stored -> (date, ticker, deliv_pct, ...).

    Each panel ticker is matched on the symbol it traded under ON THAT DATE,
    and on the EQ series only. A date with no matching EQ row gives no row.
    """
    rec = records[records["series"] == SERIES]
    by_day = {d: g.set_index("symbol") for d, g in rec.groupby("date")}
    out = []
    for ticker in tickers:
        base = ticker.replace(".NS", "")
        for day, g in by_day.items():
            sym = symbol_on(base, pd.Timestamp(day), changes)
            if sym in g.index:
                row = g.loc[sym]
                if isinstance(row, pd.DataFrame):     # a duplicated symbol: refuse
                    continue
                out.append((day, ticker, sym, float(row["deliv_pct"]),
                            float(row["deliv_qty"]), float(row["qty_traded"])))
    return pd.DataFrame(out, columns=["date", "ticker", "symbol", "deliv_pct",
                                      "deliv_qty", "qty_traded"])


def delivery_features(delivery: pd.DataFrame, grid: list[str]) -> pd.DataFrame:
    """
    (date, ticker, deliv_pct) -> (date, ticker, LEVEL_COL, ABN_COL, ABN5_COL)
    on the panel's session grid.

    Everything at t is built from sessions t-1 and earlier: the file for t is
    published after t's close. Missing sessions stay NaN, and a rolling mean
    over them uses the observations that exist (subject to its minimum),
    never a value carried forward.
    """
    cols = ["date", "ticker"] + DELIVERY_COLS
    if delivery.empty:
        return pd.DataFrame(columns=cols)
    wide = (delivery.pivot_table(index="date", columns="ticker",
                                 values="deliv_pct", aggfunc="first")
            .reindex(sorted(grid)))
    known = wide.shift(1)                                   # published after t-1's close
    base = known.rolling(TRAILING_WINDOW, min_periods=TRAILING_MIN).mean()
    short = known.rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    frames = {LEVEL_COL: known, ABN_COL: known - base, ABN5_COL: short - base}
    long = pd.concat({c: frames[c].stack(future_stack=True) for c in DELIVERY_COLS},
                     axis=1)
    long.index = long.index.set_names(["date", "ticker"])
    return long.reset_index().replace([np.inf, -np.inf], np.nan)[cols]


def rank_persistence(features: pd.DataFrame, col: str, lag: int = 250) -> float:
    """
    Correlation of a column's within-date RANK with itself `lag` sessions on.

    Outcome-blind. It is the check that exposed valuation as a persistent
    per-ticker quantity (+0.813 at 250 sessions): the kind of feature a tree
    uses to recognise WHICH COMPANY it is, which earned `pooled_xgb` a mean
    rebalance t of +0.77 from pure noise.
    """
    wide = features.pivot_table(index="date", columns="ticker", values=col)
    ranks = wide.rank(axis=1, pct=True)
    a = ranks.iloc[:-lag].to_numpy().ravel()
    b = ranks.iloc[lag:].to_numpy().ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
