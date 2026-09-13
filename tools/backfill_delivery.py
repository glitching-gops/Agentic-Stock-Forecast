"""
tools/backfill_delivery.py — NSE's daily MTO delivery files for the panel's sessions.

Stage 1, Pilot 2. Resumable and throttled to at most three requests a
second. It writes one local cache, `delivery_cache.npz` (gitignored by
`*.npz`), and touches no database and nothing Render or either scheduled job
reads.

    python tools/backfill_delivery.py --limit 5                 # smoke
    python tools/backfill_delivery.py                           # ~2,440 sessions
    python tools/backfill_delivery.py --report                  # coverage only
    python tools/backfill_delivery.py --validate 2026-09-11 2024-07-08

The coverage report is the thing to read before believing any feature built
on this: holes concentrated in the early panel would manufacture exactly the
early-fold artifact this project has been fooled by six times.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.delivery import (  # noqa: E402
    ARCHIVE,
    SERIES,
    ArchiveClient,
    mto_url,
    parse_mto,
    parse_symbol_changes,
    symbol_history,
    to_panel_tickers,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
CACHE = os.path.join(ROOT, "delivery_cache.npz")

SAVE_EVERY = 50
#: A RATE over the last 50 requests, never a lifetime count. A lifetime
#: threshold is what stopped the first news backfill at ticker 7 of 84, on
#: the ordinary transient-failure rate.
ABORT_WINDOW = 50
ABORT_RATE = 0.20

REC_COLS = ["date", "symbol", "series", "qty_traded", "deliv_qty", "deliv_pct"]
MAN_COLS = ["date", "status", "http", "sha256", "n_records", "detail"]


# ── the cache ─────────────────────────────────────────────────────────────────


def load_cache(path: str = CACHE) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not os.path.exists(path):
        return (pd.DataFrame(columns=REC_COLS), pd.DataFrame(columns=MAN_COLS), "")
    z = np.load(path, allow_pickle=True)
    rec = pd.DataFrame({c: z[f"rec_{c}"] for c in REC_COLS})
    man = pd.DataFrame({c: z[f"man_{c}"] for c in MAN_COLS})
    return rec, man, str(z["symbolchange"][0])


def save_cache(rec: pd.DataFrame, man: pd.DataFrame, symbolchange: str,
               path: str = CACHE) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        **{f"rec_{c}": rec[c].to_numpy() for c in REC_COLS},
        **{f"man_{c}": man[c].astype(object).to_numpy() for c in MAN_COLS},
        symbolchange=np.array([symbolchange], dtype=object))
    os.replace(tmp, path)


def grid_and_tickers(panel_cache: str = PANEL_CACHE) -> tuple[list[str], list[str],
                                                             pd.DataFrame]:
    panel = pd.read_parquet(panel_cache, columns=["date", "ticker"])
    panel["date"] = panel["date"].astype(str)
    return (sorted(panel["date"].unique()), sorted(panel["ticker"].unique()), panel)


# ── the backfill ──────────────────────────────────────────────────────────────


def backfill(grid: list[str], tickers: list[str], limit: int | None = None,
             path: str = CACHE, client: ArchiveClient | None = None) -> None:
    client = client or ArchiveClient()
    rec, man, sc_text = load_cache(path)
    if not sc_text:
        sc_text = client.fetch_symbol_changes()
    changes = parse_symbol_changes(sc_text)
    wanted = set().union(*(symbol_history(t.replace(".NS", ""), changes)
                           for t in tickers))

    done = set(man.loc[man["status"] == "ok", "date"])
    todo = [d for d in grid if d not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(done)} sessions already cached; fetching {len(todo)} "
          f"({len(wanted)} symbols across {len(tickers)} tickers' histories)",
          flush=True)

    recent: deque[bool] = deque(maxlen=ABORT_WINDOW)
    new_rec, new_man = [], []
    for i, day in enumerate(todo, 1):
        res = client.fetch_mto(pd.Timestamp(day))
        n = 0
        if res.status == "ok":
            f = res.frame
            f = f[f["symbol"].isin(wanted) & (f["series"] == SERIES)].assign(date=day)
            n = len(f)
            new_rec.append(f[REC_COLS])
        new_man.append({"date": day, "status": res.status, "http": res.http,
                        "sha256": res.sha256, "n_records": n, "detail": res.detail})
        recent.append(res.status in ("blocked", "error"))

        tripped = (len(recent) == ABORT_WINDOW
                   and float(np.mean(recent)) > ABORT_RATE)
        if tripped or i % SAVE_EVERY == 0 or i == len(todo):
            rec, man = _merge(rec, man, new_rec, new_man)
            new_rec, new_man = [], []
            save_cache(rec, man, sc_text, path)
            counts = man["status"].value_counts().to_dict()
            print(f"  {i}/{len(todo)} fetched; cache {len(rec):,} records; "
                  f"statuses {counts}", flush=True)
        if tripped:
            raise SystemExit(
                f"more than {ABORT_RATE:.0%} of the last {ABORT_WINDOW} requests "
                f"were blocked or failed; stopped and saved. Re-run to resume.")


def _merge(rec, man, new_rec, new_man):
    if new_rec:
        rec = pd.concat([rec] + new_rec, ignore_index=True)
    if new_man:
        m = pd.DataFrame(new_man)
        man = pd.concat([man[~man["date"].isin(m["date"])], m], ignore_index=True)
    rec = rec.drop_duplicates(subset=["date", "symbol", "series"], keep="last")
    return rec.sort_values(["date", "symbol"]).reset_index(drop=True), \
        man.sort_values("date").reset_index(drop=True)


# ── coverage ──────────────────────────────────────────────────────────────────


def coverage_report(path: str = CACHE, panel_cache: str = PANEL_CACHE) -> str:
    rec, man, sc_text = load_cache(path)
    grid, tickers, panel = grid_and_tickers(panel_cache)
    changes = parse_symbol_changes(sc_text)
    long = to_panel_tickers(rec, tickers, changes)
    cells = panel.merge(long[["date", "ticker", "deliv_pct"]],
                        on=["date", "ticker"], how="left")
    cells["year"] = cells["date"].str[:4]
    o = ["## Delivery coverage\n",
         f"Sessions on the panel grid: {len(grid):,}; manifest rows {len(man):,}.",
         f"Statuses: {man['status'].value_counts().to_dict()}.\n"]
    nf = man.loc[man["status"] != "ok", ["date", "status", "http", "detail"]]
    if len(nf):
        o.append("Sessions without a usable file:\n")
        o.append(nf.to_string(index=False))
        o.append("")
    o.append("| year | panel (date, ticker) rows | with an EQ delivery value | share |")
    o.append("|---|---|---|---|")
    for year, g in cells.groupby("year"):
        o.append(f"| {year} | {len(g):,} | {int(g['deliv_pct'].notna().sum()):,} | "
                 f"{g['deliv_pct'].notna().mean():.1%} |")
    per = cells.groupby("ticker")["deliv_pct"].apply(lambda s: s.notna().mean())
    o.append(f"\nLowest per-ticker coverage: "
             + ", ".join(f"{t} {v:.1%}" for t, v in per.nsmallest(8).items()))
    renamed = long[long["symbol"] != long["ticker"].str.replace(".NS", "", regex=False)]
    o.append(f"\nRows matched under a FORMER symbol: {len(renamed):,} "
             f"({renamed['ticker'].nunique()} tickers: "
             f"{sorted(renamed['ticker'].unique())}).")
    return "\n".join(o)


# ── validation against the post-July-2024 files ───────────────────────────────


def validate(days: list[str], tickers: list[str],
             client: ArchiveClient | None = None) -> str:
    """
    For each date: MTO against sec_bhavdata_full (the delivery figures must be
    identical), and against the UDiFF CM bhavcopy (it must carry every
    universe symbol, must carry no delivery column, and must agree on the
    close with sec_bhavdata_full). A disagreement is printed, not
    reconciled.
    """
    client = client or ArchiveClient()
    syms = [t.replace(".NS", "") for t in tickers]
    o = ["## Validation against the UDiFF-era files\n"]
    for day in days:
        d = pd.Timestamp(day)
        res = client.fetch_mto(d)
        if res.status != "ok":
            o.append(f"- {day}: MTO {res.status}")
            continue
        mto = res.frame[res.frame["series"] == SERIES]
        sec = pd.read_csv(io.BytesIO(client.get(
            f"{ARCHIVE}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv").content),
            skipinitialspace=True)
        sec.columns = [c.strip() for c in sec.columns]
        sec = sec[sec["SERIES"].str.strip() == SERIES].assign(
            SYMBOL=lambda x: x["SYMBOL"].str.strip())
        for c in ("DELIV_QTY", "DELIV_PER", "CLOSE_PRICE"):
            sec[c] = pd.to_numeric(sec[c], errors="coerce")
        z = zipfile.ZipFile(io.BytesIO(client.get(
            f"{ARCHIVE}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip").content))
        udiff = pd.read_csv(z.open(z.namelist()[0]))
        ueq = udiff[udiff["SctySrs"] == SERIES]

        j = mto.merge(sec, left_on="symbol", right_on="SYMBOL")
        k = sec.merge(ueq, left_on="SYMBOL", right_on="TckrSymb")
        o.append(
            f"- **{day}**: MTO EQ {len(mto)}, sec_bhavdata_full EQ {len(sec)}, "
            f"UDiFF EQ {len(ueq)}. Deliverable qty identical on "
            f"{(j['deliv_qty'] == j['DELIV_QTY']).mean():.2%} of {len(j)} joined; "
            f"max abs % gap {(j['deliv_pct'] - j['DELIV_PER']).abs().max():.3f}. "
            f"UDiFF delivery columns: {[c for c in udiff.columns if 'eliv' in c] or 'none'}. "
            f"Close equal UDiFF vs sec on {(k['CLOSE_PRICE'] == k['ClsPric']).mean():.2%}. "
            f"Universe present — MTO {sum(s in set(mto['symbol']) for s in syms)}/84, "
            f"UDiFF {sum(s in set(ueq['TckrSymb']) for s in syms)}/84.")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--validate", nargs="*", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    grid, tickers, _ = grid_and_tickers(args.panel_cache)
    out = []
    if args.validate is not None:
        out.append(validate(args.validate or ["2026-09-11", "2024-07-08"], tickers))
    elif not args.report:
        backfill(grid, tickers, limit=args.limit, path=args.cache)
    if os.path.exists(args.cache) and args.validate is None:
        out.append(coverage_report(args.cache, args.panel_cache))
    text = "\n\n".join(out)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
