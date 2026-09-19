"""
tools/backfill_results.py — NSE quarterly results, corporate actions and
parsed EPS for the panel's tickers.

Stage 1, Pilot 3. Resumable and throttled to at most three requests a second.
Writes one local cache, `results_cache.npz` (gitignored by `*.npz`). Touches
no database, and nothing Render or either scheduled job reads.

    python tools/backfill_results.py --limit-symbols 2          # smoke
    python tools/backfill_results.py                            # everything
    python tools/backfill_results.py --report [--markdown out.md]

Read the report before believing anything built on this. It shows coverage
by year, how many old-format pages validated and at which row shift, how
many XBRL EPS agree with their own implied EPS, when results are disclosed
relative to the close, and whether NSE's splits agree with the ones
`corporate_actions` already holds.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from urllib.parse import quote

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.earnings import (  # noqa: E402
    CORP_ACTIONS,
    FIRST_PERIOD_END,
    INTEGRATED_LIST,
    LEGACY_LIST,
    REFERER_ACTIONS,
    REFERER_INTEGRATED,
    REFERER_RESULTS,
    NSEClient,
    action_factor,
    first_disclosures,
    normalise_integrated,
    normalise_legacy,
    parse_actions,
    parse_old_html_eps,
    parse_xbrl_eps,
    timing_class,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
CACHE = os.path.join(ROOT, "results_cache.npz")

SAVE_EVERY = 50
#: A RATE over the last 50 requests, never a lifetime count.
ABORT_WINDOW = 50
ABORT_RATE = 0.20
PAGE_SIZE = 100

TABLES = {
    "lists": ["symbol", "kind", "status", "http", "n_rows", "detail"],
    "filings": ["symbol", "period_start", "period_end", "basis", "source", "doc_url",
                "broadcast", "dissemination", "disclosed", "revised", "seq"],
    "actions": ["symbol", "ex_date", "kind", "factor", "subject"],
    "yf_splits": ["ticker", "date", "ratio"],
    "docs": ["doc_url", "status", "http", "sha256", "eps", "implied", "shift",
             "eps_tag", "parse_status", "via"],
}
DATE_COLS = {"period_start", "period_end", "broadcast", "dissemination", "disclosed",
             "ex_date", "date"}


# ── the cache ─────────────────────────────────────────────────────────────────


def _empty(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=TABLES[name])


def load_cache(path: str = CACHE) -> dict[str, pd.DataFrame]:
    if not os.path.exists(path):
        return {n: _empty(n) for n in TABLES}
    z = np.load(path, allow_pickle=True)
    out = {}
    for name, cols in TABLES.items():
        if f"{name}__{cols[0]}" not in z:
            out[name] = _empty(name)
            continue
        df = pd.DataFrame({c: z[f"{name}__{c}"] for c in cols})
        for c in cols:
            if c in DATE_COLS:
                df[c] = pd.to_datetime(df[c].astype(object).where(df[c] != "", None))
        out[name] = df
    # Factors are re-derived from NSE's own subject text on every load, so a
    # parser fix reaches a cache written before it.
    a = out["actions"]
    if len(a):
        split = a["kind"].isin(["bonus", "split"])
        a.loc[split, "factor"] = a.loc[split, "subject"].map(action_factor)
    return out


def save_cache(tables: dict[str, pd.DataFrame], path: str = CACHE) -> None:
    arrays = {}
    for name, cols in TABLES.items():
        df = tables[name]
        for c in cols:
            s = df[c] if c in df else pd.Series([None] * len(df))
            if c in DATE_COLS:
                s = pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
            arrays[f"{name}__{c}"] = s.astype(object).to_numpy()
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def panel_tickers(panel_cache: str = PANEL_CACHE) -> list[str]:
    return sorted(pd.read_parquet(panel_cache, columns=["ticker"])["ticker"].unique())


# ── the backfill ──────────────────────────────────────────────────────────────


def _json(f) -> object:
    import json
    return json.loads(f.content.decode("utf-8", "replace"))


def fetch_lists(client: NSEClient, symbol: str, today: str) -> tuple[list[dict], list]:
    """The three list calls for one symbol -> (list-status rows, [legacy,
    integrated, actions] frames)."""
    enc = quote(symbol, safe="")
    status, frames = [], []

    f = client.fetch(LEGACY_LIST.format(sym=enc), REFERER_RESULTS)
    rows = _json(f) if f.status == "ok" else []
    rows = rows if isinstance(rows, list) else []
    status.append({"symbol": symbol, "kind": "legacy", "status": f.status, "http": f.http,
                   "n_rows": len(rows), "detail": f.detail})
    frames.append(normalise_legacy(rows))

    got, page, total, st = [], 1, None, "ok"
    while True:
        f = client.fetch(INTEGRATED_LIST.format(sym=enc, page=page, size=PAGE_SIZE),
                         REFERER_INTEGRATED)
        if f.status != "ok":
            st = f.status
            break
        body = _json(f)
        data = body.get("data", []) if isinstance(body, dict) else []
        total = body.get("totalCount", len(data)) if isinstance(body, dict) else len(data)
        got.extend(data)
        if not data or len(got) >= (total or 0):
            break
        page += 1
    status.append({"symbol": symbol, "kind": "integrated", "status": st, "http": f.http,
                   "n_rows": len(got), "detail": f.detail})
    frames.append(normalise_integrated(got))

    f = client.fetch(CORP_ACTIONS.format(sym=enc, to=today), REFERER_ACTIONS)
    rows = _json(f) if f.status == "ok" else []
    rows = rows if isinstance(rows, list) else []
    status.append({"symbol": symbol, "kind": "actions", "status": f.status, "http": f.http,
                   "n_rows": len(rows), "detail": f.detail})
    frames.append(parse_actions(rows, symbol))
    return status, frames


def parse_doc(content: bytes, row: pd.Series) -> dict:
    text = content.decode("utf-8", "replace")
    if row["source"] == "legacy_html":
        r = parse_old_html_eps(text, row["basis"])
        return {"eps": r["eps"], "implied": r["implied"], "shift": r["shift"],
                "eps_tag": "html", "parse_status": r["status"], "via": "html"}
    r = parse_xbrl_eps(text, pd.Timestamp(row["period_start"]), pd.Timestamp(row["period_end"]))
    return {"eps": r["eps"], "implied": r["implied"], "shift": None,
            "eps_tag": r["eps_tag"], "parse_status": r["status"], "via": r["via"]}


def backfill(tickers: list[str], path: str = CACHE, client: NSEClient | None = None,
             limit_symbols: int | None = None, limit_docs: int | None = None,
             reparse: bool = False) -> None:
    client = client or NSEClient()
    T = load_cache(path)
    today = pd.Timestamp.today().strftime("%d-%m-%Y")
    symbols = [t.replace(".NS", "") for t in tickers]
    if limit_symbols is not None:
        symbols = symbols[:limit_symbols]

    done = set()
    if len(T["lists"]):
        ok = T["lists"][T["lists"]["status"] == "ok"].groupby("symbol")["kind"].nunique()
        done = set(ok[ok == 3].index)
    todo = [s for s in symbols if s not in done]
    print(f"lists: {len(done)} symbols cached, fetching {len(todo)}", flush=True)
    recent: deque[bool] = deque(maxlen=ABORT_WINDOW)
    for i, sym in enumerate(todo, 1):
        status, (leg, integ, acts) = fetch_lists(client, sym, today)
        recent.extend(s["status"] in ("blocked", "error") for s in status)
        T["lists"] = pd.concat([T["lists"][T["lists"]["symbol"] != sym],
                                pd.DataFrame(status)], ignore_index=True)
        new = pd.concat([leg, integ], ignore_index=True)
        new["symbol"] = sym
        T["filings"] = pd.concat([T["filings"][T["filings"]["symbol"] != sym], new],
                                 ignore_index=True)
        T["actions"] = pd.concat([T["actions"][T["actions"]["symbol"] != sym], acts],
                                 ignore_index=True)
        print(f"  {sym}: legacy {status[0]['n_rows']} ({status[0]['status']}), "
              f"integrated {status[1]['n_rows']} ({status[1]['status']}), "
              f"actions {status[2]['n_rows']} ({status[2]['status']})", flush=True)
        _abort_if(recent, T, path)
        if i % 10 == 0:
            save_cache(T, path)
    save_cache(T, path)

    chosen = selected_filings(T["filings"])
    chosen = chosen[chosen["symbol"].isin(symbols)]
    d = T["docs"]
    have = set(d.loc[d["status"].isin(["ok", "not_found"]), "doc_url"])
    if reparse:
        # pages an older parser could not read are fetched and read again
        have -= set(d.loc[(d["status"] == "ok") & d["eps"].isna(), "doc_url"])
    todo_docs = chosen[~chosen["doc_url"].isin(have)]
    if limit_docs is not None:
        todo_docs = todo_docs.head(limit_docs)
    print(f"documents: {len(chosen)} first disclosures, {len(have)} cached, "
          f"fetching {len(todo_docs)}", flush=True)
    new_docs = []
    for i, (_, row) in enumerate(todo_docs.iterrows(), 1):
        f = client.fetch(row["doc_url"])
        rec = {"doc_url": row["doc_url"], "status": f.status, "http": f.http,
               "sha256": f.sha256, "eps": None, "implied": None, "shift": None,
               "eps_tag": None, "parse_status": None, "via": None}
        if f.status == "ok":
            try:
                rec.update(parse_doc(f.content, row))
            except Exception as exc:                            # noqa: BLE001
                rec["parse_status"] = f"parse_error: {type(exc).__name__}"
        new_docs.append(rec)
        recent.append(f.status in ("blocked", "error"))
        tripped = len(recent) == ABORT_WINDOW and float(np.mean(recent)) > ABORT_RATE
        if tripped or i % SAVE_EVERY == 0 or i == len(todo_docs):
            nd = pd.DataFrame(new_docs, columns=TABLES["docs"])
            T["docs"] = pd.concat([T["docs"][~T["docs"]["doc_url"].isin(nd["doc_url"])], nd],
                                  ignore_index=True)
            new_docs = []
            save_cache(T, path)
            print(f"  {i}/{len(todo_docs)} documents; statuses "
                  f"{T['docs']['status'].value_counts().to_dict()}; parse "
                  f"{T['docs']['parse_status'].value_counts().to_dict()}", flush=True)
        if tripped:
            raise SystemExit(f"more than {ABORT_RATE:.0%} of the last {ABORT_WINDOW} "
                             "requests were blocked or failed; stopped and saved. "
                             "Re-run to resume.")


def _abort_if(recent, T, path):
    if len(recent) == ABORT_WINDOW and float(np.mean(recent)) > ABORT_RATE:
        save_cache(T, path)
        raise SystemExit(f"more than {ABORT_RATE:.0%} of the last {ABORT_WINDOW} "
                         "requests were blocked or failed; stopped and saved.")


def selected_filings(filings: pd.DataFrame) -> pd.DataFrame:
    f = filings[pd.to_datetime(filings["period_end"]) >= FIRST_PERIOD_END]
    return first_disclosures(f)


def eps_table(T: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """First disclosures joined to their parsed EPS: (symbol, period_end,
    basis, source, disclosed, eps)."""
    chosen = selected_filings(T["filings"])
    docs = T["docs"].drop_duplicates("doc_url", keep="last")
    j = chosen.merge(docs[["doc_url", "status", "eps", "parse_status", "implied", "shift",
                           "via"]],
                     on="doc_url", how="left")
    j["eps"] = pd.to_numeric(j["eps"], errors="coerce")
    return j


# ── the report ────────────────────────────────────────────────────────────────


def report(path: str = CACHE, panel_cache: str = PANEL_CACHE,
           yf_splits: pd.DataFrame | None = None) -> str:
    T = load_cache(path)
    grid = sorted(pd.read_parquet(panel_cache, columns=["date"])["date"].astype(str).unique())
    E = eps_table(T)
    o = ["## Results ingestion\n"]
    o.append(f"List calls: {T['lists'].groupby(['kind', 'status']).size().to_dict()}.")
    o.append(f"Filings listed: {len(T['filings']):,}; first disclosures from "
             f"{FIRST_PERIOD_END.date()}: {len(E):,} "
             f"({E['source'].value_counts().to_dict()}).")
    o.append(f"Document fetches: {T['docs']['status'].value_counts().to_dict()}; parse "
             f"{T['docs']['parse_status'].value_counts().to_dict()}.\n")

    E["year"] = pd.to_datetime(E["period_end"]).dt.year
    o.append("| period-end year | first disclosures | EPS parsed | share | consolidated | standalone |")
    o.append("|---|---|---|---|---|---|")
    for y, g in E.groupby("year"):
        o.append(f"| {y} | {len(g):,} | {int(g['eps'].notna().sum()):,} | "
                 f"{g['eps'].notna().mean():.1%} | {int((g['basis'] == 'consolidated').sum())} | "
                 f"{int((g['basis'] == 'standalone').sum())} |")

    old = E[E["source"] == "legacy_html"]
    if len(old):
        o.append(f"\nOld-format pages: {old['parse_status'].value_counts().to_dict()}; "
                 f"accepted at shift {pd.to_numeric(old['shift'], errors='coerce').value_counts().to_dict()}.")
    xb = E[(E["source"] != "legacy_html") & E["eps"].notna()].copy()
    o.append(f"XBRL EPS by context route: {xb['via'].value_counts().to_dict()}.")
    xb["implied"] = pd.to_numeric(xb["implied"], errors="coerce")
    xi = xb.dropna(subset=["implied"])
    if len(xi):
        agree = (xi["eps"] - xi["implied"]).abs() <= np.maximum(0.15 * xi["implied"].abs(), 0.10)
        o.append(f"XBRL EPS with an implied EPS available: {len(xi):,} of {len(xb):,}; "
                 f"within 15%: {agree.mean():.1%}.")

    sessions = set(grid)
    tc = E.dropna(subset=["disclosed"])
    tc = tc[tc["disclosed"].dt.strftime("%Y-%m-%d").between(grid[0], grid[-1])].assign(
        cls=lambda x: [timing_class(t, sessions) for t in x["disclosed"]])
    o.append(f"\nDisclosure timing (first disclosures): {tc['cls'].value_counts().to_dict()}.")

    per = E.groupby("symbol")["eps"].apply(lambda s: s.notna().mean())
    o.append("Lowest per-symbol EPS coverage: "
             + ", ".join(f"{k} {v:.0%}" for k, v in per.nsmallest(10).items()) + ".")

    a = T["actions"]
    o.append(f"\nNSE corporate actions kept: {a['kind'].value_counts().to_dict()}; "
             f"demergers: {a.loc[a['kind'] == 'demerger', ['symbol', 'ex_date']].astype(str).values.tolist()}.")
    if yf_splits is not None:
        o.append(split_crosscheck(a, yf_splits))
    return "\n".join(o)


def split_crosscheck(nse: pd.DataFrame, yf: pd.DataFrame, since: str = "2012-01-01") -> str:
    """NSE bonuses/splits against `corporate_actions` (yfinance) since
    `since`: matched when the ex-dates are within 5 days and the ratios within
    1%."""
    n = nse[nse["kind"].isin(["bonus", "split"]) & (nse["ex_date"] >= since)].copy()
    y = yf[pd.to_datetime(yf["date"]) >= since].copy()
    y["symbol"] = y["ticker"].str.replace(".NS", "", regex=False)
    y["date"] = pd.to_datetime(y["date"])
    matched, only_nse = 0, []
    used = set()
    for _, r in n.iterrows():
        c = y[(y["symbol"] == r["symbol"]) & ((y["date"] - r["ex_date"]).abs().dt.days <= 5)
              & (np.abs(y["ratio"].astype(float) / float(r["factor"]) - 1) < 0.01)]
        if len(c):
            matched += 1
            used.add(c.index[0])
        else:
            only_nse.append(f"{r['symbol']} {r['ex_date'].date()} x{r['factor']:g}")
    only_yf = [f"{r['symbol']} {r['date'].date()} x{float(r['ratio']):g}"
               for i, r in y.iterrows() if i not in used and r["symbol"] in set(nse["symbol"])]
    return (f"Split/bonus cross-check since {since}: {matched} of {len(n)} NSE actions "
            f"match yfinance. Only in NSE: {only_nse or 'none'}. "
            f"Only in yfinance: {only_yf or 'none'}.")


def load_yf_splits() -> pd.DataFrame | None:
    """Read-only: the splits `corporate_actions` already holds."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from data.db import get_engine
        return pd.read_sql("SELECT ticker, date, ratio FROM corporate_actions "
                           "WHERE action_type = 'SPLIT'", get_engine())
    except Exception as exc:                                    # noqa: BLE001
        print(f"(split cross-check skipped: {type(exc).__name__})")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--limit-symbols", type=int, default=None)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reparse", action="store_true",
                    help="re-fetch and re-read documents that yielded no EPS")
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--snapshot-splits", action="store_true",
                    help="copy corporate_actions' splits (read-only) into the cache")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    if args.snapshot_splits:
        T = load_cache(args.cache)
        yf = load_yf_splits()
        if yf is None:
            raise SystemExit("could not read corporate_actions; nothing snapshotted")
        T["yf_splits"] = yf[["ticker", "date", "ratio"]]
        save_cache(T, args.cache)
        print(f"snapshotted {len(yf)} yfinance splits into {args.cache}")
        return
    if not args.report:
        backfill(panel_tickers(args.panel_cache), args.cache,
                 limit_symbols=args.limit_symbols, limit_docs=args.limit_docs,
                 reparse=args.reparse)
    text = report(args.cache, args.panel_cache,
                  None if args.no_crosscheck else load_yf_splits())
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
