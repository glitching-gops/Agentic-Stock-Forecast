"""
pipeline/earnings.py — reported quarterly EPS from NSE, point-in-time, and the
seasonal-random-walk SUE built from it.

Stage 1, Pilot 3. The method and every decision rule are fixed in
`docs/stage1c-preregistration.md`, written before any model saw these columns.

THE SOURCE IS NSE's OWN RESULTS FILINGS, IN THREE ERAS
------------------------------------------------------
Measured 2026-09-19 against www.nseindia.com and nsearchives.nseindia.com:

* `/api/corporates-financial-results?period=Quarterly` lists every quarterly
  result a company filed, from 2005 to the quarter ended December 2024, with
  the second it was disseminated (`broadCastDate`, `exchdisstime`). History
  is attached to the CURRENT symbol: TMPV returns Tata Motors' filings and LTM
  returns LTI's, so no rename walk is needed.
  - "Old" format filings (to ~2018) link an HTML page of label/value rows.
  - "New" format filings link an XBRL instance.
* From the quarter ended March 2025, results moved to SEBI's Integrated
  Filing. `/api/integrated-filing-results?type=Integrated Filing- Financials`
  lists them, with `broadcast_Date`, `creation_Date` and an
  Original/Revised flag, each linking an XBRL instance.

Both list endpoints need the cookies a listing page sets; the landing page
itself answered this machine with 403. The documents on nsearchives are
served without cookies.

THE OLD HTML CANNOT BE READ BY LABEL ALONE
------------------------------------------
In the bank template (HDFCBANK, December 2016), every value from "Face Value"
down sits ONE ROW BELOW its label. "Capital Adequacy Ratio" shows 15.90 and
"Gross/Net NPA" shows 15.00. A label regex would read the wrong number and
nothing would raise. So an old-format EPS is ACCEPTED only when it agrees
with net profit / (paid-up capital / face value), read at the same row shift
(0 or +1). A page that validates at neither shift yields no EPS.

POINT-IN-TIME
-------------
A result is usable from the first session at whose close it was public. A
filing disseminated on a trading day before 15:00 IST is usable that
session. 15:00 is when NSE's closing-price window opens, so this is stricter
than 15:30. Anything later, or on a non-trading day, is usable from the next
session. The timestamp used is the LATER of the two NSE records (broadcast,
dissemination), so a result is never treated as public earlier than NSE
says it was.
The EPS of quarter q-k is the figure AS FIRST DISCLOSED, never a restatement,
and it enters quarter q's SUE only if it was disclosed before q was.

SPLITS AND BONUSES
------------------
Reported EPS is per share as of its disclosure. Every EPS entering one SUE is
restated to the share basis at quarter q's disclosure using NSE's own
corporate actions (bonus, face-value split) with an ex-date on or before the
relevant disclosure. A split after q's disclosure cannot enter.
"""

from __future__ import annotations

import hashlib
import html as _html
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WWW = "https://www.nseindia.com"
LEGACY_LIST = (WWW + "/api/corporates-financial-results?index=equities"
               "&symbol={sym}&period=Quarterly")
INTEGRATED_LIST = (WWW + "/api/integrated-filing-results?index=equities&symbol={sym}"
                   "&type=Integrated%20Filing-%20Financials&page={page}&size={size}")
CORP_ACTIONS = (WWW + "/api/corporates-corporateActions?index=equities&symbol={sym}"
                "&from_date=01-01-2010&to_date={to}")
REFERER_RESULTS = WWW + "/companies-listing/corporate-filings-financial-results"
REFERER_INTEGRATED = WWW + "/companies-listing/corporate-integrated-filing"
REFERER_ACTIONS = WWW + "/companies-listing/corporate-filings-actions"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

#: At most three requests a second, whatever the network does.
MIN_REQUEST_INTERVAL = 1.0 / 3.0
REQUEST_TIMEOUT = 60
RETRIES = 3

#: The earliest quarter fetched. The first SUE on the panel (2016-10) needs
#: its own quarter, the year-ago quarter and at least four earlier seasonal
#: differences, so history must start around 2014.
FIRST_PERIOD_END = pd.Timestamp("2013-03-31")

#: A filing made on a trading day strictly before this time (IST) is usable
#: at that session's close.
INTRADAY_CUTOFF = pd.Timedelta(hours=15)

#: The SRW denominator: the standard deviation of the seasonal differences of
#: the SIGMA_QUARTERS quarters before q, needing at least SIGMA_MIN of them.
SIGMA_QUARTERS = 8
SIGMA_MIN = 4

#: The event window: SUE is live for this many sessions after it becomes
#: usable (the day it is usable counts as session 0).
EVENT_WINDOW = 30
#: Sessions-since-announcement is capped at one quarter.
AGE_CAP = 63

#: Old-format validation: accepted if within this share of the implied EPS,
#: or within the absolute floor (for near-zero EPS).
EPS_REL_TOL = 0.15
EPS_ABS_TOL = 0.10

SUE_EVT = "sue_evt"
SUE_AGE = "sue_age"
SUE_MISSING = "sue_missing"
SUE_FFILL = "sue_ffill"
EVENT_COLS = [SUE_EVT, SUE_AGE, SUE_MISSING]

#: Mergers into the listed company. They move its earnings base without a
#: corporate action on its own symbol, so NSE's action list cannot supply
#: them. Demergers come from that list instead.
MERGERS: dict[str, list[str]] = {
    "HDFCBANK": ["2023-07-01"],      # HDFC Ltd amalgamated into HDFC Bank
    "LTM": ["2022-11-14"],           # Mindtree amalgamated into LTI (-> LTIMindtree)
}

XBRL_EPS_TAGS = (
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "BasicEarningsPerShareAfterExtraordinaryItems",
    "BasicEarningsLossPerShareFromContinuingOperations",
    "BasicEarningsPerShareBeforeExtraordinaryItems",
)
XBRL_PROFIT_TAGS = (
    "ProfitOrLossAttributableToOwnersOfParent",
    "ProfitLossForPeriod",
    "ProfitLossForThePeriod",
)
XBRL_PAIDUP_TAGS = ("PaidUpValueOfEquityShareCapital",)
XBRL_FV_TAGS = ("FaceValueOfEquityShareCapital",)


# ── dates ─────────────────────────────────────────────────────────────────────


def parse_nse_time(value) -> pd.Timestamp:
    """'16-Jan-2025 20:20:21', '16-Jan-2025 20:20' or '17-JUL-2026 19:50:03'
    -> naive IST Timestamp. Anything else -> NaT."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    s = str(value).strip()
    if s in ("", "-", "None", "null"):
        return pd.NaT
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s.title(), format=fmt))
        except (ValueError, TypeError):
            continue
    return pd.NaT


def disclosed_at(*stamps) -> pd.Timestamp:
    """The LATEST of the recorded times: a result is never public earlier
    than NSE's last record of disseminating it."""
    ts = [t for t in stamps if pd.notna(t)]
    return max(ts) if ts else pd.NaT


# ── filing lists ──────────────────────────────────────────────────────────────


FILING_COLS = ["symbol", "period_start", "period_end", "basis", "source", "doc_url",
               "broadcast", "dissemination", "disclosed", "revised", "seq"]


def _basis(value) -> str | None:
    v = str(value).strip().lower()
    if v in ("consolidated",):
        return "consolidated"
    if v in ("non-consolidated", "standalone"):
        return "standalone"
    return None


def normalise_legacy(rows: list[dict]) -> pd.DataFrame:
    """The legacy list -> FILING_COLS. Quarter-length, non-cumulative rows
    only; a row with no readable document is dropped."""
    out = []
    for r in rows:
        start = pd.to_datetime(r.get("fromDate"), format="%d-%b-%Y", errors="coerce")
        end = pd.to_datetime(r.get("toDate"), format="%d-%b-%Y", errors="coerce")
        if pd.isna(start) or pd.isna(end) or not 80 <= (end - start).days <= 100:
            continue
        if str(r.get("cumulative", "")).strip().lower() != "non-cumulative":
            continue
        basis = _basis(r.get("consolidated"))
        if basis is None:
            continue
        if str(r.get("format", "")).strip().lower() == "new":
            url, source = r.get("xbrl"), "legacy_xbrl"
        else:
            url, source = r.get("resultDetailedDataLink"), "legacy_html"
        if not isinstance(url, str) or not url.startswith("http") or url.rstrip("/").endswith("-"):
            continue
        b = parse_nse_time(r.get("broadCastDate"))
        d = parse_nse_time(r.get("exchdisstime"))
        if pd.isna(b) and pd.isna(d):
            b = parse_nse_time(r.get("filingDate"))
        out.append({"symbol": r.get("symbol"), "period_start": start, "period_end": end,
                    "basis": basis, "source": source, "doc_url": url,
                    "broadcast": b, "dissemination": d, "disclosed": disclosed_at(b, d),
                    "revised": str(r.get("reInd", "")).strip().upper() == "R",
                    "seq": str(r.get("seqNumber", ""))})
    return pd.DataFrame(out, columns=FILING_COLS)


def normalise_integrated(rows: list[dict]) -> pd.DataFrame:
    """The Integrated Filing list -> FILING_COLS."""
    out = []
    for r in rows:
        if "Financials" not in str(r.get("type", "")):
            continue
        end = pd.to_datetime(str(r.get("qe_Date", "")).title(), format="%d-%b-%Y",
                             errors="coerce")
        basis = _basis(r.get("consolidated"))
        url = r.get("xbrl")
        if pd.isna(end) or basis is None or not isinstance(url, str) \
                or not url.lower().endswith(".xml"):
            continue
        start = (end - pd.offsets.QuarterEnd(1)) + pd.Timedelta(days=1)
        b = parse_nse_time(r.get("broadcast_Date"))
        d = parse_nse_time(r.get("creation_Date"))
        out.append({"symbol": r.get("symbol"), "period_start": start, "period_end": end,
                    "basis": basis, "source": "integrated_xbrl", "doc_url": url,
                    "broadcast": b, "dissemination": d, "disclosed": disclosed_at(b, d),
                    "revised": str(r.get("type_Sub", "Original")).strip().lower() != "original",
                    "seq": str(r.get("seq_Id", ""))})
    return pd.DataFrame(out, columns=FILING_COLS)


def first_disclosures(filings: pd.DataFrame) -> pd.DataFrame:
    """One filing per (symbol, period_end, basis): the earliest ORIGINAL one.
    A quarter with only revised filings keeps its earliest revision, flagged.
    A restatement filed later never replaces what the market first saw."""
    if filings.empty:
        return filings
    f = filings.dropna(subset=["disclosed"]).copy()
    f["_rev"] = f["revised"].astype(bool).astype(int)
    f = f.sort_values(["symbol", "period_end", "basis", "_rev", "disclosed"])
    return (f.drop_duplicates(subset=["symbol", "period_end", "basis"], keep="first")
            .drop(columns="_rev").reset_index(drop=True))


# ── EPS: XBRL ─────────────────────────────────────────────────────────────────


_CTX = re.compile(r"<(?:xbrli:)?context\b[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>",
                  re.S)


def _contexts(text: str) -> dict[str, tuple[str | None, str | None, bool]]:
    out = {}
    for cid, body in _CTX.findall(text):
        start = re.search(r"<(?:xbrli:)?startDate>([^<]+)<", body)
        end = re.search(r"<(?:xbrli:)?endDate>([^<]+)<", body)
        inst = re.search(r"<(?:xbrli:)?instant>([^<]+)<", body)
        dim = ("explicitMember" in body) or ("typedMember" in body)
        out[cid] = (start.group(1).strip() if start else None,
                    (end or inst).group(1).strip() if (end or inst) else None, dim)
    return out


def _facts(text: str, tag: str) -> list[tuple[str, str]]:
    pat = re.compile(r"<[\w\-]+:" + tag + r"\b[^>]*?contextRef=\"([^\"]+)\"[^>]*>([^<]*)<")
    return pat.findall(text)


def _num(s) -> float | None:
    try:
        v = float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def parse_xbrl_eps(text: str, period_start: pd.Timestamp,
                   period_end: pd.Timestamp) -> dict:
    """
    Basic EPS for exactly the quarter, from a non-dimensional context whose
    period is [period_start, period_end]. The tag is the first of
    XBRL_EPS_TAGS present for such a context. Several such contexts with
    DIFFERENT values is refused, not resolved by position.

    Also returns the implied EPS (profit / (paid-up / face value)) when the
    tags exist. It is a diagnostic here, not a gate: XBRL is structured.
    """
    ctx = _contexts(text)
    s, e = period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d")
    quarter = {cid for cid, (a, b, dim) in ctx.items() if a == s and b == e and not dim}
    instants = {cid for cid, (a, b, dim) in ctx.items()
                if a is None and not dim}
    out = {"eps": None, "eps_tag": None, "implied": None, "status": "no_eps",
           "via": "defined"}
    if not quarter:
        # Older NSE instances reference contextRef="OneD" without ever
        # defining it. Wherever a file DOES define OneD it is exactly the
        # quarter, so the convention is used only when OneD is undefined; a
        # defined OneD with other dates is never overridden.
        if "OneD" in ctx:
            out["status"] = "no_quarter_context"
            return out
        quarter, out["via"] = {"OneD"}, "oned_convention"
        instants = instants | {"OneI"}
    for tag in XBRL_EPS_TAGS:
        vals = {cid: _num(v) for cid, v in _facts(text, tag) if cid in quarter}
        vals = {k: v for k, v in vals.items() if v is not None}
        if not vals:
            continue
        distinct = set(round(v, 6) for v in vals.values())
        if len(distinct) > 1:
            # NSE files sometimes date the year-to-date context ("FourD") as
            # the quarter (ABB, December 2022: OneD 14.41, FourD 47.96, both
            # stamped 2022-10-01..2022-12-31). OneD is the quarter by NSE's
            # own convention; without it, the conflict is refused.
            if "OneD" not in vals:
                out["status"] = "conflicting_eps"
                return out
            out.update(eps=vals["OneD"], eps_tag=tag, status="ok", via="oned_preferred")
            break
        out.update(eps=next(iter(vals.values())), eps_tag=tag, status="ok")
        break

    def first(tags, ctxs):
        for t in tags:
            for cid, v in _facts(text, t):
                if cid in ctxs and _num(v):
                    return _num(v)
        return None
    tags = XBRL_PROFIT_TAGS if "ProfitOrLossAttributableToOwnersOfParent" in text         else XBRL_PROFIT_TAGS[1:]
    # the diagnostic reads the quarter from OneD too when the file has it
    profit = first(tags, ({"OneD"} & quarter) or quarter)
    paid = first(XBRL_PAIDUP_TAGS, quarter | instants)
    fv = first(XBRL_FV_TAGS, quarter | instants)
    if profit is not None and paid and fv:
        out["implied"] = profit / (paid / fv)
    return out


# ── EPS: old-format HTML ──────────────────────────────────────────────────────


def _cells(row: str) -> list[str]:
    return [re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S | re.I)]


def html_rows(page: str) -> list[tuple[str, str | None, str]]:
    """(label, value, header) for every row of the result table, in order, up
    to 'Notes To Accounts'. `header` is the last value-less row above it, so
    '(a) Basic' can be told apart under 'before' and 'after' extraordinary
    items."""
    out, header = [], ""
    # Split on each row's OPENING tag: the 2017 Ind-AS template never closes
    # its <TR>s, so a <tr>...</tr> match reads one row and stops.
    for row in re.split(r"<tr(?:\s[^>]*)?>", page, flags=re.I)[1:]:
        cells = _cells(row)
        if not cells:
            continue
        if cells[0].lower().startswith("notes to accounts"):
            break
        if len(cells) == 1:
            header = cells[0]
            continue
        if len(cells) == 2:
            out.append((cells[0], cells[1], header))
    return out


def _find(rows, pred) -> int | None:
    for i, (label, _, header) in enumerate(rows):
        if pred(label.lower(), header.lower()):
            return i
    return None


def parse_old_html_eps(page: str, basis: str) -> dict:
    """
    Basic EPS after extraordinary items (else before), ACCEPTED only when it
    agrees with net profit / (paid-up / face value) read at the same row
    shift, 0 or +1. The +1 shift is the bank template's: values from 'Face
    Value' down sit one row below their labels.
    """
    rows = html_rows(page)
    out = {"eps": None, "implied": None, "shift": None, "status": "no_rows"}
    if not rows:
        return out

    def under(key):
        return lambda l, h: (l.startswith("(a) basic") and "earnings per share" in h
                             and key in h)
    # In priority order. Three templates are known: the pre-Ind-AS one
    # ("(a) Basic" under an "after extraordinary items" header), the bank one
    # ("Basic EPS after Extraordinary items"), and the 2017 Ind-AS one
    # ("Basic EPS for continued and discontinued operations").
    preds = [
        lambda l, h: l.startswith("basic eps") and "discontinued operations" in l
        and ("continued and" in l or "continuing and" in l),
        lambda l, h: l.startswith("basic eps") and "after" in l,
        under("after"),
        lambda l, h: l.startswith("basic eps") and "continuing operations" in l,
        lambda l, h: l.startswith("basic eps") and "before" in l,
        under("before"),
    ]
    # Every candidate row is tried, in this order, and the first that
    # reconciles wins. The 2017 Ind-AS template often leaves "continued and
    # discontinued" at 0.00 while "continuing operations" carries the figure
    # (AMBUJACEM, June 2017: 0.00 against 1.98, implied 1.975), so stopping at
    # the first row FOUND would reject a page the validator can confirm.
    candidates = list(dict.fromkeys(i for i in (_find(rows, p) for p in preds)
                                    if i is not None))
    i_fv = _find(rows, lambda l, h: l.startswith("face value"))
    i_pu = _find(rows, lambda l, h: "paid-up" in l or "paid up" in l)
    i_np = None
    if basis == "consolidated":
        i_np = _find(rows, lambda l, h: ("minority interest and share of" in l
                                         and l.startswith("net profit"))
                     or l.startswith("consolidated net profit"))
    if i_np is None:
        i_np = _find(rows, lambda l, h: l.startswith("net profit") and "for the period" in l)
    if not candidates or None in (i_fv, i_pu, i_np):
        out["status"] = "missing_rows"
        return out

    vals = [_num(v) for _, v, _ in rows]
    profit = vals[i_np]
    for i_eps in candidates:
        for shift in (0, 1):
            idx = [i + shift for i in (i_eps, i_fv, i_pu)]
            if max(idx) >= len(vals):
                continue
            eps, fv, pu = (vals[i] for i in idx)
            if None in (eps, fv, pu, profit) or not fv or not pu:
                continue
            implied = profit / (pu / fv)
            if abs(eps - implied) <= max(EPS_REL_TOL * abs(implied), EPS_ABS_TOL):
                out.update(eps=eps, implied=implied, shift=shift, status="ok")
                return out
    out["status"] = "unvalidated"
    return out


# ── corporate actions ─────────────────────────────────────────────────────────


_RS = re.compile(r"R[se]\.?\s*(\d+(?:\.\d+)?)", re.I)


def _one_factor(s: str) -> float | None:
    m = re.search(r"bonus\s*(\d+)\s*:\s*(\d+)", s, re.I)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / b if b > 0 else None
    if re.search(r"split|sub-division", s, re.I):
        amounts = [float(x) for x in _RS.findall(s)]
        if len(amounts) >= 2 and amounts[1] > 0:
            return amounts[0] / amounts[1]
    return None


def action_factor(subject: str) -> float | None:
    """How many shares one share becomes: 'Bonus 1:2' -> 1.5, 'Face Value
    Split From Rs.10/- To Rs.2/-' -> 5. A combined subject multiplies its
    parts: 'Bonus 1:1/Face Value Split ... Rs 10/- ... To Rs 2/-' -> 10
    (BAJFINANCE, 2016). None when no part is a bonus or split."""
    s = " ".join(str(subject).split())
    parts = [_one_factor(p) for p in re.split(r"/(?=\s*(?:bonus|face|fv|split))", s,
                                                flags=re.I)]
    parts = [f for f in parts if f is not None]
    return float(np.prod(parts)) if parts else None


def merge_yf_splits(nse: pd.DataFrame, yf: pd.DataFrame | None,
                    window_days: int = 5) -> pd.DataFrame:
    """NSE's actions are primary. A yfinance split (`corporate_actions`, which
    records bonuses as splits too) with NO NSE bonus or split for that symbol
    within `window_days` is added as kind 'split_yf'. Measured 2026-09-19:
    NSE's list lacks MOTHERSON's July-2015 1:2 bonus, and yfinance lacks the
    bonus half of two same-day split+bonus pairs (BAJAJFINSV 2022, BAJFINANCE
    2025). Neither source alone is complete."""
    if yf is None or yf.empty:
        return nse
    y = yf.copy()
    y["symbol"] = y["ticker"].str.replace(".NS", "", regex=False)
    y["ex_date"] = pd.to_datetime(y["date"])
    ns = nse[nse["kind"].isin(["bonus", "split"])]
    add = []
    for _, r in y.iterrows():
        near = ns[(ns["symbol"] == r["symbol"])
                  & ((ns["ex_date"] - r["ex_date"]).abs().dt.days <= window_days)]
        if near.empty and r["symbol"] in set(nse["symbol"]) | set(ns["symbol"]):
            add.append((r["symbol"], r["ex_date"], "split_yf", float(r["ratio"]),
                        "yfinance corporate_actions"))
    extra = pd.DataFrame(add, columns=["symbol", "ex_date", "kind", "factor", "subject"])
    return pd.concat([nse, extra], ignore_index=True)


def parse_actions(rows: list[dict], symbol: str) -> pd.DataFrame:
    """NSE corporate actions -> (symbol, ex_date, kind, factor, subject) for
    bonuses, splits and demergers."""
    out = []
    for r in rows:
        subj = " ".join(str(r.get("subject", "")).split())
        ex = pd.to_datetime(str(r.get("exDate", "")).strip(), format="%d-%b-%Y",
                            errors="coerce")
        if pd.isna(ex):
            continue
        if re.search(r"demerger", subj, re.I):
            out.append((symbol, ex, "demerger", np.nan, subj))
            continue
        f = action_factor(subj)
        if f is not None and f > 0 and abs(f - 1.0) > 1e-9:
            out.append((symbol, ex, "bonus" if "bonus" in subj.lower() else "split", f, subj))
    return (pd.DataFrame(out, columns=["symbol", "ex_date", "kind", "factor", "subject"])
            .drop_duplicates(subset=["symbol", "ex_date", "kind"]))


def share_factor(actions: pd.DataFrame, when: pd.Timestamp) -> float:
    """Cumulative shares-per-original-share from actions with ex-date on or
    before `when`'s calendar day."""
    a = actions[actions["kind"].isin(["bonus", "split", "split_yf"])]
    a = a[a["ex_date"] <= pd.Timestamp(when).normalize()]
    return float(np.prod(a["factor"].to_numpy(dtype=float))) if len(a) else 1.0


def breaks_for(symbol: str, actions: pd.DataFrame) -> list[pd.Timestamp]:
    dem = actions.loc[actions["kind"] == "demerger", "ex_date"].tolist()
    return sorted([pd.Timestamp(d) for d in dem]
                  + [pd.Timestamp(d) for d in MERGERS.get(symbol, [])])


# ── SUE ───────────────────────────────────────────────────────────────────────


def srw_sue(quarters: pd.DataFrame, actions: pd.DataFrame,
            breaks: list[pd.Timestamp]) -> pd.DataFrame:
    """
    One symbol, one basis: (period_end, eps, disclosed) -> the same rows plus
    `sue`, `diff`, `sigma`, `n_sigma`.

    For quarter q, disclosed at d_q:
    * every EPS used was disclosed strictly before d_q (q's own excepted) and
      is restated to the share basis at d_q with splits ex on or before d_q;
    * D_j = E_j - E_(j-4), where j-4 is the same quarter one year earlier, and
      is void if a merger or demerger falls in (period_end_(j-4),
      period_end_j];
    * SUE_q = D_q / sd(D_(q-1) .. D_(q-SIGMA_QUARTERS)), needing SIGMA_MIN.
    """
    q = quarters.dropna(subset=["eps", "disclosed"]).sort_values("period_end")
    q = q.drop_duplicates(subset=["period_end"], keep="first").reset_index(drop=True)
    ends = q["period_end"].tolist()
    eps = dict(zip(ends, q["eps"].astype(float)))
    when = dict(zip(ends, q["disclosed"]))
    factor_at = {e: share_factor(actions, when[e]) for e in ends}

    def year_ago(e: pd.Timestamp) -> pd.Timestamp:
        return (e - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)

    def broken(a: pd.Timestamp, b: pd.Timestamp) -> bool:
        return any(a < br <= b for br in breaks)

    out = []
    for e in ends:
        d_q = when[e]
        f_q = factor_at[e]

        def adj(k):
            if k not in eps or (k != e and not when[k] < d_q):
                return None
            return eps[k] * factor_at[k] / f_q

        def diff(j):
            a, b = adj(j), adj(year_ago(j))
            if a is None or b is None or broken(year_ago(j), j):
                return None
            return a - b

        d_now = diff(e)
        prior = []
        j = e
        for _ in range(SIGMA_QUARTERS):
            j = (j - pd.offsets.QuarterEnd(1))
            j = pd.Timestamp(j) + pd.offsets.MonthEnd(0)
            v = diff(j)
            if v is not None:
                prior.append(v)
        sigma = float(np.std(prior, ddof=1)) if len(prior) >= SIGMA_MIN else float("nan")
        sue = (d_now / sigma if d_now is not None and np.isfinite(sigma) and sigma > 0
               else float("nan"))
        out.append({"period_end": e, "diff": d_now if d_now is not None else float("nan"),
                    "sigma": sigma, "n_sigma": len(prior), "sue": sue})
    return q.merge(pd.DataFrame(out), on="period_end", how="left")


def announcements(eps_table: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """
    (symbol, period_end, basis, eps, disclosed) for every symbol -> one
    announcement per (symbol, period_end): the CONSOLIDATED SUE where
    defined, else the STANDALONE one, disclosed when that filing was. If
    neither basis defines a SUE the announcement still exists, at the
    earliest disclosure, with SUE missing.
    """
    rows = []
    for sym, g in eps_table.groupby("symbol"):
        acts = actions[actions["symbol"] == sym]
        brk = breaks_for(sym, acts)
        per = {b: srw_sue(gb, acts, brk) for b, gb in g.groupby("basis")}
        ends = sorted(set(g["period_end"]))
        for e in ends:
            picked = None
            for b in ("consolidated", "standalone"):
                t = per.get(b)
                if t is None:
                    continue
                r = t[t["period_end"] == e]
                if len(r) and np.isfinite(r["sue"].iloc[0]):
                    picked = (b, r.iloc[0])
                    break
            if picked is None:
                first = g[g["period_end"] == e].sort_values("disclosed").iloc[0]
                rows.append({"symbol": sym, "period_end": e, "basis": None,
                             "disclosed": first["disclosed"], "sue": float("nan")})
            else:
                b, r = picked
                rows.append({"symbol": sym, "period_end": e, "basis": b,
                             "disclosed": r["disclosed"], "sue": float(r["sue"])})
    return pd.DataFrame(rows, columns=["symbol", "period_end", "basis", "disclosed", "sue"])


# ── from announcements to the panel ───────────────────────────────────────────


def usable_session(ts: pd.Timestamp, sessions: list[str]) -> str | None:
    """The first session at whose close a filing disseminated at `ts` (IST)
    was public: the same session if it is one and `ts` is before 15:00, else
    the next session strictly after `ts`'s calendar day."""
    if pd.isna(ts):
        return None
    day = pd.Timestamp(ts).normalize()
    ds = day.strftime("%Y-%m-%d")
    arr = np.asarray(sessions)
    if (pd.Timestamp(ts) - day) < INTRADAY_CUTOFF:
        i = int(np.searchsorted(arr, ds, side="left"))
    else:
        i = int(np.searchsorted(arr, ds, side="right"))
    return str(arr[i]) if i < len(arr) else None


def timing_class(ts: pd.Timestamp, sessions: set[str]) -> str:
    if pd.isna(ts):
        return "unknown"
    day = pd.Timestamp(ts).strftime("%Y-%m-%d")
    if day not in sessions:
        return "non_trading_day"
    return "intraday" if (pd.Timestamp(ts) - pd.Timestamp(ts).normalize()) < INTRADAY_CUTOFF \
        else "after_close"


def event_features(ann: pd.DataFrame, grid: list[str], tickers: list[str],
                   phantoms: tuple[str, ...] = (),
                   window: int = EVENT_WINDOW) -> pd.DataFrame:
    """
    Announcements -> (date, ticker, sue_evt, sue_age, sue_missing, sue_ffill),
    RAW (before imputation and z-scoring), on the panel's grid.

    Sessions exclude `phantoms`: a phantom row carries what was known at the
    previous real session's close, and no announcement is usable on one.
    * sue_evt: the latest announcement's SUE while fewer than `window`
      sessions have passed since it became usable, 0.0 after; NaN when that
      announcement has no SUE or there has been none.
    * sue_age: sessions since it became usable, capped at AGE_CAP; NaN if none.
    * sue_missing: 1.0 where the latest announcement has no SUE, or none exists.
    * sue_ffill: the latest DEFINED SUE carried forward with no window — the
      naive construction, for comparison only.
    """
    phantom = set(phantoms)
    sessions = [d for d in sorted(grid) if d not in phantom]
    pos = {d: i for i, d in enumerate(sessions)}
    grid_sorted = sorted(grid)
    # a phantom row reads the previous real session
    row_pos = []
    last = -1
    for d in grid_sorted:
        if d in pos:
            last = pos[d]
        row_pos.append(last)
    row_pos = np.asarray(row_pos)

    frames = []
    for t in tickers:
        sym = t.replace(".NS", "")
        a = ann[ann["symbol"] == sym].copy()
        a["usable"] = [usable_session(x, sessions) for x in a["disclosed"]]
        a = a.dropna(subset=["usable"])
        a["upos"] = a["usable"].map(pos)
        # several announcements usable the same session: the later period wins
        a = a.sort_values(["upos", "period_end"]).drop_duplicates("upos", keep="last")
        up = a["upos"].to_numpy(dtype=int)
        sue = a["sue"].to_numpy(dtype=float)

        k = np.searchsorted(up, row_pos, side="right") - 1
        has = (k >= 0) & (row_pos >= 0)
        age = np.where(has, row_pos - up[np.clip(k, 0, None)], -1)
        cur = np.where(has, sue[np.clip(k, 0, None)], np.nan)
        live = has & (age < window)
        evt = np.where(live, cur, np.where(has & np.isfinite(cur), 0.0, np.nan))
        missing = (~has) | ~np.isfinite(cur)
        # naive: last defined SUE carried forward
        defined = np.isfinite(sue)
        dk = np.full(len(row_pos), -1)
        if defined.any():
            dup, dsue = up[defined], sue[defined]
            dk = np.searchsorted(dup, row_pos, side="right") - 1
            ffill = np.where(dk >= 0, dsue[np.clip(dk, 0, None)], np.nan)
        else:
            ffill = np.full(len(row_pos), np.nan)
        frames.append(pd.DataFrame({
            "date": grid_sorted, "ticker": t,
            SUE_EVT: evt,
            SUE_AGE: np.where(has, np.minimum(age, AGE_CAP), np.nan).astype(float),
            SUE_MISSING: missing.astype(float),
            SUE_FFILL: ffill,
        }))
    return pd.concat(frames, ignore_index=True)


def impute_cross_sectional_median(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """NaN -> that date's cross-sectional median of the column. The
    missingness is kept in SUE_MISSING, so imputing loses nothing and never
    carries a stale value forward."""
    out = frame.copy()
    for c in cols:
        med = out.groupby("date")[c].transform("median")
        out[c] = out[c].fillna(med).fillna(0.0)
    return out


# ── fetching ──────────────────────────────────────────────────────────────────


@dataclass
class Fetched:
    status: str                      # ok | not_found | blocked | error
    http: int | None = None
    content: bytes | None = None
    sha256: str | None = None
    detail: str = ""


@dataclass
class NSEClient:
    """
    A throttled session for NSE's JSON APIs and document archive.

    The APIs need the cookies a listing page sets, so the session visits the
    relevant listing page once before its first API call, and again after a
    401/403. ``sleep`` and ``clock`` are injectable so the throttle is
    testable without a network or a wall clock.
    """
    session: object = None
    min_interval: float = MIN_REQUEST_INTERVAL
    sleep: object = time.sleep
    clock: object = time.monotonic
    _last: float = field(default=-1e9, repr=False)
    _warm: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        if self.session is None:
            import requests

            self.session = requests.Session()
            self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*",
                                         "Accept-Language": "en-US,en;q=0.9"})

    def _throttle(self) -> None:
        wait = self.min_interval - (self.clock() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = self.clock()

    def _raw(self, url: str, referer: str | None = None):
        self._throttle()
        headers = {"Referer": referer} if referer else None
        return self.session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)

    def warm(self, referer: str, force: bool = False) -> None:
        if referer in self._warm and not force:
            return
        try:
            self._raw(referer)
        except Exception:                                       # noqa: BLE001
            pass
        self._warm.add(referer)

    def fetch(self, url: str, referer: str | None = None) -> Fetched:
        if referer:
            self.warm(referer)
        last = ""
        for attempt in range(RETRIES):
            try:
                r = self._raw(url, referer)
            except Exception as exc:                            # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"[:200]
                self.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                return Fetched("ok", 200, r.content, hashlib.sha256(r.content).hexdigest())
            if r.status_code == 404:
                return Fetched("not_found", 404)
            if r.status_code in (401, 403):
                if referer and attempt == 0:
                    self.warm(referer, force=True)
                    continue
                return Fetched("blocked", r.status_code)
            last = f"HTTP {r.status_code}"
            self.sleep(2.0 * (attempt + 1))
        return Fetched("error", None, detail=last)
