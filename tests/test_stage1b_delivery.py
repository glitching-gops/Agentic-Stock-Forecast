"""
Stage 1, Pilot 2 — NSE delivery % ingestion and its features.

Every clause here fails silently when broken, which is why each is a test:
- a file for the wrong date, or a record with a shifted column, must be
  refused, not parsed;
- a 404 on a trading date must not read as a holiday, and a 403 must not
  read as "no delivery";
- a renamed ticker must be matched on the symbol it had THAT day;
- the feature at t must not see session t;
- a missing session must never be filled with a stale value.

See docs/stage1b-preregistration.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import delivery as dl

MTO_TEXT = """Security Wise Delivery Position - Compulsory Rolling Settlement
10,MTO,11092026,462506349,0000003
Trade Date <11-SEP-2026>,Settlement Type <N>,Settlement No <2026173>,Settlement Date <14-SEP-2026>
Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity(gross across client level),% of Deliverable Quantity to Traded Quantity
20,1,RELIANCE,EQ,7451218,4061266,54.50
20,2,RELIANCE,BL,100,100,100.00
20,3,TCS,EQ,1000,400,40.00
"""

CHANGES = """COMPANY NAME,OLD SYMBOL,NEW SYMBOL,DATE OF CHANGE
LTIMindtree Limited,LTI,LTIM,05-DEC-2022
LTM Limited,LTIM,LTM,27-FEB-2026
Tata Motors, Passenger Vehicles Limited,TATAMOTORS,TMPV,24-OCT-2025
"""


# ── parsing ───────────────────────────────────────────────────────────────────

def test_the_validated_mto_format_parses_to_the_right_numbers():
    f = dl.parse_mto(MTO_TEXT, expected=pd.Timestamp("2026-09-11"))
    rel = f[(f["symbol"] == "RELIANCE") & (f["series"] == "EQ")].iloc[0]
    assert (rel["qty_traded"], rel["deliv_qty"], rel["deliv_pct"]) == (7451218, 4061266, 54.50)
    assert set(f["series"]) == {"EQ", "BL"}
    assert f.attrs["file_date"] == pd.Timestamp("2026-09-11")


def test_a_file_for_another_date_is_refused():
    """NSE has already been caught answering one date's request with
    another's (the FII/DII ?date= parameter). A delivery figure filed
    under the wrong session is a look-ahead or a look-back that raises
    nothing."""
    with pytest.raises(dl.MTOFormatError, match="asked for"):
        dl.parse_mto(MTO_TEXT, expected=pd.Timestamp("2026-09-10"))


def test_a_record_with_a_shifted_column_is_refused_not_parsed():
    bad = MTO_TEXT.replace("20,3,TCS,EQ,1000,400,40.00", "20,3,TCS,LTD,EQ,1000,400,40.00")
    with pytest.raises(dl.MTOFormatError, match="8 fields"):
        dl.parse_mto(bad)


def test_symbol_changes_parse_a_company_name_that_contains_a_comma():
    ch = dl.parse_symbol_changes(CHANGES)
    assert ("TATAMOTORS", "TMPV") in set(zip(ch["old"], ch["new"]))
    assert len(ch) == 3


# ── symbols change ────────────────────────────────────────────────────────────

def test_each_session_maps_to_the_symbol_it_was_filed_under():
    ch = dl.parse_symbol_changes(CHANGES)
    assert dl.symbol_on("LTM", "2020-01-02", ch) == "LTI"
    assert dl.symbol_on("LTM", "2022-12-02", ch) == "LTI"
    assert dl.symbol_on("LTM", "2022-12-05", ch) == "LTIM", "the change date is the NEW symbol's first day"
    assert dl.symbol_on("LTM", "2026-02-26", ch) == "LTIM"
    assert dl.symbol_on("LTM", "2026-02-27", ch) == "LTM"
    assert dl.symbol_on("RELIANCE", "2017-01-02", ch) == "RELIANCE"
    assert dl.symbol_history("LTM", ch) == {"LTM", "LTIM", "LTI"}


def test_panel_tickers_match_on_that_dates_symbol_and_on_eq_only():
    ch = dl.parse_symbol_changes(CHANGES)
    rec = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2026-03-02", "2026-03-02", "2026-03-02"],
        "symbol": ["LTI", "LTM", "LTM", "RELIANCE", "RELIANCE"],
        "series": ["EQ", "EQ", "EQ", "EQ", "BL"],
        "qty_traded": [10, 99, 20, 30, 1], "deliv_qty": [5, 99, 8, 15, 1],
        "deliv_pct": [50.0, 100.0, 40.0, 50.0, 100.0]})
    out = dl.to_panel_tickers(rec, ["LTM.NS", "RELIANCE.NS"], ch).set_index(["date", "ticker"])
    assert out.loc[("2020-01-02", "LTM.NS"), "deliv_pct"] == 50.0, (
        "in 2020 LTM traded as LTI; a stray 'LTM' row that day belongs to nobody in the panel")
    assert out.loc[("2026-03-02", "LTM.NS"), "deliv_pct"] == 40.0
    assert out.loc[("2026-03-02", "RELIANCE.NS"), "deliv_pct"] == 50.0, "the BL row must not be read"
    assert ("2020-01-02", "RELIANCE.NS") not in out.index


# ── fetching ──────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, code, content=b""):
        self.status_code, self.content = code, content


class _Session:
    def __init__(self, codes):
        self.codes, self.urls = list(codes), []
        self.headers = {}

    def get(self, url, timeout=None):
        self.urls.append(url)
        code = self.codes.pop(0)
        return _Resp(code, MTO_TEXT.encode() if code == 200 else b"")


def _client(codes):
    return dl.ArchiveClient(session=_Session(codes), sleep=lambda s: None)


def test_a_404_is_not_found_and_never_filed_as_a_holiday_quietly():
    r = _client([404]).fetch_mto(pd.Timestamp("2026-09-11"))
    assert r.status == "not_found" and r.frame is None


def test_a_403_warms_cookies_once_and_is_otherwise_blocked_not_empty():
    c = _client([403, 200, 403])        # the file, the landing page, the retry
    r = c.fetch_mto(pd.Timestamp("2026-09-11"))
    assert r.status == "blocked"
    assert c.session.urls[1] == dl.HOMEPAGE, "one landing-page visit for cookies"
    assert len(c.session.urls) == 3, "exactly one retry"


def test_a_good_file_is_ok_and_its_bytes_are_hashed():
    r = _client([200]).fetch_mto(pd.Timestamp("2026-09-11"))
    assert r.status == "ok" and len(r.sha256) == 64 and len(r.frame) == 3


def test_the_throttle_holds_three_requests_a_second():
    now = [0.0]
    stamps = []

    def sleep(s):
        now[0] += s

    def clock():
        return now[0]

    sess = _Session([200] * 10)
    orig_get = sess.get

    def get(url, timeout=None):
        stamps.append(now[0])
        return orig_get(url, timeout)
    sess.get = get
    c = dl.ArchiveClient(session=sess, sleep=sleep, clock=clock)
    for _ in range(10):
        c.get("x")
    gaps = np.diff(stamps)
    assert gaps.min() >= dl.MIN_REQUEST_INTERVAL - 1e-12


# ── the features: point-in-time, and never filled ─────────────────────────────

def _delivery(n_dates=150, tickers=("A.NS", "B.NS", "C.NS"), seed=0, levels=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for j, t in enumerate(tickers):
        base = (levels or {}).get(t, 50.0)
        for d, x in zip(dates, base + rng.normal(0, 5, n_dates)):
            rows.append((d, t, float(np.clip(x, 0, 100))))
    return pd.DataFrame(rows, columns=["date", "ticker", "deliv_pct"]), list(dates)


def _wide(f, col):
    return f.pivot(index="date", columns="ticker", values=col)


def test_delivery_features_at_t_are_blind_to_session_t_and_after():
    """The file for T is published after T's close, so it may inform T+1 and
    nothing earlier. Corrupting session `cut` and every session after must
    leave every feature at dates <= cut unchanged, and must move the feature
    at cut+1 (so the lag is exactly one session, not more)."""
    deliv, grid = _delivery()
    cut, nxt = grid[100], grid[101]
    before = dl.delivery_features(deliv, grid)
    shocked = deliv.copy()
    later = shocked["date"] >= cut
    shocked.loc[later, "deliv_pct"] = np.random.default_rng(9).uniform(0, 100, int(later.sum()))
    after = dl.delivery_features(shocked, grid)

    key = ["date", "ticker"]
    a = before[before["date"] <= cut].set_index(key)[dl.DELIVERY_COLS]
    b = after[after["date"] <= cut].set_index(key)[dl.DELIVERY_COLS]
    assert a.notna().any().all(), "not vacuous: every column is defined somewhere"
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(_wide(before, dl.LEVEL_COL).loc[nxt],
                           _wide(after, dl.LEVEL_COL).loc[nxt])


def test_the_level_at_t_is_exactly_session_t_minus_one():
    deliv, grid = _delivery()
    f = dl.delivery_features(deliv, grid)
    raw = deliv.pivot(index="date", columns="ticker", values="deliv_pct")
    pd.testing.assert_frame_equal(_wide(f, dl.LEVEL_COL), raw.shift(1),
                                  check_names=False)


def test_a_missing_session_is_never_filled_with_a_stale_value():
    deliv, grid = _delivery()
    gap = grid[80]
    holed = deliv[~((deliv["ticker"] == "A.NS") & (deliv["date"] == gap))]
    f = dl.delivery_features(holed, grid)
    lvl, abn, abn5 = (_wide(f, c)["A.NS"] for c in dl.DELIVERY_COLS)
    assert np.isnan(lvl.loc[grid[81]]), "the session after a gap has no 'yesterday'"
    assert np.isnan(abn.loc[grid[81]])
    assert abn5.loc[grid[81]:grid[85]].isna().all(), "a 5-session window over the gap is void"
    assert np.isfinite(lvl.loc[grid[82]]) and np.isfinite(abn.loc[grid[82]])
    assert np.isfinite(_wide(f, dl.LEVEL_COL).loc[grid[81], "B.NS"]), "other names untouched"


def test_the_corrected_placebo_permutes_within_each_date_and_keeps_the_pairing():
    """Pilot 1's placebo shuffled PREDICTIONS, which also destroyed the
    baseline's own ordering and credited the arm with the baseline's grades.
    This one permutes the new COLUMNS within each date, jointly, and the model
    is retrained on them: each date keeps its own values, the two columns stay
    paired, and only the name-to-delivery link is broken."""
    from tools.stage1b_delivery import permute_within_date

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=30).strftime("%Y-%m-%d"), 12)
    panel = pd.DataFrame({"date": dates,
                          "ticker": np.tile([f"T{i}" for i in range(12)], 30),
                          "a": rng.normal(size=360)})
    panel["b"] = 2.0 * panel["a"] + 1.0
    out = permute_within_date(panel, ["a", "b"], seed=3)

    pd.testing.assert_frame_equal(out[["date", "ticker"]], panel[["date", "ticker"]])
    for d, g in out.groupby("date"):
        assert sorted(g["a"]) == sorted(panel.loc[panel["date"] == d, "a"]), (
            "a date may only receive its own values back, rearranged")
    np.testing.assert_allclose(out["b"], 2.0 * out["a"] + 1.0,
                               err_msg="the columns must be permuted jointly")
    assert (out["a"].to_numpy() != panel["a"].to_numpy()).mean() > 0.8, (
        "most rows must have lost their own value")


def test_the_arms_are_the_baseline_plus_exactly_the_delivery_columns():
    from pipeline.baselines import FACTORS
    from tools.stage1b_delivery import ARM_FEATURES

    assert ARM_FEATURES == {
        "baseline": list(FACTORS),
        "abnormal": list(FACTORS) + dl.ABNORMAL_COLS,
        "level": list(FACTORS) + [dl.LEVEL_COL],
    }


def test_the_abnormal_transform_removes_a_permanent_level():
    """The raw level is a per-company constant — the kind of feature a tree
    uses to recognise WHICH company it is (the +0.77-from-nothing landmine).
    Its within-date rank persists; the abnormal figure's must not."""
    deliv, grid = _delivery(n_dates=400, tickers=tuple(f"T{i}.NS" for i in range(12)),
                            levels={f"T{i}.NS": 20.0 + 5 * i for i in range(12)})
    f = dl.delivery_features(deliv, grid)
    assert dl.rank_persistence(f, dl.LEVEL_COL, lag=100) > 0.9
    assert abs(dl.rank_persistence(f, dl.ABN_COL, lag=100)) < 0.2
