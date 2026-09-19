"""
Stage 1, Pilot 3: NSE results filings, EPS parsing, SRW SUE and the event
features. The point-in-time contract lives in test_leakage.py beside the
others; this file pins everything else that could be silently wrong.

Every parser test runs on a fixture shaped like the page or instance NSE
actually serves, including the three ways a naive reader gets it wrong: the
bank template's one-row shift, the Ind-AS template's unclosed rows, and the
year-to-date context stamped with the quarter's dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import earnings as ea

# ── timestamps and filing lists ───────────────────────────────────────────────


def test_nse_times_parse_in_every_format_nse_uses():
    assert ea.parse_nse_time("16-Jan-2025 20:20:21") == pd.Timestamp("2025-01-16 20:20:21")
    assert ea.parse_nse_time("16-Jan-2025 20:20") == pd.Timestamp("2025-01-16 20:20")
    assert ea.parse_nse_time("17-JUL-2026 19:50:03") == pd.Timestamp("2026-07-17 19:50:03")
    for bad in (None, "-", "", float("nan"), "null"):
        assert pd.isna(ea.parse_nse_time(bad))


def test_the_disclosure_time_is_the_later_of_nses_records():
    a, b = pd.Timestamp("2025-01-16 20:20:21"), pd.Timestamp("2025-01-16 20:20:54")
    assert ea.disclosed_at(a, b) == b
    assert ea.disclosed_at(b, a) == b
    assert ea.disclosed_at(pd.NaT, a) == a


def _legacy(**kw):
    row = {"symbol": "X", "fromDate": "01-Oct-2024", "toDate": "31-Dec-2024",
           "cumulative": "Non-cumulative", "consolidated": "Consolidated", "format": "New",
           "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/a.xml",
           "resultDetailedDataLink": None, "broadCastDate": "16-Jan-2025 20:20:21",
           "exchdisstime": "16-Jan-2025 20:20:54", "filingDate": "16-Jan-2025 20:20",
           "reInd": "N", "seqNumber": "1"}
    row.update(kw)
    return row


def test_the_legacy_list_keeps_quarters_and_routes_each_format_to_its_document():
    rows = [
        _legacy(),
        _legacy(format="Old", xbrl="https://nsearchives.nseindia.com/corporate/xbrl/-",
                resultDetailedDataLink="https://nsearchives.nseindia.com/r.html",
                consolidated="Non-Consolidated"),
        _legacy(fromDate="01-Apr-2024", cumulative="Cumulative"),           # 9 months
        _legacy(fromDate="01-Apr-2024"),                                    # not a quarter
        _legacy(format="Old", resultDetailedDataLink="-", xbrl="-"),        # no document
    ]
    f = ea.normalise_legacy(rows)
    assert len(f) == 2
    assert f.iloc[0]["source"] == "legacy_xbrl" and f.iloc[0]["basis"] == "consolidated"
    assert f.iloc[1]["source"] == "legacy_html" and f.iloc[1]["basis"] == "standalone"
    assert f.iloc[0]["disclosed"] == pd.Timestamp("2025-01-16 20:20:54")


def test_the_integrated_list_reads_its_own_fields_and_flags_revisions():
    rows = [{"type": "Integrated Filing- Financials", "qe_Date": "30-JUN-2026",
             "consolidated": "Standalone", "xbrl": "https://x/y.xml",
             "broadcast_Date": "17-Jul-2026 19:49:04", "creation_Date": "17-Jul-2026 19:49:06",
             "type_Sub": "Revised", "symbol": "X", "seq_Id": "9"},
            {"type": "Integrated Filing- Governance", "qe_Date": "30-JUN-2026",
             "consolidated": None, "xbrl": "https://x/z.xml", "symbol": "X"}]
    f = ea.normalise_integrated(rows)
    assert len(f) == 1
    r = f.iloc[0]
    assert r["period_start"] == pd.Timestamp("2026-04-01")
    assert r["period_end"] == pd.Timestamp("2026-06-30")
    assert r["basis"] == "standalone" and bool(r["revised"])
    assert r["disclosed"] == pd.Timestamp("2026-07-17 19:49:06")


def test_a_later_restatement_never_replaces_the_first_disclosure():
    base = {"symbol": "X", "period_start": pd.Timestamp("2024-10-01"),
            "period_end": pd.Timestamp("2024-12-31"), "basis": "standalone",
            "source": "legacy_xbrl", "broadcast": pd.NaT, "dissemination": pd.NaT, "seq": ""}
    f = pd.DataFrame([
        {**base, "doc_url": "revised-early", "disclosed": pd.Timestamp("2025-01-10"),
         "revised": True},
        {**base, "doc_url": "original", "disclosed": pd.Timestamp("2025-01-16"),
         "revised": False},
        {**base, "doc_url": "original-late", "disclosed": pd.Timestamp("2025-03-01"),
         "revised": False},
    ])
    assert ea.first_disclosures(f)["doc_url"].tolist() == ["original"]


# ── EPS from the old HTML pages ───────────────────────────────────────────────


def _page(rows, closed=True):
    end = "</tr>" if closed else ""
    body = []
    for r in rows:
        if isinstance(r, str):
            body.append(f"<tr><td colspan=2>{r}</td>{end}")
        else:
            body.append(f"<TR><TD class=t1>{r[0]}</td><TD class=t0 nowrap>{r[1]}</td>{end}")
    return "<table>" + "\n".join(body) + "<tr><td>Notes To Accounts</td><td>x</td></tr></table>"


RELIANCE_2016 = [
    ("Net Profit / (Loss) for the period", "756700.00"),
    ("Net Profit / (Loss) after taxes, minority interest and share of profit / (loss) "
     "of associates", "750600"),
    ("Face Value (in Rs.)", "10.00"),
    ("Paid-up equity share capital", "295100.00"),
    "Earnings per share (before extraordinary items) (not annualised):",
    ("(a) Basic", "25.30"),
    ("(b) Diluted", "25.30"),
    "Earnings per share (after extraordinary items) (not annualised):",
    ("(a) Basic", "25.40"),
    ("(b) Diluted", "25.40"),
]


def test_the_pre_ind_as_template_reads_basic_eps_after_extraordinary_items():
    r = ea.parse_old_html_eps(_page(RELIANCE_2016), "consolidated")
    assert r["status"] == "ok" and r["shift"] == 0
    assert r["eps"] == 25.40, "the 'after' Basic, not the 'before' one"
    assert r["implied"] == pytest.approx(750600 / (295100 / 10))


HDFCBANK_2016 = [                      # NSE's own misalignment, verbatim
    ("Net Profit (+) / Loss (-) for the period", "386533.00"),
    ("Face Value (in Rs.)", "-"),
    ("Paid-up Equity Share Capital (in Rs.)", "2.00"),
    ("Reserves excluding Revaluation Reserves", "51107.00"),
    ("Percentage of shares Held by Government of India", "-"),
    ("Capital Adequacy Ratio", "0.00"),
    ("Basic EPS before Extraordinary items (in Rs.)", "15.90"),
    ("Diluted EPS before Extraordinary items (in Rs.)", "15.20"),
    ("Basic EPS after Extraordinary items (in Rs.)", "15.00"),
    ("Diluted EPS after Extraordinary items (in Rs.)", "15.20"),
    ("Gross/Net NPA", "15.00"),
]


def test_the_bank_templates_one_row_shift_is_detected_not_read_through():
    """Read by label, 'Basic EPS after' says 15.00, which is the diluted
    figure. The true basic EPS, 15.20, sits one row down. Only the +1 shift
    reconciles with net profit / (paid-up / face value)."""
    r = ea.parse_old_html_eps(_page(HDFCBANK_2016), "standalone")
    assert r["status"] == "ok" and r["shift"] == 1
    assert r["eps"] == 15.20
    assert r["implied"] == pytest.approx(386533 / (51107 / 2))


def test_the_ind_as_template_with_unclosed_rows_parses():
    rows = [("Net Profit / (Loss) for the period", "8817.00"),
            ("Consolidated Net Profit/Loss for the period", "8817.00"),
            ("Face Value (in Rs.)", "2.00"), ("Paid-up equity share capital", "4238.00"),
            ("Basic EPS  for continuing operations", "4.16"),
            ("Basic EPS  for continued and discontinued operations", "4.16")]
    r = ea.parse_old_html_eps(_page(rows, closed=False), "standalone")
    assert r["status"] == "ok" and r["eps"] == 4.16


def test_a_page_that_reconciles_at_no_shift_yields_no_eps():
    bad = [(label, "999.00" if "EPS" in label else value) for label, value in HDFCBANK_2016]
    r = ea.parse_old_html_eps(_page(bad), "standalone")
    assert r["eps"] is None and r["status"] == "unvalidated"


# ── EPS from XBRL ─────────────────────────────────────────────────────────────


def _xbrl(contexts, facts):
    ctx = "".join(
        f'<xbrli:context id="{cid}"><xbrli:entity/>'
        f'<xbrli:period><xbrli:startDate>{s}</xbrli:startDate>'
        f'<xbrli:endDate>{e}</xbrli:endDate></xbrli:period>'
        + ('<xbrli:scenario><xbrldi:explicitMember dimension="d">m</xbrldi:explicitMember>'
           '</xbrli:scenario>' if dim else "")
        + "</xbrli:context>"
        for cid, s, e, dim in contexts)
    fx = "".join(f'<in-bse-fin:{t} contextRef="{c}" unitRef="u" decimals="2">{v}</in-bse-fin:{t}>'
                 for t, c, v in facts)
    return f"<xbrli:xbrl>{ctx}{fx}</xbrli:xbrl>"


Q = (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
EPS = "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations"


def test_xbrl_eps_comes_from_the_quarters_own_context():
    x = _xbrl([("OneD", "2024-10-01", "2024-12-31", False),
               ("FourD", "2024-04-01", "2024-12-31", False),
               ("Seg", "2024-10-01", "2024-12-31", True)],
              [(EPS, "FourD", "37.13"), (EPS, "Seg", "99"), (EPS, "OneD", "13.70")])
    r = ea.parse_xbrl_eps(x, *Q)
    assert r["status"] == "ok" and r["eps"] == 13.70 and r["via"] == "defined"


def test_an_undefined_oned_is_read_by_nses_convention():
    x = _xbrl([("Seg", "2024-10-01", "2024-12-31", True)], [(EPS, "OneD", "4.84")])
    r = ea.parse_xbrl_eps(x, *Q)
    assert r["eps"] == 4.84 and r["via"] == "oned_convention"


def test_a_defined_oned_with_other_dates_is_never_overridden():
    x = _xbrl([("OneD", "2024-07-01", "2024-09-30", False)], [(EPS, "OneD", "4.84")])
    r = ea.parse_xbrl_eps(x, *Q)
    assert r["eps"] is None and r["status"] == "no_quarter_context"


def test_a_ytd_context_stamped_with_the_quarters_dates_loses_to_oned():
    """ABB, December 2022: OneD 14.41 and FourD 47.96, both dated
    2022-10-01..2022-12-31. FourD is year-to-date by NSE's convention."""
    x = _xbrl([("OneD", "2024-10-01", "2024-12-31", False),
               ("FourD", "2024-10-01", "2024-12-31", False)],
              [(EPS, "FourD", "47.96"), (EPS, "OneD", "14.41")])
    r = ea.parse_xbrl_eps(x, *Q)
    assert r["eps"] == 14.41 and r["via"] == "oned_preferred"


def test_a_conflict_without_oned_is_refused():
    x = _xbrl([("A", "2024-10-01", "2024-12-31", False),
               ("B", "2024-10-01", "2024-12-31", False)],
              [(EPS, "A", "47.96"), (EPS, "B", "14.41")])
    assert ea.parse_xbrl_eps(x, *Q)["status"] == "conflicting_eps"


# ── corporate actions ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("subject, factor", [
    ("Bonus 1:1", 2.0), (" Bonus 1:2", 1.5), ("Bonus 3:5", 1.6),
    ("Face Value Split From Rs.10/- To Rs.2/-", 5.0),
    ("Fv Split Rs.10 To Rs.5", 2.0),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 5/- Per Share", 2.0),
    ("Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share", 2.0),
    ("Rights 1:15 @ Premium Rs 1247", None), ("Dividend - Rs 10 Per Share", None),
])
def test_bonus_and_split_subjects_become_share_factors(subject, factor):
    assert ea.action_factor(subject) == factor


def test_a_split_only_counts_from_its_ex_date():
    acts = ea.parse_actions([{"subject": "Bonus 1:1", "exDate": "07-Sep-2017"},
                             {"subject": "Demerger", "exDate": "20-Jul-2023"},
                             {"subject": "Bonus 1:1", "exDate": "28-Oct-2024"}], "RELIANCE")
    assert ea.share_factor(acts, pd.Timestamp("2017-09-06 23:00")) == 1.0
    assert ea.share_factor(acts, pd.Timestamp("2017-09-07 09:00")) == 2.0
    assert ea.share_factor(acts, pd.Timestamp("2026-01-01")) == 4.0
    assert ea.breaks_for("RELIANCE", acts) == [pd.Timestamp("2023-07-20")]
    assert ea.breaks_for("HDFCBANK", acts.iloc[:0]) == [pd.Timestamp("2023-07-01")]


# ── SUE ───────────────────────────────────────────────────────────────────────


def _quarters(eps, start="2019-03-31", lag_days=40):
    ends = pd.date_range(start, periods=len(eps), freq="QE")
    return pd.DataFrame({"period_end": ends, "eps": eps,
                         "disclosed": ends + pd.Timedelta(days=lag_days)})


NO_ACTIONS = pd.DataFrame(columns=["symbol", "ex_date", "kind", "factor", "subject"])


def test_srw_sue_matches_a_hand_computation():
    rng = np.random.default_rng(1)
    eps = list(10 + rng.normal(0, 1, 16).cumsum())
    out = ea.srw_sue(_quarters(eps), NO_ACTIONS, [])
    d = [eps[i] - eps[i - 4] for i in range(4, 16)]
    # quarter 15: D_15 over sd(D_7..D_14) -> d indices 3..10
    expected = d[11] / np.std(d[3:11], ddof=1)
    assert out["sue"].iloc[15] == pytest.approx(expected)
    assert out["n_sigma"].iloc[15] == 8
    assert np.isnan(out["sue"].iloc[7]), "quarter 7 has only 3 prior differences"
    assert np.isfinite(out["sue"].iloc[8]), "quarter 8 has exactly SIGMA_MIN = 4"


def test_a_split_leaves_the_sue_unchanged():
    """A 2:1 bonus halves every later EPS. Restated to one share basis, the
    SUE must equal the one computed with no split at all."""
    rng = np.random.default_rng(2)
    eps = 10 + rng.normal(0, 1, 16).cumsum()
    q = _quarters(list(eps))
    split_day = q["disclosed"].iloc[9] + pd.Timedelta(days=5)
    halved = q.copy()
    halved.loc[10:, "eps"] = halved.loc[10:, "eps"] / 2.0
    acts = pd.DataFrame([{"symbol": "X", "ex_date": split_day, "kind": "bonus",
                          "factor": 2.0, "subject": "Bonus 1:1"}])
    a = ea.srw_sue(q, NO_ACTIONS, [])["sue"]
    b = ea.srw_sue(halved, acts, [])["sue"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    naive = ea.srw_sue(halved, NO_ACTIONS, [])["sue"]
    assert not np.allclose(a.iloc[10:14], naive.iloc[10:14]), "the split is real without it"


def test_a_quarter_disclosed_after_q_cannot_enter_qs_sue():
    rng = np.random.default_rng(3)
    q = _quarters(list(10 + rng.normal(0, 1, 16).cumsum()))
    before = ea.srw_sue(q, NO_ACTIONS, [])["sue"].iloc[15]
    late = q.copy()
    late.loc[11, "disclosed"] = late.loc[15, "disclosed"] + pd.Timedelta(days=1)
    after = ea.srw_sue(late, NO_ACTIONS, [])
    assert after["sue"].iloc[15] != pytest.approx(before)
    assert after["n_sigma"].iloc[15] < 8


def test_a_merger_voids_every_difference_that_straddles_it():
    rng = np.random.default_rng(4)
    q = _quarters(list(10 + rng.normal(0, 1, 16).cumsum()))
    brk = [pd.Timestamp("2021-07-01")]              # between 2021-06-30 and 2021-09-30
    out = ea.srw_sue(q, NO_ACTIONS, brk)
    by_end = out.set_index("period_end")
    for e in pd.date_range("2021-09-30", "2022-06-30", freq="QE"):
        assert np.isnan(by_end.loc[e, "diff"]), f"{e.date()} straddles the merger"
    assert np.isfinite(by_end.loc[pd.Timestamp("2022-09-30"), "diff"])
    assert np.isfinite(by_end.loc[pd.Timestamp("2021-06-30"), "diff"])


def test_consolidated_sue_is_preferred_and_standalone_is_the_fallback():
    rng = np.random.default_rng(5)
    s = _quarters(list(10 + rng.normal(0, 1, 16).cumsum())).assign(basis="standalone")
    c = _quarters(list(20 + rng.normal(0, 1, 16).cumsum())).assign(basis="consolidated")
    c = c.iloc[8:]                                  # consolidated history too short early on
    table = pd.concat([s, c]).assign(symbol="X")
    ann = ea.announcements(table, NO_ACTIONS).set_index("period_end")
    early = ann.loc[s["period_end"].iloc[:8]]
    assert early["basis"].isna().all() and early["sue"].isna().all(), (
        "fewer than SIGMA_MIN prior differences on either basis: no SUE")
    late = ann.loc[s["period_end"].iloc[8:]]
    assert (late["basis"] == "standalone").all() and late["sue"].notna().all(), (
        "8 consolidated quarters give no consolidated SUE, so standalone stands in")
    c_full = _quarters(list(20 + rng.normal(0, 1, 16).cumsum())).assign(basis="consolidated")
    ann2 = ea.announcements(pd.concat([s, c_full]).assign(symbol="X"), NO_ACTIONS)
    assert ann2.iloc[-1]["basis"] == "consolidated"
    early = ann2[ann2["period_end"] == s["period_end"].iloc[2]].iloc[0]
    assert pd.isna(early["basis"]) and np.isnan(early["sue"]), "an announcement with no SUE"


# ── from announcements to the panel ───────────────────────────────────────────


SESSIONS = ["2024-06-13", "2024-06-14", "2024-06-17", "2024-06-18"]   # Fri 14th, Mon 17th


@pytest.mark.parametrize("ts, expected", [
    ("2024-06-14 14:59:59", "2024-06-14"),          # intraday: usable at that close
    ("2024-06-14 15:00:00", "2024-06-17"),          # the closing window has opened
    ("2024-06-14 19:30:00", "2024-06-17"),          # after the close
    ("2024-06-15 11:00:00", "2024-06-17"),          # Saturday board meeting
    ("2024-06-18 18:00:00", None),                  # beyond the grid
])
def test_the_first_usable_session(ts, expected):
    assert ea.usable_session(pd.Timestamp(ts), SESSIONS) == expected


def _ann(rows):
    return pd.DataFrame(rows, columns=["symbol", "period_end", "basis", "disclosed", "sue"])


def test_the_event_window_is_exactly_window_sessions_and_then_zero():
    grid = list(pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d"))
    ann = _ann([("A", pd.Timestamp("2023-12-31"), "standalone",
                 pd.Timestamp(grid[5] + " 10:00"), 1.7)])
    f = ea.event_features(ann, grid, ["A.NS"], window=10).set_index("date")
    assert np.isnan(f.loc[grid[4], ea.SUE_EVT]) and f.loc[grid[4], ea.SUE_MISSING] == 1
    assert (f.loc[grid[5]:grid[14], ea.SUE_EVT] == 1.7).all(), "sessions 0..9 are live"
    assert f.loc[grid[15], ea.SUE_EVT] == 0.0 and f.loc[grid[15], ea.SUE_MISSING] == 0
    assert f.loc[grid[5], ea.SUE_AGE] == 0 and f.loc[grid[15], ea.SUE_AGE] == 10
    assert f.loc[grid[59], ea.SUE_FFILL] == 1.7, "the naive column never expires"


def test_an_announcement_without_a_sue_is_missing_not_zero():
    grid = list(pd.bdate_range("2024-01-01", periods=30).strftime("%Y-%m-%d"))
    ann = _ann([("A", pd.Timestamp("2023-09-30"), "standalone",
                 pd.Timestamp(grid[2] + " 10:00"), 0.9),
                ("A", pd.Timestamp("2023-12-31"), None, pd.Timestamp(grid[10] + " 10:00"),
                 np.nan)])
    f = ea.event_features(ann, grid, ["A.NS"], window=20).set_index("date")
    assert f.loc[grid[9], ea.SUE_EVT] == 0.9
    assert np.isnan(f.loc[grid[10], ea.SUE_EVT]) and f.loc[grid[10], ea.SUE_MISSING] == 1
    assert f.loc[grid[10], ea.SUE_AGE] == 0, "the age still resets: an announcement happened"
    assert f.loc[grid[10], ea.SUE_FFILL] == 0.9


def test_a_phantom_row_carries_the_previous_sessions_state():
    grid = list(pd.bdate_range("2024-01-01", periods=12).strftime("%Y-%m-%d"))
    phantom = grid[6]
    ann = _ann([("A", pd.Timestamp("2023-12-31"), "standalone",
                 pd.Timestamp(phantom + " 10:00"), 2.0)])
    f = ea.event_features(ann, grid, ["A.NS"], phantoms=(phantom,), window=5).set_index("date")
    assert np.isnan(f.loc[phantom, ea.SUE_EVT]), "no announcement is usable on a phantom"
    assert f.loc[grid[7], ea.SUE_EVT] == 2.0 and f.loc[grid[7], ea.SUE_AGE] == 0


def test_missing_values_take_the_dates_median_and_nothing_older():
    f = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 2, "ticker": list("ABCDAB"),
                      "x": [1.0, 2.0, 9.0, np.nan, np.nan, 5.0]})
    out = ea.impute_cross_sectional_median(f, ["x"])
    assert out["x"].tolist() == [1.0, 2.0, 9.0, 2.0, 5.0, 5.0]


# ── the harness ───────────────────────────────────────────────────────────────


def test_the_arms_are_the_baseline_plus_exactly_the_sue_columns():
    from pipeline.baselines import FACTORS
    from tools.stage1c_sue import ARM_FEATURES

    assert ARM_FEATURES == {
        "baseline": list(FACTORS),
        "sue": list(FACTORS) + [ea.SUE_EVT, ea.SUE_AGE, ea.SUE_MISSING],
        "naive": list(FACTORS) + [ea.SUE_FFILL],
        "timing": list(FACTORS) + [ea.SUE_AGE, ea.SUE_MISSING],
    }


def test_the_placebo_permutes_all_three_event_columns_jointly():
    from tools.stage1b_delivery import permute_within_date

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d"), 12)
    p = pd.DataFrame({"date": dates, "ticker": np.tile([f"T{i}" for i in range(12)], 20),
                      ea.SUE_EVT: rng.normal(size=240)})
    p[ea.SUE_AGE] = p[ea.SUE_EVT] * 3
    p[ea.SUE_MISSING] = p[ea.SUE_EVT] * -1
    out = permute_within_date(p, ea.EVENT_COLS, seed=5)
    np.testing.assert_allclose(out[ea.SUE_AGE], out[ea.SUE_EVT] * 3)
    np.testing.assert_allclose(out[ea.SUE_MISSING], -out[ea.SUE_EVT])
    assert (out[ea.SUE_EVT].to_numpy() != p[ea.SUE_EVT].to_numpy()).mean() > 0.8


def test_r4_charges_the_round_trip_on_the_arms_extra_turnover():
    """Two books with identical GROSS returns, one churning every name every
    rebalance and one holding: the churner must come out behind net, by the
    cost of its extra turnover, and R4 must see it."""
    from tools.stage1c_sue import net_of_cost

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=300).strftime("%Y-%m-%d")
    tickers = [f"T{i:02d}" for i in range(40)]
    frame = pd.DataFrame([(d, t) for d in dates for t in tickers], columns=["date", "ticker"])
    frame["y_true"] = rng.normal(0, 0.05, len(frame))
    fold = frame.assign(fold=0)
    steady = fold.assign(y_pred=fold["ticker"].str[1:].astype(float))
    order = {d: rng.permutation(40) for d in dates}
    churn = fold.assign(y_pred=[float(order[d][int(t[1:])]) for d, t in
                                zip(fold["date"], fold["ticker"])])
    r = net_of_cost(steady, churn)
    assert r["turn_b"] > r["turn_a"] + 0.3
    gross_gap = r["gross_b"] - r["gross_a"]
    assert r["diff"] == pytest.approx(gross_gap - (r["turn_b"] - r["turn_a"]) * r["round_trip"],
                                      abs=1e-9)


# ── the client ────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, code, content=b"{}"):
        self.status_code, self.content = code, content


class _Session:
    def __init__(self, codes):
        self.codes, self.urls, self.headers = list(codes), [], {}

    def get(self, url, timeout=None, headers=None):
        self.urls.append(url)
        return _Resp(self.codes.pop(0))


def test_an_api_403_re_warms_cookies_once_then_is_blocked_not_empty():
    s = _Session([200, 403, 200, 403])      # warm, api, re-warm, api
    c = ea.NSEClient(session=s, sleep=lambda x: None)
    r = c.fetch("https://www.nseindia.com/api/x", ea.REFERER_RESULTS)
    assert r.status == "blocked"
    assert s.urls == [ea.REFERER_RESULTS, "https://www.nseindia.com/api/x",
                      ea.REFERER_RESULTS, "https://www.nseindia.com/api/x"]


def test_the_client_holds_three_requests_a_second():
    now, stamps = [0.0], []
    s = _Session([200] * 12)
    orig = s.get

    def get(url, timeout=None, headers=None):
        stamps.append(now[0])
        return orig(url, timeout, headers)
    s.get = get
    c = ea.NSEClient(session=s, sleep=lambda x: now.__setitem__(0, now[0] + x),
                     clock=lambda: now[0])
    for _ in range(12):
        c.fetch("https://nsearchives.nseindia.com/doc.xml")
    assert np.diff(stamps).min() >= ea.MIN_REQUEST_INTERVAL - 1e-12


def test_a_zero_filled_eps_row_gives_way_to_the_one_that_reconciles():
    """AMBUJACEM, June 2017: 'continued and discontinued' left at 0.00 and
    'continuing operations' carrying the figure."""
    rows = [("Net Profit / (Loss) for the period", "39223.00"),
            ("Face Value (in Rs.)", "2.00"), ("Paid-up equity share capital", "39713.00"),
            ("Basic EPS for continuing operations", "1.98"),
            ("Diluted EPS for continuing operations", "1.98"),
            ("Basic EPS for continued and discontinued operations", "0.00")]
    r = ea.parse_old_html_eps(_page(rows), "standalone")
    assert r["status"] == "ok" and r["eps"] == 1.98 and r["shift"] == 0


def test_a_combined_bonus_and_split_subject_multiplies_both():
    s = ("Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To "
         "Rs 2/- Per Share")
    assert ea.action_factor(s) == 10.0


def test_yfinance_fills_only_what_nse_lacks():
    nse = ea.parse_actions([{"subject": "Bonus 1:2", "exDate": "03-Oct-2022"},
                            {"subject": "Bonus 4:1", "exDate": "16-Jun-2025"},
                            {"subject": "Face Value Split (Sub-Division) - From Rs 2/- Per "
                                        "Share To Re 1/- Per Share", "exDate": "16-Jun-2025"}],
                           "X")
    yf = pd.DataFrame({"ticker": ["X.NS", "X.NS", "X.NS", "Y.NS"],
                       "date": ["2015-07-23", "2022-10-04", "2025-06-16", "2015-01-01"],
                       "ratio": [1.5, 1.5, 2.0, 3.0]})
    m = ea.merge_yf_splits(nse, yf)
    added = m[m["kind"] == "split_yf"]
    assert added["ex_date"].tolist() == [pd.Timestamp("2015-07-23")], (
        "only the action NSE lacks, and never for a symbol NSE was not asked about")
    assert ea.share_factor(m, pd.Timestamp("2026-01-01")) == pytest.approx(1.5 * 1.5 * 5 * 2)
