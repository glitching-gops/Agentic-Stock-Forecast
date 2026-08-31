from typing import Optional

from pydantic import BaseModel, Field


class LeaderboardEntry(BaseModel):
    rank: Optional[int] = Field(
        None, description="Competition rank on the applied sort key. Tied rows "
                          "SHARE a rank (1, 2, 3, 3, 3, ...) rather than being "
                          "numbered off arbitrarily — most rows tie at "
                          "composite_score 0.0. Null when the sort key is null.")
    ticker: str
    company: Optional[str] = None
    sector: Optional[str] = None

    current_price: Optional[float] = None
    forecast_price: Optional[float] = Field(
        None, description="Implied price assuming the benchmark is flat.")
    upside_pct: Optional[float] = None
    pred_excess_return: Optional[float] = Field(
        None, description="Predicted 30-session log return in excess of the benchmark.")

    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    interval_coverage: Optional[float] = None
    prob_outperform: Optional[float] = None
    random_walk_price: Optional[float] = None

    benchmark_ticker: Optional[str] = None
    benchmark_name: Optional[str] = None
    benchmark_sector_specific: Optional[bool] = None

    composite_score: Optional[float] = Field(
        None, description="Ranking heuristic in [0,100]: predicted excess return and "
                          "conviction, multiplied by an evidence grade. NOT an "
                          "expected return.")
    score_basis: Optional[str] = Field(
        None, description="Why composite_score is what it is, and in particular why "
                          "it is zero: RANKED / NO_FORECAST / NO_EVIDENCE / "
                          "NOT_LONG / FLAGGED_OUT. Most rows score 0.0, and that "
                          "one value otherwise covers several unrelated situations.")
    critic_verdict: Optional[str] = None
    forecast_confidence: Optional[str] = Field(
        None, description="Evidence grade: STRONG / WEAK / INSUFFICIENT.")

    eval_rank_ic: Optional[float] = None
    eval_rank_ic_t: Optional[float] = Field(
        None, description="t-statistic of the out-of-sample rank IC. One of the "
                          "three held-out checks behind forecast_confidence, and "
                          "the only one that speaks to significance rather than "
                          "size. Read it WITH the sign: the gate tests |t|, so a "
                          "large negative value marks a ticker the model gets "
                          "reliably WRONG, not one it gets right.")
    eval_hit_rate: Optional[float] = None
    eval_baseline_hit_rate: Optional[float] = None
    eval_beats_random_walk: Optional[bool] = None
    evaluated_at: Optional[str] = Field(
        None, description="When the evidence behind this row was last measured "
                          "(weekly, not daily) — may be older than last_updated.")


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]
    total: int
    last_updated: str
    filters_applied: dict
    methodology: str = Field(
        default=(
            "Ranked by a composite of predicted 30-session excess return and "
            "conviction, gated by held-out evidence from purged walk-forward "
            "evaluation with a 30-session embargo. Metrics are out-of-sample. "
            "The composite score is a ranking heuristic, not an expected return, "
            "and is reported before transaction costs. It ranks LONG candidates "
            "only: conviction counts only where the point forecast also predicts "
            "outperformance, so a predicted underperformer floors at zero rather "
            "than ranking below one. Read score_basis before reading a 0.0 as "
            "'no signal', and note that tied rows SHARE a rank rather than being "
            "numbered off in an order the score does not support."
        ),
        description="How to read these numbers.",
    )
