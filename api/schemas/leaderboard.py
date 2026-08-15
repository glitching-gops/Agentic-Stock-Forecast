from typing import Optional

from pydantic import BaseModel, Field


class LeaderboardEntry(BaseModel):
    rank: int
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
    critic_verdict: Optional[str] = None
    forecast_confidence: Optional[str] = Field(
        None, description="Evidence grade: STRONG / WEAK / INSUFFICIENT.")

    eval_rank_ic: Optional[float] = None
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
            "and is reported before transaction costs."
        ),
        description="How to read these numbers.",
    )
