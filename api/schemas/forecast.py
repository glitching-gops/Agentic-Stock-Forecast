from typing import Optional

from pydantic import BaseModel, Field


class EvaluationEvidence(BaseModel):
    """
    Held-out performance, measured by purged walk-forward evaluation.

    Every field here comes from folds the model never trained on. The previous
    schema exposed `mape` and `directional_accuracy` produced by fitting an
    estimator on the validation set and scoring it on that same set, which is
    why a forecast could advertise 85% accuracy (audit finding F1).
    """

    rank_ic: Optional[float] = Field(
        None, description="Spearman rank correlation, out-of-sample.")
    hit_rate: Optional[float] = Field(
        None, description="Directional accuracy, out-of-sample, percent.")
    baseline_hit_rate: Optional[float] = Field(
        None, description="Majority-class baseline on the same window, percent. "
                          "hit_rate is only meaningful relative to this.")
    beats_random_walk: Optional[bool] = Field(
        None, description="Whether mean absolute error beats forecasting zero excess return.")
    model_version: Optional[str] = None
    evaluated_at: Optional[str] = Field(
        None, description="When this evidence was last measured (Lever 1: weekly, not "
                          "daily). Distinct from the forecast's own last_updated, which "
                          "changes every day — this may be up to a week older.")


class ForecastResponse(BaseModel):
    ticker: str
    company: Optional[str] = None
    sector: Optional[str] = None

    # What the model actually predicts
    pred_excess_return: Optional[float] = Field(
        None, description="Predicted 30-session log return in excess of the benchmark.")
    benchmark_ticker: Optional[str] = None
    benchmark_name: Optional[str] = None
    benchmark_sector_specific: Optional[bool] = Field(
        None, description="False when the stock falls back to NIFTY 50 because no "
                          "reliable sector index exists.")

    # The rupee view derived from it
    current_price: Optional[float] = None
    forecast_price: Optional[float] = Field(
        None, description="Implied price ASSUMING THE BENCHMARK IS FLAT. The model "
                          "forecasts relative performance and says nothing about "
                          "where the index goes.")
    direction: Optional[str] = Field(None, description="OUTPERFORM / UNDERPERFORM / UNAVAILABLE")
    change_pct: Optional[float] = None
    random_walk_price: Optional[float] = Field(
        None, description="Baseline forecast: today's price.")

    # Uncertainty
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    interval_coverage: Optional[float] = Field(
        None, description="Nominal conformal coverage, e.g. 0.80.")
    prob_outperform: Optional[float] = Field(
        None, description="Calibrated probability the excess return exceeds zero.")

    # Evidence and review
    evaluation: Optional[EvaluationEvidence] = None
    forecast_confidence: Optional[str] = Field(
        None, description="Evidence grade: STRONG / WEAK / INSUFFICIENT.")
    signal_narrative: Optional[str] = None
    critic_verdict: Optional[str] = None
    critic_reasoning: Optional[str] = None
    critic_flags: list[str] = []
    critic_source: Optional[str] = Field(
        None, description="Which layer set the verdict.")

    forecast_available: bool = True
    forecast_error: Optional[str] = None
    universe_rule: Optional[str] = None
    last_updated: Optional[str] = None


class CurrentForecast(BaseModel):
    """
    One stock's current forecast, as carried on the forecast_current table.

    NO RANK, NO SCORE. This replaces LeaderboardEntry, which carried `rank`,
    `composite_score` and `score_basis`. The evidence fields survive unchanged
    and are what a reader should judge a row by: a forecast whose
    forecast_confidence is INSUFFICIENT has no held-out support, and saying so
    is more informative than placing it 47th.

    Every unmeasured quantity is null, never 0.0. A zero here is a POSITION —
    "the model predicts no excess return" — and is not the same statement as
    "nothing has been measured". Conflating the two is how a dead sentiment
    scorer displayed a confident neutral reading for months.
    """

    ticker: str
    company: Optional[str] = None
    sector: Optional[str] = None

    current_price: Optional[float] = None
    forecast_price: Optional[float] = Field(
        None, description="Implied price assuming the benchmark is flat.")
    direction: Optional[str] = Field(
        None, description="OUTPERFORM / UNDERPERFORM / UNAVAILABLE")
    change_pct: Optional[float] = Field(
        None, description="Implied percentage change. Was `upside_pct`; renamed "
                          "because a forecast can point down and 'upside' asserted "
                          "otherwise.")
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

    critic_verdict: Optional[str] = None
    forecast_confidence: Optional[str] = Field(
        None, description="Evidence grade: STRONG / WEAK / INSUFFICIENT.")
    signal_narrative: Optional[str] = None

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
    model_version: Optional[str] = None
    evaluated_at: Optional[str] = Field(
        None, description="When the evidence behind this row was last measured "
                          "(weekly, not daily) — may be older than last_updated.")
    last_updated: Optional[str] = None


class ForecastListResponse(BaseModel):
    forecasts: list[CurrentForecast]
    total: int
    last_updated: str
    filters_applied: dict
    methodology: str = Field(
        default=(
            "One 30-session forecast per stock over a fixed universe of 84 NIFTY "
            "100 names, selected on data quality alone and never on measured "
            "accuracy. Predictions are of return in excess of a sector benchmark; "
            "the rupee price assumes the benchmark is flat. Every eval_* figure "
            "comes from purged walk-forward evaluation with a 30-session embargo, "
            "on folds the model never trained on. These forecasts are NOT ranked "
            "against each other: the evidence gate clears too few names for an "
            "ordering to mean anything, so each row is presented with its own "
            "evidence and read on its own terms. A null is 'not measured', which "
            "is a different statement from a zero."
        ),
        description="How to read these numbers.",
    )
