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
