from typing import Any, List, Optional, TypedDict

# How much weight an evidence grade carries. The single definition of
# "INSUFFICIENT means nothing survives".
#
# It was a MULTIPLIER on the composite score, and it lived here rather than
# beside that function so the score and critic_node's LLM gate could not hold
# different copies of the rule. The score is gone with the rest of the ranking
# layer; critic_node still reads this to decide whether an LLM review can
# change anything, so the constant stays and the values keep their meaning.
# Nothing multiplies by it any more — the name is kept because the ordering
# STRONG > WEAK > INSUFFICIENT = 0 is what callers actually depend on.
EVIDENCE_MULTIPLIER = {"STRONG": 1.0, "WEAK": 0.5, "INSUFFICIENT": 0.0}


class AgentState(TypedDict, total=False):
    """
    Shared LangGraph state.

    Phase 0 reshaped the forecast fields. The model now predicts a 30-session
    excess return against a benchmark index; the rupee target is derived from it
    and travels with an interval, a calibrated probability, and the random-walk
    reference. The old ``model_mape`` / ``model_directional_accuracy`` pair is
    gone: both were produced by fitting an estimator on the validation set and
    scoring it on that same set (audit finding F1), so neither meant what its
    name suggested.

    ``total=False`` because a failed forecast populates only the failure fields.
    """

    ticker: str
    company_name: str
    current_price: float

    # Trading Data Agent
    signals_df: Any
    latest_signals: dict

    # External Data Agent
    # None means "no scored headline exists", which is distinct from 0.0
    # ("the news balanced out"). No scorer currently runs — see
    # pipeline/sentiment.py.
    sentiment_score: Optional[float]
    macro_df: Any

    # Forecasting Agent — the forecast itself
    forecast_available: bool
    forecast_error: Optional[str]
    forecast_price: Optional[float]          # implied, benchmark assumed flat
    forecast_direction: str                  # OUTPERFORM / UNDERPERFORM / UNAVAILABLE
    forecast_change_pct: Optional[float]
    pred_excess_return: Optional[float]      # the model's actual output
    interval_low: Optional[float]
    interval_high: Optional[float]
    interval_coverage: Optional[float]
    prob_outperform: Optional[float]
    random_walk_price: Optional[float]
    benchmark_ticker: Optional[str]
    benchmark_sector_specific: Optional[bool]
    signal_narrative: str

    # Forecasting Agent — held-out evidence, all from purged walk-forward
    eval_rank_ic: Optional[float]
    eval_rank_ic_t: Optional[float]
    eval_hit_rate: Optional[float]
    eval_baseline_hit_rate: Optional[float]
    eval_beats_naive: Optional[bool]
    eval_evaluated_at: Optional[str]        # when the WEEKLY evaluation ran;
                                             # may be up to a week older than
                                             # last_updated (the forecast itself
                                             # regenerates daily)
    model_version: Optional[str]

    # Critic Agent
    evidence_grade: str                      # STRONG / WEAK / INSUFFICIENT
    evidence_reasons: List[str]
    critic_verdict: str                      # APPROVED / FLAGGED / REJECTED
    critic_reasoning: str
    critic_flags: List[str]
    critic_source: str                       # which layer set the verdict
