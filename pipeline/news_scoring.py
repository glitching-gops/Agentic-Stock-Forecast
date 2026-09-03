"""
pipeline/news_scoring.py — one scorer, for history and for live alike.

WHY ONE SCORER IS A NON-NEGOTIABLE, NOT A PREFERENCE
-----------------------------------------------------
Sentiment was removed as a model feature by audit finding F7: it existed only
for the current date, so every training row held 0.0 while the row being
predicted held a real value. A train/serve mismatch at exactly the row that
matters.

Scoring the 2016-2026 backfill with one model and the daily trickle with
another would be F7 in a new costume — the historical feature and the live
feature would be different quantities, and nothing measured in a purged fold
would describe what actually gets published. So `scorer_id` pins the checkpoint
AND its resolved revision, it is stored on every row, and features refuse to
mix two of them.

Note that CPU vs GPU is NOT that problem. The same checkpoint on either device
is the same scorer; the Chronos work already measured CPU and CUDA agreeing to
4.8e-07 with TF32 off. A different MODEL is the defect, not a different device.

WHY FinBERT, AND WHAT IS WRONG WITH IT
---------------------------------------
`ProsusAI/finbert` is the standard financial-sentiment checkpoint: 110M
parameters, three-way positive/negative/neutral with calibrated probabilities.
Chosen deliberately over an LLM, for two reasons that both matter more than
accuracy. It is DETERMINISTIC, so a re-score reproduces the archive rather than
perturbing it; and it costs no request budget, so the ~46,000-article backfill
does not consume 150 days of the Groq token allowance.

Its known weakness is recorded here rather than discovered later: it was
trained on Reuters and analyst-report English, not on Indian retail financial
media. Headlines from Moneycontrol and Livemint sit outside that register, so
expect a domain mismatch. That is a measurable quantity, not a hidden one — it
shows up as poor separation on a hand-labelled sample, and the remedy is a
different `scorer_id`, not a silent patch.

THE SCORE IS SIGNED AND CONTINUOUS, NOT A LABEL
------------------------------------------------
`score = p(positive) - p(negative)`, in [-1, +1]. A label alone throws away the
model's confidence, and confidence is most of the information when the majority
of headlines are genuinely neutral. `label` and `confidence` are stored too, so
a downstream feature can require both a direction and a strength.

TORCH IS IMPORTED INSIDE THE FUNCTION, ON PURPOSE
--------------------------------------------------
`requirements.txt` must never carry torch — it is installed by Render, by the
daily pipeline and by the weekly evaluation, none of which need it, and it was
removed in Phase 0 as the largest contributor to memory pressure on an instance
that had already been OOM-killed. The import lives inside `load_scorer` so that
importing this module (which `pipeline.baselines` and the API transitively can)
never pulls it in. A test asserts torch is absent from `sys.modules` after
importing this module in a fresh interpreter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

from data.db import get_engine

logger = logging.getLogger(__name__)

#: The checkpoint. Changing it changes `scorer_id`, which makes every existing
#: score a DIFFERENT quantity — features refuse to mix two, so a change here is
#: a re-score of the whole archive, not an edit.
SCORER_MODEL = os.getenv("NEWS_SCORER_MODEL", "ProsusAI/finbert")

#: Pin the revision when you want reproducibility across machines. Left as the
#: default branch so a fresh checkout works, but the RESOLVED commit is what
#: gets stored — see `resolve_scorer_id`. Recording "main" would be recording
#: nothing: it means a different model next month.
SCORER_REVISION = os.getenv("NEWS_SCORER_REVISION", "main")

BATCH_SIZE = int(os.getenv("NEWS_SCORER_BATCH", "64"))

#: FinBERT's own label order. Read from the model config rather than assumed —
#: the median-index landmine in the series work is the same shape: three
#: checkpoints, three different orderings, and a hardcoded index silently
#: returns the wrong class with no error anywhere.
_FALLBACK_LABELS = {0: "positive", 1: "negative", 2: "neutral"}


class ScorerUnavailable(Exception):
    """torch/transformers are absent. The caller degrades; it does not guess."""


@dataclass
class ScoringReport:
    scored: int = 0
    skipped: int = 0
    scorer_id: str = ""
    device: str = ""
    seconds: float = 0.0

    def summary(self) -> str:
        return (f"{self.scored} scored, {self.skipped} already had this "
                f"scorer's rows, id={self.scorer_id}, device={self.device}, "
                f"{self.seconds:.0f}s")


_CACHE: dict[tuple[str, str, str], tuple] = {}


def resolve_device(explicit: str | None = None) -> str:
    """Explicit -> NEWS_SCORER_DEVICE -> autodetect. Mirrors series.resolve_device."""
    if explicit:
        return explicit
    env = os.getenv("NEWS_SCORER_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                            # noqa: BLE001
        return "cpu"


def load_scorer(model_id: str = SCORER_MODEL, revision: str = SCORER_REVISION,
                device: str | None = None):
    """
    Returns ``(tokenizer, model, device, scorer_id)``, cached per (model, revision, device).

    KEYED ON DEVICE TOO. A model-only cache silently hands a CPU-resident model
    to a caller that asked for CUDA, and the run is merely slow with nothing to
    see — the exact defect the series model caches were fixed for.
    """
    device = resolve_device(device)
    key = (model_id, revision, device)
    if key in _CACHE:
        return _CACHE[key]

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:                                     # noqa: BLE001
        raise ScorerUnavailable(
            f"{type(exc).__name__}: {exc}. Install the scoring extras:\n"
            f"    pip install -r requirements-scoring.txt\n"
            f"torch is deliberately absent from requirements.txt — see the "
            f"module docstring."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision)
    model.eval().to(device)

    resolved = getattr(getattr(model, "config", None), "_commit_hash", None) or revision
    scorer_id = f"{model_id}@{resolved}"

    _CACHE[key] = (tokenizer, model, device, scorer_id)
    return _CACHE[key]


def label_map(model) -> dict[int, str]:
    """
    The checkpoint's OWN class ordering, lowercased.

    Read rather than assumed. Three series checkpoints put their median
    quantile at three different indices, and a hardcoded position returned the
    30th percentile of every prediction with no error anywhere. A sentiment
    model with a permuted label order fails exactly the same way: every score
    keeps its magnitude and flips its meaning.
    """
    cfg = getattr(model, "config", None)
    raw = getattr(cfg, "id2label", None)
    if not raw:
        return dict(_FALLBACK_LABELS)
    out = {int(k): str(v).lower() for k, v in raw.items()}
    if {"positive", "negative", "neutral"} - set(out.values()):
        raise ScorerUnavailable(
            f"{SCORER_MODEL} reports labels {sorted(set(out.values()))}, which "
            f"do not include the three this scorer stores. A different label "
            f"scheme is a different quantity and must not be written under the "
            f"same scorer_id.")
    return out


def score_texts(titles: list[str], model_id: str = SCORER_MODEL,
                revision: str = SCORER_REVISION, device: str | None = None,
                batch_size: int = BATCH_SIZE) -> list[dict]:
    """Scores headlines. Returns one dict per input, in the SAME ORDER."""
    import torch

    tokenizer, model, device, _ = load_scorer(model_id, revision, device)
    labels = label_map(model)
    out: list[dict] = []

    for start in range(0, len(titles), batch_size):
        chunk = titles[start:start + batch_size]
        encoded = tokenizer(chunk, padding=True, truncation=True,
                            max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu()

        for row in probs:
            by_label = {labels[i]: float(row[i]) for i in range(len(row))}
            # POSITIVE MINUS NEGATIVE, not argmax. A label throws away the
            # model's confidence, and confidence is most of the information
            # when the majority of headlines are genuinely neutral.
            score = by_label["positive"] - by_label["negative"]
            winner = max(by_label, key=by_label.get)
            out.append({"label": winner, "score": score,
                        "confidence": by_label[winner]})
    return out


def score_unscored(limit: int | None = None, engine=None, device: str | None = None,
                   model_id: str = SCORER_MODEL,
                   revision: str = SCORER_REVISION) -> ScoringReport:
    """
    Scores every article with no row under the CURRENT scorer_id.

    Idempotent by construction: re-running scores nothing. A checkpoint change
    makes every article unscored again under the new id, which is the correct
    and deliberate cost of changing the scorer — the alternative is an archive
    holding two incomparable quantities in one column.
    """
    import time

    engine = engine or get_engine()
    report = ScoringReport()
    began = time.time()

    _, model, device, scorer_id = load_scorer(model_id, revision, device)
    report.scorer_id, report.device = scorer_id, device

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT a.article_id, a.title
            FROM news_articles a
            WHERE NOT EXISTS (
                SELECT 1 FROM news_scores s
                WHERE s.article_id = a.article_id AND s.scorer_id = :sid)
            ORDER BY a.published_at
        """ + (f" LIMIT {int(limit)}" if limit else "")),
            {"sid": scorer_id}).fetchall()

    if not rows:
        report.seconds = time.time() - began
        return report

    ids = [r[0] for r in rows]
    scored = score_texts([r[1] or "" for r in rows], model_id, revision, device)
    now = datetime.now(timezone.utc).isoformat()

    with engine.connect() as conn:
        for article_id, result in zip(ids, scored):
            conn.execute(text("""
                INSERT INTO news_scores
                    (article_id, scorer_id, label, score, confidence, scored_at)
                VALUES (:aid, :sid, :label, :score, :conf, :at)
                ON CONFLICT (article_id, scorer_id) DO NOTHING
            """), {"aid": article_id, "sid": scorer_id,
                   "label": result["label"], "score": float(result["score"]),
                   "conf": float(result["confidence"]), "at": now})
            report.scored += 1
        conn.commit()

    report.seconds = time.time() - began
    logger.info("[news_scoring] %s", report.summary())
    return report


def current_scorer_id(engine=None) -> str | None:
    """
    The scorer_id holding the most rows, or None.

    Feature construction reads this rather than recomputing it from the
    configured checkpoint, so a half-finished re-score cannot silently produce
    a feature built from two different models' output.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT scorer_id, COUNT(*) n FROM news_scores "
            "GROUP BY scorer_id ORDER BY n DESC")).fetchone()
    return row[0] if row else None
