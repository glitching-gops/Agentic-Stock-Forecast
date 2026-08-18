"""
JSON-safe conversion for pandas frames on their way out of the API.

`df.where(df.notna(), other=None)` looks like it nulls out missing values and
does not. pandas cannot store None in a float64 column, so on every numeric
column the None is coerced straight back to NaN; only object-dtype columns
actually keep it. The frame then reaches Starlette's JSONResponse, and
`json.dumps` refuses NaN:

    ValueError: Out of range float values are not JSON compliant: nan

That is a 500, not a degraded response. It hit both signals endpoints for
EVERY ticker, because the last HORIZON_SESSIONS rows of `signals` necessarily
carry a null target — the label looks 30 sessions into a future that has not
happened yet — so any window including recent sessions contains NaN by
construction.

The failure was invisible from the dashboard: the Streamlit client wrapped its
fetch in a bare `except` and substituted an empty frame, so a hard 500 rendered
as a blank chart rather than an error.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """
    One scalar, converted to something `json.dumps` accepts.

    Handles the three ways a non-serialisable value reaches this point: pandas
    nulls (NaN, NaT, pd.NA), non-finite floats (inf/-inf, which json.dumps
    renders as bare `Infinity` — valid JavaScript, invalid JSON), and numpy
    scalars, which are not instances of the Python types json recognises.
    """
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # pd.isna raises on array-likes, so it is only safe once scalars are known.
    if not isinstance(value, (str, bytes, bool, int, list, dict, tuple)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)

    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as a list of JSON-safe dicts."""
    return [
        {key: json_safe(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def mapping(row: dict[str, Any]) -> dict[str, Any]:
    """One record dict, made JSON-safe."""
    return {key: json_safe(value) for key, value in row.items()}
