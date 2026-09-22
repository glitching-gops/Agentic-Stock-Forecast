"""
pipeline/determinism.py — one place that decides how many threads a fit uses.

P6 measured what this file exists to prevent. The same code, the same panel,
the same Optuna ``SEED``, the same ``random_state=42``, run twice and differing
only in ``OMP_NUM_THREADS``:

    threads   selected gammas per fold                cs IC       t
    20        4.744, 3.188, 3.188, 0.172, 0.172       +0.014593   +2.405
    6         3.188, 3.188, 0.172, 0.172, 0.172       +0.012050   +1.987

Zero of 162,535 predictions matched, the largest difference was 6.2 prediction
standard deviations, and the result crossed the pre-registered t = 2.0
threshold on an environment variable.

THE MECHANISM IS NOT LAST-DECIMAL NOISE. XGBoost's ``hist`` accumulates
gradient histograms in parallel and floating-point addition is not associative,
so a trial's inner-CV score depends on how the work was divided. Optuna then
selects a DIFFERENT WINNER, and a different winner is a different model. The
fragility therefore bites hardest on whatever is closest to a threshold, which
in practice means positives.

WHAT ACTUALLY PINS IT, measured rather than assumed
---------------------------------------------------
``n_jobs`` alone is NOT enough, and the obvious value is the wrong one. A fit
run at ``n_jobs=4`` reproduces across machines only where 4 threads are
actually available; with ``OMP_NUM_THREADS=2`` in the environment OpenMP caps
the team below the request and the result changes again:

    OMP_NUM_THREADS   n_jobs      prediction digest
    2                 unpinned    5f8eba01...
    20                unpinned    de5191e7...        <- differs
    2                 4           5f8eba01...
    20                4           64ff5a4f...        <- still differs
    2 / 8 / 20        **2**       5f8eba01... (all three identical)
    2 / 8 / 20        1           f3f00ce3... (all three identical)

So the rule is: **pin ``n_jobs`` at or below the smallest core count any
machine that runs this will have.** Two is that floor. It is NOT this repo's CI:
the repository is public, and GitHub's hosted runners for public repositories
have four vCPUs (measured 2026-09-21: ``ubuntu-24.04``, and the repo's
visibility is ``public``). Two is the smallest machine a recorded number might
plausibly be produced on — a private-repository runner, a two-core VM, a
laptop — and a pin that fits the smallest machine fits every larger one.

**THE PRODUCTION CONSEQUENCE.** Before this module the scheduled jobs fitted at
whatever the runner offered — four threads — and after it they fit at two. So
the first weekly run after this merges will move the persisted per-ticker
evaluations the same way the 30-session baseline moved when the pin was
applied to it (cs IC -0.00101 -> -0.00423, grades 1/1/82 -> 2/3/79). That is a
one-off step, attributable through ``tracking.environment_hash``, and it is
the price of every later run being reproducible.

The cost is real and was measured on a 160,000-row fit: 0.8 s at 20 threads,
1.2 s at 4, **2.2 s at 2**, 6.0 s at 1. A 2.75x slowdown against a workstation
default is the price of a number that means the same thing on two machines,
and it is worth paying: every table produced before this module existed is
reproducible only at whatever thread count its machine happened to default to.

WHAT THIS DOES NOT PIN, and is recorded rather than fixed:
  - Library versions. ``requirements.txt`` uses ``>=``, so xgboost, numpy,
    pandas, scikit-learn and optuna all float. ``environment_fingerprint()``
    records them so a movement is at least ATTRIBUTABLE, which is the same
    trade ``tracking.config_hash`` makes for everything else.
  - A single-core machine. ``XGB_THREADS = 2`` would be capped to 1 there and
    the results would differ. No such machine is in use; the fingerprint would
    show it.
"""

from __future__ import annotations

import os

#: Threads every XGBoost fit in this project uses.
#:
#: Two, not more, because a pin above a machine's available threads silently
#: degrades to what is available — which is exactly the failure this constant
#: exists to remove — and two is the smallest machine a recorded number might be
#: produced on. NOT this repo's CI, which has four; see the module docstring.
XGB_THREADS = 2

#: Escape hatch for exploration ONLY. Anything whose number is recorded must
#: run at the default: `environment_fingerprint()` reports the value actually
#: used, so a run at an override is identifiable rather than silently different.
THREADS_ENV_VAR = "AGENTIC_XGB_THREADS"


def xgb_threads() -> int:
    """The thread count this process will use for every XGBoost fit."""
    raw = os.environ.get(THREADS_ENV_VAR)
    if raw is None:
        return XGB_THREADS
    try:
        value = int(raw)
    except ValueError:
        return XGB_THREADS
    return value if value > 0 else XGB_THREADS


def xgb_params(**overrides) -> dict:
    """
    The keyword arguments every ``XGBRegressor`` in this project is built with.

    ``n_jobs`` and ``tree_method`` are the two that must not vary. ``n_jobs``
    for the reason in the module docstring; ``tree_method`` because its default
    is ``"auto"``, which picks a different algorithm by DATA SIZE — so a fold
    with fewer rows could silently be fitted by a different tree builder than
    its neighbours. Every call site here already passed ``"hist"`` except one,
    and "every call site except one" is how the fold-guard landmine happened.

    Caller overrides win for everything EXCEPT ``n_jobs``, so a hyperparameter
    dict from the tuner still applies in full; the point is that a site which
    specifies nothing still gets the pin, and a site which specifies a thread
    count does not get to keep it.
    """
    params = {
        "tree_method": "hist",
        "verbosity": 0,
    }
    params.update(overrides)
    # LAST, and deliberately not overridable. A caller that passes `n_jobs`
    # is asking for the one thing this module exists to take away from
    # callers; `tree_method` above is a default because a caller may have a
    # legitimate reason to change the tree builder, but nobody has a
    # legitimate reason to change the thread count of a recorded run.
    params["n_jobs"] = xgb_threads()
    return params


def threads_are_pinned() -> tuple[bool, str]:
    """
    Whether the pin can actually hold in this process.

    ``n_jobs`` is a request, not a guarantee: OpenMP caps the team at whatever
    the environment permits, so on a machine with fewer usable threads than
    ``XGB_THREADS`` the fit silently runs narrower and reproduces nothing. That
    is the one case the pin cannot cover, and it is detectable rather than
    mysterious.
    """
    wanted = xgb_threads()
    cpus = os.cpu_count() or 1
    if cpus < wanted:
        return False, f"machine has {cpus} usable core(s), below the pinned {wanted}"
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw:
        try:
            if int(raw) < wanted:
                return False, (f"OMP_NUM_THREADS={raw} caps the team below the "
                               f"pinned {wanted}; OpenMP wins")
        except ValueError:
            pass
    return True, f"pinned at {wanted} threads"


def environment_fingerprint() -> dict:
    """
    Everything outside this repository that can move a number.

    Recorded beside `config_hash` and `data_hash` for the same reason those
    exist: when a metric moves while one of them is constant, it says which of
    code, data or environment caused it. Before P6 there was no third one, and
    a thread-count change would have read as a code change.

    THE OPERATING SYSTEM IS PART OF IT (2026-09-21). With every library version
    and the thread count identical, the h=30 baseline is bit-identical run to
    run on Windows and run to run on Linux, and DIFFERENT between them - max
    drift 0.64, fold 4 choosing gamma 0.172 on Linux against 3.188 on Windows.
    The cause is XGBoost's row and column subsampling: the same seed draws a
    different sample under MSVC's C++ standard library than under GCC's,
    because the standard fixes the generator but not the distributions. A fit
    with `subsample` and `colsample_bytree` at 1.0 is identical on both.
    """
    import platform
    import sys

    import numpy
    import pandas

    def _version(module_name: str) -> str:
        try:
            module = __import__(module_name)
            return str(getattr(module, "__version__", "unknown"))
        except Exception:                                       # noqa: BLE001
            return "absent"

    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "xgb_threads": xgb_threads(),
        "xgb_threads_pinned_default": XGB_THREADS,
        "xgb_threads_overridden": xgb_threads() != XGB_THREADS,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cpu_count": os.cpu_count(),
        "xgboost": _version("xgboost"),
        "optuna": _version("optuna"),
        "sklearn": _version("sklearn"),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }
