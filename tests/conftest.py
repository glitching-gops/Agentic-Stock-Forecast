"""
Pytest configuration.

1. The repository root on sys.path.

2. THE SUITE CAN NEVER REACH A NON-SQLITE DATABASE, whatever `.env` says.

   `data.db` reads DATABASE_URL once, at import, and in a checkout whose `.env`
   points at Supabase every unpatched `get_engine()` in the suite resolved to
   PRODUCTION. It was measured, not suspected: by 2026-09-13,
   `test_weekly_job_raises_when_labels_would_regress` had written 221 fake
   ABORTED weekly runs into the production `experiment_runs` table, which held
   250 rows in all.

   Two layers, because there are two routes and each has its own test in
   `tests/test_suite_isolation.py`:

   * PIN. DATABASE_URL is set to a throwaway SQLite file and `data.db` is
     imported HERE, before any test module runs. That matters because
     `agents/critic_agent.py` calls `load_dotenv(override=True)`, which would
     otherwise rewrite the variable to Supabase's before `data.db` first read
     it, depending only on which test happened to import what first.
   * GUARD. `get_engine` refuses any non-SQLite engine for the rest of the
     session. This catches the other route: something later reassigning
     `data.db.DATABASE_URL` or reloading the module after the override.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TEST_DATABASE_URL = "sqlite:///" + (
    Path(tempfile.mkdtemp(prefix="asf-pytest-")) / "suite.sqlite").as_posix()
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

import data.db as _db  # noqa: E402  (must follow the pin)

if _db.DATABASE_URL != TEST_DATABASE_URL:
    raise RuntimeError(
        f"data.db was imported before tests/conftest.py could pin the test "
        f"database, and holds {_db.DATABASE_URL.split('@')[-1]!r}. Refusing to "
        f"run a suite that can write to it.")

_real_get_engine = _db.get_engine


def _suite_get_engine():
    engine = _real_get_engine()
    if engine.dialect.name != "sqlite":
        raise RuntimeError(
            f"the test suite reached a non-SQLite database "
            f"({engine.dialect.name}). Tests must never touch a real one; "
            f"patch get_engine to a tmp_path SQLite engine instead.")
    return engine


_db.get_engine = _suite_get_engine
