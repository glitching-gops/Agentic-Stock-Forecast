"""
The test suite must never be able to write to a real database.

Before 2026-09-13 it could, and did: in a checkout whose `.env` pointed at
Supabase, one test wrote 221 fake weekly runs into production
`experiment_runs`. tests/conftest.py now pins a throwaway SQLite database and
guards `get_engine`. Each layer is tested alone here, so neither can quietly
stand in for the other.
"""
import pytest


def test_the_suite_database_is_the_throwaway_sqlite_conftest_pinned():
    import data.db as db

    assert db.DATABASE_URL.startswith("sqlite:///") and "asf-pytest-" in db.DATABASE_URL, (
        f"the suite is pointed at {db.DATABASE_URL.split('@')[-1]!r}, not "
        f"the throwaway database tests/conftest.py pins")
    assert db.get_engine().dialect.name == "sqlite"


def test_get_engine_refuses_a_non_sqlite_database(monkeypatch):
    import data.db as db

    # No connection is attempted: create_engine is lazy, and the guard
    # refuses on the dialect before anything is sent.
    monkeypatch.setattr(db, "DATABASE_URL", "postgresql+psycopg2://u:p@127.0.0.1:1/x")
    monkeypatch.setattr(db, "_ENGINE", None)
    with pytest.raises(RuntimeError, match="non-SQLite"):
        db.get_engine()
