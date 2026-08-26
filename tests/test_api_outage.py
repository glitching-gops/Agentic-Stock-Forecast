"""
A database outage must be reported as an outage, never as an empty result.

On 2026-08-26 the live API could not reach Supabase and four read paths
disagreed about what that meant. `/api/stocks` returned HTTP 200 with
`total: 0` and `/api/leaderboard` returned HTTP 200 with `entries: []`, while
`/api/forecasts` and `/api/signals` returned 500. Nothing anywhere said the
database was down, and the frontend — which caches read-through on a timer —
replaced a good page with an empty board on the next revalidation.

The cause was a bare `except Exception` in each of those paths. Every one had
been written for a narrow case that genuinely should degrade: a fresh database
has no `index_membership` table until the first sync, and some leaderboard
columns are added lazily, so a query naming one of them must not 500. The
guard was wider than its intent, and "the schema is behind" and "the database
is gone" landed on the same branch.

These tests pin the discrimination in both directions. A guard that fails on
everything gets switched off and a guard that fails on nothing is decoration,
so the soft path is tested as hard as the loud one.
"""

import pytest
from pandas.errors import DatabaseError as PandasDatabaseError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import NoSuchTableError, OperationalError, ProgrammingError

from data.db import is_missing_relation


# ── Building the two failures, as the drivers really produce them ────────────

class _Psycopg2Error(Exception):
    """
    A psycopg2 error carries `pgcode`, which is None for a connection failure
    and a SQLSTATE for anything the server itself rejected.
    """

    def __init__(self, message: str, pgcode: str | None = None):
        super().__init__(message)
        self.pgcode = pgcode


def _unreachable_database() -> OperationalError:
    """
    What SQLAlchemy raises when the connection cannot be made.

    This is the exception the live outage produced. Note the class: psycopg2
    reports a refused connection as OperationalError, which is the SAME class
    SQLite raises for a missing table. The class alone cannot tell them apart,
    which is the whole reason is_missing_relation exists.
    """
    return OperationalError(
        "SELECT * FROM leaderboard",
        {},
        _Psycopg2Error("connection to server at \"db.abcdefgh.supabase.co\" "
                       "(10.24.0.7), port 5432 failed: FATAL: sorry, too many "
                       "clients already"),
    )


def _broken_engine(tmp_path):
    """
    A real engine that cannot connect. The path names a directory that does
    not exist, so SQLite fails to open the file — a genuine connect-time
    failure rather than a mock of one.
    """
    return create_engine(f"sqlite:///{tmp_path / 'no_such_dir' / 'x.db'}")


# ── The discriminator itself ─────────────────────────────────────────────────

def test_a_missing_table_is_recognised_and_an_outage_is_not():
    """
    The two cases the soft path has to separate, produced by executing real
    statements rather than by constructing the exceptions by hand.
    """
    import pandas as pd

    engine = create_engine("sqlite://")
    with pytest.raises(Exception) as missing:
        pd.read_sql(text("SELECT * FROM index_membership"), con=engine)

    assert is_missing_relation(missing.value), (
        "a table that does not exist yet must still fail soft; "
        f"got False for {missing.value!r}")

    assert not is_missing_relation(_unreachable_database()), (
        "an unreachable database must NOT be treated as a missing table — "
        "that is the defect that served an outage as an empty universe")


def test_the_inspector_reports_an_absent_table_as_its_own_class():
    """
    `inspect(engine).get_columns` never reaches the driver, so it raises
    NoSuchTableError whose str() is the bare table name — 'leaderboard'. That
    matches no message test, so keying the guard on the message alone would
    make the leaderboard 500 on a fresh database instead of degrading.
    """
    assert is_missing_relation(NoSuchTableError("leaderboard"))


def test_postgres_sqlstate_is_read_through_the_pandas_wrapper():
    """
    Against Postgres the message says `relation "x" does not exist`, which the
    SQLite text match does not recognise; the SQLSTATE is the only reliable
    signal. It is also two levels down — pandas wraps read_sql failures in its
    own DatabaseError, which wraps SQLAlchemy's, which holds the driver error.
    Reading only the outermost exception finds nothing.
    """
    undefined_table = ProgrammingError(
        "SELECT * FROM leaderboard", {},
        _Psycopg2Error('relation "leaderboard" does not exist', pgcode="42P01"))
    undefined_column = ProgrammingError(
        "SELECT pred_excess_return FROM leaderboard", {},
        _Psycopg2Error('column "pred_excess_return" does not exist',
                       pgcode="42703"))

    assert is_missing_relation(undefined_table)
    assert is_missing_relation(undefined_column)

    for inner in (undefined_table, undefined_column):
        wrapped = PandasDatabaseError("Execution failed on sql")
        wrapped.__cause__ = inner
        assert is_missing_relation(wrapped), (
            "the SQLSTATE must be found underneath pandas' own DatabaseError")

    # A server-side failure that is not a missing relation must not degrade.
    deadlock = ProgrammingError("SELECT 1", {},
                                _Psycopg2Error("deadlock detected",
                                               pgcode="40P01"))
    assert not is_missing_relation(deadlock), (
        "a SQLSTATE that is present and is not 42P01/42703 must propagate")


# ── The universe: /api/stocks reported "0 stocks" during the outage ──────────

def test_index_members_degrades_only_when_the_table_is_absent(tmp_path):
    import data.universe as universe

    original = universe.get_engine
    try:
        universe.get_engine = lambda: create_engine("sqlite://")
        assert universe.get_index_members("2026-08-26") == [], (
            "a database with no index_membership table must fail soft — a "
            "fresh deployment has none until the first sync")

        universe.get_engine = lambda: _broken_engine(tmp_path)
        with pytest.raises(Exception) as outage:
            universe.get_index_members("2026-08-26")
    finally:
        universe.get_engine = original

    assert not is_missing_relation(outage.value), (
        "the test must be raising on the outage, not on a missing table")


# ── The leaderboard: two guards, and the outage has to pass both ─────────────

def _leaderboard_engine():
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE leaderboard "
                          "(ticker TEXT, composite_score REAL)"))
        conn.execute(text("INSERT INTO leaderboard VALUES ('CANBK.NS', 23.86)"))
        conn.commit()
    return engine


def _call(engine, **kwargs):
    """
    Every argument is passed explicitly: FastAPI's Query() defaults are only
    resolved during a request, and calling a router function directly with
    them unresolved raises NotImplementedError instead of running the code.
    """
    import api.routers.leaderboard as lb

    args = dict(sector=None, verdict=None, evidence=None,
                sort_by="composite_score", limit=50)
    args.update(kwargs)

    original = lb.get_engine
    lb.get_engine = lambda: engine
    lb._leaderboard_columns.cache_clear()
    try:
        return lb.get_leaderboard(**args)
    finally:
        lb.get_engine = original
        lb._leaderboard_columns.cache_clear()


def test_leaderboard_degrades_when_the_table_has_not_been_created(tmp_path):
    """The soft path the guard was written for, still working."""
    response = _call(create_engine("sqlite://"))

    assert response.entries == []
    assert response.total == 0


def test_leaderboard_column_probe_propagates_an_outage(tmp_path):
    """
    The first guard. `_leaderboard_columns` asks the database what columns the
    table has; when that question cannot be asked at all, the answer is not
    "no columns", and returning frozenset() short-circuits the endpoint into
    an empty 200 before the real query is ever attempted.
    """
    with pytest.raises(Exception) as outage:
        _call(_broken_engine(tmp_path))

    assert not is_missing_relation(outage.value)


def test_leaderboard_query_propagates_an_outage(tmp_path):
    """
    The second guard, reached only once the column probe has succeeded — so
    the probe is stubbed with a real column set and the query is left to fail.
    An outage here used to be served as `entries: []` with HTTP 200, which the
    frontend cached as though it were the answer.
    """
    import api.routers.leaderboard as lb

    def _columns():
        return frozenset({"ticker", "composite_score", "last_updated"})

    _columns.cache_clear = lambda: None      # the endpoint calls this on the
                                             # missing-column path
    original_columns = lb._leaderboard_columns
    lb._leaderboard_columns = _columns
    try:
        with pytest.raises(Exception) as outage:
            _call(_broken_engine(tmp_path))
    finally:
        lb._leaderboard_columns = original_columns

    assert not is_missing_relation(outage.value)


# ── What the caller actually receives ────────────────────────────────────────

def _client():
    from fastapi.testclient import TestClient

    import api.main as main
    return TestClient(main.app, raise_server_exceptions=False)


def test_an_outage_is_served_as_503_and_is_not_cacheable():
    """
    503 plus no-store says "this is not an answer". A 200 with an empty list
    says "there are no stocks", and the frontend believes it — it revalidates
    on a timer and keeps whatever it last received, so a single revalidation
    during an outage replaces a working leaderboard with an empty one and
    nothing recovers it until the next successful fetch.
    """
    import data.universe as universe

    original = universe.get_universe
    universe.get_universe = lambda: (_ for _ in ()).throw(_unreachable_database())
    try:
        response = _client().get("/api/stocks")
    finally:
        universe.get_universe = original

    assert response.status_code == 503, (
        f"an unreachable database must not be a 2xx; got {response.status_code} "
        f"with body {response.text[:200]}")
    assert response.headers.get("cache-control") == "no-store", (
        "an outage response that can be cached is an empty result with extra "
        "steps")
    assert response.headers.get("retry-after") == "30"


def test_the_outage_response_does_not_publish_the_database_host():
    """
    `detail=str(exc)` was the previous behaviour on two endpoints. A psycopg2
    connection failure carries the database hostname and its resolved IP, and
    this response is public and unauthenticated.
    """
    import data.universe as universe

    exc = _unreachable_database()
    assert "supabase.co" in str(exc), "the fixture must carry a host to leak"

    original = universe.get_universe
    universe.get_universe = lambda: (_ for _ in ()).throw(exc)
    try:
        body = _client().get("/api/stocks").text
    finally:
        universe.get_universe = original

    assert "supabase.co" not in body and "10.24.0.7" not in body, (
        f"the connection string reached the public response: {body[:300]}")


def test_a_driver_failure_wrapped_by_pandas_is_also_an_outage():
    """
    Only a failure at CONNECT time arrives as a bare DBAPIError. A pooled
    connection that has gone stale fails mid-query, and pandas re-raises that
    as its own DatabaseError — a class the DBAPIError handler never sees. That
    is the shape a pooler outage takes once a connection has been handed out.
    """
    import data.universe as universe

    wrapped = PandasDatabaseError("Execution failed on sql 'SELECT ...'")
    wrapped.__cause__ = _unreachable_database()

    original = universe.get_universe
    universe.get_universe = lambda: (_ for _ in ()).throw(wrapped)
    try:
        response = _client().get("/api/stocks")
    finally:
        universe.get_universe = original

    assert response.status_code == 503
    assert "supabase.co" not in response.text


def test_the_sentiment_router_was_missed_by_the_first_fix(tmp_path):
    """
    Four read paths were fixed; there were five.

    `/api/sentiment/{ticker}/headlines` kept a bare `except Exception` that
    raised `HTTPException(500, detail=f"Database error: {e}")`. Two defects in
    one line. It caught DBAPIError, so an outage never reached the 503 handler
    and was served as a 500 telling the caller their request was broken; and it
    interpolated the exception, so the response published the database hostname
    and its resolved IP on a public unauthenticated endpoint.

    Found by the test suite rather than by review: retiring the Streamlit app
    broke a test that grepped its source, and reading this router to replace
    that test is what surfaced the leak.
    """
    import api.routers.sentiment as sent

    original = sent.get_engine
    sent.get_engine = lambda: _broken_engine(tmp_path)
    try:
        response = _client().get("/api/sentiment/RELIANCE.NS/headlines")
    finally:
        sent.get_engine = original

    assert response.status_code == 503, (
        f"an unreachable database must not be reported as a 500; got "
        f"{response.status_code} with body {response.text[:200]}")
    assert response.headers.get("cache-control") == "no-store"


def test_the_sentiment_router_does_not_publish_the_connection_string():
    """The other half of that line: `detail=f"...{e}"` on a public endpoint."""
    import api.routers.sentiment as sent

    exc = _unreachable_database()
    assert "supabase.co" in str(exc), "the fixture must carry a host to leak"

    # The failure has to happen INSIDE the try block. `get_engine()` is called
    # above it, so throwing from there propagates uncaught and never exercises
    # the handler under test — the first draft of this test did exactly that
    # and passed against the unfixed router.
    class _DeadEngine:
        def connect(self):
            raise exc

    original = sent.get_engine
    sent.get_engine = _DeadEngine
    try:
        body = _client().get("/api/sentiment/RELIANCE.NS/headlines").text
    finally:
        sent.get_engine = original

    assert "supabase.co" not in body and "10.24.0.7" not in body, (
        f"the connection string reached the public response: {body[:300]}")


def test_a_pandas_error_that_is_not_a_driver_failure_stays_a_500():
    """
    The counterweight. A DatabaseError with no driver failure underneath it is
    our own defect — a malformed statement, say — and calling that an outage
    would tell the caller to retry something that will never succeed.
    """
    import data.universe as universe

    original = universe.get_universe
    universe.get_universe = lambda: (_ for _ in ()).throw(
        PandasDatabaseError("Execution failed on sql: syntax error"))
    try:
        response = _client().get("/api/stocks")
    finally:
        universe.get_universe = original

    assert response.status_code == 500, (
        f"our own SQL defect must not be reported as a database outage; "
        f"got {response.status_code}")
