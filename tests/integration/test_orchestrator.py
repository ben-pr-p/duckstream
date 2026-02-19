"""Integration tests for the Orchestrator."""

from __future__ import annotations

import os
import shutil
import tempfile

import duckdb
import pytest

from duckstream.orchestrator import (
    CatalogNotFoundError,
    CyclicDependencyError,
    MissingTableError,
    NotDuckLakeError,
    Orchestrator,
)


def _make_ducklake_attach(tmpdir: str):
    """Create an attach function for a fresh DuckLake catalog."""
    meta_path = os.path.join(tmpdir, "meta.ddb")
    data_path = os.path.join(tmpdir, "data")
    os.makedirs(data_path, exist_ok=True)

    def attach(conn: duckdb.DuckDBPyConnection, name: str) -> None:
        conn.execute("INSTALL ducklake")
        conn.execute("LOAD ducklake")
        conn.execute(f"ATTACH 'ducklake:{meta_path}' AS {name} (DATA_PATH '{data_path}')")

    return attach


@pytest.fixture
def orch():
    """Fresh Orchestrator with a DuckLake catalog 'dl'."""
    tmpdir = tempfile.mkdtemp()
    conn = duckdb.connect()
    o = Orchestrator(conn=conn)
    o.add_catalog("dl", _make_ducklake_attach(tmpdir))

    yield o, conn, "dl"

    conn.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_basic_lifecycle(orch):
    """add_catalog, add_ivm, initialize, verify, maintain — single MV."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.orders (id INTEGER, amount INTEGER, region TEXT)")
    conn.execute(f"INSERT INTO {cat}.orders VALUES (1, 100, 'east'), (2, 200, 'west')")

    o.add_ivm(
        cat, "order_totals", "SELECT region, SUM(amount) AS total FROM orders GROUP BY region"
    )
    o.initialize()
    o.verify()

    # Make changes
    conn.execute(f"INSERT INTO {cat}.orders VALUES (3, 300, 'east')")

    # Maintain
    results = o.maintain()
    assert len(results) == 1
    assert results[0].duration_ms is not None
    assert results[0].duration_ms >= 0

    # Verify result
    rows = conn.execute(
        f"SELECT region, total FROM {cat}.main.order_totals ORDER BY region"
    ).fetchall()
    assert rows == [("east", 400), ("west", 200)]


def test_initialize_is_idempotent(orch):
    """Calling initialize() twice doesn't error or duplicate data."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.t (x INTEGER)")
    conn.execute(f"INSERT INTO {cat}.t VALUES (1), (2)")

    o.add_ivm(cat, "my_mv", "SELECT x FROM t")
    o.initialize()
    o.initialize()  # second call should be a no-op

    rows = conn.execute(f"SELECT x FROM {cat}.main.my_mv ORDER BY x").fetchall()
    assert rows == [(1,), (2,)]


# ---------------------------------------------------------------------------
# Chained MVs
# ---------------------------------------------------------------------------


def test_chained_mvs(orch):
    """MV_B depends on MV_A. Topo-sorts correctly. Maintains A before B."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.orders (id INTEGER, amount INTEGER, region TEXT)")
    conn.execute(
        f"INSERT INTO {cat}.orders VALUES (1, 100, 'east'), (2, 200, 'west'), (3, 150, 'east')"
    )

    o.add_ivm(
        cat,
        "order_totals",
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
    )
    o.add_ivm(
        cat,
        "big_regions",
        "SELECT region, total FROM order_totals WHERE total > 100",
    )
    o.initialize()
    o.verify()

    # Check topo order: order_totals before big_regions
    plan = o.get_maintenance_plan()
    names = [s.mv_name for s in plan.steps]
    assert names.index("order_totals") < names.index("big_regions")

    # Check levels: should be 2 levels
    assert len(plan.levels) == 2

    # Make changes and maintain
    conn.execute(f"INSERT INTO {cat}.orders VALUES (4, 500, 'west')")

    results = o.maintain()
    assert len(results) == 2

    # Verify MV_A
    rows_a = conn.execute(
        f"SELECT region, total FROM {cat}.main.order_totals ORDER BY region"
    ).fetchall()
    assert rows_a == [("east", 250), ("west", 700)]

    # Verify MV_B
    rows_b = conn.execute(
        f"SELECT region, total FROM {cat}.main.big_regions ORDER BY region"
    ).fetchall()
    assert rows_b == [("east", 250), ("west", 700)]


# ---------------------------------------------------------------------------
# verify() failure cases
# ---------------------------------------------------------------------------


def test_verify_missing_catalog():
    """verify() fails when MV's catalog was never attached."""
    tmpdir = tempfile.mkdtemp()
    conn = duckdb.connect()
    try:
        o = Orchestrator(conn=conn)
        # Register an MV in catalog "ghost" without ever calling add_catalog for it
        o.add_ivm("ghost", "my_mv", "SELECT 1 AS x FROM t")

        with pytest.raises(CatalogNotFoundError):
            o.verify()
    finally:
        conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_verify_not_ducklake():
    """verify() fails on non-DuckLake catalog."""
    conn = duckdb.connect()
    try:
        o = Orchestrator(conn=conn)
        # Attach a regular (non-ducklake) in-memory database
        conn.execute("ATTACH ':memory:' AS mem_cat")
        conn.execute("CREATE TABLE mem_cat.t (x INTEGER)")

        o.add_ivm("mem_cat", "my_mv", "SELECT x FROM t")

        with pytest.raises(NotDuckLakeError):
            o.verify()
    finally:
        conn.close()


def test_verify_missing_table(orch):
    """verify() fails when SQL references a table that doesn't exist."""
    o, conn, cat = orch

    # Register an MV referencing "nonexistent" — the table was never created
    o.add_ivm(cat, "my_mv", "SELECT x FROM nonexistent")

    with pytest.raises(MissingTableError):
        o.verify()


def test_verify_cycle(orch):
    """verify() fails on cyclic dependency."""
    o, conn, cat = orch

    # Create two real tables so the MVs compile successfully
    conn.execute(f"CREATE TABLE {cat}.a (x INTEGER)")
    conn.execute(f"CREATE TABLE {cat}.b (x INTEGER)")

    # MV named "a" reads from table "b", MV named "b" reads from table "a"
    # This creates a cycle: a -> b -> a
    o.add_ivm(cat, "a", "SELECT x FROM b")
    o.add_ivm(cat, "b", "SELECT x FROM a")

    with pytest.raises(CyclicDependencyError):
        o.verify()


# ---------------------------------------------------------------------------
# tree()
# ---------------------------------------------------------------------------


def test_tree_output(orch):
    """tree() returns formatted dependency tree."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.orders (id INTEGER, amount INTEGER, region TEXT)")
    conn.execute(f"INSERT INTO {cat}.orders VALUES (1, 100, 'east')")

    o.add_ivm(
        cat,
        "order_totals",
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
    )
    o.initialize()

    tree_str = o.tree()
    assert "ducklake catalog" in tree_str
    assert "orders" in tree_str
    assert "order_totals" in tree_str
    assert "MV" in tree_str
    assert "base table" in tree_str


# ---------------------------------------------------------------------------
# get_maintenance_plan with detail levels
# ---------------------------------------------------------------------------


def test_maintenance_plan_none(orch):
    """get_maintenance_plan with detail_level='none'."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.t (x INTEGER)")
    conn.execute(f"INSERT INTO {cat}.t VALUES (1)")

    o.add_ivm(cat, "my_mv", "SELECT x FROM t")
    o.initialize()

    plan = o.get_maintenance_plan(detail_level="none")
    assert len(plan.steps) == 1
    assert plan.steps[0].pending_snapshots is None
    assert plan.steps[0].pending_rows is None


def test_maintenance_plan_cursor_distance(orch):
    """get_maintenance_plan with detail_level='cursor_distance'."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.t (x INTEGER)")
    conn.execute(f"INSERT INTO {cat}.t VALUES (1)")

    o.add_ivm(cat, "my_mv", "SELECT x FROM t")
    o.initialize()

    # Note: DuckLake creates a snapshot for every write, including cursor
    # updates during maintain. So pending_snapshots is never truly zero after
    # setup — there's always at least 1 from the cursor update itself.
    # We verify the value increases after a real data change.
    plan_before = o.get_maintenance_plan(detail_level="cursor_distance")
    snap_before = plan_before.steps[0].pending_snapshots
    assert snap_before is not None

    # Make a change — should increase pending_snapshots
    conn.execute(f"INSERT INTO {cat}.t VALUES (2)")
    plan_after = o.get_maintenance_plan(detail_level="cursor_distance")
    snap_after = plan_after.steps[0].pending_snapshots
    assert snap_after is not None
    assert snap_after > snap_before


def test_maintenance_plan_rows_changed(orch):
    """get_maintenance_plan with detail_level='rows_changed'."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.t (x INTEGER)")
    conn.execute(f"INSERT INTO {cat}.t VALUES (1)")

    o.add_ivm(cat, "my_mv", "SELECT x FROM t")
    o.initialize()

    conn.execute(f"INSERT INTO {cat}.t VALUES (2)")
    conn.execute(f"INSERT INTO {cat}.t VALUES (3)")

    plan = o.get_maintenance_plan(detail_level="rows_changed")
    assert plan.steps[0].pending_rows is not None
    assert plan.steps[0].pending_rows >= 2


# ---------------------------------------------------------------------------
# advance_one()
# ---------------------------------------------------------------------------


def test_advance_one(orch):
    """advance_one() steps through in order."""
    o, conn, cat = orch

    conn.execute(f"CREATE TABLE {cat}.t (x INTEGER)")
    conn.execute(f"INSERT INTO {cat}.t VALUES (1)")

    o.add_ivm(cat, "my_mv", "SELECT x FROM t")
    o.initialize()

    conn.execute(f"INSERT INTO {cat}.t VALUES (2)")

    future = o.advance_one()
    assert future is not None
    step = future.result()
    assert step.mv_name == "my_mv"
    assert step.duration_ms is not None

    # No more work
    future2 = o.advance_one()
    assert future2 is None


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


def test_parallel_independent_mvs():
    """parallelism=2 with independent MVs."""
    tmpdir = tempfile.mkdtemp()
    conn = duckdb.connect()
    try:
        o = Orchestrator(conn=conn)
        o.add_catalog("dl", _make_ducklake_attach(tmpdir))
        cat = "dl"

        conn.execute(f"CREATE TABLE {cat}.t1 (x INTEGER)")
        conn.execute(f"CREATE TABLE {cat}.t2 (y INTEGER)")
        conn.execute(f"INSERT INTO {cat}.t1 VALUES (1), (2)")
        conn.execute(f"INSERT INTO {cat}.t2 VALUES (10), (20)")

        o.add_ivm(cat, "mv1", "SELECT x FROM t1")
        o.add_ivm(cat, "mv2", "SELECT y FROM t2")
        o.initialize()
        o.verify()

        # Make changes to both tables
        conn.execute(f"INSERT INTO {cat}.t1 VALUES (3)")
        conn.execute(f"INSERT INTO {cat}.t2 VALUES (30)")

        # Both should be in the same level (independent)
        plan = o.get_maintenance_plan()
        assert len(plan.levels) == 1
        assert len(plan.levels[0]) == 2

        results = o.maintain(parallelism=2)
        assert len(results) == 2

        # Verify both MVs updated
        rows1 = conn.execute(f"SELECT x FROM {cat}.main.mv1 ORDER BY x").fetchall()
        assert rows1 == [(1,), (2,), (3,)]

        rows2 = conn.execute(f"SELECT y FROM {cat}.main.mv2 ORDER BY y").fetchall()
        assert rows2 == [(10,), (20,), (30,)]
    finally:
        conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
