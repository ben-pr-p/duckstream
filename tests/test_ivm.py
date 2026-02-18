"""
Property-based tests for IVM correctness.

Core invariant: for any supported view definition, applying the generated
maintenance SQL after a delta should produce the same result as recomputing
the view from scratch on the updated base tables.

All base tables live in a DuckLake catalog. Deltas are read via
ducklake_table_changes() — no manually-populated delta tables.
"""

import duckdb
import pytest
from hypothesis import given, settings, HealthCheck

from tests.strategies import (
    Column,
    Delta,
    Scenario,
    Row,
    Table,
    single_table_select,
    single_table_aggregate,
)
from tests.conftest import DUCKLAKE_CATALOG
from ducklake_ivm.compiler import compile_ivm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sql_literal(value: object) -> str:
    """Convert a Python value to a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, float):
        return f"{value!r}"
    return str(value)


def insert_rows(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    rows: list[Row],
    catalog: str,
):
    """Insert rows into a DuckLake table."""
    if not rows:
        return
    cols = ", ".join(table.col_names)
    for row in rows:
        vals = ", ".join(sql_literal(row.values[c]) for c in table.col_names)
        con.execute(f"INSERT INTO {catalog}.{table.name} ({cols}) VALUES ({vals})")


def delete_rows(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    rows: list[Row],
    catalog: str,
):
    """Delete exact rows from a DuckLake table (matches all columns)."""
    if not rows:
        return
    for row in rows:
        conditions = " AND ".join(
            f"{c} = {sql_literal(row.values[c])}" for c in table.col_names
        )
        # Delete only one matching row
        con.execute(f"""
            DELETE FROM {catalog}.{table.name}
            WHERE rowid = (
                SELECT rowid FROM {catalog}.{table.name}
                WHERE {conditions} LIMIT 1
            )
        """)


def get_snapshot(con: duckdb.DuckDBPyConnection, catalog: str) -> int:
    """Get the current (latest) snapshot ID for the DuckLake catalog."""
    return con.execute(
        f"SELECT MAX(snapshot_id) FROM ducklake_snapshots({catalog})"
    ).fetchone()[0]


def setup_scenario(
    con: duckdb.DuckDBPyConnection,
    scenario: Scenario,
    catalog: str,
):
    """Create DuckLake tables and insert initial data."""
    for table in scenario.tables:
        con.execute(table.ddl(catalog))
        insert_rows(con, table, scenario.initial_data.get(table.name, []), catalog)


def qualified_view_sql(view_sql: str, catalog: str) -> str:
    """Qualify unqualified table names in the view SQL with the catalog.

    Simple prefix replacement — works for our generated scenarios where
    table names appear after FROM/JOIN keywords.
    """
    # For now, use DuckDB's USE to set the default catalog so that
    # unqualified names resolve to DuckLake tables.
    # The view_sql stays unqualified; the caller sets USE before executing.
    return view_sql


def recompute_view(
    con: duckdb.DuckDBPyConnection,
    view_sql: str,
    catalog: str,
) -> list[tuple]:
    """Run the view query from scratch and return sorted results."""
    con.execute(f"USE {catalog}")
    result = con.execute(view_sql).fetchall()
    con.execute("USE memory")
    return sorted(result)


def apply_deltas(
    con: duckdb.DuckDBPyConnection,
    scenario: Scenario,
    catalog: str,
):
    """Apply deltas directly to the DuckLake base tables."""
    table_by_name = {t.name: t for t in scenario.tables}
    for delta in scenario.deltas:
        table = table_by_name[delta.table_name]
        delete_rows(con, table, delta.deletes, catalog)
        insert_rows(con, table, delta.inserts, catalog)


def read_mv(con: duckdb.DuckDBPyConnection, mv_table: str = "mv") -> list[tuple]:
    """Read the materialized view table (in memory catalog) and return sorted results."""
    result = con.execute(f"SELECT * FROM memory.main.{mv_table}").fetchall()
    return sorted(result)


# ---------------------------------------------------------------------------
# The core oracle test
# ---------------------------------------------------------------------------

def assert_ivm_correct(scenario: Scenario, ducklake_fixture):
    """The core correctness check: maintained MV == recomputed view.

    Steps:
        1. Create DuckLake tables with initial data
        2. Record the snapshot after initial data load
        3. Compile IVM maintenance SQL from the view definition
        4. Materialize the view (full computation into memory.main.mv)
        5. Apply deltas to the DuckLake base tables
        6. Record the post-delta snapshot
        7. Run the compiler's maintenance SQL (reads ducklake_table_changes)
        8. Recompute the view from scratch on the updated tables
        9. Assert: maintained MV == recomputed view
    """
    con, catalog = ducklake_fixture

    # 1. Set up DuckLake tables with initial data
    setup_scenario(con, scenario, catalog)

    # 2. Record snapshot after initial data
    pre_snapshot = get_snapshot(con, catalog)

    # 3. Compile IVM
    ivm_output = compile_ivm(scenario.view_sql, dialect="duckdb")

    # 4. Initial materialization into memory catalog
    con.execute(f"USE {catalog}")
    con.execute(f"CREATE TABLE memory.main.mv AS {scenario.view_sql}")
    con.execute("USE memory")

    # 5. Apply deltas to DuckLake base tables
    apply_deltas(con, scenario, catalog)

    # 6. Record post-delta snapshot
    post_snapshot = get_snapshot(con, catalog)

    # 7. Run maintenance SQL
    #    The compiler's maintain statements will reference
    #    ducklake_table_changes() with the snapshot range.
    for stmt in ivm_output.maintain:
        con.execute(stmt)

    # 8. Read maintained MV
    maintained = read_mv(con)

    # 9. Recompute from scratch and compare
    expected = recompute_view(con, scenario.view_sql, catalog)

    assert maintained == expected, (
        f"IVM result differs from recomputation.\n"
        f"View: {scenario.view_sql}\n"
        f"Snapshots: {pre_snapshot} -> {post_snapshot}\n"
        f"Maintained: {maintained}\n"
        f"Expected:   {expected}"
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

@given(scenario=single_table_select())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_select_project_filter(scenario, ducklake):
    """SELECT/PROJECT/WHERE on a single table maintains correctly."""
    assert_ivm_correct(scenario, ducklake)


@given(scenario=single_table_aggregate())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_single_table_aggregate(scenario, ducklake):
    """GROUP BY with SUM/COUNT/AVG on a single table maintains correctly."""
    assert_ivm_correct(scenario, ducklake)


# ---------------------------------------------------------------------------
# Deterministic smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    """Hand-written scenarios to sanity-check the oracle harness."""

    def test_simple_select_all(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "INTEGER")])],
            initial_data={"t": [
                Row({"id": 1, "val": 10}),
                Row({"id": 2, "val": 20}),
                Row({"id": 3, "val": 30}),
            ]},
            view_sql="SELECT id, val FROM t",
            deltas=[Delta("t",
                inserts=[Row({"id": 4, "val": 40})],
                deletes=[Row({"id": 2, "val": 20})],
            )],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_select_with_filter(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "INTEGER")])],
            initial_data={"t": [
                Row({"id": 1, "val": 10}),
                Row({"id": 2, "val": 20}),
                Row({"id": 3, "val": 30}),
            ]},
            view_sql="SELECT id, val FROM t WHERE val > 15",
            deltas=[Delta("t",
                inserts=[Row({"id": 4, "val": 5}), Row({"id": 5, "val": 50})],
                deletes=[Row({"id": 3, "val": 30})],
            )],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_count_aggregate(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={"t": [
                Row({"grp": "a", "val": 1}),
                Row({"grp": "a", "val": 2}),
                Row({"grp": "b", "val": 3}),
            ]},
            view_sql="SELECT grp, COUNT(*) AS agg_val FROM t GROUP BY grp",
            deltas=[Delta("t",
                inserts=[Row({"grp": "a", "val": 10}), Row({"grp": "c", "val": 5})],
                deletes=[Row({"grp": "b", "val": 3})],
            )],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_sum_aggregate(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={"t": [
                Row({"grp": "a", "val": 10}),
                Row({"grp": "a", "val": 20}),
                Row({"grp": "b", "val": 30}),
            ]},
            view_sql="SELECT grp, SUM(val) AS agg_val FROM t GROUP BY grp",
            deltas=[Delta("t",
                inserts=[Row({"grp": "a", "val": 5})],
                deletes=[Row({"grp": "a", "val": 10})],
            )],
        )
        assert_ivm_correct(scenario, ducklake)
