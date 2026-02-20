import os
import shutil
import tempfile

import duckdb
import pytest

from duckstream import compile_ivm
from tests.strategies import Row, Scenario, Table

DUCKLAKE_CATALOG = "dl"


@pytest.fixture
def ducklake():
    """Fresh isolated DuckLake instance per test.

    Creates a random temp directory for the metadata catalog and data files.
    Attaches the DuckLake catalog as 'dl'. The MV and any non-DuckLake tables
    live in the default 'memory' catalog.

    Yields a (connection, catalog_name) tuple.
    """
    tmpdir = tempfile.mkdtemp()
    meta_path = os.path.join(tmpdir, "meta.ddb")
    data_path = os.path.join(tmpdir, "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL ducklake")
    con.execute("LOAD ducklake")
    con.execute(f"ATTACH 'ducklake:{meta_path}' AS {DUCKLAKE_CATALOG} (DATA_PATH '{data_path}')")

    yield con, DUCKLAKE_CATALOG

    con.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Oracle harness helpers (used by all test_*.py files)
# ---------------------------------------------------------------------------


def make_ducklake():
    """Create a fresh isolated DuckLake instance. Returns (con, catalog, cleanup_fn)."""
    tmpdir = tempfile.mkdtemp()
    meta_path = os.path.join(tmpdir, "meta.ddb")
    data_path = os.path.join(tmpdir, "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL ducklake")
    con.execute("LOAD ducklake")
    con.execute(f"ATTACH 'ducklake:{meta_path}' AS {DUCKLAKE_CATALOG} (DATA_PATH '{data_path}')")

    def cleanup():
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return con, DUCKLAKE_CATALOG, cleanup


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
            f"{c} IS NOT DISTINCT FROM {sql_literal(row.values[c])}" for c in table.col_names
        )
        con.execute(f"""
            DELETE FROM {catalog}.{table.name}
            WHERE rowid = (
                SELECT rowid FROM {catalog}.{table.name}
                WHERE {conditions} LIMIT 1
            )
        """)


def setup_scenario(
    con: duckdb.DuckDBPyConnection,
    scenario: Scenario,
    catalog: str,
):
    """Create DuckLake tables and insert initial data."""
    for table in scenario.tables:
        con.execute(table.ddl(catalog))
        insert_rows(con, table, scenario.initial_data.get(table.name, []), catalog)


def _null_safe_sort_key(row: tuple) -> tuple:
    """Sort key that handles NULLs by sorting them last."""
    return tuple((1, None) if v is None else (0, v) for v in row)


def recompute_view(
    con: duckdb.DuckDBPyConnection,
    view_sql: str,
    catalog: str,
) -> list[tuple]:
    """Run the view query from scratch and return sorted results."""
    con.execute(f"USE {catalog}")
    result = con.execute(view_sql).fetchall()
    con.execute("USE memory")
    return sorted(result, key=_null_safe_sort_key)


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


def read_mv(
    con: duckdb.DuckDBPyConnection,
    mv_fqn: str,
) -> list[tuple]:
    """Read the MV, excluding _ivm_* auxiliary columns, return sorted results."""
    result = con.execute(f"SELECT * FROM {mv_fqn} LIMIT 0")
    all_cols = [desc[0] for desc in result.description]
    visible_cols = [c for c in all_cols if not c.startswith("_ivm_")]
    if not visible_cols:
        return []
    cols_sql = ", ".join(visible_cols)
    result = con.execute(f"SELECT {cols_sql} FROM {mv_fqn}").fetchall()
    return sorted(result, key=_null_safe_sort_key)


def _initialize_mv(con, plan):
    """Initialize a MaterializedView and all its inner MVs (depth-first)."""
    # Initialize inner MVs first (they must exist before the outer MV)
    for inner in plan.inner_mvs:
        _initialize_mv(con, inner)

    con.execute(plan.create_cursors_table)
    con.execute(plan.create_mv)
    for stmt in plan.initialize_cursors:
        con.execute(stmt)


def _maintain_mv(con, plan):
    """Run maintenance for a MaterializedView and all its inner MVs (depth-first)."""
    # Maintain inner MVs first (they must be up-to-date before the outer MV)
    for inner in plan.inner_mvs:
        _maintain_mv(con, inner)

    for stmt in plan.maintain:
        con.execute(stmt)


def assert_ivm_correct(scenario: Scenario, ducklake_fixture):
    """The core correctness check: maintained MV == recomputed view."""
    con, catalog = ducklake_fixture

    # 1. Set up DuckLake tables with initial data
    setup_scenario(con, scenario, catalog)

    # 2. Compile IVM
    plan = compile_ivm(
        scenario.view_sql,
        mv_catalog=catalog,
    )

    # 3. Set up MV and cursors (including inner MVs)
    _initialize_mv(con, plan)

    # 4. Apply deltas to DuckLake base tables
    apply_deltas(con, scenario, catalog)

    # 5. Run maintenance SQL (including inner MVs)
    _maintain_mv(con, plan)

    # 6. Read maintained MV (excluding _ivm_* columns, applying HAVING if present)
    if plan.query_mv:
        result = con.execute(plan.query_mv).fetchall()
        maintained = sorted(result, key=_null_safe_sort_key)
    else:
        mv_fqn = f"{catalog}.main.mv"
        maintained = read_mv(con, mv_fqn)

    # 7. Recompute from scratch and compare
    expected = recompute_view(con, scenario.view_sql, catalog)

    assert maintained == expected, (
        f"IVM result differs from recomputation.\n"
        f"View: {scenario.view_sql}\n"
        f"Maintained ({len(maintained)} rows): {maintained}\n"
        f"Expected ({len(expected)} rows):   {expected}\n"
        f"Plan maintain SQL:\n" + "\n---\n".join(plan.maintain)
    )
