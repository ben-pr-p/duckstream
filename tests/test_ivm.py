"""
Property-based tests for IVM correctness.

Core invariant: for any supported view definition, applying the generated
maintenance SQL after a delta should produce the same result as recomputing
the view from scratch on the updated base tables.

All base tables live in a DuckLake catalog. Deltas are read via
ducklake_table_changes() — no manually-populated delta tables.
"""

import os
import shutil
import tempfile

import duckdb
from hypothesis import HealthCheck, given, settings

from ducklake_ivm import compile_ivm
from tests.conftest import DUCKLAKE_CATALOG
from tests.strategies import (
    Column,
    Delta,
    Row,
    Scenario,
    Table,
    join_then_aggregate,
    single_table_aggregate,
    single_table_distinct,
    single_table_select,
    two_table_join,
)


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
            f"{c} IS NOT DISTINCT FROM {sql_literal(row.values[c])}" for c in table.col_names
        )
        # Delete only one matching row
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


def read_mv(
    con: duckdb.DuckDBPyConnection,
    mv_fqn: str,
) -> list[tuple]:
    """Read the MV, excluding _ivm_* auxiliary columns, return sorted results."""
    # Get column names
    result = con.execute(f"SELECT * FROM {mv_fqn} LIMIT 0")
    all_cols = [desc[0] for desc in result.description]
    visible_cols = [c for c in all_cols if not c.startswith("_ivm_")]
    if not visible_cols:
        return []
    cols_sql = ", ".join(visible_cols)
    result = con.execute(f"SELECT {cols_sql} FROM {mv_fqn}").fetchall()
    return sorted(result)


# ---------------------------------------------------------------------------
# The core oracle test
# ---------------------------------------------------------------------------


def assert_ivm_correct(scenario: Scenario, ducklake_fixture):
    """The core correctness check: maintained MV == recomputed view."""
    con, catalog = ducklake_fixture

    # 1. Set up DuckLake tables with initial data
    setup_scenario(con, scenario, catalog)

    # 2. Compile IVM
    plan = compile_ivm(
        scenario.view_sql,
        dialect="duckdb",
        mv_catalog=catalog,
    )

    # 3. Set up MV and cursors
    con.execute(plan.create_cursors_table)
    con.execute(plan.create_mv)
    for stmt in plan.initialize_cursors:
        con.execute(stmt)

    # 4. Apply deltas to DuckLake base tables
    apply_deltas(con, scenario, catalog)

    # 5. Run maintenance SQL
    for stmt in plan.maintain:
        con.execute(stmt)

    # 6. Read maintained MV (excluding _ivm_* columns)
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


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(scenario=single_table_select())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_select_project_filter(scenario):
    """SELECT/PROJECT/WHERE on a single table maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


@given(scenario=single_table_aggregate())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_single_table_aggregate(scenario):
    """GROUP BY with SUM/COUNT/AVG on a single table maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


@given(scenario=two_table_join())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_two_table_join(scenario):
    """Two-table inner JOIN maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


@given(scenario=join_then_aggregate())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_join_then_aggregate(scenario):
    """Two-table JOIN + GROUP BY aggregate maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


@given(scenario=single_table_distinct())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_single_table_distinct(scenario):
    """SELECT DISTINCT on a single table maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Deterministic smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    """Hand-written scenarios to sanity-check the oracle harness."""

    def test_simple_select_all(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"id": 1, "val": 10}),
                    Row({"id": 2, "val": 20}),
                    Row({"id": 3, "val": 30}),
                ]
            },
            view_sql="SELECT id, val FROM t",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"id": 4, "val": 40})],
                    deletes=[Row({"id": 2, "val": 20})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_select_with_filter(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"id": 1, "val": 10}),
                    Row({"id": 2, "val": 20}),
                    Row({"id": 3, "val": 30}),
                ]
            },
            view_sql="SELECT id, val FROM t WHERE val > 15",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"id": 4, "val": 5}), Row({"id": 5, "val": 50})],
                    deletes=[Row({"id": 3, "val": 30})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_count_aggregate(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 1}),
                    Row({"grp": "a", "val": 2}),
                    Row({"grp": "b", "val": 3}),
                ]
            },
            view_sql="SELECT grp, COUNT(*) AS agg_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "a", "val": 10}), Row({"grp": "c", "val": 5})],
                    deletes=[Row({"grp": "b", "val": 3})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_sum_aggregate(self, ducklake):
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 10}),
                    Row({"grp": "a", "val": 20}),
                    Row({"grp": "b", "val": 30}),
                ]
            },
            view_sql="SELECT grp, SUM(val) AS agg_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "a", "val": 5})],
                    deletes=[Row({"grp": "a", "val": 10})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_insert_left_only(self, ducklake):
        """Inner join: insert into left table only -> new matches appear."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [
                    Row({"oid": 1, "cid": 1}),
                    Row({"oid": 2, "cid": 2}),
                ],
                "customers": [
                    Row({"cid": 1, "name": "alice"}),
                    Row({"cid": 2, "name": "bob"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders JOIN customers ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("orders", inserts=[Row({"oid": 3, "cid": 1})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_insert_right_only(self, ducklake):
        """Inner join: insert into right table only -> new matches appear."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [
                    Row({"oid": 1, "cid": 1}),
                    Row({"oid": 2, "cid": 2}),
                ],
                "customers": [
                    Row({"cid": 1, "name": "alice"}),
                    Row({"cid": 2, "name": "bob"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders JOIN customers ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("customers", inserts=[Row({"cid": 3, "name": "carol"})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_insert_both(self, ducklake):
        """Inner join: insert into both tables -> cross-delta term exercised."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [
                    Row({"oid": 1, "cid": 1}),
                ],
                "customers": [
                    Row({"cid": 1, "name": "alice"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders JOIN customers ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("orders", inserts=[Row({"oid": 2, "cid": 3})], deletes=[]),
                Delta("customers", inserts=[Row({"cid": 3, "name": "carol"})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_delete_participant(self, ducklake):
        """Inner join: delete row that participates in join -> matches removed."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [
                    Row({"oid": 1, "cid": 1}),
                    Row({"oid": 2, "cid": 2}),
                ],
                "customers": [
                    Row({"cid": 1, "name": "alice"}),
                    Row({"cid": 2, "name": "bob"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders JOIN customers ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[],
                    deletes=[Row({"oid": 1, "cid": 1})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_delete_nonparticipant(self, ducklake):
        """Inner join: delete row that doesn't participate -> MV unchanged."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [
                    Row({"oid": 1, "cid": 1}),
                    Row({"oid": 2, "cid": 99}),
                ],
                "customers": [
                    Row({"cid": 1, "name": "alice"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders JOIN customers ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[],
                    deletes=[Row({"oid": 2, "cid": 99})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_sum_aggregate(self, ducklake):
        """JOIN + SUM aggregate: insert an order, verify region totals update."""
        orders = Table(
            "orders",
            [Column("oid", "INTEGER"), Column("store_id", "INTEGER"), Column("amount", "INTEGER")],
        )
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])
        scenario = Scenario(
            tables=[orders, stores],
            initial_data={
                "orders": [
                    Row({"oid": 1, "store_id": 1, "amount": 100}),
                    Row({"oid": 2, "store_id": 2, "amount": 200}),
                    Row({"oid": 3, "store_id": 1, "amount": 50}),
                ],
                "stores": [
                    Row({"store_id": 1, "region": "east"}),
                    Row({"store_id": 2, "region": "west"}),
                ],
            },
            view_sql=(
                "SELECT stores.region, SUM(orders.amount) AS total"
                " FROM orders JOIN stores ON orders.store_id = stores.store_id"
                " GROUP BY stores.region"
            ),
            deltas=[
                Delta(
                    "orders", inserts=[Row({"oid": 4, "store_id": 2, "amount": 300})], deletes=[]
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_count_aggregate(self, ducklake):
        """JOIN + COUNT(*) aggregate."""
        orders = Table("orders", [Column("oid", "INTEGER"), Column("store_id", "INTEGER")])
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])
        scenario = Scenario(
            tables=[orders, stores],
            initial_data={
                "orders": [
                    Row({"oid": 1, "store_id": 1}),
                    Row({"oid": 2, "store_id": 2}),
                    Row({"oid": 3, "store_id": 1}),
                ],
                "stores": [
                    Row({"store_id": 1, "region": "east"}),
                    Row({"store_id": 2, "region": "west"}),
                ],
            },
            view_sql=(
                "SELECT stores.region, COUNT(*) AS cnt"
                " FROM orders JOIN stores ON orders.store_id = stores.store_id"
                " GROUP BY stores.region"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"oid": 4, "store_id": 1})],
                    deletes=[Row({"oid": 2, "store_id": 2})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_join_aggregate_delete_store(self, ducklake):
        """JOIN + aggregate: delete a store, verify groups disappear."""
        orders = Table(
            "orders",
            [Column("oid", "INTEGER"), Column("store_id", "INTEGER"), Column("amount", "INTEGER")],
        )
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])
        scenario = Scenario(
            tables=[orders, stores],
            initial_data={
                "orders": [
                    Row({"oid": 1, "store_id": 1, "amount": 100}),
                    Row({"oid": 2, "store_id": 2, "amount": 200}),
                ],
                "stores": [
                    Row({"store_id": 1, "region": "east"}),
                    Row({"store_id": 2, "region": "west"}),
                ],
            },
            view_sql=(
                "SELECT stores.region, SUM(orders.amount) AS total"
                " FROM orders JOIN stores ON orders.store_id = stores.store_id"
                " GROUP BY stores.region"
            ),
            deltas=[
                Delta("stores", inserts=[], deletes=[Row({"store_id": 2, "region": "west"})]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_distinct_basic(self, ducklake):
        """SELECT DISTINCT with duplicate rows."""
        scenario = Scenario(
            tables=[Table("t", [Column("a", "INTEGER"), Column("b", "VARCHAR")])],
            initial_data={
                "t": [
                    Row({"a": 1, "b": "x"}),
                    Row({"a": 1, "b": "x"}),
                    Row({"a": 2, "b": "y"}),
                ]
            },
            view_sql="SELECT DISTINCT a, b FROM t",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"a": 3, "b": "z"}), Row({"a": 1, "b": "x"})],
                    deletes=[Row({"a": 2, "b": "y"})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_distinct_with_where(self, ducklake):
        """SELECT DISTINCT with a WHERE filter."""
        scenario = Scenario(
            tables=[Table("t", [Column("a", "INTEGER"), Column("b", "VARCHAR")])],
            initial_data={
                "t": [
                    Row({"a": 1, "b": "x"}),
                    Row({"a": 2, "b": "x"}),
                    Row({"a": 3, "b": "y"}),
                    Row({"a": 3, "b": "y"}),
                ]
            },
            view_sql="SELECT DISTINCT b FROM t WHERE a > 1",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"a": 4, "b": "x"})],
                    deletes=[Row({"a": 3, "b": "y"})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_distinct_all_unique(self, ducklake):
        """SELECT DISTINCT where all rows are already unique."""
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "VARCHAR")])],
            initial_data={
                "t": [
                    Row({"id": 1, "val": "a"}),
                    Row({"id": 2, "val": "b"}),
                    Row({"id": 3, "val": "c"}),
                ]
            },
            view_sql="SELECT DISTINCT id, val FROM t",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"id": 4, "val": "d"})],
                    deletes=[Row({"id": 1, "val": "a"})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_min_delete_current(self, ducklake):
        """MIN: delete the current minimum, verify rescan picks up the next one."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 1}),
                    Row({"grp": "a", "val": 5}),
                    Row({"grp": "a", "val": 10}),
                    Row({"grp": "b", "val": 20}),
                ]
            },
            view_sql="SELECT grp, MIN(val) AS min_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[],
                    deletes=[Row({"grp": "a", "val": 1})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_min_insert_new_minimum(self, ducklake):
        """MIN: insert a value below the current minimum."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 5}),
                    Row({"grp": "a", "val": 10}),
                ]
            },
            view_sql="SELECT grp, MIN(val) AS min_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "a", "val": 2})],
                    deletes=[],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_max_delete_current(self, ducklake):
        """MAX: delete the current maximum, verify rescan picks up the next one."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 1}),
                    Row({"grp": "a", "val": 5}),
                    Row({"grp": "a", "val": 10}),
                ]
            },
            view_sql="SELECT grp, MAX(val) AS max_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[],
                    deletes=[Row({"grp": "a", "val": 10})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_max_insert_new_maximum(self, ducklake):
        """MAX: insert a value above the current maximum."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 5}),
                    Row({"grp": "a", "val": 10}),
                ]
            },
            view_sql="SELECT grp, MAX(val) AS max_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "a", "val": 20})],
                    deletes=[],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_min_group_disappears(self, ducklake):
        """MIN: delete all rows in a group, the group should disappear."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 5}),
                    Row({"grp": "b", "val": 10}),
                ]
            },
            view_sql="SELECT grp, MIN(val) AS min_val FROM t GROUP BY grp",
            deltas=[
                Delta(
                    "t",
                    inserts=[],
                    deletes=[Row({"grp": "b", "val": 10})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)
