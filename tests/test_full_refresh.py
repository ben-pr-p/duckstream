"""Tests for full-refresh + bag-diff fallback path."""

from duckstream import compile_ivm
from duckstream.materialized_view import UnsupportedSQLError
from tests.conftest import (
    _initialize_mv,
    _maintain_mv,
    _null_safe_sort_key,
    assert_ivm_correct,
    recompute_view,
)
from tests.strategies import Column, Delta, Row, Scenario, Table

# ---------------------------------------------------------------------------
# Smoke tests — queries that previously raised UnsupportedSQLError
# ---------------------------------------------------------------------------


class TestSmokeFallback:
    """Queries that were previously unsupported now produce working MVs."""

    def test_outer_join_with_aggregate(self, ducklake):
        """LEFT JOIN + SUM aggregate falls back to full refresh."""
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
                " FROM orders LEFT JOIN stores ON orders.store_id = stores.store_id"
                " GROUP BY stores.region"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"oid": 3, "store_id": 1, "amount": 50})],
                    deletes=[],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_distinct_with_group_by(self, ducklake):
        """DISTINCT + GROUP BY falls back to full refresh."""
        items = Table(
            "items",
            [Column("cat", "VARCHAR"), Column("price", "INTEGER")],
        )
        scenario = Scenario(
            tables=[items],
            initial_data={
                "items": [
                    Row({"cat": "a", "price": 10}),
                    Row({"cat": "a", "price": 20}),
                    Row({"cat": "b", "price": 30}),
                ],
            },
            view_sql="SELECT DISTINCT cat, SUM(price) AS total FROM items GROUP BY cat",
            deltas=[
                Delta("items", inserts=[Row({"cat": "a", "price": 5})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


# ---------------------------------------------------------------------------
# Strategy introspection
# ---------------------------------------------------------------------------


class TestStrategyIntrospection:
    """Verify strategy metadata is set correctly."""

    def test_incremental_strategy(self):
        """Normal queries get incremental strategy."""
        mv = compile_ivm("SELECT x FROM t", mv_catalog="dl")
        assert mv.strategy == "incremental"

    def test_full_refresh_strategy(self):
        """Outer join + aggregate gets full_refresh strategy."""
        mv = compile_ivm(
            "SELECT b.region, SUM(a.amount) AS total"
            " FROM a LEFT JOIN b ON a.id = b.id"
            " GROUP BY b.region",
            mv_catalog="dl",
        )
        assert mv.strategy == "full_refresh"
        assert "full_refresh" in mv.features

    def test_genuinely_invalid_still_raises(self):
        """no_table and having_no_agg still raise, not fallback."""
        import pytest

        with pytest.raises(UnsupportedSQLError, match="No tables found"):
            compile_ivm("SELECT 1", mv_catalog="dl")

        with pytest.raises(UnsupportedSQLError, match="HAVING requires"):
            compile_ivm("SELECT x FROM t HAVING x > 1", mv_catalog="dl")


# ---------------------------------------------------------------------------
# Correctness oracle
# ---------------------------------------------------------------------------


class TestCorrectnessOracle:
    """Full-refresh MV matches recomputed view after deltas."""

    def test_outer_join_agg_insert_and_delete(self, ducklake):
        """Outer join + aggregate with both inserts and deletes."""
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
                    Row({"oid": 3, "store_id": 1, "amount": 300}),
                ],
                "stores": [
                    Row({"store_id": 1, "region": "east"}),
                    Row({"store_id": 2, "region": "west"}),
                ],
            },
            view_sql=(
                "SELECT stores.region, SUM(orders.amount) AS total"
                " FROM orders LEFT JOIN stores ON orders.store_id = stores.store_id"
                " GROUP BY stores.region"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"oid": 4, "store_id": 2, "amount": 50})],
                    deletes=[Row({"oid": 1, "store_id": 1, "amount": 100})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_multiple_maintenance_rounds(self, ducklake):
        """Full-refresh MV stays correct across multiple maintenance rounds."""
        con, catalog = ducklake
        orders = Table(
            "orders",
            [
                Column("oid", "INTEGER"),
                Column("store_id", "INTEGER"),
                Column("amount", "INTEGER"),
            ],
        )
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])

        con.execute(orders.ddl(catalog))
        con.execute(stores.ddl(catalog))
        con.execute(f"INSERT INTO {catalog}.orders VALUES (1, 1, 100)")
        con.execute(f"INSERT INTO {catalog}.orders VALUES (2, 2, 200)")
        con.execute(f"INSERT INTO {catalog}.stores VALUES (1, 'east')")
        con.execute(f"INSERT INTO {catalog}.stores VALUES (2, 'west')")

        view_sql = (
            "SELECT stores.region, SUM(orders.amount) AS total"
            " FROM orders LEFT JOIN stores ON orders.store_id = stores.store_id"
            " GROUP BY stores.region"
        )
        plan = compile_ivm(view_sql, mv_catalog=catalog)
        assert plan.strategy == "full_refresh"

        _initialize_mv(con, plan)

        # Round 1: insert
        con.execute(f"INSERT INTO {catalog}.orders VALUES (3, 1, 50)")
        _maintain_mv(con, plan)

        result = con.execute(plan.query_mv).fetchall()
        maintained = sorted(result, key=_null_safe_sort_key)
        expected = recompute_view(con, view_sql, catalog)
        assert maintained == expected

        # Round 2: delete
        con.execute(
            f"DELETE FROM {catalog}.orders WHERE rowid = ("
            f"SELECT rowid FROM {catalog}.orders WHERE oid = 2 LIMIT 1)"
        )
        _maintain_mv(con, plan)

        result = con.execute(plan.query_mv).fetchall()
        maintained = sorted(result, key=_null_safe_sort_key)
        expected = recompute_view(con, view_sql, catalog)
        assert maintained == expected


# ---------------------------------------------------------------------------
# Bag semantics — duplicate rows handled correctly
# ---------------------------------------------------------------------------


class TestBagSemantics:
    """Verify duplicate rows are handled correctly in the bag-diff."""

    def test_duplicate_rows_preserved(self, ducklake):
        """Full refresh preserves duplicate rows in the MV."""
        items = Table("items", [Column("x", "INTEGER")])
        scenario = Scenario(
            tables=[items],
            initial_data={
                "items": [
                    Row({"x": 1}),
                    Row({"x": 1}),
                    Row({"x": 2}),
                ],
            },
            # DISTINCT + GROUP BY triggers fallback
            view_sql="SELECT DISTINCT x, COUNT(*) AS cnt FROM items GROUP BY x",
            deltas=[
                Delta("items", inserts=[Row({"x": 1})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_bag_diff_with_identical_rows(self, ducklake):
        """When view produces duplicate result rows, bag-diff handles them."""
        orders = Table(
            "orders",
            [Column("oid", "INTEGER"), Column("store_id", "INTEGER"), Column("amount", "INTEGER")],
        )
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])

        # Two stores in same region -> LEFT JOIN + GROUP BY produces two identical-looking
        # aggregate groups that get merged. This tests the ROW_NUMBER matching.
        scenario = Scenario(
            tables=[orders, stores],
            initial_data={
                "orders": [
                    Row({"oid": 1, "store_id": 1, "amount": 100}),
                    Row({"oid": 2, "store_id": 2, "amount": 100}),
                ],
                "stores": [
                    Row({"store_id": 1, "region": "east"}),
                    Row({"store_id": 2, "region": "east"}),
                ],
            },
            view_sql=(
                "SELECT stores.region, SUM(orders.amount) AS total"
                " FROM orders LEFT JOIN stores ON orders.store_id = stores.store_id"
                " GROUP BY stores.region"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"oid": 3, "store_id": 1, "amount": 50})],
                    deletes=[],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


# ---------------------------------------------------------------------------
# Minimal diff — verify only changed rows are touched
# ---------------------------------------------------------------------------


class TestMinimalDiff:
    """Verify that bag-diff produces minimal changes on the MV."""

    def test_single_row_change(self, ducklake):
        """When 1 row changes out of many, MV only gets minimal modifications."""
        con, catalog = ducklake

        orders = Table(
            "orders",
            [
                Column("oid", "INTEGER"),
                Column("store_id", "INTEGER"),
                Column("amount", "INTEGER"),
            ],
        )
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])
        con.execute(orders.ddl(catalog))
        con.execute(stores.ddl(catalog))

        # Insert 10 stores
        for i in range(10):
            con.execute(f"INSERT INTO {catalog}.stores VALUES ({i}, 'region_{i}')")
        # Insert 10 orders
        for i in range(10):
            con.execute(f"INSERT INTO {catalog}.orders VALUES ({i}, {i}, {i * 10})")

        view_sql = (
            "SELECT stores.region, SUM(orders.amount) AS total"
            " FROM orders LEFT JOIN stores"
            " ON orders.store_id = stores.store_id"
            " GROUP BY stores.region"
        )
        plan = compile_ivm(view_sql, mv_catalog=catalog)
        assert plan.strategy == "full_refresh"
        _initialize_mv(con, plan)

        # Get snapshot before change
        snap_before = con.execute(
            f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{catalog}')"
        ).fetchone()[0]

        # Change just 1 row — add a new order for store 0
        con.execute(f"INSERT INTO {catalog}.orders VALUES (99, 0, 990)")
        _maintain_mv(con, plan)

        # Verify correctness
        result = con.execute(plan.query_mv).fetchall()
        maintained = sorted(result, key=_null_safe_sort_key)
        expected = recompute_view(con, view_sql, catalog)
        assert maintained == expected

        # Verify the MV has the right number of rows (still 10 regions)
        count = con.execute(f"SELECT COUNT(*) FROM {catalog}.main.mv").fetchone()[0]
        assert count == 10

        # Check that ducklake_table_changes on the MV shows minimal changes
        mv_snap_after = con.execute(
            f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{catalog}')"
        ).fetchone()[0]

        if mv_snap_after > snap_before:
            changes = con.execute(
                f"SELECT COUNT(*) FROM ducklake_table_changes("
                f"'{catalog}', 'main', 'mv',"
                f" {snap_before + 1}, {mv_snap_after})"
            ).fetchone()[0]
            # Only region_0 changed (DELETE old + INSERT new = 2 changes)
            # not a full 10-row DELETE + 10-row INSERT
            assert changes <= 4, f"Expected minimal changes, got {changes}"


# ---------------------------------------------------------------------------
# Downstream chain — full_refresh MV -> incremental MV
# ---------------------------------------------------------------------------


class TestDownstreamChain:
    """Base table -> full_refresh MV -> incremental MV chain."""

    def test_downstream_incremental_mv(self, ducklake):
        """An incremental MV that reads from a full_refresh MV works correctly."""
        con, catalog = ducklake

        # Create base tables
        orders = Table(
            "orders",
            [Column("oid", "INTEGER"), Column("store_id", "INTEGER"), Column("amount", "INTEGER")],
        )
        stores = Table("stores", [Column("store_id", "INTEGER"), Column("region", "VARCHAR")])
        con.execute(orders.ddl(catalog))
        con.execute(stores.ddl(catalog))

        # Insert initial data
        for row in [
            ("orders", 1, 1, 100),
            ("orders", 2, 2, 200),
        ]:
            con.execute(
                f"INSERT INTO {catalog}.orders (oid, store_id, amount)"
                f" VALUES ({row[1]}, {row[2]}, {row[3]})"
            )
        for row in [
            (1, "east"),
            (2, "west"),
        ]:
            con.execute(
                f"INSERT INTO {catalog}.stores (store_id, region) VALUES ({row[0]}, '{row[1]}')"
            )

        # Compile the full_refresh MV (outer join + agg)
        from duckstream.materialized_view import Naming

        class NamedRegionTotals(Naming):
            def mv_table(self) -> str:
                return "region_totals"

        view_sql_1 = (
            "SELECT stores.region, SUM(orders.amount) AS total"
            " FROM orders LEFT JOIN stores ON orders.store_id = stores.store_id"
            " GROUP BY stores.region"
        )
        plan1 = compile_ivm(view_sql_1, naming=NamedRegionTotals(), mv_catalog=catalog)
        assert plan1.strategy == "full_refresh"

        # Initialize the full_refresh MV
        _initialize_mv(con, plan1)

        # Now compile an incremental MV that reads from region_totals
        class NamedHighRegions(Naming):
            def mv_table(self) -> str:
                return "high_regions"

        view_sql_2 = "SELECT region, total FROM region_totals WHERE total > 150"
        plan2 = compile_ivm(view_sql_2, naming=NamedHighRegions(), mv_catalog=catalog)
        assert plan2.strategy == "incremental"

        _initialize_mv(con, plan2)

        # Verify initial state
        result = con.execute(plan2.query_mv).fetchall()
        maintained = sorted(result, key=_null_safe_sort_key)
        assert maintained == [("west", 200)]

        # Add an order that increases east total above 150
        con.execute(f"INSERT INTO {catalog}.orders (oid, store_id, amount) VALUES (3, 1, 100)")

        # Maintain in order: full_refresh first, then incremental
        _maintain_mv(con, plan1)
        _maintain_mv(con, plan2)

        # Verify downstream MV is correct
        result = con.execute(plan2.query_mv).fetchall()
        maintained = sorted(result, key=_null_safe_sort_key)

        # Recompute expected: east=200, west=200, both > 150
        expected = sorted([("east", 200), ("west", 200)], key=_null_safe_sort_key)
        assert maintained == expected
