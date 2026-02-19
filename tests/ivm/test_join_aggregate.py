"""Tests for JOIN + GROUP BY aggregate."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table, join_then_aggregate


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


class TestSmokeJoinAggregate:
    """Hand-written scenarios for JOIN + aggregate."""

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
