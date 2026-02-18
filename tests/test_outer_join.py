"""Tests for outer JOIN (LEFT / RIGHT / FULL)."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table, two_table_outer_join


@given(scenario=two_table_outer_join())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_two_table_outer_join(scenario):
    """LEFT/RIGHT/FULL OUTER JOIN maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


class TestSmokeOuterJoin:
    """Hand-written scenarios for outer JOIN."""

    def test_left_join_insert_preserved(self, ducklake):
        """LEFT JOIN: insert into preserved (left) side with no match -> NULL-padded row."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [Row({"oid": 1, "cid": 1})],
                "customers": [Row({"cid": 1, "name": "alice"})],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders LEFT JOIN customers"
                " ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("orders", inserts=[Row({"oid": 2, "cid": 99})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_left_join_insert_nullable_creates_match(self, ducklake):
        """LEFT JOIN: insert into nullable (right) side creates match, removes NULL-padded."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [
                    Row({"oid": 1, "cid": 1}),
                    Row({"oid": 2, "cid": 2}),
                ],
                "customers": [Row({"cid": 1, "name": "alice"})],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders LEFT JOIN customers"
                " ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("customers", inserts=[Row({"cid": 2, "name": "bob"})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_left_join_delete_nullable_restores_null(self, ducklake):
        """LEFT JOIN: delete from right side -> NULL-padded row reappears."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [Row({"oid": 1, "cid": 1})],
                "customers": [Row({"cid": 1, "name": "alice"})],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders LEFT JOIN customers"
                " ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("customers", inserts=[], deletes=[Row({"cid": 1, "name": "alice"})]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_right_join_basic(self, ducklake):
        """RIGHT JOIN: insert into left -> new matches. Unmatched right -> NULL-padded."""
        left = Table("orders", [Column("oid", "INTEGER"), Column("cid", "INTEGER")])
        right = Table("customers", [Column("cid", "INTEGER"), Column("name", "VARCHAR")])
        scenario = Scenario(
            tables=[left, right],
            initial_data={
                "orders": [Row({"oid": 1, "cid": 1})],
                "customers": [
                    Row({"cid": 1, "name": "alice"}),
                    Row({"cid": 2, "name": "bob"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders RIGHT JOIN customers"
                " ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("orders", inserts=[Row({"oid": 2, "cid": 2})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_full_outer_join_basic(self, ducklake):
        """FULL OUTER JOIN: both unmatched sides produce NULL-padded rows."""
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
                    Row({"cid": 88, "name": "bob"}),
                ],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders FULL OUTER JOIN customers"
                " ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("orders", inserts=[Row({"oid": 3, "cid": 88})], deletes=[]),
                Delta("customers", inserts=[Row({"cid": 99, "name": "carol"})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)
