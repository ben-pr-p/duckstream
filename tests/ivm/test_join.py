"""Tests for inner JOIN (two-table and three-table)."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import (
    Column,
    Delta,
    Row,
    Scenario,
    Table,
    three_table_join,
    two_table_join,
)


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


@given(scenario=three_table_join())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_three_table_join(scenario):
    """Three-table inner JOIN chain maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


class TestSmokeJoin:
    """Hand-written scenarios for inner JOIN."""

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
                "orders": [Row({"oid": 1, "cid": 1})],
                "customers": [Row({"cid": 1, "name": "alice"})],
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
                Delta("orders", inserts=[], deletes=[Row({"oid": 1, "cid": 1})]),
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
                "customers": [Row({"cid": 1, "name": "alice"})],
            },
            view_sql=(
                "SELECT orders.oid, orders.cid, customers.name"
                " FROM orders JOIN customers ON orders.cid = customers.cid"
            ),
            deltas=[
                Delta("orders", inserts=[], deletes=[Row({"oid": 2, "cid": 99})]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_three_table_join_insert_first(self, ducklake):
        """3-table join: insert into first table."""
        r = Table("r", [Column("k1", "INTEGER"), Column("rv", "INTEGER")])
        s = Table("s", [Column("k1", "INTEGER"), Column("k2", "INTEGER")])
        t = Table("t", [Column("k2", "INTEGER"), Column("tv", "VARCHAR")])
        scenario = Scenario(
            tables=[r, s, t],
            initial_data={
                "r": [Row({"k1": 1, "rv": 10}), Row({"k1": 2, "rv": 20})],
                "s": [Row({"k1": 1, "k2": 100}), Row({"k1": 2, "k2": 200})],
                "t": [Row({"k2": 100, "tv": "a"}), Row({"k2": 200, "tv": "b"})],
            },
            view_sql=("SELECT r.k1, r.rv, t.tv FROM r JOIN s ON r.k1 = s.k1 JOIN t ON s.k2 = t.k2"),
            deltas=[Delta("r", inserts=[Row({"k1": 1, "rv": 30})], deletes=[])],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_three_table_join_insert_middle(self, ducklake):
        """3-table join: insert into middle table creates new join paths."""
        r = Table("r", [Column("k1", "INTEGER"), Column("rv", "INTEGER")])
        s = Table("s", [Column("k1", "INTEGER"), Column("k2", "INTEGER")])
        t = Table("t", [Column("k2", "INTEGER"), Column("tv", "VARCHAR")])
        scenario = Scenario(
            tables=[r, s, t],
            initial_data={
                "r": [Row({"k1": 1, "rv": 10})],
                "s": [Row({"k1": 1, "k2": 100})],
                "t": [Row({"k2": 100, "tv": "a"}), Row({"k2": 200, "tv": "b"})],
            },
            view_sql=("SELECT r.k1, r.rv, t.tv FROM r JOIN s ON r.k1 = s.k1 JOIN t ON s.k2 = t.k2"),
            deltas=[Delta("s", inserts=[Row({"k1": 1, "k2": 200})], deletes=[])],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_three_table_join_delete_middle(self, ducklake):
        """3-table join: delete from middle table breaks join paths."""
        r = Table("r", [Column("k1", "INTEGER"), Column("rv", "INTEGER")])
        s = Table("s", [Column("k1", "INTEGER"), Column("k2", "INTEGER")])
        t = Table("t", [Column("k2", "INTEGER"), Column("tv", "VARCHAR")])
        scenario = Scenario(
            tables=[r, s, t],
            initial_data={
                "r": [Row({"k1": 1, "rv": 10})],
                "s": [Row({"k1": 1, "k2": 100}), Row({"k1": 1, "k2": 200})],
                "t": [Row({"k2": 100, "tv": "a"}), Row({"k2": 200, "tv": "b"})],
            },
            view_sql=("SELECT r.k1, r.rv, t.tv FROM r JOIN s ON r.k1 = s.k1 JOIN t ON s.k2 = t.k2"),
            deltas=[Delta("s", inserts=[], deletes=[Row({"k1": 1, "k2": 100})])],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_three_table_join_insert_all(self, ducklake):
        """3-table join: insert into all three tables simultaneously."""
        r = Table("r", [Column("k1", "INTEGER"), Column("rv", "INTEGER")])
        s = Table("s", [Column("k1", "INTEGER"), Column("k2", "INTEGER")])
        t = Table("t", [Column("k2", "INTEGER"), Column("tv", "VARCHAR")])
        scenario = Scenario(
            tables=[r, s, t],
            initial_data={
                "r": [Row({"k1": 1, "rv": 10})],
                "s": [Row({"k1": 1, "k2": 100})],
                "t": [Row({"k2": 100, "tv": "a"})],
            },
            view_sql=("SELECT r.k1, r.rv, t.tv FROM r JOIN s ON r.k1 = s.k1 JOIN t ON s.k2 = t.k2"),
            deltas=[
                Delta("r", inserts=[Row({"k1": 2, "rv": 20})], deletes=[]),
                Delta("s", inserts=[Row({"k1": 2, "k2": 200})], deletes=[]),
                Delta("t", inserts=[Row({"k2": 200, "tv": "b"})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)
