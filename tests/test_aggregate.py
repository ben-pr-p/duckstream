"""Tests for GROUP BY with aggregates (SUM / COUNT / AVG / MIN / MAX)."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table, single_table_aggregate


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


class TestSmokeAggregate:
    """Hand-written scenarios for aggregates."""

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
            deltas=[Delta("t", inserts=[], deletes=[Row({"grp": "a", "val": 1})])],
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
            deltas=[Delta("t", inserts=[Row({"grp": "a", "val": 2})], deletes=[])],
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
            deltas=[Delta("t", inserts=[], deletes=[Row({"grp": "a", "val": 10})])],
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
            deltas=[Delta("t", inserts=[Row({"grp": "a", "val": 20})], deletes=[])],
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
            deltas=[Delta("t", inserts=[], deletes=[Row({"grp": "b", "val": 10})])],
        )
        assert_ivm_correct(scenario, ducklake)
