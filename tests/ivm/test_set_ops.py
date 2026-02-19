"""Tests for set operations (UNION / EXCEPT / INTERSECT)."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table, set_operation


@given(scenario=set_operation())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_set_operation(scenario):
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


class TestSmokeSetOps:
    """Smoke tests for set operations."""

    def test_union_all_basic(self, ducklake):
        """UNION ALL: rows from both tables combined."""
        t1 = Table("t1", [Column("x", "INTEGER")])
        t2 = Table("t2", [Column("x", "INTEGER")])
        scenario = Scenario(
            tables=[t1, t2],
            initial_data={
                "t1": [Row({"x": 1}), Row({"x": 2})],
                "t2": [Row({"x": 2}), Row({"x": 3})],
            },
            view_sql="SELECT x FROM t1 UNION ALL SELECT x FROM t2",
            deltas=[
                Delta("t1", inserts=[Row({"x": 4})], deletes=[Row({"x": 1})]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_union_all_both_sides(self, ducklake):
        """UNION ALL: deltas on both tables simultaneously."""
        t1 = Table("t1", [Column("x", "INTEGER")])
        t2 = Table("t2", [Column("x", "INTEGER")])
        scenario = Scenario(
            tables=[t1, t2],
            initial_data={
                "t1": [Row({"x": 1})],
                "t2": [Row({"x": 2})],
            },
            view_sql="SELECT x FROM t1 UNION ALL SELECT x FROM t2",
            deltas=[
                Delta("t1", inserts=[Row({"x": 10})], deletes=[]),
                Delta("t2", inserts=[Row({"x": 20})], deletes=[Row({"x": 2})]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_union_distinct_dedup(self, ducklake):
        """UNION (DISTINCT): duplicates across branches are deduplicated."""
        t1 = Table("t1", [Column("x", "INTEGER")])
        t2 = Table("t2", [Column("x", "INTEGER")])
        scenario = Scenario(
            tables=[t1, t2],
            initial_data={
                "t1": [Row({"x": 1}), Row({"x": 2})],
                "t2": [Row({"x": 2}), Row({"x": 3})],
            },
            view_sql="SELECT x FROM t1 UNION SELECT x FROM t2",
            deltas=[
                Delta("t1", inserts=[Row({"x": 3})], deletes=[Row({"x": 1})]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_except_all_basic(self, ducklake):
        """EXCEPT ALL: bag subtraction."""
        t1 = Table("t1", [Column("x", "INTEGER")])
        t2 = Table("t2", [Column("x", "INTEGER")])
        scenario = Scenario(
            tables=[t1, t2],
            initial_data={
                "t1": [Row({"x": 1}), Row({"x": 2}), Row({"x": 2}), Row({"x": 3})],
                "t2": [Row({"x": 2})],
            },
            view_sql="SELECT x FROM t1 EXCEPT ALL SELECT x FROM t2",
            deltas=[
                Delta("t2", inserts=[Row({"x": 1})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_intersect_all_basic(self, ducklake):
        """INTERSECT ALL: bag intersection."""
        t1 = Table("t1", [Column("x", "INTEGER")])
        t2 = Table("t2", [Column("x", "INTEGER")])
        scenario = Scenario(
            tables=[t1, t2],
            initial_data={
                "t1": [Row({"x": 1}), Row({"x": 2}), Row({"x": 2})],
                "t2": [Row({"x": 2}), Row({"x": 2}), Row({"x": 3})],
            },
            view_sql="SELECT x FROM t1 INTERSECT ALL SELECT x FROM t2",
            deltas=[
                Delta("t1", inserts=[Row({"x": 3})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)
