"""Tests for SELECT DISTINCT."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table, single_table_distinct


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


class TestSmokeDistinct:
    """Hand-written scenarios for DISTINCT."""

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
