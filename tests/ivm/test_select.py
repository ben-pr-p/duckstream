"""Tests for SELECT / PROJECT / WHERE on a single table."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table, single_table_select


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


class TestSmokeSelect:
    """Hand-written scenarios for SELECT / WHERE."""

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
