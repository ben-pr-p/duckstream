"""Tests for GROUP BY with HAVING clause."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table
from tests.strategies.having import single_table_having


@given(scenario=single_table_having())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_having_property(scenario):
    """GROUP BY with HAVING on a single table maintains correctly."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


class TestSmokeHaving:
    """Hand-written scenarios for HAVING clause."""

    def test_having_sum_gt(self, ducklake):
        """HAVING SUM(val) > threshold: only groups above threshold appear."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 50}),
                    Row({"grp": "a", "val": 60}),
                    Row({"grp": "b", "val": 10}),
                    Row({"grp": "b", "val": 20}),
                ]
            },
            view_sql="SELECT grp, SUM(val) AS total FROM t GROUP BY grp HAVING SUM(val) > 50",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "b", "val": 30})],
                    deletes=[],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_having_count_gt(self, ducklake):
        """HAVING COUNT(*) > threshold: filters by group size."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 1}),
                    Row({"grp": "a", "val": 2}),
                    Row({"grp": "a", "val": 3}),
                    Row({"grp": "b", "val": 10}),
                ]
            },
            view_sql="SELECT grp, SUM(val) AS total FROM t GROUP BY grp HAVING COUNT(*) > 1",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "b", "val": 20})],
                    deletes=[],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_having_group_appears(self, ducklake):
        """A group that initially doesn't satisfy HAVING appears after inserts."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 100}),
                    Row({"grp": "b", "val": 5}),
                ]
            },
            view_sql="SELECT grp, SUM(val) AS total FROM t GROUP BY grp HAVING SUM(val) > 50",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "b", "val": 100})],
                    deletes=[],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_having_group_disappears(self, ducklake):
        """A group that initially satisfies HAVING disappears after deletes."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 100}),
                    Row({"grp": "a", "val": 50}),
                    Row({"grp": "b", "val": 200}),
                ]
            },
            view_sql="SELECT grp, SUM(val) AS total FROM t GROUP BY grp HAVING SUM(val) > 100",
            deltas=[
                Delta(
                    "t",
                    inserts=[],
                    deletes=[Row({"grp": "a", "val": 100})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_having_all_groups_filtered(self, ducklake):
        """All groups are below HAVING threshold -> empty MV."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 1}),
                    Row({"grp": "b", "val": 2}),
                ]
            },
            view_sql="SELECT grp, SUM(val) AS total FROM t GROUP BY grp HAVING SUM(val) > 100",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "a", "val": 3})],
                    deletes=[],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_having_with_count_star_in_select(self, ducklake):
        """HAVING COUNT(*) > N when COUNT(*) is also in SELECT."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 1}),
                    Row({"grp": "a", "val": 2}),
                    Row({"grp": "a", "val": 3}),
                    Row({"grp": "b", "val": 10}),
                ]
            },
            view_sql=("SELECT grp, COUNT(*) AS cnt FROM t GROUP BY grp HAVING COUNT(*) > 2"),
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "b", "val": 20}), Row({"grp": "b", "val": 30})],
                    deletes=[Row({"grp": "a", "val": 1})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_having_compound_condition(self, ducklake):
        """HAVING with AND combining two conditions."""
        scenario = Scenario(
            tables=[Table("t", [Column("grp", "VARCHAR"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"grp": "a", "val": 10}),
                    Row({"grp": "a", "val": 20}),
                    Row({"grp": "a", "val": 30}),
                    Row({"grp": "b", "val": 100}),
                    Row({"grp": "b", "val": 200}),
                    Row({"grp": "c", "val": 1}),
                ]
            },
            view_sql=(
                "SELECT grp, SUM(val) AS total FROM t GROUP BY grp "
                "HAVING SUM(val) > 50 AND COUNT(*) > 1"
            ),
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"grp": "c", "val": 100}), Row({"grp": "c", "val": 200})],
                    deletes=[Row({"grp": "a", "val": 30})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)
