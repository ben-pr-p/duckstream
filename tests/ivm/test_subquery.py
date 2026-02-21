"""Tests for subquery rewriting and IVM maintenance."""

from hypothesis import HealthCheck, given, settings

from tests.conftest import assert_ivm_correct, make_ducklake
from tests.strategies import Column, Delta, Row, Scenario, Table
from tests.strategies.subquery import not_in_with_null_outer


class TestFromSubquery:
    """Subqueries in FROM clause (derived tables)."""

    def test_simple_derived_table(self, ducklake):
        """SELECT ... FROM (SELECT ... FROM t) alias."""
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"id": 1, "val": 10}),
                    Row({"id": 2, "val": 20}),
                    Row({"id": 3, "val": 30}),
                ]
            },
            view_sql="SELECT sub.id, sub.val FROM (SELECT id, val FROM t) sub",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"id": 4, "val": 40})],
                    deletes=[Row({"id": 2, "val": 20})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_derived_table_with_where(self, ducklake):
        """SELECT ... FROM (SELECT ... FROM t WHERE ...) alias WHERE ..."""
        scenario = Scenario(
            tables=[Table("t", [Column("id", "INTEGER"), Column("val", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"id": 1, "val": 10}),
                    Row({"id": 2, "val": 20}),
                    Row({"id": 3, "val": 30}),
                    Row({"id": 4, "val": 5}),
                ]
            },
            view_sql=(
                "SELECT sub.id, sub.val "
                "FROM (SELECT id, val FROM t WHERE val > 5) sub "
                "WHERE sub.id > 1"
            ),
            deltas=[
                Delta(
                    "t",
                    inserts=[
                        Row({"id": 5, "val": 50}),
                        Row({"id": 6, "val": 3}),  # filtered by inner WHERE
                    ],
                    deletes=[Row({"id": 3, "val": 30})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_derived_table_with_column_rename(self, ducklake):
        """SELECT ... FROM (SELECT a AS x, b AS y FROM t) alias."""
        scenario = Scenario(
            tables=[Table("t", [Column("a", "INTEGER"), Column("b", "INTEGER")])],
            initial_data={
                "t": [
                    Row({"a": 1, "b": 10}),
                    Row({"a": 2, "b": 20}),
                ]
            },
            view_sql="SELECT sub.x, sub.y FROM (SELECT a AS x, b AS y FROM t) sub",
            deltas=[
                Delta(
                    "t",
                    inserts=[Row({"a": 3, "b": 30})],
                    deletes=[Row({"a": 1, "b": 10})],
                )
            ],
        )
        assert_ivm_correct(scenario, ducklake)


class TestInSubquery:
    """IN (subquery) rewriting to semi-join."""

    def test_simple_in_subquery(self, ducklake):
        """SELECT ... FROM t1 WHERE col IN (SELECT col FROM t2)."""
        scenario = Scenario(
            tables=[
                Table("orders", [Column("id", "INTEGER"), Column("cid", "INTEGER")]),
                Table("vips", [Column("cid", "INTEGER")]),
            ],
            initial_data={
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": 2}),
                    Row({"id": 3, "cid": 3}),
                ],
                "vips": [
                    Row({"cid": 1}),
                    Row({"cid": 3}),
                ],
            },
            view_sql=(
                "SELECT orders.id, orders.cid FROM orders "
                "WHERE orders.cid IN (SELECT cid FROM vips)"
            ),
            deltas=[
                Delta("vips", inserts=[Row({"cid": 2})], deletes=[]),
                Delta("orders", inserts=[Row({"id": 4, "cid": 4})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_in_subquery_with_filter(self, ducklake):
        """IN subquery where the inner query has a WHERE clause."""
        scenario = Scenario(
            tables=[
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
                Table(
                    "customers",
                    [Column("cid", "INTEGER"), Column("active", "BOOLEAN")],
                ),
            ],
            initial_data={
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 100}),
                    Row({"id": 2, "cid": 2, "amount": 200}),
                    Row({"id": 3, "cid": 3, "amount": 300}),
                ],
                "customers": [
                    Row({"cid": 1, "active": True}),
                    Row({"cid": 2, "active": False}),
                    Row({"cid": 3, "active": True}),
                ],
            },
            view_sql=(
                "SELECT orders.id, orders.amount "
                "FROM orders "
                "WHERE orders.cid IN (SELECT cid FROM customers WHERE active = true)"
            ),
            deltas=[
                Delta(
                    "customers",
                    inserts=[Row({"cid": 2, "active": True})],
                    deletes=[Row({"cid": 2, "active": False})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


class TestNotInSubquery:
    """NOT IN (subquery) rewriting to anti-join."""

    def test_simple_not_in(self, ducklake):
        """SELECT ... FROM t1 WHERE col NOT IN (SELECT col FROM t2)."""
        scenario = Scenario(
            tables=[
                Table("orders", [Column("id", "INTEGER"), Column("cid", "INTEGER")]),
                Table("blocked", [Column("cid", "INTEGER")]),
            ],
            initial_data={
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": 2}),
                    Row({"id": 3, "cid": 3}),
                ],
                "blocked": [
                    Row({"cid": 2}),
                ],
            },
            view_sql=(
                "SELECT orders.id, orders.cid "
                "FROM orders "
                "WHERE orders.cid NOT IN (SELECT cid FROM blocked)"
            ),
            deltas=[
                Delta("blocked", inserts=[Row({"cid": 3})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_not_in_with_null_outer(self, ducklake):
        """NOT IN should exclude NULLs from the outer column."""
        scenario = Scenario(
            tables=[
                Table("orders", [Column("id", "INTEGER"), Column("cid", "INTEGER")]),
                Table("blocked", [Column("cid", "INTEGER")]),
            ],
            initial_data={
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": None}),
                    Row({"id": 3, "cid": 2}),
                ],
                "blocked": [Row({"cid": 2})],
            },
            view_sql=(
                "SELECT orders.id, orders.cid "
                "FROM orders "
                "WHERE orders.cid NOT IN (SELECT cid FROM blocked)"
            ),
            deltas=[
                Delta("blocked", inserts=[Row({"cid": 1})], deletes=[]),
                Delta("orders", inserts=[Row({"id": 4, "cid": None})], deletes=[]),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


@given(scenario=not_in_with_null_outer())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_not_in_with_null_outer_property(scenario):
    """NOT IN should never return rows with NULL outer values."""
    con, catalog, cleanup = make_ducklake()
    try:
        assert_ivm_correct(scenario, (con, catalog))
    finally:
        cleanup()


class TestExistsSubquery:
    """Correlated EXISTS/NOT EXISTS rewriting."""

    def test_correlated_exists(self, ducklake):
        """SELECT ... FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.col = t1.col)."""
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [Column("id", "INTEGER"), Column("cid", "INTEGER")],
                ),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                    Row({"id": 3, "name": "carol"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": 1}),
                    Row({"id": 3, "cid": 3}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name "
                "FROM customers "
                "WHERE EXISTS (SELECT 1 FROM orders WHERE orders.cid = customers.id)"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"id": 4, "cid": 2})],
                    deletes=[],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_correlated_not_exists(self, ducklake):
        """Customers with no orders."""
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [Column("id", "INTEGER"), Column("cid", "INTEGER")],
                ),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                    Row({"id": 3, "name": "carol"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": 3}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name "
                "FROM customers "
                "WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.cid = customers.id)"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"id": 3, "cid": 2})],
                    deletes=[Row({"id": 1, "cid": 1})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


class TestScalarSubquery:
    """Correlated scalar subquery in SELECT via recursive MV."""

    def test_scalar_subquery_max(self, ducklake):
        """SELECT col, (SELECT MAX(...) FROM t2 WHERE t2.fk = t1.id) AS alias FROM t1."""
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                    Row({"id": 3, "name": "carol"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 100}),
                    Row({"id": 2, "cid": 1, "amount": 200}),
                    Row({"id": 3, "cid": 3, "amount": 50}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name, "
                "(SELECT MAX(orders.amount) FROM orders WHERE orders.cid = customers.id) "
                "AS max_order "
                "FROM customers"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"id": 4, "cid": 2, "amount": 300})],
                    deletes=[Row({"id": 2, "cid": 1, "amount": 200})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_scalar_subquery_count(self, ducklake):
        """SELECT col, (SELECT COUNT(*) FROM t2 WHERE t2.fk = t1.id) AS alias FROM t1."""
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [Column("id", "INTEGER"), Column("cid", "INTEGER")],
                ),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": 1}),
                ],
            },
            view_sql=(
                "SELECT customers.id, "
                "(SELECT COUNT(*) FROM orders WHERE orders.cid = customers.id) "
                "AS order_count "
                "FROM customers"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"id": 3, "cid": 2})],
                    deletes=[],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


class TestFromSubqueryWithAgg:
    """FROM subqueries with aggregation via recursive MV."""

    def test_from_subquery_with_group_by(self, ducklake):
        """SELECT ... FROM (SELECT cid, SUM(amount) ... GROUP BY cid) sub."""
        scenario = Scenario(
            tables=[
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
            ],
            initial_data={
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 100}),
                    Row({"id": 2, "cid": 1, "amount": 200}),
                    Row({"id": 3, "cid": 2, "amount": 50}),
                ]
            },
            view_sql=(
                "SELECT sub.cid, sub.total "
                "FROM (SELECT cid, SUM(amount) AS total FROM orders GROUP BY cid) sub"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"id": 4, "cid": 2, "amount": 150})],
                    deletes=[Row({"id": 1, "cid": 1, "amount": 100})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)


class TestComplexCombined:
    """Complex queries combining multiple subquery types, recursive MVs, and joins.

    These are end-to-end integration tests that exercise the full compilation
    pipeline with realistic multi-table, multi-subquery queries.
    """

    def test_scalar_subquery_with_in_filter(self, ducklake):
        """VIP customers with their total order amounts.

        Combines:
        - Scalar subquery in SELECT (-> recursive inner MV with SUM aggregate)
        - IN subquery in WHERE (-> semi-join rewrite)

        Query:
            SELECT customers.id, customers.name,
                   (SELECT SUM(orders.amount) FROM orders
                    WHERE orders.cid = customers.id) AS total_spent
            FROM customers
            WHERE customers.id IN (SELECT cid FROM vip_list)
        """
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [
                        Column("id", "INTEGER"),
                        Column("name", "VARCHAR"),
                        Column("region", "VARCHAR"),
                    ],
                ),
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
                Table("vip_list", [Column("cid", "INTEGER")]),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice", "region": "east"}),
                    Row({"id": 2, "name": "bob", "region": "west"}),
                    Row({"id": 3, "name": "carol", "region": "east"}),
                    Row({"id": 4, "name": "dave", "region": "west"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 100}),
                    Row({"id": 2, "cid": 1, "amount": 250}),
                    Row({"id": 3, "cid": 2, "amount": 75}),
                    Row({"id": 4, "cid": 3, "amount": 300}),
                    Row({"id": 5, "cid": 3, "amount": 50}),
                ],
                "vip_list": [
                    Row({"cid": 1}),
                    Row({"cid": 3}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name, "
                "(SELECT SUM(orders.amount) FROM orders "
                "WHERE orders.cid = customers.id) AS total_spent "
                "FROM customers "
                "WHERE customers.id IN (SELECT cid FROM vip_list)"
            ),
            deltas=[
                # bob becomes a VIP, alice loses a big order, carol gets a new one
                Delta("vip_list", inserts=[Row({"cid": 2})], deletes=[]),
                Delta(
                    "orders",
                    inserts=[
                        Row({"id": 6, "cid": 2, "amount": 500}),
                        Row({"id": 7, "cid": 3, "amount": 200}),
                    ],
                    deletes=[Row({"id": 2, "cid": 1, "amount": 250})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_scalar_subquery_with_exists_filter(self, ducklake):
        """Customers who have orders, with their max order amount.

        Combines:
        - Scalar subquery in SELECT (-> recursive inner MV with MAX aggregate)
        - Correlated EXISTS in WHERE (-> semi-join rewrite)

        Query:
            SELECT customers.id, customers.name,
                   (SELECT MAX(orders.amount) FROM orders
                    WHERE orders.cid = customers.id) AS max_order
            FROM customers
            WHERE EXISTS (
                SELECT 1 FROM orders WHERE orders.cid = customers.id
            )
        """
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                    Row({"id": 3, "name": "carol"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 100}),
                    Row({"id": 2, "cid": 1, "amount": 200}),
                    Row({"id": 3, "cid": 3, "amount": 150}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name, "
                "(SELECT MAX(orders.amount) FROM orders "
                "WHERE orders.cid = customers.id) AS max_order "
                "FROM customers "
                "WHERE EXISTS ("
                "SELECT 1 FROM orders WHERE orders.cid = customers.id"
                ")"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[
                        # bob gets his first order -> appears in results
                        Row({"id": 4, "cid": 2, "amount": 500}),
                        # alice gets a bigger order -> max changes
                        Row({"id": 5, "cid": 1, "amount": 999}),
                    ],
                    deletes=[
                        # carol's only order removed -> disappears from results
                        Row({"id": 3, "cid": 3, "amount": 150}),
                    ],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_from_agg_subquery_with_where_filter(self, ducklake):
        """Top spenders filtered to active customers.

        Combines:
        - FROM subquery with GROUP BY + SUM (-> recursive inner MV)
        - Outer WHERE filter on the aggregated result

        Query:
            SELECT sub.cid, sub.total
            FROM (SELECT cid, SUM(amount) AS total
                  FROM orders GROUP BY cid) sub
            WHERE sub.total > 100
        """
        scenario = Scenario(
            tables=[
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
            ],
            initial_data={
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 50}),
                    Row({"id": 2, "cid": 1, "amount": 60}),  # cid=1 total=110 > 100
                    Row({"id": 3, "cid": 2, "amount": 30}),  # cid=2 total=30 < 100
                    Row({"id": 4, "cid": 3, "amount": 200}),  # cid=3 total=200 > 100
                ]
            },
            view_sql=(
                "SELECT sub.cid, sub.total "
                "FROM (SELECT cid, SUM(amount) AS total "
                "FROM orders GROUP BY cid) sub "
                "WHERE sub.total > 100"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[
                        # cid=2 goes from 30 to 130 -> now passes filter
                        Row({"id": 5, "cid": 2, "amount": 100}),
                    ],
                    deletes=[
                        # cid=1 goes from 110 to 60 -> drops below filter
                        Row({"id": 2, "cid": 1, "amount": 60}),
                    ],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_two_scalar_subqueries(self, ducklake):
        """Multiple scalar subqueries from different tables.

        Combines:
        - Two scalar subqueries in SELECT, each becoming a separate inner MV
        - One computes SUM of orders, other computes COUNT of reviews

        Query:
            SELECT customers.id, customers.name,
                   (SELECT SUM(orders.amount) FROM orders
                    WHERE orders.cid = customers.id) AS total_spent,
                   (SELECT COUNT(*) FROM reviews
                    WHERE reviews.cid = customers.id) AS review_count
            FROM customers
        """
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("amount", "INTEGER"),
                    ],
                ),
                Table(
                    "reviews",
                    [
                        Column("id", "INTEGER"),
                        Column("cid", "INTEGER"),
                        Column("stars", "INTEGER"),
                    ],
                ),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                    Row({"id": 3, "name": "carol"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1, "amount": 100}),
                    Row({"id": 2, "cid": 1, "amount": 200}),
                    Row({"id": 3, "cid": 2, "amount": 50}),
                ],
                "reviews": [
                    Row({"id": 1, "cid": 1, "stars": 5}),
                    Row({"id": 2, "cid": 3, "stars": 3}),
                    Row({"id": 3, "cid": 3, "stars": 4}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name, "
                "(SELECT SUM(orders.amount) FROM orders "
                "WHERE orders.cid = customers.id) AS total_spent, "
                "(SELECT COUNT(*) FROM reviews "
                "WHERE reviews.cid = customers.id) AS review_count "
                "FROM customers"
            ),
            deltas=[
                Delta(
                    "orders",
                    inserts=[Row({"id": 4, "cid": 3, "amount": 400})],
                    deletes=[Row({"id": 1, "cid": 1, "amount": 100})],
                ),
                Delta(
                    "reviews",
                    inserts=[Row({"id": 4, "cid": 2, "stars": 5})],
                    deletes=[Row({"id": 2, "cid": 3, "stars": 3})],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)

    def test_not_in_with_scalar_subquery(self, ducklake):
        """Non-blocked customers with their order counts.

        Combines:
        - Scalar subquery in SELECT (-> recursive inner MV with COUNT)
        - NOT IN subquery in WHERE (-> anti-join rewrite)

        Query:
            SELECT customers.id, customers.name,
                   (SELECT COUNT(*) FROM orders
                    WHERE orders.cid = customers.id) AS num_orders
            FROM customers
            WHERE customers.id NOT IN (SELECT cid FROM blocked)
        """
        scenario = Scenario(
            tables=[
                Table(
                    "customers",
                    [Column("id", "INTEGER"), Column("name", "VARCHAR")],
                ),
                Table(
                    "orders",
                    [Column("id", "INTEGER"), Column("cid", "INTEGER")],
                ),
                Table("blocked", [Column("cid", "INTEGER")]),
            ],
            initial_data={
                "customers": [
                    Row({"id": 1, "name": "alice"}),
                    Row({"id": 2, "name": "bob"}),
                    Row({"id": 3, "name": "carol"}),
                ],
                "orders": [
                    Row({"id": 1, "cid": 1}),
                    Row({"id": 2, "cid": 1}),
                    Row({"id": 3, "cid": 2}),
                ],
                "blocked": [
                    Row({"cid": 3}),
                ],
            },
            view_sql=(
                "SELECT customers.id, customers.name, "
                "(SELECT COUNT(*) FROM orders "
                "WHERE orders.cid = customers.id) AS num_orders "
                "FROM customers "
                "WHERE customers.id NOT IN (SELECT cid FROM blocked)"
            ),
            deltas=[
                # unblock carol, block bob
                Delta(
                    "blocked",
                    inserts=[Row({"cid": 2})],
                    deletes=[Row({"cid": 3})],
                ),
                # add an order for carol
                Delta(
                    "orders",
                    inserts=[Row({"id": 4, "cid": 3})],
                    deletes=[],
                ),
            ],
        )
        assert_ivm_correct(scenario, ducklake)
