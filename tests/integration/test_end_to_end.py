"""End-to-end integration test: create tables, compile IVM, apply changes, verify."""

from duckstream import compile_ivm
from tests.conftest import make_ducklake


def test_select_insert_delete_roundtrip():
    """Full lifecycle: create table, init MV, insert+delete, maintain, verify."""
    con, catalog, cleanup = make_ducklake()
    try:
        # Create a base table and seed it
        con.execute(f"CREATE TABLE {catalog}.orders (id INTEGER, amount INTEGER, region TEXT)")
        con.execute(
            f"INSERT INTO {catalog}.orders VALUES"
            f" (1, 100, 'east'), (2, 200, 'west'), (3, 150, 'east')"
        )

        # Compile IVM plan
        view_sql = "SELECT region, SUM(amount) AS total FROM orders GROUP BY region"
        plan = compile_ivm(view_sql, mv_catalog=catalog)

        # Create MV and cursors, then populate MV with initial data
        con.execute(plan.create_cursors_table)
        con.execute(plan.create_mv)
        for stmt in plan.initialize_cursors:
            con.execute(stmt)

        # Verify initial MV state
        rows = con.execute(
            f"SELECT region, total FROM {catalog}.main.mv ORDER BY region"
        ).fetchall()
        assert rows == [("east", 250), ("west", 200)]

        # Apply changes: insert a new west order, delete an east order
        con.execute(f"INSERT INTO {catalog}.orders VALUES (4, 300, 'west')")
        con.execute(f"DELETE FROM {catalog}.orders WHERE id = 1")

        # Run maintenance
        for stmt in plan.maintain:
            con.execute(stmt)

        # Verify maintained MV matches expectation
        rows = con.execute(
            f"SELECT region, total FROM {catalog}.main.mv ORDER BY region"
        ).fetchall()
        assert rows == [("east", 150), ("west", 500)]

        # Verify against full recomputation
        con.execute(f"USE {catalog}")
        expected = con.execute(
            "SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region"
        ).fetchall()
        con.execute("USE memory")
        assert rows == expected
    finally:
        cleanup()


def test_not_in_subquery_with_null():
    """NOT IN subquery should respect NULL semantics."""
    con, catalog, cleanup = make_ducklake()
    try:
        con.execute(f"CREATE TABLE {catalog}.customers (id INTEGER)")
        con.execute(f"CREATE TABLE {catalog}.blocked (cid INTEGER)")
        con.execute(f"INSERT INTO {catalog}.customers VALUES (1), (2), (3)")
        con.execute(f"INSERT INTO {catalog}.blocked VALUES (NULL)")

        view_sql = (
            "SELECT customers.id FROM customers WHERE customers.id NOT IN (SELECT cid FROM blocked)"
        )
        plan = compile_ivm(view_sql, mv_catalog=catalog)

        con.execute(plan.create_cursors_table)
        con.execute(plan.create_mv)
        for stmt in plan.initialize_cursors:
            con.execute(stmt)

        maintained = con.execute(plan.query_mv).fetchall()
        con.execute(f"USE {catalog}")
        expected = con.execute(view_sql).fetchall()
        con.execute("USE memory")

        assert maintained == expected
    finally:
        cleanup()
