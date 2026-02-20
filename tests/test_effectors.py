"""Tests for the effector plugin system."""

from __future__ import annotations

import duckdb
from pydantic import BaseModel

from duckstream import Effector, EffectorResult, Orchestrator
from duckstream.effectors import Effector as EffectorFromSubpkg


class DummyOptions(BaseModel):
    prefix: str = "processed"


class PassthroughEffector(Effector[DummyOptions]):
    """An effector that copies source data into the output table with a prefix."""

    Options = DummyOptions
    columns: list[tuple[str, str]] = [
        ("source_id", "INTEGER"),
        ("result", "VARCHAR"),
    ]

    def handle_insert(self, row: dict) -> dict | None:
        return {
            "source_id": row["id"],
            "result": f"{self.options.prefix}:inserted:{row['id']}",
        }

    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        return {
            "source_id": new_row["id"],
            "result": f"{self.options.prefix}:updated:{old_row['id']}->{new_row['id']}",
        }

    def handle_delete(self, row: dict) -> dict | None:
        return {
            "source_id": row["id"],
            "result": f"{self.options.prefix}:deleted:{row['id']}",
        }


class SkippingEffector(Effector[DummyOptions]):
    """An effector that skips rows with even IDs."""

    Options = DummyOptions
    columns: list[tuple[str, str]] = [
        ("source_id", "INTEGER"),
        ("result", "VARCHAR"),
    ]

    def handle_insert(self, row: dict) -> dict | None:
        if row["id"] % 2 == 0:
            return None
        return {"source_id": row["id"], "result": "odd"}

    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        return None

    def handle_delete(self, row: dict) -> dict | None:
        return None


class ErrorRaisingEffector(Effector[DummyOptions]):
    """An effector that raises on every insert."""

    Options = DummyOptions
    columns: list[tuple[str, str]] = [("source_id", "INTEGER"), ("result", "VARCHAR")]
    on_error = "raise"

    def handle_insert(self, row: dict) -> dict | None:
        raise ValueError(f"boom on {row['id']}")

    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        return None

    def handle_delete(self, row: dict) -> dict | None:
        return None


class ErrorSkippingEffector(Effector[DummyOptions]):
    """An effector that raises but has on_error='skip'."""

    Options = DummyOptions
    columns: list[tuple[str, str]] = [("source_id", "INTEGER"), ("result", "VARCHAR")]
    on_error = "skip"

    def handle_insert(self, row: dict) -> dict | None:
        raise ValueError(f"boom on {row['id']}")

    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        return None

    def handle_delete(self, row: dict) -> dict | None:
        return None


class ErrorStoringEffector(Effector[DummyOptions]):
    """An effector that raises but has on_error='store'."""

    Options = DummyOptions
    columns: list[tuple[str, str]] = [
        ("source_id", "INTEGER"),
        ("result", "VARCHAR"),
        ("error", "VARCHAR"),
    ]
    on_error = "store"

    def handle_insert(self, row: dict) -> dict | None:
        raise ValueError(f"boom on {row['id']}")

    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        return None

    def handle_delete(self, row: dict) -> dict | None:
        return None


class SetupTrackingEffector(Effector[DummyOptions]):
    """An effector that tracks whether setup was called."""

    Options = DummyOptions
    columns: list[tuple[str, str]] = [("source_id", "INTEGER"), ("result", "VARCHAR")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_called = False

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._setup_called = True

    def handle_insert(self, row: dict) -> dict | None:
        return {"source_id": row["id"], "result": "ok"}

    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        return None

    def handle_delete(self, row: dict) -> dict | None:
        return None


class TestEffectorABC:
    """Test the Effector base class and related dataclasses."""

    def test_fqn(self):
        e = PassthroughEffector(
            catalog="cat", table="tbl", output_table="out", options=DummyOptions()
        )
        assert e.fqn == "cat.main.tbl"

    def test_output_fqn(self):
        e = PassthroughEffector(
            catalog="cat", table="tbl", output_table="out", options=DummyOptions()
        )
        assert e.output_fqn == "cat.main.out"

    def test_effector_name(self):
        e = PassthroughEffector(
            catalog="cat", table="tbl", output_table="out", options=DummyOptions()
        )
        assert e.effector_name == "PassthroughEffector_tbl"

    def test_effector_result_defaults(self):
        r = EffectorResult()
        assert r.rows_inserted == 0
        assert r.rows_skipped == 0
        assert r.rows_errored == 0

    def test_import_from_subpackage(self):
        assert EffectorFromSubpkg is Effector


class TestEffectorIntegration:
    """Test effector integration with Orchestrator."""

    def test_basic_flush(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = PassthroughEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_processed",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        # Insert data and maintain
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        # Flush effectors
        results = orch.flush_effectors()
        assert len(results) == 1
        assert results[0].rows_inserted == 1
        assert results[0].rows_skipped == 0

        # Verify output table contents
        rows = conn.execute(
            f"SELECT source_id, result FROM {catalog}.main.orders_processed"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] == "processed:inserted:1"

    def test_setup_called(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = SetupTrackingEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_out",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        assert effector._setup_called

    def test_flush_skips_when_no_changes(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = PassthroughEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_processed",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        # Insert, maintain, flush once
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        orch.flush_effectors()

        # Flush again repeatedly until cursor catches up
        for _ in range(3):
            orch.flush_effectors()

        # Output table should still only have 1 row
        rows = conn.execute(f"SELECT * FROM {catalog}.main.orders_processed").fetchall()
        assert len(rows) == 1

    def test_multiple_inserts(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = PassthroughEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_processed",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        orch.maintain()

        results = orch.flush_effectors()
        assert results[0].rows_inserted == 2

        rows = conn.execute(
            f"SELECT source_id, result FROM {catalog}.main.orders_processed ORDER BY source_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] == "processed:inserted:1"
        assert rows[1][1] == "processed:inserted:2"

    def test_skip_returns(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = SkippingEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_filtered",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (3, 300)")
        orch.maintain()

        results = orch.flush_effectors()
        assert results[0].rows_inserted == 2  # ids 1 and 3
        assert results[0].rows_skipped == 1  # id 2

        rows = conn.execute(
            f"SELECT source_id FROM {catalog}.main.orders_filtered ORDER BY source_id"
        ).fetchall()
        assert [r[0] for r in rows] == [1, 3]


class TestEffectorOnError:
    """Test on_error behavior."""

    def test_on_error_raise(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = ErrorRaisingEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_out",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        import pytest

        with pytest.raises(ValueError, match="boom on 1"):
            orch.flush_effectors()

    def test_on_error_skip(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = ErrorSkippingEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_out",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        results = orch.flush_effectors()
        assert results[0].rows_errored == 1
        assert results[0].rows_inserted == 0

        # Output table should be empty
        rows = conn.execute(f"SELECT * FROM {catalog}.main.orders_out").fetchall()
        assert len(rows) == 0

    def test_on_error_store(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = ErrorStoringEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_out",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        results = orch.flush_effectors()
        assert results[0].rows_errored == 1

        # Output table should have an error row
        rows = conn.execute(
            f"SELECT source_id, result, error FROM {catalog}.main.orders_out"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] is None  # source_id is NULL
        assert rows[0][1] is None  # result is NULL
        assert "boom on 1" in rows[0][2]


class TestEffectorCursorAdvances:
    """Test that effector cursors advance correctly."""

    def test_cursor_advances(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = PassthroughEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_processed",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        initial_snap = orch._sink_cursors[effector.effector_name]

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        orch.flush_effectors()

        assert orch._sink_cursors[effector.effector_name] > initial_snap

    def test_second_flush_sees_only_new_changes(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        effector = PassthroughEffector(
            catalog=catalog,
            table="orders_mv",
            output_table="orders_processed",
            options=DummyOptions(),
        )
        orch.add_effector(effector)
        orch.initialize()

        # First batch
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        orch.flush_effectors()

        # Second batch
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        orch.maintain()
        orch.flush_effectors()

        # Should have processed only the new row
        # (may also pick up cursor-update snapshots, but output should reflect only real data)
        rows = conn.execute(
            f"SELECT source_id FROM {catalog}.main.orders_processed ORDER BY source_id"
        ).fetchall()
        assert [r[0] for r in rows] == [1, 2]
