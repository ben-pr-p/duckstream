"""Tests for the sink plugin system."""

from __future__ import annotations

import duckdb
from pydantic import BaseModel

from duckstream import ChangeSet, FlushResult, Orchestrator
from duckstream.sinks import Sink


class DummyOptions(BaseModel):
    pass


class BatchedDummySink(Sink[DummyOptions]):
    """A batched sink that records all ChangeSets it receives."""

    Options = DummyOptions
    batched = True

    def __init__(self, *, catalog: str, table: str, options: DummyOptions):
        super().__init__(catalog=catalog, table=table, options=options)
        self.flush_calls: list[ChangeSet] = []
        self._setup_called = False

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._setup_called = True

    def flush(self, changes: ChangeSet) -> FlushResult:
        self.flush_calls.append(changes)
        # Actually query the MV to verify it works
        rows = changes.conn.execute(f"SELECT * FROM {changes.fqn}").fetchall()
        return FlushResult(rows_inserted=len(rows))


class PerSnapshotDummySink(Sink[DummyOptions]):
    """A per-snapshot sink that records all ChangeSets it receives."""

    Options = DummyOptions
    batched = False

    def __init__(self, *, catalog: str, table: str, options: DummyOptions):
        super().__init__(catalog=catalog, table=table, options=options)
        self.flush_calls: list[ChangeSet] = []

    def flush(self, changes: ChangeSet) -> FlushResult:
        self.flush_calls.append(changes)
        rows = changes.conn.execute(f"SELECT * FROM {changes.fqn}").fetchall()
        return FlushResult(rows_inserted=len(rows))


class TestSinkABC:
    """Test the Sink base class and related dataclasses."""

    def test_fqn(self):
        sink = BatchedDummySink(catalog="cat", table="tbl", options=DummyOptions())
        assert sink.fqn == "cat.main.tbl"

    def test_sink_name(self):
        sink = BatchedDummySink(catalog="cat", table="tbl", options=DummyOptions())
        assert sink.sink_name == "BatchedDummySink_tbl"

    def test_flush_result_defaults(self):
        r = FlushResult()
        assert r.rows_inserted == 0
        assert r.rows_deleted == 0
        assert r.rows_updated == 0

    def test_changeset_fqn(self):
        cs = ChangeSet(
            catalog="cat",
            schema="main",
            table="tbl",
            conn=duckdb.connect(),
            snapshot_start=1,
            snapshot_end=5,
        )
        assert cs.fqn == "cat.main.tbl"


class TestBatchedSinkIntegration:
    """Test batched sink integration with Orchestrator."""

    def test_basic_flush(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        # Create base table and MV
        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        sink = BatchedDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        assert sink._setup_called

        # Insert data and maintain
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        # Flush — should see the new data
        results = orch.flush_sinks()
        assert len(results) == 1
        assert results[0].rows_inserted == 1
        assert len(sink.flush_calls) == 1

        cs = sink.flush_calls[0]
        assert cs.snapshot_start > 0
        assert cs.snapshot_end >= cs.snapshot_start

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

        sink = BatchedDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        # Insert, maintain, and flush once
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        orch.flush_sinks()

        # Flush repeatedly with no intervening maintain.
        # The cursor UPDATE creates a DuckLake snapshot, so the first extra
        # flush sees that snapshot (with no MV data changes). After enough
        # flushes, the cursor catches up and flush truly skips.
        for _ in range(3):
            orch.flush_sinks()
        old_flush_count = len(sink.flush_calls)
        orch.flush_sinks()
        assert len(sink.flush_calls) == old_flush_count

    def test_accumulated_changes(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        sink = BatchedDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        # Two maintains without flushing
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        orch.maintain()

        # Single flush should get all accumulated changes
        results = orch.flush_sinks()
        assert len(results) == 1
        assert results[0].rows_inserted == 2  # sees both rows
        assert len(sink.flush_calls) == 1  # single flush call

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

        sink = BatchedDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        initial_snap = orch._sink_cursors[sink.sink_name]

        # Insert, maintain, flush
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        orch.flush_sinks()

        # In-memory cursor should have advanced
        assert orch._sink_cursors[sink.sink_name] > initial_snap

        # Persisted cursor should also have advanced (may be 1 behind
        # in-memory due to the persist UPDATE creating a new snapshot)
        cursors_fqn = f"{catalog}.main._sink_cursors"
        row = conn.execute(
            f"SELECT last_snapshot FROM {cursors_fqn} WHERE sink_name = '{sink.sink_name}'"
        ).fetchone()
        assert row is not None
        assert row[0] > initial_snap


class TestPerSnapshotSinkIntegration:
    """Test per-snapshot sink integration with Orchestrator."""

    def test_per_snapshot_mode(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        sink = PerSnapshotDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        # Two maintains creating separate snapshots
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        orch.maintain()

        # Flush — should call flush once per snapshot
        results = orch.flush_sinks()
        assert len(results) >= 2  # at least one per maintain
        # Each call should have start == end
        for cs in sink.flush_calls:
            assert cs.snapshot_start == cs.snapshot_end

    def test_per_snapshot_cursor_advances_incrementally(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        sink = PerSnapshotDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        orch.flush_sinks()

        # Now another maintain + flush should only see new changes
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        orch.maintain()

        old_count = len(sink.flush_calls)
        orch.flush_sinks()
        # Should have at least one new flush call
        assert len(sink.flush_calls) > old_count


class TestSinkCanQueryMV:
    """Test that sinks can actually read MV state and deltas from ChangeSet."""

    def test_sink_reads_mv_state(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        sink = BatchedDummySink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (2, 200)")
        orch.maintain()
        results = orch.flush_sinks()

        # The BatchedDummySink queries SELECT * FROM mv in flush
        assert results[0].rows_inserted == 2

    def test_sink_reads_deltas_via_table_changes(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        conn.execute(f"CREATE TABLE {catalog}.main.orders (id INTEGER, amount INTEGER)")
        orch.add_ivm(
            catalog,
            "orders_mv",
            f"SELECT id, amount FROM {catalog}.main.orders",
        )

        # Use a sink that queries ducklake_table_changes
        class DeltaSink(Sink[DummyOptions]):
            Options = DummyOptions
            batched = True

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.delta_rows: list = []

            def flush(self, changes: ChangeSet) -> FlushResult:
                changes.conn.execute(f"SET VARIABLE _sink_start = {changes.snapshot_start}")
                changes.conn.execute(f"SET VARIABLE _sink_end = {changes.snapshot_end}")
                rows = changes.conn.execute(
                    f"SELECT * FROM ducklake_table_changes("
                    f"'{changes.catalog}', '{changes.schema}', '{changes.table}', "
                    f"getvariable('_sink_start'), getvariable('_sink_end'))"
                ).fetchall()
                self.delta_rows = rows
                return FlushResult(rows_inserted=len(rows))

        sink = DeltaSink(catalog=catalog, table="orders_mv", options=DummyOptions())
        orch.add_sink(sink)
        orch.initialize()

        conn.execute(f"INSERT INTO {catalog}.main.orders VALUES (1, 100)")
        orch.maintain()

        results = orch.flush_sinks()
        assert results[0].rows_inserted > 0
        assert len(sink.delta_rows) > 0
