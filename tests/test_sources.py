"""Tests for the source plugin system."""

from __future__ import annotations

import duckdb
from pydantic import BaseModel

from duckstream import Orchestrator, SyncResult
from duckstream.sources import Source


class DummyOptions(BaseModel):
    rows: list[dict[str, int | str]]


class DummySource(Source[DummyOptions]):
    """A trivial in-memory source for testing."""

    Options = DummyOptions

    def __init__(self, *, catalog: str, table: str, options: DummyOptions):
        super().__init__(catalog=catalog, table=table, options=options)
        self._setup_called = False

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._setup_called = True

    def sync(self, conn: duckdb.DuckDBPyConnection) -> SyncResult:
        # Create table if not exists
        conn.execute(f"CREATE TABLE IF NOT EXISTS {self.fqn} (id INTEGER, name VARCHAR)")
        # Full replace: delete all, insert all
        existing = conn.execute(f"SELECT COUNT(*) FROM {self.fqn}").fetchone()
        old_count = existing[0] if existing else 0
        if old_count > 0:
            conn.execute(f"DELETE FROM {self.fqn}")
        for row in self.options.rows:
            conn.execute(
                f"INSERT INTO {self.fqn} VALUES (?, ?)",
                [row["id"], row["name"]],
            )
        return SyncResult(
            rows_inserted=len(self.options.rows),
            rows_deleted=old_count,
        )


class TestSourceABC:
    """Test the Source base class and SyncResult."""

    def test_fqn(self):
        source = DummySource(
            catalog="cat",
            table="tbl",
            options=DummyOptions(rows=[]),
        )
        assert source.fqn == "cat.main.tbl"

    def test_sync_result_defaults(self):
        r = SyncResult()
        assert r.rows_inserted == 0
        assert r.rows_deleted == 0
        assert r.rows_updated == 0

    def test_dummy_source_setup_and_sync(self, ducklake):
        conn, catalog = ducklake
        source = DummySource(
            catalog=catalog,
            table="test_tbl",
            options=DummyOptions(rows=[{"id": 1, "name": "alice"}]),
        )
        source.setup(conn)
        assert source._setup_called

        result = source.sync(conn)
        assert result.rows_inserted == 1
        assert result.rows_deleted == 0

        rows = conn.execute(f"SELECT * FROM {source.fqn}").fetchall()
        assert rows == [(1, "alice")]


class TestOrchestratorSourceIntegration:
    """Test source integration with the Orchestrator."""

    def test_add_source_and_sync(self, ducklake):
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None  # already attached

        source = DummySource(
            catalog=catalog,
            table="src_tbl",
            options=DummyOptions(rows=[{"id": 1, "name": "bob"}]),
        )
        orch.add_source(source)
        results = orch.sync_sources()

        assert len(results) == 1
        assert results[0].rows_inserted == 1
        rows = conn.execute(f"SELECT * FROM {catalog}.main.src_tbl").fetchall()
        assert rows == [(1, "bob")]

    def test_initialize_syncs_sources_before_mvs(self, ducklake):
        """Sources are synced during initialize(), so MVs can read from source tables."""
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        source = DummySource(
            catalog=catalog,
            table="people",
            options=DummyOptions(
                rows=[
                    {"id": 1, "name": "alice"},
                    {"id": 2, "name": "bob"},
                ]
            ),
        )
        orch.add_source(source)

        # Register an MV that reads from the source table
        orch.add_ivm(
            catalog,
            "people_names",
            f"SELECT id, name FROM {catalog}.main.people",
        )

        # initialize() should sync the source first, then create the MV
        orch.initialize()

        rows = conn.execute(
            f"SELECT id, name FROM {catalog}.main.people_names ORDER BY id"
        ).fetchall()
        assert rows == [(1, "alice"), (2, "bob")]

    def test_source_sync_then_maintain_picks_up_changes(self, ducklake):
        """After initial sync + MV init, changing source data and maintaining updates the MV."""
        conn, catalog = ducklake
        orch = Orchestrator(conn=conn)
        orch._catalogs[catalog] = lambda c, n: None

        source = DummySource(
            catalog=catalog,
            table="items",
            options=DummyOptions(
                rows=[
                    {"id": 1, "name": "apple"},
                    {"id": 2, "name": "banana"},
                ]
            ),
        )
        orch.add_source(source)
        # Use a simple projection MV (not bare COUNT(*) which has a compiler edge case)
        orch.add_ivm(
            catalog,
            "item_names",
            f"SELECT id, name FROM {catalog}.main.items",
        )
        orch.initialize()

        # Verify initial state
        rows = conn.execute(
            f"SELECT id, name FROM {catalog}.main.item_names ORDER BY id"
        ).fetchall()
        assert rows == [(1, "apple"), (2, "banana")]

        # Now change the source data and re-sync
        source.options = DummyOptions(
            rows=[
                {"id": 1, "name": "apple"},
                {"id": 2, "name": "banana"},
                {"id": 3, "name": "cherry"},
            ]
        )
        orch.sync_sources()

        # Maintain should pick up the change
        orch.maintain()

        rows = conn.execute(
            f"SELECT id, name FROM {catalog}.main.item_names ORDER BY id"
        ).fetchall()
        assert rows == [(1, "apple"), (2, "banana"), (3, "cherry")]
