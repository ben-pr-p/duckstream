"""Orchestrator: runtime coordination layer for IVM maintenance.

Manages catalogs, compiles and registers MVs, validates dependencies,
and executes maintenance with optional parallelism.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import duckdb

from duckstream.compiler import compile_ivm
from duckstream.effectors.base import Effector, EffectorResult
from duckstream.materialized_view import MaterializedView, Naming
from duckstream.sinks.base import ChangeSet, FlushResult, Sink
from duckstream.sources.base import Source, SyncResult

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class OrchestratorError(Exception): ...


class CatalogNotFoundError(OrchestratorError): ...


class NotDuckLakeError(OrchestratorError): ...


class MissingTableError(OrchestratorError): ...


class MissingColumnError(OrchestratorError): ...


class CyclicDependencyError(OrchestratorError): ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MaintenanceStep:
    """One unit of maintenance work — maintaining a single MV."""

    catalog: str
    schema: str
    mv_name: str
    mv: MaterializedView
    strategy: str = "incremental"
    pending_snapshots: int | None = None
    pending_rows: int | None = None
    skipped: bool = False
    duration_ms: float | None = None


@dataclass
class MaintenancePlan:
    """Topologically sorted maintenance plan."""

    steps: list[MaintenanceStep]
    levels: list[list[MaintenanceStep]]


# ---------------------------------------------------------------------------
# Internal MV registration
# ---------------------------------------------------------------------------


@dataclass
class _RegisteredMV:
    """Internal record for a registered MV."""

    catalog: str
    schema: str
    mv_name: str
    mv: MaterializedView
    naming: Naming
    fqn: str  # catalog.schema.mv_name


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Runtime coordinator for IVM maintenance."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection | None = None,
        naming: Naming | None = None,
    ):
        self._conn = conn or duckdb.connect()
        self._naming = naming or Naming()
        self._catalogs: dict[str, Callable[[duckdb.DuckDBPyConnection, str], None]] = {}
        self._mvs: dict[str, _RegisteredMV] = {}  # keyed by fqn
        self._sources: list[Source] = []
        self._sinks: list[Sink] = []
        self._effectors: list[Effector] = []
        self._sink_cursors: dict[str, int] = {}  # in-memory cache, persisted to DuckLake
        self._executor: ThreadPoolExecutor | None = None
        self._advance_state: _AdvanceState | None = None

        # DuckLake is always required
        self._conn.execute("INSTALL ducklake")
        self._conn.execute("LOAD ducklake")

    # -- Extensions & secrets ------------------------------------------------

    def setup_extensions(
        self,
        fn: Callable[[duckdb.DuckDBPyConnection], None],
    ) -> None:
        """Install/load additional DuckDB extensions.

        Called with the orchestrator's connection so the user can run
        INSTALL/LOAD statements for any extensions needed by their catalogs.
        """
        fn(self._conn)

    def setup_secrets(
        self,
        fn: Callable[[duckdb.DuckDBPyConnection], None],
    ) -> None:
        """Configure DuckDB secrets (e.g. S3 credentials).

        Called with the orchestrator's connection so the user can run
        CREATE SECRET statements.
        """
        fn(self._conn)

    # -- Source management ---------------------------------------------------

    def add_source(self, source: Source) -> None:
        """Register a source plugin."""
        self._sources.append(source)

    def sync_sources(self) -> list[SyncResult]:
        """Sync all registered sources. Returns list of SyncResults."""
        results: list[SyncResult] = []
        for source in self._sources:
            results.append(source.sync(self._conn))
        return results

    # -- Sink management ----------------------------------------------------

    def add_sink(self, sink: Sink) -> None:
        """Register a sink plugin."""
        self._sinks.append(sink)

    def flush_sinks(self) -> list[FlushResult]:
        """Flush all registered sinks. Returns list of FlushResults.

        Uses an in-memory cursor cache for skip detection (avoiding the
        DuckLake self-referential snapshot problem where writing the cursor
        table creates a new snapshot).  Persists cursors to the DuckLake
        ``_sink_cursors`` table after each flush so they survive restarts.
        """
        results: list[FlushResult] = []
        for sink in self._sinks:
            last_snapshot = self._sink_cursors.get(sink.sink_name, 0)

            latest = self._conn.execute(
                f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{sink.catalog}')"
            ).fetchone()
            latest_snapshot = latest[0] if latest and latest[0] is not None else 0

            if latest_snapshot <= last_snapshot:
                results.append(FlushResult())
                continue

            if sink.batched:
                changes = ChangeSet(
                    catalog=sink.catalog,
                    schema=sink.schema_,
                    table=sink.table,
                    conn=self._conn,
                    snapshot_start=last_snapshot + 1,
                    snapshot_end=latest_snapshot,
                )
                results.append(sink.flush(changes))
                self._sink_cursors[sink.sink_name] = latest_snapshot
            else:
                for snap in range(last_snapshot + 1, latest_snapshot + 1):
                    changes = ChangeSet(
                        catalog=sink.catalog,
                        schema=sink.schema_,
                        table=sink.table,
                        conn=self._conn,
                        snapshot_start=snap,
                        snapshot_end=snap,
                    )
                    results.append(sink.flush(changes))
                    self._sink_cursors[sink.sink_name] = snap

            # Persist cursor to DuckLake for restart recovery.
            # The UPDATE creates a new DuckLake snapshot, so we read the new
            # max afterward and update the in-memory cache to include it.
            cursors_fqn = f"{sink.catalog}.{sink.schema_}._sink_cursors"
            self._conn.execute(
                f"UPDATE {cursors_fqn} "
                f"SET last_snapshot = {self._sink_cursors[sink.sink_name]} "
                f"WHERE sink_name = '{sink.sink_name}' AND mv_name = '{sink.table}'"
            )
            post = self._conn.execute(
                f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{sink.catalog}')"
            ).fetchone()
            if post and post[0] is not None:
                self._sink_cursors[sink.sink_name] = post[0]
        return results

    # -- Effector management ------------------------------------------------

    def add_effector(self, effector: Effector) -> None:
        """Register an effector plugin."""
        self._effectors.append(effector)

    def flush_effectors(self) -> list[EffectorResult]:
        """Flush all registered effectors. Returns list of EffectorResults.

        For each effector, queries ducklake_table_changes() on the source MV,
        calls the appropriate handle method per row, and batch INSERTs results
        into the output table.
        """
        results: list[EffectorResult] = []
        for effector in self._effectors:
            last_snapshot = self._sink_cursors.get(effector.effector_name, 0)

            latest = self._conn.execute(
                f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{effector.catalog}')"
            ).fetchone()
            latest_snapshot = latest[0] if latest and latest[0] is not None else 0

            if latest_snapshot <= last_snapshot:
                results.append(EffectorResult())
                continue

            result = self._flush_one_effector(effector, last_snapshot, latest_snapshot)
            results.append(result)

            # Persist cursor (reuses _sink_cursors table)
            cursors_fqn = f"{effector.catalog}.{effector.schema_}._sink_cursors"
            self._conn.execute(
                f"UPDATE {cursors_fqn} "
                f"SET last_snapshot = {self._sink_cursors[effector.effector_name]} "
                f"WHERE sink_name = '{effector.effector_name}' "
                f"AND mv_name = '{effector.table}'"
            )
            post = self._conn.execute(
                f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{effector.catalog}')"
            ).fetchone()
            if post and post[0] is not None:
                self._sink_cursors[effector.effector_name] = post[0]

        return results

    def _flush_one_effector(
        self,
        effector: Effector,
        last_snapshot: int,
        latest_snapshot: int,
    ) -> EffectorResult:
        """Process changes for a single effector."""
        self._conn.execute(f"SET VARIABLE _eff_start = {last_snapshot + 1}")
        self._conn.execute(f"SET VARIABLE _eff_end = {latest_snapshot}")

        rows = self._conn.execute(
            f"SELECT * FROM ducklake_table_changes("
            f"'{effector.catalog}', '{effector.schema_}', '{effector.table}', "
            f"getvariable('_eff_start'), getvariable('_eff_end'))"
        ).fetchall()

        col_names = [
            desc[0]
            for desc in self._conn.execute(
                f"SELECT * FROM ducklake_table_changes("
                f"'{effector.catalog}', '{effector.schema_}', '{effector.table}', "
                f"getvariable('_eff_start'), getvariable('_eff_end')) LIMIT 0"
            ).description
        ]

        result = EffectorResult()
        output_rows: list[dict] = []

        for row in rows:
            row_dict = dict(zip(col_names, row, strict=False))
            change_type = row_dict.pop("change_type", None)
            row_dict.pop("snapshot_id", None)
            row_dict.pop("rowid", None)

            try:
                if change_type == "insert":
                    out = effector.handle_insert(row_dict)
                elif change_type == "delete":
                    out = effector.handle_delete(row_dict)
                elif change_type == "update_before":
                    # Stash the old row for pairing with update_after
                    _update_before = row_dict
                    continue
                elif change_type == "update_after":
                    out = effector.handle_update(_update_before, row_dict)  # noqa: F821
                else:
                    continue
            except Exception as e:
                if effector.on_error == "raise":
                    raise
                elif effector.on_error == "skip":
                    result.rows_errored += 1
                    continue
                else:  # "store"
                    error_row = {name: None for name, _ in effector.columns}
                    error_row["error"] = str(e)
                    output_rows.append(error_row)
                    result.rows_errored += 1
                    continue

            if out is None:
                result.rows_skipped += 1
            else:
                output_rows.append(out)
                result.rows_inserted += 1

        if output_rows:
            col_list = ", ".join(name for name, _ in effector.columns)
            placeholders = ", ".join("?" for _ in effector.columns)
            insert_sql = f"INSERT INTO {effector.output_fqn} ({col_list}) VALUES ({placeholders})"
            for out_row in output_rows:
                values = [out_row.get(name) for name, _ in effector.columns]
                self._conn.execute(insert_sql, values)

        self._sink_cursors[effector.effector_name] = latest_snapshot
        return result

    # -- Catalog management --------------------------------------------------

    def add_catalog(
        self,
        name: str,
        attach: Callable[[duckdb.DuckDBPyConnection, str], None],
    ) -> None:
        """Register a DuckLake catalog. Calls attach(conn, name) immediately."""
        attach(self._conn, name)
        self._catalogs[name] = attach

    # -- MV registration -----------------------------------------------------

    def add_ivm(
        self,
        catalog: str,
        name: str,
        sql: str,
        *,
        schema: str = "main",
    ) -> MaterializedView:
        """Compile and register an IVM. Returns the MaterializedView."""
        # Create a per-MV naming that uses the given name but inherits
        # other conventions from the orchestrator's naming instance.
        base_naming = self._naming

        class _Named(type(base_naming)):  # type: ignore[misc]
            def mv_table(self) -> str:
                return name

        naming = _Named()
        mv = compile_ivm(
            sql,
            naming=naming,
            mv_catalog=catalog,
            mv_schema=schema,
        )
        mv_name = name
        fqn = f"{catalog}.{schema}.{mv_name}"
        self._mvs[fqn] = _RegisteredMV(
            catalog=catalog,
            schema=schema,
            mv_name=mv_name,
            mv=mv,
            naming=naming,
            fqn=fqn,
        )
        return mv

    # -- Initialization ------------------------------------------------------

    def initialize(self) -> None:
        """Idempotently create cursor tables, MVs, and initialize cursors.

        Runs source setup and initial sync before MV initialization.
        Safe to call multiple times — skips anything that already exists.
        """
        # Set up and sync sources before MV init so base tables are populated
        for source in self._sources:
            source.setup(self._conn)
            source.sync(self._conn)

        for reg in self._mvs.values():
            mv = reg.mv
            # CREATE TABLE IF NOT EXISTS for cursors
            self._conn.execute(mv.create_cursors_table)
            # Check if MV table already exists
            exists = self._conn.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_catalog = ? AND table_schema = ? AND table_name = ?",
                [reg.catalog, reg.schema, reg.mv_name],
            ).fetchone()
            if not exists:
                self._conn.execute(mv.create_mv)
                for stmt in mv.initialize_cursors:
                    self._conn.execute(stmt)

        # Set up sinks and initialize their cursor table and rows
        for sink in self._sinks:
            sink.setup(self._conn)
            cursors_fqn = f"{sink.catalog}.{sink.schema_}._sink_cursors"
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {cursors_fqn} ("
                f"sink_name VARCHAR, mv_name VARCHAR, last_snapshot BIGINT)"
            )
            # Init cursor to current max snapshot so first flush only sees
            # future changes (not the initial load).
            self._conn.execute(
                f"INSERT INTO {cursors_fqn} (sink_name, mv_name, last_snapshot) "
                f"SELECT '{sink.sink_name}', '{sink.table}', "
                f"COALESCE(MAX(snapshot_id), 0) "
                f"FROM ducklake_snapshots('{sink.catalog}') "
                f"WHERE NOT EXISTS ("
                f"SELECT 1 FROM {cursors_fqn} "
                f"WHERE sink_name = '{sink.sink_name}' AND mv_name = '{sink.table}')"
            )
            # Load persisted cursor into in-memory cache
            row = self._conn.execute(
                f"SELECT last_snapshot FROM {cursors_fqn} "
                f"WHERE sink_name = '{sink.sink_name}' AND mv_name = '{sink.table}'"
            ).fetchone()
            if row:
                self._sink_cursors[sink.sink_name] = row[0]

        # Set up effectors: create output tables and initialize cursors
        for effector in self._effectors:
            effector.setup(self._conn)

            # Create output table from effector.columns
            col_defs = ", ".join(f"{name} {sql_type}" for name, sql_type in effector.columns)
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {effector.output_fqn} ({col_defs})")

            # Reuse _sink_cursors table for effector cursors
            cursors_fqn = f"{effector.catalog}.{effector.schema_}._sink_cursors"
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {cursors_fqn} ("
                f"sink_name VARCHAR, mv_name VARCHAR, last_snapshot BIGINT)"
            )
            self._conn.execute(
                f"INSERT INTO {cursors_fqn} (sink_name, mv_name, last_snapshot) "
                f"SELECT '{effector.effector_name}', '{effector.table}', "
                f"COALESCE(MAX(snapshot_id), 0) "
                f"FROM ducklake_snapshots('{effector.catalog}') "
                f"WHERE NOT EXISTS ("
                f"SELECT 1 FROM {cursors_fqn} "
                f"WHERE sink_name = '{effector.effector_name}' "
                f"AND mv_name = '{effector.table}')"
            )
            row = self._conn.execute(
                f"SELECT last_snapshot FROM {cursors_fqn} "
                f"WHERE sink_name = '{effector.effector_name}' "
                f"AND mv_name = '{effector.table}'"
            ).fetchone()
            if row:
                self._sink_cursors[effector.effector_name] = row[0]

    # -- Verification --------------------------------------------------------

    def verify(self) -> None:
        """Validate all registered IVMs."""
        self._verify_catalogs()
        self._verify_tables_and_columns()
        self._topo_sort()  # will raise CyclicDependencyError if cycles

    def _verify_catalogs(self) -> None:
        """Check all referenced catalogs are attached and are DuckLake type."""
        # Get attached databases
        rows = self._conn.execute("SELECT database_name, type FROM duckdb_databases()").fetchall()
        db_map = {r[0]: r[1] for r in rows}

        needed: set[str] = set()
        for reg in self._mvs.values():
            needed.add(reg.catalog)
            for _table_name, cat in reg.mv.base_tables.items():
                needed.add(cat)

        for cat in needed:
            if cat not in db_map:
                raise CatalogNotFoundError(f"Catalog '{cat}' is not attached")
            if db_map[cat] != "ducklake":
                raise NotDuckLakeError(
                    f"Catalog '{cat}' has type '{db_map[cat]}', expected 'ducklake'"
                )

    def _verify_tables_and_columns(self) -> None:
        """Check all base tables exist with required columns."""
        for reg in self._mvs.values():
            for table_name, cat in reg.mv.base_tables.items():
                # Check table exists
                result = self._conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_catalog = ? AND table_schema = 'main' AND table_name = ?",
                    [cat, table_name],
                ).fetchall()
                if not result:
                    raise MissingTableError(f"Table '{cat}.main.{table_name}' not found")

    # -- Dependency graph & topo sort ----------------------------------------

    def _build_dependency_graph(self) -> dict[str, list[str]]:
        """Build adjacency list: mv_fqn -> list of mv_fqns it depends on."""
        # Map (catalog, schema, table_name) -> mv_fqn for MVs that produce those tables
        mv_table_map: dict[tuple[str, str, str], str] = {}
        for fqn, reg in self._mvs.items():
            mv_table_map[(reg.catalog, reg.schema, reg.mv_name)] = fqn

        deps: dict[str, list[str]] = {fqn: [] for fqn in self._mvs}
        for fqn, reg in self._mvs.items():
            for table_name, cat in reg.mv.base_tables.items():
                # Check if this base table is actually another MV
                key = (cat, "main", table_name)
                if key in mv_table_map and mv_table_map[key] != fqn:
                    deps[fqn].append(mv_table_map[key])
        return deps

    def _topo_sort(self) -> list[list[str]]:
        """Kahn's algorithm. Returns levels (groups that can run in parallel).

        Raises CyclicDependencyError if the graph has cycles.
        """
        deps = self._build_dependency_graph()

        # Build in-degree map and reverse adjacency
        in_degree: dict[str, int] = {fqn: 0 for fqn in self._mvs}
        reverse: dict[str, list[str]] = defaultdict(list)
        for fqn, dep_list in deps.items():
            in_degree[fqn] = len(dep_list)
            for dep in dep_list:
                reverse[dep].append(fqn)

        # Kahn's with level tracking
        queue: deque[str] = deque()
        for fqn, deg in in_degree.items():
            if deg == 0:
                queue.append(fqn)

        levels: list[list[str]] = []
        processed = 0

        while queue:
            level = list(queue)
            levels.append(level)
            next_queue: deque[str] = deque()
            for fqn in level:
                processed += 1
                for dependent in reverse[fqn]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            queue = next_queue

        if processed != len(self._mvs):
            remaining = [fqn for fqn, deg in in_degree.items() if deg > 0]
            raise CyclicDependencyError(f"Cyclic dependency detected among: {remaining}")

        return levels

    # -- Tree visualization --------------------------------------------------

    def tree(self) -> str:
        """Return a pretty-printed dependency tree string."""
        # Group base tables by catalog
        catalog_tables: dict[str, set[str]] = defaultdict(set)
        for reg in self._mvs.values():
            for table_name, cat in reg.mv.base_tables.items():
                catalog_tables[cat].add(table_name)

        # Build: base_table -> list of MVs that use it
        table_to_mvs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for fqn, reg in self._mvs.items():
            for table_name, cat in reg.mv.base_tables.items():
                table_to_mvs[(cat, table_name)].append(fqn)

        # Build: mv_fqn -> list of MVs that depend on it
        deps = self._build_dependency_graph()
        reverse_deps: dict[str, list[str]] = defaultdict(list)
        for fqn, dep_list in deps.items():
            for dep in dep_list:
                reverse_deps[dep].append(fqn)

        lines: list[str] = []

        def _add_mv_tree(mv_fqn: str, prefix: str, is_last: bool) -> None:
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            reg = self._mvs[mv_fqn]
            label = "(MV, full refresh)" if reg.mv.strategy == "full_refresh" else "(MV)"
            lines.append(f"{prefix}{connector}{mv_fqn} {label}")
            child_prefix = prefix + ("    " if is_last else "\u2502   ")
            dependents = sorted(reverse_deps.get(mv_fqn, []))
            for i, dep_fqn in enumerate(dependents):
                _add_mv_tree(dep_fqn, child_prefix, i == len(dependents) - 1)

        catalogs_sorted = sorted(catalog_tables.keys())
        for cat in catalogs_sorted:
            lines.append(f"{cat} (ducklake catalog)")
            tables = sorted(catalog_tables[cat])
            for ti, table_name in enumerate(tables):
                is_last_table = ti == len(tables) - 1
                connector = "\u2514\u2500\u2500 " if is_last_table else "\u251c\u2500\u2500 "
                lines.append(f"{connector}{table_name} (base table)")
                table_prefix = "    " if is_last_table else "\u2502   "
                mvs = sorted(table_to_mvs.get((cat, table_name), []))
                # Only show MVs that directly use this base table (not via another MV)
                for mi, mv_fqn in enumerate(mvs):
                    _add_mv_tree(mv_fqn, table_prefix, mi == len(mvs) - 1)

        return "\n".join(lines)

    # -- Maintenance planning ------------------------------------------------

    def get_maintenance_plan(
        self,
        detail_level: Literal["none", "cursor_distance", "rows_changed"] = "none",
    ) -> MaintenancePlan:
        """Build a topologically-sorted maintenance plan."""
        levels_fqns = self._topo_sort()

        all_steps: list[MaintenanceStep] = []
        level_steps: list[list[MaintenanceStep]] = []

        for level_fqns in levels_fqns:
            current_level: list[MaintenanceStep] = []
            for fqn in level_fqns:
                reg = self._mvs[fqn]
                step = MaintenanceStep(
                    catalog=reg.catalog,
                    schema=reg.schema,
                    mv_name=reg.mv_name,
                    mv=reg.mv,
                    strategy=reg.mv.strategy,
                )

                if detail_level in ("cursor_distance", "rows_changed"):
                    step.pending_snapshots = self._get_pending_snapshots(reg)

                if detail_level == "rows_changed":
                    step.pending_rows = self._get_pending_rows(reg)

                all_steps.append(step)
                current_level.append(step)
            level_steps.append(current_level)

        return MaintenancePlan(steps=all_steps, levels=level_steps)

    def _get_pending_snapshots(self, reg: _RegisteredMV) -> int:
        """Count pending snapshots for an MV across all its source catalogs."""
        cursors_fqn = f"{reg.catalog}.{reg.schema}.{reg.naming.cursors_table()}"
        total = 0
        for _table_name, cat in reg.mv.base_tables.items():
            result = self._conn.execute(
                f"SELECT (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{cat}'))"
                f" - last_snapshot AS pending "
                f"FROM {cursors_fqn} "
                f"WHERE mv_name = '{reg.mv_name}' AND source_catalog = '{cat}'",
            ).fetchone()
            if result and result[0]:
                total += result[0]
        return total

    def _get_pending_rows(self, reg: _RegisteredMV) -> int:
        """Count pending row changes for an MV."""
        cursors_fqn = f"{reg.catalog}.{reg.schema}.{reg.naming.cursors_table()}"
        total = 0
        for table_name, cat in reg.mv.base_tables.items():
            cursor_row = self._conn.execute(
                f"SELECT last_snapshot FROM {cursors_fqn} "
                f"WHERE mv_name = '{reg.mv_name}' AND source_catalog = '{cat}'",
            ).fetchone()
            if not cursor_row:
                continue
            last_snap = cursor_row[0]
            latest = self._conn.execute(
                f"SELECT MAX(snapshot_id) FROM ducklake_snapshots('{cat}')"
            ).fetchone()
            if not latest or latest[0] is None or latest[0] <= last_snap:
                continue
            self._conn.execute(f"SET VARIABLE _ivm_orch_start = {last_snap + 1}")
            self._conn.execute(f"SET VARIABLE _ivm_orch_end = {latest[0]}")
            count_result = self._conn.execute(
                f"SELECT COUNT(*) FROM ducklake_table_changes("
                f"'{cat}', 'main', '{table_name}', "
                f"getvariable('_ivm_orch_start'), getvariable('_ivm_orch_end'))"
            ).fetchone()
            if count_result:
                total += count_result[0]
        return total

    # -- Maintenance execution -----------------------------------------------

    def _execute_step(
        self,
        step: MaintenanceStep,
        conn: duckdb.DuckDBPyConnection,
        verbose: bool = False,
    ) -> MaintenanceStep:
        """Execute a single maintenance step."""
        t0 = time.monotonic()

        # Check if there's actually work to do
        if step.pending_snapshots is not None and step.pending_snapshots == 0:
            step.skipped = True
            step.duration_ms = (time.monotonic() - t0) * 1000
            return step

        for stmt in step.mv.maintain:
            if verbose:
                print(f"[{step.catalog}.{step.schema}.{step.mv_name}] {stmt[:80]}...")
            conn.execute(stmt)

        step.duration_ms = (time.monotonic() - t0) * 1000
        return step

    def advance_one(
        self,
        verbose: bool = False,
        detail_level: Literal["cursor_distance", "rows_changed"] = "cursor_distance",
    ) -> Future[MaintenanceStep] | None:
        """Submit the next pending maintenance step to a thread pool.

        Returns a Future that resolves to the completed MaintenanceStep,
        or None if nothing to do.
        """
        if self._advance_state is None:
            plan = self.get_maintenance_plan(detail_level=detail_level)
            self._advance_state = _AdvanceState(plan=plan, completed=set())
            self._executor = self._executor or ThreadPoolExecutor(max_workers=1)

        state = self._advance_state
        deps = self._build_dependency_graph()

        # Find next step whose dependencies are all complete
        for step in state.plan.steps:
            fqn = f"{step.catalog}.{step.schema}.{step.mv_name}"
            if fqn in state.completed or fqn in state.in_flight:
                continue
            dep_fqns = deps.get(fqn, [])
            if all(d in state.completed for d in dep_fqns):
                state.in_flight.add(fqn)
                cursor = self._conn.cursor()

                def _run(
                    s: MaintenanceStep = step,
                    c: duckdb.DuckDBPyConnection = cursor,
                    f: str = fqn,
                ) -> MaintenanceStep:
                    try:
                        result = self._execute_step(s, c, verbose)
                        return result
                    finally:
                        state.in_flight.discard(f)
                        state.completed.add(f)

                assert self._executor is not None
                return self._executor.submit(_run)

        # Nothing left to do — reset state
        self._advance_state = None
        return None

    async def advance_one_async(
        self,
        verbose: bool = False,
        detail_level: Literal["cursor_distance", "rows_changed"] = "cursor_distance",
    ) -> MaintenanceStep | None:
        """Async version of advance_one."""
        future = self.advance_one(verbose=verbose, detail_level=detail_level)
        if future is None:
            return None
        return await asyncio.wrap_future(future)

    def maintain(
        self,
        verbose: bool = False,
        detail_level: Literal["cursor_distance", "rows_changed"] = "cursor_distance",
        parallelism: int = 1,
    ) -> list[MaintenanceStep]:
        """Run all pending maintenance. Returns list of steps executed."""
        plan = self.get_maintenance_plan(detail_level=detail_level)
        results: list[MaintenanceStep] = []

        if parallelism <= 1:
            # Simple sequential execution
            for step in plan.steps:
                self._execute_step(step, self._conn, verbose)
                results.append(step)
            return results

        # Parallel execution by level
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for level in plan.levels:
                futures: list[Future[MaintenanceStep]] = []
                for step in level:
                    cursor = self._conn.cursor()
                    futures.append(pool.submit(self._execute_step, step, cursor, verbose))
                for f in futures:
                    results.append(f.result())

        return results

    async def maintain_async(
        self,
        verbose: bool = False,
        detail_level: Literal["cursor_distance", "rows_changed"] = "cursor_distance",
        parallelism: int = 1,
    ) -> list[MaintenanceStep]:
        """Async version of maintain."""
        plan = self.get_maintenance_plan(detail_level=detail_level)
        results: list[MaintenanceStep] = []

        loop = asyncio.get_running_loop()

        if parallelism <= 1:
            for step in plan.steps:
                await loop.run_in_executor(None, self._execute_step, step, self._conn, verbose)
                results.append(step)
            return results

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for level in plan.levels:
                tasks = []
                for step in level:
                    cursor = self._conn.cursor()
                    tasks.append(
                        loop.run_in_executor(pool, self._execute_step, step, cursor, verbose)
                    )
                level_results = await asyncio.gather(*tasks)
                results.extend(level_results)

        return results


@dataclass
class _AdvanceState:
    """Internal state for advance_one() stepping."""

    plan: MaintenancePlan
    completed: set[str] = field(default_factory=set)
    in_flight: set[str] = field(default_factory=set)
