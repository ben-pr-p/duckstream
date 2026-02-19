"""duckstream dev — watch SQL files, recompile on change."""

from __future__ import annotations

import threading
from pathlib import Path

import click
import duckdb

from duckstream.cli import load_project
from duckstream.orchestrator import Orchestrator

MAX_PREVIEW_ROWS = 20


@click.command()
@click.option(
    "--polling-interval",
    type=float,
    default=5.0,
    help="Seconds between file change checks.",
)
@click.pass_context
def dev(ctx: click.Context, polling_interval: float) -> None:
    """Watch SQL files and recompile on change."""
    project_dir = ctx.obj["project_dir"]
    _run_tui(project_dir, polling_interval)


def _try_compile_project(project_dir: str) -> tuple[Orchestrator | None, Exception | None]:
    """Attempt to load and compile a project (no verify/initialize). Returns (orch, error)."""
    try:
        orch = load_project(project_dir)
        return orch, None
    except Exception as e:
        return None, e


def _format_table(columns: list[str], rows: list[tuple], total_count: int) -> str:
    """Format query results as an ASCII table."""
    col_widths = [len(c) for c in columns]
    str_rows = []
    for row in rows:
        str_row = [str(v) if v is not None else "NULL" for v in row]
        str_rows.append(str_row)
        for i, val in enumerate(str_row):
            col_widths[i] = max(col_widths[i], min(len(val), 40))

    def _truncate(val: str, width: int) -> str:
        if len(val) > width:
            return val[: width - 1] + "…"
        return val

    lines: list[str] = []
    header = " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(columns))
    lines.append(header)
    lines.append("-+-".join("-" * w for w in col_widths))
    for str_row in str_rows:
        line = " | ".join(
            _truncate(val, col_widths[i]).ljust(col_widths[i]) for i, val in enumerate(str_row)
        )
        lines.append(line)
    if total_count > len(rows):
        lines.append(f"... ({total_count - len(rows)} more rows)")
    return "\n".join(lines)


def _run_query(conn: duckdb.DuckDBPyConnection, sql: str) -> str:
    """Execute SQL and return a formatted table string or error."""
    try:
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        all_rows = result.fetchall()
        total = len(all_rows)
        preview_rows = all_rows[:MAX_PREVIEW_ROWS]
        if preview_rows:
            return _format_table(columns, preview_rows, total)
        return "(empty — 0 rows)"
    except Exception as e:
        return f"Error: {e}"


def _run_tui(project_dir: str, polling_interval: float) -> None:
    """Dev mode TUI — watches SQL files and recompiles."""
    from watchfiles import watch

    from duckstream.tui.app import DuckStreamApp
    from duckstream.tui.tree_sidebar import MVSelected

    catalogs_dir = Path(project_dir) / "catalogs"
    orch, initial_error = _try_compile_project(project_dir)

    if orch is None:
        raise click.ClickException(f"Could not load project from {project_dir}: {initial_error}")

    # Event to signal the file watcher thread to stop
    stop_event = threading.Event()

    class DevApp(DuckStreamApp):
        BINDINGS = [
            ("q", "quit", "Quit"),
        ]

        def __init__(self, orchestrator, **kwargs):
            super().__init__(orchestrator, **kwargs)
            self._active_fqn: str | None = None
            self._initial_error = initial_error

        def on_mount(self) -> None:
            log = self.log_widget
            if self._initial_error is not None:
                log.write(f"[bold red]{self._initial_error}[/bold red]")
                log.write("")
            log.write("[dim]Select a view and press Enter to preview. Press q to quit.[/dim]")
            self.run_worker(self._watch_files())

        def on_unmount(self) -> None:
            stop_event.set()

        def on_mv_selected(self, event: MVSelected) -> None:
            """Handle MV selection from the tree sidebar (Enter key)."""
            self._active_fqn = event.fqn
            self._run_active_query()

        def _run_active_query(self) -> None:
            """Show loading state and run the active query in a worker."""
            log = self.log_widget
            log.clear()

            fqn = self._active_fqn
            if fqn is None:
                log.write("[dim]Select a view and press Enter to preview. Press q to quit.[/dim]")
                return

            reg = self.orchestrator._mvs.get(fqn)
            if reg is None:
                log.write(f"[bold red]{fqn} not found after recompile[/bold red]")
                return

            log.write(f"[bold]{fqn}[/bold]")
            log.write(f"[dim]{reg.mv.view_sql}[/dim]")
            log.write("")
            log.write("[yellow]Running query ...[/yellow]")

            # Run in a worker so the UI stays responsive
            self.run_worker(
                self._execute_and_display(fqn, reg.mv.view_sql),
                exclusive=True,
            )

        async def _execute_and_display(self, fqn: str, sql: str) -> None:
            """Execute the query in a thread and display results."""
            import asyncio

            loop = asyncio.get_running_loop()
            conn = self.orchestrator._conn
            result_str = await loop.run_in_executor(None, _run_query, conn, sql)

            log = self.log_widget
            log.clear()
            log.write(f"[bold]{fqn}[/bold]")
            log.write(f"[dim]{sql}[/dim]")
            log.write("")
            log.write(result_str)

        async def _watch_files(self) -> None:
            import asyncio
            from functools import partial

            loop = asyncio.get_running_loop()

            def _blocking_watch():
                yield from watch(
                    catalogs_dir,
                    watch_filter=lambda change, path: path.endswith(".sql"),
                    step=int(polling_interval * 1000),
                    stop_event=stop_event,
                )

            def _next_change(iterator):
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            iterator = _blocking_watch()
            while not stop_event.is_set():
                changes = await loop.run_in_executor(None, partial(_next_change, iterator))
                if changes is None:
                    break
                self._handle_changes(changes)

        def _handle_changes(self, changes) -> None:
            # Figure out which FQN the saved file corresponds to
            saved_fqn = self._fqn_for_changes(changes)

            # Recompile
            new_orch, error = _try_compile_project(project_dir)
            if error is not None:
                log = self.log_widget
                log.clear()
                log.write(f"[bold red]{error}[/bold red]")
                return

            if new_orch is not None:
                self.orchestrator = new_orch
                self.refresh_tree()

                # If the saved file maps to an MV, switch to it
                if saved_fqn and saved_fqn in self.orchestrator._mvs:
                    self._active_fqn = saved_fqn

                self._run_active_query()

        def _fqn_for_changes(self, changes) -> str | None:
            """Try to map changed files back to an MV FQN."""
            for _change_type, path in changes:
                p = Path(path)
                if p.suffix != ".sql":
                    continue
                mv_name = p.stem
                # Walk up to find catalog name: catalogs/<catalog>/<mv>.sql
                # or catalogs/<catalog>/<schema>/<mv>.sql
                parts = p.relative_to(catalogs_dir).parts
                if len(parts) == 2:
                    catalog = parts[0]
                    return f"{catalog}.main.{mv_name}"
                elif len(parts) == 3:
                    catalog = parts[0]
                    schema = parts[1]
                    return f"{catalog}.{schema}.{mv_name}"
            return None

    app = DevApp(orch, title="DuckStream — Dev")
    app.run()
