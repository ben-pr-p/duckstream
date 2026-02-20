"""duckstream watch — continuous maintenance loop."""

from __future__ import annotations

import time
from datetime import datetime

import click

from duckstream.cli import load_project


@click.command()
@click.option("--plain", is_flag=True, help="Plain text output instead of TUI.")
@click.option(
    "--polling-interval",
    type=float,
    default=5.0,
    help="Seconds between maintenance checks.",
)
@click.pass_context
def watch(ctx: click.Context, plain: bool, polling_interval: float) -> None:
    """Continuously watch for data changes and run maintenance."""
    project_dir = ctx.obj["project_dir"]

    try:
        orch = load_project(project_dir)
        orch.verify()
        orch.initialize()
    except Exception as e:
        raise click.ClickException(str(e)) from e

    if plain:
        _run_plain(orch, polling_interval)
    else:
        _run_tui(orch, polling_interval)


def _has_pending_work(orch) -> bool:
    """Check if any MV has pending snapshots."""
    plan = orch.get_maintenance_plan(detail_level="cursor_distance")
    return any(
        step.pending_snapshots is not None and step.pending_snapshots > 0 for step in plan.steps
    )


def _run_plain(orch, polling_interval: float) -> None:
    """Watch with plain text output."""
    click.echo(orch.tree())
    click.echo()
    click.echo(f"Watching for changes (polling every {polling_interval}s) ...")
    click.echo("Press Ctrl+C to stop.")
    click.echo()

    try:
        while True:
            if _has_pending_work(orch):
                now = datetime.now().strftime("%H:%M:%S")
                click.echo(f"[{now}] Changes detected, running maintenance ...")
                steps = orch.maintain(detail_level="cursor_distance")
                for step in steps:
                    status = "skipped" if step.skipped else f"{step.duration_ms:.0f}ms"
                    strategy = " [full refresh]" if step.strategy == "full_refresh" else ""
                    click.echo(f"  {step.catalog}.{step.schema}.{step.mv_name}{strategy}: {status}")
                click.echo()
            time.sleep(polling_interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")


def _run_tui(orch, polling_interval: float) -> None:
    """Watch with TUI output."""
    from duckstream.tui.app import DuckStreamApp

    class WatchApp(DuckStreamApp):
        def on_mount(self) -> None:
            log = self.log_widget
            log.write(f"[bold]Watching for changes (polling every {polling_interval}s) ...[/bold]")
            log.write("")
            self.set_interval(polling_interval, self._check_and_maintain)

        async def _check_and_maintain(self) -> None:
            log = self.log_widget

            plan = self.orchestrator.get_maintenance_plan(detail_level="cursor_distance")
            has_work = any(
                step.pending_snapshots is not None and step.pending_snapshots > 0
                for step in plan.steps
            )

            if not has_work:
                return

            now = datetime.now().strftime("%H:%M:%S")
            log.write(f"[{now}] Changes detected, running maintenance ...")

            while True:
                step_result = await self.orchestrator.advance_one_async(
                    detail_level="cursor_distance"
                )
                if step_result is None:
                    break
                fqn = f"{step_result.catalog}.{step_result.schema}.{step_result.mv_name}"
                strategy = " [full refresh]" if step_result.strategy == "full_refresh" else ""
                if step_result.skipped:
                    log.write(f"  [dim]{fqn}{strategy}: skipped[/dim]")
                else:
                    log.write(f"  [green]{fqn}{strategy}[/green]: {step_result.duration_ms:.0f}ms")

            log.write("")

    app = WatchApp(orch, title="DuckStream — Watch")
    app.run()
