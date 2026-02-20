"""duckstream maintain — run maintenance once."""

from __future__ import annotations

import click

from duckstream.cli import load_project


@click.command()
@click.option("--plain", is_flag=True, help="Plain text output instead of TUI.")
@click.pass_context
def maintain(ctx: click.Context, plain: bool) -> None:
    """Run maintenance on all materialized views once."""
    project_dir = ctx.obj["project_dir"]

    try:
        orch = load_project(project_dir)
        orch.verify()
        orch.initialize()
    except Exception as e:
        raise click.ClickException(str(e)) from e

    if plain:
        _run_plain(orch)
    else:
        _run_tui(orch)


def _run_plain(orch):
    """Run maintenance with plain text output."""
    click.echo(orch.tree())
    click.echo()
    click.echo("Running maintenance ...")

    steps = orch.maintain(detail_level="cursor_distance")

    for step in steps:
        status = "skipped" if step.skipped else f"{step.duration_ms:.0f}ms"
        pending = f" ({step.pending_snapshots} pending)" if step.pending_snapshots else ""
        strategy = " [full refresh]" if step.strategy == "full_refresh" else ""
        click.echo(f"  {step.catalog}.{step.schema}.{step.mv_name}{strategy}: {status}{pending}")

    click.echo("Maintenance complete.")


def _run_tui(orch):
    """Run maintenance with TUI output."""
    from duckstream.tui.app import DuckStreamApp

    class MaintainApp(DuckStreamApp):
        def on_mount(self) -> None:
            self.run_worker(self._do_maintain())

        async def _do_maintain(self) -> None:
            log = self.log_widget
            log.write("[bold]Running maintenance ...[/bold]")
            log.write("")

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
            log.write("[bold green]Maintenance complete.[/bold green]")
            log.write("Press [bold]q[/bold] to quit.")

    app = MaintainApp(orch, title="DuckStream — Maintain")
    app.run()
