"""duckstream plan — show maintenance strategy for each MV."""

from __future__ import annotations

import click

from duckstream.file_loader import MVCompilationError, load_directory
from duckstream.orchestrator import Orchestrator


@click.command()
@click.pass_context
def plan(ctx: click.Context) -> None:
    """Show the maintenance strategy for each materialized view."""
    project_dir = ctx.obj["project_dir"]

    try:
        orch = Orchestrator()
        errors: list[MVCompilationError] = []
        load_directory(project_dir, orch, errors=errors)
    except Exception as e:
        raise click.ClickException(str(e)) from e

    if not orch._mvs and not errors:
        click.echo("No materialized views found.")
        return

    # Build dependency levels for ordering
    try:
        levels = orch._topo_sort()
    except Exception:
        levels = [list(orch._mvs.keys())]

    # Flatten levels preserving order
    ordered_fqns: list[str] = []
    for level in levels:
        ordered_fqns.extend(sorted(level))

    click.echo()
    click.echo(click.style("Maintenance Plan", bold=True))
    click.echo(click.style("=" * 60, dim=True))

    for fqn in ordered_fqns:
        reg = orch._mvs[fqn]
        mv = reg.mv
        click.echo()

        # Header
        click.echo(click.style(f"  {fqn}", bold=True))

        # Strategy
        if mv.strategy == "incremental":
            click.echo(f"    Strategy:  {click.style('incremental', fg='green')}")
        else:
            click.echo(f"    Strategy:  {click.style('full refresh', fg='yellow')}")
            if mv.fallback_reason:
                click.echo(f"    Reason:    {mv.fallback_reason}")

        # View SQL (truncated)
        sql_display = mv.view_sql.replace("\n", " ").strip()
        if len(sql_display) > 80:
            sql_display = sql_display[:77] + "..."
        click.echo(click.style(f"    SQL:       {sql_display}", dim=True))

        # Base tables
        tables = ", ".join(f"{cat}.{name}" for name, cat in mv.base_tables.items())
        click.echo(f"    Sources:   {tables}")

        # Features
        if mv.features:
            click.echo(f"    Features:  {', '.join(sorted(mv.features))}")

        # Inner MVs
        if mv.inner_mvs:
            click.echo(f"    Inner MVs: {len(mv.inner_mvs)}")
            for inner in mv.inner_mvs:
                inner_strategy = (
                    click.style("full refresh", fg="yellow")
                    if inner.strategy == "full_refresh"
                    else click.style("incremental", fg="green")
                )
                click.echo(f"      - {inner_strategy}")
                if inner.fallback_reason:
                    click.echo(f"        Reason: {inner.fallback_reason}")

        # Maintenance steps
        click.echo(f"    Steps:     {len(mv.maintain)} SQL statements")

    # Show compilation errors
    if errors:
        click.echo()
        click.echo(click.style("  Errors", bold=True))
        click.echo(click.style("  " + "-" * 56, dim=True))
        for err in errors:
            fqn = f"{err.catalog}.{err.schema}.{err.mv_name}"
            click.echo()
            click.echo(click.style(f"  {fqn}", bold=True))
            click.echo(f"    Strategy:  {click.style('error', fg='red')}")
            click.echo(f"    Error:     {err.cause}")

    click.echo()
    click.echo(click.style("-" * 60, dim=True))

    # Summary
    total = len(ordered_fqns)
    incremental = sum(1 for fqn in ordered_fqns if orch._mvs[fqn].mv.strategy == "incremental")
    full_refresh = total - incremental
    parts = [
        f"{click.style(str(incremental), fg='green')} incremental",
        f"{click.style(str(full_refresh), fg='yellow')} full refresh",
    ]
    if errors:
        parts.append(f"{click.style(str(len(errors)), fg='red')} failed")
        total += len(errors)
    click.echo(f"  {total} MVs: {', '.join(parts)}")
    click.echo()
