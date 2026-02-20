"""duckstream plan — show maintenance strategy for each MV."""

from __future__ import annotations

import click

from duckstream.cli import load_project


@click.command()
@click.pass_context
def plan(ctx: click.Context) -> None:
    """Show the maintenance strategy for each materialized view."""
    project_dir = ctx.obj["project_dir"]

    try:
        orch = load_project(project_dir)
    except Exception as e:
        raise click.ClickException(str(e)) from e

    if not orch._mvs:
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
    click.echo()

    for i, fqn in enumerate(ordered_fqns):
        reg = orch._mvs[fqn]
        mv = reg.mv

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

        if i < len(ordered_fqns) - 1:
            click.echo()

    click.echo()
    click.echo(click.style("-" * 60, dim=True))

    # Summary
    total = len(ordered_fqns)
    incremental = sum(1 for fqn in ordered_fqns if orch._mvs[fqn].mv.strategy == "incremental")
    full_refresh = total - incremental
    click.echo(
        f"  {total} MVs: "
        f"{click.style(str(incremental), fg='green')} incremental, "
        f"{click.style(str(full_refresh), fg='yellow')} full refresh"
    )
    click.echo()
