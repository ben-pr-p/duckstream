"""duckstream init — verify config, create MVs + cursors."""

from __future__ import annotations

import click

from duckstream.cli import load_project


@click.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Verify configuration, create materialized views and cursors."""
    project_dir = ctx.obj["project_dir"]

    click.echo(f"Loading project from {project_dir} ...")
    try:
        orch = load_project(project_dir)
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.echo("Verifying configuration ...")
    try:
        orch.verify()
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.echo("Initializing materialized views ...")
    try:
        orch.initialize()
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.echo()
    click.echo(orch.tree())
    click.echo()
    click.echo("Initialization complete.")
