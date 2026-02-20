"""DuckStream CLI — command-line interface for IVM management."""

from __future__ import annotations

from pathlib import Path

import click

from duckstream.file_loader import load_directory
from duckstream.orchestrator import Orchestrator


def load_project(project_dir: str | Path) -> Orchestrator:
    """Create an Orchestrator and load a project directory into it."""
    orch = Orchestrator()
    load_directory(project_dir, orch)
    return orch


@click.group()
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=".",
    help="Path to the DuckStream project directory.",
)
@click.pass_context
def main(ctx: click.Context, project_dir: str) -> None:
    """DuckStream — incremental view maintenance for DuckLake."""
    ctx.ensure_object(dict)
    ctx.obj["project_dir"] = project_dir


# Import and register subcommands
from duckstream.cli.dev_cmd import dev  # noqa: E402
from duckstream.cli.init_cmd import init  # noqa: E402
from duckstream.cli.maintain_cmd import maintain  # noqa: E402
from duckstream.cli.plan_cmd import plan  # noqa: E402
from duckstream.cli.watch_cmd import watch  # noqa: E402

main.add_command(init)
main.add_command(dev)
main.add_command(maintain)
main.add_command(plan)
main.add_command(watch)
