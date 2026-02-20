"""File-based loader for Orchestrator configuration.

Loads catalogs and MV definitions from a directory layout:

    my_project/
    ├── setup.py          # exports extensions(), secrets(), catalogs()
    └── catalogs/
        └── my_catalog/
            ├── orders_mv.sql           # MV in default "main" schema
            └── analytics/              # schema name
                └── daily_totals.sql    # MV in "analytics" schema

setup.py exports:
- catalogs() -> list of functions  (required; each function's __name__ is the catalog name)
- extensions(conn) -> None  (optional)
- secrets(conn) -> None  (optional)

SQL files contain the view definition (a SELECT statement).
The file name (minus .sql) becomes the MV name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckstream.orchestrator import Orchestrator


class FileLoaderError(Exception): ...


class MissingSetupError(FileLoaderError): ...


class InvalidSetupError(FileLoaderError): ...


class MissingCatalogsError(FileLoaderError): ...


class MVCompilationError(FileLoaderError):
    """A single MV failed to compile."""

    def __init__(self, catalog: str, schema: str, mv_name: str, cause: Exception):
        self.catalog = catalog
        self.schema = schema
        self.mv_name = mv_name
        self.cause = cause
        fqn = f"{catalog}.{schema}.{mv_name}"
        super().__init__(f"{fqn}: {cause}")


def load_directory(
    directory: str | Path,
    orchestrator: Orchestrator,
    errors: list[MVCompilationError] | None = None,
) -> None:
    """Load a project directory into the orchestrator.

    1. Imports setup.py and calls its exported functions:
       - extensions(conn) if present
       - secrets(conn) if present
       - catalogs() to get catalog attach functions
    2. Walks catalogs/ to register MV definitions via add_ivm().

    If ``errors`` is provided, compilation failures are appended to it
    instead of raising immediately. Setup/catalog-level errors always raise.
    """
    root = Path(directory)

    # -- Load setup.py -------------------------------------------------------
    setup_path = root / "setup.py"
    if not setup_path.is_file():
        raise MissingSetupError(f"No setup.py found in {root}")

    module = _load_setup_module(setup_path)

    # extensions(conn) — optional
    if hasattr(module, "extensions"):
        fn = module.extensions
        if not callable(fn):
            raise InvalidSetupError(f"{setup_path}: `extensions` is not callable")
        orchestrator.setup_extensions(fn)

    # secrets(conn) — optional
    if hasattr(module, "secrets"):
        fn = module.secrets
        if not callable(fn):
            raise InvalidSetupError(f"{setup_path}: `secrets` is not callable")
        orchestrator.setup_secrets(fn)

    # catalogs() — required
    if not hasattr(module, "catalogs"):
        raise InvalidSetupError(
            f"{setup_path} must export a `catalogs` function, "
            f"found: {[n for n in dir(module) if not n.startswith('_')]}"
        )
    catalogs_fn = module.catalogs
    if not callable(catalogs_fn):
        raise InvalidSetupError(f"{setup_path}: `catalogs` is not callable")

    catalog_list = catalogs_fn()
    if not isinstance(catalog_list, list):
        raise InvalidSetupError(
            f"{setup_path}: `catalogs()` must return a list, got {type(catalog_list).__name__}"
        )
    for attach in catalog_list:
        if not callable(attach):
            raise InvalidSetupError(f"{setup_path}: `catalogs()` must return a list of functions")
        name = attach.__name__

        # The user's function takes (conn); add_catalog expects (conn, name).
        def _adapt(conn, _name, fn=attach):
            fn(conn)

        orchestrator.add_catalog(name, _adapt)

    # sources() — optional
    if hasattr(module, "sources"):
        sources_fn = module.sources
        if not callable(sources_fn):
            raise InvalidSetupError(f"{setup_path}: `sources` is not callable")
        source_list = sources_fn()
        if not isinstance(source_list, list):
            raise InvalidSetupError(
                f"{setup_path}: `sources()` must return a list, got {type(source_list).__name__}"
            )
        for source in source_list:
            orchestrator.add_source(source)

    # sinks() — optional
    if hasattr(module, "sinks"):
        sinks_fn = module.sinks
        if not callable(sinks_fn):
            raise InvalidSetupError(f"{setup_path}: `sinks` is not callable")
        sink_list = sinks_fn()
        if not isinstance(sink_list, list):
            raise InvalidSetupError(
                f"{setup_path}: `sinks()` must return a list, got {type(sink_list).__name__}"
            )
        for sink in sink_list:
            orchestrator.add_sink(sink)

    # -- Walk catalogs/ ------------------------------------------------------
    catalogs_dir = root / "catalogs"
    if not catalogs_dir.is_dir():
        raise MissingCatalogsError(f"No catalogs/ directory found in {root}")

    for catalog_path in sorted(catalogs_dir.iterdir()):
        if not catalog_path.is_dir():
            continue
        catalog_name = catalog_path.name
        _load_catalog(catalog_path, catalog_name, orchestrator, errors)


def _load_setup_module(setup_path: Path):
    """Import setup.py and return the module."""
    module_name = f"_duckstream_setup_{setup_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, setup_path)
    if spec is None or spec.loader is None:
        raise InvalidSetupError(f"Could not load {setup_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise InvalidSetupError(f"Error executing {setup_path}: {e}") from e
    finally:
        sys.modules.pop(module_name, None)

    return module


def _load_catalog(
    catalog_path: Path,
    catalog_name: str,
    orchestrator: Orchestrator,
    errors: list[MVCompilationError] | None = None,
) -> None:
    """Load all MVs from a catalog directory."""
    for entry in sorted(catalog_path.iterdir()):
        if entry.is_file() and entry.suffix == ".sql":
            mv_name = entry.stem
            sql = entry.read_text().strip()
            _try_add_ivm(orchestrator, catalog_name, mv_name, sql, "main", errors)
        elif entry.is_dir():
            schema_name = entry.name
            _load_schema(entry, catalog_name, schema_name, orchestrator, errors)


def _load_schema(
    schema_path: Path,
    catalog_name: str,
    schema_name: str,
    orchestrator: Orchestrator,
    errors: list[MVCompilationError] | None = None,
) -> None:
    """Load all MVs from a schema subdirectory."""
    for entry in sorted(schema_path.iterdir()):
        if entry.is_file() and entry.suffix == ".sql":
            mv_name = entry.stem
            sql = entry.read_text().strip()
            _try_add_ivm(orchestrator, catalog_name, mv_name, sql, schema_name, errors)


def _try_add_ivm(
    orchestrator: Orchestrator,
    catalog_name: str,
    mv_name: str,
    sql: str,
    schema_name: str,
    errors: list[MVCompilationError] | None,
) -> None:
    """Try to add an IVM, collecting errors if an error list is provided."""
    try:
        orchestrator.add_ivm(catalog_name, mv_name, sql, schema=schema_name)
    except Exception as e:
        err = MVCompilationError(catalog_name, schema_name, mv_name, e)
        if errors is not None:
            errors.append(err)
        else:
            raise err from e
