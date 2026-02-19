"""Tests for the file loader."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import duckdb
import pytest

from duckstream.file_loader import (
    InvalidSetupError,
    MissingCatalogsError,
    MissingSetupError,
    load_directory,
)
from duckstream.orchestrator import Orchestrator

SAMPLE_PROJECT = Path(__file__).parent / "sample_project"


@pytest.fixture
def orch_env():
    """Orchestrator + DuckDB connection with env var for setup.py."""
    tmpdir = tempfile.mkdtemp()
    conn = duckdb.connect()
    o = Orchestrator(conn=conn)

    old_env = os.environ.get("DUCKSTREAM_TEST_DATA_DIR")
    os.environ["DUCKSTREAM_TEST_DATA_DIR"] = tmpdir

    yield o, conn, tmpdir

    os.environ.pop("DUCKSTREAM_TEST_DATA_DIR", None)
    if old_env is not None:
        os.environ["DUCKSTREAM_TEST_DATA_DIR"] = old_env
    conn.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_sample_project(orch_env):
    """Loading the sample project registers catalogs and MVs."""
    o, conn, _tmpdir = orch_env

    load_directory(SAMPLE_PROJECT, o)

    # Setup should have attached the "dl" catalog
    assert "dl" in o._catalogs

    # Should have 2 MVs registered
    assert len(o._mvs) == 2

    # Check the default-schema MV
    assert "dl.main.order_totals" in o._mvs

    # Check the schema-qualified MV
    assert "dl.analytics.high_value_regions" in o._mvs


def test_loaded_mvs_work_end_to_end(orch_env):
    """MVs loaded from files can be initialized and maintained."""
    o, conn, tmpdir = orch_env

    # Build a simple project with only default-schema MVs
    # (cross-schema MV references are a compiler concern, not a loader concern)
    project = Path(tempfile.mkdtemp())
    try:
        (project / "setup.py").write_text(
            f"import os\n"
            f"def catalogs():\n"
            f"    meta = os.path.join('{tmpdir}', 'e2e_meta.ddb')\n"
            f"    data = os.path.join('{tmpdir}', 'e2e_data')\n"
            f"    os.makedirs(data, exist_ok=True)\n"
            f"    def e2e(conn):\n"
            f'        dsn = f"ducklake:{{meta}}"\n'
            f"        sql = f\"ATTACH '{{dsn}}' AS e2e"
            f" (DATA_PATH '{{data}}')\"\n"
            f"        conn.execute(sql)\n"
            f"    return [e2e]\n"
        )
        cat_dir = project / "catalogs" / "e2e"
        cat_dir.mkdir(parents=True)
        (cat_dir / "order_totals.sql").write_text(
            "SELECT region, SUM(amount) AS total FROM orders GROUP BY region"
        )

        load_directory(project, o)

        conn.execute("CREATE TABLE e2e.orders (id INTEGER, amount INTEGER, region TEXT)")
        conn.execute("INSERT INTO e2e.orders VALUES (1, 100, 'east'), (2, 200, 'west')")

        o.initialize()

        rows = conn.execute(
            "SELECT region, total FROM e2e.main.order_totals ORDER BY region"
        ).fetchall()
        assert rows == [("east", 100), ("west", 200)]

        # Make a change and maintain
        conn.execute("INSERT INTO e2e.orders VALUES (3, 300, 'east')")
        o.maintain()

        rows = conn.execute(
            "SELECT region, total FROM e2e.main.order_totals ORDER BY region"
        ).fetchall()
        assert rows == [("east", 400), ("west", 200)]
    finally:
        shutil.rmtree(project, ignore_errors=True)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_setup_py():
    """Error when setup.py is absent."""
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, "catalogs"))
    try:
        o = Orchestrator(conn=duckdb.connect())
        with pytest.raises(MissingSetupError):
            load_directory(tmpdir, o)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_missing_catalogs_dir():
    """Error when catalogs/ directory is absent."""
    tmpdir = tempfile.mkdtemp()
    Path(tmpdir, "setup.py").write_text("def catalogs(): return []\n")
    try:
        o = Orchestrator(conn=duckdb.connect())
        with pytest.raises(MissingCatalogsError):
            load_directory(tmpdir, o)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_setup_py_missing_catalogs_function():
    """Error when setup.py doesn't export `catalogs`."""
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, "catalogs"))
    Path(tmpdir, "setup.py").write_text("x = 42\n")
    try:
        o = Orchestrator(conn=duckdb.connect())
        with pytest.raises(InvalidSetupError, match="must export a `catalogs` function"):
            load_directory(tmpdir, o)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_non_sql_files_ignored(orch_env):
    """Non-.sql files in catalog directories are ignored."""
    o, conn, _tmpdir = orch_env

    # Create a temp project with a non-sql file
    tmpdir = tempfile.mkdtemp()
    try:
        Path(tmpdir, "setup.py").write_text("def catalogs(): return []\n")
        cat_dir = Path(tmpdir, "catalogs", "dl")
        cat_dir.mkdir(parents=True)
        (cat_dir / "README.md").write_text("not a view")
        (cat_dir / "notes.txt").write_text("also not a view")

        load_directory(tmpdir, o)
        assert len(o._mvs) == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
