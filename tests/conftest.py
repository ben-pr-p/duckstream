import os
import shutil
import tempfile

import duckdb
import pytest


DUCKLAKE_CATALOG = "dl"


@pytest.fixture
def ducklake():
    """Fresh isolated DuckLake instance per test.

    Creates a random temp directory for the metadata catalog and data files.
    Attaches the DuckLake catalog as 'dl'. The MV and any non-DuckLake tables
    live in the default 'memory' catalog.

    Yields a (connection, catalog_name) tuple.
    """
    tmpdir = tempfile.mkdtemp()
    meta_path = os.path.join(tmpdir, "meta.ddb")
    data_path = os.path.join(tmpdir, "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL ducklake")
    con.execute("LOAD ducklake")
    con.execute(
        f"ATTACH 'ducklake:{meta_path}' AS {DUCKLAKE_CATALOG} "
        f"(DATA_PATH '{data_path}')"
    )

    yield con, DUCKLAKE_CATALOG

    con.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
