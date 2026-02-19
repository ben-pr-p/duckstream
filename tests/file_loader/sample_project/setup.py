"""Sample setup.py for file loader tests."""

import os

import duckdb


def extensions(conn: duckdb.DuckDBPyConnection) -> None:
    """Install additional DuckDB extensions.

    In a real project you might install httpfs for S3-backed catalogs, etc.
    """


def secrets(conn: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB secrets.

    In a real project you might run:
        conn.execute("CREATE SECRET s3_creds (TYPE S3, ...)")
    """


def catalogs() -> list:
    """Return a list of attach functions. Each function's name is the catalog name."""
    data_dir = (
        os.environ["DUCKSTREAM_TEST_DATA_DIR"]
        if "DUCKSTREAM_TEST_DATA_DIR" in os.environ
        else "./data"
    )
    meta_path = os.path.join(data_dir, "dl_meta.ddb")
    dl_data_path = os.path.join(data_dir, "dl_data")
    os.makedirs(dl_data_path, exist_ok=True)

    def dl(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(f"ATTACH 'ducklake:{meta_path}' AS dl (DATA_PATH '{dl_data_path}')")

    return [dl]
