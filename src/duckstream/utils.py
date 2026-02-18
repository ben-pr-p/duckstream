"""Utility functions for IVM maintenance operations."""

from __future__ import annotations

from duckstream.plan import Naming


def safe_to_expire_sql(
    *,
    mv_catalogs: list[str],
    source_catalogs: list[str],
    naming: Naming | None = None,
) -> str:
    """Generate SQL to report which snapshots are safe to expire.

    Scans the _ivm_cursors table in each MV catalog and computes, for each
    source catalog, the minimum snapshot that is still required by any MV.
    Snapshots before that threshold are safe to expire.

    Returns a single SQL query that produces rows of:
        (source_catalog, min_required_snapshot, latest_snapshot)
    """
    naming = naming or Naming()
    cursors_table = naming.cursors_table()

    if not mv_catalogs or not source_catalogs:
        # Return a query that produces empty results with correct schema
        return (
            "SELECT "
            "CAST(NULL AS VARCHAR) AS source_catalog, "
            "CAST(NULL AS BIGINT) AS min_required_snapshot, "
            "CAST(NULL AS BIGINT) AS latest_snapshot "
            "WHERE FALSE"
        )

    # Build a UNION ALL across all MV catalogs' cursor tables
    cursor_unions = []
    for mv_cat in mv_catalogs:
        cursor_unions.append(
            f"SELECT source_catalog, MIN(last_snapshot) AS min_snap\n"
            f"FROM {mv_cat}.main.{cursors_table}\n"
            f"GROUP BY source_catalog"
        )

    all_cursors = "\nUNION ALL\n".join(cursor_unions)

    # Filter to requested source catalogs
    source_list = ", ".join(f"'{s}'" for s in source_catalogs)

    return (
        f"WITH _all_cursors AS (\n"
        f"    {all_cursors}\n"
        f"),\n"
        f"_min_per_source AS (\n"
        f"    SELECT source_catalog, MIN(min_snap) AS min_required_snapshot\n"
        f"    FROM _all_cursors\n"
        f"    WHERE source_catalog IN ({source_list})\n"
        f"    GROUP BY source_catalog\n"
        f")\n"
        f"SELECT\n"
        f"    m.source_catalog,\n"
        f"    m.min_required_snapshot,\n"
        f"    (SELECT MAX(snapshot_id)"
        f" FROM ducklake_snapshots(m.source_catalog)) AS latest_snapshot\n"
        f"FROM _min_per_source m\n"
        f"ORDER BY m.source_catalog"
    )


def pending_maintenance_sql(
    *,
    mv_catalogs: list[str],
    naming: Naming | None = None,
) -> str:
    """Generate SQL to report pending maintenance work per MV.

    Scans each MV catalog's _ivm_cursors table and counts pending changes
    and unprocessed snapshots for each MV + source catalog combination.

    Returns a single SQL query that produces rows of:
        (mv_catalog, mv_name, source_catalog, pending_snapshots)

    Note: pending_snapshots is the number of unprocessed snapshots. For the
    exact number of pending row changes, use ducklake_table_changes() directly
    (counting rows is expensive for large change sets).
    """
    naming = naming or Naming()
    cursors_table = naming.cursors_table()

    if not mv_catalogs:
        return (
            "SELECT "
            "CAST(NULL AS VARCHAR) AS mv_catalog, "
            "CAST(NULL AS VARCHAR) AS mv_name, "
            "CAST(NULL AS VARCHAR) AS source_catalog, "
            "CAST(NULL AS BIGINT) AS pending_snapshots "
            "WHERE FALSE"
        )

    # Build a UNION ALL across all MV catalogs
    parts = []
    for mv_cat in mv_catalogs:
        parts.append(
            f"SELECT\n"
            f"    '{mv_cat}' AS mv_catalog,\n"
            f"    c.mv_name,\n"
            f"    c.source_catalog,\n"
            f"    (SELECT MAX(snapshot_id)"
            f" FROM ducklake_snapshots(c.source_catalog))"
            f" - c.last_snapshot AS pending_snapshots\n"
            f"FROM {mv_cat}.main.{cursors_table} c"
        )

    return "\nUNION ALL\n".join(parts) + "\nORDER BY mv_catalog, mv_name, source_catalog"
