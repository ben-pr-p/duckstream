"""Full-refresh compiler: recompute + bag-diff maintenance strategy.

Used as a fallback when incremental IVM compilation is not supported.
Recomputes the entire view into a shadow table, then diffs against the MV
using INSERT/DELETE so DuckLake only records actual row-level changes.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from duckstream.compiler.infrastructure import (
    _extract_output_col_names,
    _gen_create_cursors,
    _gen_init_cursor,
    _resolve_source,
)
from duckstream.materialized_view import MaterializedView, Naming


def compile_full_refresh(
    view_sql: str,
    *,
    naming: Naming | None = None,
    mv_catalog: str = "dl",
    mv_schema: str = "main",
    sources: dict[str, dict] | None = None,
) -> MaterializedView:
    """Compile a view into a full-refresh + bag-diff maintenance plan."""
    dialect = "duckdb"
    naming = naming or Naming()
    parsed = sqlglot.parse_one(view_sql, dialect=dialect)

    mv_table = naming.mv_table()
    mv_fqn = f"{mv_catalog}.{mv_schema}.{mv_table}"
    cursors_fqn = f"{mv_catalog}.{mv_schema}.{naming.cursors_table()}"

    # Discover all base tables (including in subqueries, JOINs)
    all_tables = list(parsed.find_all(exp.Table))
    seen: set[str] = set()
    unique_tables: list[exp.Table] = []
    for t in all_tables:
        if t.name not in seen:
            seen.add(t.name)
            unique_tables.append(t)

    # Resolve each table's catalog/schema
    table_sources: dict[str, dict[str, str]] = {}
    for t in unique_tables:
        table_sources[t.name] = _resolve_source(t.name, sources, mv_catalog, mv_schema)

    # Qualify all table references in the view SQL
    qualified_ast = parsed.copy().transform(
        lambda node: (
            exp.table_(
                node.name,
                db=table_sources[node.name]["schema"],
                catalog=table_sources[node.name]["catalog"],
                alias=node.alias_or_name if node.alias else None,
            )
            if isinstance(node, exp.Table) and node.name in table_sources
            else node
        )
    )
    qualified_sql = qualified_ast.sql(dialect=dialect)

    # Extract output column names from the original AST
    # For set operations, use the left branch's columns
    if isinstance(parsed, exp.Select):
        col_names = _extract_output_col_names(parsed)
    else:
        # For UNION/EXCEPT/INTERSECT, find the leftmost SELECT
        left = parsed
        while not isinstance(left, exp.Select):
            left = left.this
        col_names = _extract_output_col_names(left)

    # Build base_tables dict: table_name -> catalog
    base_tables: dict[str, str] = {}
    for tname, src in table_sources.items():
        base_tables[tname] = src["catalog"]

    # --- DDL ---
    create_cursors = _gen_create_cursors(cursors_fqn)
    create_mv = f"CREATE TABLE {mv_fqn} AS {qualified_sql}"
    query_mv = f"SELECT * FROM {mv_fqn}"

    # Initialize cursors (one per source catalog)
    init_cursors: list[str] = []
    seen_catalogs: set[str] = set()
    for _tname, src in table_sources.items():
        cat = src["catalog"]
        if cat not in seen_catalogs:
            seen_catalogs.add(cat)
            init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, src))

    # --- Maintenance SQL (bag-diff) ---
    maintain = _gen_full_refresh_maintain(
        qualified_sql=qualified_sql,
        mv_fqn=mv_fqn,
        mv_table=mv_table,
        cursors_fqn=cursors_fqn,
        col_names=col_names,
        table_sources=table_sources,
    )

    features: set[str] = {"full_refresh"}

    return MaterializedView(
        view_sql=view_sql,
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
        query_mv=query_mv,
        strategy="full_refresh",
    )


def _gen_full_refresh_maintain(
    *,
    qualified_sql: str,
    mv_fqn: str,
    mv_table: str,
    cursors_fqn: str,
    col_names: list[str],
    table_sources: dict[str, dict[str, str]],
) -> list[str]:
    """Generate the bag-diff maintenance SQL statements."""
    stmts: list[str] = []
    shadow_name = f"_shadow_{mv_table}"
    all_cols = ", ".join(col_names)
    first_col = col_names[0]

    # Build IS NOT DISTINCT FROM join conditions
    join_conds = " AND ".join(f"m.{col} IS NOT DISTINCT FROM s.{col}" for col in col_names)

    # 1. Set snapshot end vars (one per source catalog)
    seen_catalogs: set[str] = set()
    for _tname, src in table_sources.items():
        cat = src["catalog"]
        if cat not in seen_catalogs:
            seen_catalogs.add(cat)
            stmts.append(
                f"SET VARIABLE _ivm_snap_end_{cat} = (\n"
                f"    SELECT MAX(snapshot_id) FROM ducklake_snapshots('{cat}')\n"
                f")"
            )

    # 2. Recompute into temp shadow table
    stmts.append(f"CREATE OR REPLACE TEMP TABLE {shadow_name} AS {qualified_sql}")

    # 3. DELETE rows from MV that are excess vs shadow
    s_cols = ", ".join(f"s.{col}" for col in col_names)

    stmts.append(
        f"DELETE FROM {mv_fqn} WHERE rowid IN (\n"
        f"    SELECT m._rid FROM (\n"
        f"        SELECT rowid AS _rid, {all_cols},\n"
        f"               ROW_NUMBER() OVER (PARTITION BY {all_cols} ORDER BY rowid) AS _rn\n"
        f"        FROM {mv_fqn}\n"
        f"    ) m\n"
        f"    LEFT JOIN (\n"
        f"        SELECT {all_cols},\n"
        f"               ROW_NUMBER() OVER (PARTITION BY {all_cols}"
        f" ORDER BY (SELECT NULL)) AS _rn\n"
        f"        FROM {shadow_name}\n"
        f"    ) s\n"
        f"        ON {join_conds} AND m._rn = s._rn\n"
        f"    WHERE s.{first_col} IS NULL AND m._rid IS NOT NULL\n"
        f")"
    )

    # 4. INSERT rows from shadow that are excess vs MV
    stmts.append(
        f"INSERT INTO {mv_fqn} ({all_cols})\n"
        f"SELECT {s_cols} FROM (\n"
        f"    SELECT {all_cols},\n"
        f"           ROW_NUMBER() OVER (PARTITION BY {all_cols}"
        f" ORDER BY (SELECT NULL)) AS _rn\n"
        f"    FROM {shadow_name}\n"
        f") s\n"
        f"LEFT JOIN (\n"
        f"    SELECT {all_cols},\n"
        f"           ROW_NUMBER() OVER (PARTITION BY {all_cols} ORDER BY rowid) AS _rn\n"
        f"    FROM {mv_fqn}\n"
        f") m\n"
        f"    ON {join_conds} AND s._rn = m._rn\n"
        f"WHERE m.{first_col} IS NULL"
    )

    # 5. Drop shadow
    stmts.append(f"DROP TABLE IF EXISTS {shadow_name}")

    # 6. Update cursors (one per source catalog)
    seen_catalogs2: set[str] = set()
    for _tname, src in table_sources.items():
        cat = src["catalog"]
        if cat not in seen_catalogs2:
            seen_catalogs2.add(cat)
            stmts.append(
                f"UPDATE {cursors_fqn}\n"
                f"SET last_snapshot = getvariable('_ivm_snap_end_{cat}')\n"
                f"WHERE mv_name = '{mv_table}' AND source_catalog = '{cat}'"
            )

    return stmts
