"""Stage 1: SELECT / PROJECT / WHERE maintenance."""

from __future__ import annotations

from sqlglot import exp

from duckstream.compiler.infrastructure import (
    _extract_output_col_names,
    _gen_changes_cte,
    _gen_set_snapshot_vars,
    _gen_update_cursor,
    _gen_where_clause,
    _repoint_columns_to_delta,
)


def _gen_select_maintenance(
    ast: exp.Select,
    mv_fqn: str,
    cursors_fqn: str,
    mv_table: str,
    src: dict[str, str],
    dialect: str,
) -> list[str]:
    set_vars = _gen_set_snapshot_vars(src, cursors_fqn, mv_table)
    changes_cte = _gen_changes_cte(src)
    where_clause = _gen_where_clause(ast, dialect)

    # Repoint projection columns to _delta
    select_exprs = [sel.copy().transform(_repoint_columns_to_delta) for sel in ast.selects]
    select_cols_sql = ", ".join(e.sql(dialect=dialect) for e in select_exprs)

    # Output column names (for matching in DELETE)
    out_col_names = _extract_output_col_names(ast)

    # --- DELETE removed rows ---
    all_proj_cols = ", ".join(out_col_names)

    # Build join condition for ROW_NUMBER matching (handle NULLs)
    join_on_parts = [f"m.{c} IS NOT DISTINCT FROM d.{c}" for c in out_col_names]
    join_on = " AND ".join(join_on_parts)

    delete_sql = (
        f"WITH {changes_cte},\n"
        f"_deletes AS (\n"
        f"    SELECT {select_cols_sql},\n"
        f"           ROW_NUMBER() OVER (\n"
        f"               PARTITION BY {all_proj_cols}"
        f" ORDER BY (SELECT NULL)) AS _dn\n"
        f"    FROM _changes AS _delta\n"
        f"    WHERE _delta.change_type"
        f" IN ('delete', 'update_preimage'){where_clause}\n"
        f"),\n"
        f"_mv_numbered AS (\n"
        f"    SELECT rowid AS _rid, {all_proj_cols},\n"
        f"           ROW_NUMBER() OVER ("
        f"PARTITION BY {all_proj_cols} ORDER BY rowid) AS _mn\n"
        f"    FROM {mv_fqn}\n"
        f")\n"
        f"DELETE FROM {mv_fqn}\n"
        f"WHERE rowid IN (\n"
        f"    SELECT _rid FROM _mv_numbered m\n"
        f"    JOIN _deletes d ON {join_on} AND m._mn = d._dn\n"
        f")"
    )

    # --- INSERT new rows ---
    col_list = ", ".join(out_col_names)
    insert_sql = (
        f"WITH {changes_cte}\n"
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT {select_cols_sql}\n"
        f"FROM _changes AS _delta\n"
        f"WHERE _delta.change_type"
        f" IN ('insert', 'update_postimage'){where_clause}"
    )

    # --- Update cursor ---
    update_cursor = _gen_update_cursor(cursors_fqn, mv_table, src)

    return [*set_vars, delete_sql, insert_sql, update_cursor]
