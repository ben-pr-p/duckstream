"""Stage 6: DISTINCT maintenance."""

from __future__ import annotations

from sqlglot import exp

from duckstream.compiler.aggregates import (
    AggInfo,
    _gen_agg_exprs,
    _gen_agg_mv_updates,
    _gen_agg_net_sql,
)
from duckstream.compiler.infrastructure import (
    _extract_output_col_names,
    _gen_changes_cte,
    _gen_set_snapshot_vars,
    _gen_update_cursor,
    _gen_where_clause,
    _repoint_columns_to_delta,
)
from duckstream.plan import Naming


def _gen_distinct_maintenance(
    ast: exp.Select,
    mv_fqn: str,
    cursors_fqn: str,
    mv_table: str,
    src: dict[str, str],
    dialect: str,
    naming: Naming,
) -> list[str]:
    """Generate maintenance SQL for SELECT DISTINCT views.

    DISTINCT is treated as GROUP BY all output columns with COUNT(*)
    tracked via _ivm_count. When multiplicity drops to 0, the row is deleted.
    """
    set_vars = _gen_set_snapshot_vars(src, cursors_fqn, mv_table)
    changes_cte = _gen_changes_cte(src)
    where_clause = _gen_where_clause(ast, dialect)

    # Group cols = all output cols; no user-visible aggregates
    out_col_names = _extract_output_col_names(ast)
    group_col_names = out_col_names
    agg_info: AggInfo = []  # No user-visible aggregates

    # Repoint projection columns to _delta
    select_exprs = [sel.copy().transform(_repoint_columns_to_delta) for sel in ast.selects]
    select_cols_sql = ", ".join(e.sql(dialect=dialect) for e in select_exprs)

    # Build aggregate expressions (only _ins_cnt / _del_cnt since agg_info is empty)
    ins_agg_parts, del_agg_parts = _gen_agg_exprs(agg_info, naming)
    ins_agg_sql = ", ".join(ins_agg_parts)
    del_agg_sql = ", ".join(del_agg_parts)

    delta_group_select = select_cols_sql

    coalesce_group = ", ".join(f"COALESCE(i.{g}, d.{g}) AS {g}" for g in group_col_names)
    net_sql = _gen_agg_net_sql(agg_info, naming)
    join_cond = " AND ".join(f"i.{g} IS NOT DISTINCT FROM d.{g}" for g in group_col_names)

    # Step 1: Create temp table with net delta aggregates
    create_delta_agg = (
        f"CREATE TEMP TABLE _delta_agg AS\n"
        f"WITH {changes_cte},\n"
        f"_ins AS (\n"
        f"    SELECT {delta_group_select}, {ins_agg_sql}\n"
        f"    FROM _changes AS _delta\n"
        f"    WHERE _delta.change_type"
        f" IN ('insert', 'update_postimage'){where_clause}\n"
        f"    GROUP BY {delta_group_select}\n"
        f"),\n"
        f"_del AS (\n"
        f"    SELECT {delta_group_select}, {del_agg_sql}\n"
        f"    FROM _changes AS _delta\n"
        f"    WHERE _delta.change_type"
        f" IN ('delete', 'update_preimage'){where_clause}\n"
        f"    GROUP BY {delta_group_select}\n"
        f")\n"
        f"SELECT {coalesce_group}, {net_sql}\n"
        f"FROM _ins i FULL OUTER JOIN _del d ON {join_cond}"
    )

    # Steps 2-5: UPDATE/INSERT/DELETE/DROP (reuse aggregate helpers)
    update_existing, insert_new, delete_empty, drop_delta = _gen_agg_mv_updates(
        agg_info, group_col_names, mv_fqn, naming
    )

    update_cursor = _gen_update_cursor(cursors_fqn, mv_table, src)

    return [
        *set_vars,
        create_delta_agg,
        update_existing,
        insert_new,
        delete_empty,
        drop_delta,
        update_cursor,
    ]
