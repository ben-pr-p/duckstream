"""Stage 2: Aggregate maintenance (GROUP BY + COUNT/SUM/AVG/MIN/MAX)."""

from __future__ import annotations

from sqlglot import exp

from duckstream.compiler.infrastructure import (
    _gen_changes_cte,
    _gen_set_snapshot_vars,
    _gen_update_cursor,
    _gen_where_clause,
)
from duckstream.materialized_view import Naming

AggInfo = list[tuple[str, str, str | None, bool]]
# Each entry: (alias, agg_type, inner_col_name, is_avg)
# agg_type is one of: "SUM", "COUNT_STAR", "COUNT_COL", "AVG", "MIN", "MAX"


def _analyze_aggregates(ast: exp.Select, dialect: str) -> tuple[list[str], AggInfo]:
    """Extract group column names and aggregate info from AST."""
    group_node = ast.args.get("group")
    group_exprs = group_node.expressions if group_node else []
    group_col_names = []
    for g in group_exprs:
        if isinstance(g, exp.Column):
            group_col_names.append(g.name)
        else:
            group_col_names.append(g.sql(dialect=dialect))

    agg_info: AggInfo = []
    for sel in ast.selects:
        alias_name = sel.alias if isinstance(sel, exp.Alias) else None
        inner = sel.this if isinstance(sel, exp.Alias) else sel
        if isinstance(inner, exp.Sum):
            col_name = inner.this.name if isinstance(inner.this, exp.Column) else inner.this.sql()
            agg_info.append((alias_name or col_name, "SUM", col_name, False))
        elif isinstance(inner, exp.Count):
            if isinstance(inner.this, exp.Star):
                agg_info.append((alias_name or "count", "COUNT_STAR", None, False))
            else:
                col_name = (
                    inner.this.name if isinstance(inner.this, exp.Column) else inner.this.sql()
                )
                agg_info.append((alias_name or col_name, "COUNT_COL", col_name, False))
        elif isinstance(inner, exp.Avg):
            col_name = inner.this.name if isinstance(inner.this, exp.Column) else inner.this.sql()
            agg_info.append((alias_name or col_name, "AVG", col_name, True))
        elif isinstance(inner, exp.Min):
            col_name = inner.this.name if isinstance(inner.this, exp.Column) else inner.this.sql()
            agg_info.append((alias_name or col_name, "MIN", col_name, False))
        elif isinstance(inner, exp.Max):
            col_name = inner.this.name if isinstance(inner.this, exp.Column) else inner.this.sql()
            agg_info.append((alias_name or col_name, "MAX", col_name, False))

    return group_col_names, agg_info


def _gen_agg_exprs(
    agg_info: AggInfo, naming: Naming, src_alias: str = "_delta"
) -> tuple[list[str], list[str]]:
    """Build insert/delete aggregate SQL expressions.

    Returns (ins_agg_parts, del_agg_parts).
    """
    ins_parts: list[str] = []
    del_parts: list[str] = []
    for alias, agg_type, col_name, _is_avg in agg_info:
        if agg_type == "SUM":
            ins_parts.append(f"SUM({src_alias}.{col_name}) AS _ins_{alias}")
            del_parts.append(f"SUM({src_alias}.{col_name}) AS _del_{alias}")
        elif agg_type == "COUNT_STAR":
            ins_parts.append(f"COUNT(*) AS _ins_{alias}")
            del_parts.append(f"COUNT(*) AS _del_{alias}")
        elif agg_type == "COUNT_COL":
            ins_parts.append(f"COUNT({src_alias}.{col_name}) AS _ins_{alias}")
            del_parts.append(f"COUNT({src_alias}.{col_name}) AS _del_{alias}")
        elif agg_type == "AVG":
            sum_col = naming.aux_column(f"sum_{alias}")
            ins_parts.append(f"SUM({src_alias}.{col_name}) AS _ins_{sum_col}")
            del_parts.append(f"SUM({src_alias}.{col_name}) AS _del_{sum_col}")
        elif agg_type == "MIN":
            ins_parts.append(f"MIN({src_alias}.{col_name}) AS _ins_{alias}")
            del_parts.append(f"MIN({src_alias}.{col_name}) AS _del_{alias}")
        elif agg_type == "MAX":
            ins_parts.append(f"MAX({src_alias}.{col_name}) AS _ins_{alias}")
            del_parts.append(f"MAX({src_alias}.{col_name}) AS _del_{alias}")
    ins_parts.append("COUNT(*) AS _ins_cnt")
    del_parts.append("COUNT(*) AS _del_cnt")
    return ins_parts, del_parts


def _gen_agg_net_sql(agg_info: AggInfo, naming: Naming) -> str:
    """Build COALESCE net expressions for the delta_agg SELECT.

    For SUM/COUNT/AVG: computes _net_X = ins - del.
    For MIN/MAX: passes through _ins_X and _del_X separately (no subtraction).
    """
    net_parts = []
    for alias, agg_type, _col_name, _is_avg in agg_info:
        if agg_type in ("SUM", "COUNT_STAR", "COUNT_COL"):
            net_parts.append(
                f"COALESCE(i._ins_{alias}, 0) - COALESCE(d._del_{alias}, 0) AS _net_{alias}"
            )
        elif agg_type == "AVG":
            sum_col = naming.aux_column(f"sum_{alias}")
            net_parts.append(
                f"COALESCE(i._ins_{sum_col}, 0) - COALESCE(d._del_{sum_col}, 0) AS _net_{sum_col}"
            )
        elif agg_type in ("MIN", "MAX"):
            net_parts.append(f"i._ins_{alias} AS _ins_{alias}")
            net_parts.append(f"d._del_{alias} AS _del_{alias}")
    net_parts.append("COALESCE(i._ins_cnt, 0) - COALESCE(d._del_cnt, 0) AS _net_count")
    return ", ".join(net_parts)


def _gen_agg_mv_updates(
    agg_info: AggInfo,
    group_col_names: list[str],
    mv_fqn: str,
    naming: Naming,
    src_fqn: str | None = None,
) -> tuple[str, str, str, str]:
    """Generate UPDATE/INSERT/DELETE/DROP for aggregate MV maintenance.

    Assumes _delta_agg temp table exists with _net_* columns.
    For MIN/MAX, src_fqn is required for rescan fallback.
    Returns (update_existing, insert_new, delete_empty, drop_delta).
    """
    ivm_count_col = naming.aux_column("count")

    # UPDATE existing groups
    update_sets = []
    for alias, agg_type, col_name, _is_avg in agg_info:
        if agg_type in ("SUM", "COUNT_STAR", "COUNT_COL"):
            update_sets.append(f"{alias} = mv.{alias} + d._net_{alias}")
        elif agg_type == "AVG":
            sum_col = naming.aux_column(f"sum_{alias}")
            update_sets.append(f"{sum_col} = mv.{sum_col} + d._net_{sum_col}")
            update_sets.append(
                f"{alias} = CAST(mv.{sum_col} + d._net_{sum_col} AS DOUBLE) "
                f"/ (mv.{ivm_count_col} + d._net_count)"
            )
        elif agg_type == "MIN":
            assert src_fqn is not None, "src_fqn required for MIN rescan"
            group_match = " AND ".join(
                f"src.{g} IS NOT DISTINCT FROM mv.{g}" for g in group_col_names
            )
            update_sets.append(
                f"{alias} = CASE"
                f" WHEN d._del_{alias} IS NOT NULL"
                f" AND d._del_{alias} IS NOT DISTINCT FROM mv.{alias}"
                f" THEN (SELECT MIN({col_name}) FROM {src_fqn} src"
                f" WHERE {group_match})"
                f" ELSE LEAST(mv.{alias}, COALESCE(d._ins_{alias}, mv.{alias}))"
                f" END"
            )
        elif agg_type == "MAX":
            assert src_fqn is not None, "src_fqn required for MAX rescan"
            group_match = " AND ".join(
                f"src.{g} IS NOT DISTINCT FROM mv.{g}" for g in group_col_names
            )
            update_sets.append(
                f"{alias} = CASE"
                f" WHEN d._del_{alias} IS NOT NULL"
                f" AND d._del_{alias} IS NOT DISTINCT FROM mv.{alias}"
                f" THEN (SELECT MAX({col_name}) FROM {src_fqn} src"
                f" WHERE {group_match})"
                f" ELSE GREATEST(mv.{alias}, COALESCE(d._ins_{alias}, mv.{alias}))"
                f" END"
            )
    update_sets.append(f"{ivm_count_col} = mv.{ivm_count_col} + d._net_count")
    update_sets_sql = ", ".join(update_sets)
    update_where = " AND ".join(f"mv.{g} IS NOT DISTINCT FROM d.{g}" for g in group_col_names)
    update_existing = (
        f"UPDATE {mv_fqn} AS mv\nSET {update_sets_sql}\nFROM _delta_agg d\nWHERE {update_where}"
    )

    # INSERT new groups
    insert_cols = list(group_col_names)
    insert_vals = [f"d.{g}" for g in group_col_names]
    for alias, agg_type, _col_name, _is_avg in agg_info:
        if agg_type in ("SUM", "COUNT_STAR", "COUNT_COL"):
            insert_cols.append(alias)
            insert_vals.append(f"d._net_{alias}")
        elif agg_type == "AVG":
            sum_col = naming.aux_column(f"sum_{alias}")
            insert_cols.append(sum_col)
            insert_vals.append(f"d._net_{sum_col}")
            insert_cols.append(alias)
            insert_vals.append(f"CAST(d._net_{sum_col} AS DOUBLE) / d._net_count")
        elif agg_type in ("MIN", "MAX"):
            insert_cols.append(alias)
            insert_vals.append(f"d._ins_{alias}")
    insert_cols.append(ivm_count_col)
    insert_vals.append("d._net_count")
    not_exists_where = " AND ".join(f"mv.{g} IS NOT DISTINCT FROM d.{g}" for g in group_col_names)
    insert_new = (
        f"INSERT INTO {mv_fqn} ({', '.join(insert_cols)})\n"
        f"SELECT {', '.join(insert_vals)}\n"
        f"FROM _delta_agg d\n"
        f"WHERE NOT EXISTS (\n"
        f"    SELECT 1 FROM {mv_fqn} mv WHERE {not_exists_where}\n"
        f")\n"
        f"AND d._net_count > 0"
    )

    # DELETE emptied groups
    delete_empty = f"DELETE FROM {mv_fqn} WHERE {ivm_count_col} <= 0"

    # DROP temp table
    drop_delta = "DROP TABLE IF EXISTS _delta_agg"

    return update_existing, insert_new, delete_empty, drop_delta


def _gen_aggregate_maintenance(
    ast: exp.Select,
    mv_fqn: str,
    cursors_fqn: str,
    mv_table: str,
    src: dict[str, str],
    dialect: str,
    naming: Naming,
) -> list[str]:
    set_vars = _gen_set_snapshot_vars(src, cursors_fqn, mv_table)
    changes_cte = _gen_changes_cte(src)
    where_clause = _gen_where_clause(ast, dialect)

    group_col_names, agg_info = _analyze_aggregates(ast, dialect)
    ins_agg_parts, del_agg_parts = _gen_agg_exprs(agg_info, naming)

    ins_agg_sql = ", ".join(ins_agg_parts)
    del_agg_sql = ", ".join(del_agg_parts)
    delta_group_cols = ", ".join(f"_delta.{g}" for g in group_col_names)
    coalesce_group = ", ".join(f"COALESCE(i.{g}, d.{g}) AS {g}" for g in group_col_names)
    net_sql = _gen_agg_net_sql(agg_info, naming)
    join_cond = " AND ".join(f"i.{g} IS NOT DISTINCT FROM d.{g}" for g in group_col_names)

    # Step 1: Create temp table with net delta aggregates
    create_delta_agg = (
        f"CREATE TEMP TABLE _delta_agg AS\n"
        f"WITH {changes_cte},\n"
        f"_ins AS (\n"
        f"    SELECT {delta_group_cols}, {ins_agg_sql}\n"
        f"    FROM _changes AS _delta\n"
        f"    WHERE _delta.change_type"
        f" IN ('insert', 'update_postimage'){where_clause}\n"
        f"    GROUP BY {delta_group_cols}\n"
        f"),\n"
        f"_del AS (\n"
        f"    SELECT {delta_group_cols}, {del_agg_sql}\n"
        f"    FROM _changes AS _delta\n"
        f"    WHERE _delta.change_type"
        f" IN ('delete', 'update_preimage'){where_clause}\n"
        f"    GROUP BY {delta_group_cols}\n"
        f")\n"
        f"SELECT {coalesce_group}, {net_sql}\n"
        f"FROM _ins i FULL OUTER JOIN _del d ON {join_cond}"
    )

    # Steps 2-5: UPDATE/INSERT/DELETE/DROP
    src_fqn = f"{src['catalog']}.{src['schema']}.{src['table']}"
    update_existing, insert_new, delete_empty, drop_delta = _gen_agg_mv_updates(
        agg_info, group_col_names, mv_fqn, naming, src_fqn=src_fqn
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
