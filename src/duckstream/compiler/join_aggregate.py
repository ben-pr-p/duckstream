"""Stage 5: JOIN + Aggregates (Composed)."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from duckstream.compiler.aggregates import (
    _analyze_aggregates,
    _gen_agg_exprs,
    _gen_agg_mv_updates,
    _gen_agg_net_sql,
)
from duckstream.compiler.infrastructure import (
    _detect_features,
    _gen_changes_cte_named,
    _gen_create_mv_join,
    _gen_init_cursor,
    _gen_query_mv,
    _gen_set_snapshot_vars_named,
    _gen_update_cursor,
    _resolve_source,
)
from duckstream.plan import IVMPlan, Naming, UnsupportedSQLError


def _compile_join_aggregate(
    ast: exp.Select,
    tables: list[exp.Table],
    joins: list,
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
    mv_table: str,
    mv_fqn: str,
    cursors_fqn: str,
    create_cursors: str,
) -> IVMPlan:
    """Compile a two-table inner join + aggregate view.

    Composes join delta decomposition with aggregate maintenance:
    1. Compute join delta rows (inserts/deletes) via three-way decomposition
    2. Aggregate those delta rows per group
    3. Apply net deltas to MV using same UPDATE/INSERT/DELETE as Stage 2
    """
    if len(joins) != 1:
        raise UnsupportedSQLError("multi_join", "Only two-table joins supported")

    # MIN/MAX with JOIN is not yet supported (rescan would need to join both tables)
    for agg in ast.find_all(exp.AggFunc):
        if isinstance(agg, (exp.Min, exp.Max)):
            raise UnsupportedSQLError("join_min_max", "MIN/MAX with JOIN is not yet supported")

    join_node = joins[0]
    left_table_node = ast.args["from_"].this
    right_table_node = join_node.this
    left_name = left_table_node.name
    right_name = right_table_node.name

    left_src = _resolve_source(left_name, sources, mv_catalog, mv_schema)
    right_src = _resolve_source(right_name, sources, mv_catalog, mv_schema)

    table_sources = {left_name: left_src, right_name: right_src}
    left_fqn = f"{left_src['catalog']}.{left_src['schema']}.{left_name}"
    right_fqn = f"{right_src['catalog']}.{right_src['schema']}.{right_name}"

    on_condition = join_node.args.get("on")

    # --- Analyze aggregates ---
    group_col_names, agg_info = _analyze_aggregates(ast, dialect)

    # Collect the "inner" columns needed before aggregation:
    # group columns + columns referenced inside aggregate functions.
    inner_col_refs: list[str] = []
    seen: set[str] = set()
    # Group columns (table-qualified)
    group_node = ast.args.get("group")
    if group_node:
        for g in group_node.expressions:
            sql = g.sql(dialect=dialect)
            if sql not in seen:
                inner_col_refs.append(sql)
                seen.add(sql)
    # Columns inside aggregates
    for _alias, _agg_type, col_name, _is_avg in agg_info:
        if col_name is not None:
            # Find the original column expression to preserve table qualification
            for sel in ast.selects:
                inner = sel.this if isinstance(sel, exp.Alias) else sel
                if isinstance(inner, (exp.Sum, exp.Count, exp.Avg)) and not isinstance(
                    inner.this, exp.Star
                ):
                    inner_col = inner.this
                    if isinstance(inner_col, exp.Column) and inner_col.name == col_name:
                        qualified = inner_col.sql(dialect=dialect)
                        if qualified not in seen:
                            inner_col_refs.append(qualified)
                            seen.add(qualified)

    inner_proj_sql = ", ".join(inner_col_refs) if inner_col_refs else "*"

    # --- DDL ---
    create_mv = _gen_create_mv_join(
        ast, mv_fqn, table_sources, dialect, has_agg=True, naming=naming
    )
    init_cursors = [_gen_init_cursor(cursors_fqn, mv_table, left_src)]
    if left_src["catalog"] != right_src["catalog"]:
        init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, right_src))

    # --- Snapshot vars and changes CTEs ---
    set_vars_left = _gen_set_snapshot_vars_named(left_src, cursors_fqn, mv_table, "l")
    set_vars_right = _gen_set_snapshot_vars_named(right_src, cursors_fqn, mv_table, "r")
    changes_l = _gen_changes_cte_named(left_src, "_changes_l", "l")
    changes_r = _gen_changes_cte_named(right_src, "_changes_r", "r")

    # --- Rewrite helpers (same as _compile_join) ---
    def _rewrite_for_alias(node: exp.Expression, delta_alias: str, table_name: str):
        def _replace(n: exp.Expression) -> exp.Expression:
            if isinstance(n, exp.Column) and n.table == table_name:
                return exp.column(n.name, table=delta_alias)
            return n

        return node.copy().transform(_replace)

    def _rewrite_proj_str(proj: str, delta_alias: str, table_name: str) -> str:
        parts = []
        for p in proj.split(","):
            parsed = sqlglot.parse_one(p.strip(), dialect=dialect)
            rewritten = _rewrite_for_alias(parsed, delta_alias, table_name)
            parts.append(rewritten.sql(dialect=dialect))
        return ", ".join(parts)

    on_delta_l = _rewrite_for_alias(on_condition, "_dl", left_name)
    on_delta_r = _rewrite_for_alias(on_condition, "_dr", right_name)
    on_cross = _rewrite_for_alias(on_delta_l, "_dr", right_name)

    on_dl_sql = on_delta_l.sql(dialect=dialect)
    on_dr_sql = on_delta_r.sql(dialect=dialect)
    on_cross_sql = on_cross.sql(dialect=dialect)

    # Rewrite inner projection for each delta term
    proj_dl_s = _rewrite_proj_str(inner_proj_sql, "_dl", left_name)
    proj_r_dr = _rewrite_proj_str(inner_proj_sql, "_dr", right_name)
    proj_cross = _rewrite_proj_str(
        _rewrite_proj_str(inner_proj_sql, "_dl", left_name),
        "_dr",
        right_name,
    )

    # Old table state for deletes
    right_old = f"(SELECT * FROM {right_fqn} AT (VERSION => getvariable('_ivm_snap_start_r') - 1))"
    left_old = f"(SELECT * FROM {left_fqn} AT (VERSION => getvariable('_ivm_snap_start_l') - 1))"

    # --- Build aggregate delta table ---
    # The join delta produces raw rows; we aggregate them per group.
    # Group col references need to be unqualified in the _join_ins/_join_del output
    # since they'll be aggregated in _ins_agg/_del_agg.

    # Output column aliases for the join delta (strip table qualification)
    inner_aliases: list[str] = []
    for ref in inner_col_refs:
        parsed = sqlglot.parse_one(ref, dialect=dialect)
        if isinstance(parsed, exp.Column):
            inner_aliases.append(parsed.name)
        else:
            inner_aliases.append(ref)

    # Build aliased projections for join delta output
    def _alias_proj(proj: str, aliases: list[str]) -> str:
        parts = proj.split(",")
        result = []
        for p, alias in zip(parts, aliases, strict=True):
            result.append(f"{p.strip()} AS {alias}")
        return ", ".join(result)

    proj_dl_s_aliased = _alias_proj(proj_dl_s, inner_aliases)
    proj_r_dr_aliased = _alias_proj(proj_r_dr, inner_aliases)
    proj_cross_aliased = _alias_proj(proj_cross, inner_aliases)

    # Aggregate expressions using unqualified inner aliases as source
    ins_agg_parts, del_agg_parts = _gen_agg_exprs(agg_info, naming, src_alias="_jd")
    ins_agg_sql = ", ".join(ins_agg_parts)
    del_agg_sql = ", ".join(del_agg_parts)

    # Group cols for aggregation (unqualified names from inner aliases)
    agg_group_cols = ", ".join(f"_jd.{g}" for g in group_col_names)
    coalesce_group = ", ".join(f"COALESCE(i.{g}, d.{g}) AS {g}" for g in group_col_names)
    net_sql = _gen_agg_net_sql(agg_info, naming)
    join_cond = " AND ".join(f"i.{g} IS NOT DISTINCT FROM d.{g}" for g in group_col_names)

    # Inner col list for UNION/EXCEPT
    inner_col_list = ", ".join(inner_aliases)

    create_delta_agg = (
        f"CREATE TEMP TABLE _delta_agg AS\n"
        f"WITH {changes_l},\n"
        f"{changes_r},\n"
        # --- Join delta inserts (post-update tables) ---
        f"_ins_from_l AS (\n"
        f"    SELECT {proj_dl_s_aliased}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN {right_fqn} AS {right_name} ON {on_dl_sql}\n"
        f"    WHERE _dl.change_type IN ('insert', 'update_postimage')\n"
        f"),\n"
        f"_ins_from_r AS (\n"
        f"    SELECT {proj_r_dr_aliased}\n"
        f"    FROM {left_fqn} AS {left_name}\n"
        f"    JOIN _changes_r AS _dr ON {on_dr_sql}\n"
        f"    WHERE _dr.change_type IN ('insert', 'update_postimage')\n"
        f"),\n"
        f"_ins_cross AS (\n"
        f"    SELECT {proj_cross_aliased}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN _changes_r AS _dr ON {on_cross_sql}\n"
        f"    WHERE _dl.change_type IN ('insert', 'update_postimage')\n"
        f"    AND _dr.change_type IN ('insert', 'update_postimage')\n"
        f"),\n"
        f"_join_ins AS (\n"
        f"    SELECT {inner_col_list} FROM _ins_from_l\n"
        f"    UNION ALL\n"
        f"    SELECT {inner_col_list} FROM _ins_from_r\n"
        f"    EXCEPT ALL\n"
        f"    SELECT {inner_col_list} FROM _ins_cross\n"
        f"),\n"
        # --- Join delta deletes (pre-update tables) ---
        f"_del_from_l AS (\n"
        f"    SELECT {proj_dl_s_aliased}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN {right_old} AS {right_name} ON {on_dl_sql}\n"
        f"    WHERE _dl.change_type IN ('delete', 'update_preimage')\n"
        f"),\n"
        f"_del_from_r AS (\n"
        f"    SELECT {proj_r_dr_aliased}\n"
        f"    FROM {left_old} AS {left_name}\n"
        f"    JOIN _changes_r AS _dr ON {on_dr_sql}\n"
        f"    WHERE _dr.change_type IN ('delete', 'update_preimage')\n"
        f"),\n"
        f"_del_cross AS (\n"
        f"    SELECT {proj_cross_aliased}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN _changes_r AS _dr ON {on_cross_sql}\n"
        f"    WHERE _dl.change_type IN ('delete', 'update_preimage')\n"
        f"    AND _dr.change_type IN ('delete', 'update_preimage')\n"
        f"),\n"
        f"_join_del AS (\n"
        f"    SELECT {inner_col_list} FROM _del_from_l\n"
        f"    UNION ALL\n"
        f"    SELECT {inner_col_list} FROM _del_from_r\n"
        f"    EXCEPT ALL\n"
        f"    SELECT {inner_col_list} FROM _del_cross\n"
        f"),\n"
        # --- Aggregate the join deltas ---
        f"_ins_agg AS (\n"
        f"    SELECT {agg_group_cols}, {ins_agg_sql}\n"
        f"    FROM _join_ins AS _jd\n"
        f"    GROUP BY {agg_group_cols}\n"
        f"),\n"
        f"_del_agg AS (\n"
        f"    SELECT {agg_group_cols}, {del_agg_sql}\n"
        f"    FROM _join_del AS _jd\n"
        f"    GROUP BY {agg_group_cols}\n"
        f")\n"
        f"SELECT {coalesce_group}, {net_sql}\n"
        f"FROM _ins_agg i FULL OUTER JOIN _del_agg d ON {join_cond}"
    )

    # MV updates (same as Stage 2)
    update_existing, insert_new, delete_empty, drop_delta = _gen_agg_mv_updates(
        agg_info, group_col_names, mv_fqn, naming
    )

    # Cursor updates
    update_cursor_l = _gen_update_cursor(cursors_fqn, mv_table, left_src)
    update_cursor_r = _gen_update_cursor(cursors_fqn, mv_table, right_src)

    maintain = [
        *set_vars_left,
        *set_vars_right,
        create_delta_agg,
        update_existing,
        insert_new,
        delete_empty,
        drop_delta,
        update_cursor_l,
        update_cursor_r,
    ]

    features = _detect_features(ast)
    features.add("join")
    base_tables = {left_name: left_src["catalog"], right_name: right_src["catalog"]}

    query_mv = _gen_query_mv(ast, mv_fqn, naming, dialect)

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
        query_mv=query_mv,
    )
