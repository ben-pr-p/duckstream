"""
IVM Compiler: SQL-to-SQL incremental view maintenance.

Takes a SQL view definition and emits SQL statements that propagate
deltas from base tables into the materialized view.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from ducklake_ivm.plan import IVMPlan, Naming, UnsupportedSQLError


def compile_ivm(
    view_sql: str,
    *,
    dialect: str = "duckdb",
    naming: Naming | None = None,
    mv_catalog: str = "dl",
    mv_schema: str = "main",
    sources: dict[str, dict] | None = None,
) -> IVMPlan:
    """Compile a view definition into IVM maintenance SQL."""
    naming = naming or Naming()
    parsed = sqlglot.parse_one(view_sql, dialect=dialect)
    assert isinstance(parsed, exp.Select), f"Expected SELECT, got {type(parsed)}"
    ast: exp.Select = parsed

    # --- Analysis ---
    tables = list(ast.find_all(exp.Table))
    has_agg = bool(list(ast.find_all(exp.AggFunc)))
    has_distinct = bool(ast.args.get("distinct"))
    joins = ast.args.get("joins")
    has_join = bool(joins)

    if not tables:
        raise UnsupportedSQLError("no_table", "No tables found in view SQL")

    mv_table = naming.mv_table()
    mv_fqn = f"{mv_catalog}.{mv_schema}.{mv_table}"
    cursors_fqn = f"{mv_catalog}.{mv_schema}.{naming.cursors_table()}"

    # --- Generate SQL ---
    create_cursors = _gen_create_cursors(cursors_fqn)

    if has_join:
        assert joins is not None
        compile_fn = _compile_join_aggregate if has_agg else _compile_join
        return compile_fn(
            ast,
            tables,
            joins,
            dialect,
            naming,
            sources,
            mv_catalog,
            mv_schema,
            mv_table,
            mv_fqn,
            cursors_fqn,
            create_cursors,
        )

    if has_distinct and (has_agg or has_join):
        raise UnsupportedSQLError(
            "distinct_combo",
            "DISTINCT combined with GROUP BY or JOIN is not yet supported",
        )

    # --- Single table path ---
    table = tables[0]
    table_name = table.name
    src = _resolve_source(table_name, sources, mv_catalog, mv_schema)

    create_mv = _gen_create_mv(ast, mv_fqn, src, dialect, has_agg or has_distinct, naming)
    init_cursors = [_gen_init_cursor(cursors_fqn, mv_table, src)]

    if has_distinct:
        maintain = _gen_distinct_maintenance(
            ast, mv_fqn, cursors_fqn, mv_table, src, dialect, naming
        )
    elif has_agg:
        maintain = _gen_aggregate_maintenance(
            ast, mv_fqn, cursors_fqn, mv_table, src, dialect, naming
        )
    else:
        maintain = _gen_select_maintenance(ast, mv_fqn, cursors_fqn, mv_table, src, dialect)

    features = _detect_features(ast)

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables={table_name: src["catalog"]},
        features=features,
    )


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def _resolve_source(
    table_name: str,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
) -> dict[str, str]:
    if sources and table_name in sources:
        return {
            "catalog": sources[table_name].get("catalog", mv_catalog),
            "schema": sources[table_name].get("schema", "main"),
            "table": table_name,
        }
    return {"catalog": mv_catalog, "schema": mv_schema, "table": table_name}


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------


def _gen_create_cursors(cursors_fqn: str) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {cursors_fqn} (\n"
        f"    mv_name VARCHAR,\n"
        f"    source_catalog VARCHAR,\n"
        f"    last_snapshot BIGINT\n"
        f")"
    )


def _gen_create_mv(
    ast: exp.Select,
    mv_fqn: str,
    src: dict[str, str],
    dialect: str,
    has_agg: bool,
    naming: Naming,
) -> str:
    # Replace unqualified table ref with fully-qualified source
    table_name = src["table"]
    qualified_ast: exp.Select = ast.copy().transform(  # type: ignore[assignment]
        lambda node: (
            exp.table_(table_name, db=src["schema"], catalog=src["catalog"])
            if isinstance(node, exp.Table) and node.name == table_name
            else node
        )
    )

    # For DISTINCT views, strip DISTINCT and add GROUP BY all output cols
    if qualified_ast.args.get("distinct"):
        qualified_ast.args.pop("distinct")
        out_cols = _extract_output_col_names(qualified_ast)
        group_exprs = [exp.column(c) for c in out_cols]
        qualified_ast.args["group"] = exp.Group(expressions=group_exprs)

    if has_agg:
        # Add _ivm_count to the SELECT for aggregate views
        count_alias = naming.aux_column("count")
        count_expr = exp.alias_(exp.Count(this=exp.Star()), count_alias)
        qualified_ast.args["expressions"].append(count_expr)

        # For AVG, we also need _ivm_sum — handle the SELECT expressions
        new_selects = []
        for sel in list(qualified_ast.selects):
            alias_name = sel.alias if isinstance(sel, exp.Alias) else None
            inner = sel.this if isinstance(sel, exp.Alias) else sel
            if isinstance(inner, exp.Avg):
                sum_col = naming.aux_column(f"sum_{alias_name or 'val'}")
                new_selects.append(exp.alias_(exp.Sum(this=inner.this.copy()), sum_col))
                new_selects.append(
                    exp.alias_(
                        exp.Div(
                            this=exp.Cast(
                                this=exp.Sum(this=inner.this.copy()),
                                to=exp.DataType(this=exp.DataType.Type.DOUBLE),
                            ),
                            expression=exp.Count(this=exp.Star()),
                        ),
                        alias_name or "avg_val",
                    )
                )
            else:
                new_selects.append(sel)
        qualified_ast.args["expressions"] = new_selects
        has_ivm_count = any(
            (isinstance(s, exp.Alias) and s.alias == count_alias) for s in qualified_ast.selects
        )
        if not has_ivm_count:
            qualified_ast.args["expressions"].append(count_expr)

    return f"CREATE TABLE {mv_fqn} AS {qualified_ast.sql(dialect=dialect)}"


def _gen_create_mv_join(
    ast: exp.Select,
    mv_fqn: str,
    table_sources: dict[str, dict[str, str]],
    dialect: str,
    has_agg: bool = False,
    naming: Naming | None = None,
) -> str:
    """Generate CREATE TABLE AS for a join view, qualifying all table refs."""
    qualified_ast: exp.Select = ast.copy().transform(  # type: ignore[assignment]
        lambda node: (
            exp.table_(
                node.name,
                db=table_sources[node.name]["schema"],
                catalog=table_sources[node.name]["catalog"],
                alias=node.alias_or_name,
            )
            if isinstance(node, exp.Table) and node.name in table_sources
            else node
        )
    )

    if has_agg and naming:
        count_alias = naming.aux_column("count")
        count_expr = exp.alias_(exp.Count(this=exp.Star()), count_alias)

        # Handle AVG decomposition
        new_selects = []
        for sel in list(qualified_ast.selects):
            alias_name = sel.alias if isinstance(sel, exp.Alias) else None
            inner = sel.this if isinstance(sel, exp.Alias) else sel
            if isinstance(inner, exp.Avg):
                sum_col = naming.aux_column(f"sum_{alias_name or 'val'}")
                new_selects.append(exp.alias_(exp.Sum(this=inner.this.copy()), sum_col))
                new_selects.append(
                    exp.alias_(
                        exp.Div(
                            this=exp.Cast(
                                this=exp.Sum(this=inner.this.copy()),
                                to=exp.DataType(this=exp.DataType.Type.DOUBLE),
                            ),
                            expression=exp.Count(this=exp.Star()),
                        ),
                        alias_name or "avg_val",
                    )
                )
            else:
                new_selects.append(sel)
        qualified_ast.args["expressions"] = new_selects

        has_ivm_count = any(
            (isinstance(s, exp.Alias) and s.alias == count_alias) for s in qualified_ast.selects
        )
        if not has_ivm_count:
            qualified_ast.args["expressions"].append(count_expr)

    return f"CREATE TABLE {mv_fqn} AS {qualified_ast.sql(dialect=dialect)}"


def _gen_init_cursor(cursors_fqn: str, mv_table: str, src: dict[str, str]) -> str:
    return (
        f"INSERT INTO {cursors_fqn} (mv_name, source_catalog, last_snapshot)\n"
        f"VALUES ('{mv_table}', '{src['catalog']}',\n"
        f"    (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src['catalog']}')))"
    )


# ---------------------------------------------------------------------------
# Column rewriting
# ---------------------------------------------------------------------------


def _repoint_columns_to_delta(node: exp.Expression) -> exp.Expression:
    """Repoint column references to the _delta alias."""
    if isinstance(node, exp.Column):
        return exp.column(node.name, table="_delta")
    return node


# ---------------------------------------------------------------------------
# Snapshot variable helpers
# ---------------------------------------------------------------------------


def _gen_set_snapshot_vars(src: dict[str, str], cursors_fqn: str, mv_table: str) -> list[str]:
    """Generate SET VARIABLE statements to capture snapshot range.

    DuckDB table functions cannot contain subqueries, so we capture values first.
    """
    return [
        (
            f"SET VARIABLE _ivm_snap_start = (\n"
            f"    SELECT last_snapshot + 1 FROM {cursors_fqn}\n"
            f"    WHERE mv_name = '{mv_table}' AND source_catalog = '{src['catalog']}'\n"
            f")"
        ),
        (
            f"SET VARIABLE _ivm_snap_end = (\n"
            f"    SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src['catalog']}')\n"
            f")"
        ),
    ]


def _gen_set_snapshot_vars_named(
    src: dict[str, str], cursors_fqn: str, mv_table: str, suffix: str
) -> list[str]:
    """Like _gen_set_snapshot_vars but with a named suffix for multi-table cases."""
    return [
        (
            f"SET VARIABLE _ivm_snap_start_{suffix} = (\n"
            f"    SELECT last_snapshot + 1 FROM {cursors_fqn}\n"
            f"    WHERE mv_name = '{mv_table}'"
            f" AND source_catalog = '{src['catalog']}'\n"
            f")"
        ),
        (
            f"SET VARIABLE _ivm_snap_end_{suffix} = (\n"
            f"    SELECT MAX(snapshot_id)"
            f" FROM ducklake_snapshots('{src['catalog']}')\n"
            f")"
        ),
    ]


def _gen_changes_cte(src: dict[str, str]) -> str:
    """Generate the _changes CTE. Assumes _ivm_snap_start/_end are set."""
    return (
        f"_changes AS (\n"
        f"    SELECT * FROM ducklake_table_changes(\n"
        f"        '{src['catalog']}', '{src['schema']}', '{src['table']}',\n"
        f"        getvariable('_ivm_snap_start'),\n"
        f"        getvariable('_ivm_snap_end')\n"
        f"    )\n"
        f")"
    )


def _gen_changes_cte_named(src: dict[str, str], cte_name: str, suffix: str) -> str:
    """Generate a named changes CTE for multi-table join cases."""
    return (
        f"{cte_name} AS (\n"
        f"    SELECT * FROM ducklake_table_changes(\n"
        f"        '{src['catalog']}', '{src['schema']}', '{src['table']}',\n"
        f"        getvariable('_ivm_snap_start_{suffix}'),\n"
        f"        getvariable('_ivm_snap_end_{suffix}')\n"
        f"    )\n"
        f")"
    )


# ---------------------------------------------------------------------------
# Stage 1: SELECT / PROJECT / WHERE maintenance
# ---------------------------------------------------------------------------


def _extract_output_col_names(ast: exp.Select) -> list[str]:
    """Get the output column names from the SELECT clause."""
    names = []
    for sel in ast.selects:
        if isinstance(sel, exp.Alias):
            names.append(sel.alias)
        elif isinstance(sel, exp.Column):
            names.append(sel.name)
        else:
            names.append(sel.sql())
    return names


def _gen_where_clause(ast: exp.Select, dialect: str) -> str:
    """Extract and repoint the WHERE clause for delta queries."""
    original_where = ast.args.get("where")
    if original_where:
        delta_where = original_where.this.copy().transform(_repoint_columns_to_delta)
        return f" AND {delta_where.sql(dialect=dialect)}"
    return ""


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


# ---------------------------------------------------------------------------
# Aggregate analysis and shared helpers
# ---------------------------------------------------------------------------

AggInfo = list[tuple[str, str, str | None, bool]]
# Each entry: (alias, agg_type, inner_col_name, is_avg)
# agg_type is one of: "SUM", "COUNT_STAR", "COUNT_COL", "AVG"


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
    ins_parts.append("COUNT(*) AS _ins_cnt")
    del_parts.append("COUNT(*) AS _del_cnt")
    return ins_parts, del_parts


def _gen_agg_net_sql(agg_info: AggInfo, naming: Naming) -> tuple[str, str, str]:
    """Build COALESCE net expressions and FULL OUTER JOIN condition.

    Returns (coalesce_group_sql, net_sql, join_cond) — but group cols
    must be supplied externally since they depend on context.
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
    net_parts.append("COALESCE(i._ins_cnt, 0) - COALESCE(d._del_cnt, 0) AS _net_count")
    return ", ".join(net_parts)


def _gen_agg_mv_updates(
    agg_info: AggInfo,
    group_col_names: list[str],
    mv_fqn: str,
    naming: Naming,
) -> tuple[str, str, str, str]:
    """Generate UPDATE/INSERT/DELETE/DROP for aggregate MV maintenance.

    Assumes _delta_agg temp table exists with _net_* columns.
    Returns (update_existing, insert_new, delete_empty, drop_delta).
    """
    ivm_count_col = naming.aux_column("count")

    # UPDATE existing groups
    update_sets = []
    for alias, agg_type, _col_name, _is_avg in agg_info:
        if agg_type in ("SUM", "COUNT_STAR", "COUNT_COL"):
            update_sets.append(f"{alias} = mv.{alias} + d._net_{alias}")
        elif agg_type == "AVG":
            sum_col = naming.aux_column(f"sum_{alias}")
            update_sets.append(f"{sum_col} = mv.{sum_col} + d._net_{sum_col}")
            update_sets.append(
                f"{alias} = CAST(mv.{sum_col} + d._net_{sum_col} AS DOUBLE) "
                f"/ (mv.{ivm_count_col} + d._net_count)"
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


# ---------------------------------------------------------------------------
# Stage 2: Aggregate maintenance (GROUP BY + COUNT/SUM/AVG)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 6: DISTINCT maintenance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 4: Two-table inner JOIN maintenance
# ---------------------------------------------------------------------------


def _compile_join(
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
    """Compile a two-table inner join view."""
    if len(joins) != 1:
        raise UnsupportedSQLError("multi_join", "Only two-table joins supported")

    join_node = joins[0]
    # Left table is in FROM, right table is in JOIN
    left_table_node = ast.args["from_"].this
    right_table_node = join_node.this
    left_name = left_table_node.name
    right_name = right_table_node.name

    left_src = _resolve_source(left_name, sources, mv_catalog, mv_schema)
    right_src = _resolve_source(right_name, sources, mv_catalog, mv_schema)

    table_sources = {left_name: left_src, right_name: right_src}
    left_fqn = f"{left_src['catalog']}.{left_src['schema']}.{left_name}"
    right_fqn = f"{right_src['catalog']}.{right_src['schema']}.{right_name}"

    # Extract the ON condition
    on_condition = join_node.args.get("on")

    # Get output col names
    out_col_names = _extract_output_col_names(ast)

    create_mv = _gen_create_mv_join(ast, mv_fqn, table_sources, dialect)
    init_cursors = [
        _gen_init_cursor(cursors_fqn, mv_table, left_src),
    ]
    # Only add second cursor if catalogs differ
    if left_src["catalog"] != right_src["catalog"]:
        init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, right_src))

    # --- Maintenance SQL ---
    # Always generate the full three-way decomposition since at runtime
    # we don't know which tables changed. The cross-delta subtraction
    # handles the case where both changed simultaneously, and is a no-op
    # (empty result) when only one table changed.

    set_vars_left = _gen_set_snapshot_vars_named(left_src, cursors_fqn, mv_table, "l")
    set_vars_right = _gen_set_snapshot_vars_named(right_src, cursors_fqn, mv_table, "r")

    changes_l = _gen_changes_cte_named(left_src, "_changes_l", "l")
    changes_r = _gen_changes_cte_named(right_src, "_changes_r", "r")

    # Rewrite ON condition for delta aliases
    def _rewrite_on_for_delta(on_node: exp.Expression, delta_alias: str, table_name: str):
        """Replace references to table_name with delta_alias in the ON condition."""

        def _replace(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column) and node.table == table_name:
                return exp.column(node.name, table=delta_alias)
            return node

        return on_node.copy().transform(_replace)

    # Rewrite projection for delta queries
    def _rewrite_proj(sel_list: list, delta_alias: str, table_name: str, dialect: str) -> str:
        """Replace references to table_name with delta_alias in projections."""
        parts = []
        for sel in sel_list:
            rewritten = sel.copy().transform(
                lambda node: (
                    exp.column(node.name, table=delta_alias)
                    if isinstance(node, exp.Column) and node.table == table_name
                    else node
                )
            )
            parts.append(rewritten.sql(dialect=dialect))
        return ", ".join(parts)

    # The ON condition rewritten for each delta scenario
    on_delta_l = _rewrite_on_for_delta(on_condition, "_dl", left_name)
    on_delta_r = _rewrite_on_for_delta(on_condition, "_dr", right_name)
    on_delta_both_l = on_delta_l.copy().transform(
        lambda node: (
            exp.column(node.name, table="_dr")
            if isinstance(node, exp.Column) and node.table == right_name
            else node
        )
    )

    on_dl_sql = on_delta_l.sql(dialect=dialect)
    on_dr_sql = on_delta_r.sql(dialect=dialect)
    on_cross_sql = on_delta_both_l.sql(dialect=dialect)

    # Projection rewritten for each term
    proj_dl_s = _rewrite_proj(list(ast.selects), "_dl", left_name, dialect)
    proj_r_dr = _rewrite_proj(list(ast.selects), "_dr", right_name, dialect)
    proj_cross = _rewrite_proj(list(ast.selects), "_dl", left_name, dialect)
    proj_cross = _rewrite_proj(
        [sqlglot.parse_one(p.strip(), dialect=dialect) for p in proj_cross.split(",")],
        "_dr",
        right_name,
        dialect,
    )

    # For DELETE we use pre-update (old) table state via DuckLake time travel.
    # The old snapshot = snap_start - 1 (the last snapshot the cursor saw).
    # AT VERSION can't be aliased directly, so we wrap in a subquery.
    right_old = f"(SELECT * FROM {right_fqn} AT (VERSION => getvariable('_ivm_snap_start_r') - 1))"
    left_old = f"(SELECT * FROM {left_fqn} AT (VERSION => getvariable('_ivm_snap_start_l') - 1))"

    # --- DELETE: (∇R ⋈ S_old) ∪ (R_old ⋈ ∇S) - (∇R ⋈ ∇S) ---
    all_proj_cols = ", ".join(out_col_names)
    join_on_parts = [f"m.{c} IS NOT DISTINCT FROM d.{c}" for c in out_col_names]
    join_on = " AND ".join(join_on_parts)

    delete_sql = (
        f"WITH {changes_l},\n"
        f"{changes_r},\n"
        f"_del_from_l AS (\n"
        f"    SELECT {proj_dl_s}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN {right_old} AS {right_name} ON {on_dl_sql}\n"
        f"    WHERE _dl.change_type IN ('delete', 'update_preimage')\n"
        f"),\n"
        f"_del_from_r AS (\n"
        f"    SELECT {proj_r_dr}\n"
        f"    FROM {left_old} AS {left_name}\n"
        f"    JOIN _changes_r AS _dr ON {on_dr_sql}\n"
        f"    WHERE _dr.change_type IN ('delete', 'update_preimage')\n"
        f"),\n"
        f"_del_cross AS (\n"
        f"    SELECT {proj_cross}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN _changes_r AS _dr ON {on_cross_sql}\n"
        f"    WHERE _dl.change_type IN ('delete', 'update_preimage')\n"
        f"    AND _dr.change_type IN ('delete', 'update_preimage')\n"
        f"),\n"
        f"_all_deletes AS (\n"
        f"    SELECT * FROM _del_from_l\n"
        f"    UNION ALL\n"
        f"    SELECT * FROM _del_from_r\n"
        f"    EXCEPT ALL\n"
        f"    SELECT * FROM _del_cross\n"
        f"),\n"
        f"_deletes_numbered AS (\n"
        f"    SELECT *, ROW_NUMBER() OVER (\n"
        f"        PARTITION BY {all_proj_cols}"
        f" ORDER BY (SELECT NULL)) AS _dn\n"
        f"    FROM _all_deletes\n"
        f"),\n"
        f"_mv_numbered AS (\n"
        f"    SELECT rowid AS _rid, {all_proj_cols},\n"
        f"        ROW_NUMBER() OVER ("
        f"PARTITION BY {all_proj_cols} ORDER BY rowid) AS _mn\n"
        f"    FROM {mv_fqn}\n"
        f")\n"
        f"DELETE FROM {mv_fqn}\n"
        f"WHERE rowid IN (\n"
        f"    SELECT _rid FROM _mv_numbered m\n"
        f"    JOIN _deletes_numbered d"
        f" ON {join_on} AND m._mn = d._dn\n"
        f")"
    )

    # --- INSERT: (ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS) ---
    col_list = ", ".join(out_col_names)
    insert_sql = (
        f"WITH {changes_l},\n"
        f"{changes_r},\n"
        f"_ins_from_l AS (\n"
        f"    SELECT {proj_dl_s}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN {right_fqn} AS {right_name} ON {on_dl_sql}\n"
        f"    WHERE _dl.change_type IN ('insert', 'update_postimage')\n"
        f"),\n"
        f"_ins_from_r AS (\n"
        f"    SELECT {proj_r_dr}\n"
        f"    FROM {left_fqn} AS {left_name}\n"
        f"    JOIN _changes_r AS _dr ON {on_dr_sql}\n"
        f"    WHERE _dr.change_type IN ('insert', 'update_postimage')\n"
        f"),\n"
        f"_ins_cross AS (\n"
        f"    SELECT {proj_cross}\n"
        f"    FROM _changes_l AS _dl\n"
        f"    JOIN _changes_r AS _dr ON {on_cross_sql}\n"
        f"    WHERE _dl.change_type IN ('insert', 'update_postimage')\n"
        f"    AND _dr.change_type IN ('insert', 'update_postimage')\n"
        f"),\n"
        f"_all_inserts AS (\n"
        f"    SELECT * FROM _ins_from_l\n"
        f"    UNION ALL\n"
        f"    SELECT * FROM _ins_from_r\n"
        f"    EXCEPT ALL\n"
        f"    SELECT * FROM _ins_cross\n"
        f")\n"
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT * FROM _all_inserts"
    )

    # --- Update cursors ---
    update_cursor_l = _gen_update_cursor(cursors_fqn, mv_table, left_src)
    update_cursor_r = _gen_update_cursor(cursors_fqn, mv_table, right_src)

    maintain = [
        *set_vars_left,
        *set_vars_right,
        delete_sql,
        insert_sql,
        update_cursor_l,
        update_cursor_r,
    ]

    features = _detect_features(ast)
    features.add("join")

    base_tables = {left_name: left_src["catalog"], right_name: right_src["catalog"]}

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
    )


# ---------------------------------------------------------------------------
# Stage 5: JOIN + Aggregates (Composed)
# ---------------------------------------------------------------------------


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

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gen_update_cursor(cursors_fqn: str, mv_table: str, src: dict[str, str]) -> str:
    return (
        f"UPDATE {cursors_fqn}\n"
        f"SET last_snapshot = (\n"
        f"    SELECT MAX(snapshot_id)"
        f" FROM ducklake_snapshots('{src['catalog']}'))\n"
        f"WHERE mv_name = '{mv_table}'"
        f" AND source_catalog = '{src['catalog']}'"
    )


def _detect_features(ast: exp.Select) -> set[str]:
    features: set[str] = {"select"}
    if ast.args.get("where"):
        features.add("where")
    if ast.args.get("distinct"):
        features.add("distinct")
    if ast.args.get("group"):
        features.add("group_by")
    if ast.args.get("joins"):
        features.add("join")
    for agg in ast.find_all(exp.AggFunc):
        if isinstance(agg, exp.Sum):
            features.add("sum")
        elif isinstance(agg, exp.Count):
            features.add("count")
        elif isinstance(agg, exp.Avg):
            features.add("avg")
    return features
