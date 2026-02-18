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

    # --- Set operations (UNION/EXCEPT/INTERSECT) ---
    if isinstance(parsed, (exp.Union, exp.Except, exp.Intersect)):
        return _compile_set_operation(
            parsed,
            dialect=dialect,
            naming=naming,
            sources=sources,
            mv_catalog=mv_catalog,
            mv_schema=mv_schema,
        )

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
        # Detect outer joins
        has_outer = any(j.side in ("LEFT", "RIGHT", "FULL") for j in joins)
        if has_outer:
            if has_agg:
                raise UnsupportedSQLError(
                    "outer_join_agg",
                    "Outer joins with aggregates are not yet supported",
                )
            compile_fn = _compile_outer_join
        elif has_agg:
            compile_fn = _compile_join_aggregate
        else:
            compile_fn = _compile_join
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
    """Compile an N-table inner join view (2+ tables).

    Uses inclusion-exclusion to compute the join delta:
    For each non-empty subset S of tables, generate a term where the tables
    in S use their delta changes and the rest use current/old state.
    Sign is (-1)^(|S|-1): positive for odd-size subsets, negative for even.
    """
    # Extract ordered table info: FROM table + each JOIN table
    from_table_node = ast.args["from_"].this
    table_names_ordered = [from_table_node.name]
    join_on_conditions = []
    for join_node in joins:
        table_names_ordered.append(join_node.this.name)
        join_on_conditions.append(join_node.args.get("on"))

    n_tables = len(table_names_ordered)

    # Resolve sources and build FQNs
    table_sources: dict[str, dict[str, str]] = {}
    table_fqns: dict[str, str] = {}
    for tname in table_names_ordered:
        src = _resolve_source(tname, sources, mv_catalog, mv_schema)
        table_sources[tname] = src
        table_fqns[tname] = f"{src['catalog']}.{src['schema']}.{tname}"

    out_col_names = _extract_output_col_names(ast)

    create_mv = _gen_create_mv_join(ast, mv_fqn, table_sources, dialect)

    # Initialize cursors — one per unique catalog
    seen_catalogs: set[str] = set()
    init_cursors = []
    for tname in table_names_ordered:
        src = table_sources[tname]
        if src["catalog"] not in seen_catalogs:
            init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, src))
            seen_catalogs.add(src["catalog"])

    # --- Snapshot variables and changes CTEs ---
    # Use table index as suffix for variable names
    all_set_vars: list[str] = []
    suffixes: dict[str, str] = {}
    changes_ctes: dict[str, str] = {}
    for idx, tname in enumerate(table_names_ordered):
        suffix = str(idx)
        suffixes[tname] = suffix
        src = table_sources[tname]
        all_set_vars.extend(_gen_set_snapshot_vars_named(src, cursors_fqn, mv_table, suffix))
        cte_name = f"_changes_{idx}"
        changes_ctes[tname] = _gen_changes_cte_named(src, cte_name, suffix)

    # Delta alias for table i when it's in the delta subset
    def _delta_alias(tname: str) -> str:
        return f"_d{suffixes[tname]}"

    # Old table reference for deletes (time travel to pre-update state)
    def _old_ref(tname: str) -> str:
        suffix = suffixes[tname]
        fqn = table_fqns[tname]
        return f"(SELECT * FROM {fqn} AT (VERSION => getvariable('_ivm_snap_start_{suffix}') - 1))"

    # Quote table name for use as alias (handles reserved keywords)
    def _quote_alias(tname: str) -> str:
        return f'"{tname}"'

    # --- Rewrite helpers ---
    def _rewrite_expr(
        expr: exp.Expression, alias_map: dict[str, str], dialect: str
    ) -> exp.Expression:
        """Replace table references according to alias_map."""

        def _replace(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column) and node.table in alias_map:
                return exp.column(node.name, table=alias_map[node.table])
            return node

        return expr.copy().transform(_replace)

    def _rewrite_proj_sql(alias_map: dict[str, str]) -> str:
        """Rewrite projection with alias_map and return SQL string."""
        parts = []
        for sel in ast.selects:
            rewritten = _rewrite_expr(sel, alias_map, dialect)
            parts.append(rewritten.sql(dialect=dialect))
        return ", ".join(parts)

    # --- Generate inclusion-exclusion terms ---
    # For each non-empty subset of tables, generate a CTE.
    # Sign: positive for odd-size subsets, negative for even-size.

    def _gen_term_cte(
        subset: list[str],
        change_type_filter: str,
        use_old_for_non_delta: bool,
    ) -> str:
        """Generate a single inclusion-exclusion term.

        subset: table names that use their delta (changes CTE)
        change_type_filter: 'insert'/'delete' filter string
        use_old_for_non_delta: if True, non-delta tables use old state (for deletes)
        """
        subset_set = set(subset)

        # Build alias map: delta tables get delta alias, others keep table name
        alias_map: dict[str, str] = {}
        for tname in table_names_ordered:
            if tname in subset_set:
                alias_map[tname] = _delta_alias(tname)

        proj_sql = _rewrite_proj_sql(alias_map)

        # Build FROM clause: first table
        first_tname = table_names_ordered[0]
        if first_tname in subset_set:
            from_clause = f"_changes_{suffixes[first_tname]} AS {_delta_alias(first_tname)}"
        elif use_old_for_non_delta:
            from_clause = f"{_old_ref(first_tname)} AS {_quote_alias(first_tname)}"
        else:
            from_clause = f"{table_fqns[first_tname]} AS {_quote_alias(first_tname)}"

        # Build JOIN clauses for remaining tables
        join_clauses = []
        for i, tname in enumerate(table_names_ordered[1:]):
            on_cond = join_on_conditions[i]
            rewritten_on = _rewrite_expr(on_cond, alias_map, dialect)
            on_sql = rewritten_on.sql(dialect=dialect)

            if tname in subset_set:
                join_ref = f"_changes_{suffixes[tname]} AS {_delta_alias(tname)}"
            elif use_old_for_non_delta:
                join_ref = f"{_old_ref(tname)} AS {_quote_alias(tname)}"
            else:
                join_ref = f"{table_fqns[tname]} AS {_quote_alias(tname)}"

            join_clauses.append(f"    JOIN {join_ref} ON {on_sql}")

        # WHERE clause: filter delta tables by change_type
        where_parts = []
        for tname in subset:
            da = _delta_alias(tname)
            where_parts.append(f"{da}.change_type IN {change_type_filter}")
        where_sql = " AND ".join(where_parts)

        joins_sql = "\n".join(join_clauses)
        return f"    SELECT {proj_sql}\n    FROM {from_clause}\n{joins_sql}\n    WHERE {where_sql}"

    def _gen_all_terms(
        change_type_filter: str, use_old: bool
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Generate all inclusion-exclusion terms, separated by sign.

        Returns (positive_term_cte_names, negative_term_cte_names).
        """
        positive_ctes: list[tuple[str, str]] = []
        negative_ctes: list[tuple[str, str]] = []

        term_idx = 0
        # Iterate over all non-empty subsets (by size for clarity)
        for size in range(1, n_tables + 1):
            sign_positive = size % 2 == 1  # odd size = positive
            for subset in _subsets_of_size(table_names_ordered, size):
                cte_body = _gen_term_cte(subset, change_type_filter, use_old)
                prefix = "del" if use_old else "ins"
                cte_name = f"_{prefix}_term_{term_idx}"
                term_idx += 1
                cte_def = f"{cte_name} AS (\n{cte_body}\n)"
                if sign_positive:
                    positive_ctes.append((cte_name, cte_def))
                else:
                    negative_ctes.append((cte_name, cte_def))

        return positive_ctes, negative_ctes

    # Changes CTEs (shared across all terms)
    changes_cte_list = [changes_ctes[tname] for tname in table_names_ordered]
    changes_cte_sql = ",\n".join(changes_cte_list)

    # --- DELETE: inclusion-exclusion with old state ---
    del_positive, del_negative = _gen_all_terms("('delete', 'update_preimage')", use_old=True)
    # --- INSERT: inclusion-exclusion with current state ---
    ins_positive, ins_negative = _gen_all_terms("('insert', 'update_postimage')", use_old=False)

    all_proj_cols = ", ".join(out_col_names)
    join_on_parts = [f"m.{c} IS NOT DISTINCT FROM d.{c}" for c in out_col_names]
    join_on = " AND ".join(join_on_parts)
    col_list = ", ".join(out_col_names)

    def _build_delta_sql(
        positive: list[tuple[str, str]],
        negative: list[tuple[str, str]],
        all_name: str,
    ) -> list[str]:
        """Build CTE definitions and the combined _all_X CTE.

        Uses subqueries to ensure correct precedence:
        (positive UNION ALL) EXCEPT ALL (negative UNION ALL).
        """
        cte_defs = [cte_def for _, cte_def in positive] + [cte_def for _, cte_def in negative]

        pos_union = "\n        UNION ALL\n        ".join(
            f"SELECT * FROM {name}" for name, _ in positive
        )
        if negative:
            neg_union = "\n        UNION ALL\n        ".join(
                f"SELECT * FROM {name}" for name, _ in negative
            )
            all_cte = (
                f"{all_name} AS (\n"
                f"    SELECT * FROM (\n        {pos_union}\n    )\n"
                f"    EXCEPT ALL\n"
                f"    SELECT * FROM (\n        {neg_union}\n    )\n"
                f")"
            )
        else:
            all_cte = f"{all_name} AS (\n    {pos_union}\n)"
        cte_defs.append(all_cte)
        return cte_defs

    # DELETE SQL
    del_cte_defs = _build_delta_sql(del_positive, del_negative, "_all_deletes")
    del_ctes_sql = ",\n".join(del_cte_defs)
    delete_sql = (
        f"WITH {changes_cte_sql},\n"
        f"{del_ctes_sql},\n"
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

    # INSERT SQL
    ins_cte_defs = _build_delta_sql(ins_positive, ins_negative, "_all_inserts")
    ins_ctes_sql = ",\n".join(ins_cte_defs)
    insert_sql = (
        f"WITH {changes_cte_sql},\n"
        f"{ins_ctes_sql}\n"
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT * FROM _all_inserts"
    )

    # --- Update cursors ---
    update_cursors = [
        _gen_update_cursor(cursors_fqn, mv_table, table_sources[tname])
        for tname in table_names_ordered
    ]

    maintain = [*all_set_vars, delete_sql, insert_sql, *update_cursors]

    features = _detect_features(ast)
    features.add("join")

    base_tables = {tname: table_sources[tname]["catalog"] for tname in table_names_ordered}

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
# Stage 9: Outer JOIN maintenance (LEFT / RIGHT / FULL)
# ---------------------------------------------------------------------------


def _compile_outer_join(
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
    """Compile a two-table outer join view (LEFT/RIGHT/FULL).

    Uses key-scoped recomputation: for each affected join key in the delta,
    delete all MV rows with that key and re-insert via the outer join on
    the updated tables. This is correct and avoids complex NULL-extension
    tracking.
    """
    if len(joins) != 1:
        raise UnsupportedSQLError(
            "multi_outer_join",
            "Only two-table outer joins are supported",
        )

    join_node = joins[0]
    join_side = join_node.side  # "LEFT", "RIGHT", or "FULL"

    from_table_node = ast.args["from_"].this
    left_name = from_table_node.name
    right_name = join_node.this.name

    left_src = _resolve_source(left_name, sources, mv_catalog, mv_schema)
    right_src = _resolve_source(right_name, sources, mv_catalog, mv_schema)
    table_sources = {left_name: left_src, right_name: right_src}
    left_fqn = f"{left_src['catalog']}.{left_src['schema']}.{left_name}"
    right_fqn = f"{right_src['catalog']}.{right_src['schema']}.{right_name}"

    out_col_names = _extract_output_col_names(ast)

    # --- Extract join key columns from ON condition ---
    on_condition = join_node.args.get("on")
    left_key_cols, right_key_cols = _extract_join_keys(on_condition, left_name, right_name)

    # --- Hidden key columns for outer join maintenance ---
    # Outer joins may NULL-pad the non-preserved side's columns, so the
    # user-visible key columns can be NULL.  We add hidden _ivm_lkey_* /
    # _ivm_rkey_* columns that always store each side's key value so the
    # maintenance DELETE can reliably match affected rows.
    hidden_lkey_names = [naming.aux_column(f"lkey_{c}") for c in left_key_cols]
    hidden_rkey_names = [naming.aux_column(f"rkey_{c}") for c in right_key_cols]

    # Build the hidden key SELECT expressions for CREATE MV
    hidden_key_exprs: list[str] = []
    for lk, hname in zip(left_key_cols, hidden_lkey_names, strict=True):
        hidden_key_exprs.append(f"{left_name}.{lk} AS {hname}")
    for rk, hname in zip(right_key_cols, hidden_rkey_names, strict=True):
        hidden_key_exprs.append(f"{right_name}.{rk} AS {hname}")

    # --- DDL ---
    base_create = _gen_create_mv_join(ast, mv_fqn, table_sources, dialect)
    # Inject hidden key columns into the SELECT
    # base_create is "CREATE TABLE <mv_fqn> AS SELECT ... FROM ..."
    # We need to add the hidden columns to the SELECT list
    as_idx = base_create.index(" AS SELECT ") + len(" AS ")
    create_select = base_create[as_idx:]
    # Parse, add hidden cols, regenerate
    create_ast = sqlglot.parse_one(create_select, dialect=dialect)
    for hexpr in hidden_key_exprs:
        parsed_expr = sqlglot.parse_one(hexpr, into=exp.Expression, dialect=dialect)
        create_ast.args["expressions"].append(parsed_expr)
    create_mv = f"CREATE TABLE {mv_fqn} AS {create_ast.sql(dialect=dialect)}"

    seen_catalogs: set[str] = set()
    init_cursors = []
    for tname in [left_name, right_name]:
        src = table_sources[tname]
        if src["catalog"] not in seen_catalogs:
            init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, src))
            seen_catalogs.add(src["catalog"])

    # --- Snapshot vars and changes CTEs ---
    set_vars_l = _gen_set_snapshot_vars_named(left_src, cursors_fqn, mv_table, "l")
    set_vars_r = _gen_set_snapshot_vars_named(right_src, cursors_fqn, mv_table, "r")
    changes_l_cte = _gen_changes_cte_named(left_src, "_changes_l", "l")
    changes_r_cte = _gen_changes_cte_named(right_src, "_changes_r", "r")

    # --- Build the join SQL fragment (for re-insertion) ---
    join_keyword = {"LEFT": "LEFT JOIN", "RIGHT": "RIGHT JOIN", "FULL": "FULL OUTER JOIN"}[
        join_side
    ]
    on_sql = on_condition.sql(dialect=dialect)

    # Projection SQL with fully-qualified table refs (includes hidden key cols)
    proj_parts = []
    for sel in ast.selects:
        proj_parts.append(sel.sql(dialect=dialect))
    for hexpr in hidden_key_exprs:
        proj_parts.append(hexpr)
    proj_sql = ", ".join(proj_parts)

    # All MV column names including hidden keys
    all_mv_col_names = out_col_names + hidden_lkey_names + hidden_rkey_names

    # Build the maintenance SQL
    maintain_stmts: list[str] = []

    if join_side == "LEFT":
        # Preserved side = left. Delete and re-insert by left key.
        maintain_stmts.extend(
            _gen_outer_join_maintain_left(
                changes_l_cte,
                changes_r_cte,
                left_fqn,
                right_fqn,
                left_name,
                right_name,
                left_key_cols,
                right_key_cols,
                hidden_lkey_names,
                hidden_rkey_names,
                mv_fqn,
                proj_sql,
                join_keyword,
                on_sql,
                all_mv_col_names,
                dialect,
            )
        )
    elif join_side == "RIGHT":
        # Preserved side = right. Mirror of LEFT.
        maintain_stmts.extend(
            _gen_outer_join_maintain_right(
                changes_l_cte,
                changes_r_cte,
                left_fqn,
                right_fqn,
                left_name,
                right_name,
                left_key_cols,
                right_key_cols,
                hidden_lkey_names,
                hidden_rkey_names,
                mv_fqn,
                proj_sql,
                join_keyword,
                on_sql,
                all_mv_col_names,
                dialect,
            )
        )
    else:
        # FULL OUTER: both sides preserved
        maintain_stmts.extend(
            _gen_outer_join_maintain_full(
                changes_l_cte,
                changes_r_cte,
                left_fqn,
                right_fqn,
                left_name,
                right_name,
                left_key_cols,
                right_key_cols,
                hidden_lkey_names,
                hidden_rkey_names,
                mv_fqn,
                proj_sql,
                join_keyword,
                on_sql,
                all_mv_col_names,
                dialect,
            )
        )

    update_cursor_l = _gen_update_cursor(cursors_fqn, mv_table, left_src)
    update_cursor_r = _gen_update_cursor(cursors_fqn, mv_table, right_src)

    maintain = [*set_vars_l, *set_vars_r, *maintain_stmts, update_cursor_l, update_cursor_r]

    features = _detect_features(ast)
    features.add("join")
    features.add(f"{join_side.lower()}_join")
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


def _extract_join_keys(
    on_condition: exp.Expression, left_name: str, right_name: str
) -> tuple[list[str], list[str]]:
    """Extract join key column names from an ON condition.

    Handles simple equi-joins: R.a = S.a AND R.b = S.b
    Returns (left_key_cols, right_key_cols) as unqualified column names.
    """
    left_keys: list[str] = []
    right_keys: list[str] = []

    eqs = list(on_condition.find_all(exp.EQ))
    if not eqs and isinstance(on_condition, exp.EQ):
        eqs = [on_condition]

    for eq in eqs:
        left_col = eq.left
        right_col = eq.right
        if isinstance(left_col, exp.Column) and isinstance(right_col, exp.Column):
            if left_col.table == left_name and right_col.table == right_name:
                left_keys.append(left_col.name)
                right_keys.append(right_col.name)
            elif left_col.table == right_name and right_col.table == left_name:
                left_keys.append(right_col.name)
                right_keys.append(left_col.name)

    return left_keys, right_keys


def _gen_outer_join_maintain_left(
    changes_l_cte: str,
    changes_r_cte: str,
    left_fqn: str,
    right_fqn: str,
    left_name: str,
    right_name: str,
    left_key_cols: list[str],
    right_key_cols: list[str],
    hidden_lkey_names: list[str],
    hidden_rkey_names: list[str],
    mv_fqn: str,
    proj_sql: str,
    join_keyword: str,
    on_sql: str,
    all_mv_col_names: list[str],
    dialect: str,
) -> list[str]:
    """Generate maintenance SQL for LEFT JOIN.

    Uses hidden _ivm_lkey_* columns (always non-NULL for preserved left side)
    to reliably identify affected MV rows.
    """
    col_list = ", ".join(all_mv_col_names)

    # Affected left keys: from left delta + right delta mapped via equi-join
    left_delta_keys = ", ".join(f"_changes_l.{k}" for k in left_key_cols)
    right_delta_as_left = ", ".join(
        f"_changes_r.{rk} AS {lk}" for lk, rk in zip(left_key_cols, right_key_cols, strict=True)
    )

    create_affected = (
        f"CREATE TEMP TABLE _affected_keys AS\n"
        f"WITH {changes_l_cte},\n"
        f"{changes_r_cte}\n"
        f"SELECT DISTINCT {left_delta_keys} FROM _changes_l\n"
        f"UNION\n"
        f"SELECT DISTINCT {right_delta_as_left} FROM _changes_r"
    )

    mv_key_match = " AND ".join(
        f"{mv_fqn}.{hlk} IS NOT DISTINCT FROM _affected_keys.{lk}"
        for hlk, lk in zip(hidden_lkey_names, left_key_cols, strict=True)
    )
    delete_sql = (
        f"DELETE FROM {mv_fqn}\n"
        f"WHERE rowid IN (\n"
        f"    SELECT {mv_fqn}.rowid FROM {mv_fqn}\n"
        f"    JOIN _affected_keys ON {mv_key_match}\n"
        f")"
    )

    ak_match = " AND ".join(
        f"{left_name}.{lk} IS NOT DISTINCT FROM _affected_keys.{lk}" for lk in left_key_cols
    )
    insert_sql = (
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT {proj_sql}\n"
        f"FROM {left_fqn} AS {left_name}\n"
        f"    {join_keyword} {right_fqn} AS {right_name} ON {on_sql}\n"
        f"WHERE EXISTS (\n"
        f"    SELECT 1 FROM _affected_keys WHERE {ak_match}\n"
        f")"
    )

    drop_affected = "DROP TABLE IF EXISTS _affected_keys"
    return [create_affected, delete_sql, insert_sql, drop_affected]


def _gen_outer_join_maintain_right(
    changes_l_cte: str,
    changes_r_cte: str,
    left_fqn: str,
    right_fqn: str,
    left_name: str,
    right_name: str,
    left_key_cols: list[str],
    right_key_cols: list[str],
    hidden_lkey_names: list[str],
    hidden_rkey_names: list[str],
    mv_fqn: str,
    proj_sql: str,
    join_keyword: str,
    on_sql: str,
    all_mv_col_names: list[str],
    dialect: str,
) -> list[str]:
    """Generate maintenance SQL for RIGHT JOIN.

    Uses hidden _ivm_rkey_* columns (always non-NULL for preserved right side)
    to reliably identify affected MV rows.
    """
    col_list = ", ".join(all_mv_col_names)

    # Affected right keys: from right delta + left delta mapped via equi-join
    right_delta_keys = ", ".join(f"_changes_r.{k}" for k in right_key_cols)
    left_delta_as_right = ", ".join(
        f"_changes_l.{lk} AS {rk}" for lk, rk in zip(left_key_cols, right_key_cols, strict=True)
    )

    create_affected = (
        f"CREATE TEMP TABLE _affected_keys AS\n"
        f"WITH {changes_l_cte},\n"
        f"{changes_r_cte}\n"
        f"SELECT DISTINCT {right_delta_keys} FROM _changes_r\n"
        f"UNION\n"
        f"SELECT DISTINCT {left_delta_as_right} FROM _changes_l"
    )

    mv_key_match = " AND ".join(
        f"{mv_fqn}.{hrk} IS NOT DISTINCT FROM _affected_keys.{rk}"
        for hrk, rk in zip(hidden_rkey_names, right_key_cols, strict=True)
    )
    delete_sql = (
        f"DELETE FROM {mv_fqn}\n"
        f"WHERE rowid IN (\n"
        f"    SELECT {mv_fqn}.rowid FROM {mv_fqn}\n"
        f"    JOIN _affected_keys ON {mv_key_match}\n"
        f")"
    )

    ak_match = " AND ".join(
        f"{right_name}.{rk} IS NOT DISTINCT FROM _affected_keys.{rk}" for rk in right_key_cols
    )
    insert_sql = (
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT {proj_sql}\n"
        f"FROM {left_fqn} AS {left_name}\n"
        f"    {join_keyword} {right_fqn} AS {right_name} ON {on_sql}\n"
        f"WHERE EXISTS (\n"
        f"    SELECT 1 FROM _affected_keys WHERE {ak_match}\n"
        f")"
    )

    drop_affected = "DROP TABLE IF EXISTS _affected_keys"
    return [create_affected, delete_sql, insert_sql, drop_affected]


def _gen_outer_join_maintain_full(
    changes_l_cte: str,
    changes_r_cte: str,
    left_fqn: str,
    right_fqn: str,
    left_name: str,
    right_name: str,
    left_key_cols: list[str],
    right_key_cols: list[str],
    hidden_lkey_names: list[str],
    hidden_rkey_names: list[str],
    mv_fqn: str,
    proj_sql: str,
    join_keyword: str,
    on_sql: str,
    all_mv_col_names: list[str],
    dialect: str,
) -> list[str]:
    """Generate maintenance SQL for FULL OUTER JOIN.

    Uses hidden _ivm_lkey_* and _ivm_rkey_* columns to reliably identify
    affected MV rows from both sides.
    """
    col_list = ", ".join(all_mv_col_names)

    # Affected left keys: from left delta + right delta mapped via equi-join
    left_delta_keys = ", ".join(f"_changes_l.{k}" for k in left_key_cols)
    right_delta_as_left = ", ".join(
        f"_changes_r.{rk} AS {lk}" for lk, rk in zip(left_key_cols, right_key_cols, strict=True)
    )
    create_affected_left = (
        f"CREATE TEMP TABLE _affected_left AS\n"
        f"WITH {changes_l_cte},\n"
        f"{changes_r_cte}\n"
        f"SELECT DISTINCT {left_delta_keys} FROM _changes_l\n"
        f"UNION\n"
        f"SELECT DISTINCT {right_delta_as_left} FROM _changes_r"
    )

    # Affected right keys: from right delta + left delta mapped via equi-join
    right_delta_keys = ", ".join(f"_changes_r.{k}" for k in right_key_cols)
    left_delta_as_right = ", ".join(
        f"_changes_l.{lk} AS {rk}" for lk, rk in zip(left_key_cols, right_key_cols, strict=True)
    )
    create_affected_right = (
        f"CREATE TEMP TABLE _affected_right AS\n"
        f"WITH {changes_l_cte},\n"
        f"{changes_r_cte}\n"
        f"SELECT DISTINCT {right_delta_keys} FROM _changes_r\n"
        f"UNION\n"
        f"SELECT DISTINCT {left_delta_as_right} FROM _changes_l"
    )

    # Delete: MV rows matching either affected left or right key
    mv_left_match = " AND ".join(
        f"{mv_fqn}.{hlk} IS NOT DISTINCT FROM _affected_left.{lk}"
        for hlk, lk in zip(hidden_lkey_names, left_key_cols, strict=True)
    )
    mv_right_match = " AND ".join(
        f"{mv_fqn}.{hrk} IS NOT DISTINCT FROM _affected_right.{rk}"
        for hrk, rk in zip(hidden_rkey_names, right_key_cols, strict=True)
    )
    delete_sql = (
        f"DELETE FROM {mv_fqn}\n"
        f"WHERE rowid IN (\n"
        f"    SELECT {mv_fqn}.rowid FROM {mv_fqn}\n"
        f"    JOIN _affected_left ON {mv_left_match}\n"
        f") OR rowid IN (\n"
        f"    SELECT {mv_fqn}.rowid FROM {mv_fqn}\n"
        f"    JOIN _affected_right ON {mv_right_match}\n"
        f")"
    )

    # Re-insert: FULL OUTER JOIN scoped to affected keys on both sides
    ak_left_match = " AND ".join(
        f"{left_name}.{lk} IS NOT DISTINCT FROM _affected_left.{lk}" for lk in left_key_cols
    )
    ak_right_match = " AND ".join(
        f"{right_name}.{rk} IS NOT DISTINCT FROM _affected_right.{rk}" for rk in right_key_cols
    )
    insert_sql = (
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT {proj_sql}\n"
        f"FROM {left_fqn} AS {left_name}\n"
        f"    {join_keyword} {right_fqn} AS {right_name} ON {on_sql}\n"
        f"WHERE EXISTS (\n"
        f"    SELECT 1 FROM _affected_left WHERE {ak_left_match}\n"
        f") OR EXISTS (\n"
        f"    SELECT 1 FROM _affected_right WHERE {ak_right_match}\n"
        f")"
    )

    drop_left = "DROP TABLE IF EXISTS _affected_left"
    drop_right = "DROP TABLE IF EXISTS _affected_right"
    return [
        create_affected_left,
        create_affected_right,
        delete_sql,
        insert_sql,
        drop_left,
        drop_right,
    ]


def _subsets_of_size(items: list[str], size: int) -> list[list[str]]:
    """Generate all subsets of the given size from items."""
    if size == 0:
        return [[]]
    if size > len(items):
        return []
    result: list[list[str]] = []

    def _backtrack(start: int, current: list[str]):
        if len(current) == size:
            result.append(list(current))
            return
        for i in range(start, len(items)):
            current.append(items[i])
            _backtrack(i + 1, current)
            current.pop()

    _backtrack(0, [])
    return result


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


# ---------------------------------------------------------------------------
# Stage 10: Set operations (UNION / EXCEPT / INTERSECT)
# ---------------------------------------------------------------------------


def _flatten_set_op(node: exp.Expression, op_type: type) -> list[exp.Select]:
    """Flatten a left-associative chain of the same set operation into branches.

    e.g. Union(Union(A, B), C) -> [A, B, C]
    Only flattens nodes of the same type (won't mix UNION and EXCEPT).
    """
    branches: list[exp.Select] = []
    if isinstance(node, op_type):
        branches.extend(_flatten_set_op(node.this, op_type))
        branches.extend(_flatten_set_op(node.expression, op_type))
    else:
        assert isinstance(node, exp.Select), f"Expected Select branch, got {type(node)}"
        branches.append(node)
    return branches


def _compile_set_operation(
    ast: exp.Union | exp.Except | exp.Intersect,
    *,
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
) -> IVMPlan:
    """Compile a set operation (UNION/EXCEPT/INTERSECT) into IVM maintenance SQL.

    Each branch must be a simple SELECT ... FROM table [WHERE ...].
    """
    is_distinct = bool(ast.args.get("distinct"))

    if isinstance(ast, exp.Union):
        op_name = "union_distinct" if is_distinct else "union_all"
    elif isinstance(ast, exp.Except):
        op_name = "except_all" if not is_distinct else "except_distinct"
    elif isinstance(ast, exp.Intersect):
        op_name = "intersect_all" if not is_distinct else "intersect_distinct"
    else:
        raise UnsupportedSQLError("unknown_set_op", f"Unknown set operation: {type(ast)}")

    # Only support EXCEPT/INTERSECT with ALL (not DISTINCT) for now
    if isinstance(ast, (exp.Except, exp.Intersect)) and is_distinct:
        raise UnsupportedSQLError(
            f"{op_name}",
            f"{op_name.upper().replace('_', ' ')} is not yet supported, use ALL variant",
        )

    # Flatten branches
    branches = _flatten_set_op(ast, type(ast))

    # EXCEPT and INTERSECT must have exactly 2 branches
    if isinstance(ast, (exp.Except, exp.Intersect)) and len(branches) != 2:
        raise UnsupportedSQLError(
            "multi_branch_set_op",
            f"{type(ast).__name__} with more than 2 branches is not supported",
        )

    # Validate each branch is a simple SELECT from one table
    branch_sources: list[dict[str, str]] = []
    for i, branch in enumerate(branches):
        tables = list(branch.find_all(exp.Table))
        if not tables:
            raise UnsupportedSQLError("set_op_no_table", f"Branch {i} has no tables")
        if len(tables) > 1 or branch.args.get("joins"):
            raise UnsupportedSQLError(
                "set_op_complex_branch",
                f"Branch {i} must be a simple single-table SELECT",
            )
        if list(branch.find_all(exp.AggFunc)):
            raise UnsupportedSQLError(
                "set_op_agg_branch",
                f"Branch {i} must not contain aggregates",
            )
        src = _resolve_source(tables[0].name, sources, mv_catalog, mv_schema)
        branch_sources.append(src)

    mv_table = naming.mv_table()
    mv_fqn = f"{mv_catalog}.{mv_schema}.{mv_table}"
    cursors_fqn = f"{mv_catalog}.{mv_schema}.{naming.cursors_table()}"
    create_cursors = _gen_create_cursors(cursors_fqn)

    # Output column names (from first branch)
    out_col_names = _extract_output_col_names(branches[0])

    # --- Resolve unique table sources for cursors ---
    seen_catalogs: set[str] = set()
    init_cursors: list[str] = []
    for src in branch_sources:
        if src["catalog"] not in seen_catalogs:
            init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, src))
            seen_catalogs.add(src["catalog"])

    if isinstance(ast, exp.Union) and not is_distinct:
        return _compile_union_all(
            ast,
            branches,
            branch_sources,
            out_col_names,
            mv_table,
            mv_fqn,
            cursors_fqn,
            create_cursors,
            init_cursors,
            dialect,
            naming,
            sources,
            mv_catalog,
            mv_schema,
        )
    elif isinstance(ast, exp.Union) and is_distinct:
        return _compile_union_distinct(
            ast,
            branches,
            branch_sources,
            out_col_names,
            mv_table,
            mv_fqn,
            cursors_fqn,
            create_cursors,
            init_cursors,
            dialect,
            naming,
            sources,
            mv_catalog,
            mv_schema,
        )
    elif isinstance(ast, exp.Except):
        return _compile_except_all(
            ast,
            branches,
            branch_sources,
            out_col_names,
            mv_table,
            mv_fqn,
            cursors_fqn,
            create_cursors,
            init_cursors,
            dialect,
            naming,
            sources,
            mv_catalog,
            mv_schema,
        )
    else:
        return _compile_intersect_all(
            ast,
            branches,
            branch_sources,
            out_col_names,
            mv_table,
            mv_fqn,
            cursors_fqn,
            create_cursors,
            init_cursors,
            dialect,
            naming,
            sources,
            mv_catalog,
            mv_schema,
        )


def _qualify_branch(branch: exp.Select, src: dict[str, str], dialect: str) -> exp.Select:
    """Qualify a branch SELECT's table references with catalog.schema."""
    table_name = src["table"]
    return branch.copy().transform(  # type: ignore[return-value]
        lambda node: (
            exp.table_(table_name, db=src["schema"], catalog=src["catalog"])
            if isinstance(node, exp.Table) and node.name == table_name
            else node
        )
    )


def _branch_suffix(idx: int) -> str:
    """Generate a unique suffix for branch-specific snapshot variables."""
    return f"b{idx}"


def _compile_union_all(
    ast: exp.Expression,
    branches: list[exp.Select],
    branch_sources: list[dict[str, str]],
    out_col_names: list[str],
    mv_table: str,
    mv_fqn: str,
    cursors_fqn: str,
    create_cursors: str,
    init_cursors: list[str],
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
) -> IVMPlan:
    """UNION ALL: delta of union = union of deltas. Each branch independently."""
    # --- CREATE MV ---
    # Qualify all branches and rebuild the full UNION ALL
    qualified_branches = [
        _qualify_branch(b, s, dialect) for b, s in zip(branches, branch_sources, strict=True)
    ]
    union_sql = " UNION ALL ".join(b.sql(dialect=dialect) for b in qualified_branches)
    create_mv = f"CREATE TABLE {mv_fqn} AS {union_sql}"

    # --- Maintenance ---
    maintain: list[str] = []
    col_list = ", ".join(out_col_names)
    all_proj_cols = col_list

    for i, (branch, src) in enumerate(zip(branches, branch_sources, strict=True)):
        suffix = _branch_suffix(i)
        set_vars = _gen_set_snapshot_vars_named(src, cursors_fqn, mv_table, suffix)
        changes_cte = _gen_changes_cte_named(src, f"_changes_{suffix}", suffix)
        where_clause = _gen_where_clause(branch, dialect)

        # Repoint projection columns to _delta
        select_exprs = [sel.copy().transform(_repoint_columns_to_delta) for sel in branch.selects]
        select_cols_sql = ", ".join(e.sql(dialect=dialect) for e in select_exprs)

        # Build join condition for ROW_NUMBER matching
        join_on_parts = [f"m.{c} IS NOT DISTINCT FROM d.{c}" for c in out_col_names]
        join_on = " AND ".join(join_on_parts)

        # DELETE removed rows
        delete_sql = (
            f"WITH {changes_cte},\n"
            f"_deletes AS (\n"
            f"    SELECT {select_cols_sql},\n"
            f"           ROW_NUMBER() OVER (\n"
            f"               PARTITION BY {all_proj_cols}"
            f" ORDER BY (SELECT NULL)) AS _dn\n"
            f"    FROM _changes_{suffix} AS _delta\n"
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

        # INSERT new rows
        insert_sql = (
            f"WITH {changes_cte}\n"
            f"INSERT INTO {mv_fqn} ({col_list})\n"
            f"SELECT {select_cols_sql}\n"
            f"FROM _changes_{suffix} AS _delta\n"
            f"WHERE _delta.change_type"
            f" IN ('insert', 'update_postimage'){where_clause}"
        )

        maintain.extend(set_vars)
        maintain.append(delete_sql)
        maintain.append(insert_sql)

    # Update cursors (one per unique catalog)
    seen: set[str] = set()
    for src in branch_sources:
        if src["catalog"] not in seen:
            maintain.append(_gen_update_cursor(cursors_fqn, mv_table, src))
            seen.add(src["catalog"])

    features: set[str] = {"select", "union_all"}
    base_tables = {}
    for b, s in zip(branches, branch_sources, strict=True):
        tbl = list(b.find_all(exp.Table))[0].name
        base_tables[tbl] = s["catalog"]

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
    )


def _compile_union_distinct(
    ast: exp.Expression,
    branches: list[exp.Select],
    branch_sources: list[dict[str, str]],
    out_col_names: list[str],
    mv_table: str,
    mv_fqn: str,
    cursors_fqn: str,
    create_cursors: str,
    init_cursors: list[str],
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
) -> IVMPlan:
    """UNION DISTINCT: maintain multiplicity via _ivm_count."""
    count_col = naming.aux_column("count")

    # --- CREATE MV: UNION ALL all branches, then GROUP BY all cols with COUNT ---
    qualified_branches = [
        _qualify_branch(b, s, dialect) for b, s in zip(branches, branch_sources, strict=True)
    ]
    union_sql = " UNION ALL ".join(b.sql(dialect=dialect) for b in qualified_branches)
    group_cols = ", ".join(out_col_names)
    select_cols = ", ".join(out_col_names)
    create_mv = (
        f"CREATE TABLE {mv_fqn} AS\n"
        f"SELECT {select_cols}, COUNT(*) AS {count_col}\n"
        f"FROM ({union_sql}) AS _src\n"
        f"GROUP BY {group_cols}"
    )

    # --- Maintenance: for each branch, compute delta counts and apply ---
    maintain: list[str] = []
    all_mv_cols = out_col_names + [count_col]

    for i, (branch, src) in enumerate(zip(branches, branch_sources, strict=True)):
        suffix = _branch_suffix(i)
        set_vars = _gen_set_snapshot_vars_named(src, cursors_fqn, mv_table, suffix)
        changes_cte = _gen_changes_cte_named(src, f"_changes_{suffix}", suffix)
        where_clause = _gen_where_clause(branch, dialect)

        select_exprs = [sel.copy().transform(_repoint_columns_to_delta) for sel in branch.selects]
        select_cols_sql = ", ".join(e.sql(dialect=dialect) for e in select_exprs)

        join_cond = " AND ".join(f"i.{g} IS NOT DISTINCT FROM d.{g}" for g in out_col_names)
        coalesce_group = ", ".join(f"COALESCE(i.{g}, d.{g}) AS {g}" for g in out_col_names)

        # Create temp table with net delta for this branch
        create_delta = (
            f"CREATE TEMP TABLE _delta_b{i} AS\n"
            f"WITH {changes_cte},\n"
            f"_ins AS (\n"
            f"    SELECT {select_cols_sql}, COUNT(*) AS _ins_cnt\n"
            f"    FROM _changes_{suffix} AS _delta\n"
            f"    WHERE _delta.change_type"
            f" IN ('insert', 'update_postimage'){where_clause}\n"
            f"    GROUP BY {select_cols_sql}\n"
            f"),\n"
            f"_del AS (\n"
            f"    SELECT {select_cols_sql}, COUNT(*) AS _del_cnt\n"
            f"    FROM _changes_{suffix} AS _delta\n"
            f"    WHERE _delta.change_type"
            f" IN ('delete', 'update_preimage'){where_clause}\n"
            f"    GROUP BY {select_cols_sql}\n"
            f")\n"
            f"SELECT {coalesce_group},\n"
            f"    COALESCE(i._ins_cnt, 0) - COALESCE(d._del_cnt, 0) AS _net_count\n"
            f"FROM _ins i FULL OUTER JOIN _del d ON {join_cond}"
        )

        # UPDATE existing rows
        mv_match = " AND ".join(
            f"{mv_fqn}.{g} IS NOT DISTINCT FROM _delta_b{i}.{g}" for g in out_col_names
        )
        update_sql = (
            f"UPDATE {mv_fqn} SET {count_col} = {count_col} + _delta_b{i}._net_count\n"
            f"FROM _delta_b{i}\n"
            f"WHERE {mv_match}"
        )

        # INSERT new groups
        not_exists = " AND ".join(
            f"{mv_fqn}.{g} IS NOT DISTINCT FROM _delta_b{i}.{g}" for g in out_col_names
        )
        insert_cols = ", ".join(all_mv_cols)
        new_group_cols = ", ".join(f"_delta_b{i}.{g}" for g in out_col_names)
        insert_sql = (
            f"INSERT INTO {mv_fqn} ({insert_cols})\n"
            f"SELECT {new_group_cols}, _delta_b{i}._net_count\n"
            f"FROM _delta_b{i}\n"
            f"WHERE _delta_b{i}._net_count > 0\n"
            f"AND NOT EXISTS (\n"
            f"    SELECT 1 FROM {mv_fqn} WHERE {not_exists}\n"
            f")"
        )

        # DELETE empty groups
        delete_sql = f"DELETE FROM {mv_fqn} WHERE {count_col} <= 0"

        drop_delta = f"DROP TABLE IF EXISTS _delta_b{i}"

        maintain.extend(set_vars)
        maintain.extend([create_delta, update_sql, insert_sql, delete_sql, drop_delta])

    # Update cursors
    seen: set[str] = set()
    for src in branch_sources:
        if src["catalog"] not in seen:
            maintain.append(_gen_update_cursor(cursors_fqn, mv_table, src))
            seen.add(src["catalog"])

    features: set[str] = {"select", "union_distinct"}
    base_tables = {}
    for b, s in zip(branches, branch_sources, strict=True):
        tbl = list(b.find_all(exp.Table))[0].name
        base_tables[tbl] = s["catalog"]

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
    )


def _compile_bag_set_op(
    ast: exp.Expression,
    branches: list[exp.Select],
    branch_sources: list[dict[str, str]],
    out_col_names: list[str],
    mv_table: str,
    mv_fqn: str,
    cursors_fqn: str,
    create_cursors: str,
    init_cursors: list[str],
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
    op_keyword: str,
    feature_name: str,
) -> IVMPlan:
    """EXCEPT ALL / INTERSECT ALL via key-scoped recomputation.

    MV stores physical rows (one per bag copy, no hidden count columns).
    Maintenance: find affected row groups from deltas, delete all MV rows
    in those groups, re-insert via the set operation query on current data
    scoped to affected groups.
    """
    left_branch, right_branch = branches
    left_src, right_src = branch_sources

    # --- CREATE MV ---
    left_q = _qualify_branch(left_branch, left_src, dialect)
    right_q = _qualify_branch(right_branch, right_src, dialect)
    create_mv = (
        f"CREATE TABLE {mv_fqn} AS\n"
        f"{left_q.sql(dialect=dialect)}\n"
        f"{op_keyword}\n"
        f"{right_q.sql(dialect=dialect)}"
    )

    # --- Maintenance ---
    maintain: list[str] = []
    col_list = ", ".join(out_col_names)

    # Collect snapshot vars for both branches
    for i, src in enumerate(branch_sources):
        suffix = _branch_suffix(i)
        maintain.extend(_gen_set_snapshot_vars_named(src, cursors_fqn, mv_table, suffix))

    # Build affected row groups: UNION of distinct row values from all deltas
    affected_parts: list[str] = []
    changes_ctes: list[str] = []
    for i, (branch, src) in enumerate(zip(branches, branch_sources, strict=True)):
        suffix = _branch_suffix(i)
        cte = _gen_changes_cte_named(src, f"_changes_{suffix}", suffix)
        changes_ctes.append(cte)
        where_clause = _gen_where_clause(branch, dialect)
        select_exprs = [sel.copy().transform(_repoint_columns_to_delta) for sel in branch.selects]
        select_cols_sql = ", ".join(e.sql(dialect=dialect) for e in select_exprs)
        # WHERE clause needs adjustment: _gen_where_clause returns " AND ..."
        filter_sql = f"\n    WHERE{where_clause[4:]}" if where_clause else ""
        affected_parts.append(
            f"SELECT DISTINCT {select_cols_sql}\nFROM _changes_{suffix} AS _delta{filter_sql}"
        )

    all_ctes = ",\n".join(changes_ctes)
    affected_union = "\nUNION\n".join(affected_parts)
    create_affected = f"CREATE TEMP TABLE _affected AS\nWITH {all_ctes}\n{affected_union}"
    maintain.append(create_affected)

    # DELETE all MV rows matching affected groups
    join_on = " AND ".join(
        f"{mv_fqn}.{c} IS NOT DISTINCT FROM _affected.{c}" for c in out_col_names
    )
    delete_sql = (
        f"DELETE FROM {mv_fqn}\n"
        f"WHERE rowid IN (\n"
        f"    SELECT {mv_fqn}.rowid FROM {mv_fqn}\n"
        f"    JOIN _affected ON {join_on}\n"
        f")"
    )
    maintain.append(delete_sql)

    # RE-INSERT via set operation scoped to affected groups
    def scoped_branch_sql(branch: exp.Select, src: dict[str, str]) -> str:
        q = _qualify_branch(branch, src, dialect)
        q_sql = q.sql(dialect=dialect)
        branch_out = _extract_output_col_names(branch)
        sel_cols = ", ".join(f"_src.{c}" for c in branch_out)
        ak_match = " AND ".join(
            f"_src.{c} IS NOT DISTINCT FROM _affected.{c}" for c in out_col_names
        )
        return (
            f"SELECT {sel_cols} FROM ({q_sql}) AS _src\n"
            f"WHERE EXISTS (\n"
            f"    SELECT 1 FROM _affected WHERE {ak_match}\n"
            f")"
        )

    left_scoped = scoped_branch_sql(left_branch, left_src)
    right_scoped = scoped_branch_sql(right_branch, right_src)

    insert_sql = f"INSERT INTO {mv_fqn} ({col_list})\n{left_scoped}\n{op_keyword}\n{right_scoped}"
    maintain.append(insert_sql)
    maintain.append("DROP TABLE IF EXISTS _affected")

    # Update cursors
    seen: set[str] = set()
    for src in branch_sources:
        if src["catalog"] not in seen:
            maintain.append(_gen_update_cursor(cursors_fqn, mv_table, src))
            seen.add(src["catalog"])

    features: set[str] = {"select", feature_name}
    base_tables = {}
    for b, s in zip(branches, branch_sources, strict=True):
        tbl = list(b.find_all(exp.Table))[0].name
        base_tables[tbl] = s["catalog"]

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
    )


def _compile_except_all(
    ast: exp.Expression,
    branches: list[exp.Select],
    branch_sources: list[dict[str, str]],
    out_col_names: list[str],
    mv_table: str,
    mv_fqn: str,
    cursors_fqn: str,
    create_cursors: str,
    init_cursors: list[str],
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
) -> IVMPlan:
    """EXCEPT ALL via key-scoped recomputation."""
    return _compile_bag_set_op(
        ast,
        branches,
        branch_sources,
        out_col_names,
        mv_table,
        mv_fqn,
        cursors_fqn,
        create_cursors,
        init_cursors,
        dialect,
        naming,
        sources,
        mv_catalog,
        mv_schema,
        op_keyword="EXCEPT ALL",
        feature_name="except_all",
    )


def _compile_intersect_all(
    ast: exp.Expression,
    branches: list[exp.Select],
    branch_sources: list[dict[str, str]],
    out_col_names: list[str],
    mv_table: str,
    mv_fqn: str,
    cursors_fqn: str,
    create_cursors: str,
    init_cursors: list[str],
    dialect: str,
    naming: Naming,
    sources: dict[str, dict] | None,
    mv_catalog: str,
    mv_schema: str,
) -> IVMPlan:
    """INTERSECT ALL via key-scoped recomputation."""
    return _compile_bag_set_op(
        ast,
        branches,
        branch_sources,
        out_col_names,
        mv_table,
        mv_fqn,
        cursors_fqn,
        create_cursors,
        init_cursors,
        dialect,
        naming,
        sources,
        mv_catalog,
        mv_schema,
        op_keyword="INTERSECT ALL",
        feature_name="intersect_all",
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
        for j in ast.args["joins"]:
            if j.side == "LEFT":
                features.add("left_join")
            elif j.side == "RIGHT":
                features.add("right_join")
            elif j.side == "FULL":
                features.add("full_join")
    for agg in ast.find_all(exp.AggFunc):
        if isinstance(agg, exp.Sum):
            features.add("sum")
        elif isinstance(agg, exp.Count):
            features.add("count")
        elif isinstance(agg, exp.Avg):
            features.add("avg")
        elif isinstance(agg, exp.Min):
            features.add("min")
        elif isinstance(agg, exp.Max):
            features.add("max")
    return features
