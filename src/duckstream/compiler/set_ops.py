"""Stage 10: Set operations (UNION / EXCEPT / INTERSECT)."""

from __future__ import annotations

from sqlglot import exp

from duckstream.compiler.infrastructure import (
    _extract_output_col_names,
    _gen_changes_cte_named,
    _gen_init_cursor,
    _gen_set_snapshot_vars_named,
    _gen_update_cursor,
    _gen_where_clause,
    _repoint_columns_to_delta,
    _resolve_source,
)
from duckstream.materialized_view import MaterializedView, Naming, UnsupportedSQLError


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
) -> MaterializedView:
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
    create_cursors = _gen_create_cursors_inline(cursors_fqn)

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


def _gen_create_cursors_inline(cursors_fqn: str) -> str:
    """Inline version of _gen_create_cursors for set_ops (avoids circular import)."""
    return (
        f"CREATE TABLE IF NOT EXISTS {cursors_fqn} (\n"
        f"    mv_name VARCHAR,\n"
        f"    source_catalog VARCHAR,\n"
        f"    last_snapshot BIGINT\n"
        f")"
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
) -> MaterializedView:
    """UNION ALL: delta of union = union of deltas. Each branch independently."""
    # --- CREATE MV ---
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

        select_exprs = [sel.copy().transform(_repoint_columns_to_delta) for sel in branch.selects]
        select_cols_sql = ", ".join(e.sql(dialect=dialect) for e in select_exprs)

        join_on_parts = [f"m.{c} IS NOT DISTINCT FROM d.{c}" for c in out_col_names]
        join_on = " AND ".join(join_on_parts)

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

    visible_cols = [c for c in out_col_names if not c.startswith("_ivm_")]
    cols_sql = ", ".join(visible_cols) if visible_cols else "*"
    query_mv = f"SELECT {cols_sql} FROM {mv_fqn}"

    return MaterializedView(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
        query_mv=query_mv,
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
) -> MaterializedView:
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

        mv_match = " AND ".join(
            f"{mv_fqn}.{g} IS NOT DISTINCT FROM _delta_b{i}.{g}" for g in out_col_names
        )
        update_sql = (
            f"UPDATE {mv_fqn} SET {count_col} = {count_col} + _delta_b{i}._net_count\n"
            f"FROM _delta_b{i}\n"
            f"WHERE {mv_match}"
        )

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

    visible_cols = [c for c in out_col_names if not c.startswith("_ivm_")]
    cols_sql = ", ".join(visible_cols) if visible_cols else "*"
    query_mv = f"SELECT {cols_sql} FROM {mv_fqn}"

    return MaterializedView(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
        query_mv=query_mv,
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
) -> MaterializedView:
    """EXCEPT ALL / INTERSECT ALL via key-scoped recomputation."""
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

    visible_cols = [c for c in out_col_names if not c.startswith("_ivm_")]
    cols_sql = ", ".join(visible_cols) if visible_cols else "*"
    query_mv = f"SELECT {cols_sql} FROM {mv_fqn}"

    return MaterializedView(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables=base_tables,
        features=features,
        query_mv=query_mv,
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
) -> MaterializedView:
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
) -> MaterializedView:
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
