"""Stage 9: Outer JOIN maintenance (LEFT / RIGHT / FULL)."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from duckstream.compiler.infrastructure import (
    _detect_features,
    _extract_output_col_names,
    _gen_changes_cte_named,
    _gen_create_mv_join,
    _gen_init_cursor,
    _gen_query_mv,
    _gen_set_snapshot_vars_named,
    _gen_update_cursor,
    _resolve_source,
)
from duckstream.materialized_view import MaterializedView, Naming, UnsupportedSQLError


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
) -> MaterializedView:
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

    query_mv = _gen_query_mv(ast, mv_fqn, naming, dialect)

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
