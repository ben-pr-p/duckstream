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
    """Compile an N-table outer join view with mixed join types.

    Uses key-scoped recomputation: for each affected join key in the delta,
    delete all MV rows with that key and re-insert via the full query on
    the updated tables. Supports any mix of INNER and LEFT JOINs.

    The "key" for recomputation is always derived from the FROM (preserved)
    table's join columns in the first LEFT JOIN's ON condition.
    """
    # --- Extract all table names from FROM + joins ---
    from_table_node = ast.args["from_"].this
    from_name = from_table_node.name

    table_names_ordered = [from_name]
    for join_node in joins:
        table_names_ordered.append(join_node.this.name)

    # Resolve sources and build FQNs for all tables
    table_sources: dict[str, dict[str, str]] = {}
    table_fqns: dict[str, str] = {}
    for tname in table_names_ordered:
        src = _resolve_source(tname, sources, mv_catalog, mv_schema)
        table_sources[tname] = src
        table_fqns[tname] = f"{src['catalog']}.{src['schema']}.{tname}"

    out_col_names = _extract_output_col_names(ast)

    # --- Find the first outer JOIN to extract the recomputation key ---
    first_outer_join = None
    for join_node in joins:
        if join_node.side in ("LEFT", "RIGHT", "FULL"):
            first_outer_join = join_node
            break

    if first_outer_join is None:
        raise UnsupportedSQLError(
            "outer_join_no_outer",
            "Outer join compiler requires at least one outer JOIN",
        )

    first_outer_side = first_outer_join.side
    first_right_name = first_outer_join.this.name
    on_condition = first_outer_join.args.get("on")
    left_key_cols, right_key_cols = _extract_join_keys(on_condition, from_name, first_right_name)

    # --- Hidden key columns for outer join maintenance ---
    hidden_lkey_names = [naming.aux_column(f"lkey_{c}") for c in left_key_cols]
    hidden_rkey_names = [naming.aux_column(f"rkey_{c}") for c in right_key_cols]

    hidden_key_exprs: list[str] = []
    for lk, hname in zip(left_key_cols, hidden_lkey_names, strict=True):
        hidden_key_exprs.append(f"{from_name}.{lk} AS {hname}")
    for rk, hname in zip(right_key_cols, hidden_rkey_names, strict=True):
        hidden_key_exprs.append(f"{first_right_name}.{rk} AS {hname}")

    # --- DDL ---
    base_create = _gen_create_mv_join(ast, mv_fqn, table_sources, dialect)
    as_idx = base_create.index(" AS SELECT ") + len(" AS ")
    create_select = base_create[as_idx:]
    create_ast = sqlglot.parse_one(create_select, dialect=dialect)
    for hexpr in hidden_key_exprs:
        parsed_expr = sqlglot.parse_one(hexpr, into=exp.Expression, dialect=dialect)
        create_ast.args["expressions"].append(parsed_expr)
    create_mv = f"CREATE TABLE {mv_fqn} AS {create_ast.sql(dialect=dialect)}"

    seen_catalogs: set[str] = set()
    init_cursors = []
    for tname in table_names_ordered:
        src = table_sources[tname]
        if src["catalog"] not in seen_catalogs:
            init_cursors.append(_gen_init_cursor(cursors_fqn, mv_table, src))
            seen_catalogs.add(src["catalog"])

    # --- Snapshot vars and changes CTEs for ALL tables ---
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

    # --- Build the full join SQL fragment (for re-insertion) ---
    join_clauses: list[str] = []
    for join_node in joins:
        jname = join_node.this.name
        jfqn = table_fqns[jname]
        jon_sql = join_node.args.get("on").sql(dialect=dialect)
        if join_node.side == "LEFT":
            join_clauses.append(f"    LEFT JOIN {jfqn} AS {jname} ON {jon_sql}")
        elif join_node.side == "RIGHT":
            join_clauses.append(f"    RIGHT JOIN {jfqn} AS {jname} ON {jon_sql}")
        elif join_node.side == "FULL":
            join_clauses.append(f"    FULL OUTER JOIN {jfqn} AS {jname} ON {jon_sql}")
        else:
            join_clauses.append(f"    JOIN {jfqn} AS {jname} ON {jon_sql}")
    joins_sql = "\n".join(join_clauses)

    # Projection SQL with fully-qualified table refs (includes hidden key cols)
    proj_parts = []
    has_distinct = bool(ast.args.get("distinct"))
    distinct_sql = "DISTINCT " if has_distinct else ""
    for sel in ast.selects:
        proj_parts.append(sel.sql(dialect=dialect))
    for hexpr in hidden_key_exprs:
        proj_parts.append(hexpr)
    proj_sql = ", ".join(proj_parts)

    all_mv_col_names = out_col_names + hidden_lkey_names + hidden_rkey_names

    original_where = ast.args.get("where")
    where_sql = ""
    if original_where:
        where_sql = f" AND {original_where.this.sql(dialect=dialect)}"

    # --- Build affected keys and maintenance SQL ---
    changes_cte_list = [changes_ctes[tname] for tname in table_names_ordered]
    changes_cte_sql = ",\n".join(changes_cte_list)
    col_list = ", ".join(all_mv_col_names)

    if first_outer_side == "FULL":
        # FULL OUTER JOIN: need separate affected key sets for both sides
        maintain_stmts = _gen_maintain_full(
            table_names_ordered,
            from_name,
            first_right_name,
            joins,
            left_key_cols,
            right_key_cols,
            hidden_lkey_names,
            hidden_rkey_names,
            changes_cte_sql,
            mv_fqn,
            col_list,
            distinct_sql,
            proj_sql,
            table_fqns,
            joins_sql,
            where_sql,
        )
    else:
        # LEFT or RIGHT JOIN: key on the preserved side
        if first_outer_side == "RIGHT":
            # Preserved side = right. Key on right_key_cols.
            key_cols = right_key_cols
            key_table = first_right_name
            hidden_key_names = hidden_rkey_names
        else:
            # LEFT JOIN (or mixed with LEFT). Preserved side = FROM (left).
            key_cols = left_key_cols
            key_table = from_name
            hidden_key_names = hidden_lkey_names

        affected_key_parts: list[str] = []
        for idx, tname in enumerate(table_names_ordered):
            cte_name = f"_changes_{idx}"
            if tname == key_table:
                cols = ", ".join(f"{cte_name}.{k}" for k in key_cols)
                affected_key_parts.append(f"SELECT DISTINCT {cols} FROM {cte_name}")
            else:
                join_for_table = None
                for jn in joins:
                    if jn.this.name == tname:
                        join_for_table = jn
                        break
                if join_for_table is None:
                    # FROM table when key_table is a joined table (RIGHT JOIN)
                    on_cond = first_outer_join.args.get("on")
                    mapped = _map_keys_through_join(on_cond, key_table, tname, key_cols)
                    if mapped:
                        cols = ", ".join(
                            f"{cte_name}.{mc} AS {kc}"
                            for mc, kc in zip(mapped, key_cols, strict=True)
                        )
                        affected_key_parts.append(f"SELECT DISTINCT {cols} FROM {cte_name}")
                    continue
                jon = join_for_table.args.get("on")
                mapped_cols = _map_keys_through_join(jon, key_table, tname, key_cols)
                if not mapped_cols:
                    # Try reverse direction (key_table might be on right side)
                    mapped_cols = _map_keys_through_join(jon, from_name, tname, key_cols)
                if mapped_cols:
                    cols = ", ".join(
                        f"{cte_name}.{mc} AS {lk}"
                        for mc, lk in zip(mapped_cols, key_cols, strict=True)
                    )
                    affected_key_parts.append(f"SELECT DISTINCT {cols} FROM {cte_name}")

        affected_union = "\nUNION\n".join(affected_key_parts)
        create_affected = (
            f"CREATE TEMP TABLE _affected_keys AS\nWITH {changes_cte_sql}\n{affected_union}"
        )

        mv_key_match = " AND ".join(
            f"{mv_fqn}.{hk} IS NOT DISTINCT FROM _affected_keys.{kc}"
            for hk, kc in zip(hidden_key_names, key_cols, strict=True)
        )
        delete_sql = (
            f"DELETE FROM {mv_fqn}\n"
            f"WHERE rowid IN (\n"
            f"    SELECT {mv_fqn}.rowid FROM {mv_fqn}\n"
            f"    JOIN _affected_keys ON {mv_key_match}\n"
            f")"
        )

        ak_match = " AND ".join(
            f"{key_table}.{kc} IS NOT DISTINCT FROM _affected_keys.{kc}" for kc in key_cols
        )
        insert_sql = (
            f"INSERT INTO {mv_fqn} ({col_list})\n"
            f"SELECT {distinct_sql}{proj_sql}\n"
            f"FROM {table_fqns[from_name]} AS {from_name}\n"
            f"{joins_sql}\n"
            f"WHERE EXISTS (\n"
            f"    SELECT 1 FROM _affected_keys WHERE {ak_match}\n"
            f"){where_sql}"
        )

        drop_affected = "DROP TABLE IF EXISTS _affected_keys"
        maintain_stmts = [create_affected, delete_sql, insert_sql, drop_affected]

    update_cursors = [
        _gen_update_cursor(cursors_fqn, mv_table, table_sources[tname])
        for tname in table_names_ordered
    ]

    maintain = [*all_set_vars, *maintain_stmts, *update_cursors]

    features = _detect_features(ast)
    features.add("join")
    for join_node in joins:
        if join_node.side:
            features.add(f"{join_node.side.lower()}_join")
    base_tables = {tname: table_sources[tname]["catalog"] for tname in table_names_ordered}

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


def _map_keys_through_join(
    on_condition: exp.Expression,
    from_name: str,
    join_table_name: str,
    from_key_cols: list[str],
) -> list[str] | None:
    """Map FROM table key columns to join table columns via ON condition.

    Given ON condition like `join_table.x = from_table.id`, and from_key_cols = ["id"],
    returns ["x"] — the join_table columns that correspond to the from_table keys.

    Returns None if the mapping cannot be established.
    """
    # Build mapping: from_col -> join_col from equi-join predicates
    from_to_join: dict[str, str] = {}

    eqs = list(on_condition.find_all(exp.EQ))
    if not eqs and isinstance(on_condition, exp.EQ):
        eqs = [on_condition]

    for eq in eqs:
        left_col = eq.left
        right_col = eq.right
        if isinstance(left_col, exp.Column) and isinstance(right_col, exp.Column):
            if left_col.table == from_name and right_col.table == join_table_name:
                from_to_join[left_col.name] = right_col.name
            elif left_col.table == join_table_name and right_col.table == from_name:
                from_to_join[right_col.name] = left_col.name

    # Map each from_key_col to the corresponding join table column
    result = []
    for fk in from_key_cols:
        if fk in from_to_join:
            result.append(from_to_join[fk])
        else:
            return None
    return result


def _gen_maintain_full(
    table_names_ordered: list[str],
    from_name: str,
    right_name: str,
    joins: list,
    left_key_cols: list[str],
    right_key_cols: list[str],
    hidden_lkey_names: list[str],
    hidden_rkey_names: list[str],
    changes_cte_sql: str,
    mv_fqn: str,
    col_list: str,
    distinct_sql: str,
    proj_sql: str,
    table_fqns: dict[str, str],
    joins_sql: str,
    where_sql: str,
) -> list[str]:
    """Generate maintenance SQL for FULL OUTER JOIN (2-table only)."""
    from_idx = table_names_ordered.index(from_name)
    right_idx = table_names_ordered.index(right_name)
    from_cte = f"_changes_{from_idx}"
    right_cte = f"_changes_{right_idx}"

    # Affected left keys
    left_delta_keys = ", ".join(f"{from_cte}.{k}" for k in left_key_cols)
    right_delta_as_left = ", ".join(
        f"{right_cte}.{rk} AS {lk}" for lk, rk in zip(left_key_cols, right_key_cols, strict=True)
    )
    create_affected_left = (
        f"CREATE TEMP TABLE _affected_left AS\n"
        f"WITH {changes_cte_sql}\n"
        f"SELECT DISTINCT {left_delta_keys} FROM {from_cte}\n"
        f"UNION\n"
        f"SELECT DISTINCT {right_delta_as_left} FROM {right_cte}"
    )

    # Affected right keys
    right_delta_keys = ", ".join(f"{right_cte}.{k}" for k in right_key_cols)
    left_delta_as_right = ", ".join(
        f"{from_cte}.{lk} AS {rk}" for lk, rk in zip(left_key_cols, right_key_cols, strict=True)
    )
    create_affected_right = (
        f"CREATE TEMP TABLE _affected_right AS\n"
        f"WITH {changes_cte_sql}\n"
        f"SELECT DISTINCT {right_delta_keys} FROM {right_cte}\n"
        f"UNION\n"
        f"SELECT DISTINCT {left_delta_as_right} FROM {from_cte}"
    )

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

    ak_left_match = " AND ".join(
        f"{from_name}.{lk} IS NOT DISTINCT FROM _affected_left.{lk}" for lk in left_key_cols
    )
    ak_right_match = " AND ".join(
        f"{right_name}.{rk} IS NOT DISTINCT FROM _affected_right.{rk}" for rk in right_key_cols
    )

    # Build join keyword for 2-table full outer join
    join_on_sql = joins[0].args.get("on").sql("duckdb")
    right_fqn = table_fqns[right_name]
    insert_sql = (
        f"INSERT INTO {mv_fqn} ({col_list})\n"
        f"SELECT {distinct_sql}{proj_sql}\n"
        f"FROM {table_fqns[from_name]} AS {from_name}\n"
        f"    FULL OUTER JOIN {right_fqn} AS {right_name} ON {join_on_sql}\n"
        f"WHERE (EXISTS (\n"
        f"    SELECT 1 FROM _affected_left WHERE {ak_left_match}\n"
        f") OR EXISTS (\n"
        f"    SELECT 1 FROM _affected_right WHERE {ak_right_match}\n"
        f")){where_sql}"
    )

    return [
        create_affected_left,
        create_affected_right,
        delete_sql,
        insert_sql,
        "DROP TABLE IF EXISTS _affected_left",
        "DROP TABLE IF EXISTS _affected_right",
    ]
