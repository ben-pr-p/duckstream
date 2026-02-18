"""Stage 4: N-table inner JOIN maintenance."""

from __future__ import annotations

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
    _subsets_of_size,
)
from duckstream.plan import IVMPlan, Naming


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
