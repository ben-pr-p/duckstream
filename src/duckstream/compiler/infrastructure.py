"""Shared infrastructure for SQL generation: DDL, cursors, snapshots, column rewriting."""

from __future__ import annotations

from sqlglot import exp

from duckstream.materialized_view import Naming


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

    # Strip HAVING — MV stores all groups; HAVING is applied at read time via query_mv
    if qualified_ast.args.get("having"):
        qualified_ast.args.pop("having")

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

    # Strip HAVING — MV stores all groups; HAVING is applied at read time via query_mv
    if qualified_ast.args.get("having"):
        qualified_ast.args.pop("having")

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


def _detect_features(ast: exp.Select) -> set[str]:
    features: set[str] = {"select"}
    if ast.args.get("where"):
        features.add("where")
    if ast.args.get("distinct"):
        features.add("distinct")
    if ast.args.get("group"):
        features.add("group_by")
    if ast.args.get("having"):
        features.add("having")
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


# ---------------------------------------------------------------------------
# HAVING support
# ---------------------------------------------------------------------------


def _having_to_where(
    having_expr: exp.Expression,
    ast: exp.Select,
    naming: Naming,
    dialect: str,
) -> str:
    """Convert a HAVING expression to a WHERE clause for query_mv.

    Maps aggregate function references to their MV column aliases.
    COUNT(*) without a SELECT alias maps to the internal _ivm_count column.
    """
    # Build mapping: normalized aggregate SQL -> MV column alias
    agg_to_alias: dict[str, str] = {}
    for sel in ast.selects:
        alias_name = sel.alias if isinstance(sel, exp.Alias) else None
        inner = sel.this if isinstance(sel, exp.Alias) else sel
        if isinstance(inner, exp.AggFunc) and alias_name:
            # Normalize: generate SQL without table qualifiers for matching
            norm_key = inner.sql(dialect=dialect)
            agg_to_alias[norm_key] = alias_name

    ivm_count = naming.aux_column("count")

    def _replace_agg(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.AggFunc):
            # Check for exact match in SELECT aliases
            norm = node.sql(dialect=dialect)
            if norm in agg_to_alias:
                return exp.column(agg_to_alias[norm])
            # COUNT(*) maps to _ivm_count
            if isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
                return exp.column(ivm_count)
        return node

    where_expr = having_expr.copy().transform(_replace_agg)
    return where_expr.sql(dialect=dialect)


def _gen_query_mv(
    ast: exp.Select,
    mv_fqn: str,
    naming: Naming,
    dialect: str,
) -> str:
    """Generate a SELECT query to read visible MV rows, excluding _ivm_* columns.

    For views with HAVING, adds a WHERE clause that filters by the HAVING condition.
    """
    # Collect visible column names (non-_ivm_* columns from the SELECT)
    visible_cols: list[str] = []
    for sel in ast.selects:
        if isinstance(sel, exp.Alias):
            name = sel.alias
        elif isinstance(sel, exp.Column):
            name = sel.name
        else:
            name = sel.sql(dialect=dialect)
        if not name.startswith("_ivm_"):
            visible_cols.append(name)

    cols_sql = ", ".join(visible_cols) if visible_cols else "*"
    query = f"SELECT {cols_sql} FROM {mv_fqn}"

    having_node = ast.args.get("having")
    if having_node:
        where_sql = _having_to_where(having_node.this, ast, naming, dialect)
        query += f" WHERE {where_sql}"

    return query
