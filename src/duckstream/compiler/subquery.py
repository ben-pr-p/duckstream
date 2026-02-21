"""Subquery rewriting pass: rewrites subqueries into joins before IVM compilation.

This is a preprocessing pass that runs before the router dispatches to specific
compiler stages. It transforms subqueries into equivalent join forms that the
existing IVM compiler can handle:

- FROM subqueries (derived tables): inlined/flattened into the outer query
- IN (subquery) -> semi-join (INNER JOIN + DISTINCT or EXISTS)
- NOT IN (subquery) -> anti-join (LEFT JOIN + IS NULL)
- EXISTS (subquery) -> semi-join
- NOT EXISTS (subquery) -> anti-join (LEFT JOIN + IS NULL)
- Scalar subquery in SELECT -> recursive inner MV + LEFT JOIN to outer query
- Complex FROM subquery (agg/group/distinct) -> recursive inner MV
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

from sqlglot import exp

from duckstream.materialized_view import MaterializedView, Naming, UnsupportedSQLError


@dataclass
class CompilationContext:
    """Context passed from the router for recursive MV compilation."""

    naming: Naming
    mv_catalog: str
    mv_schema: str
    sources: dict[str, dict] | None
    dialect: str


def rewrite_subqueries(
    ast: exp.Select,
    dialect: str = "duckdb",
    ctx: CompilationContext | None = None,
) -> tuple[exp.Select, list[MaterializedView]]:
    """Main entry point: rewrite all subqueries in the AST.

    Applies rewrites in order:
    1. FROM subqueries (derived tables) — simple ones flattened, complex ones -> inner MV
    2. WHERE subqueries (IN/NOT IN/EXISTS/NOT EXISTS)
    3. SELECT scalar subqueries -> inner MV (scalar lookup replaces aggregate)

    Returns (rewritten_ast, list_of_inner_mvs).
    """
    inner_mvs: list[MaterializedView] = []
    ast, from_mvs = _rewrite_from_subqueries(ast, dialect, ctx)
    inner_mvs.extend(from_mvs)
    ast = _rewrite_where_subqueries(ast, dialect)
    ast, select_mvs = _rewrite_select_subqueries(ast, dialect, ctx)
    inner_mvs.extend(select_mvs)
    return ast, inner_mvs


def has_subqueries(ast: exp.Expression) -> bool:
    """Check if an AST contains any subqueries that need rewriting."""
    # Check FROM clause for subqueries/derived tables
    from_clause = ast.args.get("from_")
    if from_clause and isinstance(from_clause.this, exp.Subquery):
        return True
    # Check joins for subqueries in the table position
    joins = ast.args.get("joins") or []
    for j in joins:
        if isinstance(j.this, exp.Subquery):
            return True
    # Check WHERE for IN/EXISTS with subqueries
    where = ast.args.get("where")
    if where and _has_subquery_predicates(where):
        return True
    # Check SELECT for scalar subqueries
    return any(list(sel.find_all(exp.Subquery)) for sel in ast.args.get("expressions", []))


def _has_subquery_predicates(node: exp.Expression) -> bool:
    """Check if a WHERE clause contains subquery predicates."""
    for n in node.walk():
        if (
            isinstance(n, exp.In)
            and n.args.get("query")
            and isinstance(n.args["query"], exp.Subquery)
        ):
            return True
        # EXISTS wraps Select directly, not in Subquery
        if isinstance(n, exp.Exists) and isinstance(n.this, exp.Select):
            return True
        if isinstance(n, exp.Subquery):
            return True
    return False


# ---------------------------------------------------------------------------
# FROM subqueries (derived tables)
# ---------------------------------------------------------------------------


def _rewrite_from_subqueries(
    ast: exp.Select, dialect: str, ctx: CompilationContext | None = None
) -> tuple[exp.Select, list[MaterializedView]]:
    """Flatten derived tables (subqueries in FROM) into the outer query.

    Supports simple cases:
    - SELECT ... FROM (SELECT ... FROM t [WHERE ...]) alias [WHERE ...]
    - Derived table with simple projection and filtering

    For complex cases (aggregation, joins, etc.), compiles as a recursive inner MV
    if a CompilationContext is provided.
    """
    ast = ast.copy()
    inner_mvs: list[MaterializedView] = []

    # Check FROM clause
    from_clause = ast.args.get("from_")
    if not from_clause:
        return ast, inner_mvs

    from_node = from_clause.this

    # Handle subquery in FROM position
    if isinstance(from_node, exp.Subquery):
        inner_select = from_node.this
        alias_name = from_node.alias

        if isinstance(inner_select, exp.Select):
            if _is_simple_from_subquery(inner_select):
                ast = _flatten_derived_table(ast, inner_select, alias_name, dialect)
            elif ctx is not None:
                ast, mv, _mv_name = _compile_from_subquery_as_mv(
                    ast, inner_select, alias_name, dialect, ctx
                )
                inner_mvs.append(mv)
            else:
                _raise_complex_from_error(inner_select)

    # Also handle subqueries in JOIN positions
    joins = ast.args.get("joins") or []
    for join_node in joins:
        if isinstance(join_node.this, exp.Subquery):
            inner_select = join_node.this.this
            alias_name = join_node.this.alias

            if not isinstance(inner_select, exp.Select):
                continue

            # For join subqueries, try to inline
            ast = _flatten_join_subquery(ast, join_node, inner_select, alias_name, dialect)

    return ast, inner_mvs


def _is_simple_from_subquery(inner: exp.Select) -> bool:
    """Check if a FROM subquery is simple enough to flatten (no agg/group/distinct/joins)."""
    if inner.args.get("group") or inner.args.get("having"):
        return False
    if inner.args.get("distinct"):
        return False
    if inner.args.get("joins"):
        return False
    if list(inner.find_all(exp.AggFunc)):
        return False
    inner_tables = list(inner.find_all(exp.Table))
    return len(inner_tables) == 1


def _raise_complex_from_error(inner: exp.Select) -> None:
    """Raise appropriate UnsupportedSQLError for complex FROM subqueries."""
    if inner.args.get("group") or inner.args.get("having") or list(inner.find_all(exp.AggFunc)):
        raise UnsupportedSQLError(
            "from_subquery_agg",
            "Subqueries with GROUP BY/HAVING/aggregates in FROM clause require "
            "a CompilationContext for recursive MV compilation",
        )
    if inner.args.get("distinct"):
        raise UnsupportedSQLError(
            "from_subquery_distinct",
            "Subqueries with DISTINCT in FROM clause require "
            "a CompilationContext for recursive MV compilation",
        )
    if inner.args.get("joins"):
        raise UnsupportedSQLError(
            "from_subquery_join",
            "Subqueries with JOINs in FROM clause require "
            "a CompilationContext for recursive MV compilation",
        )
    inner_tables = list(inner.find_all(exp.Table))
    if len(inner_tables) != 1:
        raise UnsupportedSQLError(
            "from_subquery_multi_table",
            "Only single-table subqueries in FROM are supported",
        )


def _compile_from_subquery_as_mv(
    outer: exp.Select,
    inner: exp.Select,
    alias: str,
    dialect: str,
    ctx: CompilationContext,
) -> tuple[exp.Select, MaterializedView, str]:
    """Compile a complex FROM subquery as a recursive inner MV.

    The inner query becomes a separate MV. The outer query is rewritten
    to reference the inner MV table instead of the subquery.

    Returns (rewritten_outer, inner_mv, inner_mv_table_name).
    """
    from duckstream.compiler.router import compile_ivm

    inner_sql = inner.sql(dialect=dialect)
    inner_mv_name = _inner_mv_name(ctx.naming.mv_table(), inner_sql)

    inner_naming = _make_inner_naming(ctx.naming, inner_mv_name)

    inner_mv = compile_ivm(
        inner_sql,
        naming=inner_naming,
        mv_catalog=ctx.mv_catalog,
        mv_schema=ctx.mv_schema,
        sources=ctx.sources,
    )

    # Replace FROM subquery with a table reference to the inner MV
    mv_table_ref = exp.table_(inner_mv_name, db=ctx.mv_schema, catalog=ctx.mv_catalog)
    outer.args["from_"] = exp.From(this=mv_table_ref)

    # Rewrite column references from alias.col to inner_mv_name.col
    # The inner MV columns are the output columns of the inner SELECT
    inner_col_names = set()
    for sel in inner.selects:
        if isinstance(sel, exp.Alias):
            inner_col_names.add(sel.alias)
        elif isinstance(sel, exp.Column):
            inner_col_names.add(sel.name)

    def _rewrite_col_ref(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and node.table == alias:
            return exp.column(node.name, table=inner_mv_name)
        return node

    outer = cast(exp.Select, outer.transform(_rewrite_col_ref))
    return outer, inner_mv, inner_mv_name


def _flatten_derived_table(
    outer: exp.Select,
    inner: exp.Select,
    alias: str,
    dialect: str,
) -> exp.Select:
    """Flatten a derived table into the outer query.

    Given: SELECT ... FROM (SELECT ... FROM t WHERE ...) AS alias WHERE ...
    Produce: SELECT ... FROM t WHERE ... AND ...

    Only handles simple cases without aggregation/joins/distinct in the inner query.
    The caller must verify _is_simple_from_subquery() before calling this.
    """
    # Get inner table
    inner_tables = list(inner.find_all(exp.Table))
    assert len(inner_tables) == 1, "Caller must verify single-table subquery"
    inner_table = inner_tables[0]
    inner_table_name = inner_table.alias_or_name

    # Build column mapping: alias.col -> inner expression
    # The inner SELECT may rename columns: SELECT a AS x, b AS y FROM t
    col_mapping: dict[str, exp.Expression] = {}
    for sel in inner.selects:
        if isinstance(sel, exp.Alias):
            out_name = sel.alias
            col_mapping[out_name] = sel.this.copy()
        elif isinstance(sel, exp.Column):
            col_mapping[sel.name] = sel.copy()
        elif isinstance(sel, exp.Star):
            # SELECT * - all columns pass through, no mapping needed
            pass

    # Replace the FROM clause with the inner table
    outer.args["from_"] = exp.From(this=inner_table.copy())

    # Rewrite outer column references: alias.col -> inner_table.col
    def _rewrite_col_ref(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and node.table == alias:
            # Look up in mapping
            if node.name in col_mapping:
                mapped = col_mapping[node.name].copy()
                # If mapped expr is a Column and has no table, qualify with inner table
                if isinstance(mapped, exp.Column) and not mapped.table:
                    return exp.column(mapped.name, table=inner_table_name)
                return mapped
            else:
                # No mapping (SELECT *) - just repoint to inner table
                return exp.column(node.name, table=inner_table_name)
        return node

    outer = cast(exp.Select, outer.transform(_rewrite_col_ref))

    # Merge inner WHERE into outer WHERE
    inner_where = inner.args.get("where")
    if inner_where:
        outer_where = outer.args.get("where")
        if outer_where:
            # Combine: outer_where AND inner_where
            combined = exp.And(this=outer_where.this, expression=inner_where.this.copy())
            outer.args["where"] = exp.Where(this=combined)
        else:
            outer.args["where"] = inner_where.copy()

    return outer


def _flatten_join_subquery(
    outer: exp.Select,
    join_node: exp.Join,
    inner: exp.Select,
    alias: str,
    dialect: str,
) -> exp.Select:
    """Flatten a subquery used as a join target.

    Replaces JOIN (SELECT ... FROM t WHERE ...) AS alias ON ...
    with JOIN t AS alias ON ... (with WHERE conditions merged into ON).
    """
    # Only handle simple inner queries
    if (
        inner.args.get("group")
        or inner.args.get("having")
        or inner.args.get("distinct")
        or inner.args.get("joins")
        or list(inner.find_all(exp.AggFunc))
    ):
        raise UnsupportedSQLError(
            "join_subquery_complex",
            "Only simple single-table subqueries are supported in JOIN position",
        )

    inner_tables = list(inner.find_all(exp.Table))
    if len(inner_tables) != 1:
        raise UnsupportedSQLError(
            "join_subquery_multi_table",
            "Only single-table subqueries in JOIN are supported",
        )

    inner_table = inner_tables[0]

    # Replace the subquery with the actual table, preserving the alias
    new_table = inner_table.copy()
    if alias:
        new_table.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    join_node.set("this", new_table)

    # Build column mapping for the inner SELECT
    inner_table_name = inner_table.alias_or_name
    col_mapping: dict[str, exp.Expression] = {}
    for sel in inner.selects:
        if isinstance(sel, exp.Alias):
            col_mapping[sel.alias] = sel.this.copy()
        elif isinstance(sel, exp.Column):
            col_mapping[sel.name] = sel.copy()

    # Rewrite outer references to the alias
    def _rewrite_col_ref(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and node.table == alias:
            if node.name in col_mapping:
                mapped = col_mapping[node.name].copy()
                if isinstance(mapped, exp.Column) and not mapped.table:
                    return exp.column(mapped.name, table=alias)
                # Repoint table qualifier to alias
                if isinstance(mapped, exp.Column):
                    return exp.column(mapped.name, table=alias)
                return mapped
            else:
                return exp.column(node.name, table=alias)
        return node

    outer = cast(exp.Select, outer.transform(_rewrite_col_ref))

    # Merge inner WHERE into join ON condition
    inner_where = inner.args.get("where")
    if inner_where:
        on_condition = join_node.args.get("on")
        if on_condition:
            # Rewrite inner WHERE refs to use alias
            inner_where_rewritten = inner_where.this.copy().transform(
                lambda n: (
                    exp.column(n.name, table=alias)
                    if isinstance(n, exp.Column) and (n.table == inner_table_name or not n.table)
                    else n
                )
            )
            combined = exp.And(this=on_condition.this, expression=inner_where_rewritten)
            join_node.set("on", combined)

    return outer


# ---------------------------------------------------------------------------
# WHERE subqueries (IN / NOT IN / EXISTS / NOT EXISTS)
# ---------------------------------------------------------------------------


def _rewrite_where_subqueries(ast: exp.Select, dialect: str) -> exp.Select:
    """Rewrite subquery predicates in WHERE to joins."""
    where = ast.args.get("where")
    if not where:
        return ast

    ast = ast.copy()
    where = ast.args.get("where")
    if not where:
        return ast

    # Collect all subquery predicates
    # We need to process them and convert to joins
    ast = _process_where_node(ast, where.this, dialect)

    return ast


def _process_where_node(ast: exp.Select, node: exp.Expression, dialect: str) -> exp.Select:
    """Process a WHERE expression, rewriting subquery predicates to joins."""
    # Handle IN (subquery)
    if isinstance(node, exp.In):
        subquery_node = node.args.get("query")
        if subquery_node and isinstance(subquery_node, exp.Subquery):
            return _rewrite_in_subquery(ast, node, subquery_node, negated=False, dialect=dialect)

    # Handle NOT IN (subquery) - sqlglot represents as Not(In(...))
    if isinstance(node, exp.Not) and isinstance(node.this, exp.In):
        in_node = node.this
        subquery_node = in_node.args.get("query")
        if subquery_node and isinstance(subquery_node, exp.Subquery):
            return _rewrite_in_subquery(ast, in_node, subquery_node, negated=True, dialect=dialect)

    # Handle EXISTS (subquery) — EXISTS wraps Select directly, not Subquery
    if isinstance(node, exp.Exists):
        inner = node.this
        if isinstance(inner, (exp.Subquery, exp.Select)):
            return _rewrite_exists_subquery(ast, node, inner, negated=False, dialect=dialect)

    # Handle NOT EXISTS (subquery)
    if isinstance(node, exp.Not) and isinstance(node.this, exp.Exists):
        exists_node = node.this
        inner = exists_node.this
        if isinstance(inner, (exp.Subquery, exp.Select)):
            return _rewrite_exists_subquery(ast, exists_node, inner, negated=True, dialect=dialect)

    # For AND expressions, process both sides
    if isinstance(node, exp.And):
        ast = _process_where_node(ast, node.this, dialect)
        # Re-fetch the where after potential modification
        where = ast.args.get("where")
        if where:
            # Re-walk to find remaining subqueries
            ast = _process_where_node(ast, node.expression, dialect)

    return ast


def _rewrite_in_subquery(
    ast: exp.Select,
    in_node: exp.In,
    subquery_node: exp.Subquery,
    negated: bool,
    dialect: str,
) -> exp.Select:
    """Rewrite IN/NOT IN (subquery) to semi-join/anti-join.

    IN (subquery) -> add INNER JOIN (with DISTINCT)
    NOT IN (subquery) -> add LEFT JOIN + IS NULL filter

    The joined table uses its original name (no alias) so the IVM join compiler
    can properly track it as a DuckLake base table.
    """
    inner_select = subquery_node.this
    if not isinstance(inner_select, exp.Select):
        return ast

    # The outer column being compared
    outer_col = in_node.this
    if not isinstance(outer_col, exp.Column):
        raise UnsupportedSQLError(
            "in_subquery_complex_expr",
            "IN (subquery) only supported with a simple column reference on the left side",
        )

    # The inner select should produce one column
    inner_selects = inner_select.selects
    if len(inner_selects) != 1:
        raise UnsupportedSQLError(
            "in_subquery_multi_col",
            "IN (subquery) requires the subquery to return exactly one column",
        )

    # Extract inner table and column
    inner_tables = list(inner_select.find_all(exp.Table))
    if not inner_tables:
        raise UnsupportedSQLError("in_subquery_no_table", "Subquery has no tables")

    inner_table = inner_tables[0]
    inner_table_name = inner_table.alias_or_name

    # Get the inner column expression
    inner_col_expr = inner_selects[0]
    if isinstance(inner_col_expr, exp.Alias):
        inner_col_expr = inner_col_expr.this

    if inner_select.args.get("joins") or inner_select.args.get("group"):
        raise UnsupportedSQLError(
            "in_subquery_complex",
            "IN (subquery) with JOINs or GROUP BY is not yet supported",
        )

    # Use inner table name directly (no alias) so join compiler can track it
    join_ref = inner_table_name

    if not negated:
        # IN -> semi-join via INNER JOIN + DISTINCT
        _remove_predicate(ast, in_node)

        # Build ON condition: outer.col = inner_table.col
        if isinstance(inner_col_expr, exp.Column):
            on_col = exp.column(inner_col_expr.name, table=join_ref)
        else:
            on_col = inner_col_expr.copy()

        on_cond = exp.EQ(this=outer_col.copy(), expression=on_col)

        # Add inner WHERE as additional ON condition
        inner_where = inner_select.args.get("where")
        if inner_where:
            inner_where_rewritten = inner_where.this.copy().transform(
                lambda n: (
                    exp.column(n.name, table=join_ref)
                    if isinstance(n, exp.Column)
                    and (n.table == inner_table_name or not n.table)
                    and n.table not in _get_table_names(ast)
                    else n
                )
            )
            on_cond = exp.And(this=on_cond, expression=inner_where_rewritten)

        join_table = inner_table.copy()
        new_join = exp.Join(this=join_table, on=on_cond)
        existing_joins = ast.args.get("joins") or []
        existing_joins.append(new_join)
        ast.set("joins", existing_joins)

        # Add DISTINCT to prevent row duplication from semi-join
        ast.set("distinct", exp.Distinct())

    else:
        # NOT IN -> rewrite to correlated NOT EXISTS with NULL-aware predicate
        _remove_predicate(ast, exp.Not(this=in_node))

        outer_col_ref = outer_col.copy()
        if not outer_col_ref.table:
            outer_tables = _get_table_names(ast)
            if len(outer_tables) == 1:
                outer_col_ref = exp.column(outer_col_ref.name, table=next(iter(outer_tables)))
            else:
                raise UnsupportedSQLError(
                    "unqualified_column",
                    "NOT IN requires table-qualified column when multiple tables present",
                )

        outer_tables = _get_table_names(ast)

        def _rewrite_inner_refs(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column):
                if node.table == inner_table_name or (
                    not node.table and node.table not in outer_tables
                ):
                    return exp.column(node.name, table=join_ref)
            return node

        inner_expr = inner_col_expr.copy().transform(_rewrite_inner_refs)
        exists_pred = exp.Or(
            this=exp.EQ(this=inner_expr.copy(), expression=outer_col_ref.copy()),
            expression=exp.Is(this=inner_expr.copy(), expression=exp.Null()),
        )

        inner_where = inner_select.args.get("where")
        if inner_where:
            inner_where_rewritten = inner_where.this.copy().transform(_rewrite_inner_refs)
            combined_where = exp.And(this=inner_where_rewritten, expression=exists_pred)
        else:
            combined_where = exists_pred

        inner_select_copy = inner_select.copy()
        inner_select_copy.set("where", exp.Where(this=combined_where))

        exists_node = exp.Exists(this=inner_select_copy)
        ast = _rewrite_exists_subquery(
            ast, exists_node, inner_select_copy, negated=True, dialect=dialect
        )

    return ast


def _rewrite_exists_subquery(
    ast: exp.Select,
    exists_node: exp.Exists,
    inner_node: exp.Subquery | exp.Select,
    negated: bool,
    dialect: str,
) -> exp.Select:
    """Rewrite EXISTS/NOT EXISTS (subquery) to semi-join/anti-join.

    Correlated EXISTS: extract correlation predicate, convert to join
    Non-correlated EXISTS: if true, query is unchanged; if could change, wrap
    """
    # EXISTS may wrap Select directly or via Subquery
    inner_select = inner_node.this if isinstance(inner_node, exp.Subquery) else inner_node
    if not isinstance(inner_select, exp.Select):
        return ast

    inner_tables = list(inner_select.find_all(exp.Table))
    if not inner_tables:
        raise UnsupportedSQLError("exists_no_table", "EXISTS subquery has no tables")

    inner_table = inner_tables[0]
    inner_table_name = inner_table.alias_or_name

    is_correlated = _is_correlated(inner_select, ast)

    if not is_correlated:
        # Non-correlated EXISTS/NOT EXISTS is a boolean gate.
        # For IVM, changes to the inner table could toggle this.
        # We treat this as: rewrite to a semi/anti-join on a synthetic condition.
        # Actually, non-correlated EXISTS is rare. For now, raise unsupported.
        raise UnsupportedSQLError(
            "non_correlated_exists",
            "Non-correlated EXISTS is not yet supported for IVM. "
            "Consider rewriting as a correlated EXISTS or IN.",
        )

    # Correlated EXISTS: extract the correlation predicate from inner WHERE
    inner_where = inner_select.args.get("where")
    if not inner_where:
        raise UnsupportedSQLError(
            "exists_no_correlation",
            "Correlated EXISTS requires a WHERE clause with correlation predicate",
        )

    # Extract correlation predicates (conditions referencing outer tables)
    outer_table_names = _get_table_names(ast)
    correlation_preds, non_correlation_preds = _split_correlation_predicates(
        inner_where.this, outer_table_names
    )

    if not correlation_preds:
        raise UnsupportedSQLError(
            "exists_no_correlation",
            "EXISTS subquery WHERE clause must reference outer table columns",
        )

    join_ref = inner_table_name

    # Remove the EXISTS/NOT EXISTS from WHERE
    if negated:
        _remove_predicate(ast, exp.Not(this=exists_node))
    else:
        _remove_predicate(ast, exists_node)

    # Build ON condition from correlation predicates
    on_parts: list[exp.Expression] = []
    for pred in correlation_preds:
        # Rewrite inner table refs to use join_ref (the table name)
        rewritten = pred.copy().transform(
            lambda n: (
                exp.column(n.name, table=join_ref)
                if isinstance(n, exp.Column)
                and (n.table == inner_table_name or not n.table)
                and n.table not in outer_table_names
                else n
            )
        )
        on_parts.append(rewritten)

    # Add non-correlation predicates to ON as well
    for pred in non_correlation_preds:
        rewritten = pred.copy().transform(
            lambda n: (
                exp.column(n.name, table=join_ref)
                if isinstance(n, exp.Column) and (n.table == inner_table_name or not n.table)
                else n
            )
        )
        on_parts.append(rewritten)

    on_cond = on_parts[0]
    for part in on_parts[1:]:
        on_cond = exp.And(this=on_cond, expression=part)

    join_table = inner_table.copy()

    if not negated:
        # EXISTS -> semi-join (INNER JOIN + DISTINCT)
        new_join = exp.Join(this=join_table, on=on_cond)
        existing_joins = ast.args.get("joins") or []
        existing_joins.append(new_join)
        ast.set("joins", existing_joins)
        ast.set("distinct", exp.Distinct())
    else:
        # NOT EXISTS -> anti-join (LEFT JOIN + IS NULL)
        new_join = exp.Join(this=join_table, on=on_cond, side="LEFT")
        existing_joins = ast.args.get("joins") or []
        existing_joins.append(new_join)
        ast.set("joins", existing_joins)

        # Use rowid to detect existence (nullable columns are unsafe)
        null_check = exp.Is(this=exp.column("rowid", table=join_ref), expression=exp.Null())
        existing_where = ast.args.get("where")
        if existing_where:
            combined = exp.And(this=existing_where.this, expression=null_check)
            ast.set("where", exp.Where(this=combined))
        else:
            ast.set("where", exp.Where(this=null_check))

    return ast


# ---------------------------------------------------------------------------
# SELECT scalar subqueries
# ---------------------------------------------------------------------------


def _rewrite_select_subqueries(
    ast: exp.Select, dialect: str, ctx: CompilationContext | None = None
) -> tuple[exp.Select, list[MaterializedView]]:
    """Rewrite scalar subqueries in SELECT via recursive MV compilation.

    Correlated scalar subqueries like:
        SELECT t1.id, (SELECT MAX(t2.val) FROM t2 WHERE t2.fk = t1.id) AS max_val FROM t1
    become:
        1. Inner MV (DuckLake table): SELECT fk, MAX(val) AS max_val FROM t2 GROUP BY fk
        2. Outer rewritten as LEFT JOIN:
           SELECT t1.id, mv_xxx.max_val
           FROM t1 LEFT JOIN dl.main.mv_xxx ON mv_xxx.fk = t1.id

    The LEFT JOIN makes the outer MV track the inner MV as a real source table,
    so changes to either base table propagate through the existing join IVM compiler.
    """
    inner_mvs: list[MaterializedView] = []
    ast = ast.copy()

    new_selects = []

    for sel in ast.selects:
        subqueries = list(sel.find_all(exp.Subquery))
        if not subqueries:
            new_selects.append(sel)
            continue

        if ctx is None:
            raise UnsupportedSQLError(
                "scalar_subquery",
                "Scalar subqueries in SELECT require a CompilationContext "
                "for recursive MV compilation.",
            )

        # Handle the scalar subquery
        sq = subqueries[0]
        inner_select = sq.this
        if not isinstance(inner_select, exp.Select):
            new_selects.append(sel)
            continue

        # Skip subqueries without aggregates (already simple lookups)
        if not list(inner_select.find_all(exp.AggFunc)):
            new_selects.append(sel)
            continue

        # Get the alias for this SELECT expression
        out_alias = sel.alias if isinstance(sel, exp.Alias) else f"_sq_{len(inner_mvs)}"

        # Extract the aggregate expression and correlation predicates
        outer_table_names = _get_table_names(ast)
        inner_where = inner_select.args.get("where")
        if not inner_where:
            raise UnsupportedSQLError(
                "scalar_subquery_no_where",
                "Scalar subqueries in SELECT must have a WHERE clause with correlation",
            )

        correlation_preds, non_correlation_preds = _split_correlation_predicates(
            inner_where.this, outer_table_names
        )
        if not correlation_preds:
            raise UnsupportedSQLError(
                "scalar_subquery_no_correlation",
                "Scalar subqueries in SELECT must be correlated",
            )

        # Build the inner MV query:
        # SELECT <correlation_cols>, <agg_expr> AS <out_alias> FROM <inner_table>
        # [WHERE <non_correlation_preds>] GROUP BY <correlation_cols>
        inner_tables = list(inner_select.find_all(exp.Table))
        if not inner_tables:
            raise UnsupportedSQLError("scalar_subquery_no_table", "Scalar subquery has no tables")
        inner_table = inner_tables[0]
        inner_table_name = inner_table.alias_or_name

        # Extract correlation join keys: inner_col = outer_col
        join_keys: list[tuple[str, exp.Column]] = []  # (inner_col_name, outer_col_expr)
        for pred in correlation_preds:
            inner_col, outer_col = _extract_eq_columns(pred, inner_table_name, outer_table_names)
            if inner_col is None or outer_col is None:
                raise UnsupportedSQLError(
                    "scalar_subquery_complex_correlation",
                    "Only equi-join correlation predicates are supported in scalar subqueries",
                )
            join_keys.append((inner_col, outer_col))

        # Build inner SELECT: group_cols + aggregate
        inner_agg = inner_select.selects[0]
        inner_agg_expr = inner_agg.this if isinstance(inner_agg, exp.Alias) else inner_agg

        group_col_exprs = [exp.column(k[0], table=inner_table_name) for k in join_keys]

        inner_mv_selects = [
            *[exp.column(k[0], table=inner_table_name) for k in join_keys],
            exp.alias_(inner_agg_expr.copy(), out_alias),
        ]

        inner_mv_ast = exp.Select(
            expressions=inner_mv_selects,
        ).from_(inner_table.copy())

        # Add non-correlation WHERE
        if non_correlation_preds:
            where_expr = non_correlation_preds[0]
            for pred in non_correlation_preds[1:]:
                where_expr = exp.And(this=where_expr, expression=pred)
            inner_mv_ast = inner_mv_ast.where(where_expr)

        inner_mv_ast = inner_mv_ast.group_by(*group_col_exprs)

        inner_sql = inner_mv_ast.sql(dialect=dialect)
        inner_mv_name = _inner_mv_name(ctx.naming.mv_table(), inner_sql)

        # Compile the inner MV
        from duckstream.compiler.router import compile_ivm

        inner_naming = _make_inner_naming(ctx.naming, inner_mv_name)

        inner_mv = compile_ivm(
            inner_sql,
            naming=inner_naming,
            mv_catalog=ctx.mv_catalog,
            mv_schema=ctx.mv_schema,
            sources=ctx.sources,
        )
        inner_mvs.append(inner_mv)

        # Replace the scalar subquery with a LEFT JOIN to the inner MV.
        # The SELECT expression becomes a simple column reference: mv_xxx.out_alias
        mv_fqn_table = exp.table_(inner_mv_name, db=ctx.mv_schema, catalog=ctx.mv_catalog)

        # Build ON condition: mv_xxx.inner_col = outer.col
        on_parts: list[exp.Expression] = []
        for inner_col, outer_col in join_keys:
            on_parts.append(
                exp.EQ(
                    this=exp.column(inner_col, table=inner_mv_name),
                    expression=outer_col.copy(),
                )
            )
        on_cond = on_parts[0]
        for part in on_parts[1:]:
            on_cond = exp.And(this=on_cond, expression=part)

        new_join = exp.Join(this=mv_fqn_table, on=on_cond, side="LEFT")
        existing_joins = ast.args.get("joins") or []
        existing_joins.append(new_join)
        ast.set("joins", existing_joins)

        # Replace the subquery SELECT expression with a column reference
        new_selects.append(exp.alias_(exp.column(out_alias, table=inner_mv_name), out_alias))

    ast.args["expressions"] = new_selects
    return ast, inner_mvs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inner_naming(base_naming: Naming, name: str) -> Naming:
    """Create a Naming instance that returns a specific mv_table name."""

    class _InnerNaming(type(base_naming)):  # type: ignore[misc]
        def mv_table(self) -> str:
            return name

    return _InnerNaming()


def _inner_mv_name(parent_mv_name: str, inner_sql: str) -> str:
    """Generate a deterministic name for an inner MV: <parent>_<hash>."""
    h = hashlib.md5(inner_sql.encode()).hexdigest()[:8]
    return f"{parent_mv_name}_{h}"


def _is_correlated(inner: exp.Select, outer: exp.Select) -> bool:
    """Check if inner query references any tables from the outer query."""
    outer_tables = _get_table_names(outer)
    return any(col.table in outer_tables for col in inner.find_all(exp.Column))


def _get_table_names(ast: exp.Select) -> set[str]:
    """Get all table names/aliases referenced in a SELECT's FROM/JOIN."""
    names: set[str] = set()
    from_clause = ast.args.get("from_")
    if from_clause:
        node = from_clause.this
        if isinstance(node, exp.Table):
            names.add(node.alias_or_name)
        elif isinstance(node, exp.Subquery) and node.alias:
            names.add(node.alias)
    for join in ast.args.get("joins") or []:
        node = join.this
        if isinstance(node, exp.Table):
            names.add(node.alias_or_name)
        elif isinstance(node, exp.Subquery) and node.alias:
            names.add(node.alias)
    return names


def _split_correlation_predicates(
    where_expr: exp.Expression,
    outer_table_names: set[str],
) -> tuple[list[exp.Expression], list[exp.Expression]]:
    """Split WHERE predicates into correlation and non-correlation predicates.

    A correlation predicate references at least one column from an outer table.
    """
    predicates = _flatten_and(where_expr)
    correlated: list[exp.Expression] = []
    non_correlated: list[exp.Expression] = []

    for pred in predicates:
        refs_outer = False
        for col in pred.find_all(exp.Column):
            if col.table in outer_table_names:
                refs_outer = True
                break
        if refs_outer:
            correlated.append(pred)
        else:
            non_correlated.append(pred)

    return correlated, non_correlated


def _flatten_and(expr: exp.Expression) -> list[exp.Expression]:
    """Flatten AND expressions into a list of predicates."""
    if isinstance(expr, exp.And):
        return _flatten_and(expr.this) + _flatten_and(expr.expression)
    return [expr]


def _find_inner_join_key(
    correlation_preds: list[exp.Expression],
    inner_table_name: str,
) -> str:
    """Find an inner table column name from correlation predicates (for IS NULL check)."""
    for pred in correlation_preds:
        for col in pred.find_all(exp.Column):
            if col.table == inner_table_name or not col.table:
                return col.name
    return "rowid"  # fallback


def _extract_eq_columns(
    pred: exp.Expression,
    inner_table_name: str,
    outer_table_names: set[str],
) -> tuple[str | None, exp.Column | None]:
    """Extract inner and outer column from an equi-join predicate.

    Returns (inner_col_name, outer_col_expression) or (None, None).
    """
    if not isinstance(pred, exp.EQ):
        return None, None

    left = pred.left
    right = pred.right

    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
        if (left.table == inner_table_name or not left.table) and right.table in outer_table_names:
            return left.name, right
        if (right.table == inner_table_name or not right.table) and left.table in outer_table_names:
            return right.name, left

    return None, None


def _remove_predicate(ast: exp.Select, target: exp.Expression) -> None:
    """Remove a predicate from the WHERE clause.

    If the WHERE is just this predicate, remove WHERE entirely.
    If it's part of an AND chain, remove just this predicate.
    """
    where = ast.args.get("where")
    if not where:
        return

    where_expr = where.this

    # Check if the entire WHERE is the target
    if _exprs_equal(where_expr, target):
        ast.set("where", None)
        return

    # Try to remove from AND chain
    new_where = _remove_from_and(where_expr, target)
    if new_where is None:
        ast.set("where", None)
    elif new_where is not where_expr:
        ast.set("where", exp.Where(this=new_where))


def _remove_from_and(expr: exp.Expression, target: exp.Expression) -> exp.Expression | None:
    """Remove target from an AND chain. Returns remaining expression or None."""
    if _exprs_equal(expr, target):
        return None

    if isinstance(expr, exp.And):
        left = _remove_from_and(expr.this, target)
        right = _remove_from_and(expr.expression, target)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        return exp.And(this=left, expression=right)

    return expr


def _exprs_equal(a: exp.Expression, b: exp.Expression) -> bool:
    """Check if two expressions are structurally equal."""
    return a.sql() == b.sql()
