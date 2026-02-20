"""
IVM Compiler: SQL-to-SQL incremental view maintenance.

Takes a SQL view definition and emits SQL statements that propagate
deltas from base tables into the materialized view.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from duckstream.compiler.aggregates import _gen_aggregate_maintenance
from duckstream.compiler.distinct import _gen_distinct_maintenance
from duckstream.compiler.infrastructure import (
    _detect_features,
    _gen_create_cursors,
    _gen_create_mv,
    _gen_init_cursor,
    _gen_query_mv,
    _resolve_source,
)
from duckstream.compiler.join import _compile_join
from duckstream.compiler.join_aggregate import _compile_join_aggregate
from duckstream.compiler.outer_join import _compile_outer_join
from duckstream.compiler.select import _gen_select_maintenance
from duckstream.compiler.set_ops import _compile_set_operation
from duckstream.compiler.subquery import CompilationContext, has_subqueries, rewrite_subqueries
from duckstream.materialized_view import MaterializedView, Naming, UnsupportedSQLError


def compile_ivm(
    view_sql: str,
    *,
    naming: Naming | None = None,
    mv_catalog: str = "dl",
    mv_schema: str = "main",
    sources: dict[str, dict] | None = None,
) -> MaterializedView:
    """Compile a view definition into IVM maintenance SQL.

    Tries incremental compilation first. On UnsupportedSQLError, falls back
    to full-refresh + bag-diff maintenance (unless the error is a genuinely
    invalid query like no_table or having_no_agg).
    """
    naming = naming or Naming()
    try:
        return _compile_ivm_inner(
            view_sql,
            naming=naming,
            mv_catalog=mv_catalog,
            mv_schema=mv_schema,
            sources=sources,
        )
    except UnsupportedSQLError as e:
        # Don't fall back on genuinely invalid queries
        if e.feature in ("no_table", "having_no_agg"):
            raise
        from duckstream.compiler.full_refresh import compile_full_refresh

        mv = compile_full_refresh(
            view_sql,
            naming=naming,
            mv_catalog=mv_catalog,
            mv_schema=mv_schema,
            sources=sources,
        )
        mv.fallback_reason = e.message
        return mv


def _compile_ivm_inner(
    view_sql: str,
    *,
    naming: Naming,
    mv_catalog: str = "dl",
    mv_schema: str = "main",
    sources: dict[str, dict] | None = None,
) -> MaterializedView:
    """Core incremental IVM compilation. Raises UnsupportedSQLError on failure."""
    dialect = "duckdb"
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

    # --- Subquery rewriting pass ---
    inner_mvs: list[MaterializedView] = []
    if has_subqueries(ast):
        ctx = CompilationContext(
            naming=naming,
            mv_catalog=mv_catalog,
            mv_schema=mv_schema,
            sources=sources,
            dialect=dialect,
        )
        ast, inner_mvs = rewrite_subqueries(ast, dialect, ctx)

    # --- Analysis ---
    tables = list(ast.find_all(exp.Table))
    has_agg = bool(list(ast.find_all(exp.AggFunc)))
    has_distinct = bool(ast.args.get("distinct"))
    has_having = bool(ast.args.get("having"))
    joins = ast.args.get("joins")
    has_join = bool(joins)

    # HAVING requires GROUP BY / aggregates
    if has_having and not has_agg:
        raise UnsupportedSQLError("having_no_agg", "HAVING requires GROUP BY with aggregates")

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
        mv = compile_fn(
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
        mv.inner_mvs = inner_mvs
        return mv

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
    query_mv = _gen_query_mv(ast, mv_fqn, naming, dialect)

    return MaterializedView(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables={table_name: src["catalog"]},
        features=features,
        query_mv=query_mv,
        inner_mvs=inner_mvs,
    )
