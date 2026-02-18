# Codebase Structure

## What This Is

A **SQL-to-SQL incremental view maintenance (IVM) compiler** for DuckLake. Given a SQL view definition, it emits the SQL statements needed to incrementally maintain a materialized view as base tables change — without recomputing the view from scratch.

Built on **sqlglot** for SQL parsing/generation and **DuckLake** for change tracking via `ducklake_table_changes()`. Pure Python, no C++ dependencies, no separate runtime.

## Project Layout

```
duckstream/
├── pyproject.toml                  # Project config, dependencies, build system
├── main.py                         # Entry point placeholder
│
├── src/duckstream/                 # Library source
│   ├── __init__.py                 # Package root (public API re-exports)
│   ├── plan.py                     # IVMPlan, Naming, UnsupportedSQLError dataclasses
│   ├── utils.py                    # safe_to_expire_sql(), pending_maintenance_sql()
│   └── compiler/                   # Core compiler modules
│       ├── __init__.py             # Package root
│       ├── router.py               # compile_ivm() orchestrator — routes to feature modules
│       ├── select.py               # Stage 1: SELECT/PROJECT/WHERE
│       ├── aggregates.py           # Stage 2-3: GROUP BY + SUM/COUNT/AVG/MIN/MAX
│       ├── join.py                 # Stage 4,8: N-table inner JOIN
│       ├── join_aggregate.py       # Stage 5: JOIN + aggregates composed
│       ├── distinct.py             # Stage 6: SELECT DISTINCT
│       ├── outer_join.py           # Stage 9: LEFT/RIGHT/FULL OUTER JOIN
│       ├── set_ops.py              # Stage 10: UNION/EXCEPT/INTERSECT
│       └── infrastructure.py       # Shared helpers: DDL, cursors, snapshots, column rewriting
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # DuckLake fixture, oracle harness (assert_ivm_correct)
│   ├── strategies/                 # Hypothesis strategies for random test generation
│   │   ├── __init__.py             # Re-exports all strategies
│   │   ├── primitives.py           # Table, Column, Row, Delta, Scenario types
│   │   ├── select.py               # single_table_select()
│   │   ├── aggregate.py            # single_table_aggregate()
│   │   ├── having.py               # single_table_having()
│   │   ├── distinct.py             # single_table_distinct()
│   │   ├── join.py                 # two_table_join(), three_table_join()
│   │   ├── join_aggregate.py       # join_then_aggregate()
│   │   ├── outer_join.py           # two_table_outer_join()
│   │   └── set_ops.py              # set_operation()
│   ├── test_select.py              # Stage 1 tests
│   ├── test_aggregate.py           # Stage 2-3 tests
│   ├── test_having.py              # HAVING tests
│   ├── test_distinct.py            # Stage 6 tests
│   ├── test_join.py                # Stage 4,8 tests
│   ├── test_join_aggregate.py      # Stage 5 tests
│   ├── test_outer_join.py          # Stage 9 tests
│   ├── test_set_ops.py             # Stage 10 tests
│   └── test_utils.py               # Utility function tests
│
├── bag-algebra-vs-z-sets.md        # Why classical bag algebra, not DBSP Z-sets
├── PLAN.md                         # Concrete implementation plan with SQL patterns
├── REQUIREMENTS.md                 # Python API specification
├── EVOLUTION.md                    # Stage-by-stage roadmap
└── CODEBASE.md                     # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `sqlglot` | SQL parsing, AST manipulation, dialect-aware code generation |
| `duckdb` | Execution engine for testing (in-memory DuckLake instances) |
| `hypothesis` | Property-based test generation |
| `pytest` | Test runner |
| `pytz` | Required by DuckLake's snapshot timestamp handling |

Managed by **uv**. Python 3.13+.

## Key Concepts

### The Compiler (`src/duckstream/compiler/`)

The library's public API is a single function:

```python
compile_ivm(view_sql, *, dialect, naming, mv_catalog, mv_schema, sources) -> IVMPlan
```

It takes a SQL SELECT statement and returns an `IVMPlan` dataclass containing:
- `create_cursors_table` — DDL for the shared snapshot cursor tracking table
- `create_mv` — DDL to create and populate the materialized view
- `initialize_cursors` — DML to set initial cursor positions
- `maintain` — ordered list of DML statements that read deltas from `ducklake_table_changes()`, apply them to the MV, and update the cursors
- `query_mv` — SELECT query to read the MV, excluding auxiliary columns and applying HAVING filter if present

The compiler is pure — no side effects, no connections. It just transforms SQL into SQL.

### Compiler Modules

| Module | Responsibility |
|--------|---------------|
| `router.py` | Top-level `compile_ivm()` orchestrator — detects features and routes to the right module |
| `select.py` | Single-table SELECT/PROJECT/WHERE maintenance |
| `aggregates.py` | GROUP BY with SUM, COUNT, AVG, MIN, MAX |
| `join.py` | N-table inner JOIN via inclusion-exclusion decomposition |
| `join_aggregate.py` | JOIN + aggregates composition |
| `distinct.py` | DISTINCT via multiplicity tracking |
| `outer_join.py` | LEFT/RIGHT/FULL OUTER JOIN with NULL extension handling |
| `set_ops.py` | UNION/EXCEPT/INTERSECT (ALL and DISTINCT variants) |
| `infrastructure.py` | Shared helpers: DDL generation, snapshot variables, column rewriting, feature detection, HAVING support |

### Utility Functions (`src/duckstream/utils.py`)

| Function | Purpose |
|----------|---------|
| `safe_to_expire_sql()` | Reports which snapshots are safe to expire across source catalogs |
| `pending_maintenance_sql()` | Reports pending maintenance work per MV |

### DuckLake Integration

All base tables and the MV are DuckLake tables. Source tables can live in **different DuckLake catalogs**; the MV can live in yet another.

Deltas are not manually populated — the compiler generates SQL that reads changes from DuckLake's native change feed:

```
ducklake_table_changes(catalog, schema, table, start_snapshot, end_snapshot)
```

This returns rows tagged with `change_type`: `insert`, `delete`, `update_preimage`, `update_postimage`. The compiler treats inserts/update_postimage as `ΔR` (rows to add) and deletes/update_preimage as `∇R` (rows to remove), following classical bag algebra.

### Snapshot Cursor Tracking

A shared `_ivm_cursors` table in the MV's catalog tracks, per MV and per source catalog, the last-processed snapshot ID. The maintenance SQL is self-contained — it reads the cursor, processes changes, and updates the cursor.

```
_ivm_cursors (mv_name, source_catalog, last_snapshot)
```

## Delta Rules (How the Rewrites Work)

The compiler uses **classical bag algebra** (as in pg_ivm) rather than DBSP Z-sets. DuckLake gives us post-update table state and separate insert/delete transition tables via `ducklake_table_changes()`, which maps directly to the bag algebra model. See `bag-algebra-vs-z-sets.md` for the full rationale.

Notation: `R` = post-update (current) table state, `ΔR` = inserted rows, `∇R` = deleted rows.

| SQL Feature | Incremental Rule |
|-------------|-----------------|
| SELECT / WHERE | `ΔV = σ_p(ΔR)`, `∇V = σ_p(∇R)` — apply predicate to both delta streams |
| PROJECT | `ΔV = π(ΔR)`, `∇V = π(∇R)` — project both delta streams |
| JOIN (one table changes) | `ΔV = ΔR ⋈ S`, `∇V = ∇R ⋈ S` — join delta against current other table |
| JOIN (both change) | `ΔV = (ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS)` — subtract to avoid double-counting |
| SUM / COUNT | Separate add/subtract passes per group from ΔR and ∇R |
| AVG | Maintained via hidden SUM and COUNT columns |
| MIN / MAX | Rescan fallback when current extremum is deleted |
| DISTINCT | Multiplicity counter; row appears/disappears at 0-crossing |
| HAVING | MV stores all groups; `query_mv` filters by HAVING condition at read time |

The critical property: **the incremental form of a composition is the composition of incremental forms**. Each node is rewritten independently.

## Test Architecture

### Oracle Pattern

Every test follows the same invariant:

1. Create DuckLake tables with initial data
2. Materialize the view (full computation)
3. Apply changes to base tables (DuckLake records snapshots)
4. Run the compiler's maintenance SQL
5. Recompute the view from scratch
6. **Assert: maintained MV == recomputed view**

### Isolation

Each test gets its own DuckLake instance via a random temp directory (`tempfile.mkdtemp()`). This ensures full isolation even under parallel execution with `pytest-xdist`.

### Test Fixture (`tests/conftest.py`)

The `ducklake` fixture creates a fresh DuckLake catalog (`dl`) in a temp directory, yields `(connection, catalog_name)`, and cleans up after.

### Hypothesis Strategies (`tests/strategies/`)

Strategies generate random `Scenario` objects containing:
- `tables` — random schemas (2-5 columns, mixed types)
- `initial_data` — random rows
- `view_sql` — a valid SELECT statement using the generated tables
- `deltas` — random inserts + deletes (deletes drawn from initial data)

Current strategies:
- `single_table_select()` — SELECT/PROJECT/WHERE
- `single_table_aggregate()` — GROUP BY with SUM/COUNT/AVG/MIN/MAX
- `single_table_having()` — GROUP BY with HAVING clause
- `single_table_distinct()` — SELECT DISTINCT
- `two_table_join()` / `three_table_join()` — N-table inner JOINs
- `join_then_aggregate()` — JOIN + GROUP BY composed
- `two_table_outer_join()` — LEFT/RIGHT/FULL OUTER JOIN
- `set_operation()` — UNION/EXCEPT/INTERSECT

### Running Tests

```bash
uv run pytest                                  # all tests
uv run pytest tests/test_having.py -x          # HAVING tests
uv run pytest tests/test_utils.py -v           # utility function tests
uv run pytest -n auto                          # parallel execution
```

## Implementation Status

All 10 stages from EVOLUTION.md are complete, plus:
- **HAVING** support — MV stores all groups, `query_mv` filters at read time
- **Utility functions** — `safe_to_expire_sql()` and `pending_maintenance_sql()`
- **`query_mv`** field on `IVMPlan` — SELECT query for reading the MV (excludes auxiliary columns, applies HAVING)

See `EVOLUTION.md` for the stage-by-stage plan and future/deferred features.

## Documentation Map

| File | Contents |
|------|----------|
| `bag-algebra-vs-z-sets.md` | Why classical bag algebra (not DBSP Z-sets) is the right model — delta rules, SQL patterns, algorithmic complexity |
| `REQUIREMENTS.md` | Python API specification: `compile_ivm()`, `IVMPlan`, `Naming`, snapshot cursors, cross-catalog support, `safe_to_expire_sql()`, `pending_maintenance_sql()`, error handling, full workflow examples |
| `EVOLUTION.md` | Implementation roadmap: 10 stages from simple SELECT through set operations, testing philosophy, per-stage compiler work + test strategies + done criteria |
| `PLAN.md` | Concrete implementation plan with exact SQL patterns, sqlglot API usage |
| `CODEBASE.md` | This file: project layout, key concepts, test architecture, implementation status |
