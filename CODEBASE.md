# Codebase Structure

## What This Is

A **SQL-to-SQL incremental view maintenance (IVM) compiler** for DuckLake. Given a SQL view definition, it emits the SQL statements needed to incrementally maintain a materialized view as base tables change — without recomputing the view from scratch.

Built on **sqlglot** for SQL parsing/generation and **DuckLake** for change tracking via `ducklake_table_changes()`. Pure Python, no C++ dependencies, no separate runtime.

## Project Layout

```
ducklake-ivm/
├── pyproject.toml                  # Project config, dependencies, build system
├── main.py                         # Entry point placeholder
│
├── src/ducklake_ivm/               # Library source
│   ├── __init__.py                 # Package root (public API re-exports)
│   └── compiler.py                 # compile_ivm() — the core compiler (stub)
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # DuckLake test fixture (isolated temp dirs)
│   ├── strategies.py               # Hypothesis strategies for random test scenarios
│   └── test_ivm.py                 # Property-based + smoke tests
│
├── CONVERSATION.md                 # Research context: prior art, DBSP theory, initial plan
├── REQUIREMENTS.md                 # Python API specification
├── EVOLUTION.md                    # Stage-by-stage implementation + testing plan
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

### The Compiler (`src/ducklake_ivm/compiler.py`)

The library's public API is a single function:

```python
compile_ivm(view_sql, *, dialect, naming, mv_catalog, mv_schema, sources) -> IVMPlan
```

It takes a SQL SELECT statement and returns an `IVMPlan` dataclass containing:
- `create_cursors_table` — DDL for the shared snapshot cursor tracking table
- `create_mv` — DDL to create and populate the materialized view
- `initialize_cursors` — DML to set initial cursor positions
- `maintain` — ordered list of DML statements that read deltas from `ducklake_table_changes()`, apply them to the MV, and update the cursors

The compiler is pure — no side effects, no connections. It just transforms SQL into SQL.

### Planned Source Modules

As the compiler grows beyond `compiler.py`, it will split into:

| Module | Responsibility |
|--------|---------------|
| `compiler.py` | Top-level `compile_ivm()` orchestrator |
| `rewriter.py` | Bag algebra delta rules — the core AST transformation logic |
| `naming.py` | `Naming` class for customizable table/column naming |
| `plan.py` | `IVMPlan`, `SnapshotSafety`, `PendingMaintenance` dataclasses |
| `introspection.py` | `safe_to_expire_sql()`, `pending_maintenance_sql()` |

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

### Introspection Functions

Two additional functions generate SQL queries (not part of `IVMPlan`):

- **`safe_to_expire_sql(mv_catalogs, source_catalogs)`** — reports which snapshots are safe to expire across all source catalogs, by finding the minimum cursor across all MVs
- **`pending_maintenance_sql(mv_catalogs)`** — reports how many pending changes and unprocessed snapshots exist per MV per source table

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

### Hypothesis Strategies (`tests/strategies.py`)

Strategies generate random `Scenario` objects containing:
- `tables` — random schemas (2-5 columns, mixed types)
- `initial_data` — random rows
- `view_sql` — a valid SELECT statement using the generated tables
- `deltas` — random inserts + deletes (deletes drawn from initial data)

Current strategies:
- `single_table_select()` — SELECT/PROJECT/WHERE
- `single_table_aggregate()` — GROUP BY with SUM/COUNT/AVG

### Test File (`tests/test_ivm.py`)

- **Property tests** — `test_select_project_filter`, `test_single_table_aggregate` — 50 Hypothesis examples each
- **Smoke tests** — `TestSmoke` class with 4 hand-written deterministic scenarios

### Running Tests

```bash
uv run pytest                                          # all tests
uv run pytest tests/test_ivm.py::TestSmoke             # smoke tests only
uv run pytest tests/test_ivm.py::test_select_project_filter -x  # one property test
uv run pytest -n auto                                  # parallel execution
```

## Implementation Status

The compiler is a stub (`NotImplementedError`). The test infrastructure is fully wired — tests are collected and fail with `NotImplementedError` as expected, ready for implementation.

See `EVOLUTION.md` for the stage-by-stage plan from Stage 1 (SELECT/WHERE) through Stage 10 (set operations).

## Documentation Map

| File | Contents |
|------|----------|
| `bag-algebra-vs-z-sets.md` | Why classical bag algebra (not DBSP Z-sets) is the right model — delta rules, SQL patterns, algorithmic complexity |
| `CONVERSATION.md` | Research context: prior art survey (RisingWave, Feldera, OpenIVM, pg_ivm), DBSP theory, initial architecture sketch |
| `REQUIREMENTS.md` | Python API specification: `compile_ivm()`, `IVMPlan`, `Naming`, snapshot cursors, cross-catalog support, `safe_to_expire_sql()`, `pending_maintenance_sql()`, error handling, full workflow examples |
| `EVOLUTION.md` | Implementation roadmap: 10 stages from simple SELECT through set operations, testing philosophy, per-stage compiler work + test strategies + done criteria |
| `CODEBASE.md` | This file: project layout, key concepts, test architecture, implementation status |
