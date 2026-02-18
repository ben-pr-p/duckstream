# DuckLake IVM — Project Instructions

## Tooling

Use **uv** exclusively for all Python operations:

- `uv run pytest` — run tests (never bare `pytest`)
- `uv run ruff check src/ tests/` — lint
- `uv run ruff format src/ tests/` — format
- `uv run ty check src/` — type check
- `uv add <pkg>` / `uv add --dev <pkg>` — add dependencies
- `uv run python` — run Python scripts

Do not use pip, conda, poetry, or any other package manager.

## Quality gates

Before any code is considered complete, all of the following must pass:

1. `uv run ruff check src/ tests/` — no lint errors
2. `uv run ruff format --check src/ tests/` — consistent formatting
3. `uv run ty check src/` — no type errors in library code
4. `uv run pytest` — all tests pass

## Plan files

Read these before working on the codebase:

- **bag-algebra-vs-z-sets.md** — why classical bag algebra (not DBSP Z-sets) is the right model; contains all delta rules and SQL patterns
- **PLAN.md** — concrete implementation plan with exact code patterns, SQL templates, and sqlglot API usage for each stage
- **EVOLUTION.md** — stage-by-stage roadmap (Stage 1: SELECT/WHERE through Stage 10: set operations) with test strategies and done criteria
- **REQUIREMENTS.md** — Python API specification (`compile_ivm()`, `IVMPlan`, `Naming`, snapshot cursors, cross-catalog support)
- **CODEBASE.md** — project layout, key concepts, delta rule summary, test architecture

## Architecture

- Pure SQL-to-SQL compiler: no side effects, no connections, no state
- Built on **sqlglot** for SQL parsing/generation
- All base tables and MVs are **DuckLake** tables
- Deltas via `ducklake_table_changes()` — no manual delta tables
- Tests use **Hypothesis** property-based testing with DuckDB as the oracle
- Each test gets its own isolated DuckLake instance via `tempfile.mkdtemp()`
