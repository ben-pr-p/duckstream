# Implementation Plan

Concrete implementation plan for the IVM compiler, starting from the current stub and building through Stage 5 (join + aggregates). Each section specifies the exact code changes, the SQL patterns to generate, and the sqlglot API calls to use.

The compiler uses **classical bag algebra** (not DBSP Z-sets). See `bag-algebra-vs-z-sets.md` for the full rationale. Key implications:
- Separate INSERT (ΔR) and DELETE (∇R) passes, not unified weights
- Post-update (current) table state for joins, not pre-update
- Join delta: `(ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS)` — subtract to avoid double-counting

## Prerequisites: Tooling Knowledge

### sqlglot patterns we'll use

```python
import sqlglot
from sqlglot import exp

# Parse
ast = sqlglot.parse_one(view_sql, dialect="duckdb")

# Inspect structure
ast.find_all(exp.Table)          # all table references
ast.find_all(exp.AggFunc)        # Sum, Count, Avg, Min, Max
ast.args.get("where")            # Where node or None
ast.args.get("group")            # Group node or None
ast.args.get("joins")            # list of Join nodes or None
ast.selects                      # projection list (SELECT clause)

# Build SQL programmatically
q = exp.select("a", "b").from_("t").where("a > 5")
q.sql(dialect="duckdb")          # → "SELECT a, b FROM t WHERE a > 5"

# Transform (returns new tree, original untouched)
new_ast = ast.transform(lambda node: replacement if condition else node)

# Table substitution
def replace_table(node):
    if isinstance(node, exp.Table) and node.name == "t":
        return exp.table_("delta_t")
    return node
delta_query = ast.transform(replace_table)

# UNION ALL
exp.union(q1, q2, distinct=False)

# Column/alias building
exp.alias_(exp.Sum(this=exp.column("val")), "total")
exp.column("col_name", table="table_name")
```

### ducklake_table_changes behavior

```sql
-- 5 args: catalog, schema, table, start_snapshot, end_snapshot
-- Both start and end are INCLUSIVE
-- Returns: snapshot_id, rowid, change_type, <table columns...>
-- change_type: 'insert', 'delete', 'update_preimage', 'update_postimage'

SELECT * FROM ducklake_table_changes('dl', 'main', 't', 3, 7)

-- For bag algebra IVM:
--   ΔR (inserted rows): WHERE change_type IN ('insert', 'update_postimage')
--   ∇R (deleted rows):  WHERE change_type IN ('delete', 'update_preimage')
-- After processing through snap N, next call uses start=N+1
-- Works in CTEs, subqueries, JOINs, INSERT INTO...SELECT
-- Inverted range (start > end) returns empty, no error
-- Non-existent future end snapshot throws error
```

---

## Step 0: Dataclasses and Module Structure

**Files to create/modify:**
- `src/duckstream/plan.py` — new file for `IVMPlan` and `Naming`
- `src/duckstream/__init__.py` — re-export public API
- `src/duckstream/compiler.py` — update signature

### `src/duckstream/plan.py`

```python
from dataclasses import dataclass, field


class Naming:
    def mv_table(self) -> str:
        return "mv"

    def cursors_table(self) -> str:
        return "_ivm_cursors"

    def aux_column(self, purpose: str) -> str:
        return f"_ivm_{purpose}"


@dataclass
class IVMPlan:
    view_sql: str
    create_cursors_table: str
    create_mv: str
    initialize_cursors: list[str]
    maintain: list[str]
    base_tables: dict[str, str]       # table_name -> catalog
    features: set[str] = field(default_factory=set)


class UnsupportedSQLError(Exception):
    def __init__(self, feature: str, message: str):
        self.feature = feature
        self.message = message
        super().__init__(message)
```

### `src/duckstream/__init__.py`

```python
from duckstream.plan import IVMPlan, Naming, UnsupportedSQLError
from duckstream.compiler import compile_ivm
```

### `src/duckstream/compiler.py` — new signature

```python
def compile_ivm(
    view_sql: str,
    *,
    dialect: str = "duckdb",
    naming: Naming | None = None,
    mv_catalog: str = "dl",
    mv_schema: str = "main",
    sources: dict[str, dict] | None = None,
) -> IVMPlan:
```

### Update test harness

The test harness currently calls `compile_ivm(scenario.view_sql, dialect="duckdb")` and expects `ivm_output.maintain`. After Step 0 the return type changes from `dict` to `IVMPlan`, but the test already accesses `.maintain` — so this is compatible. We need to update the call to pass `mv_catalog`:

```python
ivm_output = compile_ivm(
    scenario.view_sql,
    dialect="duckdb",
    mv_catalog=catalog,   # the DuckLake catalog from fixture
)
```

Also update `assert_ivm_correct` to:
- Execute `plan.create_cursors_table` and `plan.create_mv` instead of the manual `CREATE TABLE memory.main.mv AS ...`
- Execute `plan.initialize_cursors`
- Read the MV from the DuckLake catalog instead of `memory.main.mv`

---

## Step 1: Stage 1 — SELECT / PROJECT / WHERE (No Aggregates, No Joins)

This is the simplest case. The view is `SELECT <cols> FROM <table> [WHERE <pred>]`.

### Analysis phase (in compiler)

```python
ast = sqlglot.parse_one(view_sql, dialect=dialect)

# Extract tables
tables = list(ast.find_all(exp.Table))
# For Stage 1: assert len(tables) == 1

# Check for unsupported features
if ast.find_all(exp.AggFunc):
    # Will be handled in Stage 2
    ...
if ast.args.get("joins"):
    raise UnsupportedSQLError("join", "Joins not yet supported")

table_name = tables[0].name
```

### Resolve source catalog

```python
if sources and table_name in sources:
    src_catalog = sources[table_name].get("catalog", mv_catalog)
    src_schema = sources[table_name].get("schema", "main")
else:
    src_catalog = mv_catalog
    src_schema = mv_schema
```

### Generate `create_cursors_table`

```sql
CREATE TABLE IF NOT EXISTS {mv_catalog}.{mv_schema}.{naming.cursors_table()} (
    mv_name VARCHAR,
    source_catalog VARCHAR,
    last_snapshot BIGINT,
    PRIMARY KEY (mv_name, source_catalog)
)
```

### Generate `create_mv`

```sql
CREATE TABLE {mv_catalog}.{mv_schema}.{naming.mv_table()} AS
SELECT {cols} FROM {src_catalog}.{src_schema}.{table_name} [WHERE ...]
```

Implementation: take the original AST, replace the table reference with the fully-qualified name, wrap in `CREATE TABLE ... AS`.

```python
qualified_ast = ast.transform(
    lambda node: exp.table_(table_name, db=src_schema, catalog=src_catalog)
    if isinstance(node, exp.Table) and node.name == table_name
    else node
)
create_mv_sql = f"CREATE TABLE {mv_catalog}.{mv_schema}.{mv_table} AS {qualified_ast.sql(dialect=dialect)}"
```

### Generate `initialize_cursors`

One INSERT per source catalog:

```sql
INSERT INTO {mv_catalog}.{mv_schema}._ivm_cursors (mv_name, source_catalog, last_snapshot)
VALUES ('{mv_table}', '{src_catalog}',
    (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}')))
```

### Generate maintenance SQL

For SELECT/PROJECT/WHERE, the bag algebra delta rule is: apply the same projection and filter to both ΔR and ∇R. The maintenance needs to:

1. **Delete removed rows (∇V)** — ∇R rows that pass the WHERE filter, matched against existing MV rows
2. **Insert new rows (ΔV)** — ΔR rows that pass the WHERE filter
3. **Update cursor**

Delete first to avoid false matches with newly inserted rows.

#### Performance: single scan of `ducklake_table_changes()`

Even though we reason about ΔR and ∇R separately (classical bag algebra semantics), the generated SQL scans `ducklake_table_changes()` only **once** by using a CTE that both the INSERT and DELETE logic reference:

```sql
WITH _changes AS (
    SELECT *
    FROM ducklake_table_changes(
        '{src_catalog}', '{src_schema}', '{table_name}',
        (SELECT last_snapshot + 1 FROM {cursors_fqn}
         WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}'),
        (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
    )
),
_to_insert AS (
    SELECT {select_cols}
    FROM _changes AS _delta
    WHERE _delta.change_type IN ('insert', 'update_postimage')
    {and_original_where_clause}
),
_to_delete AS (
    SELECT {select_cols}
    FROM _changes AS _delta
    WHERE _delta.change_type IN ('delete', 'update_preimage')
    {and_original_where_clause}
)
-- Both _to_insert and _to_delete share a single scan of ducklake_table_changes()
```

This gives classical semantics (separate insert/delete reasoning, readable SQL) with single-scan performance.

#### Maintenance statement 1: DELETE removed rows

DuckDB doesn't support `DELETE ... LIMIT 1`, so when the MV has duplicate rows we need ROW_NUMBER pairing to delete exactly one MV row per delta-delete:

```sql
WITH _changes AS (
    SELECT * FROM ducklake_table_changes('{src_catalog}', '{src_schema}', '{table_name}',
        (SELECT last_snapshot + 1 FROM {cursors_fqn}
         WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}'),
        (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
    )
),
_deletes AS (
    SELECT {select_cols},
           ROW_NUMBER() OVER (PARTITION BY {all_proj_cols} ORDER BY (SELECT NULL)) AS _dn
    FROM _changes AS _delta
    WHERE _delta.change_type IN ('delete', 'update_preimage')
    {and_original_where_clause}
),
_mv_numbered AS (
    SELECT rowid AS _rid, {all_proj_cols},
           ROW_NUMBER() OVER (PARTITION BY {all_proj_cols} ORDER BY rowid) AS _mn
    FROM {mv_table}
)
DELETE FROM {mv_table}
WHERE rowid IN (
    SELECT _rid FROM _mv_numbered m
    JOIN _deletes d ON {join_on_all_cols} AND m._mn = d._dn
)
```

This pairs each delta-delete with exactly one MV row by matching on (column values + row number within duplicate group).

#### Maintenance statement 2: INSERT new rows

```sql
WITH _changes AS (
    SELECT * FROM ducklake_table_changes('{src_catalog}', '{src_schema}', '{table_name}',
        (SELECT last_snapshot + 1 FROM {cursors_fqn}
         WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}'),
        (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
    )
)
INSERT INTO {mv_fqn} ({col_list})
SELECT {select_cols}
FROM _changes AS _delta
WHERE _delta.change_type IN ('insert', 'update_postimage')
{and_original_where_clause}
```

#### Maintenance statement 3: Update cursor

```sql
UPDATE {cursors_fqn}
SET last_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}'
```

**Note:** In DuckDB, the DELETE and INSERT are separate statements (DuckDB doesn't support multi-statement CTEs). Each repeats the `_changes` CTE, but DuckDB's query optimizer can cache the table function result. If profiling shows redundant scans, consider materializing the changes into a temp table first.

### Column rewriting with sqlglot

The `{select_cols}` and `{and_original_where_clause}` come from the original view SQL. We need to repoint column references to `_delta`:

```python
def repoint_columns(node):
    if isinstance(node, exp.Column):
        return exp.column(node.name, table="_delta")
    return node

insert_select_exprs = [expr.transform(repoint_columns) for expr in ast.selects]
```

For the WHERE clause:
```python
original_where = ast.args.get("where")
if original_where:
    delta_where = original_where.this.transform(repoint_columns)
    where_sql = f" AND {delta_where.sql(dialect=dialect)}"
else:
    where_sql = ""
```

### Building the SQL with sqlglot

Use a hybrid approach:

**For the table_changes call** — sqlglot doesn't have a built-in node for table functions with positional args. Build it as a string and embed it using `sqlglot.parse_one(fragment, dialect=dialect)`.

**For the cursor subqueries** — simple enough to build as strings.

**For column rewriting** — use `ast.transform()` to repoint column references.

**For column rewriting** — use `ast.transform()` to repoint column references.

### Implementation sketch

```python
def compile_ivm(view_sql, *, dialect="duckdb", naming=None, mv_catalog="dl",
                mv_schema="main", sources=None):
    naming = naming or Naming()
    ast = sqlglot.parse_one(view_sql, dialect=dialect)

    # --- Analysis ---
    tables = list(ast.find_all(exp.Table))
    has_agg = bool(list(ast.find_all(exp.AggFunc)))
    has_join = bool(ast.args.get("joins"))
    has_group = ast.args.get("group") is not None

    if has_join:
        raise UnsupportedSQLError("join", "Joins not yet supported")

    # --- Resolve sources ---
    table = tables[0]
    table_name = table.name
    src = _resolve_source(table_name, sources, mv_catalog, mv_schema)

    mv_table = naming.mv_table()
    mv_fqn = f"{mv_catalog}.{mv_schema}.{mv_table}"
    cursors_fqn = f"{mv_catalog}.{mv_schema}.{naming.cursors_table()}"

    # --- Projection columns ---
    proj_col_names = _extract_projection_names(ast)

    # --- Generate SQL ---
    create_cursors = _gen_create_cursors(cursors_fqn)
    create_mv = _gen_create_mv(ast, mv_fqn, src, dialect)
    init_cursors = [_gen_init_cursor(cursors_fqn, mv_table, src)]

    if has_agg:
        maintain = _gen_aggregate_maintenance(...)  # Stage 2
    else:
        maintain = _gen_select_maintenance(
            ast, mv_fqn, cursors_fqn, mv_table, src, proj_col_names, dialect
        )

    return IVMPlan(
        view_sql=ast.sql(dialect=dialect),
        create_cursors_table=create_cursors,
        create_mv=create_mv,
        initialize_cursors=init_cursors,
        maintain=maintain,
        base_tables={table_name: src["catalog"]},
        features=_detect_features(ast),
    )
```

---

## Step 2: Stage 2 — GROUP BY with COUNT, SUM

### Analysis

When the AST has `GROUP BY` and aggregate functions:

```python
group_node = ast.args.get("group")
group_cols = [col.name for col in group_node.expressions]  # e.g. ["grp"]
agg_exprs = list(ast.find_all(exp.AggFunc))               # Sum, Count, Avg
```

### MV schema changes

For aggregates, the MV needs an auxiliary `_ivm_count` column to track the number of base-table rows contributing to each group. This is essential for:
- Knowing when to delete a group (count drops to 0)
- Computing AVG as SUM/COUNT

```sql
CREATE TABLE {mv_fqn} AS
SELECT {group_cols}, {agg_exprs}, COUNT(*) AS _ivm_count
FROM {src_table}
GROUP BY {group_cols}
```

But the user's view SQL might already have COUNT — we need to handle that. The aux column is always added internally. When reading the MV for comparison with the oracle, we need to exclude `_ivm_count` from the comparison.

**Test harness change:** `read_mv` needs to exclude `_ivm_count` columns. We can do this by selecting only the columns that match the view's output columns.

### Maintenance SQL for aggregates

The bag algebra approach uses separate reasoning for ΔR (inserts) and ∇R (deletes), but the generated SQL scans `ducklake_table_changes()` only **once** via a shared CTE, then splits by change_type for aggregation.

#### Step 1: Compute delta aggregates (single scan)

```sql
CREATE TEMP TABLE _delta_agg AS
WITH _changes AS (
    SELECT * FROM ducklake_table_changes('{src_catalog}', '{src_schema}', '{table_name}',
        (SELECT last_snapshot + 1 FROM {cursors_fqn}
         WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}'),
        (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
    )
),
_ins AS (
    SELECT {group_cols}, {agg_exprs}, COUNT(*) AS _cnt
    FROM _changes WHERE change_type IN ('insert', 'update_postimage')
    GROUP BY {group_cols}
),
_del AS (
    SELECT {group_cols}, {agg_exprs}, COUNT(*) AS _cnt
    FROM _changes WHERE change_type IN ('delete', 'update_preimage')
    GROUP BY {group_cols}
)
SELECT
    COALESCE(i.{group_col}, d.{group_col}) AS {group_col},
    COALESCE(i._sum_val, 0) - COALESCE(d._sum_val, 0) AS _net_sum_val,
    COALESCE(i._cnt, 0) - COALESCE(d._cnt, 0) AS _net_count
FROM _ins i FULL OUTER JOIN _del d ON i.{group_col} = d.{group_col};
```

This scans the change feed once, computes insert and delete aggregates as separate CTEs, then combines them via FULL OUTER JOIN into net deltas per group.

#### Step 2: Update existing groups

```sql
UPDATE {mv_fqn} AS mv
SET sum_val = mv.sum_val + d._net_sum_val,
    _ivm_count = mv._ivm_count + d._net_count
FROM _delta_agg d
WHERE mv.{group_col} = d.{group_col};
```

#### Step 3: Insert new groups

```sql
INSERT INTO {mv_fqn} ({group_cols}, sum_val, _ivm_count)
SELECT d.{group_cols}, d._net_sum_val, d._net_count
FROM _delta_agg d
WHERE NOT EXISTS (
    SELECT 1 FROM {mv_fqn} mv WHERE mv.{group_col} = d.{group_col}
)
AND d._net_count > 0;
```

#### Step 4: Delete emptied groups

```sql
DELETE FROM {mv_fqn}
WHERE _ivm_count <= 0
```

#### Step 5: Cleanup and update cursor

```sql
DROP TABLE IF EXISTS _delta_agg;

UPDATE {cursors_fqn}
SET last_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}';
```

### Handling different aggregate functions

| Original | Insert aggregate (ΔR) | Delete aggregate (∇R) | Net delta |
|----------|----------------------|----------------------|-----------|
| `SUM(x)` | `SUM(x)` from inserts | `SUM(x)` from deletes | `ins - del` |
| `COUNT(*)` | `COUNT(*)` from inserts | `COUNT(*)` from deletes | `ins - del` |
| `COUNT(x)` | `COUNT(x)` from inserts | `COUNT(x)` from deletes | `ins - del` |
| `AVG(x)` | Hidden `_ivm_sum` and `_ivm_count` | Same | `avg = sum/count` |

### Putting it together for aggregates

```python
maintain = [
    # 1. Compute net delta aggregates (single scan of change feed)
    create_delta_agg_sql,
    # 2. Update existing groups
    update_existing_groups_sql,
    # 3. Insert new groups
    insert_new_groups_sql,
    # 4. Delete empty groups
    delete_empty_groups_sql,
    # 5. Cleanup and update cursor
    drop_delta_agg_sql,
    update_cursor_sql,
]
```

Using a temp table since the net delta is referenced by UPDATE, INSERT, and DELETE. The single `_changes` CTE ensures one scan of `ducklake_table_changes()`.

---

## Step 3: Test Harness Updates

### Changes to `assert_ivm_correct`

The current test creates the MV manually as `memory.main.mv`. With the compiler now generating `create_mv`, we need to:

1. Use `plan.create_cursors_table` and `plan.create_mv`
2. Use `plan.initialize_cursors`
3. Read the MV from wherever the plan puts it (in the DuckLake catalog)
4. For comparison, exclude auxiliary `_ivm_*` columns

Updated `assert_ivm_correct`:

```python
def assert_ivm_correct(scenario, ducklake_fixture):
    con, catalog = ducklake_fixture

    # 1. Set up DuckLake tables with initial data
    setup_scenario(con, scenario, catalog)

    # 2. Compile IVM
    plan = compile_ivm(
        scenario.view_sql,
        dialect="duckdb",
        mv_catalog=catalog,
    )

    # 3. Set up MV and cursors
    con.execute(plan.create_cursors_table)
    con.execute(plan.create_mv)
    for stmt in plan.initialize_cursors:
        con.execute(stmt)

    # 4. Apply deltas to DuckLake base tables
    apply_deltas(con, scenario, catalog)

    # 5. Run maintenance SQL
    for stmt in plan.maintain:
        con.execute(stmt)

    # 6. Read maintained MV (excluding _ivm_* columns)
    mv_cols = get_non_aux_columns(con, catalog, plan)
    maintained = read_mv_cols(con, catalog, plan, mv_cols)

    # 7. Recompute from scratch and compare
    expected = recompute_view(con, scenario.view_sql, catalog)

    assert maintained == expected, ...
```

Helper to get non-auxiliary columns:

```python
def get_non_aux_columns(con, catalog, plan):
    """Get MV column names, excluding _ivm_* auxiliary columns."""
    schema = con.execute(
        f"DESCRIBE {catalog}.main.{plan.view_sql}"  # or use PRAGMA
    ).fetchall()
    # Actually simpler: just query and filter column names
    result = con.execute(f"SELECT * FROM {catalog}.main.mv LIMIT 0")
    return [desc[0] for desc in result.description if not desc[0].startswith("_ivm_")]
```

---

## Step 4: Implementation Order

### Phase 1: Infrastructure (Step 0)
1. Create `src/duckstream/plan.py` with `IVMPlan`, `Naming`, `UnsupportedSQLError`
2. Update `src/duckstream/__init__.py` to re-export
3. Update `src/duckstream/compiler.py` with new signature (still raises `NotImplementedError`)
4. Update `tests/test_ivm.py` to use the new `IVMPlan` protocol in `assert_ivm_correct`
5. Verify tests still fail with `NotImplementedError`

### Phase 2: Stage 1 — SELECT/WHERE
1. Implement `compile_ivm` for the non-aggregate, non-join case
2. Generate `create_cursors_table`, `create_mv`, `initialize_cursors`
3. Generate maintenance: DELETE + INSERT + UPDATE cursor
4. Run smoke tests: `test_simple_select_all`, `test_select_with_filter`
5. Run property test: `test_select_project_filter` (50 examples)
6. Debug any failures (Hypothesis will shrink to minimal cases)

### Phase 3: Stage 2 — Aggregates (COUNT, SUM, AVG)
1. Add aggregate detection to the compiler
2. Generate MV with `_ivm_count` auxiliary column
3. Generate maintenance: temp delta_agg table, UPDATE existing, INSERT new, DELETE empty, cursor
4. Handle SUM, COUNT(*), AVG (as SUM/COUNT)
5. Update test harness to exclude `_ivm_*` columns from comparison
6. Run smoke tests: `test_count_aggregate`, `test_sum_aggregate`
7. Run property test: `test_single_table_aggregate` (50 examples)

### Phase 4: Stage 4 — Two-Table Inner JOIN
1. Detect JOIN nodes in the AST; extract join condition
2. Resolve source catalogs for both tables
3. Read deltas for each base table via `ducklake_table_changes()`
4. Generate maintenance SQL implementing the three-way decomposition:
   - `ΔR ⋈ S_current` — delta from left table joined against current right table
   - `R_current ⋈ ΔS` — current left table joined against delta from right table
   - `ΔR ⋈ ΔS` — cross-delta term (both changed simultaneously)
5. Each term uses the same join condition and WHERE/projection from the original view
6. Results are UNION ALL'd; inserts (+1 weight) go to INSERT, deletes (-1 weight) go to DELETE
7. Write `two_table_join()` Hypothesis strategy and smoke tests
8. Run property tests

### Phase 5: Stage 5 — JOIN + Aggregates (Composed)
1. Compose join delta with aggregate maintenance (bottom-up AST rewrite)
2. The join delta feeds into the weighted aggregate pattern from Phase 3
3. Write `join_then_aggregate()` strategy and smoke tests

### Phase 6: Refinement
1. Handle edge cases found by Hypothesis
2. Handle NULL values in projections (NULL = NULL matching for deletes)
3. Handle empty delta ranges (no-op maintenance)
4. Add `features` detection to `IVMPlan`

### Quality gates (every phase)
1. `uv run ruff check src/ tests/` — all linting passes
2. `uv run ruff format --check src/ tests/` — formatting is consistent
3. `uv run ty check src/` — type checking passes
4. `uv run pytest` — all tests pass

---

## Step 5: Stage 4 — Two-Table Inner JOIN

### Bag algebra join rule

When only R changes (common fast path):
```
ΔV = ΔR ⋈ S       (rows to insert into MV)
∇V = ∇R ⋈ S       (rows to delete from MV)
```

When both R and S change simultaneously (subtract to avoid double-counting):
```
ΔV = (ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS)
∇V = (∇R ⋈ S) ∪ (R ⋈ ∇S) - (∇R ⋈ ∇S)
```

Here `R` and `S` are the **post-update (current)** table state. Since `R` already includes `ΔR`, the term `R ⋈ ΔS` already counts `ΔR ⋈ ΔS` — we must subtract it. This is opposite to the DBSP Z-set formula (which uses pre-update state and adds the third term).

### Analysis phase

```python
joins = ast.args.get("joins")  # list of Join nodes
if joins and len(joins) == 1:
    join = joins[0]
    right_table = join.this  # Table node for right side
    left_table = ast.args["from"].this  # Table node for left side
    on_condition = join.args["on"]  # EQ or And(EQ, EQ, ...)
```

### Maintenance SQL structure

For a view `SELECT ... FROM R JOIN S ON R.k = S.k [WHERE ...] [GROUP BY ...]`:

**When only one table changes (common fast path):**

Separate INSERT and DELETE passes, each joining the delta against the current other table:

```sql
-- INSERT: ΔR ⋈ S (new R rows joined against current S)
INSERT INTO {mv_fqn} ({proj_cols})
SELECT {proj_cols}
FROM ducklake_table_changes('{r_catalog}', '{r_schema}', '{r_table}',
    (SELECT last_snapshot + 1 FROM {cursors_fqn}
     WHERE mv_name = '{mv_table}' AND source_catalog = '{r_catalog}'),
    (SELECT MAX(snapshot_id) FROM ducklake_snapshots('{r_catalog}'))
) AS _delta_r
JOIN {s_fqn} AS s ON {join_cond}
WHERE _delta_r.change_type IN ('insert', 'update_postimage')
{and_where_clause};

-- DELETE: ∇R ⋈ S (deleted R rows joined against current S)
-- Use ROW_NUMBER matching to delete exactly one MV row per delta match
DELETE FROM {mv_fqn} WHERE rowid IN (
    ... -- same ROW_NUMBER pairing pattern as Stage 1
);
```

**When both tables change simultaneously:**

The double-counting correction uses `EXCEPT` or `NOT EXISTS` to subtract `ΔR ⋈ ΔS`:

```sql
-- INSERT: (ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS)
INSERT INTO {mv_fqn} ({proj_cols})
SELECT {proj_cols} FROM (
    -- Term 1: new R rows joined against current S
    SELECT {proj_cols} FROM delta_R_inserts JOIN S ON ...
    UNION ALL
    -- Term 2: current R joined against new S rows
    SELECT {proj_cols} FROM R JOIN delta_S_inserts ON ...
) AS combined
WHERE NOT EXISTS (
    -- Subtract double-counted cross-delta matches
    SELECT 1 FROM delta_R_inserts dr JOIN delta_S_inserts ds
    ON dr.join_key = ds.join_key
    WHERE combined.{pk_cols} = <derived_pk>
);

-- DELETE: analogous structure for (∇R ⋈ S) ∪ (R ⋈ ∇S) - (∇R ⋈ ∇S)
```

### Optimization: single-table-change detection

The compiler should detect which source tables actually have pending changes (via the cursor table) and skip the cross-delta correction term when only one table changed. This is the common case and produces simpler, faster SQL — a single join instead of UNION ALL + EXCEPT.

### Cross-delta term rationale

Since `R` and `S` are post-update (current) state:
- `R ⋈ ΔS` already includes `ΔR ⋈ ΔS` (because R contains ΔR)
- `ΔR ⋈ S` also includes `ΔR ⋈ ΔS` (because S contains ΔS)
- So the union double-counts `ΔR ⋈ ΔS` — we subtract it

This is the classical bag algebra result. Contrast with DBSP Z-sets which use pre-update state and *add* the third term.

### New test infrastructure

**Strategy: `two_table_join()`**

```python
@st.composite
def two_table_join(draw):
    """Two tables with an inner join on a shared key column."""
    key_name = draw(col_names)
    used = {key_name}

    # Left table: key + extra columns
    left_extras = [Column(draw(col_names.filter(lambda n: n not in used)), draw(col_type()))
                   for _ in range(draw(st.integers(1, 3)))]
    left_table = Table(draw(table_names.filter(lambda n: n not in used)),
                       [Column(key_name, "INTEGER")] + left_extras)
    used.add(left_table.name)

    # Right table: key + extra columns
    right_extras = [Column(draw(col_names.filter(lambda n: n not in used)), draw(col_type()))
                    for _ in range(draw(st.integers(1, 3)))]
    right_table = Table(draw(table_names.filter(lambda n: n not in used)),
                        [Column(key_name, "INTEGER")] + right_extras)

    # Generate data with overlapping keys
    left_rows = draw(rows_for_table(left_table, min_size=2, max_size=10))
    right_rows = draw(rows_for_table(right_table, min_size=2, max_size=10))

    # View SQL
    proj = draw(...)  # subset of columns from both tables
    view_sql = f"SELECT {proj} FROM {left_table.name} JOIN {right_table.name} ON ..."

    # Deltas to one or both tables
    ...

    return Scenario(tables=[left_table, right_table], ...)
```

**Smoke tests:**
- Inner join, insert into left table only → new matches appear
- Inner join, insert into right table only → new matches appear
- Inner join, insert into both tables → cross-delta term exercised
- Inner join, delete row that participates in join → matches removed
- Inner join, delete row that doesn't participate → MV unchanged

### Column qualification challenge

With joins, columns need table qualifiers (`R.col` vs `S.col`). The compiler must:
1. Know which columns come from which table
2. When rewriting for the delta query, repoint columns to `_delta_r` or `_delta_s` as appropriate
3. Use `sqlglot.optimizer.qualify` to ensure all columns are table-qualified before transformation

```python
from sqlglot.optimizer.qualify import qualify

# Build a schema dict for qualify
schema = {table.name: {col.name: col.dtype for col in table.columns} for table in tables}
qualified_ast = qualify(ast, schema=schema, dialect=dialect)
# Now every Column node has .table populated
```

---

## Step 6: Stage 5 — JOIN + Aggregates (Composed)

The incremental form of a composition is the composition of incremental forms. This means:

1. Compute the join delta (as in Step 5)
2. Feed the join delta through the aggregate maintenance (as in Step 2)

The compiler walks the AST bottom-up:
- First, rewrite the join to produce a delta stream
- Then, rewrite the aggregate to consume that delta stream

In practice, this means the aggregate maintenance receives a weighted delta (from the join decomposition) and applies the same UPDATE/INSERT/DELETE logic from Stage 2, but the "delta source" is the join delta rather than a single table's change feed.

### Implementation approach

Rather than generating separate join + aggregate SQL, generate a single set of maintenance statements where the aggregate's delta source is the join delta. The join produces separate ΔV (rows to insert) and ∇V (rows to delete), which feed into the same single-scan aggregate pattern from Stage 2:

```sql
-- Single scan: compute join deltas for both inserts and deletes,
-- then aggregate each into net delta per group
CREATE TEMP TABLE _delta_agg AS
WITH _changes_r AS (
    SELECT * FROM ducklake_table_changes('{r_catalog}', ..., {snap_range_r})
),
_changes_s AS (
    SELECT * FROM ducklake_table_changes('{s_catalog}', ..., {snap_range_s})
),
_join_ins AS (
    -- ΔR ⋈ S (with correction if both changed)
    SELECT {proj_cols} FROM _changes_r cr JOIN {s_fqn} s ON ...
    WHERE cr.change_type IN ('insert', 'update_postimage')
    {correction_for_double_count}
),
_join_del AS (
    -- ∇R ⋈ S (with correction if both changed)
    SELECT {proj_cols} FROM _changes_r cr JOIN {s_fqn} s ON ...
    WHERE cr.change_type IN ('delete', 'update_preimage')
    {correction_for_double_count}
),
_ins_agg AS (
    SELECT {group_cols}, {agg_exprs}, COUNT(*) AS _cnt FROM _join_ins GROUP BY {group_cols}
),
_del_agg AS (
    SELECT {group_cols}, {agg_exprs}, COUNT(*) AS _cnt FROM _join_del GROUP BY {group_cols}
)
SELECT COALESCE(i.{group_col}, d.{group_col}) AS {group_col},
       COALESCE(i._sum_val, 0) - COALESCE(d._sum_val, 0) AS _net_sum_val,
       COALESCE(i._cnt, 0) - COALESCE(d._cnt, 0) AS _net_count
FROM _ins_agg i FULL OUTER JOIN _del_agg d ON i.{group_col} = d.{group_col};

-- Then UPDATE / INSERT / DELETE as in Stage 2
```

This composes cleanly: the join delta is split into insert/delete streams via CTE, aggregated separately, then combined into net deltas — all in a single scan of each table's change feed.

---

## Key Design Decisions

### String building vs. sqlglot AST

For the generated maintenance SQL, use a hybrid approach:
- Use sqlglot AST manipulation for the parts derived from the user's view SQL (projections, WHERE clause, column rewriting)
- Use string formatting for the DuckLake-specific boilerplate (`ducklake_table_changes(...)`, cursor reads/writes, `CREATE TABLE IF NOT EXISTS`)

Rationale: `ducklake_table_changes()` is a DuckDB-specific table function that sqlglot may not model perfectly. The cursor tracking SQL is simple and fixed. The user's view SQL is the part that benefits from AST manipulation.

### Single-scan pattern

Even though we reason about ΔR and ∇R separately (classical bag algebra), the generated SQL scans `ducklake_table_changes()` once per statement via a shared CTE. For aggregates, we materialize the net delta into a temp table (since it's referenced by UPDATE, INSERT, and DELETE as separate statements). DuckDB `CREATE TEMP TABLE` is fast and scoped to the connection.

### Row-matching for non-aggregate deletes

Use the ROW_NUMBER partition approach described above. This correctly handles duplicate rows in the MV — each delta-delete removes exactly one matching MV row.

### Column qualification

Before generating maintenance SQL, use sqlglot's column extraction to identify which columns come from which table. For Stage 1 (single table), all columns come from one table, so we just repoint them to `_delta`. For Stage 4+ (joins), we'll need proper table qualification.

### MV location

The MV lives in the DuckLake catalog (`mv_catalog.mv_schema.mv_table`), not in the `memory` catalog. This gives the MV time travel support and is consistent with the requirements doc.

### Snapshot subquery pattern

The cursor read is always:
```sql
(SELECT last_snapshot + 1 FROM {cursors_fqn}
 WHERE mv_name = '{mv_table}' AND source_catalog = '{src_catalog}')
```

And the current snapshot is:
```sql
(SELECT MAX(snapshot_id) FROM ducklake_snapshots('{src_catalog}'))
```

These are embedded as subqueries in the `ducklake_table_changes()` call, making the maintenance SQL fully self-contained.

### Linting and type checking

Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [ty](https://docs.astral.sh/ty/) for type checking. Both are from Astral and work with uv.

**Setup:** Add as dev dependencies:
```bash
uv add --dev ruff ty
```

**Run on every change:**
```bash
uv run ruff check src/ tests/       # lint
uv run ruff format src/ tests/       # format (or --check to verify)
uv run ty check src/                 # type check library code
```

**Ruff configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
# E/F/W: pyflakes + pycodestyle basics
# I: isort (import sorting)
# UP: pyupgrade (modern Python syntax)
# B: flake8-bugbear (common bugs)
# SIM: flake8-simplify
```

All code must pass `ruff check` and `ty check` before being considered complete. Format with `ruff format` before committing.
