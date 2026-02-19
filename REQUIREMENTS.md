# Python API Requirements

## Overview

The library is a **compiler**: SQL in, SQL out. It takes a view definition and emits the SQL statements needed to incrementally maintain that view as a DuckLake table. It does not execute anything — the caller owns the database connection and execution.

**All base tables and the materialized view are DuckLake tables.** Deltas are discovered automatically via DuckLake's `ducklake_table_changes()` function — there are no manually-populated delta tables. The caller just writes to their base tables normally, and the generated maintenance SQL reads the changes from the DuckLake change feed.

**Source tables can live in different DuckLake catalogs.** A view can join tables from `catalog_a` and `catalog_b`, with the MV written to `catalog_c`. The compiler tracks per-source snapshot cursors in a shared tracking table (`_ivm_cursors`) in the MV's catalog.

An optional execution helper may be added later as a separate module.

## Core API

### `compile_ivm(view_sql, *, dialect, naming, mv_catalog, mv_schema, sources) -> MaterializedView`

The single entry point. Takes a view definition (a SELECT statement as a string) and returns a structured plan containing all the SQL needed to create and maintain the materialized view.

```python
from duckstream import compile_ivm

# Simple case: all tables in one catalog
plan = compile_ivm(
    "SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
    dialect="duckdb",
    mv_catalog="dl",
)

# Cross-catalog: sources in different DuckLakes, MV in a third
plan = compile_ivm(
    "SELECT o.customer_id, SUM(o.amount) AS total, c.name "
    "FROM orders o JOIN customers c ON o.customer_id = c.id "
    "GROUP BY o.customer_id, c.name",
    dialect="duckdb",
    mv_catalog="analytics",
    sources={
        "orders": {"catalog": "sales_dl", "schema": "main"},
        "customers": {"catalog": "crm_dl", "schema": "main"},
    },
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `view_sql` | `str` | required | The SELECT statement defining the view. Table names should be unqualified. |
| `dialect` | `str` | `"duckdb"` | Target SQL dialect for output |
| `naming` | `Naming` | `Naming()` | Controls naming of generated tables and columns |
| `mv_catalog` | `str` | `"dl"` | The DuckLake catalog where the MV and cursor tracking table live |
| `mv_schema` | `str` | `"main"` | The schema for the MV within its catalog |
| `sources` | `dict` | `None` | Per-table source catalog/schema overrides. If `None`, all tables are assumed to be in `mv_catalog`/`mv_schema`. Keys are unqualified table names, values are dicts with `catalog` and optional `schema` (defaults to `"main"`). |

### `Naming`

A class with overridable methods that control all generated names. Subclass to customize.

```python
from duckstream import Naming

class Naming:
    """Default naming strategy. Subclass to override any method."""

    def mv_table(self) -> str:
        """Name of the materialized view table."""
        return "mv"

    def cursors_table(self) -> str:
        """Name of the shared snapshot cursor tracking table."""
        return "_ivm_cursors"

    def aux_column(self, purpose: str) -> str:
        """Name of an auxiliary column. Purpose is e.g. 'count', 'sum', 'weight', 'multiplicity'."""
        return f"_ivm_{purpose}"

# Using defaults
plan = compile_ivm(view_sql)

# Custom naming via subclass
class MyNaming(Naming):
    def mv_table(self) -> str:
        return "sales_by_region_mv"

plan = compile_ivm(view_sql, naming=MyNaming())
```

### `MaterializedView`

The output dataclass. All SQL strings are ready to execute in the target dialect.

```python
@dataclass
class MaterializedView:
    """Complete set of SQL statements for IVM maintenance."""

    # The original view SQL, normalized by sqlglot
    view_sql: str

    # DDL: create the snapshot cursor tracking table (shared across all MVs in this catalog).
    # Safe to execute multiple times — uses CREATE TABLE IF NOT EXISTS.
    create_cursors_table: str

    # DDL: create the materialized view table with initial data
    create_mv: str

    # DML: initialize cursor rows for this MV (one per source catalog)
    initialize_cursors: list[str]

    # DML: maintenance statements that:
    #   1. Read current cursors from the tracking table
    #   2. Read deltas from ducklake_table_changes() for each source
    #   3. Apply changes to the MV
    #   4. Update the cursor rows
    # Execute in order. No parameters needed — snapshot ranges are
    # read from and written to the cursors table automatically.
    maintain: list[str]

    # The base tables referenced by the view, with their source catalogs
    base_tables: dict[str, str]     # table_name -> catalog

    # Which SQL features this view uses (for caller introspection)
    features: set[str]              # e.g. {"select", "where", "join", "group_by", "sum", "distinct"}
```

## Snapshot Cursor Tracking

The compiler emits a shared tracking table in the MV's DuckLake catalog that records, for each MV, the last-processed snapshot per source catalog:

```sql
CREATE TABLE IF NOT EXISTS analytics._ivm_cursors (
    mv_name VARCHAR,
    source_catalog VARCHAR,
    last_snapshot BIGINT,
    PRIMARY KEY (mv_name, source_catalog)
);
```

Example contents when two MVs exist in `analytics`:

```
analytics._ivm_cursors:
┌─────────────────────┬────────────────┬───────────────┐
│ mv_name             │ source_catalog │ last_snapshot  │
├─────────────────────┼────────────────┼───────────────┤
│ sales_by_region     │ sales_dl       │ 42             │
│ sales_by_region     │ crm_dl         │ 17             │
│ customer_totals     │ sales_dl       │ 38             │
└─────────────────────┴────────────────┴───────────────┘
```

The generated maintenance SQL reads the current cursor values, uses them as the start snapshot for `ducklake_table_changes()`, and updates them after processing. This makes maintenance calls self-contained — the caller just executes `plan.maintain` without needing to pass snapshot parameters.

## Delta Discovery via DuckLake

There are no delta tables to create or populate. The compiler generates maintenance SQL that reads changes directly from DuckLake's change feed:

```sql
-- The compiler generates maintenance SQL like this:
-- (simplified example for SELECT id, val FROM t WHERE val > 10)
-- Source table 't' is in catalog 'dl', MV is in catalog 'analytics'

INSERT INTO analytics.mv (id, val)
SELECT id, val
FROM ducklake_table_changes('dl', 'main', 't',
    (SELECT last_snapshot FROM analytics._ivm_cursors
     WHERE mv_name = 'mv' AND source_catalog = 'dl'),
    (SELECT MAX(snapshot_id) FROM ducklake_snapshots(dl)))
WHERE change_type IN ('insert', 'update_postimage')
  AND val > 10;

DELETE FROM analytics.mv
WHERE rowid IN (
    SELECT mv.rowid FROM analytics.mv AS mv
    JOIN ducklake_table_changes('dl', 'main', 't',
        (SELECT last_snapshot FROM analytics._ivm_cursors
         WHERE mv_name = 'mv' AND source_catalog = 'dl'),
        (SELECT MAX(snapshot_id) FROM ducklake_snapshots(dl))) AS delta
      ON mv.id = delta.id AND mv.val = delta.val
    WHERE delta.change_type IN ('delete', 'update_preimage')
);

-- Update cursor
UPDATE analytics._ivm_cursors
SET last_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshots(dl))
WHERE mv_name = 'mv' AND source_catalog = 'dl';
```

DuckLake's `ducklake_table_changes()` returns rows with a `change_type` column:
- `'insert'` — a new row (ΔR — rows to add to the MV)
- `'delete'` — a removed row (∇R — rows to remove from the MV)
- `'update_preimage'` — row state before update (treated as ∇R)
- `'update_postimage'` — row state after update (treated as ΔR)

The compiler uses **classical bag algebra** — separate insert (ΔR) and delete (∇R) passes against the post-update (current) table state. See `bag-algebra-vs-z-sets.md` for the rationale.

## Execution Order Contract

The caller is responsible for executing the SQL in the correct order. The contract:

1. **Setup (once):**
   - Execute `plan.create_cursors_table` (safe to run multiple times)
   - Execute `plan.create_mv` to create and populate the MV
   - Execute `plan.initialize_cursors` to record the initial snapshot per source

2. **Per maintenance cycle:**
   - Write to the base tables normally (DuckLake records snapshots automatically)
   - Execute `plan.maintain` statements **in order**
   - That's it — the maintenance SQL reads and updates the cursors automatically

The maintenance SQL is fully self-contained. No snapshot parameters need to be passed by the caller.

## Snapshot Safety

DuckLake allows expiring old snapshots to reclaim storage. However, expiring a snapshot that an MV's cursor still depends on will break `ducklake_table_changes()`. The library provides a function to generate SQL that reports which snapshots are safe to drop across all IVM catalogs.

### `safe_to_expire_sql(mv_catalogs, source_catalogs) -> str`

Generates a single SQL query that reports, for each source catalog, the minimum snapshot that is still required by any MV and therefore the maximum snapshot that is safe to expire.

```python
from duckstream import safe_to_expire_sql

# Which catalogs have MV cursors tables, and which catalogs are sources
sql = safe_to_expire_sql(
    mv_catalogs=["analytics", "reporting"],
    source_catalogs=["sales_dl", "crm_dl", "analytics"],
)

result = con.execute(sql).fetchall()
# Returns rows like:
# [
#   ("sales_dl",  42, 150),   -- (catalog, min_required_snapshot, latest_snapshot)
#   ("crm_dl",    17, 88),
#   ("analytics", 38, 200),
# ]
# For sales_dl: snapshots 0–41 are safe to expire. 42+ are still needed.
# For crm_dl:   snapshots 0–16 are safe to expire. 17+ are still needed.
```

The generated SQL scans the `_ivm_cursors` table in each MV catalog and takes the minimum `last_snapshot` per source catalog across all of them. This is the oldest snapshot still needed by any MV anywhere.

```python
@dataclass
class SnapshotSafety:
    """Per-catalog snapshot expiration guidance."""
    source_catalog: str
    min_required_snapshot: int      # oldest snapshot still needed (do NOT expire)
    latest_snapshot: int            # current latest snapshot in this catalog
    safe_to_expire_before: int      # = min_required_snapshot (snapshots < this are safe)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mv_catalogs` | `list[str]` | All DuckLake catalogs that contain `_ivm_cursors` tables (i.e., catalogs where MVs live) |
| `source_catalogs` | `list[str]` | All DuckLake catalogs to check expiration safety for. Typically the union of all source catalogs across all MVs. |
| `naming` | `Naming` | Optional. Same `Naming` class to find the cursors table name. Defaults to `Naming()`. |

### Why both `mv_catalogs` and `source_catalogs`?

An MV in `analytics` might depend on `sales_dl` and `crm_dl`. A different MV in `reporting` might also depend on `sales_dl`. To know the safe-to-expire threshold for `sales_dl`, you need to check the cursors tables in *both* `analytics` and `reporting` and take the minimum.

The caller provides the full list of MV catalogs so the query can scan all of them. `source_catalogs` controls which catalogs appear in the output.

### Example: Expire old snapshots

```python
from duckstream import safe_to_expire_sql

sql = safe_to_expire_sql(
    mv_catalogs=["analytics", "reporting"],
    source_catalogs=["sales_dl", "crm_dl"],
)

for catalog, min_required, latest in con.execute(sql).fetchall():
    safe_before = min_required
    if safe_before > 0:
        print(f"{catalog}: expiring snapshots before {safe_before}")
        con.execute(
            f"CALL ducklake_expire_snapshots('{catalog}', older_than := "
            f"(SELECT snapshot_timestamp FROM ducklake_snapshots({catalog}) "
            f"WHERE snapshot_id = {safe_before}))"
        )
    else:
        print(f"{catalog}: no snapshots safe to expire")
```

## Pending Maintenance Status

Before running maintenance, the caller may want to know: which MVs have pending work, how much has changed, and what will be touched? The library provides a function to generate a SQL query that reports this.

### `pending_maintenance_sql(mv_catalogs) -> str`

Generates a SQL query that reports, for each MV in the given catalogs, how many changes are pending per source table and the MV that would be updated.

```python
from duckstream import pending_maintenance_sql

sql = pending_maintenance_sql(mv_catalogs=["analytics", "reporting"])

result = con.execute(sql).fetchall()
# Returns rows like:
# [
#   ("analytics", "sales_by_region", "sales_dl",  "orders",    82, 3),
#   ("analytics", "sales_by_region", "crm_dl",    "customers",  5, 1),
#   ("reporting", "daily_totals",    "sales_dl",  "orders",    82, 3),
# ]
```

```python
@dataclass
class PendingMaintenance:
    """Per-MV, per-source-table maintenance status."""
    mv_catalog: str                 # catalog the MV lives in
    mv_name: str                    # the MV table name
    source_catalog: str             # catalog the source table lives in
    source_table: str               # the source table name
    pending_changes: int            # number of change rows from ducklake_table_changes()
    pending_snapshots: int          # number of unprocessed snapshots
```

The generated SQL works by joining each `_ivm_cursors` row against `ducklake_table_changes()` (counting rows) and `ducklake_snapshots()` (counting snapshots since the cursor). This gives the caller a quick picture of how stale each MV is and how much work maintenance will do.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mv_catalogs` | `list[str]` | All DuckLake catalogs that contain `_ivm_cursors` tables |
| `naming` | `Naming` | Optional. Same `Naming` class to find the cursors table name. Defaults to `Naming()`. |

### Example: Selective maintenance

```python
from duckstream import pending_maintenance_sql

sql = pending_maintenance_sql(mv_catalogs=["analytics"])

for mv_cat, mv_name, src_cat, src_table, n_changes, n_snaps in con.execute(sql).fetchall():
    print(f"{mv_cat}.{mv_name}: {n_changes} changes from {src_cat}.{src_table} ({n_snaps} snapshots behind)")

# analytics.sales_by_region: 82 changes from sales_dl.orders (3 snapshots behind)
# analytics.sales_by_region: 5 changes from crm_dl.customers (1 snapshots behind)

# The caller can decide:
# - Skip maintenance if pending_changes is 0
# - Prioritize MVs with the most pending work
# - Alert if an MV is too many snapshots behind
```

## Error Handling

```python
from duckstream import compile_ivm, UnsupportedSQLError

try:
    plan = compile_ivm("SELECT *, ROW_NUMBER() OVER (...) FROM t")
except UnsupportedSQLError as e:
    # e.feature  -> "window_function"
    # e.message  -> "Window functions are not supported for incremental maintenance"
    print(e)
```

The compiler raises `UnsupportedSQLError` for SQL features it cannot incrementally maintain. It should **never** silently produce incorrect maintenance SQL.

Supported features and their status:

| Feature | Status |
|---------|--------|
| SELECT / PROJECT / WHERE | MVP |
| JOIN (inner) | MVP |
| GROUP BY + SUM, COUNT, AVG | MVP |
| GROUP BY + MIN, MAX | MVP (rescan fallback) |
| DISTINCT | MVP |
| LEFT / RIGHT / FULL OUTER JOIN | Post-MVP |
| UNION / EXCEPT / INTERSECT | Post-MVP |
| Window functions | Unsupported — raises error |
| Correlated subqueries | Unsupported — raises error |
| Recursive CTEs | Unsupported — raises error |
| ORDER BY / LIMIT | Unsupported — raises error |

## Example: Full Workflow

### Single catalog

```python
import duckdb
from duckstream import compile_ivm

con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute("ATTACH 'ducklake:meta.ddb' AS dl (DATA_PATH './data')")

con.execute("CREATE TABLE dl.orders (id INT, customer_id INT, amount DECIMAL)")
con.execute("INSERT INTO dl.orders VALUES (1, 7, 49.99), (2, 3, 120.00)")

# 1. Compile
plan = compile_ivm(
    "SELECT customer_id, SUM(amount) AS total, COUNT(*) AS num_orders "
    "FROM orders GROUP BY customer_id",
    dialect="duckdb",
    mv_catalog="dl",
)

# 2. Setup (once)
con.execute(plan.create_cursors_table)
con.execute(plan.create_mv)
for stmt in plan.initialize_cursors:
    con.execute(stmt)

# 3. Later, new orders arrive — just write normally
con.execute("INSERT INTO dl.orders VALUES (3, 7, 25.00)")
con.execute("DELETE FROM dl.orders WHERE id = 2")

# 4. Maintain the MV
for stmt in plan.maintain:
    con.execute(stmt)

# Done. The cursors table tracks where we left off.
# Next time, just run plan.maintain again.
```

### Cross-catalog

```python
import duckdb
from duckstream import compile_ivm

con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute("ATTACH 'ducklake:sales.ddb' AS sales_dl (DATA_PATH './sales_data')")
con.execute("ATTACH 'ducklake:crm.ddb' AS crm_dl (DATA_PATH './crm_data')")
con.execute("ATTACH 'ducklake:analytics.ddb' AS analytics (DATA_PATH './analytics_data')")

# Source tables in different catalogs
con.execute("CREATE TABLE sales_dl.orders (id INT, customer_id INT, amount DECIMAL)")
con.execute("CREATE TABLE crm_dl.customers (id INT, name VARCHAR, region VARCHAR)")

# MV in a third catalog, joining across the other two
plan = compile_ivm(
    "SELECT c.region, SUM(o.amount) AS total "
    "FROM orders o JOIN customers c ON o.customer_id = c.id "
    "GROUP BY c.region",
    dialect="duckdb",
    mv_catalog="analytics",
    sources={
        "orders": {"catalog": "sales_dl"},
        "customers": {"catalog": "crm_dl"},
    },
)

# Setup
con.execute(plan.create_cursors_table)
con.execute(plan.create_mv)
for stmt in plan.initialize_cursors:
    con.execute(stmt)

# Write to source tables in their respective catalogs
con.execute("INSERT INTO sales_dl.orders VALUES (1, 1, 100.00)")
con.execute("INSERT INTO crm_dl.customers VALUES (1, 'Alice', 'West')")

# Maintain — reads changes from both source catalogs, updates MV in analytics
for stmt in plan.maintain:
    con.execute(stmt)

# analytics._ivm_cursors now tracks:
#   mv_name='mv', source_catalog='sales_dl', last_snapshot=<latest>
#   mv_name='mv', source_catalog='crm_dl',   last_snapshot=<latest>
```

## Design Principles

1. **Pure compiler.** No side effects, no connections, no state. Given the same inputs, `compile_ivm` always returns the same `MaterializedView`.

2. **Correct or loud.** The compiler either produces correct maintenance SQL or raises `UnsupportedSQLError`. No silent incorrectness.

3. **DuckLake-native.** Base tables and the MV are all DuckLake tables. Deltas come from `ducklake_table_changes()`, not from manually-managed delta tables. The MV gets DuckLake time travel for free.

4. **Cross-catalog.** Source tables can live in different DuckLake catalogs. The MV can live in yet another. Snapshot cursors are tracked per-source in the MV's catalog.

5. **Self-contained maintenance.** The generated maintenance SQL reads and updates its own cursor state. The caller just executes the statements — no snapshot tracking required in application code.
