# IVM on DuckLake: Why Classical Bag Algebra Is the Right Choice

## Context

We're building a SQL-to-SQL IVM compiler in Python (using sqlglot) that targets DuckDB with DuckLake as the storage layer. DuckLake provides full time travel and a cheap `ducklake_table_changes()` function that outputs transition tables — the set of rows inserted into and deleted from a base table between two versions.

This document explains why the classical bag algebra approach (as used by pg_ivm) is the correct choice over the DBSP Z-set formulation, and works through the algorithmic implications in detail.

## Why Classical Bag Algebra, Not Z-Sets

The Z-set formulation from DBSP exists to solve two problems we don't have:

**Problem 1: Unknown transaction timing.** In a portable, CDC-driven system, you don't know whether the base table reflects the pre-update or post-update state when your maintenance logic runs. Z-sets solve this by working entirely with the pre-update state and requiring a three-term join decomposition. We don't have this problem — DuckLake time travel gives us access to both the old and new states of every table at any point. We can always query `R AT VERSION old` or `R AT VERSION new`.

**Problem 2: Composability across external systems.** Z-sets unify inserts and deletes into signed weights, which compose cleanly when deltas flow between systems that don't share transaction boundaries. We don't have this problem — we're in a single DuckDB instance with DuckLake.

What we *do* have is cheap access to transition tables via `ducklake_table_changes()`, which gives us exactly the `ΔR` (inserted rows) and `∇R` (deleted rows) that classical bag algebra expects. And we have access to the new state of every base table (the current version). This maps directly onto pg_ivm's model, where triggers fire after the base table is modified, transition tables provide the deltas, and the base table is already in its post-update state.

The classical approach is simpler, produces more readable SQL, and the generated queries are easier to debug. Each delta rule has a clear operational meaning: "join the new rows against the current state of the other table."

## The Delta Rules

### Notation

For a base table R changing between version `v_old` and `v_new`:

- `R` — the table at version `v_new` (post-update state, i.e., the current state)
- `ΔR` — rows inserted: output of `ducklake_table_changes(R, v_old, v_new)` where change type is INSERT
- `∇R` — rows deleted: output of `ducklake_table_changes(R, v_old, v_new)` where change type is DELETE

Updates are modeled as a delete of the old row followed by an insert of the new row. `ducklake_table_changes()` handles this — an updated row appears as both a deletion (old values) and an insertion (new values).

For the materialized view `V`, we need to compute:
- `ΔV` — rows to INSERT into the materialized view
- `∇V` — rows to DELETE from the materialized view

### Selection: V = σ_p(R)

```
ΔV = σ_p(ΔR)
∇V = σ_p(∇R)
```

Apply the same predicate to the inserted and deleted rows. Trivial — the filter doesn't interact with the delta at all.

**Generated SQL:**
```sql
-- Rows to insert into MV
INSERT INTO mv SELECT * FROM ducklake_table_changes('R', v_old, v_new)
WHERE change_type = 'INSERT' AND <predicate>;

-- Rows to delete from MV
DELETE FROM mv WHERE EXISTS (
  SELECT 1 FROM ducklake_table_changes('R', v_old, v_new) d
  WHERE d.change_type = 'DELETE' AND <predicate> AND mv.pk = d.pk
);
```

### Projection: V = π_cols(R)

```
ΔV = π_cols(ΔR)
∇V = π_cols(∇R)
```

Same idea — project the delta columns. One subtlety: if the projection introduces duplicates that weren't in the original, you need multiplicity tracking (the `__ivm_count__` column from pg_ivm). If the projection preserves a key, no issue.

### Inner Join: V = R ⋈ S

This is the most important rule. When R changes:

```
ΔV = ΔR ⋈ S
∇V = ∇R ⋈ S
```

When S changes:

```
ΔV = R ⋈ ΔS
∇V = R ⋈ ∇S
```

When both change simultaneously:

```
ΔV = (ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS)
∇V = (∇R ⋈ S) ∪ (R ⋈ ∇S) - (∇R ⋈ ∇S)
```

**Critical detail: this is a two-term formula, not three.** Because `R` and `S` refer to the post-update (current) state, the new rows in R are already included in R. So `R ⋈ ΔS` already includes `ΔR ⋈ ΔS`. We must *subtract* it to avoid double-counting. This is the classic bag algebra result.

Contrast with DBSP's Z-set formula, which uses pre-update states and *adds* the third term:
```
Δ(R ⋈ S) = (ΔR ⋈ S_old) + (R_old ⋈ ΔS) + (ΔR ⋈ ΔS)
```

Both are correct. Ours is a better fit because we naturally have the post-update state.

**Optimization: when only one table changes.** In practice, most maintenance cycles involve changes to a single base table. When only R changes and S is unchanged:

```
ΔV = ΔR ⋈ S
∇V = ∇R ⋈ S
```

No correction term needed. This is the common fast path and it's a single join.

**Generated SQL (single table change):**
```sql
-- Rows to insert into MV
INSERT INTO mv
SELECT R_new.*, S.*
FROM ducklake_table_changes('R', v_old, v_new) AS R_new
JOIN S ON R_new.join_key = S.join_key
WHERE R_new.change_type = 'INSERT';

-- Rows to delete from MV
DELETE FROM mv WHERE EXISTS (
  SELECT 1
  FROM ducklake_table_changes('R', v_old, v_new) AS R_del
  JOIN S ON R_del.join_key = S.join_key
  WHERE R_del.change_type = 'DELETE'
  AND mv.pk = <derived_pk>
);
```

**Generated SQL (both tables change):**
```sql
-- Compute insertions
INSERT INTO mv
SELECT * FROM (
  -- New R rows joined against current S
  SELECT ... FROM delta_R_inserts JOIN S ON ...
  UNION ALL
  -- Current R joined against new S rows
  SELECT ... FROM R JOIN delta_S_inserts ON ...
) combined
-- Subtract the double-counted intersection
WHERE NOT EXISTS (
  SELECT 1 FROM delta_R_inserts dr JOIN delta_S_inserts ds
  ON dr.join_key = ds.join_key
  WHERE combined.pk = <derived_pk_from_dr_ds>
);

-- (Analogous for deletions)
```

### Multi-Way Joins: V = R ⋈ S ⋈ T

The two-way rule extends. When only R changes:

```
ΔV = ΔR ⋈ S ⋈ T
∇V = ∇R ⋈ S ⋈ T
```

The unchanged tables act as filters. The delta only flows through the changed table. When multiple tables change simultaneously, you get inclusion-exclusion terms, but the single-table-change case dominates in practice.

The compiler should decompose multi-way joins into a tree of binary joins (which sqlglot's optimizer already does) and apply the binary join rule at each node.

### Aggregation: V = γ_{group, agg}(R)

Aggregation is where the real complexity lives. The approach depends on the aggregate function.

#### Distributive Aggregates: SUM, COUNT

For `V = SELECT group_key, SUM(val) AS s, COUNT(*) AS c FROM R GROUP BY group_key`:

**On insertion of rows ΔR:**
```sql
-- For existing groups: update in place
UPDATE mv SET
  s = s + delta.s,
  c = c + delta.c
FROM (
  SELECT group_key, SUM(val) AS s, COUNT(*) AS c
  FROM ducklake_table_changes('R', v_old, v_new)
  WHERE change_type = 'INSERT'
  GROUP BY group_key
) delta
WHERE mv.group_key = delta.group_key;

-- For new groups: insert
INSERT INTO mv (group_key, s, c)
SELECT group_key, SUM(val), COUNT(*)
FROM ducklake_table_changes('R', v_old, v_new)
WHERE change_type = 'INSERT'
GROUP BY group_key
HAVING group_key NOT IN (SELECT group_key FROM mv);
```

**On deletion of rows ∇R:**
```sql
-- Subtract from existing groups
UPDATE mv SET
  s = s - delta.s,
  c = c - delta.c
FROM (
  SELECT group_key, SUM(val) AS s, COUNT(*) AS c
  FROM ducklake_table_changes('R', v_old, v_new)
  WHERE change_type = 'DELETE'
  GROUP BY group_key
) delta
WHERE mv.group_key = delta.group_key;

-- Remove empty groups
DELETE FROM mv WHERE c = 0;
```

**This is why the materialized view must always store COUNT(*) as a hidden column**, even if the original query doesn't request it. Without it, you can't detect when a group becomes empty.

#### AVG

AVG is not directly maintainable, but it decomposes into SUM / COUNT. The materialized view stores SUM and COUNT as hidden columns, and AVG is derived:

```sql
-- MV schema: (group_key, _ivm_sum, _ivm_count, avg_val)
-- After updating _ivm_sum and _ivm_count as above:
UPDATE mv SET avg_val = _ivm_sum::FLOAT / _ivm_count
WHERE _ivm_count > 0;
```

#### MIN / MAX

These are the hard case. MIN and MAX are not distributive — you can't compute the new MIN from the old MIN and the delta alone.

**On insertion:** easy. The new MIN is `MIN(current_min, MIN(ΔR))`.

**On deletion:** hard. If the deleted row's value equals the current MIN, you don't know what the new MIN is without rescanning the group. This requires a fallback:

```sql
-- If the current min might have been deleted, rescan
UPDATE mv SET min_val = (
  SELECT MIN(val) FROM R WHERE R.group_key = mv.group_key
)
WHERE mv.group_key IN (
  SELECT group_key
  FROM ducklake_table_changes('R', v_old, v_new)
  WHERE change_type = 'DELETE'
  AND val = mv.min_val  -- only rescan if the deleted value was the current min
);
```

This rescan touches the full group in the base table. For large groups, this is expensive. Optimizations exist (maintaining a sorted structure, counting occurrences of the min value) but add complexity. For an MVP, the rescan fallback is correct and sufficient.

### DISTINCT: V = δ(R)

DISTINCT is maintained via a hidden multiplicity counter `__ivm_count__` on the materialized view. Each row in the MV tracks how many times it appears in the underlying query result.

**On insertion:**
```sql
-- Increment count for existing rows
UPDATE mv SET __ivm_count__ = __ivm_count__ + delta.cnt
FROM (
  SELECT <all_cols>, COUNT(*) AS cnt
  FROM ducklake_table_changes('R', v_old, v_new)
  WHERE change_type = 'INSERT'
  GROUP BY <all_cols>
) delta
WHERE mv.<all_cols> = delta.<all_cols>;

-- Insert new rows (count was 0, now positive)
INSERT INTO mv (<all_cols>, __ivm_count__)
SELECT <all_cols>, COUNT(*)
FROM ducklake_table_changes('R', v_old, v_new)
WHERE change_type = 'INSERT'
GROUP BY <all_cols>
HAVING <all_cols> NOT IN (SELECT <all_cols> FROM mv);
```

**On deletion:**
```sql
-- Decrement count
UPDATE mv SET __ivm_count__ = __ivm_count__ - delta.cnt
FROM (
  SELECT <all_cols>, COUNT(*) AS cnt
  FROM ducklake_table_changes('R', v_old, v_new)
  WHERE change_type = 'DELETE'
  GROUP BY <all_cols>
) delta
WHERE mv.<all_cols> = delta.<all_cols>;

-- Remove rows whose count dropped to 0
DELETE FROM mv WHERE __ivm_count__ = 0;
```

### UNION ALL

Trivial. Delta of a UNION ALL is the UNION ALL of deltas.

### UNION (with dedup)

Treated as UNION ALL followed by DISTINCT. Apply the DISTINCT maintenance rules.

### EXCEPT ALL

```
V = R EXCEPT ALL S
```

When R gains rows (ΔR), those rows are added to V. When S gains rows (ΔS), those rows are removed from V. Deletions are the reverse. This is straightforward bag subtraction maintenance, handled the same way as a DISTINCT counter — maintain multiplicity as `count_in_R - count_in_S`.

### LEFT / RIGHT / FULL OUTER JOIN

Outer joins are more complex because they produce NULL-padded rows when there's no match. The delta rules must account for:

- An insertion into R might cause a previously NULL-padded row in V (from S having no match) to be replaced by a real joined row
- A deletion from R might cause a row in V to revert to a NULL-padded row

This requires checking whether the join partner exists. For LEFT JOIN `R LEFT JOIN S`:

**When S gains a row that matches an existing R row:**
```sql
-- Delete the old NULL-padded row
DELETE FROM mv WHERE mv.r_key = <key> AND mv.s_cols IS NULL;
-- Insert the real joined row
INSERT INTO mv SELECT R.*, S_new.* FROM R JOIN delta_S_inserts ON ...;
```

**When S loses a row that was the only match for an R row:**
```sql
-- Delete the real joined row
DELETE FROM mv WHERE mv.r_key = <key> AND mv.s_key = <deleted_s_key>;
-- Re-insert the NULL-padded row if no other S match exists
INSERT INTO mv SELECT R.*, NULL, NULL, ...
FROM R WHERE R.r_key = <key>
AND NOT EXISTS (SELECT 1 FROM S WHERE S.join_key = R.r_key);
```

Outer join maintenance is the most intricate part of the system. For an MVP, it's reasonable to support only inner joins and add outer join support later.

## Version Tracking

The materialized view needs to know which version of each base table it was last maintained against. This requires a metadata table:

```sql
CREATE TABLE _ivm_metadata (
  mv_name TEXT,
  base_table TEXT,
  last_maintained_version BIGINT
);
```

A maintenance cycle:
1. Read `last_maintained_version` for each base table
2. Get current version of each base table from DuckLake
3. For each base table that changed, compute `ducklake_table_changes(table, old_version, new_version)`
4. Apply the delta propagation SQL
5. Update `last_maintained_version`

## Algorithmic Complexity Summary

| Operation | Maintenance Cost (per delta) | Notes |
|-----------|------------------------------|-------|
| Selection | O(\|ΔR\|) | Filter the delta |
| Projection | O(\|ΔR\|) | Project the delta |
| Inner Join (one table changes) | O(\|ΔR\| × index_lookup) | Single join against unchanged table |
| Inner Join (both change) | O(\|ΔR\| × \|S\| + \|R\| × \|ΔS\|) | Two joins plus correction |
| SUM / COUNT | O(\|ΔR\| + affected_groups) | Aggregate delta, update groups |
| AVG | O(\|ΔR\| + affected_groups) | Via SUM and COUNT |
| MIN / MAX (insert) | O(\|ΔR\|) | Compare against current extremum |
| MIN / MAX (delete of extremum) | O(\|group\|) | Must rescan the full group |
| DISTINCT | O(\|ΔR\| + affected_rows) | Update counters |

The key insight: all of these are proportional to the size of the delta, not the size of the base tables — except MIN/MAX deletion and the correction term for simultaneous multi-table changes. This is the fundamental win of IVM over recomputation.

## Performance: Single-Scan Optimization

The one place where separate INSERT/DELETE passes could cost performance is scanning `ducklake_table_changes()` twice. We avoid this by using a shared CTE:

```sql
WITH _changes AS (
    SELECT * FROM ducklake_table_changes('R', v_old, v_new)
),
_to_insert AS (
    SELECT ... FROM _changes WHERE change_type IN ('insert', 'update_postimage') ...
),
_to_delete AS (
    SELECT ... FROM _changes WHERE change_type IN ('delete', 'update_preimage') ...
)
-- Both _to_insert and _to_delete share a single scan
```

This gives classical semantics (separate insert/delete reasoning, readable SQL, two-term join formula) with single-scan performance. The CTE is referenced by both the INSERT and DELETE logic without re-scanning the change feed.

For aggregates, the same pattern applies — a single `_changes` CTE feeds both the insert-aggregate and delete-aggregate computations, which are then combined via FULL OUTER JOIN into net deltas per group.

## What the Compiler Emits

For a given view definition, the compiler produces:

1. **DDL** — CREATE TABLE for the materialized view, including hidden columns (`__ivm_count__`, auxiliary SUM/COUNT for AVG, etc.)
2. **Initialization query** — A SELECT that populates the MV from scratch (for first load or full refresh)
3. **Per-base-table maintenance SQL** — For each base table, a set of INSERT/UPDATE/DELETE statements against the MV, parameterized by version numbers, using `ducklake_table_changes()` as the delta source
4. **Metadata management SQL** — Version tracking updates

The compiler does NOT emit triggers, scheduling logic, or orchestration code. It's the caller's responsibility to invoke the maintenance SQL at the right time. This keeps the compiler simple and the output portable within the DuckDB/DuckLake ecosystem.
