# Incremental View Maintenance: Conversation Summary & Implementation Plan

## Conversation Arc

We started from RisingWave's incremental computation framework and progressively narrowed toward a specific gap in the ecosystem: a **standalone, SQL-to-SQL incremental view maintenance (IVM) compiler** — something that takes a SQL view definition and emits the SQL statements needed to propagate deltas from base tables into the materialized view, without requiring a separate execution engine.

### Key findings along the way

- **RisingWave** uses an actor-based dataflow engine inspired by Differential Dataflow / Timely Dataflow / DBSP for incremental computation. It exposes Iceberg tables over the Postgres wire protocol, but it is not the only system doing so (Snowflake Labs' **pg_lake** / Crunchy Data Warehouse does as well).

- **Feldera** is the most mature standalone IVM engine. It has a SQL-to-DBSP compiler that uses Apache Calcite for parsing and emits Rust DBSP circuits. Powerful, but the output is Rust code, not SQL.

- **OpenIVM** (SIGMOD 2024 demo) is the closest to the goal: a SQL-to-SQL compiler following DBSP principles, implemented as a DuckDB extension in C++. However, it appears to be an unmaintained research prototype — the GitHub repo may be private or removed, it has limited SQL coverage, and it's tightly coupled to DuckDB's C++ internals.

- **pg_ivm** is well-maintained (supports PG 13–18) but is Postgres-specific — it uses AFTER triggers and transition tables, not a portable SQL-to-SQL compilation approach.

- **cwida/ivm-extension** is another DuckDB IVM extension from CWI, but it's limited (no joins, no HAVING, no nested subqueries) and not actively maintained.

## Your Goal

A **well-maintained, standalone, SQL-to-SQL IVM compiler** that:

- Takes a SQL view definition as input
- Outputs standard SQL statements that propagate deltas (inserts/deletes) from base tables into the materialized view
- Stays entirely within the SQL world — no separate runtime, no Rust circuits, no C++ engine
- Is portable across databases (not tied to Postgres triggers or DuckDB internals)

This tool does not exist in a production-quality form today. The gap is real.

## The DBSP Rewrite Rules (What OpenIVM Implements)

The core rules from DBSP that power OpenIVM are surprisingly compact:

### Data Model: Z-Sets

Everything is represented as **Z-sets** — multisets where each row has an integer weight. Insertions have weight `+1`, deletions have weight `-1`. This unifies inserts and deletes into a single algebraic framework. OpenIVM simplified this to a boolean multiplicity column.

### Incremental Rewrite Rules

The rules are applied **bottom-up** on the relational algebra tree:

| Operator | Incremental Form | Complexity |
|----------|-----------------|------------|
| **Selection** (σ) | `Δ(σ_p(R)) = σ_p(ΔR)` — apply same predicate to delta | Trivial (identity) |
| **Projection** (π) | `Δ(π(R)) = π(ΔR)` — project the delta | Trivial (identity) |
| **Map / Scalar** | Apply same expression to delta | Trivial (identity) |
| **Join** (R ⋈ S) | `(ΔR ⋈ S) ∪ (R ⋈ ΔS) ∪ (ΔR ⋈ ΔS)` | Moderate — 3 joins + union |
| **Aggregation** (SUM, COUNT) | Add/subtract delta contribution per group | Moderate — needs auxiliary state |
| **Aggregation** (AVG) | Maintain via SUM and COUNT | Moderate |
| **Aggregation** (MIN, MAX) | Fallback to rescan of group on delete of current min/max | Hard — needs full group access |
| **DISTINCT** | Maintain multiplicity counter; emit row when count crosses 0 | Moderate |
| **UNION** | Delta of union is union of deltas | Trivial |
| **EXCEPT / INTERSECT** | Z-set subtraction / min of weights | Moderate |

### What Makes This Tractable

The key property from DBSP: **the incremental form of a composition is the composition of incremental forms**. You don't need special rules for nested queries — you just walk the tree and rewrite each node independently.

## Implementation Plan: sqlglot-Based Python IVM Compiler

### Why sqlglot

- Pure Python, no C++ dependencies
- Excellent SQL parser with dialect support (Postgres, DuckDB, MySQL, Snowflake, etc.)
- Produces a clean AST (expressions, not strings)
- Has a built-in SQL optimizer and transpiler
- Can parse in one dialect and emit in another — enabling cross-database IVM
- Well-maintained (active development, ~6k GitHub stars)

### Architecture

```
SQL View Definition
        │
        ▼
┌─────────────────┐
│  sqlglot.parse() │  ← Parse SQL into AST
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Logical Plan Build  │  ← Convert AST to relational algebra tree
└────────┬────────────┘   (sqlglot.optimizer.optimize already does much of this)
         │
         ▼
┌─────────────────────┐
│  IVM Rewriter        │  ← Bottom-up tree walk applying DBSP rules
│  (the core logic)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────┐
│  SQL Code Generator          │  ← Emit:
│  (sqlglot.Generator)         │     1. DDL for delta tables
└────────┬────────────────────┘     2. DDL for materialized view table
         │                           3. DML to propagate deltas
         ▼                           4. (Optional) DML to initialize from full data
   Output SQL Statements
```

### Components to Build

#### 1. Delta Table Schema Generator (~100 lines)
For each base table referenced in the view, generate a delta table schema that mirrors the base table plus a `_multiplicity` column (integer or boolean).

```sql
-- For base table: orders(id INT, customer_id INT, amount DECIMAL)
CREATE TABLE delta_orders (LIKE orders);
ALTER TABLE delta_orders ADD COLUMN _ivm_weight INTEGER DEFAULT 1;
```

#### 2. Materialized View Table Generator (~50 lines)
Generate the CREATE TABLE for the materialized view output, including any auxiliary columns needed for maintenance (e.g., `_ivm_count` for DISTINCT, hidden SUM/COUNT for AVG).

#### 3. IVM Rewriter — The Core (~500–800 lines)
A recursive AST visitor that walks the sqlglot expression tree bottom-up and rewrites each node:

- **Select/Where/Project nodes:** Replace base table references with delta table references. The predicate/projection stays the same.

- **Join nodes:** The most complex rewrite. For `R JOIN S ON condition`:
  ```sql
  -- Delta propagation when R changes:
  (SELECT ... FROM delta_R JOIN S ON condition)
  UNION ALL
  (SELECT ... FROM R JOIN delta_S ON condition)
  UNION ALL
  (SELECT ... FROM delta_R JOIN delta_S ON condition)
  ```
  The third term can optionally be folded into the first two by using `R ∪ ΔR` in one of the branches (this is an optimization choice).

- **GROUP BY + Aggregate nodes:** Generate update statements that:
  - For SUM: `UPDATE mv SET sum_col = sum_col + delta_sum WHERE group_key = ...`
  - For COUNT: `UPDATE mv SET count_col = count_col + delta_count WHERE group_key = ...`
  - For AVG: Update via SUM and COUNT
  - For MIN/MAX: Conditional — if the deleted value equals current min/max, rescan the group
  - Handle group creation (INSERT when a new group appears) and group deletion (DELETE when count reaches 0)

- **DISTINCT nodes:** Maintain a `_ivm_count` column. Increment/decrement on delta. Emit to output only on transitions across 0.

#### 4. SQL Serializer (~100 lines, mostly free via sqlglot)
Use `sqlglot.Generator` to serialize the rewritten AST back to SQL in the target dialect. This is largely free — sqlglot handles dialect-specific syntax.

#### 5. Orchestrator (~200 lines)
Ties everything together:
- Accepts a view definition string and target dialect
- Calls the parser, rewriter, and generator
- Emits a complete set of SQL statements: DDL + DML for delta propagation
- Optionally emits a "refresh from scratch" query for initial population

### Estimated Complexity

| Component | Lines of Python | Difficulty |
|-----------|----------------|------------|
| Delta table schema gen | ~100 | Easy |
| MV table gen | ~50 | Easy |
| IVM rewriter (SPJ) | ~300 | Moderate |
| IVM rewriter (aggregates) | ~300 | Moderate–Hard |
| IVM rewriter (DISTINCT) | ~100 | Moderate |
| SQL serializer | ~100 | Easy (sqlglot does the work) |
| Orchestrator + API | ~200 | Easy |
| Tests | ~500+ | Necessary |
| **Total** | **~1,500–2,000** | |

### What's Hard

- **Correct handling of NULL semantics** in aggregates and joins
- **Multi-table deltas in the same transaction** — when both R and S change simultaneously, the three-way join decomposition must be correct
- **Correlated subqueries** — require flattening or special handling before the IVM rewrite
- **Window functions** — not naturally incrementalizable; may need to punt on these initially
- **MIN/MAX** — the rescan fallback is correct but expensive; optimizations exist (e.g., maintaining a sorted index or using Reactive Aggregator patterns) but add complexity

### Suggested MVP Scope

Start with **Selection-Projection-Join + distributive aggregates (SUM, COUNT, AVG) + DISTINCT**. This covers the same fragment that pg_ivm supports and handles the vast majority of real-world materialized view definitions. Add MIN/MAX with rescan fallback. Defer window functions and recursive queries.

### Key Design Decisions

1. **Integer weights vs. boolean multiplicity:** Integer weights (Z-sets) are more general and compose better. Boolean is simpler but breaks on queries that naturally produce duplicates. Recommend integer weights.

2. **Eager vs. lazy maintenance:** The generated SQL can be wrapped in triggers (eager, like pg_ivm) or called on-demand (lazy/deferred). The compiler should be agnostic — it just emits the DML.

3. **Cross-database support:** sqlglot's dialect system means you can parse Postgres SQL and emit DuckDB SQL (or vice versa). This enables the cross-system HTAP scenario that OpenIVM demonstrated.

4. **State management:** The IVM rewriter needs to know the current state of base tables (for the `R ⋈ ΔS` term in join maintenance). The generated SQL assumes the base tables contain the pre-delta state. The caller is responsible for applying deltas to base tables after propagation, or the generated SQL should reference the pre-update state (e.g., via transition tables in Postgres, or explicit snapshot semantics).
