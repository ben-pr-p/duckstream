# Evolution Plan

How the compiler and its test suite grow together, stage by stage.

## Testing Philosophy

**Oracle testing via DuckLake on DuckDB.** Every test follows the same pattern:

1. Set up DuckLake tables with initial data (records a snapshot)
2. Materialize the view (full computation)
3. Apply changes (inserts/deletes) to the DuckLake tables (records a new snapshot)
4. Run the compiler's maintenance SQL, which uses `ducklake_table_changes()` to read deltas from the snapshot range
5. Recompute the view from scratch on the updated tables
6. Assert: maintained MV == recomputed view

This is the only invariant we need. If it holds across random inputs, the compiler is correct.

**DuckLake isolation via random temp directories.** DuckLake requires a metadata catalog file and a data directory on disk — it cannot run purely in-memory. Each test must create its own randomly-named temp directory (e.g. via `tempfile.mkdtemp()`) for the metadata DB and data path. This ensures full isolation between tests, even under parallel execution with `pytest-xdist`. The fixture cleans up the directory after the test.

**Hypothesis generates the inputs.** For each SQL fragment class, we write a Hypothesis strategy that produces random `Scenario` objects (schemas, data, view SQL, deltas). Hypothesis handles shrinking — when a failure is found, it minimizes the example to the smallest reproducing case.

**Hand-written smoke tests first.** Each stage starts with a few deterministic scenarios that exercise the new feature in isolation. These are fast to debug. The property tests then hammer the feature with random inputs.

**DuckDB as the oracle.** The oracle (recompute from scratch) is always DuckDB itself, so we're testing our maintenance SQL against DuckDB's own query engine. Parallel execution works via `pytest-xdist` since each test has its own DuckLake catalog in its own temp directory.

## Code Quality

**Linting and formatting** with [ruff](https://docs.astral.sh/ruff/) (Astral). All code must pass `uv run ruff check src/ tests/` and be formatted with `uv run ruff format src/ tests/`.

**Type checking** with [ty](https://docs.astral.sh/ty/) (Astral). Library code must pass `uv run ty check src/`.

**Quality gate for every stage:** Before a stage is considered complete, all of the following must pass:
1. `uv run ruff check src/ tests/` — no lint errors
2. `uv run ruff format --check src/ tests/` — consistent formatting
3. `uv run ty check src/` — no type errors in library code
4. `uv run pytest` — all tests pass

---

## Stage 1: Single-Table SELECT / PROJECT / WHERE

**Compiler work:**
- Parse view SQL with sqlglot
- Identify base tables
- Generate `CREATE TABLE mv AS <view_sql>` for initial materialization
- For maintenance: separate INSERT and DELETE passes against the MV, reading ΔR (inserts) and ∇R (deletes) from `ducklake_table_changes()`

**Bag algebra rule:** Selection and projection are identity transforms on both delta streams — `ΔV = σ_p(ΔR)`, `∇V = σ_p(∇R)`. Same for projection.

**Test strategies (already written):**
- `single_table_select()` — random table (2-5 cols), random data, random column subset projection, optional WHERE on an integer column, random inserts + deletes

**Smoke tests (already written):**
- `test_simple_select_all` — SELECT all columns, one insert, one delete
- `test_select_with_filter` — WHERE filter, mixed inserts (some pass filter, some don't), one delete

**Done when:** All smoke tests and 50 Hypothesis examples pass. `ruff check`, `ruff format --check`, and `ty check` all clean.

---

## Stage 2: Single-Table GROUP BY with Distributive Aggregates (COUNT, SUM)

**Compiler work:**
- Detect GROUP BY + aggregate expressions in the AST
- Generate MV table with an auxiliary `_ivm_count` column (tracks row count per group)
- Maintenance SQL:
  - For existing groups: `UPDATE mv SET agg_col = agg_col + delta_agg, _ivm_count = _ivm_count + delta_count WHERE group_key = ...`
  - For new groups: `INSERT INTO mv ... SELECT ... FROM delta_table GROUP BY ... HAVING ...` (only groups not already in MV)
  - For emptied groups: `DELETE FROM mv WHERE _ivm_count = 0`

**Bag algebra rule:** Separate passes for ΔR and ∇R — for SUM, add `SUM(val)` from inserts and subtract `SUM(val)` from deletes. For COUNT, add `COUNT(*)` from inserts and subtract `COUNT(*)` from deletes.

**Test strategies (already written):**
- `single_table_aggregate()` — random table, random grouping column (VARCHAR), random numeric agg column (INTEGER), SUM or COUNT or AVG, random inserts + deletes

**Smoke tests (already written):**
- `test_count_aggregate` — GROUP BY with COUNT(*), insert new groups + delete existing
- `test_sum_aggregate` — GROUP BY with SUM, insert and delete within same group

**Done when:** All smoke + Hypothesis tests pass, including edge cases like groups appearing and disappearing. Quality gates pass.

---

## Stage 3: AVG (Derived from SUM + COUNT)

**Compiler work:**
- When AVG is requested, maintain hidden `_ivm_sum` and `_ivm_count` columns in the MV
- MV exposes `_ivm_sum / _ivm_count` as the AVG column
- Maintenance is the same as Stage 2 — update sum and count, recompute the ratio

**New tests:**
- Add AVG to the `single_table_aggregate` strategy (already included)
- Smoke test for AVG specifically, including groups where count drops to 0

**Done when:** AVG matches recomputed result across random inputs. Float comparison needs an epsilon tolerance. Quality gates pass.

---

## Stage 4: Two-Table JOIN (Inner Join)

**Compiler work:**
- Detect JOIN nodes in the AST
- Read deltas for each base table via `ducklake_table_changes()`
- Maintenance SQL implements the three-way decomposition:
  ```
  Δ(R ⋈ S) = (ΔR ⋈ S) ∪ (R ⋈ ΔS) ∪ (ΔR ⋈ ΔS)
  ```
- Each term is a SELECT with the same join condition, one side reading from `ducklake_table_changes()`
- Results are UNION ALL'd and applied to the MV (inserts for +1 weight, deletes for -1)

**New test strategy:**
- `two_table_join()` — two random tables sharing a join column (same type), random data in both, random deltas to one or both tables, view is `SELECT ... FROM R JOIN S ON R.key = S.key`

**Smoke tests:**
- Inner join, insert into left table only
- Inner join, insert into right table only
- Inner join, insert into both tables simultaneously
- Inner join, delete a row that participates in a join match

**Done when:** All join property tests pass. Particular attention to: simultaneous deltas to both tables, dangling deltas (insert into R but no match in S), and the ΔR ⋈ ΔS cross-delta term. Quality gates pass.

**Bag algebra rule:** When one table changes: `ΔV = ΔR ⋈ S`, `∇V = ∇R ⋈ S`. When both change: `ΔV = (ΔR ⋈ S) ∪ (R ⋈ ΔS) - (ΔR ⋈ ΔS)` — subtract the cross-delta to avoid double-counting (R and S are post-update/current state). See `bag-algebra-vs-z-sets.md` for rationale, PLAN.md Step 5 for SQL patterns.

---

## Stage 5: JOIN + Aggregates (Composed)

**Compiler work:**
- This should fall out naturally from the composition property: the incremental form of a join feeding into an aggregate is just the join delta piped through the aggregate maintenance.
- May require the compiler to walk the AST bottom-up, rewriting join first, then aggregate on top.

**New test strategy:**
- `join_then_aggregate()` — two-table join with GROUP BY + SUM/COUNT/AVG on the result

**Smoke tests:**
- `SELECT s.region, SUM(o.amount) FROM orders o JOIN stores s ON o.store_id = s.id GROUP BY s.region` — insert an order, insert a store, delete an order

**Done when:** Composed join + aggregate matches recomputed result. This is the real test of the bottom-up rewriting approach. Quality gates pass.

**Implementation:** The join delta (weighted rows from the three-way decomposition) feeds directly into the aggregate maintenance from Stage 2. The compiler composes these by using the join delta as the source for the weighted aggregation CTE. See PLAN.md Step 6.

---

## Stage 6: DISTINCT

**Compiler work:**
- Maintain a hidden `_ivm_multiplicity` column per row in the MV
- On delta: increment/decrement multiplicity
- Row appears in output when multiplicity transitions from 0 to positive
- Row disappears when multiplicity drops to 0

**New test strategy:**
- `single_table_distinct()` — SELECT DISTINCT on a subset of columns, with data that naturally produces duplicates

**Done when:** DISTINCT works standalone and composed with WHERE and JOIN. Quality gates pass.

---

## Stage 7: MIN / MAX (Rescan Fallback)

**Compiler work:**
- For inserts: if the new value is less/greater than the current MIN/MAX, update
- For deletes: if the deleted value equals the current MIN/MAX, rescan the group to find the new MIN/MAX

**New test strategy:**
- Extend `single_table_aggregate` to include MIN/MAX

**Smoke tests:**
- Delete the current MIN value, verify rescan picks up the next one
- Insert a new MIN value below the current

**Done when:** MIN/MAX correct under all delta patterns. This is the first case where maintenance can be expensive (group rescan). Quality gates pass.

---

## Stage 8: Multi-Way Joins (3+ Tables)

**Compiler work:**
- Generalize the two-table join decomposition to N tables
- For N tables, a delta to table i means joining ΔTi with all other tables (using their current state), plus cross-delta terms

**New test strategy:**
- `three_table_join()` — three tables with a chain of join conditions

**Done when:** Three-table joins produce correct results. The combinatorial explosion of delta terms is the main challenge. Quality gates pass.

---

## Stage 9: LEFT / RIGHT / FULL OUTER Joins

**Compiler work:**
- Outer joins require tracking NULL-extended rows
- A delta to the preserved side may create or destroy NULL extensions
- A delta to the nullable side may replace NULL extensions with real matches or vice versa

**New test strategies:**
- `left_join()`, `right_join()`, `full_outer_join()` — same as inner join strategies but with outer join semantics

**Done when:** Outer joins correctly handle NULL extension creation/destruction. Quality gates pass.

---

## Stage 10: UNION / EXCEPT / INTERSECT

**Compiler work:**
- UNION ALL: delta of union is union of deltas (trivial)
- UNION (DISTINCT): combine with DISTINCT maintenance
- EXCEPT / INTERSECT: bag subtraction / multiplicity tracking

**Done when:** Set operations work standalone and composed. Quality gates pass.

---

## Future / Deferred

These are explicitly out of scope for the MVP but noted for later:

- **Window functions** — not naturally incrementalizable; would need to maintain sorted buffers
- **Correlated subqueries** — require decorrelation (flattening to joins) before IVM rewrite
- **Recursive CTEs** — require fixed-point iteration; a different algorithm entirely
- **Cross-dialect emission** — the compiler already uses sqlglot's dialect system, but testing against Postgres/MySQL/etc. requires those engines in CI
- **HAVING** — should compose from aggregate + filter, but needs explicit testing
- **ORDER BY / LIMIT** — fundamentally incompatible with incremental maintenance in the general case; may support for specific patterns

---

## Running Tests

```bash
# All tests
uv run pytest

# Just smoke tests (fast)
uv run pytest tests/test_ivm.py::TestSmoke

# Just property tests for a specific stage
uv run pytest tests/test_ivm.py::test_select_project_filter -x
uv run pytest tests/test_ivm.py::test_single_table_aggregate -x

# More Hypothesis examples (slow, thorough)
uv run pytest tests/test_ivm.py --hypothesis-seed=0 -x

# Parallel execution
uv run pytest -n auto
```
