"""Strategy: set operations (UNION / EXCEPT / INTERSECT)."""

from hypothesis import strategies as st

from tests.strategies.primitives import (
    Column,
    Delta,
    Row,
    Scenario,
    Table,
    safe_col_names,
    safe_table_names,
    value_for_type,
)


@st.composite
def set_operation(draw):
    """Scenario: two tables with same column schema, combined via a set operation.

    Generates UNION ALL, UNION, EXCEPT ALL, or INTERSECT ALL between
    SELECT ... FROM t1 and SELECT ... FROM t2.
    """
    # Build two tables with same column types but different names
    n_cols = draw(st.integers(min_value=1, max_value=3))
    used_names: set[str] = set()
    columns_spec: list[tuple[str, str]] = []
    for i in range(n_cols):
        name = draw(safe_col_names.filter(lambda n, u=used_names: n not in u))
        used_names.add(name)
        # Use INTEGER for first col, small range to get duplicates
        dtype = "INTEGER" if i == 0 else draw(st.sampled_from(["INTEGER", "VARCHAR"]))
        columns_spec.append((name, dtype))

    left_name = draw(safe_table_names.filter(lambda n: n not in used_names))
    used_names.add(left_name)
    right_name = draw(safe_table_names.filter(lambda n: n not in used_names))
    used_names.add(right_name)

    left_cols = [Column(name=n, dtype=d) for n, d in columns_spec]
    right_cols = [Column(name=n, dtype=d) for n, d in columns_spec]
    left_table = Table(name=left_name, columns=left_cols)
    right_table = Table(name=right_name, columns=right_cols)

    # Small value ranges to ensure overlap between tables
    def small_val(dtype):
        if dtype == "INTEGER":
            return st.integers(min_value=1, max_value=5)
        elif dtype == "VARCHAR":
            return st.sampled_from(["a", "b", "c", "d"])
        return value_for_type(dtype)

    def make_row(table):
        vals = {}
        for col in table.columns:
            vals[col.name] = draw(small_val(col.dtype))
        return Row(vals)

    left_rows = [make_row(left_table) for _ in range(draw(st.integers(2, 8)))]
    right_rows = [make_row(right_table) for _ in range(draw(st.integers(2, 8)))]

    # Choose set operation
    op = draw(st.sampled_from(["UNION ALL", "UNION", "EXCEPT ALL", "INTERSECT ALL"]))
    col_names_str = ", ".join(c.name for c in left_cols)
    view_sql = (
        f"SELECT {col_names_str} FROM {left_name} {op} SELECT {col_names_str} FROM {right_name}"
    )

    # Generate deltas for one or both tables
    deltas: list[Delta] = []
    delta_target = draw(st.sampled_from(["left", "right", "both"]))

    for tname, rows, table in [
        (left_name, left_rows, left_table),
        (right_name, right_rows, right_table),
    ]:
        if delta_target == "left" and tname != left_name:
            continue
        if delta_target == "right" and tname != right_name:
            continue

        ins = [make_row(table) for _ in range(draw(st.integers(0, 3)))]
        n_del = draw(st.integers(0, min(2, len(rows))))
        del_idx = (
            draw(
                st.lists(
                    st.integers(0, len(rows) - 1),
                    min_size=n_del,
                    max_size=n_del,
                    unique=True,
                )
            )
            if n_del > 0
            else []
        )
        deltas.append(Delta(tname, inserts=ins, deletes=[rows[i] for i in del_idx]))

    return Scenario(
        tables=[left_table, right_table],
        initial_data={left_name: left_rows, right_name: right_rows},
        view_sql=view_sql,
        deltas=deltas,
    )
