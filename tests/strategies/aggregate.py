"""Strategy: single-table GROUP BY with aggregates."""

from hypothesis import strategies as st

from tests.strategies.primitives import (
    Column,
    Delta,
    Scenario,
    Table,
    col_type,
    rows_for_table,
    safe_col_names,
    safe_table_names,
)


@st.composite
def single_table_aggregate(draw):
    """Scenario: SELECT <group_cols>, <agg>(<col>) FROM <table> GROUP BY <group_cols>."""
    group_name = draw(safe_col_names)
    used = {group_name}
    agg_name = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(agg_name)

    extra_cols = []
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        name = draw(safe_col_names.filter(lambda n: n not in used))
        used.add(name)
        extra_cols.append(Column(name=name, dtype=draw(col_type())))

    columns = [
        Column(name=group_name, dtype="VARCHAR"),
        Column(name=agg_name, dtype="INTEGER"),
        *extra_cols,
    ]

    tname = draw(safe_table_names.filter(lambda n: n not in used))
    table = Table(name=tname, columns=columns)

    initial_rows = draw(rows_for_table(table, min_size=2, max_size=15))

    agg_func = draw(st.sampled_from(["SUM", "COUNT", "AVG", "MIN", "MAX"]))
    agg_expr = f"{agg_func}({agg_name})" if agg_func != "COUNT" else "COUNT(*)"

    view_sql = f"SELECT {group_name}, {agg_expr} AS agg_val FROM {tname} GROUP BY {group_name}"

    delta_inserts = draw(rows_for_table(table, min_size=0, max_size=5))
    n_deletes = draw(st.integers(min_value=0, max_value=min(3, len(initial_rows))))
    delete_indices = (
        draw(
            st.lists(
                st.integers(min_value=0, max_value=len(initial_rows) - 1),
                min_size=n_deletes,
                max_size=n_deletes,
                unique=True,
            )
        )
        if n_deletes > 0
        else []
    )
    delta_deletes = [initial_rows[i] for i in delete_indices]

    delta = Delta(table_name=tname, inserts=delta_inserts, deletes=delta_deletes)

    return Scenario(
        tables=[table],
        initial_data={tname: initial_rows},
        view_sql=view_sql,
        deltas=[delta],
    )
