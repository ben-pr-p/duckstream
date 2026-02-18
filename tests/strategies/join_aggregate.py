"""Strategy: JOIN + GROUP BY aggregate."""

from hypothesis import strategies as st

from tests.strategies.primitives import (
    Column,
    Delta,
    Row,
    Scenario,
    Table,
    safe_col_names,
    safe_table_names,
)


@st.composite
def join_then_aggregate(draw):
    """Scenario: SELECT <group_col>, <agg>(<col>) FROM R JOIN S ON ... GROUP BY <group_col>."""
    used: set[str] = set()

    # Shared join key
    key_name = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(key_name)

    # Group column (from right table, e.g. region)
    group_name = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(group_name)

    # Agg column (from left table, e.g. amount)
    agg_name = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(agg_name)

    left_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(left_name)
    right_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(right_name)

    left_table = Table(
        name=left_name,
        columns=[Column(key_name, "INTEGER"), Column(agg_name, "INTEGER")],
    )
    right_table = Table(
        name=right_name,
        columns=[Column(key_name, "INTEGER"), Column(group_name, "VARCHAR")],
    )

    # Generate data with overlapping keys
    key_pool = draw(st.lists(st.integers(1, 20), min_size=3, max_size=8, unique=True))

    left_rows = []
    for _ in range(draw(st.integers(3, 10))):
        left_rows.append(
            Row(
                values={
                    key_name: draw(st.sampled_from(key_pool)),
                    agg_name: draw(st.integers(-100, 100)),
                }
            )
        )

    # Right table: use a small set of group values to get interesting aggregations
    group_pool = draw(
        st.lists(
            st.text(alphabet="abcdef", min_size=1, max_size=3),
            min_size=2,
            max_size=4,
            unique=True,
        )
    )
    right_rows = []
    for _ in range(draw(st.integers(2, 6))):
        right_rows.append(
            Row(
                values={
                    key_name: draw(st.sampled_from(key_pool)),
                    group_name: draw(st.sampled_from(group_pool)),
                }
            )
        )

    # Choose aggregate function
    agg_func = draw(st.sampled_from(["SUM", "COUNT"]))
    agg_expr = f"SUM({left_name}.{agg_name})" if agg_func == "SUM" else "COUNT(*)"

    view_sql = (
        f"SELECT {right_name}.{group_name}, {agg_expr} AS agg_val"
        f" FROM {left_name}"
        f" JOIN {right_name} ON {left_name}.{key_name} = {right_name}.{key_name}"
        f" GROUP BY {right_name}.{group_name}"
    )

    # Deltas
    delta_target = draw(st.sampled_from(["left", "right", "both"]))
    deltas = []

    if delta_target in ("left", "both"):
        left_inserts = []
        for _ in range(draw(st.integers(0, 3))):
            left_inserts.append(
                Row(
                    values={
                        key_name: draw(st.sampled_from(key_pool)),
                        agg_name: draw(st.integers(-100, 100)),
                    }
                )
            )
        n_del = draw(st.integers(0, min(2, len(left_rows))))
        left_del_idx = (
            draw(
                st.lists(
                    st.integers(0, len(left_rows) - 1),
                    min_size=n_del,
                    max_size=n_del,
                    unique=True,
                )
            )
            if n_del > 0
            else []
        )
        deltas.append(
            Delta(left_name, inserts=left_inserts, deletes=[left_rows[i] for i in left_del_idx])
        )

    if delta_target in ("right", "both"):
        right_inserts = []
        for _ in range(draw(st.integers(0, 2))):
            right_inserts.append(
                Row(
                    values={
                        key_name: draw(st.sampled_from(key_pool)),
                        group_name: draw(st.sampled_from(group_pool)),
                    }
                )
            )
        n_del = draw(st.integers(0, min(2, len(right_rows))))
        right_del_idx = (
            draw(
                st.lists(
                    st.integers(0, len(right_rows) - 1),
                    min_size=n_del,
                    max_size=n_del,
                    unique=True,
                )
            )
            if n_del > 0
            else []
        )
        deltas.append(
            Delta(right_name, inserts=right_inserts, deletes=[right_rows[i] for i in right_del_idx])
        )

    return Scenario(
        tables=[left_table, right_table],
        initial_data={left_name: left_rows, right_name: right_rows},
        view_sql=view_sql,
        deltas=deltas,
    )
