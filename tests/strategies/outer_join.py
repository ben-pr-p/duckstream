"""Strategy: two-table outer JOIN (LEFT / RIGHT / FULL)."""

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
def two_table_outer_join(draw):
    """Scenario: SELECT ... FROM R LEFT/RIGHT/FULL JOIN S ON R.key = S.key.

    Uses small key pools so some keys have no match on one side (exercises
    NULL-extension logic). The join type is randomly chosen.
    """
    used: set[str] = set()

    key_name = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(key_name)

    # Left table: key + 1 extra column
    col_l = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(col_l)
    left_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(left_name)
    left_table = Table(
        left_name,
        [Column(key_name, "INTEGER"), Column(col_l, "INTEGER")],
    )

    # Right table: key + 1 extra column
    col_r = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(col_r)
    right_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(right_name)
    right_table = Table(
        right_name,
        [Column(key_name, "INTEGER"), Column(col_r, "VARCHAR")],
    )

    # Key pools: intentionally different ranges so some keys have no match
    left_key_pool = [1, 2, 3, 4]
    right_key_pool = [3, 4, 5, 6]

    left_rows = []
    for _ in range(draw(st.integers(2, 6))):
        left_rows.append(
            Row(
                {
                    key_name: draw(st.sampled_from(left_key_pool)),
                    col_l: draw(st.integers(-50, 50)),
                }
            )
        )

    right_rows = []
    for _ in range(draw(st.integers(2, 6))):
        right_rows.append(
            Row(
                {
                    key_name: draw(st.sampled_from(right_key_pool)),
                    col_r: draw(st.sampled_from(["x", "y", "z"])),
                }
            )
        )

    # Join type
    join_type = draw(st.sampled_from(["LEFT", "RIGHT", "FULL OUTER"]))

    # Projection: always include key from left and extras from both
    proj_cols = [
        f"{left_name}.{key_name}",
        f"{left_name}.{col_l}",
        f"{right_name}.{col_r}",
    ]
    select_clause = ", ".join(proj_cols)

    view_sql = (
        f"SELECT {select_clause}"
        f" FROM {left_name}"
        f" {join_type} JOIN {right_name}"
        f" ON {left_name}.{key_name} = {right_name}.{key_name}"
    )

    # Deltas
    delta_target = draw(st.sampled_from(["left", "right", "both"]))
    deltas = []

    if delta_target in ("left", "both"):
        inserts = []
        for _ in range(draw(st.integers(0, 2))):
            inserts.append(
                Row(
                    {
                        key_name: draw(st.sampled_from(left_key_pool + right_key_pool)),
                        col_l: draw(st.integers(-50, 50)),
                    }
                )
            )
        n_del = draw(st.integers(0, min(2, len(left_rows))))
        del_idx = (
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
            Delta(
                left_name,
                inserts=inserts,
                deletes=[left_rows[i] for i in del_idx],
            )
        )

    if delta_target in ("right", "both"):
        inserts = []
        for _ in range(draw(st.integers(0, 2))):
            inserts.append(
                Row(
                    {
                        key_name: draw(st.sampled_from(left_key_pool + right_key_pool)),
                        col_r: draw(st.sampled_from(["x", "y", "z"])),
                    }
                )
            )
        n_del = draw(st.integers(0, min(2, len(right_rows))))
        del_idx = (
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
            Delta(
                right_name,
                inserts=inserts,
                deletes=[right_rows[i] for i in del_idx],
            )
        )

    return Scenario(
        tables=[left_table, right_table],
        initial_data={left_name: left_rows, right_name: right_rows},
        view_sql=view_sql,
        deltas=deltas,
    )
