"""Strategies: two-table and three-table inner JOIN."""

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
def two_table_join(draw):
    """Scenario: SELECT ... FROM R JOIN S ON R.key = S.key."""
    used = set()

    # Shared join key
    key_name = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(key_name)

    # Left table: key + 1-2 extra columns
    left_extras = []
    for _ in range(draw(st.integers(1, 2))):
        name = draw(safe_col_names.filter(lambda n: n not in used))
        used.add(name)
        left_extras.append(Column(name=name, dtype=draw(st.sampled_from(["INTEGER", "VARCHAR"]))))

    left_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(left_name)
    left_table = Table(name=left_name, columns=[Column(key_name, "INTEGER"), *left_extras])

    # Right table: key + 1-2 extra columns
    right_extras = []
    for _ in range(draw(st.integers(1, 2))):
        name = draw(safe_col_names.filter(lambda n: n not in used))
        used.add(name)
        right_extras.append(Column(name=name, dtype=draw(st.sampled_from(["INTEGER", "VARCHAR"]))))

    right_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(right_name)
    right_table = Table(name=right_name, columns=[Column(key_name, "INTEGER"), *right_extras])

    # Generate data with overlapping keys to ensure join matches
    key_pool = draw(st.lists(st.integers(1, 20), min_size=3, max_size=8, unique=True))

    left_rows = []
    for _ in range(draw(st.integers(2, 8))):
        row_vals: dict[str, object] = {key_name: draw(st.sampled_from(key_pool))}
        for col in left_extras:
            row_vals[col.name] = draw(value_for_type(col.dtype))
        left_rows.append(Row(values=row_vals))

    right_rows = []
    for _ in range(draw(st.integers(2, 8))):
        row_vals = {key_name: draw(st.sampled_from(key_pool))}
        for col in right_extras:
            row_vals[col.name] = draw(value_for_type(col.dtype))
        right_rows.append(Row(values=row_vals))

    # Build projection: table-qualified columns from both tables
    left_proj = [f"{left_name}.{key_name}"]
    for col in left_extras:
        left_proj.append(f"{left_name}.{col.name}")
    right_proj = []
    for col in right_extras:
        right_proj.append(f"{right_name}.{col.name}")

    all_proj = left_proj + right_proj
    proj_cols = draw(
        st.lists(st.sampled_from(all_proj), min_size=1, max_size=len(all_proj), unique=True)
    )
    select_clause = ", ".join(proj_cols)

    view_sql = (
        f"SELECT {select_clause} FROM {left_name} "
        f"JOIN {right_name} ON {left_name}.{key_name} = {right_name}.{key_name}"
    )

    # Deltas: choose which tables get changes
    delta_target = draw(st.sampled_from(["left", "right", "both"]))
    deltas = []

    if delta_target in ("left", "both"):
        left_inserts = []
        for _ in range(draw(st.integers(0, 3))):
            row_vals = {key_name: draw(st.sampled_from(key_pool))}
            for col in left_extras:
                row_vals[col.name] = draw(value_for_type(col.dtype))
            left_inserts.append(Row(values=row_vals))
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
        left_deletes = [left_rows[i] for i in left_del_idx]
        deltas.append(Delta(left_name, inserts=left_inserts, deletes=left_deletes))

    if delta_target in ("right", "both"):
        right_inserts = []
        for _ in range(draw(st.integers(0, 3))):
            row_vals = {key_name: draw(st.sampled_from(key_pool))}
            for col in right_extras:
                row_vals[col.name] = draw(value_for_type(col.dtype))
            right_inserts.append(Row(values=row_vals))
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
        right_deletes = [right_rows[i] for i in right_del_idx]
        deltas.append(Delta(right_name, inserts=right_inserts, deletes=right_deletes))

    return Scenario(
        tables=[left_table, right_table],
        initial_data={left_name: left_rows, right_name: right_rows},
        view_sql=view_sql,
        deltas=deltas,
    )


@st.composite
def three_table_join(draw):
    """Scenario: SELECT ... FROM R JOIN S ON ... JOIN T ON ... (chain join)."""
    used: set[str] = set()

    # Two join keys: R-S share key1, S-T share key2
    key1 = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(key1)
    key2 = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(key2)

    # Extra column per table
    col_r = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(col_r)
    col_s = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(col_s)
    col_t = draw(safe_col_names.filter(lambda n: n not in used))
    used.add(col_t)

    r_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(r_name)
    s_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(s_name)
    t_name = draw(safe_table_names.filter(lambda n: n not in used))
    used.add(t_name)

    r_table = Table(r_name, [Column(key1, "INTEGER"), Column(col_r, "INTEGER")])
    s_table = Table(
        s_name, [Column(key1, "INTEGER"), Column(key2, "INTEGER"), Column(col_s, "VARCHAR")]
    )
    t_table = Table(t_name, [Column(key2, "INTEGER"), Column(col_t, "VARCHAR")])

    # Overlapping key pools
    key1_pool = draw(st.lists(st.integers(1, 10), min_size=3, max_size=6, unique=True))
    key2_pool = draw(st.lists(st.integers(1, 10), min_size=3, max_size=6, unique=True))

    r_rows = []
    for _ in range(draw(st.integers(2, 6))):
        r_rows.append(
            Row({key1: draw(st.sampled_from(key1_pool)), col_r: draw(st.integers(-50, 50))})
        )

    s_rows = []
    for _ in range(draw(st.integers(2, 6))):
        s_rows.append(
            Row(
                {
                    key1: draw(st.sampled_from(key1_pool)),
                    key2: draw(st.sampled_from(key2_pool)),
                    col_s: draw(st.sampled_from(["x", "y", "z"])),
                }
            )
        )

    t_rows = []
    for _ in range(draw(st.integers(2, 6))):
        t_rows.append(
            Row({key2: draw(st.sampled_from(key2_pool)), col_t: draw(st.sampled_from(["p", "q"]))})
        )

    # Projection: pick from all three tables
    all_proj = [
        f"{r_name}.{key1}",
        f"{r_name}.{col_r}",
        f"{s_name}.{col_s}",
        f"{t_name}.{col_t}",
    ]
    proj_cols = draw(
        st.lists(st.sampled_from(all_proj), min_size=2, max_size=len(all_proj), unique=True)
    )
    select_clause = ", ".join(proj_cols)

    view_sql = (
        f"SELECT {select_clause} FROM {r_name}"
        f" JOIN {s_name} ON {r_name}.{key1} = {s_name}.{key1}"
        f" JOIN {t_name} ON {s_name}.{key2} = {t_name}.{key2}"
    )

    # Deltas: choose 1, 2, or all 3 tables
    delta_targets = draw(
        st.lists(st.sampled_from([r_name, s_name, t_name]), min_size=1, max_size=3, unique=True)
    )
    deltas = []

    table_map = {r_name: (r_table, r_rows), s_name: (s_table, s_rows), t_name: (t_table, t_rows)}
    for tname in delta_targets:
        table, rows = table_map[tname]
        inserts = []
        for _ in range(draw(st.integers(0, 2))):
            vals: dict[str, object] = {}
            for col in table.columns:
                if col.name == key1:
                    vals[col.name] = draw(st.sampled_from(key1_pool))
                elif col.name == key2:
                    vals[col.name] = draw(st.sampled_from(key2_pool))
                else:
                    vals[col.name] = draw(value_for_type(col.dtype))
            inserts.append(Row(vals))
        n_del = draw(st.integers(0, min(2, len(rows))))
        del_idx = (
            draw(
                st.lists(st.integers(0, len(rows) - 1), min_size=n_del, max_size=n_del, unique=True)
            )
            if n_del > 0
            else []
        )
        deletes = [rows[i] for i in del_idx]
        deltas.append(Delta(tname, inserts=inserts, deletes=deletes))

    return Scenario(
        tables=[r_table, s_table, t_table],
        initial_data={r_name: r_rows, s_name: s_rows, t_name: t_rows},
        view_sql=view_sql,
        deltas=deltas,
    )
