"""Strategy: SELECT / PROJECT / WHERE on a single table."""

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
def single_table_select(draw):
    """Scenario: SELECT <cols> FROM <table> [WHERE <predicate>]."""
    n_cols = draw(st.integers(min_value=2, max_value=5))
    columns = []
    used_names = set()
    for i in range(n_cols):
        name = draw(safe_col_names.filter(lambda n: n not in used_names))
        used_names.add(name)
        dtype = draw(col_type()) if i > 0 else "INTEGER"  # first col always INT
        columns.append(Column(name=name, dtype=dtype))

    tname = draw(safe_table_names.filter(lambda n: n not in used_names))
    table = Table(name=tname, columns=columns)

    initial_rows = draw(rows_for_table(table, min_size=1, max_size=15))

    proj_cols = draw(
        st.lists(
            st.sampled_from(table.col_names),
            min_size=1,
            max_size=len(table.col_names),
            unique=True,
        )
    )
    select_clause = ", ".join(proj_cols)

    add_where = draw(st.booleans())
    threshold = draw(st.integers(min_value=-500, max_value=500))
    where_clause = f" WHERE {columns[0].name} > {threshold}" if add_where else ""

    # Unqualified table name — harness qualifies at runtime
    view_sql = f"SELECT {select_clause} FROM {tname}{where_clause}"

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
        if n_deletes > 0 and initial_rows
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
