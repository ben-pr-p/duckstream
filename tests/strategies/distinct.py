"""Strategy: SELECT DISTINCT on a single table."""

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
def single_table_distinct(draw):
    """Scenario: SELECT DISTINCT <cols> FROM <table> [WHERE <predicate>].

    Uses small value ranges to ensure duplicate rows exist, making DISTINCT
    meaningful. Projects a subset of columns to increase duplicate probability.
    """
    n_cols = draw(st.integers(min_value=2, max_value=4))
    columns = []
    used_names: set[str] = set()
    for i in range(n_cols):
        name = draw(safe_col_names.filter(lambda n: n not in used_names))
        used_names.add(name)
        # Use small-range types to produce duplicates
        dtype = draw(st.sampled_from(["INTEGER", "VARCHAR"])) if i > 0 else "INTEGER"
        columns.append(Column(name=name, dtype=dtype))

    tname = draw(safe_table_names.filter(lambda n: n not in used_names))
    table = Table(name=tname, columns=columns)

    # Generate initial data with small value ranges to produce duplicates
    initial_rows = []
    for _ in range(draw(st.integers(min_value=3, max_value=15))):
        vals: dict[str, object] = {}
        for col in columns:
            if col.dtype == "INTEGER":
                vals[col.name] = draw(st.integers(min_value=1, max_value=5))
            else:
                vals[col.name] = draw(st.sampled_from(["a", "b", "c"]))
        initial_rows.append(Row(values=vals))

    # Project a subset of columns (1 to n_cols) for DISTINCT
    proj_cols = draw(
        st.lists(
            st.sampled_from(table.col_names),
            min_size=1,
            max_size=len(table.col_names),
            unique=True,
        )
    )
    select_clause = ", ".join(proj_cols)

    # Optional WHERE filter
    add_where = draw(st.booleans())
    threshold = draw(st.integers(min_value=1, max_value=4))
    where_clause = f" WHERE {columns[0].name} > {threshold}" if add_where else ""

    view_sql = f"SELECT DISTINCT {select_clause} FROM {tname}{where_clause}"

    # Deltas: inserts with small ranges (likely duplicates) + some deletes
    delta_inserts = []
    for _ in range(draw(st.integers(min_value=0, max_value=5))):
        vals = {}
        for col in columns:
            if col.dtype == "INTEGER":
                vals[col.name] = draw(st.integers(min_value=1, max_value=5))
            else:
                vals[col.name] = draw(st.sampled_from(["a", "b", "c"]))
        delta_inserts.append(Row(values=vals))

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
