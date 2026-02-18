"""Hypothesis strategies for generating IVM test scenarios."""

from dataclasses import dataclass
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

# Column types we support in generated schemas
COLUMN_TYPES = ["INTEGER", "BIGINT", "DOUBLE", "VARCHAR", "BOOLEAN"]

col_names = st.from_regex(r"[a-z]{1}[a-z0-9_]{0,7}", fullmatch=True)
table_names = st.from_regex(r"[a-z]{1}[a-z0-9_]{0,7}", fullmatch=True)


@st.composite
def col_type(draw):
    return draw(st.sampled_from(COLUMN_TYPES))


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

@dataclass
class Column:
    name: str
    dtype: str


@dataclass
class Table:
    name: str
    columns: list[Column]

    def ddl(self, catalog: str) -> str:
        """CREATE TABLE in the given DuckLake catalog."""
        cols = ", ".join(f"{c.name} {c.dtype}" for c in self.columns)
        return f"CREATE TABLE {catalog}.{self.name} ({cols})"

    @property
    def col_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class Row:
    values: dict[str, object]


@dataclass
class Delta:
    """A set of rows to insert (+1) or delete (-1) from a table."""
    table_name: str
    inserts: list[Row]
    deletes: list[Row]


@dataclass
class Scenario:
    """Complete test scenario: tables, initial data, view SQL, and deltas.

    view_sql uses unqualified table names. The test harness qualifies them
    with the DuckLake catalog name at runtime.
    """
    tables: list[Table]
    initial_data: dict[str, list[Row]]  # table_name -> rows
    view_sql: str
    deltas: list[Delta]


# ---------------------------------------------------------------------------
# Value generators per column type
# ---------------------------------------------------------------------------

def value_for_type(dtype: str):
    if dtype == "INTEGER":
        return st.integers(min_value=-1000, max_value=1000)
    elif dtype == "BIGINT":
        return st.integers(min_value=-100000, max_value=100000)
    elif dtype == "DOUBLE":
        return st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    elif dtype == "VARCHAR":
        return st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=8)
    elif dtype == "BOOLEAN":
        return st.booleans()
    raise ValueError(f"Unknown type: {dtype}")


@st.composite
def row_for_table(draw, table: Table):
    values = {}
    for col in table.columns:
        values[col.name] = draw(value_for_type(col.dtype))
    return Row(values=values)


@st.composite
def rows_for_table(draw, table: Table, min_size=0, max_size=10):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(row_for_table(table)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Scenario strategies — one per SQL fragment class
# ---------------------------------------------------------------------------

@st.composite
def single_table_select(draw):
    """Scenario: SELECT <cols> FROM <table> [WHERE <predicate>]."""
    n_cols = draw(st.integers(min_value=2, max_value=5))
    columns = []
    used_names = set()
    for i in range(n_cols):
        name = draw(col_names.filter(lambda n: n not in used_names))
        used_names.add(name)
        dtype = draw(col_type()) if i > 0 else "INTEGER"  # first col always INT
        columns.append(Column(name=name, dtype=dtype))

    tname = draw(table_names.filter(lambda n: n not in used_names))
    table = Table(name=tname, columns=columns)

    initial_rows = draw(rows_for_table(table, min_size=1, max_size=15))

    proj_cols = draw(st.lists(
        st.sampled_from(table.col_names),
        min_size=1,
        max_size=len(table.col_names),
        unique=True,
    ))
    select_clause = ", ".join(proj_cols)

    add_where = draw(st.booleans())
    threshold = draw(st.integers(min_value=-500, max_value=500))
    where_clause = f" WHERE {columns[0].name} > {threshold}" if add_where else ""

    # Unqualified table name — harness qualifies at runtime
    view_sql = f"SELECT {select_clause} FROM {tname}{where_clause}"

    delta_inserts = draw(rows_for_table(table, min_size=0, max_size=5))
    n_deletes = draw(st.integers(min_value=0, max_value=min(3, len(initial_rows))))
    delete_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(initial_rows) - 1),
            min_size=n_deletes,
            max_size=n_deletes,
            unique=True,
        )
    ) if n_deletes > 0 and initial_rows else []
    delta_deletes = [initial_rows[i] for i in delete_indices]

    delta = Delta(table_name=tname, inserts=delta_inserts, deletes=delta_deletes)

    return Scenario(
        tables=[table],
        initial_data={tname: initial_rows},
        view_sql=view_sql,
        deltas=[delta],
    )


@st.composite
def single_table_aggregate(draw):
    """Scenario: SELECT <group_cols>, <agg>(<col>) FROM <table> GROUP BY <group_cols>."""
    group_name = draw(col_names)
    used = {group_name}
    agg_name = draw(col_names.filter(lambda n: n not in used))
    used.add(agg_name)

    extra_cols = []
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        name = draw(col_names.filter(lambda n: n not in used))
        used.add(name)
        extra_cols.append(Column(name=name, dtype=draw(col_type())))

    columns = [
        Column(name=group_name, dtype="VARCHAR"),
        Column(name=agg_name, dtype="INTEGER"),
        *extra_cols,
    ]

    tname = draw(table_names.filter(lambda n: n not in used))
    table = Table(name=tname, columns=columns)

    initial_rows = draw(rows_for_table(table, min_size=2, max_size=15))

    agg_func = draw(st.sampled_from(["SUM", "COUNT", "AVG"]))
    agg_expr = f"{agg_func}({agg_name})" if agg_func != "COUNT" else "COUNT(*)"

    view_sql = f"SELECT {group_name}, {agg_expr} AS agg_val FROM {tname} GROUP BY {group_name}"

    delta_inserts = draw(rows_for_table(table, min_size=0, max_size=5))
    n_deletes = draw(st.integers(min_value=0, max_value=min(3, len(initial_rows))))
    delete_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(initial_rows) - 1),
            min_size=n_deletes,
            max_size=n_deletes,
            unique=True,
        )
    ) if n_deletes > 0 else []
    delta_deletes = [initial_rows[i] for i in delete_indices]

    delta = Delta(table_name=tname, inserts=delta_inserts, deletes=delta_deletes)

    return Scenario(
        tables=[table],
        initial_data={tname: initial_rows},
        view_sql=view_sql,
        deltas=[delta],
    )
