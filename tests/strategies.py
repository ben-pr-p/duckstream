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

# SQL reserved words to avoid as identifiers
_RESERVED = {
    "select", "from", "where", "join", "on", "group", "by", "as", "and", "or",
    "not", "in", "is", "null", "true", "false", "insert", "delete", "update",
    "create", "table", "into", "values", "set", "order", "having", "limit",
    "count", "sum", "avg", "min", "max", "all", "distinct", "between", "like",
    "exists", "case", "when", "then", "else", "end", "union", "except", "left",
    "right", "inner", "outer", "full", "cross", "with", "key", "rowid",
}  # fmt: skip

safe_col_names = col_names.filter(lambda n: n not in _RESERVED)
safe_table_names = table_names.filter(lambda n: n not in _RESERVED)


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
        name = draw(col_names.filter(lambda n: n not in used_names))
        used_names.add(name)
        # Use small-range types to produce duplicates
        dtype = draw(st.sampled_from(["INTEGER", "VARCHAR"])) if i > 0 else "INTEGER"
        columns.append(Column(name=name, dtype=dtype))

    tname = draw(table_names.filter(lambda n: n not in used_names))
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
