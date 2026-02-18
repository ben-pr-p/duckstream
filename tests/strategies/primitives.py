"""Primitive types and value generators for IVM test scenarios."""

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
    "do", "if", "at", "to", "go", "no", "of", "for", "int", "add", "drop",
    "alter", "index", "check", "primary", "foreign", "references", "column",
    "version", "type", "row", "asc", "desc", "over", "partition",
    "mv", "_ivm_cursors",  # IVM internal names
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
