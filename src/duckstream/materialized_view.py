from dataclasses import dataclass, field


class Naming:
    """Default naming strategy for generated tables and columns. Subclass to override."""

    def mv_table(self) -> str:
        return "mv"

    def cursors_table(self) -> str:
        return "_ivm_cursors"

    def aux_column(self, purpose: str) -> str:
        return f"_ivm_{purpose}"


@dataclass
class MaterializedView:
    """Complete set of SQL statements for IVM maintenance."""

    view_sql: str
    create_cursors_table: str
    create_mv: str
    initialize_cursors: list[str]
    maintain: list[str]
    base_tables: dict[str, str]  # table_name -> catalog
    features: set[str] = field(default_factory=set)
    query_mv: str = ""


class UnsupportedSQLError(Exception):
    def __init__(self, feature: str, message: str):
        self.feature = feature
        self.message = message
        super().__init__(message)
