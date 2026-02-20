"""Source plugin base class for replicating external data into DuckLake tables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import duckdb
from pydantic import BaseModel


@dataclass
class SyncResult:
    """Result of a source sync operation."""

    rows_inserted: int = 0
    rows_deleted: int = 0
    rows_updated: int = 0


class Source[TOptions: BaseModel](ABC):
    """Base class for source plugins that replicate external data into DuckLake.

    Subclasses must define an ``Options`` class attribute (a Pydantic model)
    and implement ``sync()``.
    """

    Options: ClassVar[type[BaseModel]]

    def __init__(self, *, catalog: str, table: str, options: TOptions):
        self.catalog = catalog
        self.table = table
        self.schema_ = "main"
        self.options = options

    @property
    def fqn(self) -> str:
        """Fully qualified table name: catalog.schema.table."""
        return f"{self.catalog}.{self.schema_}.{self.table}"

    @abstractmethod
    def sync(self, conn: duckdb.DuckDBPyConnection) -> SyncResult:
        """Fetch data from the external source and write it into the DuckLake table.

        Responsible for creating/altering the table if needed.
        """
        ...

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:  # noqa: B027
        """Optional one-time setup (install extensions, create secrets)."""
