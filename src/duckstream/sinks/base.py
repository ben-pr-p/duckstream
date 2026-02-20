"""Sink plugin base class for pushing MV changes to external systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import duckdb
from pydantic import BaseModel


@dataclass
class FlushResult:
    """Result of a sink flush operation."""

    rows_inserted: int = 0
    rows_deleted: int = 0
    rows_updated: int = 0


@dataclass
class ChangeSet:
    """Passed to Sink.flush() — everything needed to read MV state or deltas."""

    catalog: str
    schema: str
    table: str
    conn: duckdb.DuckDBPyConnection
    snapshot_start: int  # first new snapshot (last_flushed + 1)
    snapshot_end: int  # latest snapshot

    @property
    def fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.table}"


class Sink[TOptions: BaseModel](ABC):
    """Base class for sink plugins that push MV changes to external systems.

    Subclasses must define an ``Options`` class attribute (a Pydantic model)
    and implement ``flush()``.
    """

    Options: ClassVar[type[BaseModel]]
    batched: ClassVar[bool] = True

    def __init__(self, *, catalog: str, table: str, options: TOptions):
        self.catalog = catalog
        self.table = table
        self.schema_ = "main"
        self.options = options

    @property
    def fqn(self) -> str:
        """Fully qualified table name: catalog.schema.table."""
        return f"{self.catalog}.{self.schema_}.{self.table}"

    @property
    def sink_name(self) -> str:
        """Unique name for cursor tracking."""
        return f"{type(self).__name__}_{self.table}"

    @abstractmethod
    def flush(self, changes: ChangeSet) -> FlushResult:
        """Push changes from the MV to the external system."""
        ...

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:  # noqa: B027
        """Optional one-time setup (install extensions, create secrets)."""
