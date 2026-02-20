"""Effector plugin base class for processing MV changes and storing results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import duckdb
from pydantic import BaseModel


@dataclass
class EffectorResult:
    """Result of an effector flush operation."""

    rows_inserted: int = 0
    rows_skipped: int = 0
    rows_errored: int = 0


class Effector[TOptions: BaseModel](ABC):
    """Base class for effector plugins that process MV changes and store results.

    Effectors sit between an MV and a downstream DuckLake output table.
    They execute side effects (API calls, notifications, etc.) for each row
    change and store results back into the output table.

    Subclasses must define:
    - ``Options``: a Pydantic model for configuration
    - ``columns``: list of (name, sql_type) tuples for the output table
    - ``handle_insert``, ``handle_update``, ``handle_delete``
    """

    Options: ClassVar[type[BaseModel]]
    columns: ClassVar[list[tuple[str, str]]]
    on_error: ClassVar[Literal["raise", "skip", "store"]] = "raise"
    batched: ClassVar[bool] = True

    def __init__(
        self,
        *,
        catalog: str,
        table: str,
        output_table: str,
        options: TOptions,
    ):
        self.catalog = catalog
        self.table = table
        self.output_table = output_table
        self.schema_ = "main"
        self.options = options

    @property
    def fqn(self) -> str:
        """Fully qualified source MV name: catalog.schema.table."""
        return f"{self.catalog}.{self.schema_}.{self.table}"

    @property
    def output_fqn(self) -> str:
        """Fully qualified output table name: catalog.schema.output_table."""
        return f"{self.catalog}.{self.schema_}.{self.output_table}"

    @property
    def effector_name(self) -> str:
        """Unique name for cursor tracking."""
        return f"{type(self).__name__}_{self.table}"

    @abstractmethod
    def handle_insert(self, row: dict) -> dict | None:
        """Handle an inserted row. Return a dict matching ``columns``, or None to skip."""
        ...

    @abstractmethod
    def handle_update(self, old_row: dict, new_row: dict) -> dict | None:
        """Handle an updated row. Return a dict matching ``columns``, or None to skip."""
        ...

    @abstractmethod
    def handle_delete(self, row: dict) -> dict | None:
        """Handle a deleted row. Return a dict matching ``columns``, or None to skip."""
        ...

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:  # noqa: B027
        """Optional one-time setup (install extensions, create secrets)."""
