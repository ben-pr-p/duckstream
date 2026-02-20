"""Interactive tree sidebar widget for navigating MVs."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from textual.message import Message
from textual.widgets import Tree

if TYPE_CHECKING:
    from duckstream.orchestrator import Orchestrator


class MVSelected(Message):
    """Posted when a user selects an MV in the tree."""

    def __init__(self, fqn: str) -> None:
        super().__init__()
        self.fqn = fqn


class TreeSidebar(Tree):
    """Interactive tree displaying catalogs → tables → MVs."""

    DEFAULT_CSS = """
    TreeSidebar {
        width: 40;
        dock: left;
        border-right: solid $accent;
        padding: 1;
    }
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__("DuckStream")
        self._orchestrator = orchestrator
        self._mv_fqns: list[str] = []

    def on_mount(self) -> None:
        self.rebuild(self._orchestrator)
        self.root.expand_all()

    def rebuild(self, orchestrator: Orchestrator) -> None:
        """Rebuild the tree from the orchestrator's current state."""
        self._orchestrator = orchestrator
        self._mv_fqns = []
        self.root.remove_children()

        # Group base tables by catalog
        catalog_tables: dict[str, set[str]] = defaultdict(set)
        for reg in orchestrator._mvs.values():
            for table_name, cat in reg.mv.base_tables.items():
                catalog_tables[cat].add(table_name)

        # Map base_table -> MVs that use it
        table_to_mvs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for fqn, reg in orchestrator._mvs.items():
            for table_name, cat in reg.mv.base_tables.items():
                table_to_mvs[(cat, table_name)].append(fqn)

        for cat in sorted(catalog_tables):
            cat_node = self.root.add(f"📁 {cat}", expand=True)
            for table_name in sorted(catalog_tables[cat]):
                table_node = cat_node.add(f"📄 {table_name}", expand=True)
                for fqn in sorted(table_to_mvs.get((cat, table_name), [])):
                    reg = orchestrator._mvs[fqn]
                    icon = "🔄" if reg.mv.strategy == "full_refresh" else "⚡"
                    mv_node = table_node.add_leaf(f"{icon} {fqn}")
                    mv_node.data = fqn
                    self._mv_fqns.append(fqn)

        self.root.expand_all()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """When a node with MV data is selected (Enter), post MVSelected."""
        if event.node.data is not None:
            self.post_message(MVSelected(event.node.data))
