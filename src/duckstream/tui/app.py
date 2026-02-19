"""Shared Textual App for DuckStream TUI modes."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, RichLog

from duckstream.orchestrator import Orchestrator
from duckstream.tui.tree_sidebar import TreeSidebar


class DuckStreamApp(App):
    """Base TUI app with tree sidebar and RichLog content area."""

    CSS = """
    #content-area {
        width: 1fr;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, orchestrator: Orchestrator, title: str = "DuckStream") -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.title = title

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield TreeSidebar(self.orchestrator)
            yield RichLog(id="content-area", highlight=True, markup=True)
        yield Footer()

    @property
    def log_widget(self) -> RichLog:
        return self.query_one("#content-area", RichLog)

    def refresh_tree(self) -> None:
        """Refresh the tree sidebar with current orchestrator state."""
        self.query_one(TreeSidebar).rebuild(self.orchestrator)
