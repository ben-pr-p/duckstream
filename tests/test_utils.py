"""Tests for utility functions (safe_to_expire_sql, pending_maintenance_sql)."""

from duckstream import pending_maintenance_sql, safe_to_expire_sql
from duckstream.plan import Naming


class TestSafeToExpireSql:
    """Tests for safe_to_expire_sql."""

    def test_generates_valid_sql(self):
        sql = safe_to_expire_sql(
            mv_catalogs=["analytics"],
            source_catalogs=["sales_dl"],
        )
        assert "source_catalog" in sql
        assert "min_required_snapshot" in sql
        assert "latest_snapshot" in sql
        assert "analytics.main._ivm_cursors" in sql
        assert "'sales_dl'" in sql

    def test_multiple_mv_catalogs(self):
        sql = safe_to_expire_sql(
            mv_catalogs=["analytics", "reporting"],
            source_catalogs=["sales_dl"],
        )
        assert "analytics.main._ivm_cursors" in sql
        assert "reporting.main._ivm_cursors" in sql
        assert "UNION ALL" in sql

    def test_multiple_source_catalogs(self):
        sql = safe_to_expire_sql(
            mv_catalogs=["analytics"],
            source_catalogs=["sales_dl", "crm_dl"],
        )
        assert "'sales_dl'" in sql
        assert "'crm_dl'" in sql

    def test_empty_mv_catalogs(self):
        sql = safe_to_expire_sql(
            mv_catalogs=[],
            source_catalogs=["sales_dl"],
        )
        assert "WHERE FALSE" in sql

    def test_empty_source_catalogs(self):
        sql = safe_to_expire_sql(
            mv_catalogs=["analytics"],
            source_catalogs=[],
        )
        assert "WHERE FALSE" in sql

    def test_custom_naming(self):
        class MyNaming(Naming):
            def cursors_table(self) -> str:
                return "_my_cursors"

        sql = safe_to_expire_sql(
            mv_catalogs=["analytics"],
            source_catalogs=["sales_dl"],
            naming=MyNaming(),
        )
        assert "_my_cursors" in sql
        assert "_ivm_cursors" not in sql


class TestPendingMaintenanceSql:
    """Tests for pending_maintenance_sql."""

    def test_generates_valid_sql(self):
        sql = pending_maintenance_sql(mv_catalogs=["analytics"])
        assert "mv_catalog" in sql
        assert "mv_name" in sql
        assert "source_catalog" in sql
        assert "pending_snapshots" in sql
        assert "analytics.main._ivm_cursors" in sql

    def test_multiple_mv_catalogs(self):
        sql = pending_maintenance_sql(mv_catalogs=["analytics", "reporting"])
        assert "analytics.main._ivm_cursors" in sql
        assert "reporting.main._ivm_cursors" in sql
        assert "UNION ALL" in sql

    def test_empty_mv_catalogs(self):
        sql = pending_maintenance_sql(mv_catalogs=[])
        assert "WHERE FALSE" in sql

    def test_custom_naming(self):
        class MyNaming(Naming):
            def cursors_table(self) -> str:
                return "_my_cursors"

        sql = pending_maintenance_sql(
            mv_catalogs=["analytics"],
            naming=MyNaming(),
        )
        assert "_my_cursors" in sql
        assert "_ivm_cursors" not in sql

    def test_order_by(self):
        sql = pending_maintenance_sql(mv_catalogs=["a", "b"])
        assert "ORDER BY mv_catalog, mv_name, source_catalog" in sql
