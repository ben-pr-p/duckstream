"""Strategies for subquery-based scenarios."""

from hypothesis import strategies as st

from tests.strategies.primitives import Column, Delta, Row, Scenario, Table


@st.composite
def not_in_with_null_outer(draw):
    """Scenario: NOT IN with NULLs in the outer column.

    Ensures at least one NULL in the outer table and at least one NULL inserted
    during deltas to exercise NOT IN null semantics.
    """
    orders_name = "orders"
    blocked_name = "blocked"
    orders = Table(orders_name, [Column("id", "INTEGER"), Column("cid", "INTEGER")])
    blocked = Table(blocked_name, [Column("cid", "INTEGER")])

    n_initial = draw(st.integers(min_value=2, max_value=6))
    initial_cids = draw(
        st.lists(
            st.one_of(st.integers(min_value=1, max_value=5), st.none()),
            min_size=n_initial,
            max_size=n_initial,
        )
    )
    if all(cid is not None for cid in initial_cids):
        initial_cids[0] = None

    initial_orders = [Row({"id": i + 1, "cid": cid}) for i, cid in enumerate(initial_cids)]

    blocked_initial = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=0,
            max_size=3,
            unique=True,
        )
    )
    blocked_rows = [Row({"cid": cid}) for cid in blocked_initial]

    insert_cids = draw(
        st.lists(
            st.one_of(st.integers(min_value=1, max_value=5), st.none()),
            min_size=1,
            max_size=3,
        )
    )
    if all(cid is not None for cid in insert_cids):
        insert_cids[0] = None
    max_id = max(row.values["id"] for row in initial_orders)
    insert_orders = [Row({"id": max_id + i + 1, "cid": cid}) for i, cid in enumerate(insert_cids)]

    n_deletes = draw(st.integers(min_value=0, max_value=min(2, len(initial_orders))))
    delete_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(initial_orders) - 1),
            min_size=n_deletes,
            max_size=n_deletes,
            unique=True,
        )
        if n_deletes > 0
        else st.just([])
    )
    delete_orders = [initial_orders[i] for i in delete_indices]

    blocked_inserts = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=0,
            max_size=2,
            unique=True,
        )
    )
    blocked_delta = Delta(
        blocked_name, inserts=[Row({"cid": c}) for c in blocked_inserts], deletes=[]
    )
    orders_delta = Delta(orders_name, inserts=insert_orders, deletes=delete_orders)

    view_sql = (
        "SELECT orders.id, orders.cid FROM orders WHERE orders.cid NOT IN (SELECT cid FROM blocked)"
    )

    return Scenario(
        tables=[orders, blocked],
        initial_data={orders_name: initial_orders, blocked_name: blocked_rows},
        view_sql=view_sql,
        deltas=[blocked_delta, orders_delta],
    )
