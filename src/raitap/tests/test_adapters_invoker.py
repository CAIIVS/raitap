"""The generic Invoker protocol is structural (#266)."""

from __future__ import annotations

from raitap._adapters import Invoker  # runtime import: deletion of Invoker would fail this test


def test_plain_callable_satisfies_invoker() -> None:
    def invoke(ctx: object) -> str:
        return f"ran:{ctx}"

    fn: Invoker[object, str] = invoke  # structural: a 1-arg callable is an Invoker
    assert fn("x") == "ran:x"


def test_invoker_is_importable_at_runtime() -> None:
    # Guards against the protocol being deleted or hidden behind TYPE_CHECKING.
    assert Invoker is not None
