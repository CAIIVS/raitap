from raitap.types import Capability


def test_seeded_members_exist() -> None:
    # AUTOGRAD is live today; TREE_MODEL / PREDICT_PROBA are pre-added roadmap
    # members (no backend provides them and no algorithm requires them yet).
    assert {c.value for c in Capability} >= {"autograd", "tree_model", "predict_proba"}


def test_is_str_enum() -> None:
    assert Capability.AUTOGRAD == "autograd"
    assert isinstance(Capability.AUTOGRAD, str)


def test_forward_only_is_empty_capability_set() -> None:
    from raitap.types import FORWARD_ONLY

    assert frozenset() == FORWARD_ONLY
