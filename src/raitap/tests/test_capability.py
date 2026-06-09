from raitap.types import Capability


def test_seeded_members_exist() -> None:
    # AUTOGRAD (torch) and TREE_MODEL / PREDICT_PROBA (tree backends) are all live.
    assert {c.value for c in Capability} >= {"autograd", "tree_model", "predict_proba"}


def test_is_str_enum() -> None:
    assert Capability.AUTOGRAD == "autograd"
    assert isinstance(Capability.AUTOGRAD, str)


def test_forward_only_is_empty_capability_set() -> None:
    from raitap.types import FORWARD_ONLY

    assert frozenset() == FORWARD_ONLY


def test_resolved_hardware_table_holds_three_axes() -> None:
    from raitap.types import Hardware, ResolvedHardware

    # member (resolved name) | pyproject_extra_suffix | config_hardware_value
    assert ResolvedHardware.cpu.pyproject_extra_suffix == "cpu"
    assert ResolvedHardware.cuda.pyproject_extra_suffix == "cuda"
    assert ResolvedHardware.xpu.pyproject_extra_suffix == "intel"  # XPU wheels labelled intel
    assert ResolvedHardware.cpu.config_hardware_value is Hardware.cpu
    assert ResolvedHardware.cuda.config_hardware_value is Hardware.gpu
    assert ResolvedHardware.xpu.config_hardware_value is Hardware.gpu


def test_resolved_hardware_is_str_enum() -> None:
    from raitap.types import ResolvedHardware

    # StrEnum members compare/serialise as their value, so probe results stay
    # string-compatible with existing call sites.
    assert ResolvedHardware.xpu == "xpu"
    assert f"{ResolvedHardware.xpu}" == "xpu"
