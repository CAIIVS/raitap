import pytest

from raitap.configs import registry_resolve as rr


def test_reject_config_target_raises_on_target_key() -> None:
    with pytest.raises(rr.UnsafeConfigTargetError, match="use:"):
        rr.reject_config_target({"_target_": "os.system", "command": "x"})


def test_reject_config_target_allows_clean_cfg() -> None:
    rr.reject_config_target({"use": "captum", "algorithm": "IntegratedGradients"})  # no raise


def test_reject_config_target_raises_on_nested_target_in_dict() -> None:
    cfg = {
        "use": "captum",
        "constructor": {
            "baseline": {"_target_": "os.system", "_args_": ["id"]},
        },
    }
    with pytest.raises(rr.UnsafeConfigTargetError, match="use:"):
        rr.reject_config_target(cfg)


def test_reject_config_target_raises_on_target_nested_in_list() -> None:
    cfg = {
        "use": "captum",
        "visualisers": [{"use": "clean"}, {"_target_": "os.system"}],
    }
    with pytest.raises(rr.UnsafeConfigTargetError, match="use:"):
        rr.reject_config_target(cfg)


def test_reject_config_target_allows_clean_deeply_nested_cfg() -> None:
    cfg = {
        "use": "captum",
        "constructor": {
            "baseline": {"use": "zeros", "shape": [1, 2, 3]},
        },
        "visualisers": [{"use": "clean"}, {"use": "another", "options": {"a": 1}}],
    }
    rr.reject_config_target(cfg)  # no raise


def test_resolve_target_fqn_returns_registered_fqn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "raitap._adapters._TARGET_FQN",
        {
            "transparency": {
                "captum": "raitap.transparency.explainers.captum_explainer.CaptumExplainer"
            }
        },
    )
    assert rr.resolve_target_fqn("transparency", "captum").endswith("CaptumExplainer")


def test_resolve_target_fqn_unknown_key_lists_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "raitap._adapters._TARGET_FQN", {"transparency": {"captum": "x", "shap": "y"}}
    )
    with pytest.raises(ValueError, match="captum, shap"):
        rr.resolve_target_fqn("transparency", "capdum")
