import pytest

from raitap.configs import registry_resolve as rr


def test_reject_config_target_raises_on_target_key() -> None:
    with pytest.raises(rr.UnsafeConfigTargetError, match="use:"):
        rr.reject_config_target({"_target_": "os.system", "command": "x"})


def test_reject_config_target_allows_clean_cfg() -> None:
    rr.reject_config_target({"use": "captum", "algorithm": "IntegratedGradients"})  # no raise


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
