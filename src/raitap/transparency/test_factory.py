from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from raitap.transparency.factory import explain_and_log


def test_explain_and_log_logs_transparency_artifacts(monkeypatch, tmp_path):
    logger = MagicMock()
    result = {"run_dir": tmp_path / "transparency"}
    result["run_dir"].mkdir()

    monkeypatch.setattr(
        "raitap.transparency.factory.explain",
        lambda *args, **kwargs: result,
    )

    out = explain_and_log(
        config=SimpleNamespace(),
        model=object(),
        inputs=object(),
        logger=logger,
        output_dir=result["run_dir"],
    )

    assert out == result
    logger.log_artifacts.assert_called_once_with(
        result["run_dir"],
        artifact_path="transparency",
    )
