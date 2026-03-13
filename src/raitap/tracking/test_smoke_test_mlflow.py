from __future__ import annotations

from types import SimpleNamespace

from raitap.tracking.smoke_test_mlflow import (
    DEFAULT_TRACKING_URI,
    _extract_scalar_metrics,
    build_parser,
)


def test_smoke_test_defaults_to_local_tracking_server():
    args = build_parser().parse_args([])

    assert args.tracking_uri == DEFAULT_TRACKING_URI


def test_extract_scalar_metrics_returns_only_scalar_values():
    result = {
        "result": SimpleNamespace(
            metrics={
                "accuracy": 1.0,
                "count": 4,
                "flag": True,
                "non_scalar": [1, 2, 3],
            }
        )
    }

    assert _extract_scalar_metrics(result) == {
        "accuracy": 1.0,
        "count": 4,
        "flag": True,
    }
