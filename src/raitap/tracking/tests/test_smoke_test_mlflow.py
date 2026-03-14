from __future__ import annotations

from .smoke_test_mlflow import DEFAULT_TRACKING_URI, build_parser


def test_smoke_test_defaults_to_local_tracking_server():
    args = build_parser().parse_args([])

    assert args.tracking_uri == DEFAULT_TRACKING_URI
