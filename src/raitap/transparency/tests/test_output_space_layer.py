from __future__ import annotations

from types import SimpleNamespace

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    InputKind,
    InputSpec,
    MethodFamily,
    TensorLayout,
)
from raitap.transparency.semantics import infer_output_space


def _image_spec() -> InputSpec:
    return InputSpec(
        kind=InputKind.IMAGE,
        shape=(1, 3, 8, 8),
        layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        feature_names=None,
        metadata=None,
    )


def test_layer_path_non_cam_yields_layer_activation() -> None:
    spec = _image_spec()
    attributions = SimpleNamespace(shape=(1, 16, 4, 4))  # layer-shaped, not input-shaped
    out = infer_output_space(
        input_spec=spec,
        attributions=attributions,
        method_families=frozenset({MethodFamily.GRADIENT}),
        layer_path="1.layer4.2.conv3",
    )
    assert out.space is ExplanationOutputSpace.LAYER_ACTIVATION
    assert out.layer_path == "1.layer4.2.conv3"


def test_cam_with_layer_path_still_image_spatial_map() -> None:
    spec = _image_spec()
    attributions = SimpleNamespace(shape=(1, 1, 4, 4))
    out = infer_output_space(
        input_spec=spec,
        attributions=attributions,
        method_families=frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
        layer_path="1.layer4.2.conv3",
    )
    assert out.space is ExplanationOutputSpace.IMAGE_SPATIAL_MAP
    assert out.layer_path == "1.layer4.2.conv3"


def test_no_layer_path_unaffected() -> None:
    spec = _image_spec()
    attributions = SimpleNamespace(shape=(1, 3, 8, 8))
    out = infer_output_space(
        input_spec=spec,
        attributions=attributions,
        method_families=frozenset({MethodFamily.GRADIENT}),
        layer_path=None,
    )
    assert out.space is ExplanationOutputSpace.INPUT_FEATURES
    assert out.layer_path is None
