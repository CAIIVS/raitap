from __future__ import annotations

import pytest
from hydra_zen import instantiate

from raitap._adapters import _BUILDERS


def test_detection_builder_carries_nested_iou_field() -> None:
    import dataclasses

    builder = _BUILDERS["metrics"]["detection"]
    fields = {f.name: f for f in dataclasses.fields(builder)}
    assert "iou" in fields
    iou_field_type = fields["iou"].type
    assert "IoUConfig" in repr(iou_field_type) or "iou" in str(iou_field_type).lower()


@pytest.mark.parametrize(
    "registry_name",
    ["binary_classification", "multiclass_classification", "multilabel_classification"],
)
def test_classification_builders_have_no_cross_task_fields(registry_name: str) -> None:
    import dataclasses

    builder = _BUILDERS["metrics"][registry_name]
    field_names = {f.name for f in dataclasses.fields(builder)}
    if registry_name == "binary_classification":
        assert "num_classes" not in field_names
        assert "num_labels" not in field_names
        assert "average" not in field_names
    if registry_name == "multiclass_classification":
        assert "num_labels" not in field_names
        assert "threshold" not in field_names
    if registry_name == "multilabel_classification":
        assert "num_classes" not in field_names


def test_detection_instantiate_with_nested_iou() -> None:
    pytest.importorskip("torchmetrics")
    pytest.importorskip("faster_coco_eval")
    cfg = {
        "_target_": "raitap.metrics.DetectionMetrics",
        "_convert_": "all",
        "iou": {"type": "bbox", "thresholds": [0.5]},
    }
    metric = instantiate(cfg)
    assert metric.metric.iou_thresholds == [0.5]
