"""End-to-end pytest covering the detection pipeline against a small torchvision detector.

We deliberately use ``fasterrcnn_mobilenet_v3_large_fpn`` (~80 MB; ~5 s on CPU
for two images) rather than the contributor-facing ``fasterrcnn_resnet50_fpn_v2``
(~180 MB; ~30 s on CPU). The resnet50 variant is the acceptance criterion for
issue #146 and is exercised by the dedicated contributor config + CI workflow;
the pytest layer keeps a fast smoke variant so iteration on the detection
pipeline stays cheap.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.metrics import MetricsEvaluation


@pytest.mark.cuda
def test_detection_pipeline_e2e_via_fasterrcnn_mobilenet(tmp_path: Path) -> None:
    # Heavy optional deps — skip if not installed in this environment.
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")

    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
    except ImportError:
        pytest.skip("torchvision detection extras not available")

    from raitap.data.samples import SAMPLE_SOURCES, _load_sample

    if "UdacitySelfDriving" not in SAMPLE_SOURCES:
        pytest.skip("UdacitySelfDriving sample registry missing")

    # Download (or hit the cache) and clip to a single sample at reduced
    # resolution — full-res IG on detection is RAM-heavy and OOMs the
    # GitHub-hosted runner. The test asserts plumbing, not attribution
    # quality.
    try:
        full_tensor, sample_ids = _load_sample("UdacitySelfDriving")
    except Exception as error:
        pytest.skip(f"udacity samples unreachable: {error}")

    # Downsample 1280x720 → 320x180 to slash IG memory budget.
    images_tensor = torch.nn.functional.interpolate(
        full_tensor[:1], size=(180, 320), mode="bilinear", align_corners=False
    )
    sample_ids = sample_ids[:1]
    assert images_tensor.shape[0] == 1

    # --- model + backend ---------------------------------------------------
    from raitap.models.backend import TorchBackend
    from raitap.models.model import Model

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    torch_model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    torch_model.eval()

    backend = TorchBackend(torch_model, device=torch.device("cpu"))
    model = Model.__new__(Model)
    model.backend = backend

    # --- data + detection labels ------------------------------------------
    from raitap.data.data import Data

    data = Data.__new__(Data)
    # Detection batches are ragged ``list[Tensor]`` (one native-resolution
    # ``(C, H, W)`` per image), not a dense ``NCHW`` tensor — exercise the
    # canonical detection input path (issue #197).
    data.tensor = [images_tensor[0]]
    data.sample_ids = sample_ids
    data.name = "udacity-e2e"
    data.source = "UdacitySelfDriving"

    # COCO class 3 = "car"; coordinates are plausible but don't need to match
    # real ground truth — the assertions cover pipeline plumbing, not mAP.
    # Coords in downsampled (320x180) space; one box, one label.
    labels_payload = [
        {
            "sample_id": sample_ids[0],
            "boxes": [[25.0, 75.0, 150.0, 150.0]],
            "labels": [3],
        },
    ]
    labels_path = tmp_path / "detection_labels.json"
    labels_path.write_text(json.dumps(labels_payload))

    # Bypass Data.__init__ and call DetectionFamily.load_labels directly; the
    # detection label loader has its own dedicated coverage in
    # src/raitap/data/tests/test_detection_labels.py. This test focuses on the
    # pipeline plumbing downstream of Data.
    from raitap.task_families.detection import DetectionFamily

    labels_cfg = SimpleNamespace(source=str(labels_path), kind="detection")
    load_cfg = cast(
        "AppConfig",
        SimpleNamespace(data=SimpleNamespace(labels=labels_cfg)),
    )
    data.labels = DetectionFamily().load_labels(
        load_cfg, tensor=data.tensor, sample_ids=data.sample_ids
    )

    # --- app config --------------------------------------------------------
    from raitap.configs import set_output_root
    from raitap.configs.schema import (
        AppConfig,
        DetectionMetricsConfig,
        TransparencyConfig,
    )

    transparency_cfg = TransparencyConfig(
        _target_="CaptumExplainer",
        algorithm="IntegratedGradients",
        # ``n_steps=4`` + ``internal_batch_size=1`` keep IG memory bounded so
        # the test fits comfortably on the GitHub-hosted runner (~7 GB RAM).
        # Default ``n_steps=50`` blows up CPU peak memory on a detection model.
        call={"target": 0, "n_steps": 4, "internal_batch_size": 1},
        raitap={
            "detection": {
                "score_threshold": 0.5,
                "max_boxes": 1,
                "iou_threshold": 0.5,
            },
            "batch_size": 1,
        },
        visualisers=[{"_target_": "DetectionImageVisualiser"}],
    )

    config = AppConfig(
        experiment_name="detection-e2e-test",
        metrics=DetectionMetricsConfig(),
        transparency={"ig_det": transparency_cfg},
    )
    set_output_root(config, tmp_path)

    # --- run pipeline ------------------------------------------------------
    from raitap.pipeline.orchestrator import run_without_tracking
    from raitap.types import TaskKind

    outputs = run_without_tracking(config, model, data)

    assert outputs.forward_output.task_kind is TaskKind.detection
    assert len(outputs.forward_output.as_detection()) == 1

    assert outputs.metrics is not None
    metrics_evaluation = cast("MetricsEvaluation", outputs.phase_results["metrics"])
    assert metrics_evaluation.resolved_target == "raitap.metrics.DetectionMetrics"

    # At least one detection should pass score_threshold=0.5 in dashcam frames
    # with a COCO-pretrained Faster R-CNN.
    assert len(outputs.transparency) >= 1, "expected at least one box above score threshold"

    for explanation in outputs.transparency:
        assert explanation.detection_box is not None
        assert explanation.original_sample_index is not None

    assert len([v for r in outputs.transparency for v in r.visualisations]) >= 1
