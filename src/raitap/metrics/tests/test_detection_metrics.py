from __future__ import annotations

from typing import Any

import pytest
import torch

from raitap.metrics import DetectionMetrics


class _DeviceAwareMapSpy:
    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def to(self, device: torch.device) -> _DeviceAwareMapSpy:
        self.devices.append(device)
        return self


class TestDetectionMetricsInitialization:
    """Test DetectionMetrics initialization with various configurations."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"box_format": "xyxy"},
            {"box_format": "xywh"},
            {"iou_type": "bbox"},
            {"iou_type": "segm"},
            {"iou_type": ("bbox", "segm")},
            {"iou_thresholds": [0.5, 0.75]},
            {"rec_thresholds": [0.0, 0.5, 1.0]},
            {"max_detection_thresholds": [1, 10, 100]},
            {"class_metrics": True},
            {"extended_summary": True},
            {"average": "macro"},
            {"average": "micro"},
            {"backend": "faster_coco_eval"},
        ],
    )
    def test_initialization(self, kwargs: dict[str, Any]) -> None:
        """Test initialization with various valid parameters."""
        # Ensure thresholds are always None unless specified to avoid mixing
        full_kwargs: dict[str, Any] = {
            "iou_thresholds": None,
            "rec_thresholds": None,
            "max_detection_thresholds": None,
        }
        full_kwargs.update(kwargs)
        metrics = DetectionMetrics(**full_kwargs)
        assert metrics.metric is not None


class TestDetectionMetricsUpdate:
    """Test DetectionMetrics update functionality."""

    @pytest.fixture
    def detection_metrics(self) -> DetectionMetrics:
        """Create a detection metrics instance."""
        return DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )

    def test_update_with_valid_data(self, detection_metrics: DetectionMetrics) -> None:
        """Test update with valid predictions and targets."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        # Should not raise any exception
        detection_metrics.update(predictions, targets)

    def test_prepare_inputs_moves_metric_state_and_nested_targets_to_prediction_device(
        self,
    ) -> None:
        detection_metrics = DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )
        metric_spy = _DeviceAwareMapSpy()
        detection_metrics.metric = metric_spy  # type: ignore[assignment]

        predictions = [
            {
                "boxes": torch.empty((0, 4), device="meta"),
                "scores": torch.empty((0,), device="meta"),
                "labels": torch.empty((0,), dtype=torch.int64, device="meta"),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        prepared_predictions, prepared_targets = detection_metrics._prepare_inputs(
            predictions, targets
        )

        assert prepared_predictions[0]["boxes"].device.type == "meta"
        assert prepared_targets[0]["boxes"].device.type == "meta"
        expected_device = torch.device("meta")
        assert metric_spy.devices == [expected_device]

    def test_update_with_multiple_images(self, detection_metrics: DetectionMetrics) -> None:
        """Test update with multiple images."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "scores": torch.tensor([0.85]),
                "labels": torch.tensor([1]),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "labels": torch.tensor([1]),
            },
        ]

        detection_metrics.update(predictions, targets)

    def test_update_with_empty_predictions(self, detection_metrics: DetectionMetrics) -> None:
        """Test update with empty predictions list."""
        predictions = [
            {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        # Should not raise any exception
        detection_metrics.update(predictions, targets)

    def test_update_with_empty_targets(self, detection_metrics: DetectionMetrics) -> None:
        """Test update with empty targets list."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]

        # Should not raise any exception
        detection_metrics.update(predictions, targets)

    def test_update_rejects_non_list_predictions(self, detection_metrics: DetectionMetrics) -> None:
        """Test that non-list predictions raise TypeError."""
        predictions = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]])}
        targets = [{"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]), "labels": torch.tensor([0])}]

        with pytest.raises(TypeError, match="Expected lists of predictions and targets"):
            detection_metrics.update(predictions, targets)

    def test_update_rejects_non_list_targets(self, detection_metrics: DetectionMetrics) -> None:
        """Test that non-list targets raise TypeError."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]), "labels": torch.tensor([0])}

        with pytest.raises(TypeError, match="Expected lists of predictions and targets"):
            detection_metrics.update(predictions, targets)

    def test_update_rejects_mismatched_lengths(self, detection_metrics: DetectionMetrics) -> None:
        """Test that mismatched prediction/target lengths raise ValueError."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "labels": torch.tensor([1]),
            },
        ]

        with pytest.raises(ValueError, match="must have the same length"):
            detection_metrics.update(predictions, targets)

    def test_multiple_updates(self, detection_metrics: DetectionMetrics) -> None:
        """Test multiple sequential updates."""
        predictions1 = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets1 = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        predictions2 = [
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "scores": torch.tensor([0.85]),
                "labels": torch.tensor([1]),
            }
        ]
        targets2 = [
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        detection_metrics.update(predictions1, targets1)
        detection_metrics.update(predictions2, targets2)


class TestDetectionMetricsCompute:
    """Test DetectionMetrics compute functionality."""

    @pytest.fixture
    def detection_metrics(self) -> DetectionMetrics:
        """Create a detection metrics instance."""
        return DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )

    def test_compute_returns_metric_result(self, detection_metrics: DetectionMetrics) -> None:
        """Test that compute returns a MetricResult object."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        detection_metrics.update(predictions, targets)
        result = detection_metrics.compute()

        assert hasattr(result, "metrics")
        assert hasattr(result, "artifacts")
        assert isinstance(result.metrics, dict)
        assert isinstance(result.artifacts, dict)

    def test_compute_separates_scalars_and_tensors(
        self, detection_metrics: DetectionMetrics
    ) -> None:
        """Test that scalar values go to metrics and tensors to artifacts."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        detection_metrics.update(predictions, targets)
        result = detection_metrics.compute()

        # Check that metrics contains only float values
        for key, value in result.metrics.items():
            assert isinstance(value, float), f"Metric '{key}' should be float, got {type(value)}"

        # Check that artifacts contains only non-scalar values (lists)
        for key, value in result.artifacts.items():
            assert isinstance(value, (list, dict)), f"Artifact '{key}' should be list or dict"

    def test_compute_perfect_predictions(self, detection_metrics: DetectionMetrics) -> None:
        """Test compute with perfect predictions."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        detection_metrics.update(predictions, targets)
        result = detection_metrics.compute()

        # With perfect predictions, mAP should be high
        assert "map" in result.metrics
        assert result.metrics["map"] >= 0.0
        assert result.metrics["map"] <= 1.0

    def test_compute_with_multiple_classes(self) -> None:
        """Test compute with multiple object classes."""
        metrics = DetectionMetrics(
            class_metrics=True,
            iou_thresholds=None,
            rec_thresholds=None,
            max_detection_thresholds=None,
        )

        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
                "scores": torch.tensor([0.9, 0.85]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
                "labels": torch.tensor([0, 1]),
            }
        ]

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "map" in result.metrics

    def test_compute_empty_predictions_and_targets(
        self, detection_metrics: DetectionMetrics
    ) -> None:
        """Test compute with completely empty predictions and targets."""
        predictions = [
            {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]
        targets = [
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]

        detection_metrics.update(predictions, targets)
        result = detection_metrics.compute()

        # Should still return a valid result structure
        assert isinstance(result.metrics, dict)
        assert isinstance(result.artifacts, dict)


class TestDetectionMetricsReset:
    """Test DetectionMetrics reset functionality."""

    @pytest.fixture
    def detection_metrics(self) -> DetectionMetrics:
        """Create a detection metrics instance."""
        return DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )

    def test_reset_clears_state(self, detection_metrics: DetectionMetrics) -> None:
        """Test that reset clears accumulated state."""
        predictions1 = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets1 = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        detection_metrics.update(predictions1, targets1)

        detection_metrics.reset()

        # After reset, update with different data
        predictions2 = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.5]),
                "labels": torch.tensor([1]),
            }
        ]
        targets2 = [
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        detection_metrics.update(predictions2, targets2)
        result2 = detection_metrics.compute()

        # Results should be independent after reset
        assert isinstance(result2.metrics, dict)
        assert isinstance(result2.artifacts, dict)


class TestDetectionMetricsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_detection_single_target(self) -> None:
        """Test with a single detection and single target."""
        metrics = DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )

        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "map" in result.metrics
        assert 0.0 <= result.metrics["map"] <= 1.0

    def test_multiple_detections_per_image(self) -> None:
        """Test with multiple detections in a single image."""
        metrics = DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )

        predictions = [
            {
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 30.0, 40.0],
                        [50.0, 60.0, 70.0, 80.0],
                        [90.0, 100.0, 110.0, 120.0],
                    ]
                ),
                "scores": torch.tensor([0.9, 0.85, 0.8]),
                "labels": torch.tensor([0, 0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 30.0, 40.0],
                        [50.0, 60.0, 70.0, 80.0],
                    ]
                ),
                "labels": torch.tensor([0, 0]),
            }
        ]

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "map" in result.metrics

    def test_xywh_box_format(self) -> None:
        """Test with xywh box format."""
        metrics = DetectionMetrics(
            box_format="xywh",
            iou_thresholds=None,
            rec_thresholds=None,
            max_detection_thresholds=None,
        )

        # xywh format: [x_center, y_center, width, height]
        predictions = [
            {
                "boxes": torch.tensor([[20.0, 30.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[20.0, 30.0, 20.0, 20.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "map" in result.metrics

    def test_large_batch(self) -> None:
        """Test with a large batch of images."""
        metrics = DetectionMetrics(
            iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None
        )

        num_images = 100
        predictions = []
        targets = []

        for i in range(num_images):
            predictions.append(
                {
                    "boxes": torch.tensor([[float(i), float(i), float(i + 10), float(i + 10)]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            )
            targets.append(
                {
                    "boxes": torch.tensor([[float(i), float(i), float(i + 10), float(i + 10)]]),
                    "labels": torch.tensor([0]),
                }
            )

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "map" in result.metrics
