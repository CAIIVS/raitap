from __future__ import annotations

from typing import Any

import pytest
import torch

from raitap.metrics import ClassificationMetrics


class TestClassificationMetricsInitialization:
    """Test ClassificationMetrics initialization and validation."""

    @pytest.mark.parametrize(
        ("task", "num_classes", "num_labels", "match"),
        [
            ("invalid", 3, None, "Unknown task"),
            ("multiclass", None, None, "must provide num_classes > 0"),
            ("multiclass", 0, None, "must provide num_classes > 0"),
            ("multilabel", None, None, "provide num_labels"),
            ("multilabel", None, 0, "num_labels must be > 0"),
        ],
    )
    def test_initialization_errors(
        self, task: Any, num_classes: Any, num_labels: Any, match: str
    ) -> None:
        """Test that invalid configurations raise appropriate errors."""
        with pytest.raises(ValueError, match=match):
            ClassificationMetrics(
                task=task,
                num_classes=num_classes,
                num_labels=num_labels,
                average="macro",
                ignore_index=None,
            )

    @pytest.mark.parametrize(
        ("task", "num_classes", "num_labels", "average"),
        [
            ("multilabel", 3, None, "macro"),
            ("binary", None, None, "macro"),
            ("multiclass", 5, None, "weighted"),
            ("multilabel", None, 4, "micro"),
        ],
    )
    def test_successful_initialization(
        self, task: Any, num_classes: Any, num_labels: Any, average: Any
    ) -> None:
        """Test successful initialization for different tasks."""
        metrics = ClassificationMetrics(
            task=task,
            num_classes=num_classes,
            num_labels=num_labels,
            average=average,
            ignore_index=None,
        )
        assert metrics.task == task
        assert metrics.average == average


class TestBinaryClassification:
    """Test ClassificationMetrics for binary classification."""

    @pytest.fixture
    def binary_metrics(self) -> ClassificationMetrics:
        """Create a binary classification metrics instance."""
        return ClassificationMetrics(
            task="binary", num_classes=None, num_labels=None, average="macro", ignore_index=None
        )

    def test_perfect_predictions(self, binary_metrics: ClassificationMetrics) -> None:
        """Test metrics with perfect predictions."""
        predictions = torch.tensor([1, 0, 1, 0, 1])
        targets = torch.tensor([1, 0, 1, 0, 1])

        binary_metrics.update(predictions, targets)
        result = binary_metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["f1"] == pytest.approx(1.0)
        assert len(result.artifacts) == 0

    def test_imperfect_predictions(self, binary_metrics: ClassificationMetrics) -> None:
        """Test metrics with some errors."""
        predictions = torch.tensor([1, 0, 1, 1, 0])
        targets = torch.tensor([1, 0, 1, 0, 1])

        binary_metrics.update(predictions, targets)
        result = binary_metrics.compute()

        assert 0.0 < result.metrics["accuracy"] < 1.0
        assert 0.0 < result.metrics["precision"] < 1.0
        assert 0.0 < result.metrics["recall"] < 1.0
        assert 0.0 < result.metrics["f1"] < 1.0

    def test_reset_clears_state(self, binary_metrics: ClassificationMetrics) -> None:
        """Test that reset clears accumulated state."""
        predictions = torch.tensor([1, 0, 1, 0])
        targets = torch.tensor([1, 0, 1, 0])

        binary_metrics.update(predictions, targets)
        binary_metrics.reset()

        # After reset, compute should give different results
        predictions_new = torch.tensor([0, 0, 0, 0])
        targets_new = torch.tensor([1, 1, 1, 1])

        binary_metrics.update(predictions_new, targets_new)
        result = binary_metrics.compute()

        # All wrong predictions
        assert result.metrics["accuracy"] == pytest.approx(0.0)
        assert result.metrics["recall"] == pytest.approx(0.0)

    def test_multiple_updates(self, binary_metrics: ClassificationMetrics) -> None:
        """Test that multiple updates accumulate correctly."""
        # First batch
        binary_metrics.update(torch.tensor([1, 1]), torch.tensor([1, 1]))
        # Second batch
        binary_metrics.update(torch.tensor([0, 0]), torch.tensor([0, 0]))

        result = binary_metrics.compute()
        assert result.metrics["accuracy"] == pytest.approx(1.0)


class TestMulticlassClassification:
    """Test ClassificationMetrics for multiclass classification."""

    @pytest.fixture
    def multiclass_metrics(self) -> ClassificationMetrics:
        """Create a multiclass metrics instance."""
        return ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="macro", ignore_index=None
        )

    def test_perfect_predictions(self, multiclass_metrics: ClassificationMetrics) -> None:
        """Test metrics with perfect predictions."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])

        multiclass_metrics.update(predictions, targets)
        result = multiclass_metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["f1"] == pytest.approx(1.0)

    def test_weighted_average(self) -> None:
        """Test with weighted average."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="weighted", ignore_index=None
        )

        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)

    def test_micro_average(self) -> None:
        """Test with micro average."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="micro", ignore_index=None
        )

        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 0, 1])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)

    def test_none_average_stores_artifacts(self) -> None:
        """Test that average='none' stores per-class metrics in artifacts."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="none", ignore_index=None
        )

        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])

        metrics.update(predictions, targets)
        result = metrics.compute()

        # With average='none', metrics should be in artifacts, not metrics dict
        assert len(result.metrics) == 0
        assert "accuracy" in result.artifacts
        assert "precision" in result.artifacts
        assert "recall" in result.artifacts
        assert "f1" in result.artifacts

        # Each should be a list (per-class values)
        assert isinstance(result.artifacts["accuracy"], list)
        assert len(result.artifacts["accuracy"]) == 3

    def test_ignore_index(self) -> None:
        """Test that ignore_index works correctly."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="macro", ignore_index=-100
        )

        # Include some samples with ignore_index
        predictions = torch.tensor([0, 1, 2, 0])
        targets = torch.tensor([0, 1, -100, 0])  # Third sample should be ignored

        metrics.update(predictions, targets)
        result = metrics.compute()

        # Should only consider the 3 valid samples
        assert result.metrics["accuracy"] == pytest.approx(1.0)


class TestMultilabelClassification:
    """Test ClassificationMetrics for multilabel classification."""

    @pytest.fixture
    def multilabel_metrics(self) -> ClassificationMetrics:
        """Create a multilabel metrics instance."""
        return ClassificationMetrics(
            task="multilabel",
            num_classes=None,
            num_labels=3,
            average="macro",
            ignore_index=None,
            threshold=0.5,
        )

    def test_perfect_predictions(self, multilabel_metrics: ClassificationMetrics) -> None:
        """Test metrics with perfect predictions."""
        predictions = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        targets = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])

        multilabel_metrics.update(predictions, targets)
        result = multilabel_metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["f1"] == pytest.approx(1.0)

    def test_none_average_stores_artifacts(self) -> None:
        """Test that average='none' stores per-label metrics in artifacts."""
        metrics = ClassificationMetrics(
            task="multilabel", num_classes=None, num_labels=3, average="none", ignore_index=None
        )

        predictions = torch.tensor([[1, 0, 1], [0, 1, 0]])
        targets = torch.tensor([[1, 0, 1], [0, 1, 0]])

        metrics.update(predictions, targets)
        result = metrics.compute()

        # With average='none', metrics should be in artifacts
        assert len(result.metrics) == 0
        assert "accuracy" in result.artifacts
        assert "precision" in result.artifacts
        assert "recall" in result.artifacts
        assert "f1" in result.artifacts

        # Each should be a list (per-label values)
        assert isinstance(result.artifacts["precision"], list)
        assert len(result.artifacts["precision"]) == 3

    def test_default_threshold(self) -> None:
        """Test that default threshold of 0.5 is applied."""
        # Don't pass threshold explicitly
        metrics = ClassificationMetrics(
            task="multilabel", num_classes=None, num_labels=2, average="macro", ignore_index=None
        )

        # Binary predictions work with default threshold
        predictions = torch.tensor([[1, 0], [0, 1]])
        targets = torch.tensor([[1, 0], [0, 1]])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
