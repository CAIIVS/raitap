from __future__ import annotations

import pytest
import torch

from src.raitap.metrics.classification_tm import ClassificationMetrics
from src.raitap.metrics.utils import tensor_to_python


class TestTensorToPython:
    """Test the _tensor_to_python helper function."""

    def test_scalar_tensor_to_float(self):
        """Test converting a scalar tensor to float."""
        tensor = torch.tensor(3.14)
        result = tensor_to_python(tensor)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_1d_tensor_to_list(self):
        """Test converting a 1D tensor to list."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_to_python(tensor)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_2d_tensor_to_list(self):
        """Test converting a 2D tensor to nested list."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = tensor_to_python(tensor)
        assert isinstance(result, list)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_non_tensor_passthrough(self):
        """Test that non-tensor values are returned unchanged."""
        assert tensor_to_python(42) == 42
        assert tensor_to_python("string") == "string"
        assert tensor_to_python([1, 2, 3]) == [1, 2, 3]
        assert tensor_to_python(None) is None

    def test_tensor_requires_grad(self):
        """Test converting a tensor with gradient tracking."""
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = tensor_to_python(tensor)
        assert isinstance(result, list)
        assert result == [1.0, 2.0]


class TestClassificationMetricsInitialization:
    """Test ClassificationMetrics initialization and validation."""

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            ClassificationMetrics(
                task="invalid", num_classes=3, num_labels=None, average="macro", ignore_index=None
            )

    def test_multiclass_without_num_classes_raises_error(self):
        """Test that multiclass without num_classes raises ValueError."""
        with pytest.raises(ValueError, match="must provide num_classes > 0"):
            ClassificationMetrics(
                task="multiclass",
                num_classes=None,
                num_labels=None,
                average="macro",
                ignore_index=None,
            )

    def test_multiclass_with_zero_classes_raises_error(self):
        """Test that multiclass with num_classes=0 raises ValueError."""
        with pytest.raises(ValueError, match="must provide num_classes > 0"):
            ClassificationMetrics(
                task="multiclass",
                num_classes=0,
                num_labels=None,
                average="macro",
                ignore_index=None,
            )

    def test_multilabel_without_num_labels_raises_error(self):
        """Test that multilabel without num_labels raises ValueError."""
        with pytest.raises(ValueError, match="provide num_labels"):
            ClassificationMetrics(
                task="multilabel",
                num_classes=None,
                num_labels=None,
                average="macro",
                ignore_index=None,
            )

    def test_multilabel_with_zero_labels_raises_error(self):
        """Test that multilabel with num_labels=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_labels must be > 0"):
            ClassificationMetrics(
                task="multilabel",
                num_classes=None,
                num_labels=0,
                average="macro",
                ignore_index=None,
            )

    def test_multilabel_accepts_num_classes_as_alias(self):
        """Test that multilabel accepts num_classes as alias for num_labels."""
        metrics = ClassificationMetrics(
            task="multilabel", num_classes=3, num_labels=None, average="macro", ignore_index=None
        )
        assert metrics.task == "multilabel"

    def test_binary_initialization(self):
        """Test successful binary classification initialization."""
        metrics = ClassificationMetrics(
            task="binary", num_classes=None, num_labels=None, average="macro", ignore_index=None
        )
        assert metrics.task == "binary"
        assert metrics.average == "macro"

    def test_multiclass_initialization(self):
        """Test successful multiclass initialization."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=5, num_labels=None, average="weighted", ignore_index=-100
        )
        assert metrics.task == "multiclass"
        assert metrics.average == "weighted"

    def test_multilabel_initialization(self):
        """Test successful multilabel initialization."""
        metrics = ClassificationMetrics(
            task="multilabel", num_classes=None, num_labels=4, average="micro", ignore_index=None
        )
        assert metrics.task == "multilabel"
        assert metrics.average == "micro"


class TestBinaryClassification:
    """Test ClassificationMetrics for binary classification."""

    @pytest.fixture
    def binary_metrics(self):
        """Create a binary classification metrics instance."""
        return ClassificationMetrics(
            task="binary", num_classes=None, num_labels=None, average="macro", ignore_index=None
        )

    def test_perfect_predictions(self, binary_metrics):
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

    def test_imperfect_predictions(self, binary_metrics):
        """Test metrics with some errors."""
        predictions = torch.tensor([1, 0, 1, 1, 0])
        targets = torch.tensor([1, 0, 1, 0, 1])

        binary_metrics.update(predictions, targets)
        result = binary_metrics.compute()

        assert 0.0 < result.metrics["accuracy"] < 1.0
        assert 0.0 < result.metrics["precision"] < 1.0
        assert 0.0 < result.metrics["recall"] < 1.0
        assert 0.0 < result.metrics["f1"] < 1.0

    def test_reset_clears_state(self, binary_metrics):
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

    def test_multiple_updates(self, binary_metrics):
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
    def multiclass_metrics(self):
        """Create a multiclass metrics instance."""
        return ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="macro", ignore_index=None
        )

    def test_perfect_predictions(self, multiclass_metrics):
        """Test metrics with perfect predictions."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])

        multiclass_metrics.update(predictions, targets)
        result = multiclass_metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["f1"] == pytest.approx(1.0)

    def test_weighted_average(self):
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

    def test_micro_average(self):
        """Test with micro average."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="micro", ignore_index=None
        )

        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 0, 1])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)

    def test_none_average_stores_artifacts(self):
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

    def test_ignore_index(self):
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
    def multilabel_metrics(self):
        """Create a multilabel metrics instance."""
        return ClassificationMetrics(
            task="multilabel",
            num_classes=None,
            num_labels=3,
            average="macro",
            ignore_index=None,
            threshold=0.5,
        )

    def test_perfect_predictions(self, multilabel_metrics):
        """Test metrics with perfect predictions."""
        predictions = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        targets = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])

        multilabel_metrics.update(predictions, targets)
        result = multilabel_metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["f1"] == pytest.approx(1.0)

    def test_none_average_stores_artifacts(self):
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

    def test_default_threshold(self):
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


class TestMetricResultStructure:
    """Test the structure of MetricResult objects."""

    def test_result_has_required_fields(self):
        """Test that compute returns a properly structured MetricResult."""
        metrics = ClassificationMetrics(
            task="binary", num_classes=None, num_labels=None, average="macro", ignore_index=None
        )

        predictions = torch.tensor([1, 0, 1])
        targets = torch.tensor([1, 0, 1])

        metrics.update(predictions, targets)
        result = metrics.compute()

        # Check that result has the expected attributes
        assert hasattr(result, "metrics")
        assert hasattr(result, "artifacts")
        assert isinstance(result.metrics, dict)
        assert isinstance(result.artifacts, dict)

    def test_metrics_are_tensors_before_conversion(self):
        """Test that metric values can be tensors (handled by torchmetrics)."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=2, num_labels=None, average="macro", ignore_index=None
        )

        predictions = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])

        metrics.update(predictions, targets)
        result = metrics.compute()

        # Verify all metrics are present
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1" in result.metrics


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_sample(self):
        """Test with a single sample."""
        metrics = ClassificationMetrics(
            task="binary", num_classes=None, num_labels=None, average="macro", ignore_index=None
        )

        predictions = torch.tensor([1])
        targets = torch.tensor([1])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)

    def test_all_same_class(self):
        """Test when all predictions and targets are the same class."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=3, num_labels=None, average="macro", ignore_index=None
        )

        predictions = torch.tensor([0, 0, 0, 0])
        targets = torch.tensor([0, 0, 0, 0])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)

    def test_large_batch(self):
        """Test with a large batch size."""
        metrics = ClassificationMetrics(
            task="multiclass", num_classes=10, num_labels=None, average="macro", ignore_index=None
        )

        predictions = torch.randint(0, 10, (1000,))
        targets = predictions.clone()  # Perfect predictions

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert result.metrics["accuracy"] == pytest.approx(1.0)
