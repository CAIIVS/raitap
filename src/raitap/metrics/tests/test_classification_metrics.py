from __future__ import annotations

import pytest
import torch

from raitap.metrics import (
    BinaryClassificationMetrics,
    MulticlassClassificationMetrics,
    MultilabelClassificationMetrics,
)


class TestBinary:
    def test_perfect_predictions(self) -> None:
        m = BinaryClassificationMetrics()
        m.update(torch.tensor([1, 0, 1, 0, 1]), torch.tensor([1, 0, 1, 0, 1]))
        r = m.compute()
        assert r.scalars["accuracy"] == pytest.approx(1.0)
        assert r.scalars["precision"] == pytest.approx(1.0)
        assert r.scalars["recall"] == pytest.approx(1.0)
        assert r.scalars["f1"] == pytest.approx(1.0)

    def test_reset_clears_state(self) -> None:
        m = BinaryClassificationMetrics()
        m.update(torch.tensor([1, 0, 1, 0]), torch.tensor([1, 0, 1, 0]))
        m.reset()
        m.update(torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1]))
        assert m.compute().scalars["accuracy"] == pytest.approx(0.0)


class TestMulticlass:
    def test_requires_num_classes(self) -> None:
        with pytest.raises(TypeError):
            MulticlassClassificationMetrics()  # type: ignore[call-arg]

    def test_rejects_non_positive_num_classes(self) -> None:
        with pytest.raises(ValueError, match="num_classes must be > 0"):
            MulticlassClassificationMetrics(num_classes=0)

    def test_perfect_predictions(self) -> None:
        m = MulticlassClassificationMetrics(num_classes=3)
        m.update(torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 2, 0, 1, 2]))
        r = m.compute()
        assert r.scalars["accuracy"] == pytest.approx(1.0)

    def test_average_none_stores_artifacts(self) -> None:
        m = MulticlassClassificationMetrics(num_classes=3, average="none")
        m.update(torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 2, 0, 1, 2]))
        r = m.compute()
        assert len(r.scalars) == 0
        assert "accuracy" in r.artifacts
        assert len(r.artifacts["accuracy"]) == 3

    def test_ignore_index(self) -> None:
        m = MulticlassClassificationMetrics(num_classes=3, ignore_index=-100)
        m.update(torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, -100, 0]))
        assert m.compute().scalars["accuracy"] == pytest.approx(1.0)


class TestMultilabel:
    def test_requires_num_labels(self) -> None:
        with pytest.raises(TypeError):
            MultilabelClassificationMetrics()  # type: ignore[call-arg]

    def test_rejects_non_positive_num_labels(self) -> None:
        with pytest.raises(ValueError, match="num_labels must be > 0"):
            MultilabelClassificationMetrics(num_labels=0)

    def test_perfect_predictions(self) -> None:
        m = MultilabelClassificationMetrics(num_labels=3)
        preds = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        targets = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        m.update(preds, targets)
        assert m.compute().scalars["accuracy"] == pytest.approx(1.0)

    def test_default_threshold(self) -> None:
        m = MultilabelClassificationMetrics(num_labels=2)
        m.update(torch.tensor([[1, 0], [0, 1]]), torch.tensor([[1, 0], [0, 1]]))
        assert m.compute().scalars["accuracy"] == pytest.approx(1.0)
