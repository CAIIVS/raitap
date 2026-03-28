from __future__ import annotations

import pytest
import torch

from raitap.metrics.utils import tensor_to_python


class TestTensorToPython:
    """Test the tensor_to_python helper function."""

    def test_scalar_tensor_to_float(self) -> None:
        """Test converting a scalar tensor to float."""
        tensor = torch.tensor(3.14)
        result = tensor_to_python(tensor)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_1d_tensor_to_list(self) -> None:
        """Test converting a 1D tensor to list."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_to_python(tensor)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_2d_tensor_to_list(self) -> None:
        """Test converting a 2D tensor to nested list."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = tensor_to_python(tensor)
        assert isinstance(result, list)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_non_tensor_passthrough(self) -> None:
        """Test that non-tensor values are returned unchanged."""
        assert tensor_to_python(42) == 42
        assert tensor_to_python("string") == "string"
        assert tensor_to_python([1, 2, 3]) == [1, 2, 3]
        assert tensor_to_python(None) is None

    def test_tensor_requires_grad(self) -> None:
        """Test converting a tensor with gradient tracking."""
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = tensor_to_python(tensor)
        assert isinstance(result, list)
        assert result == [1.0, 2.0]
