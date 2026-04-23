"""Tests for global_grad_norm."""

import math

import torch
from torch import nn

from titans.observability.grad_norm import global_grad_norm


def test_returns_zero_when_no_grads() -> None:
    """Without backward(), no param.grad is attached → norm is 0."""
    model = nn.Linear(4, 4)
    assert global_grad_norm(model) == 0.0


def test_returns_l2_of_known_gradients() -> None:
    """Manually set param.grad tensors; result must equal sqrt(sum of squared norms)."""
    model = nn.Linear(3, 2, bias=True)
    # Weight: 2x3 with known values
    with torch.no_grad():
        model.weight.grad = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
        )
        model.bias.grad = torch.tensor([0.0, 0.0], dtype=torch.float32)

    # L2 norm = sqrt(1^2 + 2^2) = sqrt(5)
    expected = math.sqrt(5.0)
    assert global_grad_norm(model) == pytest_approx(expected)


def test_ignores_params_without_grad() -> None:
    """A param with grad=None must not affect the result."""
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[0].weight.grad = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    model[0].bias.grad = torch.tensor([0.0, 0.0], dtype=torch.float32)
    # model[1] params have no .grad

    expected = 5.0  # sqrt(9 + 16)
    assert global_grad_norm(model) == pytest_approx(expected)


def test_real_backward_gives_positive_norm() -> None:
    """End-to-end: loss.backward() produces a positive grad norm."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    norm = global_grad_norm(model)
    assert norm > 0
    assert math.isfinite(norm)


def pytest_approx(value: float, rel: float = 1e-6) -> float:  # noqa: PT002
    """Shim so tests don't need to import pytest.approx in every assert."""
    import pytest

    return pytest.approx(value, rel=rel)
