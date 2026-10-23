import numpy as np
import pytest
import torch

from data import JOINT_LIMIT, fk_numpy
from model import IKModel, decode_angles, fk_torch


@pytest.mark.parametrize("shape", [(3,), (8, 3), (2, 4, 3)])
def test_model_output_shapes_and_joint_pairs(shape):
    model = IKModel()
    output = model(torch.zeros(shape))
    assert output.shape == (*shape[:-1], 6)
    assert torch.isfinite(output).all()
    pairs = output.reshape(*shape[:-1], 3, 2)
    torch.testing.assert_close(torch.linalg.vector_norm(pairs, dim=-1), torch.ones(shape[:-1] + (3,)))
    angles = decode_angles(output)
    assert angles.shape == shape
    assert torch.all(torch.abs(angles) <= JOINT_LIMIT + 1e-6)


def test_differentiable_forward_kinematics_matches_numpy():
    q = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, -0.7, 1.2], [-2.0, 1.0, -2.0]],
        dtype=torch.float64,
    )
    actual = fk_torch(q)
    assert actual.shape == q.shape
    np.testing.assert_allclose(actual.detach().numpy(), fk_numpy(q.numpy()), atol=1e-12)


def test_forward_kinematics_gradients():
    # Keep orientation away from its discontinuity at +/- pi.
    q = torch.tensor([[0.3, -0.7, 1.2]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fk_torch, (q,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_model_loss_propagates_to_parameters():
    model = IKModel()
    target = torch.tensor([[0.45, 0.15, 0.2], [0.20, -0.30, -0.5]])
    prediction = fk_torch(decode_angles(model(target)))
    (prediction[:, :2] - target[:, :2]).square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
