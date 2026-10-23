"""Normalized 3 -> 256 -> 256 -> 256 -> 6 inverse kinematics network."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import JOINT_LIMIT, LINK_LENGTHS, ROOT

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "model.pt"


class IKModel(nn.Module):
    """Inputs: (x metres, y metres, wrapped phi radians).

    Outputs: (sin q1, cos q1, sin q2, cos q2, sin q3, cos q3).
    Training-set normalization is stored in the state dict, never fit on test data.
    """

    def __init__(self, input_mean=None, input_std=None):
        super().__init__()
        mean = torch.zeros(3) if input_mean is None else torch.as_tensor(input_mean, dtype=torch.float32)
        std = torch.ones(3) if input_std is None else torch.as_tensor(input_std, dtype=torch.float32)
        self.register_buffer("input_mean", mean.clone())
        self.register_buffer("input_std", std.clone().clamp_min(1e-6))
        self.net = nn.Sequential(
            nn.Linear(3, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 6),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0., 1., 0., 1., 0., 1.]))

    def forward(self, poses):
        raw = self.net((poses - self.input_mean) / self.input_std)
        pairs = raw.reshape(*raw.shape[:-1], 3, 2)
        return F.normalize(pairs, dim=-1, eps=1e-8).reshape_as(raw)


def decode_angles(outputs, enforce_limits=True):
    """Decode interleaved sin/cos pairs; bound all returned joint commands."""
    pairs = outputs.reshape(*outputs.shape[:-1], 3, 2)
    angles = torch.atan2(pairs[..., 0], pairs[..., 1])
    return angles.clamp(-JOINT_LIMIT, JOINT_LIMIT) if enforce_limits else angles


def fk_torch(angles):
    """Differentiable batched FK; phi is wrapped to [-pi, pi]."""
    cumulative = torch.cumsum(angles, dim=-1)
    lengths = angles.new_tensor(LINK_LENGTHS)
    x = (lengths * torch.cos(cumulative)).sum(dim=-1)
    y = (lengths * torch.sin(cumulative)).sum(dim=-1)
    phi = torch.atan2(torch.sin(cumulative[..., -1]), torch.cos(cumulative[..., -1]))
    return torch.stack((x, y, phi), dim=-1)


def ik_loss(outputs, targets, target_angles, angle_weight=1e-4):
    """Metres squared + circular orientation/joint errors + limit penalty.

    Orientation weight has units m^2; 0.01 corresponds to a 10 cm length
    scale. Small joint supervision breaks branch ambiguity without strongly
    penalizing another valid IK solution. Limit penalty retains gradients for
    raw predictions outside the allowed command interval.
    """
    raw_angles = decode_angles(outputs, enforce_limits=False)
    angles = raw_angles.clamp(-JOINT_LIMIT, JOINT_LIMIT)
    predicted = fk_torch(angles)
    position = (predicted[..., :2] - targets[..., :2]).square().sum(dim=-1).mean()
    orientation = (1 - torch.cos(predicted[..., 2] - targets[..., 2])).mean()
    joint = (1 - torch.cos(angles - target_angles)).mean()
    limits = F.relu(raw_angles.abs() - JOINT_LIMIT).square().mean()
    return position + 0.01 * orientation + angle_weight * joint + 0.01 * limits


def load_model(checkpoint=DEFAULT_CHECKPOINT):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}. Run python train.py first.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = IKModel()
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model


@torch.inference_mode()
def predict_angles(model, poses):
    """CPU prediction, including normalization, pair decoding and joint bounds."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape[-1] != 3 or not np.isfinite(poses).all():
        raise ValueError("poses must have final dimension 3 and finite values")
    poses = poses.copy()
    poses[..., 2] = np.arctan2(np.sin(poses[..., 2]), np.cos(poses[..., 2]))
    return decode_angles(model(torch.from_numpy(poses))).numpy()
