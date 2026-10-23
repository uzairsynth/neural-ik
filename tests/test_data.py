import numpy as np
import pytest

from data import JOINT_LIMIT, fk_numpy, generate_dataset, load_dataset, save_dataset


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ([0.0, 0.0, 0.0], [0.70, 0.0, 0.0]),
        ([np.pi / 2, 0.0, 0.0], [0.0, 0.70, np.pi / 2]),
        ([0.0, np.pi / 2, -np.pi / 2], [0.45, 0.25, 0.0]),
        ([0.0, np.pi, 0.0], [-0.10, 0.0, -np.pi]),
    ],
)
def test_forward_kinematics_known_poses(angles, expected):
    np.testing.assert_allclose(fk_numpy(angles), expected, atol=1e-12)


def test_forward_kinematics_batches_and_orientation_wrapping():
    angles = np.zeros((2, 4, 3), dtype=np.float64)
    angles[..., 2] = 3 * np.pi / 2
    poses = fk_numpy(angles)
    assert poses.shape == (2, 4, 3)
    np.testing.assert_allclose(poses[..., 2], -np.pi / 2)


@pytest.mark.parametrize("angles", [0.0, [1.0, 2.0], np.zeros((2, 4))])
def test_forward_kinematics_rejects_invalid_shapes(angles):
    with pytest.raises(ValueError, match="shape"):
        fk_numpy(angles)


def test_seeded_dataset_has_disjoint_correct_splits():
    first = generate_dataset(samples=100, seed=42)
    repeat = generate_dataset(samples=100, seed=42)
    different = generate_dataset(samples=100, seed=43)
    for split, expected_count in (("train", 80), ("val", 10), ("test", 10)):
        q = first[f"q_{split}"]
        pose = first[f"pose_{split}"]
        assert q.shape == pose.shape == (expected_count, 3)
        assert q.dtype == pose.dtype == np.float32
        assert np.all(np.abs(q) <= JOINT_LIMIT)
        np.testing.assert_allclose(pose, fk_numpy(q), atol=2e-7)
        np.testing.assert_array_equal(q, repeat[f"q_{split}"])
        assert not np.array_equal(q, different[f"q_{split}"])
    combined = np.concatenate([first[f"q_{split}"] for split in ("train", "val", "test")])
    assert len(np.unique(combined, axis=0)) == 100


def test_dataset_round_trip(tmp_path):
    path = tmp_path / "nested" / "small.npz"
    generated = save_dataset(path, samples=30, seed=7)
    loaded = load_dataset(path)
    assert loaded.keys() == generated.keys()
    for key in generated:
        np.testing.assert_array_equal(loaded[key], generated[key])


def test_missing_dataset_error_explains_generation(tmp_path):
    with pytest.raises(FileNotFoundError, match="python data.py"):
        load_dataset(tmp_path / "missing.npz")
