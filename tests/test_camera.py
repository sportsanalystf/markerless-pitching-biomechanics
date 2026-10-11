import numpy as np
import pytest

from pitchlab.camera import NAMED_VIEWS, Camera, project_keypoints


def test_target_projects_to_image_centre():
    for az in NAMED_VIEWS.values():
        cam = Camera(az)
        uv = cam.project(np.asarray(cam.target))
        assert np.allclose(uv, [cam.width / 2, cam.height / 2], atol=1e-6)


def test_higher_points_appear_higher_in_image():
    cam = Camera(NAMED_VIEWS["arm_side"])
    lo = cam.project(np.asarray(cam.target))
    hi = cam.project(np.asarray(cam.target) + [0, 0, 0.5])
    assert hi[1] < lo[1]  # image y points down


@pytest.mark.parametrize("view,expected", [("arm_side", None), ("glove_side", None)])
def test_side_views_see_home_plate_left_or_right(view, expected):
    s = Camera(NAMED_VIEWS[view]).home_direction_px()
    assert s in (-1.0, 1.0)


def test_arm_and_glove_side_are_mirror_images():
    assert Camera(NAMED_VIEWS["arm_side"]).home_direction_px() == -Camera(NAMED_VIEWS["glove_side"]).home_direction_px()


def test_project_keypoints_keeps_shapes():
    T = 30
    kp = {"wrist": np.random.default_rng(0).normal(size=(T, 3)) * 0.2 + [1.0, -0.3, 1.0]}
    out = project_keypoints(kp, Camera())
    assert out["wrist"].shape == (T, 2)
    assert np.all(np.isfinite(out["wrist"]))
