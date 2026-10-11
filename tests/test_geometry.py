import numpy as np

from pitchlab import geometry as g


def test_flexion_straight_and_right_angle():
    a, b = np.array([0.0, 1.0]), np.array([0.0, 0.0])
    assert np.isclose(g.flexion(a, b, np.array([0.0, -1.0])), 0.0)
    assert np.isclose(g.flexion(a, b, np.array([1.0, 0.0])), 90.0)


def test_joint_angle_broadcasts_over_time():
    T = 50
    th = np.linspace(0, np.pi, T)
    a = np.tile([1.0, 0.0, 0.0], (T, 1))
    b = np.zeros((T, 3))
    c = np.stack([np.cos(th), np.sin(th), np.zeros(T)], axis=1)
    assert np.allclose(g.joint_angle(a, b, c), np.degrees(th), atol=1e-6)


def test_line_acute_angle_is_symmetric_and_bounded():
    p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    q1, q2 = np.array([0.0, 0.0]), np.array([-1.0, 1.0])
    assert np.isclose(g.line_acute_angle(p1, p2, q1, q2), 45.0)
    assert np.isclose(g.line_acute_angle(p1, p2, q2, q1), 45.0)


def test_tilt_from_vertical_sign_follows_forward_direction():
    lower, upper = np.array([100.0, 500.0]), np.array([110.0, 490.0])  # up and to the right (y down)
    assert np.isclose(g.tilt_from_vertical_2d(lower, upper, +1.0), 45.0)
    assert np.isclose(g.tilt_from_vertical_2d(lower, upper, -1.0), -45.0)


def test_elevation_above_horizontal():
    assert np.isclose(g.elevation_above_horizontal_2d(np.array([0.0, 0.0]), np.array([1.0, -1.0])), 45.0)
    assert np.isclose(g.elevation_above_horizontal_2d(np.array([0.0, 0.0]), np.array([-1.0, 1.0])), -45.0)


def test_project_to_plane_removes_normal_component():
    v = np.array([1.0, 2.0, 3.0])
    out = g.project_to_plane(v, np.array([0.0, 0.0, 2.0]))
    assert np.allclose(out, [1.0, 2.0, 0.0])
