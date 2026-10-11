"""Vector geometry shared by the 3D (marker) and 2D (pixel) code paths.

Every function accepts arrays of shape (..., d) with d = 2 or 3 and broadcasts
over leading axes, so the same call measures a single frame, a time series, or
a batch of trials.

2D convention: pixel coordinates with image y pointing *down*.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Included angle at b between segments b->a and b->c, in degrees (0..180)."""
    u, v = _unit(a - b), _unit(c - b)
    cos = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Flexion at b: 0 deg when a-b-c is straight, increasing as the joint bends."""
    return 180.0 - joint_angle(a, b, c)


def line_acute_angle(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Unsigned acute angle between line p1-p2 and line q1-q2, in degrees (0..90)."""
    u, v = _unit(p2 - p1), _unit(q2 - q1)
    cos = np.abs(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0))
    return np.degrees(np.arccos(cos))


def tilt_from_vertical_2d(lower: np.ndarray, upper: np.ndarray, forward_sign: float = 1.0) -> np.ndarray:
    """Signed tilt of segment lower->upper from image vertical, degrees.

    Positive means the upper point leans toward ``forward_sign`` along image x.
    Image y points down, so 'up' is -y.
    """
    d = upper - lower
    return np.degrees(np.arctan2(forward_sign * d[..., 0], -d[..., 1]))


def elevation_above_horizontal_2d(proximal: np.ndarray, distal: np.ndarray) -> np.ndarray:
    """Angle of segment proximal->distal above image horizontal, degrees (-90..90)."""
    d = distal - proximal
    return np.degrees(np.arctan2(-d[..., 1], np.abs(d[..., 0])))


def project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Remove the component of 3D vectors v along a plane normal."""
    n = _unit(np.asarray(normal, float))
    return v - np.sum(v * n, axis=-1, keepdims=True) * n
