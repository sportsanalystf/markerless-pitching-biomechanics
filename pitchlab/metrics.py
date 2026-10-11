"""Pitching metrics, each defined twice: once in 3D (truth) and once in 2D (pixels).

Both definitions use the same joint centres and the same event, so the
difference between them is the measurement error that a single camera adds.
That difference is what ``scripts/03_projection_study.py`` measures.

Trunk axis = mid-hip -> mid-shoulder, the only trunk line a 2D pose model
can see. Every pitcher is mirrored to a right-hander (throwing arm on -y).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .geometry import (elevation_above_horizontal_2d, flexion, line_acute_angle,
                       project_to_plane, tilt_from_vertical_2d)

# shoulder height ~0.818 H and ankle height ~0.039 H (Drillis & Contini, 1966)
ANKLE_TO_SHOULDER_FRAC = 0.818 - 0.039

X, Y, Z = np.eye(3)


def _mid(a, b):
    return 0.5 * (a + b)


def _trunk(kp):
    return _mid(kp["rear_hip"], kp["lead_hip"]), _mid(kp["shoulder"], kp["glove_shoulder"])


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str
    event: str        # event at which the metric is read ("fp" or "br")
    plane: str        # anatomical plane the quantity lives in
    truth: Callable   # (kp3d at frames, ctx) -> value
    image: Callable   # (kp2d at frames, ctx) -> value


# ---- 3D truth definitions -------------------------------------------------

def _t_elbow(kp, ctx):
    return flexion(kp["shoulder"], kp["elbow"], kp["wrist"])


def _t_knee(kp, ctx):
    return flexion(kp["lead_hip"], kp["lead_knee"], kp["lead_ankle"])


def _t_trunk_fwd(kp, ctx):
    lo, hi = _trunk(kp)
    v = project_to_plane(hi - lo, Y)               # sagittal plane (x-z)
    return np.degrees(np.arctan2(v[..., 0], v[..., 2]))


def _t_trunk_lat(kp, ctx):
    lo, hi = _trunk(kp)
    v = project_to_plane(hi - lo, X)               # frontal plane (y-z)
    return np.abs(np.degrees(np.arctan2(v[..., 1], v[..., 2])))


def _t_arm_slot(kp, ctx):
    v = project_to_plane(kp["hand"] - kp["elbow"], X)
    return np.degrees(np.arctan2(v[..., 2], np.abs(v[..., 1])))


def _t_hss(kp, ctx):
    p = project_to_plane(kp["lead_hip"] - kp["rear_hip"], Z)
    s = project_to_plane(kp["glove_shoulder"] - kp["shoulder"], Z)
    z = np.zeros_like(p)
    return line_acute_angle(z, p, z, s)


def _t_stride(kp, ctx):
    d = project_to_plane(kp["lead_ankle"] - kp["rear_ankle"], Z)
    return 100.0 * np.linalg.norm(d, axis=-1) / ctx["height_m"]


# ---- 2D image definitions -------------------------------------------------

def _i_elbow(kp, ctx):
    return flexion(kp["shoulder"], kp["elbow"], kp["wrist"])


def _i_knee(kp, ctx):
    return flexion(kp["lead_hip"], kp["lead_knee"], kp["lead_ankle"])


def _i_trunk_fwd(kp, ctx):
    lo, hi = _trunk(kp)
    return tilt_from_vertical_2d(lo, hi, forward_sign=ctx["home_sign"])


def _i_trunk_lat(kp, ctx):
    lo, hi = _trunk(kp)
    return np.abs(tilt_from_vertical_2d(lo, hi))


def _i_arm_slot(kp, ctx):
    return elevation_above_horizontal_2d(kp["elbow"], kp["hand"])


def _i_hss(kp, ctx):
    return line_acute_angle(kp["rear_hip"], kp["lead_hip"], kp["shoulder"], kp["glove_shoulder"])


def _i_stride(kp, ctx):
    d = np.linalg.norm(kp["lead_ankle"] - kp["rear_ankle"], axis=-1)
    return 100.0 * d / ctx["height_px"]


METRICS: Dict[str, Metric] = {m.name: m for m in [
    Metric("elbow_flexion_fp", "Elbow flexion at foot plant", "deg", "fp", "3D", _t_elbow, _i_elbow),
    Metric("elbow_flexion_br", "Elbow flexion at release", "deg", "br", "3D", _t_elbow, _i_elbow),
    Metric("lead_knee_flexion_fp", "Lead knee flexion at foot plant", "deg", "fp", "sagittal", _t_knee, _i_knee),
    Metric("lead_knee_flexion_br", "Lead knee flexion at release", "deg", "br", "sagittal", _t_knee, _i_knee),
    Metric("trunk_forward_tilt_br", "Trunk forward tilt at release", "deg", "br", "sagittal", _t_trunk_fwd, _i_trunk_fwd),
    Metric("trunk_lateral_tilt_br", "Trunk lateral tilt at release", "deg", "br", "frontal", _t_trunk_lat, _i_trunk_lat),
    Metric("arm_slot_br", "Arm slot at release", "deg", "br", "frontal", _t_arm_slot, _i_arm_slot),
    Metric("hip_shoulder_sep_fp", "Hip-shoulder separation at foot plant", "deg", "fp", "transverse", _t_hss, _i_hss),
    Metric("stride_length_fp", "Stride length at foot plant", "% height", "fp", "transverse", _t_stride, _i_stride),
]}

# derived: lead knee extension from foot plant to release (positive = bracing)
DERIVED = {"lead_knee_extension_fp_br": ("lead_knee_flexion_fp", "lead_knee_flexion_br")}
DERIVED_INFO = {"lead_knee_extension_fp_br": ("Lead knee extension, foot plant to release", "deg", "sagittal")}


def _at(kp: dict, i: int) -> dict:
    return {k: v[i] for k, v in kp.items()}


def compute(kp: dict, idx: Dict[str, int], ctx: dict, mode: str) -> Dict[str, float]:
    """Compute every metric at the given event frame indices.

    mode = "truth" (3D keypoints) or "image" (2D keypoints).
    ctx needs height_m (truth) or height_px and home_sign (image).
    """
    out: Dict[str, float] = {}
    for m in METRICS.values():
        i = idx.get(m.event)
        if i is None or i < 0:
            out[m.name] = np.nan
            continue
        fn = m.truth if mode == "truth" else m.image
        out[m.name] = float(fn(_at(kp, i), ctx))
    for name, (a, b) in DERIVED.items():
        out[name] = out[a] - out[b]
    return out


def height_px_from_frame(kp2d: dict, frame: int = 0) -> float:
    """Pixel height estimate from the set position: rear ankle to throwing shoulder."""
    d = np.linalg.norm(kp2d["shoulder"][frame] - kp2d["rear_ankle"][frame])
    return float(d / ANKLE_TO_SHOULDER_FRAC)


def series(kp: dict, ctx: dict, mode: str = "image") -> Dict[str, np.ndarray]:
    """Time series used by the dashboard."""
    if mode == "image":
        return {
            "elbow_flexion": _i_elbow(kp, ctx),
            "lead_knee_flexion": _i_knee(kp, ctx),
            "trunk_forward_tilt": _i_trunk_fwd(kp, ctx),
        }
    return {
        "elbow_flexion": _t_elbow(kp, ctx),
        "lead_knee_flexion": _t_knee(kp, ctx),
        "trunk_forward_tilt": _t_trunk_fwd(kp, ctx),
    }
