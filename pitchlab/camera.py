"""A pinhole camera placed around the pitcher, used to turn 3D markers into pixels.

Azimuth convention (lab frame, after mirroring every pitcher to right-handed):

    0   deg  behind home plate, looking at the pitcher's face   ("catcher")
    90  deg  first-base side, the glove side                     ("glove_side")
    180 deg  behind the pitcher, toward center field              ("center_field")
    270 deg  third-base side, the throwing-arm side               ("arm_side")

The default rig is a sideline phone on a tripod: 10 m from the delivery,
lens at 1.2 m, 1920x1080, with a focal length that frames a 2.5 m tall
volume in about 70% of the image height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NAMED_VIEWS = {"catcher": 0.0, "glove_side": 90.0, "center_field": 180.0, "arm_side": 270.0}


@dataclass
class Camera:
    azimuth_deg: float = 270.0
    distance_m: float = 10.0
    height_m: float = 1.2
    target: np.ndarray = field(default_factory=lambda: np.array([1.0, -0.3, 1.0]))
    width: int = 1920
    height: int = 1080
    f_px: float = 3024.0  # 0.7 * 1080 px * 10 m / 2.5 m

    def __post_init__(self) -> None:
        a = np.radians(self.azimuth_deg)
        tgt = np.asarray(self.target, float)
        self.center = np.array([tgt[0] + self.distance_m * np.cos(a),
                                tgt[1] + self.distance_m * np.sin(a),
                                self.height_m])
        fwd = tgt - self.center
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        # rows map world offsets to camera (x right, y down, z forward)
        self.R = np.stack([right, down, fwd])

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project (..., 3) world points to (..., 2) pixels (image y down)."""
        Xc = (np.asarray(X, float) - self.center) @ self.R.T
        z = Xc[..., 2:3]
        uv = self.f_px * Xc[..., :2] / z
        return uv + np.array([self.width / 2.0, self.height / 2.0])

    def home_direction_px(self) -> float:
        """+1 if home plate (+x in the lab) appears to the right in the image, else -1."""
        p0 = self.project(np.asarray(self.target, float))
        p1 = self.project(np.asarray(self.target, float) + np.array([0.5, 0.0, 0.0]))
        dx = p1[0] - p0[0]
        return 1.0 if dx >= 0 else -1.0


def project_keypoints(kp3d: dict, cam: Camera) -> dict:
    return {k: cam.project(v) for k, v in kp3d.items()}
