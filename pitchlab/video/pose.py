"""MediaPipe BlazePose wrapper returning 33 landmarks per frame in pixels.

Supports the MediaPipe Tasks API (mediapipe >= 0.10.x, needs a .task model
file) and the legacy ``solutions.pose`` API where it still exists. The model
is downloaded once to ~/.cache/pitchlab if no path is given.

Other estimators (RTMPose, ViTPose) can be added by returning the same
(T, 33, 3) array of x_px, y_px, visibility.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def _model_path(path: Optional[str]) -> str:
    if path:
        return path
    cache = Path(os.environ.get("PITCHLAB_CACHE", Path.home() / ".cache" / "pitchlab"))
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / "pose_landmarker_heavy.task"
    if not dst.exists():
        urllib.request.urlretrieve(MODEL_URL, dst)
    return str(dst)


def read_video(path: str) -> Tuple[np.ndarray, float]:
    """Return (frames as list-like array of BGR images, fps)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, float(fps)


def estimate(frames, fps: float, model_path: Optional[str] = None) -> np.ndarray:
    """Run BlazePose on every frame. Returns (T, 33, 3): x_px, y_px, visibility (NaN if no person)."""
    import cv2
    T = len(frames)
    out = np.full((T, 33, 3), np.nan)
    if T == 0:
        return out
    h, w = frames[0].shape[:2]
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=_model_path(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.3, min_tracking_confidence=0.3)
        with vision.PoseLandmarker.create_from_options(opts) as lm:
            for i, f in enumerate(frames):
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                res = lm.detect_for_video(img, int(round(1000.0 * i / fps)))
                if res.pose_landmarks:
                    for j, p in enumerate(res.pose_landmarks[0]):
                        out[i, j] = (p.x * w, p.y * h, getattr(p, "visibility", 1.0) or 1.0)
        return out
    except (ImportError, AttributeError):
        pass
    from mediapipe.python.solutions import pose as mp_pose  # legacy API
    with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.3) as pose:
        for i, f in enumerate(frames):
            res = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                for j, p in enumerate(res.pose_landmarks.landmark):
                    out[i, j] = (p.x * w, p.y * h, p.visibility)
    return out
