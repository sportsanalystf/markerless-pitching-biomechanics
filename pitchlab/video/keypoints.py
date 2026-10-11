"""Map BlazePose-33 landmarks to PitchLab's canonical keypoints.

Throwing side = the pitcher's throwing hand; lead side = the other side.
The hand centre is the mean of wrist, index and pinky landmarks.
"""

from __future__ import annotations

import numpy as np

BLAZE = {
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}


def to_pitchlab(lm: np.ndarray, throws: str, min_vis: float = 0.3) -> dict:
    """lm: (T, 33, 3) -> {keypoint: (T, 2)} pixel arrays, low-visibility points set to NaN."""
    th, ld = ("r", "l") if throws.upper() == "R" else ("l", "r")
    xy = lm[..., :2].copy()
    xy[lm[..., 2] < min_vis] = np.nan

    def p(name):
        return xy[:, BLAZE[name]]

    def hand(s):
        return np.nanmean(np.stack([p(f"{s}_wrist"), p(f"{s}_index"), p(f"{s}_pinky")]), axis=0)

    return {
        "shoulder": p(f"{th}_shoulder"), "elbow": p(f"{th}_elbow"), "wrist": p(f"{th}_wrist"),
        "hand": hand(th),
        "glove_shoulder": p(f"{ld}_shoulder"), "glove_elbow": p(f"{ld}_elbow"), "glove_hand": hand(ld),
        "rear_hip": p(f"{th}_hip"), "lead_hip": p(f"{ld}_hip"),
        "rear_knee": p(f"{th}_knee"), "lead_knee": p(f"{ld}_knee"),
        "rear_ankle": p(f"{th}_ankle"), "lead_ankle": p(f"{ld}_ankle"),
    }


def fill_gaps(kp: dict, max_gap: int = 6) -> dict:
    """Linearly interpolate short NaN gaps; longer gaps stay NaN."""
    out = {}
    for k, v in kp.items():
        v = v.copy()
        for d in range(v.shape[1]):
            x = v[:, d]
            bad = ~np.isfinite(x)
            if bad.all() or not bad.any():
                continue
            idx = np.arange(len(x))
            filled = np.interp(idx, idx[~bad], x[~bad])
            # restore long gaps
            run_start = None
            for i in range(len(x) + 1):
                b = i < len(x) and bad[i]
                if b and run_start is None:
                    run_start = i
                elif not b and run_start is not None:
                    if i - run_start <= max_gap:
                        x[run_start:i] = filled[run_start:i]
                    run_start = None
            v[:, d] = x
        out[k] = v
    return out
