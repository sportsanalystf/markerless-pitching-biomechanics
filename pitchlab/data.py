"""Loaders for the OpenBiomechanics (OBP) baseball pitching release.

OBP data are CC BY-NC-SA 4.0 and are not redistributed here. ``scripts/01_get_data.sh``
downloads them and checks them against Driveline's published SHA-256 sums.

Coordinate frame (OBP): +x second base -> home plate, +y toward first base,
+z up, metres. For a pitcher facing home, +y is his left. Left-handed trials
are mirrored (y -> -y) so every trial is expressed as a right-hander, which
puts the throwing arm on -y (third-base side) for all 411 trials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# canonical keypoint name -> OBP landmark column prefix
OBP_KEYPOINTS: Dict[str, str] = {
    "shoulder": "shoulder_jc",
    "elbow": "elbow_jc",
    "wrist": "wrist_jc",
    "hand": "hand_jc",
    "glove_shoulder": "glove_shoulder_jc",
    "glove_elbow": "glove_elbow_jc",
    "glove_hand": "glove_hand_jc",
    "rear_hip": "rear_hip",
    "lead_hip": "lead_hip",
    "rear_knee": "rear_knee_jc",
    "lead_knee": "lead_knee_jc",
    "rear_ankle": "rear_ankle_jc",
    "lead_ankle": "lead_ankle_jc",
    "com": "centerofmass",
}

EVENT_COLS = {
    "pkh": "pkh_time",      # peak knee height
    "fc": "fp_10_time",     # foot contact: lead-foot force crosses 10% body weight
    "fp": "fp_100_time",    # foot plant: lead-foot force crosses 100% body weight
    "mer": "MER_time",      # maximum shoulder external rotation (layback)
    "br": "BR_time",        # ball release
    "mir": "MIR_time",      # maximum shoulder internal rotation
}

FS_HZ = 360.0  # OBP marker sampling rate


def data_dir() -> Path:
    """Locate the OBP data folder: $PITCHLAB_DATA, ./data/obp, or ../data/obp."""
    candidates = []
    if os.environ.get("PITCHLAB_DATA"):
        candidates.append(Path(os.environ["PITCHLAB_DATA"]))
    candidates += [REPO_ROOT / "data" / "obp", REPO_ROOT.parent / "data" / "obp"]
    for c in candidates:
        if (c / "poi_metrics.csv").exists():
            return c
    raise FileNotFoundError(
        "OBP data not found. Run scripts/01_get_data.sh or set PITCHLAB_DATA. "
        f"Looked in: {[str(c) for c in candidates]}"
    )


def _read(name: str) -> pd.DataFrame:
    d = data_dir()
    for fn in (d / name, d / name.replace(".csv", ".zip")):
        if fn.exists():
            return pd.read_csv(fn)
    raise FileNotFoundError(f"{name} not found in {d}")


@lru_cache(maxsize=1)
def load_poi() -> pd.DataFrame:
    return _read("poi_metrics.csv")


@lru_cache(maxsize=1)
def load_metadata() -> pd.DataFrame:
    return _read("metadata.csv")


@lru_cache(maxsize=1)
def trials() -> pd.DataFrame:
    """One row per pitch: POI metrics joined to metadata, with a pitcher id."""
    poi = load_poi()
    meta = load_metadata()[
        ["session_pitch", "user", "session_mass_kg", "session_height_m", "age_yrs", "playing_level"]
    ]
    df = poi.merge(meta, on="session_pitch", how="left", validate="one_to_one")
    df = df.rename(columns={"user": "pitcher_id"})
    # elbow varus moment normalised by body weight x height, in %BW*Ht
    g = 9.81
    df["elbow_varus_moment_bwht"] = (
        100.0 * df["elbow_varus_moment"] / (df["session_mass_kg"] * g * df["session_height_m"])
    )
    return df


@lru_cache(maxsize=1)
def load_landmarks() -> pd.DataFrame:
    d = data_dir()
    for fn in (d / "landmarks.csv", d / "lm" / "landmarks.csv", d / "pitching_landmarks.zip"):
        if fn.exists():
            return pd.read_csv(fn)
    raise FileNotFoundError("pitching landmarks not found; run scripts/01_get_data.sh")


@lru_cache(maxsize=1)
def load_joint_angles() -> pd.DataFrame:
    d = data_dir()
    for fn in (d / "joint_angles.csv", d / "ja" / "joint_angles.csv", d / "pitching_joint_angles.zip"):
        if fn.exists():
            return pd.read_csv(fn)
    raise FileNotFoundError("pitching joint angles not found; run scripts/01_get_data.sh")


def mirror_left(arr: np.ndarray, throws: str) -> np.ndarray:
    """Mirror a (..., 3) array of lab coordinates for a left-hander (y -> -y)."""
    if throws == "L":
        arr = arr.copy()
        arr[..., 1] *= -1.0
    return arr


def trial_keypoints(lm: pd.DataFrame, throws: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """Return (time, {keypoint: (T,3)}, {event: time}) for one trial's landmark rows."""
    lm = lm.sort_values("time")
    t = lm["time"].to_numpy(float)
    kp = {}
    for name, pref in OBP_KEYPOINTS.items():
        xyz = lm[[f"{pref}_x", f"{pref}_y", f"{pref}_z"]].to_numpy(float)
        kp[name] = mirror_left(xyz, throws)
    first = lm.iloc[0]
    ev = {k: float(first[c]) if pd.notna(first[c]) else np.nan for k, c in EVENT_COLS.items()}
    return t, kp, ev


def iter_trials() -> Iterator[Tuple[pd.Series, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]]:
    """Yield (trial_row, time, keypoints, events) for every pitch with landmarks."""
    tr = trials().set_index("session_pitch")
    lm = load_landmarks()
    for sp, g in lm.groupby("session_pitch", sort=True):
        if sp not in tr.index:
            continue
        row = tr.loc[sp]
        t, kp, ev = trial_keypoints(g, row["p_throws"])
        yield row, t, kp, ev
