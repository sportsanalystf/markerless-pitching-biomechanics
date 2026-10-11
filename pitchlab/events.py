"""Phase events: read from OBP (truth) or detected from 2D keypoints (video).

Phases reported:
    wind-up         start -> peak knee height (PKH)
    stride          PKH -> foot plant (FP)
    arm action      FP -> ball release (BR)
    follow-through  BR -> end

Maximum external rotation cannot be located reliably from one camera, so arm
cocking and acceleration are reported together as "arm action".

Detectors (2D only, after low-pass filtering), anchored on release because
the wrist-speed peak is the most stable landmark in the delivery:

    PKH  lead knee at its highest image position
    BR   peak wrist speed after PKH, shifted by a view-specific offset that is
         calibrated on OBP marker data (``configs/event_offsets.json``)
    FP   inside [PKH, BR - 30 ms]: first frame after the lead ankle's peak
         speed at which its speed drops below FP_SPEED_FRAC of that peak

FP is compared against OBP's 100%-body-weight foot plant, the moment the foot
is visibly planted. OBP's 10% "foot contact" is a force event that a camera
does not see.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt

FP_SPEED_FRAC = 0.15
BR_GUARD_S = 0.03
_OFFSETS = Path(__file__).resolve().parents[1] / "configs" / "event_offsets.json"


def lowpass(x: np.ndarray, fs: float, cutoff: float = 12.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0; cutoff clipped below Nyquist."""
    nyq = 0.5 * fs
    wn = min(cutoff, 0.8 * nyq) / nyq
    b, a = butter(order, wn)
    if x.shape[0] <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x, axis=0)


def smooth_keypoints(kp: dict, fs: float, cutoff: float = 12.0) -> dict:
    return {k: lowpass(v, fs, cutoff) for k, v in kp.items()}


def truth_indices(t: np.ndarray, ev: Dict[str, float]) -> Dict[str, int]:
    return {k: (int(np.argmin(np.abs(t - v))) if np.isfinite(v) else -1) for k, v in ev.items()}


def plausible(ev: Dict[str, float]) -> bool:
    """Reject OBP trials whose event labels are physiologically implausible.

    Pre-registered bounds, set from pitching physiology rather than tuned to
    the detector: foot plant to release 80-250 ms, knee lift to foot plant
    0.30-1.00 s. Eighteen of 411 trials fail (BR before FP, FP before PKH, or
    a missing force-plate event).
    """
    fp, br, pkh = ev.get("fp", np.nan), ev.get("br", np.nan), ev.get("pkh", np.nan)
    if not np.all(np.isfinite([fp, br, pkh])):
        return False
    return 0.08 <= br - fp <= 0.25 and 0.30 <= fp - pkh <= 1.00


def _speed(p: np.ndarray, fs: float) -> np.ndarray:
    return np.linalg.norm(np.gradient(p, axis=0) * fs, axis=-1)


def load_offsets() -> Dict[str, float]:
    if _OFFSETS.exists():
        return json.loads(_OFFSETS.read_text())["br_offset_s"]
    return {}


def br_offset_for(view: str) -> float:
    return float(load_offsets().get(view, 0.0))


def raw_release_peak(kp: dict, fs: float, pkh: int) -> int:
    return pkh + int(np.argmax(_speed(kp["wrist"], fs)[pkh:]))


def detect_events_2d(kp2d: dict, fs: float, view: str = "arm_side",
                     br_offset_s: Optional[float] = None, smoothed: bool = False) -> Dict[str, int]:
    """Detect PKH, FP and BR frame indices from 2D keypoints sampled at fs Hz."""
    kp = kp2d if smoothed else smooth_keypoints(kp2d, fs)
    n = len(kp["lead_knee"])
    pkh = int(np.argmin(kp["lead_knee"][:, 1]))
    if pkh >= n - 3:
        return {"pkh": pkh, "fp": -1, "br": -1}

    peak = raw_release_peak(kp, fs, pkh)
    off = br_offset_for(view) if br_offset_s is None else br_offset_s
    br = int(np.clip(round(peak + off * fs), 0, n - 1))

    hi = max(pkh + 2, peak - int(round(BR_GUARD_S * fs)))
    sp = _speed(kp["lead_ankle"], fs)
    pk = pkh + int(np.argmax(sp[pkh:hi]))
    below = np.where(sp[pk:hi] < FP_SPEED_FRAC * sp[pk])[0]
    fp = int(pk + below[0]) if below.size else int(hi)
    return {"pkh": pkh, "fp": fp, "br": br}


def downsample(t: np.ndarray, kp: dict, fps: float, phase: int = 0) -> tuple:
    """Simulate a camera at ``fps`` by taking every k-th frame of a 360 Hz trial."""
    step = max(1, int(round(360.0 / fps)))
    sl = slice(phase % step, None, step)
    return t[sl], {k: v[sl] for k, v in kp.items()}, 360.0 / step


def nearest_index(t_sub: np.ndarray, t_event: float) -> int:
    return int(np.argmin(np.abs(t_sub - t_event))) if np.isfinite(t_event) else -1
