"""Video -> keypoints -> events -> metrics -> gated, normed report.

Every metric is passed through two gates before it is shown:

1. Measurability (configs/measurability.json): the metric's tier at this camera
   view, measured against marker truth. Tier C is withheld with the reason.
2. Reference data (configs/norms.json): percentiles exist only for baseball
   overhand deliveries, because OpenBiomechanics is baseball fastballs.
   Softball windmill output is reported without percentiles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import events, metrics
from ..camera import NAMED_VIEWS

CFG = Path(__file__).resolve().parents[2] / "configs"
STYLES = ("overhand", "windmill")


def _load(name: str) -> dict:
    p = CFG / name
    return json.loads(p.read_text()) if p.exists() else {}


def _home_sign(kp: dict, ev: Dict[str, int]) -> float:
    a, b = ev.get("pkh", -1), ev.get("fp", -1)
    if a < 0 or b < 0:
        return 1.0
    dx = kp["lead_ankle"][b, 0] - kp["lead_ankle"][a, 0]
    return 1.0 if dx >= 0 else -1.0


def percentile_of(value: float, pct: Dict[str, float]) -> Optional[int]:
    """Approximate percentile of value against stored p5..p95 cut points."""
    if value is None or not np.isfinite(value) or not pct:
        return None
    qs = sorted((int(k[1:]), v) for k, v in pct.items() if k.startswith("p"))
    xs, ps = [v for _, v in qs], [q for q, _ in qs]
    return int(round(float(np.interp(value, xs, ps, left=ps[0], right=ps[-1]))))


def analyze_keypoints(kp2d: dict, fps: float, view: str = "arm_side", style: str = "overhand",
                      height_px: Optional[float] = None, level: str = "college",
                      smoothed: bool = False) -> dict:
    """Core analysis on canonical 2D keypoints (pixels). Shared by video and demo."""
    if view not in NAMED_VIEWS:
        raise ValueError(f"view must be one of {list(NAMED_VIEWS)}")
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    kp = kp2d if smoothed else events.smooth_keypoints(kp2d, fps)
    off = None if style == "overhand" else 0.0
    ev = events.detect_events_2d(kp, fps, view=view, br_offset_s=off, smoothed=True)
    ctx = {"height_px": height_px or metrics.height_px_from_frame(kp), "home_sign": _home_sign(kp, ev)}
    vals = metrics.compute(kp, ev, ctx, "image")

    meas = _load("measurability.json").get("views", {}).get(view, {})
    norms = _load("norms.json").get("views", {}).get(view, {}) if style == "overhand" else {}
    report = {}
    for name, v in vals.items():
        info = metrics.METRICS.get(name)
        label, unit = (info.label, info.unit) if info else metrics.DERIVED_INFO[name][:2]
        m = meas.get(name, {"tier": "C", "rmse": None, "r": None})
        entry = {"label": label, "unit": unit, "tier": m["tier"],
                 "validation_rmse": m.get("rmse"), "validation_r": m.get("r")}
        if m["tier"] == "C":
            entry.update(value=None, withheld=f"not measurable from the {view.replace('_', ' ')} view "
                                              f"(RMSE {m.get('rmse')}, r {m.get('r')} vs markers)")
        else:
            entry["value"] = round(float(v), 1) if np.isfinite(v) else None
            nm = norms.get(name, {})
            ref = nm.get(level) or nm.get("all")
            entry["percentile"] = percentile_of(entry["value"], ref) if ref else None
            entry["reference"] = (f"OBP baseball, {level if nm.get(level) else 'all levels'}"
                                  if ref else "none (no reference data for this delivery style)")
        report[name] = entry

    t = np.arange(len(kp["wrist"])) / fps
    phases = {}
    if min(ev.values()) >= 0:
        phases = {"wind_up": [0.0, t[ev["pkh"]]], "stride": [t[ev["pkh"]], t[ev["fp"]]],
                  "arm_action": [t[ev["fp"]], t[ev["br"]]], "follow_through": [t[ev["br"]], t[-1]]}
    return {"fps": fps, "view": view, "style": style,
            "events_s": {k: (round(float(t[i]), 4) if i >= 0 else None) for k, i in ev.items()},
            "events_idx": ev, "phases_s": phases, "metrics": report,
            "series": {k: v.tolist() for k, v in metrics.series(kp, ctx, "image").items()},
            "time_s": t.tolist(),
            "notes": ([] if style == "overhand" else
                      ["Windmill delivery: release offset is uncalibrated and no reference "
                       "percentiles exist; values are descriptive only."])}


def analyze_video(path: str, throws: str = "R", view: str = "arm_side", style: str = "overhand",
                  level: str = "college", model_path: Optional[str] = None) -> dict:
    from .keypoints import fill_gaps, to_pitchlab
    from .pose import estimate, read_video
    frames, fps = read_video(path)
    lm = estimate(frames, fps, model_path)
    kp = fill_gaps(to_pitchlab(lm, throws))
    missing = {k: float(np.mean(~np.isfinite(v[:, 0]))) for k, v in kp.items()}
    if max(missing.values()) > 0.2:
        raise RuntimeError(f"pose lost on >20% of frames for: "
                           f"{[k for k, m in missing.items() if m > 0.2]}")
    kp = {k: np.where(np.isfinite(v), v, np.nanmean(v, axis=0)) for k, v in kp.items()}
    out = analyze_keypoints(kp, fps, view=view, style=style, level=level)
    out["source"] = {"video": str(path), "frames": len(frames), "pose": "MediaPipe BlazePose (heavy)"}
    if fps < 100:
        out["notes"].append(f"{fps:.0f} fps: release-dependent metrics degrade below ~120 fps "
                            "(see results/end_to_end_metrics.csv).")
    return out
