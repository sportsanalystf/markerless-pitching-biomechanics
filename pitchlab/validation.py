"""Single-camera validation against marker truth.

For every OBP trial, the 3D joint centres are pushed through a virtual camera
(``pitchlab.camera``), optionally degraded with pixel noise and a lower frame
rate, and then measured with exactly the code the video pipeline uses. The
same metric computed in 3D at the marker-based event is the reference.

Two error sources are separated:

* geometry   2D metric read at the *true* event frame. Error comes only from
             projecting a 3D quantity onto the image plane.
* end-to-end 2D metric read at the *detected* event frame, with noise and a
             realistic frame rate. This is what a coach filming on a phone gets.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import data, events, metrics
from .camera import NAMED_VIEWS, Camera, project_keypoints

ALL_METRICS = list(metrics.METRICS) + list(metrics.DERIVED)
EVENT_KEYS = ("pkh", "fp", "br")


def load_clean_trials() -> List[tuple]:
    out = []
    for row, t, kp, ev in data.iter_trials():
        if events.plausible(ev):
            out.append((row, t, kp, ev))
    return out


def _prep(t, kp, cam, fps, noise_px, rng):
    k2 = project_keypoints(kp, cam)
    step = max(1, int(round(360.0 / fps)))
    t2, k2, fs = events.downsample(t, k2, fps, phase=int(rng.integers(step)))
    if noise_px > 0:
        k2 = {k: v + rng.normal(0.0, noise_px, v.shape) for k, v in k2.items()}
    k2s = events.smooth_keypoints(k2, fs)
    ctx = {"height_px": metrics.height_px_from_frame(k2s), "home_sign": cam.home_direction_px()}
    return t2, k2s, fs, ctx


def geometry_sweep(trials, azimuths: Iterable[float], noise_px: float = 0.0,
                   fps: float = 360.0, seed: int = 0) -> pd.DataFrame:
    """Image metric at the true event frame vs 3D truth, for each camera azimuth."""
    rng = np.random.default_rng(seed)
    rows = []
    for az in azimuths:
        cam = Camera(float(az))
        for row, t, kp, ev in trials:
            truth = metrics.compute(kp, events.truth_indices(t, ev), {"height_m": row["session_height_m"]}, "truth")
            t2, k2s, fs, ctx = _prep(t, kp, cam, fps, noise_px, rng)
            idx = {k: events.nearest_index(t2, ev[k]) for k in EVENT_KEYS}
            img = metrics.compute(k2s, idx, ctx, "image")
            for m in ALL_METRICS:
                rows.append((az, row.name, row["pitcher_id"], m, truth[m], img[m]))
    return pd.DataFrame(rows, columns=["azimuth", "session_pitch", "pitcher_id", "metric", "truth", "image"])


def end_to_end(trials, view: str, fps: float, noise_px: float, n_splits: int = 5,
               seed: int = 0) -> tuple:
    """Detected events + image metrics vs truth, with the release offset cross-validated by pitcher.

    Returns (metric_rows, timing_rows, final_offset_s).
    """
    rng = np.random.default_rng(seed)
    cam = Camera(NAMED_VIEWS[view])
    cache = []
    for row, t, kp, ev in trials:
        truth = metrics.compute(kp, events.truth_indices(t, ev), {"height_m": row["session_height_m"]}, "truth")
        t2, k2s, fs, ctx = _prep(t, kp, cam, fps, noise_px, rng)
        det0 = events.detect_events_2d(k2s, fs, br_offset_s=0.0, smoothed=True)
        raw_delta = ev["br"] - t2[det0["br"]] if det0["br"] >= 0 else np.nan
        cache.append(dict(row=row, t2=t2, k2s=k2s, fs=fs, ctx=ctx, ev=ev, det0=det0,
                          truth=truth, raw_delta=raw_delta))

    groups = np.array([c["row"]["pitcher_id"] for c in cache])
    deltas = np.array([c["raw_delta"] for c in cache])
    offsets = np.full(len(cache), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(deltas, groups=groups):
        offsets[te] = np.nanmedian(deltas[tr])
    final_offset = float(np.nanmedian(deltas))

    mrows, trows = [], []
    for c, off in zip(cache, offsets):
        det = dict(c["det0"])
        n = len(c["t2"])
        if det["br"] >= 0:
            det["br"] = int(np.clip(round(det["br"] + off * c["fs"]), 0, n - 1))
        img = metrics.compute(c["k2s"], det, c["ctx"], "image")
        r = c["row"]
        for m in ALL_METRICS:
            mrows.append((view, fps, noise_px, r.name, r["pitcher_id"], m, c["truth"][m], img[m]))
        for k in EVENT_KEYS:
            err = (c["t2"][det[k]] - c["ev"][k]) * 1000.0 if det[k] >= 0 else np.nan
            trows.append((view, fps, noise_px, r.name, r["pitcher_id"], k, err))
    mdf = pd.DataFrame(mrows, columns=["view", "fps", "noise_px", "session_pitch", "pitcher_id", "metric", "truth", "image"])
    tdf = pd.DataFrame(trows, columns=["view", "fps", "noise_px", "session_pitch", "pitcher_id", "event", "error_ms"])
    return mdf, tdf, final_offset


def summarise(df: pd.DataFrame, by: List[str], n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    """Bias, RMSE, MAE and Pearson r of image vs truth, optionally with a pitcher bootstrap CI on RMSE."""
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df.groupby(by):
        g = g.dropna(subset=["truth", "image"])
        e = g["image"] - g["truth"]
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(n=len(g), bias=e.mean(), mae=e.abs().mean(), rmse=float(np.sqrt((e ** 2).mean())),
                   r=float(np.corrcoef(g["truth"], g["image"])[0, 1]) if len(g) > 2 else np.nan)
        if n_boot:
            pids = g["pitcher_id"].unique()
            se = {p: (e[g["pitcher_id"] == p] ** 2).to_numpy() for p in pids}
            bs = []
            for _ in range(n_boot):
                s = rng.choice(pids, size=len(pids), replace=True)
                v = np.concatenate([se[p] for p in s])
                bs.append(np.sqrt(v.mean()))
            rec["rmse_lo"], rec["rmse_hi"] = np.percentile(bs, [2.5, 97.5])
        out.append(rec)
    return pd.DataFrame(out)


def tier(rmse: float, r: float) -> str:
    """Measurability tier: A = report, B = report with caution, C = do not report."""
    if not np.isfinite(rmse) or not np.isfinite(r):
        return "C"
    if rmse <= 5.0 and r >= 0.80:
        return "A"
    if rmse <= 10.0 and r >= 0.60:
        return "B"
    return "C"
