"""Study B: how much error does a single camera add?

1. Geometry sweep: 24 camera azimuths, 360 fps, no noise and 4 px noise.
   Image metric at the *true* event vs 3D truth.
2. End-to-end: the four named views x {360, 240, 120, 60} fps, 4 px noise,
   detected events, release offset cross-validated by pitcher (GroupKFold).
3. Measurability tiers per metric and view (120 fps, 4 px), written to
   configs/measurability.json and used by the video pipeline to gate output.
4. Release offsets per view and frame rate, written to configs/event_offsets.json.

Run:  python scripts/03_projection_study.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab import metrics, validation  # noqa: E402
from pitchlab.camera import NAMED_VIEWS  # noqa: E402

RES = ROOT / "results"
CFG = ROOT / "configs"
NOISE_PX = 4.0
TIER_FPS = 120


def main() -> None:
    t0 = time.time()
    trials = validation.load_clean_trials()
    n_pitchers = len({r["pitcher_id"] for r, *_ in trials})
    print(f"{len(trials)} clean trials from {n_pitchers} pitchers")

    # 1. geometry sweep
    az = list(range(0, 360, 15))
    geo = []
    for noise in (0.0, NOISE_PX):
        g = validation.geometry_sweep(trials, az, noise_px=noise, fps=360.0, seed=1)
        g["noise_px"] = noise
        geo.append(g)
    geo = pd.concat(geo, ignore_index=True)
    geo_sum = validation.summarise(geo, ["noise_px", "azimuth", "metric"])
    geo_sum.to_csv(RES / "geometry_by_azimuth.csv", index=False)
    print(f"geometry sweep done ({time.time() - t0:.0f}s)")

    # 2. end-to-end
    mets, tims, offsets = [], [], {}
    for view in NAMED_VIEWS:
        offsets[view] = {}
        for fps in (360, 240, 120, 60):
            m, t, off = validation.end_to_end(trials, view, fps, NOISE_PX, seed=2)
            mets.append(m)
            tims.append(t)
            offsets[view][str(fps)] = round(off, 5)
        print(f"  end-to-end {view} done ({time.time() - t0:.0f}s)")
    mets = pd.concat(mets, ignore_index=True)
    tims = pd.concat(tims, ignore_index=True)
    e2e = validation.summarise(mets, ["view", "fps", "metric"])
    e2e_boot = validation.summarise(mets[mets.fps == TIER_FPS], ["view", "metric"], n_boot=1000, seed=3)
    e2e.to_csv(RES / "end_to_end_metrics.csv", index=False)
    e2e_boot.to_csv(RES / "end_to_end_metrics_120fps_ci.csv", index=False)

    tsum = (tims.dropna().groupby(["view", "fps", "event"]).error_ms
            .agg(n="size", median="median", mae=lambda x: x.abs().mean(),
                 p90=lambda x: np.percentile(np.abs(x), 90)).reset_index())
    tsum.to_csv(RES / "event_timing.csv", index=False)

    # 3. tiers
    tiers = {}
    for view in NAMED_VIEWS:
        sub = e2e[(e2e.view == view) & (e2e.fps == TIER_FPS)].set_index("metric")
        tiers[view] = {m: {"tier": validation.tier(sub.loc[m, "rmse"], sub.loc[m, "r"]),
                           "rmse": round(float(sub.loc[m, "rmse"]), 2),
                           "r": round(float(sub.loc[m, "r"]), 3)}
                       for m in validation.ALL_METRICS}
    (CFG / "measurability.json").write_text(json.dumps({
        "basis": f"OBP marker truth, {len(trials)} trials / {n_pitchers} pitchers, "
                 f"{TIER_FPS} fps, {NOISE_PX:.0f} px keypoint noise, detected events",
        "rule": "A: RMSE<=5 and r>=0.80; B: RMSE<=10 and r>=0.60; else C (not reported)",
        "views": tiers}, indent=2))

    # norms: distribution of the *image* metric (what video produces) at 120 fps,
    # for tier A/B metrics only, overall and by playing level (n >= 20 pitches)
    level = {r.name: r["playing_level"] for r, *_ in trials}
    nm = mets[mets.fps == TIER_FPS].copy()
    nm["level"] = nm["session_pitch"].map(level)
    qs = (5, 10, 25, 50, 75, 90, 95)
    norms = {}
    for view in NAMED_VIEWS:
        norms[view] = {}
        for m in validation.ALL_METRICS:
            if tiers[view][m]["tier"] == "C":
                continue
            g = nm[(nm.view == view) & (nm.metric == m)].dropna(subset=["image"])
            entry = {"all": {f"p{q}": round(float(np.percentile(g.image, q)), 2) for q in qs}}
            for lv, gl in g.groupby("level"):
                if len(gl) >= 20:
                    entry[lv] = {f"p{q}": round(float(np.percentile(gl.image, q)), 2) for q in qs}
            entry["n"] = int(len(g))
            norms[view][m] = entry
    (CFG / "norms.json").write_text(json.dumps({
        "basis": f"image-space metrics from OBP trials projected through each view at {TIER_FPS} fps "
                 f"with {NOISE_PX:.0f} px noise and detected events; baseball fastballs only",
        "views": norms}, indent=2))

    # 4. release offsets: one per view, taken at the frame rate the pipeline sees
    (CFG / "event_offsets.json").write_text(json.dumps({
        "note": "seconds added to the peak-wrist-speed frame to estimate ball release; "
                "median over OBP trials (GroupKFold-validated), by view and frame rate",
        "by_view_fps": offsets,
        "br_offset_s": {v: offsets[v][str(TIER_FPS)] for v in offsets}}, indent=2))

    # headline numbers
    arm = e2e_boot[e2e_boot.view == "arm_side"].set_index("metric")
    tm = tsum[(tsum.view == "arm_side") & (tsum.fps == TIER_FPS)].set_index("event")
    summary = {
        "n_trials": len(trials), "n_pitchers": n_pitchers, "noise_px": NOISE_PX, "fps": TIER_FPS,
        "arm_side_120fps": {m: {k: round(float(arm.loc[m, k]), 2) for k in ("rmse", "rmse_lo", "rmse_hi", "bias", "r")}
                            for m in validation.ALL_METRICS},
        "arm_side_event_mae_ms": {e: round(float(tm.loc[e, "mae"]), 1) for e in tm.index},
        "tiers": {v: {m: tiers[v][m]["tier"] for m in tiers[v]} for v in tiers},
        "runtime_s": round(time.time() - t0, 1),
    }
    (RES / "projection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
