"""Study A: which mechanics go with velocity, and which with elbow load?

Unit of analysis is the pitcher (n = 100), not the pitch: each pitcher's 2-5
trials are averaged first, so a pitcher with five pitches does not count five
times. For each pre-registered mechanic:

    outcome ~ mechanic_z + body_mass_z + height_z      (pitcher means, OLS)

Effect = change in outcome per 1 SD of the mechanic, holding body size fixed.
95% CI from 2,000 pitcher bootstrap resamples; p from 5,000 permutations of
the mechanic across pitchers; Benjamini-Hochberg across all 30 tests.

Outcomes: pitch speed (mph) and peak elbow varus moment normalised by body
weight x height (%BW*Ht), the usual injury-load proxy for the UCL.

Run:  python scripts/02_biomech_drivers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab import data  # noqa: E402

RES = ROOT / "results"

CANDIDATES = {
    "max_shoulder_external_rotation": "Layback (max shoulder ER)",
    "rotation_hip_shoulder_separation_fp": "Hip-shoulder separation at FP",
    "max_rotation_hip_shoulder_separation": "Peak hip-shoulder separation",
    "stride_length": "Stride length",
    "lead_knee_extension_from_fp_to_br": "Lead knee extension FP to BR",
    "torso_anterior_tilt_br": "Trunk forward tilt at BR",
    "torso_lateral_tilt_br": "Trunk lateral tilt at BR",
    "max_torso_rotational_velo": "Peak trunk rotation velocity",
    "max_pelvis_rotational_velo": "Peak pelvis rotation velocity",
    "max_cog_velo_x": "Peak centre-of-mass velocity to home",
    "arm_slot": "Arm slot",
    "elbow_flexion_fp": "Elbow flexion at FP",
    "shoulder_abduction_fp": "Shoulder abduction at FP",
    "shoulder_horizontal_abduction_fp": "Shoulder horizontal abduction at FP",
    "timing_peak_torso_to_peak_pelvis_rot_velo": "Pelvis-to-trunk peak timing",
}
OUTCOMES = {"pitch_speed_mph": "Pitch speed (mph)",
            "elbow_varus_moment_bwht": "Elbow varus moment (%BW*Ht)"}


def _z(x):
    return (x - x.mean()) / x.std(ddof=0)


def _beta(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0][1]


def bh(p):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main(n_boot: int = 2000, n_perm: int = 5000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = data.trials()
    cols = list(CANDIDATES) + list(OUTCOMES) + ["session_mass_kg", "session_height_m"]
    pm = df.groupby("pitcher_id")[cols].mean().dropna()
    n = len(pm)
    size = np.column_stack([_z(pm["session_mass_kg"]), _z(pm["session_height_m"])])

    rows = []
    for out, out_lbl in OUTCOMES.items():
        y = pm[out].to_numpy()
        for c, lbl in CANDIDATES.items():
            x = _z(pm[c]).to_numpy()
            X = np.column_stack([x, size])
            b = _beta(y, X)
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, n, n)
                xb = _z(pd.Series(pm[c].to_numpy()[i])).to_numpy()
                sb = np.column_stack([_z(pd.Series(pm["session_mass_kg"].to_numpy()[i])),
                                      _z(pd.Series(pm["session_height_m"].to_numpy()[i]))])
                boots.append(_beta(y[i], np.column_stack([xb, sb])))
            perm = np.array([_beta(y, np.column_stack([rng.permutation(x), size])) for _ in range(n_perm)])
            p = (1 + np.sum(np.abs(perm) >= abs(b))) / (n_perm + 1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(outcome=out, outcome_label=out_lbl, mechanic=c, label=lbl,
                             effect_per_sd=b, ci_lo=lo, ci_hi=hi, p_perm=p))
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res["p_perm"].to_numpy())
    res = res.sort_values(["outcome", "q_bh"])
    res.to_csv(RES / "biomech_drivers.csv", index=False)

    # velocity vs elbow load across pitchers
    r = np.corrcoef(pm["pitch_speed_mph"], pm["elbow_varus_moment_bwht"])[0, 1]
    r_abs = np.corrcoef(pm["pitch_speed_mph"], df.groupby("pitcher_id")["elbow_varus_moment"].mean().loc[pm.index])[0, 1]
    rb = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rb.append(np.corrcoef(pm["pitch_speed_mph"].to_numpy()[i], pm["elbow_varus_moment_bwht"].to_numpy()[i])[0, 1])

    sig = res[res.q_bh < 0.05]
    summary = {
        "n_pitchers": n, "n_pitches": int(len(df)),
        "velocity_mph_mean_sd": [round(pm.pitch_speed_mph.mean(), 1), round(pm.pitch_speed_mph.std(), 1)],
        "elbow_load_bwht_mean_sd": [round(pm.elbow_varus_moment_bwht.mean(), 2), round(pm.elbow_varus_moment_bwht.std(), 2)],
        "r_velocity_vs_normalised_elbow_load": [round(r, 2)] + [round(v, 2) for v in np.percentile(rb, [2.5, 97.5])],
        "r_velocity_vs_absolute_elbow_load_Nm": round(r_abs, 2),
        "significant_after_BH": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rr.items()
             if k in ("outcome", "label", "effect_per_sd", "ci_lo", "ci_hi", "q_bh")}
            for rr in sig.to_dict("records")],
    }
    (RES / "biomech_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(res[["outcome", "label", "effect_per_sd", "ci_lo", "ci_hi", "p_perm", "q_bh"]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
