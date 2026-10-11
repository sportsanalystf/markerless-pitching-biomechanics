"""Study C: random split vs pitcher-out evaluation of a BiLSTM velocity model.

Models, each scored under both splits (5 folds, 3 seeds for the network):
  * BiLSTM on joint-angle sequences (PKH -> BR)
  * Ridge on the 76 numeric OBP point-of-interest metrics (fixed alpha, no tuning)
  * Pitcher memory: mean speed of the same pitcher's training pitches

Hyper-parameters are fixed in advance (pitchlab/sequence.py) so nothing is
tuned on a test fold.

Run:  python scripts/04_sequence_leakage.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab import data, sequence, validation  # noqa: E402

RES = ROOT / "results"
SEEDS = (0, 1, 2)


def main() -> None:
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    trials = validation.load_clean_trials()
    X, y, g, chans = sequence.build_sequences(data.load_joint_angles(), trials)
    sp = [r.name for r, *_ in trials]
    poi = data.trials().set_index("session_pitch").loc[sp]
    drop = {"session", "p_throws", "pitch_type", "pitch_speed_mph", "pitcher_id", "playing_level",
            "elbow_varus_moment_bwht", "session_mass_kg", "session_height_m", "age_yrs"}
    pcols = [c for c in poi.columns if c not in drop and np.issubdtype(poi[c].dtype, np.number)]
    P = poi[pcols].to_numpy(float)
    P = np.where(np.isfinite(P), P, np.nanmedian(P, axis=0))
    print(f"X {X.shape}, {len(set(g))} pitchers, POI features {P.shape[1]}")

    splits = {
        "random": lambda s: KFold(5, shuffle=True, random_state=s).split(X),
        "pitcher_out": lambda s: GroupKFold(5).split(X, groups=g),
    }
    rows = []
    for split, gen in splits.items():
        for seed in SEEDS:
            preds = {k: np.full(len(y), np.nan) for k in ("bilstm", "ridge_poi", "pitcher_memory")}
            for tr, te in gen(seed):
                preds["bilstm"][te] = sequence.fit_predict_bilstm(X[tr], y[tr], X[te], seed=seed)
                preds["ridge_poi"][te] = sequence.fit_predict_ridge(P[tr], y[tr], P[te])
                preds["pitcher_memory"][te] = sequence.pitcher_memory(y[tr], g[tr], g[te])
            for model, p in preds.items():
                m = sequence._metrics(y, p)
                rows.append(dict(split=split, seed=seed, model=model, **m))
            print(f"  {split} seed {seed} done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sequence_leakage_runs.csv", index=False)
    s = df.groupby(["model", "split"]).agg(mae=("mae", "mean"), mae_sd=("mae", "std"),
                                           r2=("r2", "mean"), r2_sd=("r2", "std")).reset_index()
    s.to_csv(RES / "sequence_leakage.csv", index=False)
    out = {"n_pitches": int(len(y)), "n_pitchers": int(len(set(g))), "target": "pitch_speed_mph",
           "target_sd_mph": round(float(y.std()), 2), "input": "45 joint-angle channels, PKH->BR, 64 steps",
           "results": {f"{r.model}|{r.split}": {"mae": round(r.mae, 2), "r2": round(r.r2, 3)}
                       for r in s.itertuples()},
           "runtime_s": round(time.time() - t0, 1)}
    (RES / "sequence_summary.json").write_text(json.dumps(out, indent=2))
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
