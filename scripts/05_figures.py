"""Make every figure in results/figures from the CSV/JSON outputs of scripts 02-04,
plus a demo dashboard from one OBP trial pushed through the arm-side camera.

Run:  python scripts/05_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab import events, metrics, report, validation  # noqa: E402
from pitchlab.camera import NAMED_VIEWS, Camera, project_keypoints  # noqa: E402
from pitchlab.report import GRID, INK, INK2, MUTED, SERIES, SURFACE, plt, style  # noqa: E402
from pitchlab.video.pipeline import analyze_keypoints  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
LABEL = {**{k: m.label for k, m in metrics.METRICS.items()},
         **{k: v[0] for k, v in metrics.DERIVED_INFO.items()}}


def fig_view_error() -> None:
    g = pd.read_csv(RES / "geometry_by_azimuth.csv")
    g = g[g.noise_px == g.noise_px.max()]
    ms = validation.ALL_METRICS
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), facecolor=SURFACE, sharex=True)
    for ax, m in zip(axes.flat, ms):
        style(ax)
        d = g[g.metric == m].sort_values("azimuth")
        ax.plot(d.azimuth, d.rmse, color=SERIES[0], lw=2, marker="o", ms=4)
        ax.axhline(5, color=MUTED, lw=1, ls="--")
        ax.set_title(LABEL[m], fontsize=10, color=INK, loc="left", wrap=True)
        ax.set_xticks(list(NAMED_VIEWS.values()))
        ax.set_xticklabels(["catcher", "glove", "CF", "arm"], fontsize=8)
        ax.set_ylim(0, None)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE vs markers", color=INK2)
    fig.suptitle("Geometry error by camera position (true event frames, 4 px keypoint noise). Dashed line = 5.",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig1_error_by_view.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_frame_rate() -> None:
    t = pd.read_csv(RES / "event_timing.csv")
    t = t[t.view == "arm_side"]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style(ax)
    names = {"pkh": "Peak knee height", "fp": "Foot plant", "br": "Ball release"}
    for i, e in enumerate(("fp", "pkh", "br")):
        d = t[t.event == e].sort_values("fps")
        ax.plot(d.fps, d.mae, color=SERIES[i], lw=2, marker="o", ms=7)
        ax.text(d.fps.iloc[-1] + 8, d.mae.iloc[-1], f"{names[e]}  {d.mae.iloc[-1]:.0f} ms", va="center",
                fontsize=10, color=INK)
    ax.set_xticks([60, 120, 240, 360])
    ax.set_xlim(40, 470)
    ax.set_ylim(0, None)
    ax.set_xlabel("Camera frame rate (fps)", color=INK2)
    ax.set_ylabel("Mean absolute timing error (ms)", color=INK2)
    ax.set_title("Event detection error vs marker events, arm-side view", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_event_timing.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_leakage() -> None:
    s = pd.read_csv(RES / "sequence_leakage.csv")
    models = [("bilstm", "BiLSTM on joint-angle sequences"), ("ridge_poi", "Ridge on 76 mechanics metrics"),
              ("pitcher_memory", "Pitcher memory (no mechanics)")]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    style(ax, grid_axis="x")
    h = 0.36
    for i, (m, lbl) in enumerate(models):
        for j, (sp, c, name) in enumerate((("random", SERIES[1], "Random split"),
                                            ("pitcher_out", SERIES[0], "New pitchers"))):
            r2 = float(s[(s.model == m) & (s.split == sp)].r2.iloc[0])
            y = i + (j - 0.5) * (h + 0.04)
            ax.barh(y, max(r2, 0), height=h, color=c)
            ax.text(max(r2, 0) + 0.015, y, f"{r2:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([l for _, l in models], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("R² predicting pitch speed", color=INK2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[1]), plt.Rectangle((0, 0), 1, 1, color=SERIES[0])],
              labels=["Random split (same pitchers in train and test)", "Pitcher-out (unseen pitchers)"],
              frameon=False, loc="lower right", fontsize=9)
    ax.set_title("A random split rewards recognising the pitcher, not reading the mechanics",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_leakage.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_drivers() -> None:
    d = pd.read_csv(RES / "biomech_drivers.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=SURFACE, sharey=True)
    order = (d[d.outcome == "pitch_speed_mph"].sort_values("effect_per_sd").label.tolist())
    titles = {"pitch_speed_mph": "Pitch speed (mph per SD)",
              "elbow_varus_moment_bwht": "Elbow varus load (%BW·Ht per SD)"}
    for ax, (out, ttl) in zip(axes, titles.items()):
        style(ax, grid_axis="x")
        g = d[d.outcome == out].set_index("label").loc[order]
        for i, (lbl, r) in enumerate(g.iterrows()):
            sig = r.q_bh < 0.05
            c = SERIES[0] if sig else MUTED
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=2)
            ax.plot(r.effect_per_sd, i, "o", ms=8, color=c, mfc=c if sig else SURFACE, mew=2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9.5, color=INK)
        ax.set_title(ttl, loc="left", color=INK, fontsize=12)
    fig.suptitle("What goes with velocity and elbow load: 100 pitchers, adjusted for body size. "
                 "Filled = significant after correcting for 30 tests.", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig4_drivers.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def demo_dashboard() -> None:
    trials = validation.load_clean_trials()
    row, t, kp, ev = trials[0]
    cam = Camera(NAMED_VIEWS["arm_side"])
    rng = np.random.default_rng(7)
    t2, k2, fs = events.downsample(t, project_keypoints(kp, cam), 120)
    k2 = {k: v + rng.normal(0, 4.0, v.shape) for k, v in k2.items()}
    res = analyze_keypoints(k2, fs, view="arm_side", style="overhand", level=row["playing_level"])
    (RES / "demo_report.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("series", "time_s")},
                                                     indent=2))
    report.dashboard(res, str(FIG / "fig5_demo_dashboard.png"),
                     title="Pitch report (demo)",
                     subtitle=f"Input: OBP trial {row.name} projected through a virtual arm-side camera at 120 fps "
                              f"with 4 px keypoint noise  ·  {row['pitch_speed_mph']:.1f} mph")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_view_error()
    fig_frame_rate()
    fig_leakage()
    fig_drivers()
    demo_dashboard()
    print("figures written to", FIG)
