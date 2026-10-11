"""Light-mode figures: the per-pitch dashboard and shared chart styling."""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e7e6e2"
BAND = "#f3f2ef"
SURFACE = "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dashboard(res: dict, path: str, title: Optional[str] = None, subtitle: str = "") -> None:
    """Joint-angle time series with phase bands (left) and the gated metric table (right)."""
    t = np.asarray(res["time_s"])
    ser = res["series"]
    ev = res["events_s"]
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    fig.text(0.04, 0.95, title or "Pitch report", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, 0.915, subtitle or f"{res['view'].replace('_', ' ')} view  ·  {res['fps']:.0f} fps  ·  "
             f"{res['style']} delivery", fontsize=11, color=INK2)

    panels = [("elbow_flexion", "Elbow flexion (deg)"),
              ("lead_knee_flexion", "Lead knee flexion (deg)"),
              ("trunk_forward_tilt", "Trunk forward tilt (deg)")]
    phases = res.get("phases_s", {})
    for i, (k, lbl) in enumerate(panels):
        ax = fig.add_axes([0.06, 0.64 - i * 0.27, 0.50, 0.21])
        style(ax)
        for j, (name, (a, b)) in enumerate(phases.items()):
            if j % 2 == 0:
                ax.axvspan(a, b, color=BAND, lw=0)
        ax.plot(t, ser[k], color=SERIES[0], lw=2)
        for name, tv in ev.items():
            if tv is not None:
                ax.axvline(tv, color=MUTED, lw=1, ls="--")
                if i == 0:
                    ax.text(tv, 1.02, name.upper(), transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_ylabel(lbl, color=INK2, fontsize=10)
        ax.set_xlim(t[0], t[-1])
        if i == 2:
            ax.set_xlabel("Time (s)", color=INK2, fontsize=10)
        if i == 0 and phases:
            for name, (a, b) in phases.items():
                ax.text((a + b) / 2, 0.04, name.replace("_", " "), transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=8, color=MUTED)

    # metric table
    x0, y0 = 0.60, 0.86
    cols = [(x0, "Metric"), (x0 + 0.20, "Value"), (x0 + 0.26, "Pctl"), (x0 + 0.31, "Validity vs markers")]
    for x, h in cols:
        fig.text(x, y0, h, fontsize=10, fontweight="bold", color=INK2)
    y = y0 - 0.045
    order = sorted(res["metrics"].items(), key=lambda kv: (kv[1]["tier"], kv[0]))
    for name, m in order:
        withheld = m.get("value") is None
        c = MUTED if withheld else INK
        fig.text(x0, y, m["label"], fontsize=10, color=c)
        if withheld:
            fig.text(x0 + 0.20, y, "withheld", fontsize=10, color=MUTED, style="italic")
        else:
            fig.text(x0 + 0.20, y, f"{m['value']:.1f} {'%' if 'height' in m['unit'] else '°'}", fontsize=10, color=INK)
            p = m.get("percentile")
            fig.text(x0 + 0.26, y, f"{p}" if p is not None else "–", fontsize=10, color=INK)
        rm = m.get("validation_rmse")
        fig.text(x0 + 0.31, y, f"Tier {m['tier']}  ·  RMSE {rm:.1f}" if rm is not None else f"Tier {m['tier']}",
                 fontsize=10, color=c)
        y -= 0.04
    fig.text(x0, y - 0.01, "Tier A: RMSE ≤ 5 and r ≥ 0.80 against marker truth.  Tier B: RMSE ≤ 10, r ≥ 0.60.\n"
             "Tier C is withheld at this view. Percentiles vs OBP collegiate baseball fastballs.",
             fontsize=8.5, color=INK2, va="top")
    for n in res.get("notes", []):
        y -= 0.05
        fig.text(x0, y - 0.04, n, fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
