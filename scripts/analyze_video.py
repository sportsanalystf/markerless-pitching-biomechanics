"""Analyze one pitching video and write a JSON report plus a dashboard PNG.

Examples
--------
python scripts/analyze_video.py clip.mp4 --throws R --view arm_side
python scripts/analyze_video.py clip.mp4 --throws L --view glove_side --style windmill --out reports/

The camera should be on a tripod, level with the pitcher's hips, roughly
perpendicular to the throwing direction for side views. 120 fps or higher is
recommended; release-dependent metrics degrade below that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab import report  # noqa: E402
from pitchlab.camera import NAMED_VIEWS  # noqa: E402
from pitchlab.video.pipeline import STYLES, analyze_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--throws", choices=["R", "L"], required=True, help="throwing hand")
    ap.add_argument("--view", choices=list(NAMED_VIEWS), default="arm_side")
    ap.add_argument("--style", choices=list(STYLES), default="overhand",
                    help="overhand = baseball; windmill = fastpitch softball (no reference percentiles)")
    ap.add_argument("--level", default="college", help="reference group for percentiles (OBP playing_level)")
    ap.add_argument("--model", default=None, help="path to a MediaPipe pose_landmarker .task file")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    res = analyze_video(a.video, throws=a.throws, view=a.view, style=a.style, level=a.level, model_path=a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem
    (out / f"{stem}.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("series", "time_s")},
                                                 indent=2))
    report.dashboard(res, str(out / f"{stem}.png"), title=f"Pitch report: {stem}")

    print(f"events (s): {res['events_s']}")
    for name, m in res["metrics"].items():
        val = "withheld" if m.get("value") is None else f"{m['value']} {m['unit']}"
        print(f"  {m['label']:<45} {val:<14} tier {m['tier']}")
    for n in res["notes"]:
        print("note:", n)
    print(f"wrote {out / (stem + '.json')} and {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
