"""Event detection on a synthetic 2D delivery with known event frames."""
import numpy as np

from pitchlab import events

FS = 240.0


def synthetic_delivery(pkh_s=0.6, fp_s=1.1, br_s=1.25, dur=1.6):
    t = np.arange(0, dur, 1 / FS)
    n = len(t)
    # lead knee: lifts to its highest point (smallest image y) at pkh
    knee_y = 600 + 120 * ((t - pkh_s) / 0.4) ** 2
    knee = np.stack([np.full(n, 900.0), np.clip(knee_y, None, 760)], axis=1)
    # lead ankle: moves forward between pkh and fp, then stops
    s = np.clip((t - pkh_s) / (fp_s - pkh_s), 0, 1)
    ankle_x = 900 + 600 * (3 * s ** 2 - 2 * s ** 3)
    ankle = np.stack([ankle_x, np.full(n, 900.0)], axis=1)
    # wrist: speed peaks at br (gaussian velocity pulse)
    v = 4000 * np.exp(-0.5 * ((t - br_s) / 0.025) ** 2)
    wrist_x = 800 + np.cumsum(v) / FS
    wrist = np.stack([wrist_x, np.full(n, 400.0)], axis=1)
    kp = {"lead_knee": knee, "lead_ankle": ankle, "wrist": wrist}
    return t, kp


def test_detects_pkh_fp_br_within_two_frames():
    t, kp = synthetic_delivery()
    ev = events.detect_events_2d(kp, FS, br_offset_s=0.0)
    assert abs(t[ev["pkh"]] - 0.6) <= 2 / FS
    assert abs(t[ev["br"]] - 1.25) <= 2 / FS
    assert abs(t[ev["fp"]] - 1.1) <= 0.03  # foot plant is the hardest event
    assert ev["pkh"] < ev["fp"] < ev["br"]


def test_br_offset_shifts_release():
    t, kp = synthetic_delivery()
    a = events.detect_events_2d(kp, FS, br_offset_s=0.0)["br"]
    b = events.detect_events_2d(kp, FS, br_offset_s=0.025)["br"]
    assert b - a == round(0.025 * FS)


def test_plausibility_filter():
    assert events.plausible({"pkh": 0.5, "fp": 1.0, "br": 1.15})
    assert not events.plausible({"pkh": 0.5, "fp": 1.0, "br": 0.98})   # release before plant
    assert not events.plausible({"pkh": 0.9, "fp": 1.0, "br": 1.15})   # stride too short
    assert not events.plausible({"pkh": 0.5, "fp": np.nan, "br": 1.15})


def test_downsample_rates():
    t = np.arange(0, 1, 1 / 360)
    kp = {"a": np.zeros((len(t), 2))}
    for fps in (360, 240, 120, 60):
        t2, k2, fs = events.downsample(t, kp, fps)
        assert len(t2) == len(k2["a"])
        assert np.isclose(np.median(np.diff(t2)), 1 / fs)


def test_lowpass_preserves_constant_signal():
    x = np.full((200, 2), 5.0)
    assert np.allclose(events.lowpass(x, 120.0), 5.0)
