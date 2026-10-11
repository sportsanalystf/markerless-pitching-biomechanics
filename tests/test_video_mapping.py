import numpy as np

from pitchlab.video.keypoints import BLAZE, fill_gaps, to_pitchlab


def fake_landmarks(T=10):
    lm = np.zeros((T, 33, 3))
    for name, i in BLAZE.items():
        lm[:, i, 0] = i          # x encodes the landmark index
        lm[:, i, 1] = 100 + i
    lm[..., 2] = 1.0            # visibility
    return lm


def test_right_hander_uses_right_arm():
    kp = to_pitchlab(fake_landmarks(), "R")
    assert kp["elbow"][0, 0] == BLAZE["r_elbow"]
    assert kp["lead_knee"][0, 0] == BLAZE["l_knee"]


def test_left_hander_swaps_sides():
    kp = to_pitchlab(fake_landmarks(), "L")
    assert kp["elbow"][0, 0] == BLAZE["l_elbow"]
    assert kp["lead_ankle"][0, 0] == BLAZE["r_ankle"]


def test_low_visibility_becomes_nan():
    lm = fake_landmarks()
    lm[3, BLAZE["r_wrist"], 2] = 0.1
    kp = to_pitchlab(lm, "R")
    assert np.isnan(kp["wrist"][3]).all()


def test_fill_gaps_short_and_long():
    x = np.arange(30, dtype=float)
    v = np.stack([x, x], axis=1)
    v[5:8] = np.nan       # 3-frame gap: filled
    v[15:25] = np.nan     # 10-frame gap: kept
    out = fill_gaps({"k": v}, max_gap=6)["k"]
    assert np.allclose(out[5:8, 0], [5, 6, 7])
    assert np.isnan(out[15:25]).all()
