"""Integration tests on the OpenBiomechanics data. Skipped if the data is not downloaded."""
import numpy as np
from sklearn.model_selection import GroupKFold

from conftest import needs_data


@needs_data
def test_trials_table():
    from pitchlab import data
    df = data.trials()
    assert len(df) == 411
    assert df["pitcher_id"].nunique() == 100
    assert {"pitch_speed_mph", "elbow_varus_moment_bwht"} <= set(df.columns)


@needs_data
def test_clean_trials_and_events_ordered():
    from pitchlab import validation
    trials = validation.load_clean_trials()
    assert 380 <= len(trials) <= 411
    for row, t, kp, ev in trials[:25]:
        assert ev["pkh"] < ev["fp"] < ev["br"]
        assert kp["wrist"].shape[1] == 3


@needs_data
def test_mirroring_puts_throwing_arm_on_negative_y():
    from pitchlab import validation
    trials = validation.load_clean_trials()
    for row, t, kp, ev in trials[:25]:
        i = int(np.argmin(np.abs(t - ev["br"])))
        assert kp["shoulder"][i, 1] < kp["glove_shoulder"][i, 1]


@needs_data
def test_end_to_end_demo_reproduces_tier_a_metrics():
    from pitchlab import events, validation
    from pitchlab.camera import NAMED_VIEWS, Camera, project_keypoints
    from pitchlab.video.pipeline import analyze_keypoints
    row, t, kp, ev = validation.load_clean_trials()[0]
    t2, k2, fs = events.downsample(t, project_keypoints(kp, Camera(NAMED_VIEWS["arm_side"])), 120)
    res = analyze_keypoints(k2, fs, view="arm_side")
    assert res["metrics"]["trunk_forward_tilt_br"]["value"] is not None
    assert res["metrics"]["hip_shoulder_sep_fp"]["value"] is None   # tier C is withheld
    assert abs(res["events_s"]["br"] - ev["br"]) < 0.02


def test_groupkfold_never_shares_pitchers():
    groups = np.repeat(np.arange(40), 4)
    X = np.zeros((len(groups), 1))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=groups):
        assert not set(groups[tr]) & set(groups[te])
