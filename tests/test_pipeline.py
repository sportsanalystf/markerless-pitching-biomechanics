import json

import numpy as np
import pytest

from pitchlab.validation import tier
from pitchlab.video import pipeline


def test_tier_rule():
    assert tier(1.4, 0.99) == "A"
    assert tier(5.2, 0.79) == "B"
    assert tier(17.0, 0.34) == "C"
    assert tier(np.nan, 0.9) == "C"


def test_percentile_interpolation():
    ref = {"p5": 10.0, "p50": 20.0, "p95": 30.0}
    assert pipeline.percentile_of(20.0, ref) == 50
    assert pipeline.percentile_of(0.0, ref) == 5
    assert pipeline.percentile_of(99.0, ref) == 95
    assert pipeline.percentile_of(None, ref) is None


def test_configs_cover_every_view_and_metric():
    from pitchlab.camera import NAMED_VIEWS
    from pitchlab.validation import ALL_METRICS
    meas = json.loads((pipeline.CFG / "measurability.json").read_text())["views"]
    assert set(meas) == set(NAMED_VIEWS)
    for v in meas.values():
        assert set(v) == set(ALL_METRICS)
        assert all(x["tier"] in "ABC" for x in v.values())


def test_norms_only_for_reportable_metrics():
    meas = json.loads((pipeline.CFG / "measurability.json").read_text())["views"]
    norms = json.loads((pipeline.CFG / "norms.json").read_text())["views"]
    for view, ms in norms.items():
        for m in ms:
            assert meas[view][m]["tier"] in ("A", "B")


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        pipeline.analyze_keypoints({}, 120, view="dugout")
    with pytest.raises(ValueError):
        pipeline.analyze_keypoints({}, 120, style="sidearm")
