import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _has_data() -> bool:
    try:
        from pitchlab import data
        d = data.data_dir()
        return (d / "poi_metrics.csv").exists()
    except Exception:
        return False


needs_data = pytest.mark.skipif(not _has_data(), reason="OBP data not downloaded (run scripts/01_get_data.sh)")
