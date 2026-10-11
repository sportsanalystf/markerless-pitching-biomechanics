PY ?= python

.PHONY: setup data test drivers projection sequence figures all clean

setup:
	$(PY) -m pip install -r requirements.txt

data:
	bash scripts/01_get_data.sh

test:
	$(PY) -m pytest -q

drivers:      ## Study A: what goes with velocity and elbow load (~10 s)
	$(PY) scripts/02_biomech_drivers.py

projection:   ## Study B: single-camera error and measurability tiers (~3 min)
	$(PY) scripts/03_projection_study.py

sequence:     ## Study C: random split vs pitcher-out (~7 min, needs torch)
	$(PY) scripts/04_sequence_leakage.py

figures:
	$(PY) scripts/05_figures.py

all: drivers projection sequence figures

clean:
	rm -rf results/*.log .pytest_cache **/__pycache__
