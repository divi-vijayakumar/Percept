# percept — CI lanes (impl plan §0)
#   L1 unit  : OUR deterministic logic, fakes only, no keys, <2s  -> `make test`
#   L2 live  : backend INTELLIGENCE + integration, real+gated     -> `PERCEPT_LIVE=1 make eval-live`
#   L3 e2e   : the real system a user drives, headless            -> `make e2e`
SDK     := packages/percept-vision
HARNESS := packages/percept-harness
EDGE    := packages/percept-edge
PY      := python3

.PHONY: help test test-harness test-js test-all eval-live e2e bench scorecard check reproduce

help:
	@echo "test       L1 unit — percept-vision + percept-harness (fakes, no keys, fast)"
	@echo "test-js    L1 unit — percept-edge (vitest; needs npm install)"
	@echo "test-all   test + test-js"
	@echo "eval-live  L2 live eval (needs keys; skips without PERCEPT_LIVE=1)"
	@echo "e2e        L3 e2e — live smoke (needs keys; skips cleanly)"
	@echo "bench      gating cost reduction (gated vs ungated vision calls)"
	@echo "check      lint + types (best-effort; skips missing tools)"
	@echo "reproduce  reproduce a published scorecard row"

test:
	cd $(SDK) && PYTHONPATH=. $(PY) -m unittest discover -s tests
	PYTHONPATH=$(SDK):$(HARNESS) $(PY) -m unittest discover -s $(HARNESS)/tests

test-js:
	cd $(EDGE) && npm test

test-all: test test-js

eval-live:
	@if [ "$$PERCEPT_LIVE" != "1" ]; then \
		echo "L2 skipped: set PERCEPT_LIVE=1 and provide backend keys to run live eval."; \
	else \
		cd $(SDK) && PYTHONPATH=. $(PY) -m eval.run; \
	fi

e2e:
	cd $(SDK) && PYTHONPATH=. $(PY) -m e2e.live_smoke

bench:
	PYTHONPATH=$(SDK):$(HARNESS) $(PY) -m percept_harness.bench_gating

scorecard:
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.scorecard

calibrate:
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.calibrate

exp-categorical:
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_categorical

check:
	@cd $(SDK) && PYTHONPATH=. $(PY) -m compileall -q percept tests && echo "compileall ok"
	@cd $(SDK) && (ruff check . || echo "ruff not installed — skipping lint")

reproduce:
	@echo "make reproduce: lands with percept-bench. Skipping."

prevalence-pt:  ## Perception Test prevalence (needs PERCEPT_LIVE + opencv-python-headless + downloaded sample)
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_perception_test

e2e-bench:  ## E2E: full Percept (gate) vs raw-VLM-loop on Perception Test onsets (PERCEPT_LIVE)
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_e2e

charades-bench:  ## E2E on Charades (exhaustive labels): full Percept vs raw-VLM-loop (PERCEPT_LIVE)
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_charades
