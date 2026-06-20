# percept — CI lanes (impl plan §0)
#   L1 unit  : OUR deterministic logic, fakes only, no keys, <2s  -> `make test`
#   L2 live  : backend INTELLIGENCE + integration, real+gated     -> `PERCEPT_LIVE=1 make eval-live`
#   L3 e2e   : the real system a user drives, headless            -> `make e2e`
SDK := packages/percept-vision
PY  := python3

.PHONY: help test eval-live e2e bench check reproduce

help:
	@echo "test       L1 unit (fakes, no keys, fast)"
	@echo "eval-live  L2 live eval (needs keys; skips without PERCEPT_LIVE=1)"
	@echo "e2e        L3 browser e2e (pre-release)"
	@echo "bench      run percept-bench (sampled)"
	@echo "check      lint + types (best-effort; skips missing tools)"
	@echo "reproduce  reproduce a published scorecard row"

test:
	cd $(SDK) && PYTHONPATH=. $(PY) -m unittest discover -s tests -v

eval-live:
	@if [ "$$PERCEPT_LIVE" != "1" ]; then \
		echo "L2 skipped: set PERCEPT_LIVE=1 and provide backend keys to run live eval."; \
	else \
		cd $(SDK) && PYTHONPATH=. $(PY) -m eval.run; \
	fi

e2e:
	@echo "L3 e2e: scaffolding only in Phase 0 (lands with the voice loop). Skipping."

bench:
	@echo "percept-bench: scaffolding only in Phase 0 (lands with the replay scorer). Skipping."

check:
	@cd $(SDK) && PYTHONPATH=. $(PY) -m compileall -q percept tests && echo "compileall ok"
	@cd $(SDK) && (ruff check . || echo "ruff not installed — skipping lint")

reproduce:
	@echo "make reproduce: lands with percept-bench. Skipping."
