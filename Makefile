# percept — CI lanes (impl plan §0)
#   L1 unit  : OUR deterministic logic, fakes only, no keys, <2s  -> `make test`
#   L2 live  : backend INTELLIGENCE + integration, real+gated     -> `PERCEPT_LIVE=1 make eval-live`
#   L3 e2e   : the real system a user drives, headless            -> `make e2e`
SDK       := packages/percept-vision
EDGE      := packages/percept-edge
PY        := python3
# absolute path to the zero-dep contract pkg (vision runs from inside $(SDK), so it needs this)

.PHONY: help test test-harness test-js test-all test-templates test-capability-gate parity eval-live eval-benchmark e2e bench audit-benchmark scorecard benchmark-prepare benchmark-fetch benchmark-fetch-charades benchmark-fetch-proactiveqa benchmark-index-proactiveqa benchmark-manifest benchmark-run benchmark-collect check check-skills-sync reproduce clean architecture architecture-check

help:
	@echo "test       L1 unit — percept-vision (incl. percept.harness + percept.contracts; fakes, no keys, fast) + template lane"
	@echo "test-templates  L1 unit — templates/ reference apps (keyless; pipecat-dependent tests skip cleanly)"
	@echo "test-capability-gate  L1 capability gate — Y16 task-level suite (CAPABILITY asserts, no keys, no stat bars)"
	@echo "test-js    L1 unit — percept-edge (vitest; needs npm install)"
	@echo "test-all   test + test-js"
	@echo "eval-live  L2 live eval (needs keys; skips without PERCEPT_LIVE=1)"
	@echo "eval-benchmark  per-concern P/R on a labelled-clip manifest (offline; MANIFEST=path)"
	@echo "e2e        L3 e2e — live smoke (needs keys; skips cleanly)"
	@echo "bench      gating cost reduction (gated vs ungated vision calls)"
	@echo "audit-benchmark  validate golden manifest + baseline registry artifacts"
	@echo "benchmark-prepare  create ignored local benchmark folders + inventory"
	@echo "benchmark-fetch  pre-cache benchmark media (VIDEOS=12 REPCOUNT=12)"
	@echo "benchmark-fetch-charades  pre-cache Charades reference videos (CHARADES_N=100)"
	@echo "benchmark-fetch-proactiveqa  pre-cache ProactiveVideoQA EGO videos (PVQA_N=100)"
	@echo "benchmark-index-proactiveqa  rewrite ProactiveQA subset from local cache"
	@echo "benchmark-manifest  write ignored content-hashed candidate manifest (PT_ACTION=200 PT_SOUND=100)"
	@echo "benchmark-run  run reproducible local scorecard wrapper (TRACKS=stable N=8 WORKERS=2)"
	@echo "benchmark-collect  re-render a local scorecard (RUN_DIR=eval/local-benchmark/runs/<id>)"
	@echo "check      lint + types (best-effort; skips missing tools)"
	@echo "reproduce  reproduce a published scorecard row"

# PERCEPT_MODE=offline: the suite is offline BY CONSTRUCTION (one declared property — RFC-offline-scripted-mode).
# pytest declares it in tests/conftest.py; unittest discovery loads no conftest, so declare it here. Without it,
# voice tests that unskip once pipecat is installed hit live-mode ModeError.
test:
	cd $(SDK) && PYTHONPATH=. PERCEPT_MODE=offline $(PY) -m unittest discover -s tests
	$(MAKE) test-templates

# templates/ = copyable reference applications; each ships its own tests. Keyless: pipecat-dependent
# tests skipUnless the template's deps are installed; everything else runs on the bare SDK checkout.
test-templates:
	cd templates/realtime_voice_watch_agent && PYTHONPATH=.:$(abspath $(SDK)) PERCEPT_NO_BANNER=1 $(PY) -m unittest discover -s tests -t .

test-js:
	cd $(EDGE) && npm test

test-all: test test-js

# --------------------------------------------------------------------------------------------------
# Profile-acceptance CI gate — CAPABILITY assertions, not a statistical bar.
#
# Runs the Y16 task-level capability suite (9 capabilities over percept.harness.synthetic.corpora,
# replayed through a REAL ScriptedJudge standing in for a perfect classifier — what's under test is
# the deterministic runtime: EpisodeLog, percept.claims, interval_coverage, Gate, BehaviorExecutor)
# plus the synthetic-fixture validity smoke test those capabilities are built on. Every assertion here
# is pass/fail against a labelled fixture — no API key, no network, no measured rate against a
# threshold anywhere in this target (contrast percept/slo.py's $/stream-hour-style statistical bars,
# which pre-register a NUMBER; this gate pre-registers nothing because it has no number to tune).
#
# STRICT XFAIL, NO EXTRA FLAG: the suite's 2 documented defects are marked
# @unittest.expectedFailure (see test_y16_capability_suite.py's module docstring table — capability 5's
# full-trace discrepancy and capability 7's boundary off-by-one). Both unittest and pytest already
# treat an expectedFailure that unexpectedly PASSES as a hard FAILURE ("Unexpected success") by
# default — there is no separate "strict" switch to flip. -rxX below only makes the tolerated xfails
# (and any XPASS) print in the summary instead of collapsing into the dot count, so a fixed defect
# whose assertion should be promoted out of expectedFailure is flagged loudly, not silently lost.
#
# DESELECTED, NOT FIXED (card Z9, owned elsewhere): tests/test_p_lat_actionable.py and
# tests/test_p_qual.py fail at COLLECTION — `ModuleNotFoundError: core_archetypes` — because they
# import a gitignored viability fixture that is not part of every checkout. That import failure is
# pre-existing and unrelated to capability regressions; --ignore both explicitly so this gate's
# result depends only on the capabilities it exists to assert, never on an unrelated collection
# error. (On a base that already has [Z9]'s importorskip guard, both files skip cleanly instead of
# erroring and these --ignore flags become redundant, not wrong — leave them; they still document
# the exclusion.)
#
# PRE-REGISTRATION DISCIPLINE, PRESERVED: a profile/behavior change still ships with its own frozen
# statistical run wherever that discipline already applies (percept/slo.py, eval/benchmark_ci_gate) —
# this target does not touch that machinery and adds none of its own; it only asserts the fixed,
# labelled capability contract.
# --------------------------------------------------------------------------------------------------
test-capability-gate:
	cd $(SDK) && PYTHONPATH=. $(PY) -m pytest \
		tests/test_y16_capability_suite.py tests/test_synthetic_fixtures.py \
		--ignore=tests/test_p_lat_actionable.py --ignore=tests/test_p_qual.py \
		-rxX -q

clean:  ## remove build + tooling artifacts (leaves out/ — that's intentional local run output)
	rm -rf packages/*/build packages/*/dist packages/*/*.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache packages/*/.pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

parity:  ## Edge<->harness conformance — JS percept-edge vs Python percept.harness, byte-for-byte
	PYTHONPATH=$(SDK) $(PY) $(EDGE)/parity/compare.py

eval-live:
	@if [ "$$PERCEPT_LIVE" != "1" ]; then \
		echo "L2 skipped: set PERCEPT_LIVE=1 and provide backend keys to run live eval."; \
	else \
		cd $(SDK) && PYTHONPATH=. $(PY) -m eval.run; \
	fi

# Per-concern precision/recall on a labelled-clip manifest (offline; fake backend; no keys).
#   make eval-benchmark MANIFEST=eval/benchmark_ci_gate/manifest.json
MANIFEST ?= $(abspath eval/benchmark_ci_gate/manifest.json)
eval-benchmark:
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.benchmark $(abspath $(MANIFEST))

e2e:
	cd $(SDK) && PYTHONPATH=. $(PY) -m e2e.live_smoke

bench:
	PYTHONPATH=$(SDK) $(PY) -m percept.harness.bench_gating

audit-benchmark:
	PYTHONPATH=$(SDK) $(PY) -m percept.bench audit --manifest eval/golden-v1/MANIFEST.json --baselines packages/percept-vision/eval/baselines.json

scorecard:
	PYTHONPATH=$(SDK) $(PY) -m percept.bench scorecard --manifest eval/golden-v1/MANIFEST.json --out scorecard.txt

benchmark-prepare:
	$(PY) tools/percept_benchmark.py prepare

VIDEOS ?= 12
REPCOUNT ?= 12
benchmark-fetch:
	$(PY) tools/percept_benchmark.py fetch --videos $(VIDEOS) --repcount $(REPCOUNT)

CHARADES_N ?= 100
benchmark-fetch-charades:
	$(PY) tools/percept_benchmark.py fetch-charades --n $(CHARADES_N)

PVQA_N ?= 100
benchmark-fetch-proactiveqa:
	$(PY) tools/percept_benchmark.py fetch-proactiveqa --n $(PVQA_N)

benchmark-index-proactiveqa:
	$(PY) tools/percept_benchmark.py index-proactiveqa

PT_ACTION ?= 200
PT_SOUND ?= 100
MANIFEST_CHARADES ?= 100
MANIFEST_PVQA ?= 100
MANIFEST_REPCOUNT ?= 100
benchmark-manifest:
	$(PY) tools/percept_benchmark.py manifest-candidate --pt-action $(PT_ACTION) --pt-sound $(PT_SOUND) --charades $(MANIFEST_CHARADES) --proactiveqa $(MANIFEST_PVQA) --repcount $(MANIFEST_REPCOUNT)

TRACKS ?= stable
N ?= 8
WORKERS ?= 2
ATTEMPTS ?= 2
LIVE ?= 0
LIVE_FLAG := $(if $(filter 1,$(LIVE)),--live,)
benchmark-run:
	$(PY) tools/percept_benchmark.py run $(LIVE_FLAG) --tracks $(TRACKS) --n $(N) --workers $(WORKERS) --attempts $(ATTEMPTS)

RUN_DIR ?=
benchmark-collect:
	@if [ -z "$(RUN_DIR)" ]; then echo "set RUN_DIR=eval/local-benchmark/runs/<run-id>"; exit 2; fi
	$(PY) tools/percept_benchmark.py collect --run-dir $(RUN_DIR)

calibrate:
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.calibrate

exp-categorical:
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_categorical

check-skills-sync:  ## Verify skill directories are byte-identical
	@diff -r .claude/skills/percept-agent .agents/skills/percept-agent --exclude="__pycache__" > /dev/null && echo "skills sync ok" || (echo "ERROR: skill directories out of sync (KEEP IN SYNC: edit BOTH)"; exit 1)

check: architecture-check check-skills-sync
	@cd $(SDK) && PYTHONPATH=. $(PY) -m compileall -q percept tests && echo "compileall ok"
	@$(PY) -m compileall -q templates && echo "templates compileall ok"
	@cd $(SDK) && (ruff check . || echo "ruff not installed — skipping lint")

architecture:  ## (re)generate architecture.html + docs/architecture.svg from code + meta
	$(PY) tools/build_architecture.py --meta docs/architecture.meta.json --out architecture.html --svg docs/architecture.svg

architecture-check:  ## CI gate: fail if the committed diagram is stale vs code/meta
	$(PY) tools/build_architecture.py --check --meta docs/architecture.meta.json --out architecture.html --svg docs/architecture.svg
	git diff --exit-code architecture.html docs/architecture.svg

reproduce:
	PYTHONPATH=$(SDK) $(PY) -m percept.bench scorecard --manifest eval/golden-v1/MANIFEST.json --out scorecard.txt

prevalence-pt:  ## Perception Test prevalence (needs PERCEPT_LIVE + opencv-python-headless + downloaded sample)
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_perception_test

e2e-bench:  ## E2E: full Percept (gate) vs raw-VLM-loop on Perception Test onsets (PERCEPT_LIVE)
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_e2e

charades-bench:  ## E2E on Charades (exhaustive labels): full Percept vs raw-VLM-loop (PERCEPT_LIVE)
	cd $(SDK) && PYTHONPATH=. $(PY) -m eval.exp_charades
