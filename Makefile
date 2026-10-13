PY ?= python

.PHONY: test guardrails check validate prepublish

test:
	$(PY) -m pytest -q -rf packages/percept-vision/tests tools/guardrails/tests benchmarks/tests   # -rf: a failure is always named

guardrails:
	$(PY) tools/guardrails/guardrails.py all

check: guardrails test

validate:
	$(PY) tools/validate.py --python $(PY)

# Pre-publication scan of the tracked tree and a freshly built wheel (needs .prepublish-terms).
prepublish:
	$(PY) tools/prepublish_check.py --python $(PY)
