PY ?= python

.PHONY: test guardrails check validate

test:
	$(PY) -m pytest -q packages/percept-vision/tests tools/guardrails/tests

guardrails:
	$(PY) tools/guardrails/guardrails.py all

check: guardrails test

validate:
	$(PY) tools/validate.py --python $(PY)
