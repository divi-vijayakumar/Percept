#!/bin/bash
# Build the interpreter the eval layers run on.
#
# Half the SV runners refuse to start if `percept` resolves to packages/percept-vision — they are
# measuring a shippable wheel, not a source tree, and a source checkout can pass on code that was
# never packaged. So: build a wheel from the current checkout, install it into a clean venv, and
# hand that interpreter to run_layers.sh.
#
# Rebuild this whenever the source moves. A stale wheel silently measures old code — the 0.8.3
# wheel that was lying around predated Settings.env_rejections entirely.
set -euo pipefail
REPO=/Users/dhivyavijayakumar/git/Percept-sdk
VENV="${VENV:-/private/tmp/percept-launch/eval-venv}"
WHEELDIR=/private/tmp/percept-launch/wheel

cd "$REPO/packages/percept-vision"
rm -rf "$WHEELDIR"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --disable-pip-version-check build
"$VENV/bin/python" -m build --wheel --outdir "$WHEELDIR" >/dev/null
WHEEL=$(ls "$WHEELDIR"/*.whl | head -1)

# The extras the layers actually exercise: gemini judging, the tier-0 harness/edge, audio, and the
# opencv video runtime the long runs decode with.
"$VENV/bin/pip" install -q --disable-pip-version-check --force-reinstall \
  "${WHEEL}[gemini,harness,edge,audio,runtime,claude]"

"$VENV/bin/python" - <<'PY'
import percept, pathlib
p = pathlib.Path(percept.__file__)
assert "site-packages" in p.parts, f"percept resolved to a source tree: {p}"
print(f"eval venv ready: percept {percept.__version__} <- {p}")
PY
