#!/bin/bash
# Run every evaluation layer, in order, with independent kill conditions.
#
# NO LLM drives this. It is a plain sequential runner; the dashboard cron reads whatever it has
# written so far. If this script dies, the results already on disk stay valid and the dashboard
# keeps showing them.
#
# RUNTIME UNDER TEST — the installed wheel, never the repo checkout. Half the SV runners refuse to
# start if `percept` resolves to packages/percept-vision (they are measuring a shippable artifact,
# not a source tree), so every layer runs on $PY with PYTHONPATH unset. Build the wheel with
# eval/dashboard/build_eval_venv.sh; this script refuses to start without it.
#
# CAPS — whichever binds first:
#   * spend      MAX_CALLS provider calls per long run, enforced by the runners' own
#                --max-provider-calls (BudgetedVision). long_run prices Gemini directly
#                ($0.25/$1.50 per Mtok, its own documented default) and reports
#                estimated_cost_usd, so the cap is checked against real money after each run and
#                the whole script stops at MAX_USD.
#   * errors     MAX_PROVIDER_ERRORS consecutive provider failures kills the layer. Without this a
#                bad credential burns the entire call budget on errors in under a minute — which is
#                exactly what happened on the 2026-08-01 11:43 run (1,845 failed calls, no results).
#   * wall clock DEADLINE_S from start, checked before each layer.
set -uo pipefail
REPO=/Users/dhivyavijayakumar/git/Percept-sdk
cd "$REPO" || exit 1

PY="${PY:-/private/tmp/percept-launch/eval-venv/bin/python}"
# Measured on a 5-call smoke against gemini-3.1-flash-lite: ~1.5k input tokens/call, ~$0.0004/call.
# 30k calls is therefore ~$12 per long run — two runs stay an order of magnitude under MAX_USD, and
# at the measured 1.2s p50 with judge-concurrency 1 the 72h wall clock binds first, as it should.
MAX_CALLS="${MAX_CALLS:-30000}"
MAX_USD="${MAX_USD:-100}"
MAX_PROVIDER_ERRORS="${MAX_PROVIDER_ERRORS:-100}"
DEADLINE_S="${DEADLINE_S:-259200}"          # 72 h
STARTED=$(date +%s)
STATE=eval/dashboard/run-state.json
LOG=eval/dashboard/logs/run.log
RESULTS=eval/stage-validation/results

mkdir -p eval/dashboard/logs "$RESULTS"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ ! -x "$PY" ]; then
  log "REFUSING: no eval wheel interpreter at $PY — run eval/dashboard/build_eval_venv.sh first"
  exit 4
fi

# --- state the dashboard reads (pure data; it never interprets) --------------------------------
py_state() { "$PY" - "$@" <<'PY'
import json, sys, time
p = "eval/dashboard/run-state.json"
try: s = json.load(open(p))
except Exception: s = {"lanes": {}}
lane, status = sys.argv[1], sys.argv[2]
s.setdefault("lanes", {})[lane] = {"status": status, "at": time.strftime("%F %T")}
if len(sys.argv) > 3: s["note"] = sys.argv[3]
json.dump(s, open(p, "w"), indent=1)
PY
}

remaining() { echo $(( DEADLINE_S - ($(date +%s) - STARTED) )); }

# The google-genai 2.9.0 client raises AttributeError from its own aclose() on every request
# teardown. It is a library shutdown bug, unrelated to whether the call succeeded, and it triples
# the log. Drop exactly that trace — never anything carrying an outcome.
denoise() { grep -v -e "Task exception was never retrieved" \
                   -e "BaseApiClient.aclose" \
                   -e "_async_httpx_client" \
                   -e "^Traceback (most recent call last):$" \
                   -e "^future: <Task finished" \
                   -e "^  File .*_api_client.py" \
                   -e "^    await self\._async_httpx_client" \
                   -e "^          \^*$" ; }

# run_layer <lane> <result-json> <label> <command...>
#
# Success is "the runner wrote its result file", NOT exit status. Every SV runner exits non-zero
# when its SUITE verdict is FAIL — and a FAIL verdict is a MEASUREMENT, the entire point of the
# run. Keying the dashboard off exit status would report a layer that measured 96 real compile
# failures as "not run" and show nothing at all.
run_layer() {
  local lane="$1" result="$2" label="$3"; shift 3
  local path="$RESULTS/$result"
  if [ "$(remaining)" -le 0 ]; then
    py_state "$lane" "failed" "72h deadline reached before $label"
    log "DEADLINE reached, skipping $label"; return 1
  fi
  py_state "$lane" "running"
  log "START $label :: $*"
  local before=0
  [ -f "$path" ] && before=$(stat -f %m "$path" 2>/dev/null || echo 0)

  env -u PYTHONPATH "$@" 2>&1 | denoise >> "$LOG"
  local rc=${PIPESTATUS[0]}

  local after=0
  [ -f "$path" ] && after=$(stat -f %m "$path" 2>/dev/null || echo 0)
  if [ "$after" -gt "$before" ]; then
    py_state "$lane" "done"; log "OK $label (rc=$rc, wrote $result)"
    return 0
  fi
  py_state "$lane" "failed"; log "FAILED $label (rc=$rc, no fresh $result)"
  return 1
}

# --- preflight: prove ONE real provider call works before spending on a whole layer -------------
# A wrong or missing key is indistinguishable from a broken product in the aggregate metrics — both
# render as provider_error. Find out for one call, not two thousand.
preflight() {
  env -u PYTHONPATH "$PY" - <<'PY' 2>&1 | tail -3
import os, sys
key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
       or os.environ.get("GOOGLE_AISTUDIO_API_KEY"))
if not key:
    print("PREFLIGHT FAIL: no Gemini key in env"); sys.exit(1)
try:
    from google import genai
    c = genai.Client(api_key=key)
    m = os.environ.get("PERCEPT_VISION_MODEL", "gemini-3.1-flash-lite")
    r = c.models.generate_content(model=m, contents="Reply with the single word: ok")
    print(f"PREFLIGHT OK: {m} -> {(r.text or '').strip()[:20]!r}")
except Exception as e:
    print(f"PREFLIGHT FAIL: {type(e).__name__}: {e}"); sys.exit(1)
PY
}

py_state "run" "running" "started $(date '+%F %T') · cap ${MAX_CALLS} calls / \$${MAX_USD} · 72h deadline"

# ================================================================================================
# L0 capability — deterministic, offline, free
# ================================================================================================
PERCEPT_MODE=offline run_layer L0 sv0_ingest.json  "sv0_ingest"  "$PY" eval/stage-validation/sv0_ingest.py
PERCEPT_MODE=offline run_layer L0 sv0_compile.json "sv0_compile" "$PY" eval/stage-validation/sv0_compile.py

# ================================================================================================
# L1 quality — admission and senses are offline; pool and cognition spend
# ================================================================================================
PERCEPT_MODE=offline run_layer L1 sv1_admission.json "sv1_admission" "$PY" eval/stage-validation/sv1_admission.py
PERCEPT_MODE=offline run_layer L1 sv2_senses.json    "sv2_senses"    "$PY" eval/stage-validation/sv2_senses.py --quick

# ================================================================================================
# L1a audio — offline
# ================================================================================================
PERCEPT_MODE=offline run_layer L1a sv_audio.json "sv_audio" "$PY" eval/stage-validation/sv_audio.py

# ================================================================================================
# Everything below spends. Credentials load HERE and not before: the offline layers above must not
# see a key, or PERCEPT_MODE=offline stops being the thing being measured.
# ================================================================================================
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export PERCEPT_VISION_BACKEND="${PERCEPT_VISION_BACKEND:-gemini}"

log "PREFLIGHT: one live ${PERCEPT_VISION_MODEL:-gemini-3.1-flash-lite} call"
PF="$(preflight)"; PF_RC=$?
log "$PF"
if [ $PF_RC -ne 0 ]; then
  log "ABORTING every spending layer — no credential. Offline results above stand."
  for lane in L1 L2 L3; do py_state "$lane" "blocked"; done
  py_state "run" "failed" "no provider credential — offline layers only ($(date '+%F %T'))"
  exit 3
fi

run_layer L1 sv3_pool.json      "sv3_pool"      "$PY" eval/stage-validation/sv3_pool.py
run_layer L1 sv4_cognition.json "sv4_cognition" "$PY" eval/stage-validation/sv4_cognition.py

# --- L2 temporal / L3 long-horizon — the runs that actually spend -------------------------------
# grep -c always prints a count but exits 1 on zero matches; `|| echo 0` would append a SECOND
# line and break the arithmetic below. Take the printed count and default only if it is empty.
count_provider_errors() {
  local n; n=$(grep -c '"outcome": "provider_error"' "$LOG" 2>/dev/null | head -1)
  echo "${n:-0}"
}

spend_guard() {   # kill $1 once the log shows MAX_PROVIDER_ERRORS failures for this layer
  local pid="$1" base
  base=$(count_provider_errors)
  while kill -0 "$pid" 2>/dev/null; do
    local now; now=$(count_provider_errors)
    if [ $(( now - base )) -ge "$MAX_PROVIDER_ERRORS" ]; then
      log "SPEND GUARD: $(( now - base )) provider errors — killing pid $pid"
      kill -TERM "$pid" 2>/dev/null; return
    fi
    sleep 20
  done
}

run_long() {  # run_long <lane> <out-stem> <wall-share 0..1> <extra args...>
  local lane="$1" stem="$2" share="$3"; shift 3
  # Split the remaining window rather than handing it all to the first run. L2 given the whole
  # 72h would simply never leave any for L3, and a missing L3 is indistinguishable on the
  # dashboard from an L3 that failed.
  local wall
  wall=$("$PY" -c "print(int($(remaining) * $share))")
  py_state "$lane" "running"
  log "START $stem :: cap ${MAX_CALLS} calls, wall ${wall}s (${share} of $(remaining)s left)"
  local before=0
  [ -f "$RESULTS/$stem.json" ] && before=$(stat -f %m "$RESULTS/$stem.json" 2>/dev/null || echo 0)

  env -u PYTHONPATH "$PY" eval/stage-validation/long_run.py \
      --max-provider-calls "$MAX_CALLS" --max-wall-s "$wall" \
      --out-stem "$stem" "$@" 2>&1 | denoise >> "$LOG" &
  local pid=$!
  spend_guard "$pid" &
  local guard=$!
  wait "$pid"; local rc=$?
  kill "$guard" 2>/dev/null

  local after=0
  [ -f "$RESULTS/$stem.json" ] && after=$(stat -f %m "$RESULTS/$stem.json" 2>/dev/null || echo 0)
  if [ "$after" -gt "$before" ]; then
    py_state "$lane" "done"; log "OK $stem (rc=$rc)"
  else
    py_state "$lane" "failed"; log "FAILED $stem (rc=$rc, no fresh $stem.json)"
  fi

  # Real money, read back off the run's own receipt — not an estimate made here.
  local spent
  spent=$("$PY" -c "
import json,sys
try: print(json.load(open('$RESULTS/$stem.json')).get('accounting',{}).get('estimated_cost_usd',0))
except Exception: print(0)" 2>/dev/null || echo 0)
  log "SPEND $stem: \$$spent (cap \$$MAX_USD)"
  "$PY" -c "import sys; sys.exit(0 if float('$spent') < float('$MAX_USD') else 1)" || {
    log "BUDGET CAP REACHED (\$$spent >= \$$MAX_USD) — stopping"
    py_state "run" "done" "stopped at \$$MAX_USD budget cap"
    exit 0
  }
}

run_long L2 long_run 0.90

# L3 long-horizon is NOT run. It used to be `run_long L3 long_run_2h`, which passed no --manifest
# and therefore silently re-ran L2's own source under a second output name — two identical passes
# reported as two layers. There is no 2h fixture in this repo to point it at: the declared
# long_sources/inject_2h_v1.mp4 is referenced by nothing but the dashboard and was never built.
# Re-enable this only with an explicit --manifest naming a real long-horizon source.
py_state "L3" "no fixture"

# L4 is deliberately not run: 180/205 gate cases are unreviewed, so any figure over it would be a
# metric with no ground truth. The dashboard says that instead of showing a number.
py_state "L4" "idle"
py_state "run" "done" "finished $(date '+%F %T')"
log "RUN COMPLETE"
