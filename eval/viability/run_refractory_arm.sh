#!/usr/bin/env bash
# WG D_op + time-refractory arm (Addendum 9): the precision fix. WG base (block_max+window+gate-relax, <=3
# goals) + PERCEPT_GATE_REFRACTORY_S at two values. Waits for any in-flight benchmark first. Appends to LEDGER.
set -u
cd /Users/dhivyavijayakumar/git/Percept-sdk
LEDGER=eval/viability/LADDER_LEDGER.md
export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)"
export PERCEPT_VISION_BACKEND=openrouter PERCEPT_VISION_MODEL="qwen/qwen3-vl-32b-instruct"
export PERCEPT_BENCH_VIDEO_CONCURRENCY=8 PERCEPT_BENCH_BLOCKMAX_THRESHOLD=0.20
# WG D_op base, held fixed:
export PERCEPT_SALIENCE_STAT=block_max PERCEPT_GATE_REQUIRE_FALSE_FIRST=0 PERCEPT_BENCH_MATCHED_PLAN=window PERCEPT_BENCH_MAX_GOALS=3
BENCH="python3 tools/percept_benchmark.py run --live --n 20 --tracks charades-e2e --workers 1 --no-resume"

record () {
  local res; res=$(ls -t eval/local-benchmark/runs/*/results/charades-e2e.json 2>/dev/null | head -1)
  python3 - "$1" "$2" "$res" >> "$LEDGER" <<'PY'
import json, sys
label, cfg, res = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    c = json.load(open(res))["cells"]; r=c.get('e2e:notification_recall')
    row=f"| {label} | {cfg} | {r} | {c.get('e2e:notification_precision')} | {c.get('edge:candidate_recall')} | {c.get('e2e:candidate_calls')} | `{res.split('/runs/')[1].split('/')[0]}` |"
    if r is None: row+="  **VOID**"
except Exception as e: row=f"| {label} | {cfg} | ERR | ERR | ERR | ERR | {type(e).__name__} VOID |"
print(row)
PY
}

echo "waiting for any in-flight benchmark to clear..."
while pgrep -f "percept_benchmark.py run --live" >/dev/null 2>&1; do sleep 30; done
echo "" >> "$LEDGER"; echo "### Addendum 9 — WG D_op + time-refractory (the precision fix)" >> "$LEDGER"
for R in 10 30; do
  label="WGR${R} D_op"
  echo "=== $(date +%H:%M:%S) $label : refractory_s=$R (WG base: block_max+window+gate-relax, <=3 goals) ==="
  PERCEPT_GATE_REFRACTORY_S="$R" $BENCH > "/tmp/refract_${R}.log" 2>&1
  record "$label" "block_max,0,window,3,refractory=$R"
  echo "  $(date +%H:%M:%S) recorded $label"
done
echo "=== REFRACTORY ARM COMPLETE ==="; grep -E "^\| WGR" "$LEDGER"
