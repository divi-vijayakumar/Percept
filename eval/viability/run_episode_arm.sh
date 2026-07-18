#!/usr/bin/env bash
# Episode-memory arm (Addendum 8): PLAN=episode on the block_max base. Waits for any in-flight benchmark
# (e.g. L3's measurement) to finish first to avoid OpenRouter rate-limit contention. Appends to LADDER_LEDGER.
set -u
cd /Users/dhivyavijayakumar/git/Percept-sdk
LEDGER=eval/viability/LADDER_LEDGER.md
export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)"
export PERCEPT_VISION_BACKEND=openrouter PERCEPT_VISION_MODEL="qwen/qwen3-vl-32b-instruct"
export PERCEPT_BENCH_VIDEO_CONCURRENCY=8 PERCEPT_BENCH_BLOCKMAX_THRESHOLD=0.20
BENCH="python3 tools/percept_benchmark.py run --live --n 20 --tracks charades-e2e --workers 1"

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

echo "waiting for any in-flight benchmark (L3 measurement) to clear..."
while pgrep -f "percept_benchmark.py run --live" >/dev/null 2>&1; do sleep 30; done
echo "" >> "$LEDGER"; echo "### Addendum 8 — episode-memory arm (block_max base, PLAN=episode)" >> "$LEDGER"
CELLS=(
  "E D_op|block_max|1|episode|3"
  "EG D_op|block_max|0|episode|3"
)
for spec in "${CELLS[@]}"; do
  IFS='|' read -r label SAL RFF PLAN MAXG <<< "$spec"
  echo "=== $(date +%H:%M:%S) $label : SAL=$SAL RFF=$RFF PLAN=$PLAN MAXG=$MAXG ==="
  PERCEPT_SALIENCE_STAT="$SAL" PERCEPT_GATE_REQUIRE_FALSE_FIRST="$RFF" \
  PERCEPT_BENCH_MATCHED_PLAN="$PLAN" PERCEPT_BENCH_MAX_GOALS="$MAXG" \
    $BENCH > "/tmp/episode_${label// /_}.log" 2>&1
  record "$label" "$SAL,$RFF,$PLAN,$MAXG"
  echo "  $(date +%H:%M:%S) recorded $label"
done
echo "=== EPISODE ARM COMPLETE ==="; grep -E "^\| (E |EG )" "$LEDGER"
