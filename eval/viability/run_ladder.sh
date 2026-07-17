#!/usr/bin/env bash
# Ablation-ladder × goal-density runner (QUALITY_GATE_PROTOCOL Addendum 4 + 5).
# Judge = qwen3-vl-32b via OpenRouter (cheap, hosted, parallel — no prepaid-credit wall). Videos judged
# concurrently (PERCEPT_BENCH_VIDEO_CONCURRENCY). block_max cells use the calibrated 0.20 cutoff (Addendum 5).
# Each cell is ONE frozen run, reported as-is; a 429/errored cell is recorded VOID and the loop continues.
set -u
cd /Users/dhivyavijayakumar/git/Percept-sdk
LEDGER=eval/viability/LADDER_LEDGER.md

# --- judge + parallelism config (shared by every cell) ---
export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)"
export PERCEPT_VISION_BACKEND=openrouter
export PERCEPT_VISION_MODEL="qwen/qwen3-vl-32b-instruct"
export PERCEPT_BENCH_VIDEO_CONCURRENCY=8
export PERCEPT_BENCH_BLOCKMAX_THRESHOLD=0.20   # calibrated ~= mean's op-rate; only affects block_max cells
BENCH="python3 tools/percept_benchmark.py run --live --n 20 --tracks charades-e2e --workers 1"

record () {  # $1=label $2=config-desc
  local res
  res=$(ls -t eval/local-benchmark/runs/*/results/charades-e2e.json 2>/dev/null | head -1)
  python3 - "$1" "$2" "$res" >> "$LEDGER" <<'PY'
import json, sys
label, cfg, res = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    c = json.load(open(res))["cells"]
    r = c.get('e2e:notification_recall'); p = c.get('e2e:notification_precision')
    row = f"| {label} | {cfg} | {r} | {p} | {c.get('edge:candidate_recall')} | {c.get('e2e:candidate_calls')} | `{res.split('/runs/')[1].split('/')[0]}` |"
    if r is None: row += "  **VOID**"
except Exception as e:
    row = f"| {label} | {cfg} | ERR | ERR | ERR | ERR | {type(e).__name__} VOID |"
print(row)
PY
}

echo "# Ladder ledger (Addendum 4+5) — judge qwen3-vl-32b, one frozen run per cell" > "$LEDGER"
echo "" >> "$LEDGER"
echo "| cell | config (SAL,RFF,PLAN,MAXG) | recall | precision | edge_recall | calls | run |" >> "$LEDGER"
echo "|---|---|---|---|---|---|---|" >> "$LEDGER"

# Full ladder on qwen (label | SAL | RFF | PLAN | MAXG)
CELLS=(
  "R0 D_stress|mean|1|0|0"
  "R1 D_stress|block_max|1|0|0"
  "R2 D_stress|block_max|1|1|0"
  "R3 D_stress|block_max|0|1|0"
  "R0 D_op|mean|1|0|3"
  "R1 D_op|block_max|1|0|3"
  "R2 D_op|block_max|1|1|3"
  "R3 D_op|block_max|0|1|3"
)
for spec in "${CELLS[@]}"; do
  IFS='|' read -r label SAL RFF PLAN MAXG <<< "$spec"
  echo "=== $(date +%H:%M:%S) running $label : SAL=$SAL RFF=$RFF PLAN=$PLAN MAXG=$MAXG ==="
  PERCEPT_SALIENCE_STAT="$SAL" PERCEPT_GATE_REQUIRE_FALSE_FIRST="$RFF" \
  PERCEPT_BENCH_MATCHED_PLAN="$PLAN" PERCEPT_BENCH_MAX_GOALS="$MAXG" \
    $BENCH > "/tmp/ladder_${label// /_}.log" 2>&1
  record "$label" "$SAL,$RFF,$PLAN,$MAXG"
  echo "  $(date +%H:%M:%S) recorded $label"
done
echo "=== LADDER COMPLETE ==="
cat "$LEDGER"
