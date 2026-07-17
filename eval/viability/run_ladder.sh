#!/usr/bin/env bash
# Ablation-ladder × goal-density runner (QUALITY_GATE_PROTOCOL Addendum 4).
# Steps sequentially through the matrix cells, one live benchmark at a time (workers=1, respects rate
# limits), and appends each result to eval/viability/LADDER_LEDGER.md (gitignored). Idempotent-ish:
# it waits for any in-flight benchmark (the R1 cell) to finish first, records it, then runs the rest.
# Each cell is ONE frozen run, reported as-is. A 429/errored cell is recorded as VOID and the loop continues.
set -u
cd /Users/dhivyavijayakumar/git/Percept-sdk
LEDGER=eval/viability/LADDER_LEDGER.md
BENCH="python3 tools/percept_benchmark.py run --live --n 20 --tracks charades-e2e --workers 1"

record () {  # $1=label $2=config-desc
  local label="$1" cfg="$2"
  local res
  res=$(ls -t eval/local-benchmark/runs/*/results/charades-e2e.json 2>/dev/null | head -1)
  python3 - "$label" "$cfg" "$res" >> "$LEDGER" <<'PY'
import json, sys
label, cfg, res = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    c = json.load(open(res))["cells"]
    row = f"| {label} | {cfg} | {c.get('e2e:notification_recall')} | {c.get('e2e:notification_precision')} | {c.get('edge:candidate_recall')} | {c.get('e2e:candidate_calls')} | `{res.split('/runs/')[1].split('/')[0]}` |"
    if c.get('e2e:notification_recall') is None: row += "  **VOID(null cells)**"
except Exception as e:
    row = f"| {label} | {cfg} | ERR | ERR | ERR | ERR | {type(e).__name__} VOID |"
print(row)
PY
}

echo "waiting for any in-flight benchmark (R1) to finish..."
while pgrep -f "percept_benchmark.py run --live" >/dev/null 2>&1; do sleep 30; done

# header (once)
grep -q "^| cell |" "$LEDGER" 2>/dev/null || {
  echo "# Ladder ledger (Addendum 4) — one frozen run per cell, reported as-is" > "$LEDGER"
  echo "" >> "$LEDGER"
  echo "| cell | config (SAL,RFF,PLAN,MAXG) | recall | precision | edge_recall | calls | run |" >> "$LEDGER"
  echo "|---|---|---|---|---|---|---|" >> "$LEDGER"
}

# R0 D_stress is already done (recall 0.30). Record R1 (the just-finished in-flight block_max cell):
record "R1 D_stress" "block_max,1,0,0"

# Remaining cells: label | SAL | RFF | PLAN | MAXG
CELLS=(
  "R2 D_stress|block_max|1|1|0"
  "R3 D_stress|block_max|0|1|0"
  "R0 D_op|mean|1|0|3"
  "R1 D_op|block_max|1|0|3"
  "R2 D_op|block_max|1|1|3"
  "R3 D_op|block_max|0|1|3"
)
for spec in "${CELLS[@]}"; do
  IFS='|' read -r label SAL RFF PLAN MAXG <<< "$spec"
  echo "=== running $label : SAL=$SAL RFF=$RFF PLAN=$PLAN MAXG=$MAXG ==="
  PERCEPT_SALIENCE_STAT="$SAL" PERCEPT_GATE_REQUIRE_FALSE_FIRST="$RFF" \
  PERCEPT_BENCH_MATCHED_PLAN="$PLAN" PERCEPT_BENCH_MAX_GOALS="$MAXG" \
    $BENCH > "/tmp/ladder_${label// /_}.log" 2>&1
  record "$label" "$SAL,$RFF,$PLAN,$MAXG"
  echo "  recorded $label"
done
echo "=== LADDER COMPLETE ==="
cat "$LEDGER"
