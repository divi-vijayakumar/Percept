"""Generate the Charades e2e deep-dive from a traced run's per_goal.json.

Classifies WHY recall/precision are low, at the (video, goal) level:
- recall misses: GATE miss (judge said yes in-segment but no fire) vs PERCEPTION miss (judge never yes in-segment)
- false fires: RE-FIRE (goal that also had a hit) vs OUT-OF-SEGMENT (fire where no labeled segment)
Writes eval/viability/CHARADES_DEEPDIVE.md (gitignored, local-only).
"""
import json, sys
from pathlib import Path
from collections import Counter

REPO = Path("/Users/dhivyavijayakumar/git/Percept-sdk")
run = sorted((REPO/"eval/viability/results/charades").glob("*/per_goal.json"))[-1]
pg = json.loads(run.read_text())
metrics = json.loads((run.parent/"metrics.json").read_text())

seg_total = hits = gate_miss = perc_miss = 0
fp_total = refire = oos = 0
goals_hit = 0
rows = []
for g in pg:
    g_hit = False
    for s in g["segments"]:
        seg_total += 1
        if s["hit_at"] is not None:
            hits += 1; g_hit = True
        elif s["judge_yes_in_seg"]:
            gate_miss += 1
        else:
            perc_miss += 1
    if g_hit: goals_hit += 1
    ff = g["false_fires"]
    fp_total += len(ff)
    if g_hit: refire += len(ff)   # extra fires in a goal that DID hit a segment
    else: oos += len(ff)          # fires in a goal with no hit at all
    rows.append(g)

def pct(a, b): return f"{100*a/b:.0f}%" if b else "-"

lines = []
lines.append("# Charades e2e deep-dive — why recall/precision are low\n")
lines.append(f"**Run:** `{run.parent.name}` · model `{metrics['model']}` · gate `require_false_first=True` · tolerance `[onset, end+1.0s]`")
lines.append(f"**Aggregate:** recall **{metrics['recall']:.2f}** / precision **{metrics['precision']:.2f}** · tp {metrics['tp']} fp {metrics['fp']} fn {metrics['fn']} · {metrics['videos']} videos, {metrics['goals']} goals\n")

lines.append("## Recall — where the misses come from")
lines.append(f"- segments total: **{seg_total}** · hit: **{hits}** ({pct(hits,seg_total)}) · missed: **{gate_miss+perc_miss}**")
lines.append(f"- of the misses: **GATE miss** (judge said YES in-segment, gate didn't fire): **{gate_miss}** ({pct(gate_miss,gate_miss+perc_miss)}) · **PERCEPTION miss** (judge never yes in-segment): **{perc_miss}** ({pct(perc_miss,gate_miss+perc_miss)})")
lines.append("  - GATE misses are fixable (rising-edge already fired/latched, or dwell). PERCEPTION misses are model/label limits.\n")

lines.append("## Precision — where the false fires come from")
lines.append(f"- false fires total: **{fp_total}**")
lines.append(f"- **RE-FIRE** (extra fires inside a goal that also HIT a real segment — flicker re-arm, no time-refractory): **{refire}** ({pct(refire,fp_total)})")
lines.append(f"- **OUT-OF-SEGMENT** (fires in a goal with NO hit — judge said yes where nothing is labeled: over-detection or unlabeled-but-real Charades activity): **{oos}** ({pct(oos,fp_total)})\n")
lines.append(f"- goals that hit >=1 segment: **{goals_hit}/{len(pg)}**\n")

lines.append("## Read")
if refire >= oos:
    lines.append(f"- Precision is dominated by **RE-FIRES ({pct(refire,fp_total)})** — the gate fires repeatedly within one true episode (no time-refractory). This is the single highest-value fix and is squarely a gate-tuning issue, not perception.")
else:
    lines.append(f"- Precision is dominated by **OUT-OF-SEGMENT fires ({pct(oos,fp_total)})** — the judge fires where nothing is labeled, i.e. over-detection and/or Charades' sparse labels (correct detections of unlabeled activities scored as false).")
if gate_miss >= perc_miss:
    lines.append(f"- Recall misses are mostly **GATE** ({pct(gate_miss,gate_miss+perc_miss)}) — the model saw the action in-segment but the gate didn't fire. Fixable.")
else:
    lines.append(f"- Recall misses are mostly **PERCEPTION** ({pct(perc_miss,gate_miss+perc_miss)}) — the model didn't confirm the action in-segment. Model/label limit.\n")

# worst offenders for precision (most false fires)
lines.append("\n## Top false-fire goals (precision drain)")
lines.append("| video | condition | judge yes-frac | #segments | #hits | false fires |")
lines.append("|---|---|---|---|---|---|")
for g in sorted(rows, key=lambda r: -len(r["false_fires"]))[:12]:
    nh = sum(1 for s in g["segments"] if s["hit_at"] is not None)
    lines.append(f"| {g['video']} | {g['condition'][:42]} | {g['judge_yes_frac']} | {len(g['segments'])} | {nh} | {len(g['false_fires'])} ({g['false_fires'][:5]}) |")

# a few recall-miss examples split by class
lines.append("\n## Recall-miss examples")
gm = [(g,s) for g in rows for s in g["segments"] if s["hit_at"] is None and s["judge_yes_in_seg"]][:6]
pm = [(g,s) for g in rows for s in g["segments"] if s["hit_at"] is None and not s["judge_yes_in_seg"]][:6]
lines.append("**GATE misses (judge saw it in-segment, no fire):**")
for g,s in gm:
    lines.append(f"- `{g['video']}` \"{g['condition'][:40]}\" seg {s['seg']} — judge YES in-seg @ {s['judge_yes_in_seg']}, fires {g['fires'][:5]}")
lines.append("\n**PERCEPTION misses (judge never yes in-segment):**")
for g,s in pm:
    lines.append(f"- `{g['video']}` \"{g['condition'][:40]}\" seg {s['seg']} — judge yes-frac {g['judge_yes_frac']} over clip, fires {g['fires'][:5]}")

(REPO/"eval/viability/CHARADES_DEEPDIVE.md").write_text("\n".join(lines))
print("wrote eval/viability/CHARADES_DEEPDIVE.md")
print(f"recall misses: gate={gate_miss} perception={perc_miss} | false fires: refire={refire} out-of-seg={oos}")
