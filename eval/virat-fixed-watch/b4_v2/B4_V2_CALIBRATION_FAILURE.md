# B4 v2 persistent-tracking calibration

Status: **FAIL — architecture stop; validation remains sealed**

The founder-approved v2 candidate was frozen before its complete calibration result. It reused the
same immutable Faster R-CNN detection cache and changed only the coherent tracking/state layer:
motion-aware global assignment, persistent identity, outside observability, boundary ambiguity,
duplicate detection suppression, scene refractory, ASK, and exact scene-goal rising-edge behavior.

All 972 predeclared configurations were scored across all 21 admitted exposed calibration sources.
No configuration passed. No configuration reached full recall or zero provisional FP; the maximum
TP was 12 of 16 and the minimum FN was 4.

The best-ranked result measured 12 TP, 22 provisional FP, 4 provisional FN, precision
0.35294117647058826, recall 0.75, 34 fires, zero emitted duplicates, 4 misses, and 5 ASK. The exact
source-level distribution is in [`b4-v2-calibration-result.json`](b4-v2-calibration-result.json).
The gate suppressed five simultaneous or consecutive raw tracker outcomes, demonstrating that the
reported fire count follows Percept's one-goal rising-edge semantics.

Detector latency remained within the predeclared calibration bound: p50 0.745623 s, p95 0.803756 s,
p99 0.868332 s, max 1.313474 s. Tracker-state latency was p50 0.00000767 s, p95 0.00002817 s, p99
0.00004871 s, max 0.000113 s. Reconciliation was exact at 21 sources and 2,230 samples.

Timely recall and source-onset-to-receiver-ACK latency were not measured. Provider calls and provider
cost were zero; all-in local cost is absent because no resource rate was supplied. Complete-zone and
occlusion audit remains pending, so every quality count and ratio is provisional.

Compared with v1, persistent state reduced provisional FP from 26 to 22 and eliminated three emitted
duplicates, but TP fell from 13 to 12 and FN rose from 3 to 4. The fixed detection evidence therefore
cannot meet the joint gate through tracking/state changes alone. A materially different detector,
model, cadence/evidence architecture, or newly authorized resource is required.

Per the requested stop rule, validation, qualification labels, PILOT-276, provider execution, and the
72-hour soak were not opened.
