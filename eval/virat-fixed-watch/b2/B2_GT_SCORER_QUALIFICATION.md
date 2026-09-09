# B2 GT transformer and scorer qualification

Status: **ENGINEERING PASS**

The F2-Z1 transformer and one-to-one scorer are implemented independently of the Percept runtime.
Qualification labels were not opened. No provider call or paid work occurred.

Measured verification:

- Nine deterministic tests pass.
- An independently implemented brute-force oracle agrees with the scorer on all `100` enumerated
  three-event/three-fire timeline cases.
- Synthetic fixtures verify boundary inclusion, clean outside evidence, left-censoring, gaps,
  excluded/ambiguous frames, minimum height, occlusion, dwell, maximum-cardinality matching, ASK,
  duplicates, misses, ACK timeliness, stage percentiles, and explicit reconciliation.
- The exposed calibration group `0502` probe admitted `21` files, rejected `3` malformed annotation
  files, and measured `16` provisional entries at each declared dwell (`0.0`, `0.25`, `0.5` seconds).

`BUG-B2-001` caught and fixed a dwell off-by-one before any validation result was produced.
`BUG-B2-002` caught and fixed cross-source timestamp matching while integrating B4 detection caches,
also before any aggregate calibration or validation result was produced.

The legacy eight-column VIRAT object format has no occlusion field. The parser records
`not_provided`; the calibration counts are therefore provisional engineering evidence. The complete
zone audit must supply/adjudicate occlusion and exclusions before PILOT-276 quality scoring. This does
not weaken the publication hold.

Next action: commit this atomic B2 implementation, then execute B3's zero-paid production-path
rehearsal with deterministic perception and a durable local receiver.
