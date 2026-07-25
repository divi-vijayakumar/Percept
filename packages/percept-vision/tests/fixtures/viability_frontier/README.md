# Synthetic viability-frontier contract fixture

This fixture validates the matched-arm evaluator for the perception runtime and programmable
Watches over live multimodal streams.

Every count is deterministic and synthetic. The two archetypes intentionally preregister different
policies, and each has two operating points that trade event recall against judge-call reduction.
Each case and event denominator is anchored in the manifest, each operating point has a canonical
configuration fingerprint, and every FIRE decision is tied to an event id so precision and false
positives are explicit. The fixture can prove that matching, denominator checks, policy evaluation,
and Pareto selection work. It cannot qualify a launch or establish a measured operating point.

Release qualification is unconditionally disabled until a separately verified
`percept.eval.viability_frontier.measured_evidence/v1` adapter supplies provenance-preserving
matched evidence. Relabeling this fixture as `measured` cannot bypass that blocker.

Run:

```bash
make viability-frontier-contract
```
