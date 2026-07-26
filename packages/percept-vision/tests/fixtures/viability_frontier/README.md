# Synthetic viability-frontier contract fixture

This fixture validates the matched-arm evaluator for the perception runtime and programmable
Watches over live multimodal streams.

Every count is deterministic and synthetic. The two archetypes intentionally preregister different
policies, and each has two operating points that trade event recall against judge-call reduction.
Each case and event denominator is anchored in the manifest, each operating point has a canonical
configuration fingerprint, and every FIRE decision is tied to an event id so precision and false
positives are explicit. `runtime_dispatches` is the execution-completeness denominator; `judge_calls`
is the independently measured provider-call count used for economics because one transition
dispatch can make multiple provider calls. The fixture can prove that matching, denominator checks,
policy evaluation, and Pareto selection work. It cannot qualify a launch or establish a measured
operating point.

Release qualification is unconditionally disabled on this raw manifest-and-row path because it
does not consume a validated `percept.eval.viability_frontier.measured_evidence/v1` bundle or
establish external release authority. Relabeling this fixture as `measured` cannot bypass that
blocker.

The manifest is schema v3. V3 intentionally requires the explicit `runtime_dispatches` field and
renames the unbounded output metric to `judge_call_intensity`; v2 evidence is rejected rather than
silently inferring dispatches from provider calls. Each archetype also preregisters provider-call
cardinality: the transition fixture requires two calls per completed dispatch (and one-to-two for a
missing partial dispatch), while the brief-action fixture requires one. This prevents complete
decisions with zero recorded provider cost from false-greening call reduction.

Run:

```bash
make viability-frontier-contract
```
