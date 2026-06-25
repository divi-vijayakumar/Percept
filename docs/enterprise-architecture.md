# Enterprise architecture

Percept OSS proves the cognition loop: state a goal, gate cheap signals, spend expensive perception only
when it matters, fire once on the rising edge, and ask instead of guessing. The enterprise version turns
that loop into an operated service with scale, SLAs, governance, and continuous model improvement.

## OSS vs Enterprise

| layer | OSS | enterprise |
|---|---|---|
| Cognition core | local Python SDK, fake/live backends, runtime worker | hosted multi-tenant control plane and managed workers |
| Edge | Python `percept.edge` and npm `percept-edge` | managed device fleet, capability inventory, rollout rings |
| Runtime | process-local video/webcam/RTSP loop | supervised workers, reconnect/backoff policy, source health, regional routing |
| Evidence | local incidents/snippets | durable incident store, retention, redaction, review workflows |
| Benchmark | public `percept-bench`, draft scorecards | pinned golden sets, competitor baselines, regression gates per tenant |
| Operations | logs/tests | SLA metrics, audit log, billing, policy controls |

## Reference Topologies

### A. Server-side Runtime

Camera/video/RTSP streams enter a managed worker. The Python edge gate runs near the worker, the cognition
core owns goals and incidents, and perception calls go to the configured model backend only on escalations.

Use this when cameras are already network-addressable and central processing is acceptable.

### B. Device Edge

`percept-edge` runs on the device/browser/edge host. It advertises capabilities, gates frames locally, and
sends `Tier0Signal` / `BoundaryReport` upstream. The server receives fewer frames and spends fewer model
tokens.

Use this when privacy, bandwidth, or latency makes local filtering important.

## Control Plane

Enterprise needs a control plane around the OSS loop:

- Tenant/project/source inventory.
- Goal and detector policy management.
- Edge capability registry and version pinning.
- Runtime worker orchestration.
- Incident review and feedback.
- Benchmark/regression runs before rollout.
- Model/backend routing with cost ceilings.

## Data Governance

Percept should treat media as sensitive by default.

- Consent profile per source and goal.
- Snippet redaction while retaining incident metadata.
- DLP hooks before webhook/export.
- Retention policy per tenant.
- Audit log for goal changes, source access, incident reads, and outbound actions.
- Region pinning for media and embeddings.

## SLA Hooks

Production runtime hardening targets:

- Source health: open failures, read failures, reconnect count, frame lag, queue drops.
- Edge health: salience rate, detector boundary rate, parity version.
- Perception health: call rate, latency, timeout, error rate, model spend.
- Cognition health: fires, asks, false-fire labels, refractory suppressions.
- Sink health: webhook attempts, retries, non-2xx, dead-letter count.

## Continuous Innovation Loop

The enterprise flywheel is benchmark-driven:

1. Incidents and asks feed human review.
2. Reviewed examples become tenant-private golden cases.
3. Public benchmark tracks catch OSS regressions.
4. Competitor/reference baselines stay machine-readable.
5. Runtime rollout gates require no false-fire regression and bounded cost/latency.

## v1.0 Hardening Checklist

- Stable public APIs for `percept`, `percept.edge`, `percept.runtime`, `percept.bench`, and `percept-edge`.
- Reproducible benchmark card with pinned data manifest.
- Runtime retries, webhook dead-lettering, health events, and source backpressure policy.
- Contract compatibility tests across Python and JS.
- Security review for media handling, webhook signing, secret loading, and audit logs.
- Upgrade guide from OSS local runtime to managed enterprise topology.
