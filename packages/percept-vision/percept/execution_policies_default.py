"""The shipped :class:`~percept.execution_policy.ExecutionPolicy` set (ADR-008).

Five policies, each named ``{id}@1`` and built ONLY from the tunables-registry paths
:data:`percept.tunables.REGISTRY` assigns owner ``"ExecutionPolicy"`` (see
:mod:`percept.execution_policy`'s module docstring for the full list). No new temporal shapes, no
``event_capsule`` defaults — every value below is an existing ``Settings``/``WatchProfile.cadence``
field.

- :data:`BALANCED_V1` — today's shipped defaults. Deliberately EMPTY ``values``: a policy that
  contributes nothing to the precedence chain is, by construction, byte-for-byte identical to
  resolving with no ``execution_policy`` at all (``tests/test_execution_policy.py`` proves this).
- :data:`BURST_V1` — aggressive cadence (higher quiet floor, higher escalation ceiling, quicker
  and longer-held saccades) plus a deeper pre-roll ring, for a deployment that would rather spend
  more judge calls and more pre-roll memory than miss a fast event.
- :data:`LOW_COST_V1` — sparser judge dispatch: a lower, slower cadence; the cache cascade
  short-circuit turned ON (``Settings.cache.enabled``) with a real threshold so repeat frames skip
  the judge; the local router set to defer to the conductor less often
  (``routing.confidence_threshold`` lowered); and tighter entity/episode/pre-roll budgets.
- :data:`PRIVACY_EDGE_V1` — a minimal-retention posture using only knobs that EXIST: the cache
  cascade OFF (nothing hashed/retained for cross-frame matching) and the entity/episode/pre-roll
  ceilings turned down to the smallest still-functional bounds. This does NOT enforce "edge-only"
  placement — ``ExecutionPolicy`` owns no placement/backend-selection field (that is
  ``DeploymentConfig``'s, per the tunables registry) — it only turns down every retention knob
  ``ExecutionPolicy`` actually owns. Inventing an enforcement mechanism here would be out of this
  policy's scope.
- :data:`STRICT_ADMISSION_V1` — signal-only admission: ``Settings.edge.coast_enabled`` set False so
  a SourceProfile-declared coast floor is NOT bound into the runtime. Judge calls are spent only on
  tuned-signal escalations (and audio wakes), never on a fixed coast cadence over static content.
  A profile that declares ``coast_required`` is overridden LOUDLY (the resolver emits a receipt
  warning and the run resolves ``applied_with_warnings``), never silently.

Import this module for its side effect (seeds :data:`~percept.execution_policy.DEFAULT_EXECUTION_POLICY_REGISTRY`)
before reading the registry directly; :func:`percept.execution_policy.get_execution_policy` already
does this for you.
"""

from __future__ import annotations

from .execution_policy import ExecutionPolicy, register_execution_policy

#: Today's shipped defaults. Empty values -> contributes nothing to the precedence chain -> a
#: resolve() call with this policy is byte-for-byte identical to omitting execution_policy entirely.
BALANCED_V1 = ExecutionPolicy(
    id="balanced",
    version="1",
    values={},
    notes="today's shipped defaults; a deliberate no-op (contributes no override).",
)

#: Aggressive cadence + deeper pre-roll: spend more to catch more.
BURST_V1 = ExecutionPolicy(
    id="burst",
    version="1",
    values={
        "WatchProfile.cadence.base_fps": 8.0,  # default 4.0 — higher quiet floor
        "WatchProfile.cadence.max_fps": 30.0,  # default 15.0 — higher cost ceiling
        "WatchProfile.cadence.escalate_threshold": 0.08,  # default 0.15 — saccade triggers sooner
        "WatchProfile.cadence.hold_s": 1.0,  # default 0.5 — hold the escalation longer
        "Settings.events.preroll_max_bytes": 268_435_456,  # default 128 MiB -> 256 MiB pre-roll ring
    },
    notes="aggressive cadence ceilings + a deeper pre-roll ring; spends more to miss less.",
)

#: Sparser judge dispatch + tighter budgets: spend less, accept a coarser cadence.
LOW_COST_V1 = ExecutionPolicy(
    id="low_cost",
    version="1",
    values={
        "WatchProfile.cadence.base_fps": 2.0,  # default 4.0 — lower quiet floor
        "WatchProfile.cadence.max_fps": 8.0,  # default 15.0 — lower cost ceiling
        "WatchProfile.cadence.escalate_threshold": 0.25,  # default 0.15 — escalate less readily
        "WatchProfile.cadence.hold_s": 0.25,  # default 0.5 — release the saccade sooner
        "Settings.cache.enabled": True,  # default False — wire the cost cascade short-circuit in
        "Settings.cache.threshold": 0.15,  # default 0.0 — a real cutoff now that caching is on
        "Settings.cache.audit_rate": 0.02,  # default 0.05 — fewer audited HITs (cheaper)
        "Settings.routing.confidence_threshold": 0.45,  # default 0.6 — defer to the conductor less often
        "Settings.entity.max_entities": 32,  # default 64 — tighter eviction bound
        "Settings.episodes.max_episodes": 32,  # default 64 — tighter eviction bound
        "Settings.events.preroll_max_bytes": 67_108_864,  # default 128 MiB -> 64 MiB pre-roll ring
    },
    notes="sparser judge dispatch (cache cascade on, router defers less) + tighter resource budgets.",
)

#: Minimal-retention posture, built only from ExecutionPolicy-owned retention knobs.
PRIVACY_EDGE_V1 = ExecutionPolicy(
    id="privacy_edge",
    version="1",
    values={
        "Settings.cache.enabled": False,  # default False — explicit: no cross-frame hash retention
        "Settings.entity.max_entities": 16,  # default 64 — smallest still-functional entity bound
        "Settings.episodes.max_episodes": 16,  # default 64 — smallest still-functional episode log
        "Settings.events.preroll_max_bytes": 16_777_216,  # default 128 MiB -> 16 MiB pre-roll ring
    },
    notes=(
        "minimal-retention posture using only knobs ExecutionPolicy owns today; does not itself "
        "enforce an edge-only placement (ExecutionPolicy owns no placement/backend field)."
    ),
)

#: Signal-only admission: a SourceProfile-declared coast floor is NOT bound into the runtime.
STRICT_ADMISSION_V1 = ExecutionPolicy(
    id="strict_admission",
    version="1",
    values={
        "Settings.edge.coast_enabled": False,  # default True — do not bind any coast floor
    },
    notes=(
        "signal-only admission: a SourceProfile-declared coast floor is NOT bound; judge calls are "
        "spent only on tuned-signal escalations (and audio wakes). A profile-declared coast_required "
        "is overridden LOUDLY (receipt warning), never silently."
    ),
)

_SHIPPED: tuple[ExecutionPolicy, ...] = (
    BALANCED_V1,
    BURST_V1,
    LOW_COST_V1,
    PRIVACY_EDGE_V1,
    STRICT_ADMISSION_V1,
)

for _policy in _SHIPPED:
    register_execution_policy(_policy)

__all__ = [
    "BALANCED_V1",
    "BURST_V1",
    "LOW_COST_V1",
    "PRIVACY_EDGE_V1",
    "STRICT_ADMISSION_V1",
]
