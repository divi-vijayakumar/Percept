---
title: Hosted BYOK invocation contract
updated: 2026-07-26
applies_to: percept-vision 1.0.0
status: stable
---
# Hosted BYOK invocation contract

`percept.hosted_byok` is the public boundary for a hosted service that receives an end user's provider
credential for one model invocation. It is separate from `Percept.create(...)`,
`resolve_vision(...)`, and the backend registry because those existing APIs intentionally support
process-environment configuration and long-lived adapters.

Contract version: `percept.HOSTED_BYOK_CONTRACT_VERSION` (`"1.1.0"`).
Executable versions: `percept.HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS` (`("1.0.0", "1.1.0")`).

`1.1.0` is purely additive: it introduces the gated detection capability
(`vision.detect.gated`) and its DTOs. Nothing in `1.0.0` changed shape or behaviour, and `judge()`
is byte-for-byte the same call, so a deployment pinned to `contract_version="1.0.0"` keeps working
unmodified. A `1.0.0` request is refused only for capabilities that version never carried — it is
never silently upgraded into a surface its caller did not agree to.

`factory.build_info()` (and `vision.build_info()` on a bound adapter) reports the contract version,
the executable versions, the per-provider capability table, and the detection bounds below.
`percept.build_info().hosted_byok_contract_version` reports the same version from the package
accessor. It is deliberately absent from `build_info().to_dict()`, which is the frozen four-field
build-stamp receipt that published benchmark receipts seal and re-verify.

## Cloud usage

Create one immutable request for one provider call, then call the factory's `judge` method:

```python
import asyncio
import time

from percept import (
    HostedByokAdapterFactory,
    HostedByokRequest,
    HostedByokSubprocessSupervisor,
)

hosted_byok = HostedByokAdapterFactory(
    process_supervisor=HostedByokSubprocessSupervisor()
)

async def judge_for_request(
    *,
    provider_credential: str,
    lease_id: str,
    principal_id: str,
    request_disconnected: asyncio.Event,
    jpeg: bytes,
):
    invocation = HostedByokRequest(
        provider="openrouter",
        model="qwen/qwen3-vl-32b-instruct",
        credential=provider_credential,
        credential_lease_id=lease_id,
        principal_id=principal_id,
        timeout_s=15.0,
        deadline_unix_s=time.time() + 20.0,
        cancellation=request_disconnected,
    )
    return await hosted_byok.judge(
        invocation,
        "a person has entered the restricted area",
        ("image/jpeg", jpeg),
    )
```

To use the normal Percept cognition/runtime surface, bind the same request to the public `Vision`
interface. The bound adapter carries no client, backend, or result cache; each goal judgment creates a
fresh provider client:

```python
from percept import Goal, Percept

vision = hosted_byok.bind(invocation)
core = Percept.create(vision=vision, mode="offline", discover_plugins=False)
core.add_goal(Goal(id="restricted-entry", condition="a person enters", say="Entry detected"))
events = await core.perceive_judged(("image/jpeg", jpeg))
```

`mode="offline"` above prevents unrelated Percept seams from requiring credentials; it does not fake
the explicitly supplied Hosted BYOK vision adapter. A hosted process may keep the factory, but should
bind a new request for each incoming credential lease.

For Gemini, use `provider="gemini"` and an exact `gemini-*` model id. Gemini's fused image/audio
path must also declare `capability="vision.judge.fused"` and pass both `frame=` and `audio=`.
Plain `capability="vision.judge"` requires an image and forbids audio for every provider; audio-only
input is not part of this contract. OpenRouter accepts exact `vendor/model` ids and does not advertise
the fused capability. The anchor, temporal `frames`, exemplar, and audio carriers are validated by
media kind before construction. A typed `AudioFrame` must be exact, contain exactly one non-empty
PCM or encoded payload, have valid rate/layout/format metadata, and use an explicit `audio/*` MIME for
encoded bytes; missing or non-audio encoded MIME is rejected before a client is constructed. Contract
1.0 rejects the presence (even a false value) of reserved control keys such as `process_qa`,
`observe_step`, `running_answer`, `claim_mode`, `extractor`, and `judge_static_context` rather than
silently routing into an unadvertised provider subprotocol.

A fused request-bound Gemini adapter composes with a normal compiled `WatchPlan(sense="fused")`.
Watch preparation captures the immutable request and trusted Hosted BYOK dispatch in the runtime's
sealed audio-input binding. A plain request-bound adapter cannot arm an audio or fused Watch, and an
OpenRouter request cannot declare fused capability.

`timeout_s` is relative to entry into `judge`; `deadline_unix_s` is an absolute UNIX timestamp. If
both are supplied, the earlier limit wins. Omitting both uses the contract's 120-second safety ceiling.
`asyncio.Event` implements the public cancellation protocol. Production calls require an exact,
immutable `HostedByokSubprocessSupervisor`; there is no boolean or virtual health hook that can waive
this requirement. Gemini and OpenRouter are both classified `production_process_isolated` because their
HTTP stacks may use DNS threads, AnyIO worker threads, or implementation-dependent background work.
The parent starts one monotonic deadline before validation, launches a clean interpreter under an
irreversible OS boundary that denies all child-process creation, and authenticates that boundary before
sending the credential-bearing payload. On success, failure, cancellation, or timeout it terminates and
waits for the worker before releasing the outcome. A provider cannot escape with `fork`, `posix_spawn`,
`setsid`, or a reparented descendant because the boundary permits no descendant to be created; clean
`Context` tasks, deferred event-loop Handles, close workers, and threads die with the waited worker.
The process-isolation contract version is `HOSTED_BYOK_PROCESS_ISOLATION_VERSION` (`"2.3.0"`).
The v2.3 supervisor uses validated root-owned Darwin `sandbox-exec`, or an authenticated Linux/AMD64
seccomp filter that allows `CLONE_THREAD` while denying process-form `clone`, `fork`, `vfork`, `clone3`,
`execve`, and `execveat`. Linux provider DNS, default-executor, AnyIO, and HTTP worker threads therefore
remain available without permitting a child process. Unsupported or ineffective platform capabilities
fail closed before the credential payload is sent. The child independently attests its clean environment,
absolute interpreter identity, native platform/ELF architecture, emulation exclusions, and active
`no_new_privs`/seccomp/fork/exec/thread state. Parent IPC uses a bounded four-byte network-order frame
header with nonblocking incremental send/receive, so partial frames remain deadline- and
cancellation-responsive. It assumes ownership immediately when `Popen` returns: every later socket
setup, readiness, framing, or cleanup failure closes both peers and synchronously kills and joins the
exact worker.
Request and Verdict IPC are bounded to 32 MiB;
the wire is canonical tagged JSON inside authenticated socket frames, never pickle.
Exact recursive carrier validation rejects hostile subclasses, duplicate keys, non-finite values,
unsupported DTOs, malformed or oversized payloads before construction/unmarshal. The supervisor owns
a recursively frozen validation graph, and the clean child independently repeats exact validation on
the canonical decoded values before constructing a provider factory. Every non-credential input is also
rejected if it contains the credential. The credential crosses only the private inherited socket: it is
never placed in the worker's argv or environment. The allowlisted child environment is installed by the
parent before the interpreter, `sitecustomize`, or SDK starts, excluding provider configuration,
proxies, custom CA paths, TLS key logging, dynamic-loader controls, and other inherited transport
routing. Hosted Gemini and OpenRouter HTTP clients also set environment trust off explicitly.

`factory.transport_safety(request)` exposes this machine-readable classification. Every built-in route
sets `requires_process_supervisor=True`; a factory without the exact SDK supervisor fails before provider
construction. Factory routing/configuration is immutable, subclasses and instance method shadows are
rejected, and the permission path uses a private non-virtual classifier. Injected backend/client seams
are classified `test_only_in_process`, cannot be combined with a production supervisor, and remain only
for deterministic SDK conformance tests. Callers protecting a credential lease or rotation with a
lifecycle guard must hold it for the complete `await`.

Validation, canonical serialization, clean startup, boundary authentication, provider construction,
execution, teardown, worker termination/wait, result decoding, and final result acceptance all consume
the single monotonic budget created at public `judge()` entry. An over-budget construction is rejected
when it returns, before any provider task is scheduled.
Cancellation is checked before and after construction and again immediately before dispatch, so a
cancellation observed during construction spends zero provider calls. The monotonic deadline is also
checked after provider completion and after the worker is terminal, so neither a coroutine that blocked
the event loop nor slow teardown can have a late result accepted. Injected test-only transports remain
in-process and do not carry this production containment claim.

## Gated detection — attributing an incident to a moment (`vision.detect.gated`)

`judge()` answers *is this true*. It returns one verdict for the evidence it was handed, so a
hosted run over a whole clip yields one verdict with no *when* and no *how many*. `detect_gated()`
answers the other half: **which moments in this clip are worth judging**.

It is the SDK's own tier-0 salience gate (`percept.harness.tier0.MotionGate` — the same gate the
runtime escalates on), run over an ordered sequence of gate frames. There is no second gate, and
none may be added: two gates drift, and a hosted tenant would then attribute incidents under
different admission semantics than the SDK's own.

**This route contacts no provider and spends no provider call.** Its entire output is temporal
identity plus the reason a moment was admitted.

```python
from percept import (
    HostedByokAdapterFactory,
    HostedByokDetectionGoal,
    HostedByokGateFrame,
    HostedByokDetectionStarved,
    HostedByokRequest,
)

request = HostedByokRequest(
    provider="gemini",
    model="gemini-3.5-flash",
    credential=provider_credential,
    credential_lease_id=lease_id,
    principal_id=principal_id,
    capability="vision.detect.gated",
)
vision = HostedByokAdapterFactory().bind(request)

goals = [HostedByokDetectionGoal(goal_id="dock", condition="a person enters", refractory_s=15.0)]
frames = [
    HostedByokGateFrame(
        width=48, height=36, pixels=luma,       # row-major 8-bit luma, len == width*height
        frame_index=i, offset_s=i / fps, timestamp_unix_s=clip_start_unix + i / fps,
    )
    for i, luma in enumerate(gate_frames)
]

try:
    proposals = await vision.detect_gated(goals, frames)
except HostedByokDetectionStarved as starved:
    # NOT an empty list — see "Zero admissions" below.
    report(starved.to_dict())
else:
    for proposal in proposals:
        # proposal.frame_index / .offset_s / .timestamp_unix_s bind an incident to the moment.
        verdict = await vision.judge(goals[0].condition, full_frame_at(proposal.frame_index))
```

`stream_gated_detections(goals, frames)` is the same route as an async iterator, for a long clip a
caller wants to act on incrementally. The whole route is preflighted before the iterator yields
anything, and a caller that stops early stops paying for the rest of the clip.

### What the route expects on the wire

Detection is defined over **tier-0 gate frames, not full frames**. `percept.runtime.source`
downsamples every captured frame to `GATE_W x GATE_H` (48x36 grayscale, 1,728 luma bytes) *before*
the gate computes anything, so a full-resolution frame is never needed to decide *when*. The
boundary enforces this: a `HostedByokGateFrame` larger than
`percept.HOSTED_BYOK_MAX_GATE_FRAME_PIXELS` (1,728) is refused, so a full frame cannot cross the
seam by accident.

The practical budget: 1,200 gate frames (600 s at 2 fps) is **~2 MB** against the 32 MiB boundary
ceiling. Full frames are then fetched by the caller only for the moments detection admitted, and
handed to `judge()`.

Every frame in one call must share one exact gate resolution, carry a strictly increasing
`frame_index`, and be ordered by non-decreasing `offset_s`. `offset_s` is what the gate's
cadence/schedule policy is evaluated against; `timestamp_unix_s` is optional and is echoed back
untouched.

### Bounded worst case

`refractory_s` is the minimum separation between two proposals for one goal. It is the gate's own
per-goal cadence (`WatchCondition.cadence_s`), shared with every other admission path for that
goal, so a proposal cannot double-spend inside the window.

At the default `refractory_s=15.0`, a **600 s clip yields at most 40 proposals per goal** —
measured, not assumed: a 1,200-frame clip whose every frame is maximally salient admits at
`t = 0.5, 15.5, … 585.5`. (The first frame has no predecessor, so it can never admit; the ceiling is
`⌈(duration − 1/fps) / refractory_s⌉`.) **Provider calls spent by detection: 0.** The number that
matters downstream is the caller's own judge calls, capped at one per proposal: **≤ 40 per goal per
600 s**, versus 1,200 for an ungated per-frame loop.

### Thresholds follow the deployment's tier-0 configuration

A goal that names no `threshold` gets the calibrated cutoff for the resolved salience statistic,
using the same per-stat rule `compile_watchspec` applies: `block_max` (the shipped default eye) uses
`Settings.edge.motion_gate_block_max` (0.20), `p95` uses `motion_gate_p95` (0.15), and `mean` uses
`motion_gate` (0.10). The localized statistics sit on a higher value scale than a whole-frame mean,
so naming a statistic without a threshold never silently applies the mean's.

`condition` is **not** interpreted: it rides through as the watch condition's `anchor_text` for
provenance only. Goal-semantics authoring is not this seam's job.

Optional `structural_adapter=` / `semantic_adapter=` accept caller-supplied cheap tier-0 channels
for material no pixel-motion statistic can admit (screen content, documents). They run inside the
credential firewall described below.

### Zero admissions is a diagnostic, never an empty list

A run that admits nothing raises `HostedByokDetectionStarved` instead of returning `()`. An empty
list is indistinguishable from "nothing happened" to a caller who never sees the salience
distribution, and a silently starving or deaf gate is a failure this project has actually shipped.
The exception carries the numbers: `frames`, `salience_stat`, `thresholds`, `max_salience`,
`mean_salience`, and `misconfigured` — `True` when real salience was observed that never cleared the
cutoff (retune, or supply an admission signal), `False` when essentially no salience was observed
(the clip was still, or the frames are not the gate downsample). `.to_dict()` renders it as a hosted
error payload.

### Isolation properties specific to this route

- **No provider credential is readable.** For the duration of the scan, a context-scoped firewall
  replaces `os.environ` for the *scanning task only*: any read of a provider key
  (`GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_AISTUDIO_API_KEY`, `OPENROUTER_API_KEY`,
  `OPENROUTER_KEY`, and the rest of `_PROVIDER_ENV_KEYS`) raises. Caller-supplied salience adapters
  run inside it, so a tenant's own tier-0 channel is not an exfiltration path either. Concurrent
  unrelated coroutines read the real environment untouched. Deployment configuration is resolved
  *before* the firewall arms.
- **No provider client or backend is constructed.** Injected client/backend seams are never called.
- **Cancellation is genuine.** The gate is synchronous arithmetic over a 1,728-byte buffer, so it is
  never offloaded to a thread or executor — it yields to the loop between frames instead. A cancelled
  task stops between two frames owning nothing, and no child process exists to orphan. The
  invocation's `cancellation` event and its monotonic deadline are both checked per frame.
- **Containment is unchanged.** This route adds no process, so `_select_containment_capability` and
  the `judge()` supervisor path are untouched.

## Isolation and secret handling

The explicit path:

- reads no provider key, model, timeout, or prompt-mode value from the process environment;
- creates and discards a fresh vendor client and Percept adapter for every call;
- gives its request-bound adapter no provider resources and has no internal result/client/backend cache;
- never falls back to another provider, model, or capability;
- requires the caller to rotate `credential_lease_id` when the opaque credential rotates;
- excludes the credential from request repr, equality, and hashing, and makes the request unhashable;
- forwards neither vendor exception text nor exception chaining, since a client error may echo headers;
- accepts only an exact SDK `Verdict`, recursively rejects opaque mapping/subclass carriers, and rejects
  any approved public carrier containing the opaque credential;
- records a client as soon as construction acquires it, including partial backend construction;
- completes invocation-local client/backend closure after success, failure, timeout, or cancellation;
- authenticates the no-descendant OS boundary before sending the credential, then terminates and waits
  for the production worker before `judge` returns, leaving no provider, deferred callback, thread, or
  child process from that invocation alive; and
- never writes the request or credential to argv, environment variables, logs, tracebacks, or temporary
  storage; the bounded private IPC payload exists only for the isolated invocation.

Principal and lease ids are non-secret isolation/audit identities; the provider credential is not.
Do not put the request in an application artifact or log even though its SDK repr is redacted.

## Reviewed fake and transport seams

Tests can inject `backend_factories={"gemini": ...}` / `{"openrouter": ...}` without importing a
vendor SDK. To exercise the real Percept adapters with fake transports, inject
`openrouter_client_factory=` or `gemini_client_factory=`. A Gemini client callback returns
`HostedByokGeminiClient(client, transport_types)`, where `transport_types` supplies compatible
`Part` and `GenerateContentConfig` objects.

Every callback is invoked once per judgment and must return a fresh backend/client; object reuse is
detected by weak identity and rejected before dispatch. These are
dependency-injection seams for deterministic conformance tests and reviewed hosted transports, not a
hook for shared client pools.

## Failures

- `HostedByokContractError`: malformed request or incompatible contract version.
- `HostedByokUnsupportedError`: unsupported provider, exact model shape, evidence, or capability.
- `HostedByokDeadlineExceeded`: relative timeout or absolute deadline elapsed.
- `HostedByokCancelled`: invocation cancellation signal fired.
- `HostedByokInvocationError`: client/backend/provider failure with provider details withheld.
- `HostedByokDetectionStarved`: a gated detection run admitted nothing anywhere, for any goal. The
  request was well-formed — this is a *diagnostic*, not a contract error, and it carries the
  observed salience distribution so the caller can tell a still clip from a deaf gate.

This is a source contract. A package/wheel provenance claim must be made only after the normal release
pipeline builds and verifies an artifact.
