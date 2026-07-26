# Hosted BYOK invocation contract

`percept.hosted_byok` is the public boundary for a hosted service that receives an end user's provider
credential for one model invocation. It is separate from `Percept.create(...)`,
`resolve_vision(...)`, and the backend registry because those existing APIs intentionally support
process-environment configuration and long-lived adapters.

Contract version: `percept.HOSTED_BYOK_CONTRACT_VERSION` (`"1.0.0"`).

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
The process-isolation contract version is `HOSTED_BYOK_PROCESS_ISOLATION_VERSION` (`"2.0.0"`).
The v2 supervisor currently requires a validated root-owned Darwin `sandbox-exec` or Linux `prlimit`
boundary plus a successful no-fork attestation; unsupported or ineffective hosts fail closed before
the credential payload is sent.
Request and Verdict IPC are bounded to 32 MiB;
the wire is canonical tagged JSON over `send_bytes`/`recv_bytes`, never pickle. Exact recursive carrier
validation rejects hostile subclasses, duplicate keys, non-finite values, unsupported DTOs, malformed
or oversized payloads before construction/unmarshal. Every non-credential input is also rejected if it
contains the credential. The credential crosses only the private inherited pipe: it is never placed in
the worker's argv or environment. The allowlisted child environment is installed by the parent before
the interpreter, `sitecustomize`, or SDK starts, excluding provider configuration, proxies, custom CA
paths, TLS key logging, dynamic-loader controls, and other inherited transport routing. Hosted Gemini
and OpenRouter HTTP clients also set environment trust off explicitly.

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

This is a source contract. A package/wheel provenance claim must be made only after the normal release
pipeline builds and verifies an artifact.
