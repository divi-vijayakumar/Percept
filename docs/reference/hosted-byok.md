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

from percept import HostedByokAdapterFactory, HostedByokRequest

hosted_byok = HostedByokAdapterFactory()

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
`asyncio.Event` implements the public cancellation protocol. Cancelling the calling task also cancels
the provider task. The deadline bounds result acceptance and cancellation initiation, not cleanup
latency. The SDK installs a ref-counted, context-scoped asyncio task recorder for the complete call:
provider/client/close descendant tasks are cancelled and joined before any accepted, cancelled, or
timeout outcome is released. A close getter, non-callable close, or close exception produces the fixed
fail-closed provider error only after this task scope has drained. Concurrent Hosted BYOK calls use
separate scopes; the loop's original task factory is restored after the last scope exits. If the host
already installed a task factory, Hosted BYOK refuses before provider construction rather than replacing
it and claiming containment.

This is **asyncio-task containment**, not a general thread/process kill switch. The reviewed production
Gemini/OpenRouter async transports must not start unmanaged threads, executors, subprocesses, or raw
background work for a Hosted BYOK invocation. The injected client/backend callbacks are deterministic
test/review seams, not a supported Cloud isolation mechanism; a Cloud deployment using a transport that
can create such work must run the whole credential-bearing call in a separately supervised process and
terminate/join that process before releasing its lease. The SDK deliberately makes no promise that it
can stop arbitrary foreign threads in-process. Callers that protect a credential lease or rotation with
a lifecycle guard must hold (and, where applicable, renew) that guard for the complete `await`.

Cloud can enforce that rule mechanically. `factory.transport_safety(request)` returns the immutable
`HostedByokTransportSafety`: built-in routes are classified `sdk_async_allowlist`; any injected backend
or client route is classified `injected_foreign_work` and sets `requires_process_supervisor=True`.
Construct a Cloud factory with `require_process_supervisor_for_foreign_work=True` and a
`HostedByokProcessSupervisor` whose `is_available()` reports the external supervisor's live status.
The SDK then rejects an unsafe route before any provider/client construction when that supervisor is
absent or unavailable. This check is deliberately a Cloud policy gate; it does not pretend an arbitrary
in-process callback can be force-killed.

Provider construction is synchronous and included in the monotonic deadline accounting: an
over-budget construction is rejected when it returns, before any provider task is scheduled.
Cancellation is checked before and after construction and again immediately before dispatch, so a
cancellation observed during construction spends zero provider calls. The monotonic deadline is also
checked after provider completion, so a coroutine that blocked the event loop cannot have a late
result accepted. A killable worker/process boundary can provide a bounded hard-stop contract, but this
in-process SDK boundary does not claim one.

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
- leaves no tracked provider, cancellation-watcher, close task, or asyncio descendant worker running
  after `judge` returns (see the process-isolation requirement above for foreign work); and
- never serializes or persists the request or credential.

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
