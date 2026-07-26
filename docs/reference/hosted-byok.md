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
OpenRouter accepts exact `vendor/model` ids and fails closed if audio is supplied.

`timeout_s` is relative to entry into `judge`; `deadline_unix_s` is an absolute UNIX timestamp. If
both are supplied, the earlier limit wins. Omitting both uses the contract's 120-second safety ceiling.
`asyncio.Event` implements the public cancellation protocol. Cancelling the calling task also cancels
the provider task. Provider construction is synchronous and included in the monotonic deadline
accounting: an over-budget construction is rejected when it returns, before any provider task is
scheduled. Cancellation is checked before and after construction and again immediately before
dispatch, so a cancellation observed during construction spends zero provider calls.

## Isolation and secret handling

The explicit path:

- reads no provider key, model, timeout, or prompt-mode value from the process environment;
- creates and discards a fresh vendor client and Percept adapter for every call;
- gives its request-bound adapter no provider resources and has no internal result/client/backend cache;
- never falls back to another provider, model, or capability;
- requires the caller to rotate `credential_lease_id` when the opaque credential rotates;
- excludes the credential from request repr, equality, and hashing, and makes the request unhashable;
- forwards neither vendor exception text nor exception chaining, since a client error may echo headers;
- best-effort closes invocation-local clients after success, failure, timeout, or cancellation; and
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
