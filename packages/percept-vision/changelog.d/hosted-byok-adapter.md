### Added — versioned invocation-scoped Hosted BYOK

`percept.hosted_byok` adds the public `HOSTED_BYOK_CONTRACT_VERSION = "1.0.0"`,
`HostedByokRequest`, and `HostedByokAdapterFactory.judge(...)` boundary for hosted services invoking
Gemini or OpenRouter with a caller-owned credential. Provider, exact model, capability, principal,
credential lease, timeout/deadline, and cancellation are explicit and immutable. The path performs no
environment fallback, creates and closes a fresh backend/client for every call, has no cache or
shared provider-resource surface, fails closed on unsupported routing/capabilities, and withholds
vendor exception details so credentials cannot enter repr/logs/errors/artifacts. `factory.bind(request)`
returns a versioned request-bound `Vision` accepted by `Percept.create(vision=...)`; it creates fresh
provider resources on every judgment. Exact capability admission makes plain `vision.judge` image-only
and reserves image+audio for Gemini `vision.judge.fused`; a fused bound adapter composes through the
compiled Watch runtime's sealed trusted-audio binding. Deadline/cancellation initiates cancellation but
retains invocation ownership until provider work and teardown actually finish, leaving no detached
tasks or threads; post-completion monotonic expiry also rejects event-loop-blocking late results.
Partially constructed clients are closed, media/context routes cannot downgrade the declared modality,
and credentials found recursively in a returned Verdict fail closed. Fake backend and fake client
factories provide deterministic conformance seams without live providers and reject reused objects.
Existing registry/direct backend APIs are unchanged; this source contract makes no wheel provenance
claim.
