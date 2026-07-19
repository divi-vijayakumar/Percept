# Security Policy

## Reporting a vulnerability

Please report security issues **privately** to **security@velvee.ai** (or divi@velvee.ai). We aim to
acknowledge within 72 hours. Do not open a public issue for a vulnerability.

## Supported versions

Research preview (0.7.x). Security fixes target the latest released version.

## Security posture

Percept is built defense-in-depth for an LLM perception runtime:

- **No unsafe execution** — no `eval` / `exec` / `pickle` / unsafe `yaml.load`. LLM-authored `fire_when`
  micro-expressions are parsed by a hand-written AST walker with a strict name whitelist, never `eval`'d.
- **Untrusted-input firewalls** — every conductor/LLM-authored sensor plan, sense, and rhythm parameter is
  clamped and enum-whitelisted (`coerce_authored_*`) before it reaches the runtime.
- **Defensive verdict parsing** — model output is type-coerced into the firing `Verdict`; a parse miss
  degrades to a low-confidence ASK, never a confident-wrong fire and never code execution.
- **No network exposure** — the runtime worker is a driver loop, not a server; webhook sinks are POST-only
  and HMAC-signed.
- **Outbound URL policy (SSRF)** — `WebhookSink` defaults to https-only and blocks non-globally-routable
  targets (loopback, private, link-local incl. the metadata IP, CGNAT, reserved), so a user-supplied
  webhook URL can't be aimed at an internal service. Dead-letter records are redacted (URL reduced to
  `scheme://host[:port]`, body stored as a SHA-256, no headers/signature).
- **Secrets** — never hardcoded; read from the environment / a gitignored `.env`.

## Known limitations

- **Webhook SSRF — DNS-rebinding residual.** The private-IP check re-resolves the host just before sending,
  but `urllib` resolves again at connect time, leaving a narrow time-of-check/time-of-use window a
  short-TTL rebinding attacker could exploit. For an untrusted-input deployment, pin hosts
  (`UrlPolicy(allowed_hosts=…)`) **and** restrict egress at the network layer (no route to
  `169.254.169.254` / internal ranges) — the real backstop for SSRF. Pinning the resolved IP through the
  TLS connection to close it in-library is planned. Capture *source* URIs (RTSP / `file://`) are operator
  config and are not URL-policy-checked.

## Model checkpoints (supply chain)

The optional `[pose]` extra downloads RTMO / MMPose checkpoints. As with any ML checkpoint, install only from
trusted sources — a malicious `.pth` is a code-execution risk at load time (the standard ML supply-chain
caveat, not specific to Percept).
