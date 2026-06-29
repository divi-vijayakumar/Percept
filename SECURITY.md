# Security Policy

## Reporting a vulnerability

Please report security issues **privately** to **security@velvee.ai** (or divi@velvee.ai). We aim to
acknowledge within 72 hours. Do not open a public issue for a vulnerability.

## Supported versions

Research preview (0.6.x). Security fixes target the latest released version.

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
- **Secrets** — never hardcoded; read from the environment / a gitignored `.env`.

## Model checkpoints (supply chain)

The optional `[pose]` extra downloads RTMO / MMPose checkpoints. As with any ML checkpoint, install only from
trusted sources — a malicious `.pth` is a code-execution risk at load time (the standard ML supply-chain
caveat, not specific to Percept).
