# Security Policy

## Reporting a vulnerability

Please report security issues **privately** to **security@velvee.ai**. We aim to acknowledge within
72 hours. Do not open a public issue for a vulnerability.

## Supported versions

Percept is in Developer Preview (0.9.0). Fixes target `main`.

## Principles

- No `eval`, `exec`, `pickle` or unsafe deserialisation of model output. Model replies are parsed
  against a fixed schema; anything else is treated as a failure, never as an answer.
- Video stays where you run Percept. Frames are sent only to the judges you configure.
- API keys are read from the environment and are never written to traces or stored state.
