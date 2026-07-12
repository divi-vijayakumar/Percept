---
id: PCPT-024
title: Paid product (G-0 local-VLM, dashboard, API, playground, supervise) — SEPARATE PRIVATE REPO
status: blocked
priority: P0
owner: human
wave: paid
pr: ""
human_test: n/a
updated: 2026-07-11
---

## Why blocked (licensing + infra)
This repo is Apache-2.0 (public). Proprietary paid code CANNOT live here (world-readable + permissively licensed = competitors can lift it). Repo strategy (2026-07-03): paid = private repo `percept-cloud` consuming the OSS as a dependency; meet at the LabelRecord seam only.

## What a human must do
1. Create the private `percept-cloud` repo (not a fork).
2. G-0 local-VLM (Moondream/SmolVLM via ONNX/llama.cpp) — needs model-weight download + runtime provisioning (#1 paid critical path; without it the falling-cost claim dies).
3. P-1..4 supervise/API/dashboard/playground — greenfield UIs/services.
4. Freeze L-1 (PCPT-020) + C-0 calibration schema first (a meeting, not a milestone).

## Note
NOT attempted autonomously overnight. OSS prerequisites (L-1 schema freeze) are being prepared in PCPT-020 so the seam is ready when the private repo starts.
