# Overnight OSS-hardening — morning report (2026-07-12)

Branch: `feat/overnight-oss-hardening` (off `main`). **Nothing pushed, no PRs, nothing published** — all commits local, per the guardrails.

## Shipped + verified (7 cards, all gate-checked)

| Card | What | Gate result |
|---|---|---|
| **R-2** (018) | Removed the DetectorBank **double-push bug** (every boundary + the published inference count was doubled) | py 917 green |
| **R-3** (019) | `mute()`/`unmute()` aliases on RuntimeSession | test green |
| **R-0** (017) | **`percept.run(source=, goals=[...])`** wrapper — the one-line quickstart is now real; `run(core,…)` unchanged | py 922 green |
| **Docs P0** (016) | Fixed the corrupted/stale `.agents` skill (`Codex` bad-replace gone), both skills lead with `percept.run`, added a live `run()` template | templates compile, 0 `Codex` |
| **Docs P1/P2** (022) | Full `run()` param reference, motion-gate-vs-cadence + salience-mode callout, `on_trace` stages, README live snippet | docs-only |
| **L-1** (020) | **LabelRecord rail** (keystone): schema + LabelStore + disposition log; media-free-head/local-tail egress guarantee; folded into contract 1.0.0 additively | **py 953 (+31), parity 21/21, js 135**, egress-cannot-leak verified |
| **Demo** (023) | v2v harness headless-verified (full import chain resolves); human run-book written | py_compile + import OK |

**Plan item 1 (docs DX gap) is fully closed. Plan item 2 (v2v demo) is headless-verified — awaiting your live browser test.**

## Blocked — needs you (documented, not force-built)

- **Paid product** (024) — G-0 local-VLM, dashboard, API, playground, supervise. **Cannot live in this Apache-2.0 public repo** (licensing: world-readable + permissive = competitors can lift it). Needs the private `percept-cloud` repo (open-core). This resolves your licensing question: OSS here, paid in a private repo consuming it, meeting only at the LabelRecord seam.
- **Live browser demo test** — run-book in PCPT-023; needs a human at a webcam + a `GEMINI_API_KEY`.

## Pending your decision

- **R-1** (021) sustained-duration `hold_s` — the last R-track item. It's flagged **risky**: it adds another field to the (frozen) `WatchCondition` contract AND must preserve the gate's byte-for-byte goldens. I held it for your go-ahead now that you're back, rather than making a second contract/gate change unattended.

## Human-action checklist

1. **Ratify L-1 design calls** (PCPT-020 feedback): `record_label()` method name (vs spec's `label()`), `verdict_ref` hash scheme, `deployment_id` source, and the **LabelRecord schema freeze** (three producers depend on it).
2. **Decide R-1** — proceed or hold.
3. **Create the private `percept-cloud` repo** to start the paid tracks.
4. **Run the live v2v demo** (PCPT-023 run-book).
5. **Review + push/PR** `feat/overnight-oss-hardening` (9 code/docs commits). Consider renaming it off the stale `feat/pcpt-001-contract-1.3.0` lineage.

## Note
The contract is now **1.0.0** (renumbered from the unreleased 1.3.0 during the salience work); the specs still say 1.3.0→1.4.0 — every "bump" was folded additively into 1.0.0 instead, since nothing shipped under the old numbers.
