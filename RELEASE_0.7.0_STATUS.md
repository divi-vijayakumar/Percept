# Release 0.7.0 — hackathon runtime · overnight status

Goal (6h scope, founder-directed): a **fully working, publishable runtime** as **0.7.0** for the hackathon.
Explicitly OUT: the PRD Rev-B, wave-2 (M2/P2/L1), health/evidence/delivery, M3/J1 re-scope.

## ✅ Done
- [x] Version 0.6.0 → **0.7.0** (> published 0.6.0, so `pip install` resolves to this)
- [x] Wheel + sdist build cleanly
- [x] **Clean-venv install works** — `run()` fires end-to-end from the installed wheel (deps: cv2 5.0, numpy 2.5, openai 2.46)
- [x] **`percept-watch` CLI** shipped (console entry point) — `pip install` → `percept-watch --goal "…"` verified in a clean venv
- [x] Full test suite green (1398 passed)
- [x] `--egomotion` preset (validated on real first-person footage: 98%→30% escalation)
- [x] Claude skill (`percept-agent`) updated to 0.7.0 (CLI, qwen, block_max/refractory/ego) — both copies

## �doing (overnight)
- [ ] Docs cleanup: top-level README quickstart, integration doc coherence, examples/README
- [ ] CHANGELOG / release notes for 0.7.0 + honest stability statement (research-preview)
- [ ] Production-readiness pass: leftover worktree cleanup, error-path sanity, re-run full suite after edits
- [ ] Prep the publish command (twine) + this morning summary

## In the morning (YOUR action)
- Publish: `cd packages/percept-vision && python3 -m twine upload dist/*` (needs your PyPI token). Command left ready below.
- Decide whether to push the branch (currently 56 local commits ahead of origin/main, unpushed per your rule).

## Honest scope of 0.7.0
Working runtime (`watch`/`run` → notifications) + winning defaults (block_max, refractory, ego-motion via
env/CLI) + the CLI + docs. NOT: wired WatchProfiles, window/episode in the live judge, health/evidence.
A research-preview runtime for the hackathon.

## 🔒 Production-readiness audit (done)
- [x] **Secrets** — `.env` gitignored + never committed (0 commits); no real keys in tracked files (only `sk-or-...` placeholders in docs). Real-key regex: NONE.
- [x] **Sdist leakage** — 276 files, clean: no `.env`, no datasets (Charades/Ego4D NOT in package), no `.mp4`, no keys. Top-level: LICENSE, README, percept, tests.
- [x] **License** — Apache-2.0, present (root + package) + declared in pyproject.
- [x] **Vuln scan (pip-audit)** — opencv-headless / numpy / openai: NO vulns. (6 findings are in `pip` itself = user tooling, not the package.)
- [x] **Egress/SSRF** — webhook sinks block private/loopback + require https by DEFAULT (`UrlPolicy`); `.dev()` opt-in for local http. Frames go only to the chosen judge provider.
- [x] **Safety framing** — `consent.py`; README "envelope" (assistant-class, not surveillance/safety); skill warns off surveillance demos.
- [x] **SECURITY.md** — real private-disclosure policy (security@velvee.ai, 72h); bumped 0.6.x → 0.7.x.
- [x] **CONTRIBUTING.md** present.

### Honest remaining gaps (minor for a research-preview)
- No CODE_OF_CONDUCT.md (nice-to-have).
- No deep manual code-security review (RTSP/webcam URL handling; goal text → VLM prompt = low prompt-injection risk for a watch tool). SSRF on webhooks IS covered.
- Docs are comprehensive but a final coherence pass would help.
