---
id: PCPT-007
title: Polygon watch zones + parity-suite extension
status: backlog
priority: P1
owner: unassigned
wave: C
pr: PR-7a
human_test: pending
---

## Why
Zones are axis-aligned Box rectangles on the motion branch only; IRA authors polygon zones. Also the parity suite is narrow - this widens it.

## Scope
polygon on WatchCondition beside region_hint; point-in-polygon mask in _motion_score_region; port mask to motion.js; extend parity/compare.py FRAME_PAIRS with polygon cases.

## Acceptance criteria
- [ ] tier0 unit tests: motion inside vs outside polygon, degenerate/concave polygons, Box fallback
- [ ] JS mask byte-for-byte identical to Python in make parity
- [ ] Zero delta on existing baselines

## Human test
Draw a non-rectangular zone on a sample scene and confirm motion outside it never escalates.
