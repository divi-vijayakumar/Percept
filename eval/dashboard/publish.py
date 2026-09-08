#!/usr/bin/env python3
"""Render the evaluation-layer dashboard and publish it to Netlify.

NO LLM IS IN THIS LOOP. This is a pure scan-render-deploy script driven by cron. It reads whatever
is on disk, states plainly what is missing, and never infers, summarises or interprets. If a layer
has no result it says so; it does not guess.

    python3 eval/dashboard/publish.py              # render only, to eval/dashboard/site/
    python3 eval/dashboard/publish.py --deploy     # render + netlify deploy --prod

Design rules, in order of importance:

1. **A missing result is rendered as missing.** Never as zero, never as a dash that could be read as
   a measurement. The failure mode this dashboard exists to avoid is a tile that looks green because
   nothing ran.
2. **L4 is never given a number.** 180 of its 205 gate cases are unreviewed, so any figure computed
   over it would be a metric with no ground truth behind it. The tab says that instead.
3. **Every panel carries its own provenance** — the file it came from and that file's mtime — so a
   stale tile is visibly stale rather than silently old.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "eval" / "stage-validation" / "results"
SITE = REPO / "eval" / "dashboard" / "site"
STATE = REPO / "eval" / "dashboard" / "run-state.json"
# No site id lives here any more. The one that did was the SOAK site's, and this file deploying
# anywhere is what caused the damage — see main(). cron.sh owns the deploy target.

# layer id, title, subtitle, the fixture it runs over, result files to look for
LAYERS = [
    ("L0", "Capability", "Deterministic. Belongs in unit tests — never a quality claim.",
     "eval/synthetic/ — 7 corpora, 22 cases",
     ["sv0_ingest.json", "sv0_compile.json"]),
    ("L1", "Quality", "Statistical. 100 cases, 2.98 h of media.",
     "eval/stage-validation/corpus.v1.1.json",
     ["sv1_admission.json", "sv2_senses.json", "sv3_pool.json", "sv4_cognition.json"]),
    ("L1a", "Audio", "53 cases, 0.472 h. No key required.",
     "eval/stage-validation/audio_corpus.json",
     ["sv_audio.json"]),
    # long_run_2h.json sits under L2, not L3: it is a REPEAT of this same source, not a separate
    # measurement. See L2_REPEAT_NOTE.
    ("L2", "Temporal", "long_run_all.mp4, 81.6 min, 73 goals — run twice.",
     "eval/stage-validation/long_sources/long_run_all.mp4",
     ["long_run.json", "long_run_2h.json"]),
    ("L3", "Long-horizon", "Fixture not built.",
     "eval/stage-validation/long_sources/ — no 2 h source present",
     []),
    ("L4", "Trust", "", "eval/fixtures-golden/", []),
]

L3_CAVEAT = (
    "No 2 h fixture exists in this repo. The layer was declared here against "
    "'long_sources/inject_2h_v1.mp4', and that file is referenced by nothing else in the tree — no "
    "builder, no manifest, no corpus entry. It was never built. The runner that filled this lane "
    "passed no --manifest, so it silently re-ran L2's default source under a different output "
    "name; that result is shown under L2 as the repeat it actually is, and is NOT shown here. "
    "Building a genuine long-horizon source is real work (build_long_sources.py concatenates "
    "same-goal cases) and a fresh provider spend, so this tab stays empty until that is a decision "
    "someone has made. An empty tab is the honest rendering; L2's number relabelled would not be."
)

L2_REPEAT_NOTE = (
    "Both files are the SAME source and the same configuration — long_run_2h was an unintended "
    "second pass, not a distinct fixture. Read as a reproducibility check it is a real result: two "
    "independent live-provider runs agreed exactly on incidents (6 vs 6), frames yielded and "
    "emitted (24,484 vs 24,484) and drops (0 with receipt, 0 without), with provider calls "
    "differing 0.07% (5,722 vs 5,726). One number did NOT reproduce: rss_end_mb was 132.9 on the "
    "first pass and 253.3 on the second, from near-identical starts (223.0 / 223.2). Same source, "
    "same frame count, same work — unexplained, and it lands on the run whose stated purpose "
    "includes bounded retention."
)

L4_CAVEAT = (
    "180 of 205 gate cases are unreviewed, and all 48 scenes are uncleared third-party media. "
    "No number is shown here because any figure computed over this fixture would have no ground "
    "truth behind it. This tab stays empty until the labels are reviewed — an empty tab is the "
    "honest rendering, a percentage would not be."
)


PRIMITIVE_COLLAPSE_CAVEAT = (
    "The long runs arm every goal from plain condition text, and SV-0.7 measured that path at "
    "4/100: nothing in the compile chain reads the condition for meaning, so all four goal kinds "
    "produce a byte-identical boolean presence Watch (type='watch'). Verified through long_run's "
    "own plan builder — count, duration, transition and state all compile to the same primitive. "
    "Every goal here therefore executes as 'is it present', while the results still stamp each "
    "one with its manifest goal_kind. Entity tracks, episode state, ledger reconciliation, judge "
    "allocation, RSS, latency and cost are unaffected: none of them depend on which primitive "
    "compiled. The per-goal-kind accuracy view is the part that is measuring the wrong thing. "
    "funnel.dropped is separately unreliable — SV-4.8-D1 found _JudgeScheduler carries no funnel "
    "reference at all (worker.py:1417); read the primary ledger, not the mirror."
)


def _mtime(p: pathlib.Path) -> str:
    try:
        return _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "—"


def _load(name: str) -> tuple[dict | None, pathlib.Path]:
    p = RESULTS / name
    if not p.exists():
        return None, p
    try:
        return json.loads(p.read_text()), p
    except (json.JSONDecodeError, OSError):
        return None, p


def _counts(d: dict) -> tuple[int, int, int]:
    """(passed, failed, total) from a stage-validation result, or zeros if it has no verdicts."""
    tests = d.get("tests") or d.get("verdicts") or []
    if isinstance(tests, dict):
        tests = list(tests.values())
    p = f = 0
    for t in tests:
        if not isinstance(t, dict):
            continue
        v = str(t.get("verdict") or t.get("status") or t.get("result") or "").lower()
        if v in ("pass", "passed", "ok", "true"):
            p += 1
        elif v in ("fail", "failed", "error"):
            f += 1
    return p, f, len(tests)


def _run_state() -> dict:
    """Whatever the runner harness last wrote. Absent is normal before the first lane reports."""
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _panel(layer_id: str, files: list[str]) -> str:
    if layer_id == "L4":
        return f'<div class="caveat"><strong>Not a metric.</strong> {html.escape(L4_CAVEAT)}</div>'
    if layer_id == "L3":
        return f'<div class="caveat"><strong>Never ran.</strong> {html.escape(L3_CAVEAT)}</div>'
    if not files:
        return '<div class="missing">No result files declared for this layer.</div>'
    prefix = ""
    if layer_id == "L2":
        prefix = (
            f'<div class="caveat"><strong>Read the per-goal-kind breakdown as invalid.</strong> '
            f"{html.escape(PRIMITIVE_COLLAPSE_CAVEAT)}</div>"
            f'<div class="caveat"><strong>The two files are one measurement, run twice.</strong> '
            f"{html.escape(L2_REPEAT_NOTE)}</div>"
        )

    rows = []
    for name in files:
        d, p = _load(name)
        if d is None:
            rows.append(
                f'<tr class="r-missing"><td><code>{html.escape(name)}</code></td>'
                f'<td colspan="4"><span class="tag tag-missing">not run</span> '
                f'<span class="dim">no file at {html.escape(str(p.relative_to(REPO)))}</span></td></tr>'
            )
            continue
        passed, failed, total = _counts(d)
        stage = html.escape(str(d.get("stage") or d.get("suite") or "—"))
        cls = "tag-fail" if failed else ("tag-pass" if total else "tag-none")
        label = f"{failed} failed" if failed else (f"{passed}/{total} pass" if total else "no verdicts")
        rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td><td>{stage}</td>"
            f'<td><span class="tag {cls}">{label}</span></td>'
            f"<td class=\"num\">{total}</td><td class=\"dim\">{_mtime(p)}</td></tr>"
        )
    return (
        prefix
        + '<table><thead><tr><th>result</th><th>stage</th><th>verdict</th>'
        '<th class="num">cases</th><th>file mtime</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render() -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = _run_state()
    lanes = state.get("lanes", {})

    tabs, panels = [], []
    for i, (lid, title, sub, fixture, files) in enumerate(LAYERS):
        active = " active" if i == 0 else ""
        lane = lanes.get(lid, {})
        status = lane.get("status", "idle")
        # The runner marked L3 "done" because a process exited and wrote a file — but what it ran
        # was L2's source again. Never let the harness's lane bookkeeping assert a measurement the
        # fixture cannot support; L3 has no fixture, so it reports as such regardless.
        if lid == "L3":
            status = "no fixture"
        dot = {"running": "d-run", "done": "d-done", "failed": "d-fail"}.get(status, "d-idle")
        tabs.append(
            f'<button class="tab{active}" onclick="show(\'{lid}\')" id="t-{lid}">'
            f'<span class="dot {dot}"></span>{lid}<span class="tsub">{html.escape(title)}</span></button>'
        )
        panels.append(
            f'<section class="panel{active}" id="p-{lid}">'
            f"<h2>{lid} · {html.escape(title)}</h2>"
            f'<p class="sub">{html.escape(sub)}</p>'
            f'<p class="fixture"><span class="dim">fixture</span> <code>{html.escape(fixture)}</code>'
            f' &nbsp; <span class="dim">lane</span> <span class="tag tag-{status}">{html.escape(status)}</span></p>'
            f"{_panel(lid, files)}</section>"
        )

    note = html.escape(state.get("note", "")) if state.get("note") else ""
    return f"""<!doctype html>
<meta charset="utf-8"><title>Percept · evaluation layers</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#0f1115;--fg:#e6e8ec;--dim:#8b93a1;--line:#222733;--card:#151922;
--pass:#2f9e6b;--fail:#c8483f;--run:#c98a2b;--idle:#4a5163}}
@media(prefers-color-scheme:light){{:root{{--bg:#fbfbfd;--fg:#14171c;--dim:#666e7d;
--line:#e3e6ec;--card:#fff}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,"SF Pro Text",Segoe UI,sans-serif}}
header{{padding:22px 26px 0;max-width:1040px;margin:0 auto}}
h1{{font-size:19px;margin:0 0 4px;font-weight:600;letter-spacing:-.01em}}
.meta{{color:var(--dim);font-size:12.5px;margin:0 0 18px}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line);
padding:0 26px;max-width:1040px;margin:0 auto}}
.tab{{background:none;border:0;border-bottom:2px solid transparent;color:var(--dim);
padding:9px 12px;cursor:pointer;font:inherit;font-weight:500;display:flex;align-items:center;gap:7px}}
.tab.active{{color:var(--fg);border-bottom-color:var(--fg)}}
.tsub{{color:var(--dim);font-weight:400;font-size:12px}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block;flex:none}}
.d-run{{background:var(--run)}}.d-done{{background:var(--pass)}}
.d-fail{{background:var(--fail)}}.d-idle{{background:var(--idle)}}
main{{max-width:1040px;margin:0 auto;padding:20px 26px 60px}}
.panel{{display:none}} .panel.active{{display:block}}
h2{{font-size:16px;margin:6px 0 3px;font-weight:600}}
.sub{{color:var(--dim);margin:0 0 12px;font-size:13px}}
.fixture{{margin:0 0 16px;font-size:12.5px}}
table{{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:8px;overflow:hidden}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);font-size:13px}}
th{{color:var(--dim);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}}
tr:last-child td{{border-bottom:0}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--fg)}}
.dim{{color:var(--dim)}}
.tag{{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;font-weight:500}}
.tag-pass{{background:color-mix(in srgb,var(--pass) 18%,transparent);color:var(--pass)}}
.tag-fail{{background:color-mix(in srgb,var(--fail) 18%,transparent);color:var(--fail)}}
.tag-missing,.tag-none,.tag-idle{{background:color-mix(in srgb,var(--idle) 22%,transparent);color:var(--dim)}}
.tag-running{{background:color-mix(in srgb,var(--run) 18%,transparent);color:var(--run)}}
.tag-done{{background:color-mix(in srgb,var(--pass) 18%,transparent);color:var(--pass)}}
.r-missing td{{opacity:.72}}
.caveat{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--run);
padding:14px 16px;border-radius:8px;font-size:13.5px}}
.missing{{color:var(--dim);font-size:13px;padding:14px 0}}
.note{{max-width:1040px;margin:0 auto;padding:0 26px 8px;color:var(--dim);font-size:12.5px}}
footer{{max-width:1040px;margin:0 auto;padding:0 26px 40px;color:var(--dim);font-size:12px;
border-top:1px solid var(--line);padding-top:14px}}
</style>
<header>
  <h1>Percept · evaluation layers</h1>
  <p class="meta">Rendered {now} · refreshes every 15 min · no model is in this loop —
  cron scans the results directory and publishes what is there.</p>
</header>
{f'<p class="note">{note}</p>' if note else ''}
<div class="tabs">{''.join(tabs)}</div>
<main>{''.join(panels)}</main>
<footer>
  A layer with no result file reads <em>not run</em>, never zero. Capability (L0) is deterministic
  and can never answer a quality question; quality (L1) is statistical and needs its stated coverage
  beside it. L4 carries no number by design.
</footer>
<script>
function show(id){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('p-'+id).classList.add('active');
  document.getElementById('t-'+id).classList.add('active');
  try{{localStorage.setItem('percept-eval-tab',id)}}catch(e){{}}
}}
try{{var s=localStorage.getItem('percept-eval-tab');if(s&&document.getElementById('p-'+s))show(s)}}catch(e){{}}
setTimeout(()=>location.reload(), 15*60*1000);
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="netlify deploy --prod after rendering")
    a = ap.parse_args()

    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    out.write_text(render())
    print(f"rendered {out.relative_to(REPO)} ({out.stat().st_size} bytes)")

    if not a.deploy:
        return 0
    # Deliberately dead. This used to run `netlify deploy --prod --dir <SITE>` against SITE_ID —
    # a BARE directory against the SOAK site — and it did two things wrong at once: it published
    # the eval dashboard over someone else's site, and because a bare dir carries no
    # netlify/edge-functions/, the deploy silently removed that site's password. It did exactly
    # that, twice, on 2026-08-01.
    #
    # Deployment now belongs to cron.sh alone, which ships the PROJECT dir (auth function included)
    # to the eval site and refuses to run if auth.js is missing. Rendering stays here; publishing
    # does not.
    sys.stderr.write(
        "--deploy is disabled. Use eval/dashboard/cron.sh, which deploys the project dir "
        "(with the edge auth function) to the eval site. Deploying a bare dir strips the "
        "target site's password.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
