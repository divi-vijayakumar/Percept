#!/usr/bin/env python3
"""Generate ``architecture.html`` + ``docs/architecture.svg`` from code + ``docs/architecture.meta.json``.

The architecture diagram is a BUILD ARTIFACT, never hand-edited. This generator makes drift
impossible: every hop's ``file:line`` pointer is validated against the source by AST, and every
adaptor's ● integrated / ○ seam marker is computed from the code (registry membership + module
resolvability) — never from the human-authored meta. A stale pointer fails the build.

Determinism (for the CI ``git diff --exit-code`` gate): output is byte-stable for a given HEAD —
authored-order iteration, integer SVG coords, version from pyproject, and a HEAD-derived sha/date
(never wall-clock). CI regenerates at the same HEAD it diffs, so the bytes match.

Usage:
    python tools/build_architecture.py                 # write html + svg
    python tools/build_architecture.py --check         # verify mode: exit 1 if they'd change
    python tools/build_architecture.py --date 2026-06-30
Exit codes: 0 ok · 1 --check found drift · 2 stale code-pointer (no output written).
"""
from __future__ import annotations

import argparse
import ast
import functools
import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "packages" / "percept-vision"
REGISTRY_FILE = PKG_ROOT / "percept" / "registry.py"
PYPROJECT = PKG_ROOT / "pyproject.toml"
DEFAULT_META = REPO_ROOT / "docs" / "architecture.meta.json"
DEFAULT_HTML = REPO_ROOT / "docs" / "architecture.html"
DEFAULT_SVG = REPO_ROOT / "docs" / "architecture.svg"

DRIFT_TOL = 10  # ±lines OK; >this is a WARNING (never fatal)
TIER_ORDER = ("vlm", "cog", "meas", "feed")
TIER_GLYPH = {"vlm": "◆", "cog": "●", "meas": "●", "feed": "●"}
RESERVED_NODES = {"n-out"}  # the cognition tier's FireEvent output sink

# meta adaptor name -> the registry (seam,name) names it owns (for the integration predicate)
BACKEND_REGISTRY_NAMES = {
    "anthropic": {"claude"},
    "gemini": {"gemini", "gemini-live"},
    "deepgram": {"deepgram"},
}
# measure/feed-tier adaptors that are NOT registry-backed: ● iff module is non-null and resolves
NON_REGISTRY_PROVIDERS = {"rtmo", "blazepose", "whisper", "pipecat"}

# tier colors (literal hex; GitHub-rendered SVG ignores CSS vars) — mirror the HTML :root tokens
COLORS = {
    "bg": "#0d1117", "panel": "#161b22", "line": "#30363d", "ink": "#e6edf3", "mut": "#8b949e",
    "feed": "#e06c9f", "meas": "#58a6ff", "cog": "#3fb950", "vlm": "#bc8cff", "seam": "#8b949e",
}


@dataclass(frozen=True)
class Stale:
    kind: str
    ref: str
    file: str
    detail: str


STALE: list[Stale] = []
WARNINGS: list[str] = []


# ----------------------------------------------------------------------------- symbol engine
@dataclass(frozen=True)
class FileSymbols:
    funcs: frozenset
    methods: frozenset   # "Class.method"
    classes: frozenset
    assigns: frozenset
    str_literals: frozenset
    lineno: dict


@functools.lru_cache(maxsize=None)
def parse_file_symbols(path: Path) -> FileSymbols:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    funcs, methods, classes, assigns, lits = set(), set(), set(), set(), set()
    lineno: dict = {}

    def walk(node, cls=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if cls:
                    methods.add(f"{cls}.{child.name}")
                    lineno.setdefault(f"{cls}.{child.name}", child.lineno)
                funcs.add(child.name)
                lineno.setdefault(child.name, child.lineno)
                walk(child, cls)
            elif isinstance(child, ast.ClassDef):
                classes.add(child.name)
                lineno.setdefault(child.name, child.lineno)
                walk(child, child.name)
            else:
                if isinstance(child, ast.Assign):
                    for t in child.targets:
                        if isinstance(t, ast.Name):
                            assigns.add(t.id)
                            lineno.setdefault(t.id, child.lineno)
                walk(child, cls)

    walk(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            lits.add(n.value)
    return FileSymbols(frozenset(funcs), frozenset(methods), frozenset(classes),
                       frozenset(assigns), frozenset(lits), lineno)


def resolve_hop_path(raw: str):
    """('a/b.py:21') -> (Path, line|None). Tries package-relative then repo-relative."""
    path_part, _, line_part = raw.partition(":")
    line = int(line_part) if line_part.isdigit() else None
    for cand in (PKG_ROOT / path_part, PKG_ROOT / "percept" / path_part, REPO_ROOT / path_part):
        if cand.is_file():
            return cand, line
    return None, line


_SEP = re.compile(r"\s*(?:→|\+|·|,)\s*|\s+builds\s+|\s+stream\b")


def normalize_symbol(fn: str):
    """A human call-expression -> (kind, symbol). kind in {name, method, none}."""
    first = _SEP.split(fn.strip())[0].strip()
    first = re.sub(r"\([^()]*\)", "", first)   # drop one level of call args
    first = re.sub(r"\([^()]*\)", "", first)   # ...and a nested level
    parts = [p for p in first.split(".") if p.isidentifier()]
    if not parts:
        return ("none", first)
    if len(parts) == 1:
        return ("name", parts[0])
    # explicit Class.method (capitalized receiver) -> qualified; instance receiver -> bare tail
    receiver = parts[-2]
    if receiver[:1].isupper():
        return ("method", f"{receiver}.{parts[-1]}")
    return ("name", parts[-1])


def registration_literals(fn: str):
    return re.findall(r"\b(?:make_detector|register|make)\(\s*[\"']([^\"']+)[\"']", fn)


def symbol_present(fs: FileSymbols, kind: str, sym: str) -> bool:
    if kind == "none":
        return True
    if kind == "method":
        if sym in fs.methods:
            return True
        tail = sym.split(".")[-1]
        return tail in fs.funcs or any(m.endswith("." + tail) for m in fs.methods)
    # name
    if sym in fs.funcs or sym in fs.classes or sym in fs.assigns:
        return True
    return any(m.endswith("." + sym) for m in fs.methods)


def symbol_lineno(fs: FileSymbols, sym: str):
    if sym in fs.lineno:
        return fs.lineno[sym]
    tail = sym.split(".")[-1]
    for m in fs.methods:
        if m.endswith("." + tail):
            return fs.lineno.get(m)
    return None


def validate_hop(hop: dict, scenario_id: str) -> None:
    raw = hop.get("file", "")
    fn = hop.get("fn", "")
    if ".py" not in raw:
        return  # narration hop (e.g. "percept-edge"): not a code pointer
    path, line = resolve_hop_path(raw)
    if path is None:
        STALE.append(Stale("missing-file", fn, raw, "file not found under package root"))
        return
    fs = parse_file_symbols(path)
    for lit in registration_literals(fn):
        if lit not in fs.str_literals:
            STALE.append(Stale("missing-literal", lit, str(path),
                               f'no register("{lit}") / name="{lit}" in file'))
    kind, sym = normalize_symbol(fn)
    if not symbol_present(fs, kind, sym):
        STALE.append(Stale("missing-symbol", sym, str(path),
                           f"symbol {sym!r} not defined (stale hop fn={fn!r}, scenario {scenario_id!r})"))
        return
    if line is not None:
        actual = symbol_lineno(fs, sym)
        if actual is not None and abs(actual - line) > DRIFT_TOL:
            WARNINGS.append(f"DRIFT {sym} in {path.relative_to(REPO_ROOT)}: "
                            f"cited :{line}, found :{actual} (>±{DRIFT_TOL}) — update meta")


# ------------------------------------------------------------------- integration (vendor-free)
def _ensure_path():
    p = str(PKG_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


@functools.lru_cache(maxsize=None)
def module_resolves(dotted: str) -> bool:
    import importlib.util
    _ensure_path()
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


@functools.lru_cache(maxsize=1)
def registry_names() -> frozenset:
    """Registered (seam,name) names, read by AST so no factory/vendor import ever runs."""
    tree = ast.parse(REGISTRY_FILE.read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "register"
                and len(n.args) >= 2
                and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in n.args[:2])):
            names.add(n.args[1].value)
    return frozenset(names)


def adaptor_integrated(name: str, adaptor: dict) -> bool:
    module = adaptor.get("module")
    if name in NON_REGISTRY_PROVIDERS:
        return bool(module) and module_resolves(module)
    if not module or not module_resolves(module):
        return False
    want = BACKEND_REGISTRY_NAMES.get(name, {name})
    return bool(want & registry_names())


# ----------------------------------------------------------------------------- provenance
def get_version() -> str:
    """[project].version from pyproject — install-independent so dev and CI agree byte-for-byte."""
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib
        return tomllib.loads(text)["project"]["version"]
    except Exception:
        m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
        return m.group(1) if m else "0.0.0"


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def get_sha() -> str:
    try:
        return _git("rev-parse", "--short", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def get_date(arg_date: str | None) -> str:
    if arg_date:
        return arg_date
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).date().isoformat()
    try:
        return _git("log", "-1", "--format=%cI")[:10]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass(frozen=True)
class Provenance:
    """Only the VERSION is embedded in the artifact — it is deterministic from pyproject and is the
    real release identifier. The git sha/commit-date are intentionally NOT embedded: a committed file
    cannot contain its own commit's sha, so embedding it would make the ``git diff --exit-code`` drift
    gate fail by construction (the sha changes the moment you commit the file). The operator still sees
    sha/date on stdout for provenance; the on-disk bytes stay HEAD-independent so the gate is exact."""
    version: str

    @property
    def stamp(self) -> str:
        return f"percept-vision {self.version}"


# ----------------------------------------------------------------------------- assembly
def build_model(meta: dict):
    """Validate everything and return (tiers, scenarios, glossary, adaptor_status, vlm_nodes)."""
    adaptors = meta.get("adaptors", {})
    status = {name: adaptor_integrated(name, ad) for name, ad in adaptors.items()}
    # If meta ASSERTS integration ("integrated": true/false), the code is the authority: a mismatch
    # is a build failure. (Lets meta document intent while making code-vs-claim drift impossible.)
    for name, ad in adaptors.items():
        if "integrated" in ad and bool(ad["integrated"]) != status[name]:
            STALE.append(Stale("integration-mismatch", name, "docs/architecture.meta.json",
                               f'meta claims integrated={ad["integrated"]!r} but code says '
                               f'{"● integrated" if status[name] else "○ seam"} — let code decide (remove the override)'))

    node_ids = set(RESERVED_NODES)
    tiers = []
    for tier in sorted(meta["tiers"], key=lambda t: TIER_ORDER.index(t["id"])):
        for node in tier["nodes"]:
            if node["id"] in node_ids:
                STALE.append(Stale("dup-node", node["id"], tier["id"], "duplicate node id"))
            node_ids.add(node["id"])
        tiers.append(tier)

    vlm_nodes = set()
    scenarios = []
    for s in meta["scenarios"]:
        hops = []
        for h in s["hops"]:
            if h["node"] not in node_ids:
                STALE.append(Stale("bad-node", h["node"], s["id"], "hop targets unknown tier node"))
            if h.get("vlm"):
                vlm_nodes.add(h["node"])
            validate_hop(h, s["id"])
            hop = {"node": h["node"], "fn": h["fn"], "file": h["file"], "ds": h["ds"]}
            if h.get("vlm"):
                hop["vlm"] = True
            hops.append(hop)
        scenarios.append({"id": s["id"], "label": s["label"], "vf": s["vf"],
                          "title": s["title"], "hops": hops})

    return tiers, scenarios, meta.get("glossary", []), adaptors, status, vlm_nodes


# ----------------------------------------------------------------------------- HTML rendering
CSS = """:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#30363d;--ink:#e6edf3;--mut:#8b949e;
--feed:#e06c9f;--meas:#58a6ff;--cog:#3fb950;--vlm:#bc8cff;--tune:#f0c040;--seam:#8b949e;--mono:ui-monospace,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px -apple-system,Segoe UI,sans-serif;padding-bottom:50px}
header{padding:14px 22px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:12.5px}
.ok{color:var(--cog);font-family:var(--mono);font-size:11px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:11px}
.scen{display:flex;gap:6px;flex-wrap:wrap}
.scn{padding:6px 11px;border:1px solid var(--line);border-radius:7px;background:var(--panel);cursor:pointer;font-size:12px}
.scn.on{border-color:var(--cog);background:#13201a;color:#fff}
.scn .vf{font-family:var(--mono);font-size:9.5px;color:var(--mut);display:block}
button.play{margin-left:auto;background:var(--cog);color:#04130a;border:none;border-radius:7px;padding:7px 16px;cursor:pointer;font-weight:700;font-size:13px}
button.step{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 12px;cursor:pointer}
.layout{display:grid;grid-template-columns:1.35fr 1fr;gap:16px;max-width:1240px;margin:0 auto;padding:16px 22px}
.stage{position:relative;border:1px solid var(--line);border-radius:12px;padding:12px;background:#0b0f14}
.tier{border-radius:10px;padding:9px 11px;margin-bottom:9px;border:1px solid var(--line)}
.tier.vlm{border-color:var(--vlm);background:linear-gradient(180deg,#16111f,#0b0f14)}
.tier.cog{border-color:var(--cog);background:linear-gradient(180deg,#0f1a13,#0b0f14)}
.tier.meas{border-color:var(--meas);background:linear-gradient(180deg,#0d151f,#0b0f14)}
.tier.feed{border-color:var(--feed);background:linear-gradient(180deg,#1c0e15,#0b0f14)}
.tl{font-family:var(--mono);font-size:9.5px;letter-spacing:.04em;text-transform:uppercase;margin-bottom:7px;display:flex;justify-content:space-between}
.vlm .tl{color:var(--vlm)}.cog .tl{color:var(--cog)}.meas .tl{color:var(--meas)}.feed .tl{color:var(--feed)}
.tl .sys{color:var(--mut)}
.nodes{display:flex;gap:7px;flex-wrap:wrap}
.node{flex:1;min-width:96px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 8px;transition:all .25s;position:relative}
.node .nn{font-weight:600;font-size:11.5px}.node .nd{color:var(--mut);font-size:10px;line-height:1.3;margin-top:2px}
.node.active{border-color:#fff;box-shadow:0 0 0 2px #ffffff55,0 0 16px #ffffff33;background:var(--panel2);transform:translateY(-1px)}
.node.vlmcall.active{box-shadow:0 0 0 2px var(--vlm),0 0 22px var(--vlm)}
.chips{display:flex;gap:3px;flex-wrap:wrap;margin-top:5px}
.ad{font-family:var(--mono);font-size:9px;padding:1px 5px;border-radius:9px;border:1px solid var(--line)}
.ad.in{border-color:var(--cog);color:var(--cog)}.ad.seam{border-color:var(--seam);color:var(--seam);border-style:dashed}
.tune{position:absolute;top:5px;right:6px;font-size:10px;color:var(--tune)}
.outnode{margin-top:4px;border:1px solid var(--cog);border-radius:9px;padding:8px 11px;text-align:center;background:#0f1a13;transition:all .25s}
.outnode.active{box-shadow:0 0 0 2px var(--cog),0 0 18px var(--cog)}.outnode b{color:var(--cog)}
.packet{position:absolute;width:15px;height:15px;border-radius:50%;background:#fff;box-shadow:0 0 16px 5px #ffffffaa;left:0;top:0;opacity:0;transition:left .5s cubic-bezier(.5,0,.3,1),top .5s cubic-bezier(.5,0,.3,1);z-index:9;pointer-events:none}
.side{display:flex;flex-direction:column;gap:14px}
.card{border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:12px 14px}
.card h3{margin:0 0 9px;font-size:13px}
.trace{min-height:200px}
.row{font-family:var(--mono);font-size:11px;padding:6px 8px;border-left:2px solid var(--line);margin-bottom:5px;background:var(--panel2);border-radius:0 6px 6px 0;opacity:.35;transition:opacity .2s}
.row.lit{opacity:1;border-left-color:#fff}
.row .h{color:var(--mut)}.row .fn{color:var(--ink);font-weight:600}.row .fl{color:var(--meas);font-size:10px}
.row .ds{color:var(--cog);font-size:10.5px;display:block;margin-top:2px}
.row.vlm{border-left-color:var(--vlm)}.row.vlm .fn{color:var(--vlm)}
.legend{font-family:var(--mono);font-size:10.5px;color:var(--mut);margin-top:6px}
.legend .ad{margin-right:4px}
.glos summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--ink)}
.gl{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.gi{font-size:11.5px;line-height:1.45}.gi b{color:var(--meas)}
.modmap{font-family:var(--mono);font-size:10px;color:var(--mut);margin-top:8px;line-height:1.7}.modmap b{color:var(--ink)}"""

# JS controller — preserved verbatim from the established template (scratch/architecture.html).
JS_CONTROLLER = """const $=s=>document.querySelector(s),stage=$('#stage'),packet=$('#packet');
let cur=S[0],idx=-1,timer=null;
const scen=$('#scen');
S.forEach((s,i)=>{const b=document.createElement('div');b.className='scn'+(i==0?' on':'');
 b.innerHTML=`${s.label}<span class="vf">${s.vf}</span>`;
 b.onclick=()=>{document.querySelectorAll('.scn').forEach(x=>x.classList.remove('on'));b.classList.add('on');pick(s);};scen.appendChild(b);});
function pick(s){cur=s;reset();$('#scentitle').textContent='— '+s.title;
 $('#trace').innerHTML=s.hops.map((h,i)=>`<div class="row ${h.vlm?'vlm':''}" id="row${i}"><span class="h">${i+1}.</span> <span class="fn">${h.fn}</span> <span class="fl">${h.file}</span><span class="ds">${h.ds}</span></div>`).join('');}
function reset(){idx=-1;if(timer){clearInterval(timer);timer=null;}packet.style.opacity=0;
 document.querySelectorAll('.node,.outnode').forEach(n=>n.classList.remove('active'));
 document.querySelectorAll('.row').forEach(r=>r.classList.remove('lit'));}
function moveTo(id){const n=document.getElementById(id),sr=stage.getBoundingClientRect(),nr=n.getBoundingClientRect();
 packet.style.opacity=1;packet.style.left=(nr.left-sr.left+nr.width/2-7)+'px';packet.style.top=(nr.top-sr.top+nr.height/2-7)+'px';}
function go(i){if(i>=cur.hops.length){return;}idx=i;const h=cur.hops[i];
 document.querySelectorAll('.node,.outnode').forEach(n=>n.classList.remove('active'));
 const n=document.getElementById(h.node);n.classList.add('active');moveTo(h.node);
 const r=document.getElementById('row'+i);r.classList.add('lit');r.scrollIntoView({block:'nearest'});}
$('#stepb').onclick=()=>{if(idx+1>=cur.hops.length)reset();go(idx+1);};
$('#playb').onclick=()=>{reset();let i=0;go(0);timer=setInterval(()=>{i++;if(i>=cur.hops.length){clearInterval(timer);timer=null;return;}go(i);},950);};
pick(S[0]);"""

E = html.escape


def _chips(adaptor_names, adaptors, status):
    out = []
    for name in adaptor_names:
        label = adaptors.get(name, {}).get("label", name)
        wired = status.get(name, False)
        cls, mark = ("in", "●") if wired else ("seam", "○")
        out.append(f'<span class="ad {cls}">{E(label)} {mark}</span>')
    return f'<div class="chips">{"".join(out)}</div>' if out else ""


def render_tier(tier, adaptors, status, vlm_nodes):
    tid = tier["id"]
    tl = f'{TIER_GLYPH[tid]} {E(tier["name"])} · {E(tier["path"])}'
    nodes_html = []
    for node in tier["nodes"]:
        cls = "node vlmcall" if node["id"] in vlm_nodes else "node"
        chips = _chips(node.get("adaptors", []), adaptors, status)
        nodes_html.append(
            f'<div class="{cls}" id="{E(node["id"])}"><div class="nn">{E(node["label"])}</div>'
            f'<div class="nd">{E(node["desc"])}</div>{chips}</div>')
    out = (f'<div class="tier {tid}"><div class="tl"><span>{tl}</span>'
           f'<span class="sys">{E(tier["system"])}</span></div>'
           f'<div class="nodes">{"".join(nodes_html)}</div>')
    if tid == "cog" and tier.get("output"):
        left, _, right = tier["output"].partition("→")
        out += f'<div class="outnode" id="n-out"><b>{E(left.strip())}</b> &nbsp;→ {E(right.strip())}</div>'
    return out + "</div>"


def render_html(model, prov: Provenance) -> str:
    tiers, scenarios, glossary, adaptors, status, vlm_nodes = model
    stage = "".join(render_tier(t, adaptors, status, vlm_nodes) for t in tiers)
    s_json = json.dumps(scenarios, ensure_ascii=False, separators=(",", ":"))
    # Escape <, >, & so a "</script>" or "<…>" in meta can't break out of the <script> block or corrupt
    # the artifact. These unescape back to the real characters when JSON.parse runs (rendered text is
    # unchanged); it just can't terminate the tag.
    s_json = s_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    glos = "".join(f'<div class="gi"><b>{E(g["term"])}</b> — {E(g["def"])}</div>' for g in glossary)
    n_int = sum(1 for v in status.values() if v)
    n_seam = len(status) - n_int
    parts = [
        "<!doctype html>",
        f"<!-- generated by tools/build_architecture.py · {prov.stamp} · DO NOT EDIT -->",
        f'<html lang="en" data-version="{prov.version}">',
        '<head><meta charset="utf-8"><title>Percept · the real architecture, animated</title>',
        f"<style>{CSS}</style></head><body>",
        "<header>",
        f' <h1>Percept — the real architecture <span class="ok">✓ generated from code · {E(prov.stamp)}</span></h1>',
        f' <div class="sub">Pick a scenario, hit play, watch a real call animate through the stack. '
        f'Adaptors marked ● integrated ({n_int}) / ○ seam ({n_seam}) — decided from the code, not this page.</div>',
        ' <div class="controls"><div class="scen" id="scen"></div>'
        '<button class="step" id="stepb">step ›</button>'
        '<button class="play" id="playb">▶ play call</button></div>',
        "</header>",
        '<div class="layout"><div class="stage" id="stage">',
        stage,
        '<div class="packet" id="packet"></div></div>',
        '<div class="side">',
        '<div class="card trace"><h3>call trace <span style="color:var(--mut);font-weight:400;'
        'font-size:11px" id="scentitle"></span></h3><div id="trace"></div></div>',
        '<div class="card"><h3>integration</h3>'
        '<div class="legend"><span class="ad in">● integrated</span> '
        '<span class="ad seam">○ seam (register to add)</span> &nbsp;— adaptors resolve via '
        '<code>registry.make(seam,name)</code>; the marker is computed from the code at build time.</div></div>',
        '<div class="card"><details class="glos" open><summary>glossary</summary>',
        f'<div class="gl">{glos}</div></details>',
        '<div class="modmap">module map ✓ &nbsp; <b>percept.contracts</b> frozen wire · '
        '<b>percept.harness</b> measures/edge · <b>percept</b> cognition + VLM wiring · '
        '<b>percept-edge</b> JS device transport</div></div>',
        "</div></div>",
        f"<script>\nconst S={s_json};\n{JS_CONTROLLER}\n</script>",
        "</body></html>",
        "",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------------- SVG rendering
def render_svg(model, prov: Provenance) -> str:
    tiers, _scen, _glos, adaptors, status, _vlm = model
    W, PAD, HEADER_H, BAND_H, GAP, FOOTER_H = 1000, 24, 64, 132, 16, 34
    H = HEADER_H + len(tiers) * BAND_H + (len(tiers) - 1) * GAP + FOOTER_H
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" data-version="{prov.version}" '
         f'font-family="-apple-system,Segoe UI,sans-serif">',
         f"<!-- generated by tools/build_architecture.py · {prov.stamp} · DO NOT EDIT -->",
         f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
         f'<text x="{PAD}" y="34" fill="{COLORS["ink"]}" font-size="20" font-weight="700">'
         f'Percept — the architecture</text>',
         f'<text x="{W-PAD}" y="34" fill="{COLORS["mut"]}" font-size="12" text-anchor="end" '
         f'font-family="monospace">{E(prov.stamp)}</text>']
    for i, tier in enumerate(tiers):
        tid = tier["id"]
        col = COLORS[tid]
        by = HEADER_H + i * (BAND_H + GAP)
        p.append(f'<rect x="{PAD}" y="{by}" width="{W-2*PAD}" height="{BAND_H}" rx="10" '
                 f'fill="{COLORS["panel"]}" stroke="{col}" stroke-width="1.5"/>')
        glyph = "FIRE · ASK · SILENT" if tid == "cog" else ""
        p.append(f'<text x="{PAD+14}" y="{by+24}" fill="{col}" font-size="13" font-weight="700">'
                 f'{TIER_GLYPH[tid]} {E(tier["name"])}</text>')
        p.append(f'<text x="{PAD+14}" y="{by+40}" fill="{COLORS["mut"]}" font-size="10.5" '
                 f'font-family="monospace">{E(tier["path"])}{("  ·  gate: " + glyph) if glyph else ""}</text>')
        nodes = tier["nodes"]
        n = len(nodes)
        inner_x = PAD + 14
        avail = (W - 2 * PAD - 28) - (n - 1) * 10
        cw = avail // n if n else 0  # a tier with 0 nodes draws an empty band (no ZeroDivision)
        cy = by + 54
        ch = BAND_H - 66
        for j, node in enumerate(nodes):
            cx = inner_x + j * (cw + 10)
            w = cw if j < n - 1 else (W - PAD - 14 - cx)
            p.append(f'<rect x="{cx}" y="{cy}" width="{int(w)}" height="{ch}" rx="7" '
                     f'fill="{COLORS["bg"]}" stroke="{COLORS["line"]}"/>')
            p.append(f'<text x="{cx+9}" y="{cy+20}" fill="{COLORS["ink"]}" font-size="12" '
                     f'font-weight="600">{E(node["label"])}</text>')
            p.append(f'<text x="{cx+9}" y="{cy+36}" fill="{COLORS["mut"]}" font-size="9.5">'
                     f'{E(node["desc"][:34])}</text>')
            ax = cx + 9
            for name in node.get("adaptors", []):
                label = adaptors.get(name, {}).get("label", name)
                wired = status.get(name, False)
                mark, mc = ("●", COLORS["cog"]) if wired else ("○", COLORS["seam"])
                txt = f"{label} {mark}"
                p.append(f'<text x="{ax}" y="{cy+54}" fill="{mc}" font-size="9.5" '
                         f'font-family="monospace">{E(txt)}</text>')
                ax += int(len(txt) * 6.0) + 8
    fy = H - 12
    p.append(f'<text x="{PAD}" y="{fy}" fill="{COLORS["mut"]}" font-size="10.5" '
             f'font-family="monospace">generated from code · drift fails the build · '
             f'{E(prov.stamp)}</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


# ----------------------------------------------------------------------------- CLI
def fail_report() -> int:
    print(f"\narchitecture build FAILED — {len(STALE)} stale reference(s); no output written.\n",
          file=sys.stderr)
    for s in sorted(STALE, key=lambda x: (x.kind, x.ref)):
        print(f"  [{s.kind}] {s.ref}\n    file: {s.file}\n    → {s.detail}\n", file=sys.stderr)
    print("regenerate meta or fix the code, then re-run `make architecture`.", file=sys.stderr)
    return 2


def emit(path: Path, text: str, check: bool) -> bool:
    """Return True if the file would change. Compares RAW BYTES (not read_text, which would translate
    CRLF->LF on an autocrlf checkout and falsely report 'no drift' while the CI git-diff gate flags it)."""
    want = text.encode("utf-8")
    existing = path.read_bytes() if path.exists() else None
    if check:
        if existing != want:
            import difflib
            diff = difflib.unified_diff(
                (existing.decode("utf-8", "replace") if existing else "").splitlines(True),
                text.splitlines(True),
                fromfile=f"{path} (committed)", tofile=f"{path} (regenerated)")
            sys.stderr.writelines(diff)
            return True
        return False
    path.write_bytes(want)  # bytes -> no newline translation on any platform
    return existing != want


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the architecture diagram from code + meta.")
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--out", default=str(DEFAULT_HTML))
    ap.add_argument("--svg", default=str(DEFAULT_SVG))
    ap.add_argument("--date", default=None, help="ISO build date (else SOURCE_DATE_EPOCH / HEAD date)")
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 if output would change")
    args = ap.parse_args(argv)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    model = build_model(meta)
    if STALE:
        return fail_report()

    prov = Provenance(get_version())
    html_text = render_html(model, prov)
    svg_text = render_svg(model, prov)

    for w in WARNINGS:
        print(f"  warning: {w}", file=sys.stderr)

    changed = False
    changed |= emit(Path(args.out), html_text, args.check)
    changed |= emit(Path(args.svg), svg_text, args.check)

    # Operator-facing provenance (HEAD sha + date) — printed, never embedded in the gated bytes.
    origin = f"{prov.stamp} · {get_sha()} · {get_date(args.date)}"
    if args.check:
        if changed:
            print("architecture diagram is STALE — run `make architecture` and commit.", file=sys.stderr)
            return 1
        print(f"architecture diagram up to date · {origin}")
        return 0
    print(f"wrote {args.out} + {args.svg} · {origin}"
          + (f" · {len(WARNINGS)} drift warning(s)" if WARNINGS else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
