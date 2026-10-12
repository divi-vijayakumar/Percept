"""Structural guardrails for the percept package.

Run from the repo root:

    python tools/guardrails/guardrails.py all
    python tools/guardrails/guardrails.py size --max-lines 10000

Each check prints its findings and exits non-zero on a violation. They read source with
``ast`` and never import the package, so they work on a broken tree.

    exports       __all__ in percept/__init__.py has at most MAX_EXPORTS names
    boundaries    imports point down: contracts imports no percept module, core modules
                  import only contracts/core and no I/O library
    reachability  every module under percept/ is reachable by imports from percept/__init__.py
    size          total lines under percept/ (exit 2 when over the review threshold: growth
                  past it is reviewed by a maintainer, not a hard failure)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_EXPORTS = 30
MAX_LINES = 10_000

CONTRACTS = "percept.contracts"
# The functional core: pure, no I/O, time passed in.
CORE = {
    "percept.contracts",
    "percept.decide",
    "percept.attention",
    "percept.triggers",
    "percept.zones",
    "percept.clips",
    "percept.judges.base",
    "percept.replay",
    "percept.plans",
    "percept.plans.labels",
    "percept.plans.score",
}
# Packages whose every module is core, except the shell modules listed in SHELL.
CORE_PACKAGES = ("percept.plans",)
# Shell modules inside a core package: the evaluation driver reads video and runs a Monitor.
SHELL = {"percept.plans.run"}
# Shell modules a core module may import to build a value it never calls: plans builds a
# cascade judge for the runtime to drive.
CORE_BUILDS = {"percept.plans": {"percept.judges._cascade"}}
# Top-level modules a core module must never import.
IO_MODULES = {
    "asyncio", "socket", "threading", "multiprocessing", "subprocess", "concurrent",
    "sqlite3", "http", "urllib", "ssl", "selectors", "signal",
    "cv2", "httpx", "requests", "aiohttp",
    "google", "anthropic", "openai",
}
# Modules a core module may import from numpy-free stdlib or these third parties.
CORE_THIRD_PARTY = {"numpy"}


@dataclass(frozen=True)
class Module:
    name: str
    path: Path
    is_package: bool


def package_dir(root: Path) -> Path:
    return root / "packages" / "percept-vision" / "percept"


def discover(root: Path) -> dict[str, Module]:
    pkg = package_dir(root)
    mods: dict[str, Module] = {}
    for path in sorted(pkg.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(pkg.parent).with_suffix("")
        parts = list(rel.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        name = ".".join(parts)
        mods[name] = Module(name, path, is_package)
    return mods


def imports_of(mod: Module) -> list[tuple[str, int]]:
    """Absolute dotted names imported by ``mod``, with line numbers."""
    tree = ast.parse(mod.path.read_text(encoding="utf-8"), filename=str(mod.path))
    base = mod.name.split(".") if mod.is_package else mod.name.split(".")[:-1]
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                target = ".".join(anchor + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            found.append((target, node.lineno))
            # `from percept.x import y` may name a submodule y.
            found += [(f"{target}.{a.name}", node.lineno) for a in node.names]
    return found


def resolve(name: str, mods: dict[str, Module]) -> str | None:
    """Longest percept module that ``name`` refers to, if any."""
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in mods:
            return candidate
        parts.pop()
    return None


def check_exports(root: Path, limit: int = MAX_EXPORTS) -> list[str]:
    init = package_dir(root) / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                return ["__all__ must be a literal list or tuple"]
            names = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
            if len(names) > limit:
                return [f"__all__ has {len(names)} names; the limit is {limit}"]
            return []
    return ["percept/__init__.py defines no __all__"]


def is_core(name: str) -> bool:
    return name not in SHELL and (name in CORE or name.startswith(tuple(p + "." for p in CORE_PACKAGES)))


def check_boundaries(root: Path) -> list[str]:
    mods = discover(root)
    problems: list[str] = []
    for mod in mods.values():
        in_core = is_core(mod.name)
        owner = next((p for p in CORE_PACKAGES if mod.name == p or mod.name.startswith(p + ".")), mod.name)
        for name, line in imports_of(mod):
            top = name.split(".")[0]
            where = f"{mod.path.relative_to(root)}:{line}"
            target = resolve(name, mods) if top == "percept" else None
            if mod.name == CONTRACTS and target is not None and target != "percept":
                problems.append(f"{where}: contracts must not import {target}")
            if not in_core:
                continue
            if top in IO_MODULES:
                problems.append(f"{where}: core module {mod.name} imports I/O module {name}")
            elif target is not None and target != "percept" and not is_core(target) \
                    and target not in CORE_BUILDS.get(owner, ()):
                problems.append(f"{where}: core module {mod.name} imports shell module {target}")
            elif top not in ("percept", "__future__") and top not in CORE_THIRD_PARTY \
                    and top not in sys.stdlib_module_names:
                problems.append(f"{where}: core module {mod.name} imports third-party {top}")
    return problems


def check_reachability(root: Path) -> list[str]:
    mods = discover(root)
    if "percept" not in mods:
        return ["percept/__init__.py is missing"]
    seen = {"percept"}
    stack = ["percept"]
    while stack:
        current = mods[stack.pop()]
        for name, _ in imports_of(current):
            target = resolve(name, mods) if name.startswith("percept") else None
            if target and target not in seen:
                seen.add(target)
                stack.append(target)
        # A package is reachable when any of its submodules is: importing
        # percept.judges.gemini also imports percept.judges.
        parts = current.name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent in mods and parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return [f"{m.path.relative_to(root)} is not reachable from percept/__init__.py"
            for m in mods.values() if m.name not in seen]


def count_lines(root: Path) -> int:
    return sum(len(m.path.read_text(encoding="utf-8").splitlines())
               for m in discover(root).values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=["exports", "boundaries", "reachability", "size", "all"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-lines", type=int, default=MAX_LINES)
    args = parser.parse_args(argv)

    checks = {
        "exports": check_exports,
        "boundaries": check_boundaries,
        "reachability": check_reachability,
    }
    selected = list(checks) if args.check == "all" else [c for c in checks if c == args.check]
    failed = False
    for name in selected:
        problems = checks[name](args.root)
        print(f"{name}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        for problem in problems:
            print(f"  {problem}")
        failed = failed or bool(problems)

    if args.check in ("size", "all"):
        lines = count_lines(args.root)
        over = lines > args.max_lines
        print(f"size: {lines} lines (review threshold {args.max_lines})")
        if over and args.check == "size":
            print("  over the review threshold: needs a maintainer's review")
            return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
