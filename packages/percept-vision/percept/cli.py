"""``percept`` — the thin umbrella CLI over the package's independent console scripts.

Eight `[project.scripts]` entry points already exist (``percept-watch``, ``percept-doctor``,
``percept-replay``, ``percept-bench``, ``percept-eval``, ``percept-derive-profile``,
``percept-optimize``, ``percept-schema``), each a self-contained ``main(argv) -> int``. This module invents no new
argument parsing for any of them — it only routes ``percept <name> ...`` to the SAME unmodified
``main`` function the standalone script calls, so ``percept watch --source 0`` and
``percept-watch --source 0`` are byte-identical in behavior. A caller who already has muscle
memory for the standalone scripts loses nothing; a caller who wants one root verb gets it.

Three subcommands are NEW here, not delegated, because no earlier entry point owned this surface:

- ``percept resolve`` — run a :class:`~percept.contracts.watchplan.WatchPlan` through
  :class:`~percept.resolver.ConfigResolver` and print the result's per-field provenance
  (:meth:`~percept.contracts.watchplan.ResolvedWatch.explain`) or its full JSON shape.
- ``percept config tunables`` — list the live :data:`percept.tunables.REGISTRY` inventory (the same
  content ``tunables.json`` ships), without requiring a caller to import Python to see it.
- ``percept components list`` / ``percept components describe`` — list or describe one entry from
  :mod:`percept.catalog`, the read-only survey over the five selectable-component registries.

Every subcommand here is byte-identical whether reached through ``python -m percept.cli`` or the
``percept`` console script — this module owns no state, no I/O beyond stdout/stderr and the files a
caller names on the command line.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

#: subcommand name -> dotted module path whose unmodified ``main(argv) -> int`` this command
#: delegates to VERBATIM. Kept as a plain mapping (not a registry class) so adding a future
#: standalone script here is a one-line change, not a new abstraction.
_DELEGATES: dict[str, str] = {
    "watch": "percept.watch_cli",
    "doctor": "percept.doctor",
    "replay": "percept.replay",
    "bench": "percept.bench",
    "eval": "percept.eval.differential",
    "derive-profile": "percept.eval.derivation",
    "optimize": "percept.optimizer",
    "schema": "percept.contracts.jsonschema",
}

#: New, non-delegated subcommands this module owns directly.
_OWN_COMMANDS: tuple[str, ...] = ("resolve", "config", "components")

_ALL_COMMANDS: tuple[str, ...] = tuple(sorted(set(_DELEGATES) | set(_OWN_COMMANDS)))


def _print_top_help() -> None:
    lines = [
        "usage: percept <command> [args...]",
        "",
        "Delegating commands (identical to the standalone percept-<name> script):",
    ]
    for name in sorted(_DELEGATES):
        lines.append(f"  {name:<16} -> percept-{name}")
    lines.append("  eval ingest-trace -> percept.eval.telemetry_fixtures (a .ptrace + labels sidecar "
                 "-> labeled derivation fixture cases)")
    lines.append("")
    lines.append("Own commands:")
    lines.append("  resolve          resolve a WatchPlan config and show per-field provenance")
    lines.append("  config tunables  list the tunables registry (tunables.json inventory)")
    lines.append("  components list|describe   list or describe one percept.catalog entry")
    print("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_top_help()
        return 2
    if args[0] in ("-h", "--help"):
        _print_top_help()
        return 0
    command, rest = args[0], args[1:]
    if command == "eval" and rest and rest[0] == "ingest-trace":
        # NOT a delegate: percept.eval.differential.main() parses `eval`'s OWN argv shape
        # (--baseline/--candidate/...) and would reject "ingest-trace" as an unknown positional.
        # This is a second, independent job under the same verb — card L2's telemetry -> fixtures
        # pipeline — so it is routed to its own module before the delegate table is consulted.
        mod = importlib.import_module("percept.eval.telemetry_fixtures")
        return int(mod.main(rest[1:]))
    if command in _DELEGATES:
        mod = importlib.import_module(_DELEGATES[command])
        return int(mod.main(rest))
    if command == "resolve":
        return _cmd_resolve(rest)
    if command == "config":
        return _cmd_config(rest)
    if command == "components":
        return _cmd_components(rest)
    print(f"percept: unknown command {command!r}", file=sys.stderr)
    print(f"percept: known commands: {', '.join(_ALL_COMMANDS)}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# percept resolve
# ---------------------------------------------------------------------------


def _load_structured(path: Path) -> dict[str, Any]:
    """Load a ``.json`` (always) or ``.yaml``/``.yml`` (needs PyYAML) plan config document.

    Mirrors :func:`percept.eval.differential._load_structured`'s own JSON-always/YAML-optional
    split (percept core deliberately has zero runtime dependencies) without importing that module,
    since ``percept.cli`` is meant to stay a thin, independent surface over the rest of the package.
    """
    if not path.exists():
        raise SystemExit(f"percept resolve: file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415 - optional; percept core has zero runtime deps
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                f"percept resolve: {path} is YAML but PyYAML is not installed. Install it "
                f"(`pip install pyyaml`) or pass the same document as JSON."
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise SystemExit(f"percept resolve: {path}: expected a mapping at the top level")
    return data


def _build_resolve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="percept resolve", description=(
        "Resolve a WatchPlan (from a config file, or --condition on the command line) through "
        "ConfigResolver and print its per-field provenance."
    ))
    parser.add_argument(
        "config", nargs="?", default=None,
        help="path to a JSON/YAML document shaped like WatchPlan's constructor kwargs, plus an "
             "optional top-level 'source_profile' name to look up",
    )
    parser.add_argument("--condition", default=None, help="condition text, when not using --config")
    parser.add_argument(
        "--source-profile", default=None,
        help="a shipped SourceProfile name (e.g. physical_scene, screen_document, conversation)",
    )
    parser.add_argument("--explain", action="store_true", help="print ResolvedWatch.explain()")
    parser.add_argument("--json", action="store_true", help="print ResolvedWatch.to_dict() as JSON")
    return parser


def _cmd_resolve(argv: Sequence[str]) -> int:
    from .contracts.watchplan import WatchPlan
    from .resolver import ConfigResolver
    from .sourceprofile import SourceProfileNotFoundError, get_source_profile

    parser = _build_resolve_parser()
    ns = parser.parse_args(list(argv))

    source_profile_name = ns.source_profile
    if ns.config:
        data = _load_structured(Path(ns.config))
        data = dict(data)
        source_profile_name = data.pop("source_profile", None) or source_profile_name
        if "condition" not in data:
            if ns.condition is None:
                print("percept resolve: no 'condition' in config and no --condition given", file=sys.stderr)
                return 2
            data["condition"] = ns.condition
        try:
            # from_dict (not WatchPlan(**data)) so the flat judge_context_* wire keys round-trip back
            # into an ArtifactBinding; a half-specified binding raises ValueError, an unknown field TypeError.
            plan = WatchPlan.from_dict(data)
        except (TypeError, ValueError) as exc:
            print(f"percept resolve: invalid WatchPlan fields in {ns.config}: {exc}", file=sys.stderr)
            return 2
    else:
        if not ns.condition:
            print("percept resolve: pass a config file or --condition", file=sys.stderr)
            return 2
        plan = WatchPlan(condition=ns.condition)

    source_profile = None
    if source_profile_name:
        try:
            source_profile = get_source_profile(source_profile_name)
        except SourceProfileNotFoundError as exc:
            print(f"percept resolve: {exc}", file=sys.stderr)
            return 2

    resolved = ConfigResolver().resolve(
        plan, source_profile=source_profile, source_profile_explicit=source_profile is not None,
    )

    if ns.json:
        print(json.dumps(resolved.to_dict(), indent=2, sort_keys=True, default=str))
    elif ns.explain:
        print(resolved.explain())
    else:
        print(f"resolved {len(resolved.effective)} field(s); plan_hash={resolved.plan_hash[:12]}")
        print("(pass --explain for per-field provenance, --json for the full document)")
    return 0


# ---------------------------------------------------------------------------
# percept config
# ---------------------------------------------------------------------------


def _cmd_config(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="percept config")
    sub = parser.add_subparsers(dest="subcommand")
    tunables_p = sub.add_parser("tunables", help="list the tunables registry")
    tunables_p.add_argument("--owner", default=None, help="filter to one owner (e.g. Gate, SourceProfile)")
    tunables_p.add_argument("--json", action="store_true", help="print the full tunables.json document")
    ns = parser.parse_args(list(argv))
    if ns.subcommand != "tunables":
        parser.print_help()
        return 0 if ns.subcommand is None else 2
    return _cmd_config_tunables(ns)


def _cmd_config_tunables(ns: argparse.Namespace) -> int:
    from .tunables import REGISTRY, generate

    if ns.json:
        print(generate())
        return 0
    specs = REGISTRY
    if ns.owner:
        specs = tuple(s for s in specs if s.owner == ns.owner)
    for spec in specs:
        env = f" env={spec.env_var}" if spec.env_var else ""
        print(f"{spec.path}  ({spec.owner}, {spec.risk})  default={spec.default!r}{env}")
    print(f"{len(specs)} tunable(s)")
    return 0


# ---------------------------------------------------------------------------
# percept components
# ---------------------------------------------------------------------------


def _cmd_components(argv: Sequence[str]) -> int:
    from . import catalog

    parser = argparse.ArgumentParser(prog="percept components")
    sub = parser.add_subparsers(dest="subcommand")

    list_p = sub.add_parser("list", help="list every catalog entry")
    list_p.add_argument("--role", default=None, choices=catalog.ROLES, help="filter to one role")
    list_p.add_argument("--json", action="store_true")

    describe_p = sub.add_parser("describe", help="describe one catalog entry")
    describe_p.add_argument("ref", help="'name' or 'role:name' (role prefix disambiguates)")
    describe_p.add_argument("--json", action="store_true")

    ns = parser.parse_args(list(argv))
    if ns.subcommand == "list":
        return _cmd_components_list(ns, catalog)
    if ns.subcommand == "describe":
        return _cmd_components_describe(ns, catalog)
    parser.print_help()
    return 0 if ns.subcommand is None else 2


def _cmd_components_list(ns: argparse.Namespace, catalog: Any) -> int:
    components = catalog.list_components(role=ns.role)
    if ns.json:
        print(json.dumps([c.to_dict() for c in components], indent=2, sort_keys=True))
        return 0
    for c in components:
        seam = f" seam={c.seam}" if c.seam else ""
        print(f"{c.role}:{c.name}{seam}  {c.summary}")
    print(f"{len(components)} component(s)")
    return 0


def _cmd_components_describe(ns: argparse.Namespace, catalog: Any) -> int:
    ref = ns.ref
    if ":" in ref:
        role, _, name = ref.partition(":")
        matches = [c for c in catalog.list_components(role=role) if c.name == name]
    else:
        matches = [c for c in catalog.list_components() if c.name == ref]
    if not matches:
        print(f"percept components describe: no component matches {ref!r}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        roles = ", ".join(sorted(f"{c.role}:{c.name}" for c in matches))
        print(
            f"percept components describe: {ref!r} is ambiguous across roles ({roles}); "
            f"pass 'role:name' to disambiguate",
            file=sys.stderr,
        )
        return 2
    descriptor = matches[0]
    if ns.json:
        print(json.dumps(descriptor.to_dict(), indent=2, sort_keys=True))
        return 0
    print(f"role:      {descriptor.role}")
    print(f"name:      {descriptor.name}")
    print(f"seam:      {descriptor.seam or '(none)'}")
    print(f"category:  {descriptor.category or '(none)'}")
    print(f"summary:   {descriptor.summary or '(none)'}")
    print(f"metadata:  {json.dumps(descriptor.metadata, indent=2, sort_keys=True)}")
    return 0


__all__ = ["main"]
