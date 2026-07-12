"""Zero-dependency JSON Schema export for the contract DTO dataclasses.

Stdlib only (``dataclasses`` + ``typing`` + ``json``). We deliberately avoid pydantic: the
contract package is zero-dep by design. This emits a *pragmatic* JSON Schema (draft 2020-12)
for the vendor-neutral DTOs — enough for clients to validate wire payloads and for tooling to
discover the shape — not a perfect reflection of every Python type.

Regenerate the committed schema with::

    python3 -m percept.contracts.jsonschema            # -> percept/contracts/schema/*.json

or call :func:`generate_all` / :func:`write_schema_files` from code.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from . import (
    CONTRACT_VERSION,
    Attribute,
    AudioFrame,
    Box,
    ClipEvent,
    Concern,
    Consequence,
    Decision,
    Evidence,
    Goal,
    Incident,
    IncidentSnippet,
    LabelRecord,
    Observation,
    Priors,
    Schedule,
    STTEvent,
    Subject,
    Tier0Signal,
    TrackingBehavior,
    TurnResult,
    Verdict,
    WatchCondition,
    WatchSpec,
)

_NoneType = type(None)

#: The DTO dataclasses we export schema for. Order = stable output / one $defs entry each.
DTOS: tuple[type, ...] = (
    Box,
    Observation,
    ClipEvent,
    Verdict,
    Attribute,
    Goal,
    Consequence,
    Evidence,
    TrackingBehavior,
    Subject,
    Priors,
    Concern,
    Decision,
    STTEvent,
    TurnResult,
    Schedule,
    WatchCondition,
    WatchSpec,
    Tier0Signal,
    AudioFrame,
    IncidentSnippet,
    Incident,
    LabelRecord,
)

_PRIMITIVE: dict[Any, dict] = {
    bool: {"type": "boolean"},
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bytes: {"type": "string", "contentEncoding": "base64"},
    _NoneType: {"type": "null"},
}


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _type_schema(tp: Any) -> dict:
    """Map a Python type annotation to a JSON Schema fragment (best-effort, stdlib only)."""
    if tp in _PRIMITIVE:
        return dict(_PRIMITIVE[tp])

    origin = get_origin(tp)

    # typing.Literal["a", "b"] -> enum of those values
    if origin is typing.Literal:
        values = list(get_args(tp))
        return {"enum": values}

    # Optional[X] / Union[...] / PEP 604 `X | None` -> anyOf, with None as a {"type": "null"}
    if origin is Union or origin is types.UnionType:
        sub = [_type_schema(a) for a in get_args(tp)]
        return {"anyOf": sub}

    # list / List[X]
    if origin in (list, typing.List):  # noqa: UP006 — runtime origin check
        args = get_args(tp)
        if args:
            return {"type": "array", "items": _type_schema(args[0])}
        return {"type": "array"}

    # dict / Dict[K, V]
    if origin in (dict, typing.Dict):  # noqa: UP006
        args = get_args(tp)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _type_schema(args[1])}
        return {"type": "object"}

    # nested DTO dataclass -> $ref into $defs
    if _is_dataclass_type(tp):
        return {"$ref": f"#/$defs/{tp.__name__}"}

    # typing.Any / unknown -> permissive
    return {}


def _dataclass_schema(cls: type) -> dict:
    """JSON Schema for one dataclass: properties + required (fields without a default)."""
    hints = typing.get_type_hints(cls)
    props: dict[str, dict] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        props[f.name] = _type_schema(hints.get(f.name, f.type))
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if not has_default:
            required.append(f.name)
    schema: dict[str, Any] = {
        "type": "object",
        "title": cls.__name__,
        "properties": props,
    }
    doc = (cls.__doc__ or "").strip()
    if doc:
        schema["description"] = doc.splitlines()[0].strip()
    if required:
        schema["required"] = required
    return schema


def generate_all(dtos: tuple[type, ...] = DTOS) -> dict:
    """Build a single JSON Schema document with every DTO under ``$defs``."""
    defs = {cls.__name__: _dataclass_schema(cls) for cls in dtos}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://percept.velvee.ai/contracts/percept-contracts.schema.json",
        "title": "percept-contracts",
        "description": "Vendor-neutral DTOs of the Percept frozen contract.",
        "x-contract-version": CONTRACT_VERSION,
        "$defs": defs,
    }


def schema_dir() -> Path:
    """The committed schema output directory: ``percept/contracts/schema``."""
    return Path(__file__).resolve().parent / "schema"


def write_schema_files(out_dir: Path | None = None) -> list[Path]:
    """Write the combined document + one file per DTO. Returns the paths written."""
    out_dir = out_dir or schema_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    combined = generate_all()
    combined_path = out_dir / "percept-contracts.schema.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")
    written.append(combined_path)

    for cls in DTOS:
        single = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://percept.velvee.ai/contracts/{cls.__name__}.schema.json",
            "x-contract-version": CONTRACT_VERSION,
            **_dataclass_schema(cls),
            "$defs": {c.__name__: _dataclass_schema(c) for c in DTOS if c is not cls},
        }
        p = out_dir / f"{cls.__name__}.schema.json"
        p.write_text(json.dumps(single, indent=2) + "\n")
        written.append(p)

    return written


def main(argv: list[str] | None = None) -> int:
    paths = write_schema_files()
    print(f"wrote {len(paths)} schema file(s) to {schema_dir()}")
    for p in paths:
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
