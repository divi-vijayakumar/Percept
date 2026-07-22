"""The portable trace bundle — ONE file a client attaches to a bug report, and we replay their run.

A media REFERENCE is not self-contained. A trace that says ``file:///Users/them/clip.mp4`` is a
perfectly good local artifact on their machine and useless on ours. So this module names two explicit
modes, and the mode is always written into the artifact so nobody has to guess which one they hold:

============  ==============================================================================
``local``     manifest + events + summary + media **references** (uri + sha256). Media stays
              outside. This is the default, and the only mode that touches no client pixels.
``portable``  manifest + events + summary + **selected media bytes**, packed into a
              ``run-123.ptrace`` directory (or a single ``run-123.ptrace.zip`` to attach).
============  ==============================================================================

::

    run-123.ptrace/
    |-- manifest.json      (run manifest + bundle header: mode, hashes, retention, expiry)
    |-- events.jsonl       (one TraceRecord per line, ingest order)
    |-- summary.json       (the derived funnel — percept.tracing.TraceSummary)
    `-- evidence/          (ONLY in portable mode: frame-001.jpg, capsule-007.json, ...)

**Media inclusion is opt-in and defaults OFF, and that is a security decision, not a convenience
one.** The first client workload is accounting screens. A bundle that ships pixels by default is an
exfiltration vector wearing a support-ticket costume: the person who runs ``export_bug_report()`` at
2am is not the person who decides what may leave the building. So :class:`BundlePolicy` starts at
``include_media=False``; evidence handed to a local-mode bundle is DROPPED (not written, not
hashed into the payload) and the count of what was dropped is written into the manifest, so the
omission is visible rather than silent.

Everything is content-hashed (``sha256`` per file, plus a ``payload_sha256`` over the ordered file
digests), carries a retention/expiry stamp, and can be verified after transport with
:meth:`TraceBundle.verify` — a bundle that was truncated in an email pipeline says so instead of
replaying half a run.

What a bundle DOES and DOES NOT let you do
------------------------------------------
It lets you run a **cognition replay** (:mod:`percept.replay`): the recorded normalized verdicts and
timestamps back through the Gate and the incident logic, deterministically, with no model call. It
does NOT let you reproduce perception — re-sending the bundled media to a live model is **media
reevaluation**, which is not deterministic and belongs to ``percept eval``. A bundle never claims to
"replay the run" on the strength of a media URI (see ADR-004 §2).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import zipfile
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from .tracing import (
    TRACE_SCHEMA_VERSION,
    RunTraceManifest,
    TraceRecord,
    TraceRecorder,
    TraceSummary,
    canonical_json,
)

#: Wire version of the BUNDLE layout (the four-file shape + its manifest header). Independent of
#: :data:`percept.tracing.TRACE_SCHEMA_VERSION`, which versions the trace payload itself.
BUNDLE_SCHEMA_VERSION = 1

#: The directory suffix of an unpacked bundle; the single-file form is this plus ``.zip``.
BUNDLE_SUFFIX = ".ptrace"

MANIFEST_NAME = "manifest.json"
EVENTS_NAME = "events.jsonl"
SUMMARY_NAME = "summary.json"
EVIDENCE_DIR = "evidence"

#: ``local`` = media stays outside, referenced by uri+hash. ``portable`` = selected media bytes are
#: inside the artifact. There is no third, ambiguous state.
BundleMode = Literal["local", "portable"]

#: Default retention for an exported bundle. A support artifact that never expires becomes a
#: forgotten copy of a client's screens; an explicit ``None`` (keep forever) has to be typed.
DEFAULT_RETENTION_DAYS = 30.0


class BundleError(Exception):
    """A bundle could not be built, read or verified. Typed, never a silent empty result."""


@dataclass(frozen=True)
class BundlePolicy:
    """What a bundle is allowed to CARRY. Media is opt-in; everything else is a bound.

    ``include_media=False`` (the default) is the whole security posture of this module: evidence
    bytes are dropped rather than packed, and :attr:`TraceBundle.media_omitted` records how many.
    Turning it on is an explicit, auditable act by whoever exports the bundle.
    """

    include_media: bool = False
    retention_days: float | None = DEFAULT_RETENTION_DAYS
    max_media_files: int = 512
    max_media_bytes: int = 64_000_000
    #: Record-data keys stripped from every event before writing (on top of the recorder's own
    #: media stripping) — a client-named redaction list, e.g. ``("ocr_text", "window_title")``.
    redact_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.retention_days is not None and self.retention_days <= 0:
            raise ValueError("retention_days must be positive or None")
        if self.max_media_files < 0:
            raise ValueError("max_media_files must be >= 0")
        if self.max_media_bytes < 0:
            raise ValueError("max_media_bytes must be >= 0")

    @classmethod
    def local(cls, **kw: Any) -> "BundlePolicy":
        """The default posture: no media leaves the client's machine."""
        return cls(include_media=False, **kw)

    @classmethod
    def portable(cls, **kw: Any) -> "BundlePolicy":
        """Opt IN to packing media. Named so it cannot be reached by accident."""
        kw.setdefault("include_media", True)
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_media": self.include_media,
            "retention_days": self.retention_days,
            "max_media_files": self.max_media_files,
            "max_media_bytes": self.max_media_bytes,
            "redact_keys": list(self.redact_keys),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundlePolicy":
        return cls(
            include_media=bool(payload.get("include_media", False)),
            retention_days=payload.get("retention_days", DEFAULT_RETENTION_DAYS),
            max_media_files=int(payload.get("max_media_files", 512)),
            max_media_bytes=int(payload.get("max_media_bytes", 64_000_000)),
            redact_keys=tuple(payload.get("redact_keys") or ()),
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of media/context BYTES, eligible to be packed — only if the policy says so."""

    name: str                      # file name inside evidence/, e.g. "frame-001.jpg"
    data: bytes = b""
    media_type: str = "application/octet-stream"
    trace_id: str | None = None
    source_ts: float | None = None
    goal_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name or self.name.startswith("."):
            raise ValueError(f"evidence name must be a bare file name, got {self.name!r}")

    @property
    def sha256(self) -> str:
        return _sha256(self.data)

    @property
    def bytes_len(self) -> int:
        return len(self.data)

    def to_entry(self) -> dict[str, Any]:
        return {
            "name": self.name, "media_type": self.media_type, "bytes": self.bytes_len,
            "sha256": self.sha256, "trace_id": self.trace_id, "source_ts": self.source_ts,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_path(cls, path: Any, **kw: Any) -> "EvidenceItem":
        p = pathlib.Path(path)
        return cls(name=kw.pop("name", p.name), data=p.read_bytes(), **kw)


@dataclass(frozen=True)
class MediaReference:
    """Media that stays OUTSIDE the artifact — a uri plus a content hash, and nothing more.

    This is what a ``local`` bundle carries, and it is exactly why a local bundle cannot claim to
    replay perception: the bytes are somewhere else, possibly mutable, possibly gone.
    """

    uri: str
    sha256: str | None = None
    bytes_len: int | None = None
    media_type: str | None = None
    trace_id: str | None = None
    source_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri, "sha256": self.sha256, "bytes": self.bytes_len,
            "media_type": self.media_type, "trace_id": self.trace_id, "source_ts": self.source_ts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MediaReference":
        return cls(
            uri=str(payload["uri"]), sha256=payload.get("sha256"),
            bytes_len=payload.get("bytes"), media_type=payload.get("media_type"),
            trace_id=payload.get("trace_id"), source_ts=payload.get("source_ts"),
        )


@dataclass(frozen=True)
class TraceBundle:
    """A trace artifact plus its mode, its hashes and its expiry — writable as a dir or one file."""

    manifest: RunTraceManifest
    records: tuple[TraceRecord, ...]
    summary: TraceSummary
    policy: BundlePolicy = field(default_factory=BundlePolicy)
    evidence: tuple[EvidenceItem, ...] = ()
    media_refs: tuple[MediaReference, ...] = ()
    created_wall_ts: float = 0.0
    expires_wall_ts: float | None = None
    media_omitted: int = 0
    #: Why media was omitted, per reason (``policy_opt_out`` / ``max_media_files`` /
    #: ``max_media_bytes``) — an omission is never a silent one.
    media_omitted_reasons: dict[str, int] = field(default_factory=dict)
    note: str = ""

    # -- construction -----------------------------------------------------
    @property
    def mode(self) -> BundleMode:
        """``portable`` iff media bytes are actually inside this bundle."""
        return "portable" if (self.policy.include_media and self.evidence) else "local"

    @property
    def self_contained(self) -> bool:
        """True iff nothing outside this artifact is needed for a COGNITION replay.

        Note the scope: a cognition replay needs only the recorded verdicts, so a ``local`` bundle
        is self-contained for it. Media reevaluation needs the pixels AND a live model, and is not
        deterministic in either mode — see the module docstring.
        """
        return True

    @classmethod
    def from_recorder(
        cls,
        recorder: TraceRecorder,
        *,
        policy: BundlePolicy | None = None,
        evidence: Sequence[EvidenceItem] = (),
        media_refs: Sequence[MediaReference] = (),
        observed_unit: str = "frames",
        wall_clock: Callable[[], float] = time.time,
        note: str = "",
    ) -> "TraceBundle":
        return cls.build(
            recorder.manifest, recorder.records, recorder.summary(observed_unit=observed_unit),
            policy=policy, evidence=evidence, media_refs=media_refs, wall_clock=wall_clock, note=note,
        )

    @classmethod
    def build(
        cls,
        manifest: RunTraceManifest,
        records: Sequence[TraceRecord],
        summary: TraceSummary,
        *,
        policy: BundlePolicy | None = None,
        evidence: Sequence[EvidenceItem] = (),
        media_refs: Sequence[MediaReference] = (),
        wall_clock: Callable[[], float] = time.time,
        note: str = "",
    ) -> "TraceBundle":
        """Apply the policy to the offered media and stamp retention. THIS is the opt-in choke point.

        Every piece of evidence is either accepted (policy allows media and the bounds hold) or
        dropped WITH a counted reason. Nothing is packed by default.
        """
        pol = policy if policy is not None else BundlePolicy()
        kept: list[EvidenceItem] = []
        dropped: dict[str, int] = {}
        total = 0
        for item in evidence:
            reason: str | None = None
            if not pol.include_media:
                reason = "policy_opt_out"
            elif len(kept) >= pol.max_media_files:
                reason = "max_media_files"
            elif total + item.bytes_len > pol.max_media_bytes:
                reason = "max_media_bytes"
            if reason is not None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            kept.append(item)
            total += item.bytes_len

        created = float(wall_clock())
        expires = None if pol.retention_days is None else created + pol.retention_days * 86400.0
        cleaned = tuple(_redact(r, pol.redact_keys) for r in records)
        return cls(
            manifest=manifest, records=cleaned, summary=summary, policy=pol,
            evidence=tuple(kept), media_refs=tuple(media_refs),
            created_wall_ts=created, expires_wall_ts=expires,
            media_omitted=sum(dropped.values()), media_omitted_reasons=dropped, note=note,
        )

    # -- retention --------------------------------------------------------
    def expired(self, *, now: float | None = None) -> bool:
        """Has this bundle passed its retention stamp? ``None`` expiry = kept forever (explicit)."""
        if self.expires_wall_ts is None:
            return False
        return float(now if now is not None else time.time()) > self.expires_wall_ts

    # -- serialization ----------------------------------------------------
    def _events_bytes(self) -> bytes:
        lines = [canonical_json(r.to_dict()) for r in self.records]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    def _summary_bytes(self) -> bytes:
        return (canonical_json(self.summary.to_dict()) + "\n").encode("utf-8")

    def _files(self) -> list[tuple[str, bytes]]:
        """Every file EXCEPT the manifest (which hashes them), in written order."""
        out: list[tuple[str, bytes]] = [
            (EVENTS_NAME, self._events_bytes()),
            (SUMMARY_NAME, self._summary_bytes()),
        ]
        out.extend((f"{EVIDENCE_DIR}/{e.name}", e.data) for e in self.evidence)
        return out

    def manifest_dict(self) -> dict[str, Any]:
        """The bundle header: run manifest + mode + per-file hashes + retention + replay contract."""
        media_types = {f"{EVIDENCE_DIR}/{e.name}": e.media_type for e in self.evidence}
        entries = [
            {"path": p, "bytes": len(b), "sha256": _sha256(b), "media_type": media_types.get(p)}
            for p, b in self._files()
        ]
        return {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "mode": self.mode,
            "media_included": bool(self.evidence),
            "media_omitted": self.media_omitted,
            "media_omitted_reasons": dict(self.media_omitted_reasons),
            "created_wall_ts": self.created_wall_ts,
            "expires_wall_ts": self.expires_wall_ts,
            "retention_days": self.policy.retention_days,
            "policy": self.policy.to_dict(),
            "note": self.note,
            # The honest contract, written INTO the artifact so it travels with it.
            "replay": {
                "cognition_replay": True,
                "cognition_replay_deterministic": True,
                "media_reevaluation": False,
                "statement": (
                    "This bundle supports COGNITION REPLAY: recorded normalized verdicts + timestamps "
                    "re-fed through the Gate and incident logic, deterministically, with no model "
                    "call. It does NOT reproduce perception — re-judging media against a live model "
                    "(MEDIA REEVALUATION) is not deterministic and is out of scope here."
                ),
            },
            "manifest": self.manifest.to_dict(),
            "files": entries,
            "media_refs": [m.to_dict() for m in self.media_refs],
            "payload_sha256": _sha256(canonical_json(entries).encode("utf-8")),
        }

    def write(self, path: Any) -> pathlib.Path:
        """Write the unpacked ``.ptrace`` directory. ``path`` may omit the suffix."""
        root = _with_suffix(pathlib.Path(path))
        root.mkdir(parents=True, exist_ok=True)
        if self.evidence:
            (root / EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)
        for rel, data in self._files():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (root / MANIFEST_NAME).write_text(
            json.dumps(self.manifest_dict(), indent=2, default=str), encoding="utf-8"
        )
        return root

    def write_archive(self, path: Any) -> pathlib.Path:
        """Write the SINGLE-FILE form (``run-123.ptrace.zip``) — the thing a client attaches."""
        p = pathlib.Path(path)
        if p.suffix != ".zip":
            p = _with_suffix(p).with_suffix(BUNDLE_SUFFIX + ".zip")
        p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_NAME, json.dumps(self.manifest_dict(), indent=2, default=str))
            for rel, data in self._files():
                zf.writestr(rel, data)
        return p

    # -- reading + verification -------------------------------------------
    @classmethod
    def read(cls, path: Any) -> "TraceBundle":
        """Read a bundle back from either form (directory or ``.zip``). Raises :class:`BundleError`."""
        p = pathlib.Path(path)
        if p.is_dir():
            reader: Callable[[str], bytes] = lambda rel: (p / rel).read_bytes()  # noqa: E731
            names = {str(q.relative_to(p)).replace("\\", "/") for q in p.rglob("*") if q.is_file()}
        elif zipfile.is_zipfile(p):
            zf = zipfile.ZipFile(p)
            names = set(zf.namelist())
            reader = zf.read
        else:
            raise BundleError(f"not a .ptrace directory or archive: {p}")
        if MANIFEST_NAME not in names:
            raise BundleError(f"bundle has no {MANIFEST_NAME}: {p}")
        header = json.loads(reader(MANIFEST_NAME).decode("utf-8"))
        manifest = RunTraceManifest.from_dict(header["manifest"])
        records = [
            TraceRecord.from_dict(json.loads(line))
            for line in reader(EVENTS_NAME).decode("utf-8").splitlines() if line.strip()
        ] if EVENTS_NAME in names else []
        summary = (
            _summary_from_dict(json.loads(reader(SUMMARY_NAME).decode("utf-8")), manifest, records)
            if SUMMARY_NAME in names
            else TraceSummary.derive(manifest, records)
        )
        evidence = tuple(
            EvidenceItem(
                name=entry["path"].split("/", 1)[1],
                data=reader(entry["path"]),
                media_type=str(entry.get("media_type") or "application/octet-stream"),
            )
            for entry in header.get("files") or []
            if entry["path"].startswith(EVIDENCE_DIR + "/") and entry["path"] in names
        )
        return cls(
            manifest=manifest, records=tuple(records), summary=summary,
            policy=BundlePolicy.from_dict(header.get("policy") or {}),
            evidence=evidence,
            media_refs=tuple(MediaReference.from_dict(m) for m in header.get("media_refs") or ()),
            created_wall_ts=float(header.get("created_wall_ts", 0.0)),
            expires_wall_ts=header.get("expires_wall_ts"),
            media_omitted=int(header.get("media_omitted", 0)),
            media_omitted_reasons=dict(header.get("media_omitted_reasons") or {}),
            note=str(header.get("note", "")),
        )

    @classmethod
    def verify_path(cls, path: Any, *, now: float | None = None) -> list[str]:
        """Read a bundle and check every file against the hashes its manifest declares.

        Returns a list of PROBLEMS (empty = intact). Checked against the manifest as written, so a
        truncated attachment or an edited event line is caught rather than replayed.
        """
        p = pathlib.Path(path)
        if p.is_dir():
            reader: Callable[[str], bytes] = lambda rel: (p / rel).read_bytes()  # noqa: E731
            names = {str(q.relative_to(p)).replace("\\", "/") for q in p.rglob("*") if q.is_file()}
        elif zipfile.is_zipfile(p):
            zf = zipfile.ZipFile(p)
            names = set(zf.namelist())
            reader = zf.read
        else:
            return [f"not a .ptrace directory or archive: {p}"]
        problems: list[str] = []
        if MANIFEST_NAME not in names:
            return [f"missing {MANIFEST_NAME}"]
        header = json.loads(reader(MANIFEST_NAME).decode("utf-8"))
        entries = header.get("files") or []
        for entry in entries:
            rel = entry["path"]
            if rel not in names:
                problems.append(f"missing file: {rel}")
                continue
            data = reader(rel)
            if _sha256(data) != entry.get("sha256"):
                problems.append(f"content hash mismatch: {rel}")
            elif len(data) != int(entry.get("bytes", -1)):
                problems.append(f"size mismatch: {rel}")
        payload = _sha256(canonical_json(entries).encode("utf-8"))
        if payload != header.get("payload_sha256"):
            problems.append("payload_sha256 mismatch (the file list itself was altered)")
        exp = header.get("expires_wall_ts")
        if exp is not None and float(now if now is not None else time.time()) > float(exp):
            problems.append("bundle is past its retention/expiry stamp")
        return problems

    # -- the point of the whole card --------------------------------------
    def replay(self, **kw: Any) -> Any:
        """Run the deterministic COGNITION replay over this bundle's records.

        Keyword arguments are forwarded to :func:`percept.replay.replay_cognition`. No model is
        called and no media is read: same bundle in, byte-identical decisions out.
        """
        from .replay import replay_cognition  # noqa: PLC0415 - avoid an import cycle at module load

        return replay_cognition(self.manifest, list(self.records), **kw)


def _with_suffix(p: pathlib.Path) -> pathlib.Path:
    return p if p.suffix == BUNDLE_SUFFIX else p.with_name(p.name + BUNDLE_SUFFIX)


def _redact(rec: TraceRecord, keys: Sequence[str]) -> TraceRecord:
    if not keys:
        return rec
    drop = set(keys)
    if not (drop & set(rec.data)):
        return rec
    return replace(rec, data={k: v for k, v in rec.data.items() if k not in drop})


def _summary_from_dict(
    payload: Mapping[str, Any], manifest: RunTraceManifest, records: Sequence[TraceRecord]
) -> TraceSummary:
    """Rebuild a :class:`~percept.tracing.TraceSummary` from its own ``to_dict`` form.

    Reuses :func:`percept.tracing.load_trace` so there is ONE deserializer for the funnel — the
    numbers a bundle reports are the numbers ``tracing`` derived, never a second interpretation.
    """
    from .tracing import load_trace  # noqa: PLC0415

    _, _, summary = load_trace({
        "manifest": manifest.to_dict(),
        "records": [r.to_dict() for r in records],
        "summary": dict(payload),
    })
    return summary


def export_bug_report(
    recorder: TraceRecorder,
    path: Any,
    *,
    include_media: bool = False,
    evidence: Sequence[EvidenceItem] = (),
    media_refs: Sequence[MediaReference] = (),
    retention_days: float | None = DEFAULT_RETENTION_DAYS,
    redact_keys: Iterable[str] = (),
    note: str = "",
    archive: bool = True,
    observed_unit: str = "frames",
    wall_clock: Callable[[], float] = time.time,
) -> pathlib.Path:
    """The client-facing path: one call, one file, attach it to the ticket.

    ``include_media`` defaults to ``False`` here too — the convenience wrapper does not get to be
    the place the safe default quietly flips. With it off, any ``evidence`` offered is dropped and
    the count is written into the bundle manifest.

    Returns the path written (``run-123.ptrace.zip`` by default, or the unpacked directory with
    ``archive=False``).
    """
    policy = BundlePolicy(
        include_media=bool(include_media),
        retention_days=retention_days,
        redact_keys=tuple(redact_keys),
    )
    bundle = TraceBundle.from_recorder(
        recorder, policy=policy, evidence=evidence, media_refs=media_refs,
        observed_unit=observed_unit, wall_clock=wall_clock, note=note,
    )
    return bundle.write_archive(path) if archive else bundle.write(path)


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_SUFFIX",
    "DEFAULT_RETENTION_DAYS",
    "EVENTS_NAME",
    "EVIDENCE_DIR",
    "MANIFEST_NAME",
    "SUMMARY_NAME",
    "BundleError",
    "BundleMode",
    "BundlePolicy",
    "EvidenceItem",
    "MediaReference",
    "TraceBundle",
    "export_bug_report",
]
