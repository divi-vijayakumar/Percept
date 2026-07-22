"""``percept.build_info()`` — the public accessor for wheel provenance.

Every wheel/sdist built through this package's packaging config stamps ``percept/_build_info.py``
with the package version, the git commit it was built from, the frozen wire
``CONTRACT_VERSION`` it shipped with, and a UTC build timestamp (see ``_build_backend.py`` at the
repo root of this package). A bare source checkout that never went through that build step keeps
the checked-in fallback stub, which reports ``"unreleased"`` for every field instead of guessing.

A returned trace (``.ptrace``) carries this same information via
:meth:`percept.tracing.RunTraceManifest.create`, so a trace can be tied back to the exact build
that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildInfo:
    """Provenance of the installed/imported ``percept`` package."""

    package_version: str
    git_commit: str
    contract_version: str
    build_timestamp: str

    def to_dict(self) -> dict[str, str]:
        return {
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "contract_version": self.contract_version,
            "build_timestamp": self.build_timestamp,
        }


def build_info() -> BuildInfo:
    """Return this build's provenance: version, git commit, CONTRACT_VERSION, build timestamp.

    Falls back to ``"unreleased"`` for every field in a source checkout that hasn't gone through
    the packaging build step — never raises.
    """
    try:
        from . import _build_info as _stamp  # noqa: PLC0415
    except Exception:  # pragma: no cover - defensive
        return BuildInfo(
            package_version="unreleased",
            git_commit="unreleased",
            contract_version="unreleased",
            build_timestamp="unreleased",
        )
    return BuildInfo(
        package_version=getattr(_stamp, "PACKAGE_VERSION", "unreleased"),
        git_commit=getattr(_stamp, "GIT_COMMIT", "unreleased"),
        contract_version=getattr(_stamp, "CONTRACT_VERSION", "unreleased"),
        build_timestamp=getattr(_stamp, "BUILD_TIMESTAMP", "unreleased"),
    )
