"""Pre-publication scan: nothing private leaves in the repository or the wheel.

Scans every tracked file (``git ls-files``, as it is in the working tree) and every file of a
wheel freshly built from a clean copy of ``packages/percept-vision`` for:

* secrets: key-shaped strings, private keys and URLs that carry a user name and password;
* absolute home paths (``/Users/<name>``, ``/home/<name>``);
* the repository URL placeholder (``REPLACE`` + ``-ME``), which must be replaced by the public
  repository URL before a release;
* media and dataset files, and any tracked file over 1 MB;
* private terms (names that must not be published) listed, one per line, in the untracked file ``.prepublish-terms``
  (copy ``.prepublish-terms.example`` to start; ``#`` starts a comment; matching ignores case and
  respects word boundaries).

It also runs ``gitleaks`` over both trees when ``gitleaks`` is on ``PATH``. Test fixtures that
exercise credential scrubbing are allow-listed by path (``ALLOW``). Exits 1 on any finding, 2 when
it can't run (no term list, a failed build), 0 when clean.

    python tools/prepublish_check.py [--python PATH] [--terms FILE] [--no-wheel]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "percept-vision"
MAX_BYTES = 1_000_000

SECRETS = [
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("OpenAI or Anthropic key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("secret assignment", re.compile(
        r"(?i)\b[\w-]*(?:api[_-]?key|secret|token|passw(?:or)?d)[\w-]*[\"']?\s*[:=]\s*[\"'][A-Za-z0-9/+_.=-]{16,}[\"']")),
    ("credentials in a URL", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@'\"<>`]+:[^\s/@'\"<>`]+@")),
]
# Assembled at run time so this file does not match itself.
PLACEHOLDER = "REPLACE" + "-ME"
HOME = re.compile(r"(?:/Users/|/home/)[A-Za-z0-9_.-]+")
MEDIA = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".jpg", ".jpeg", ".png", ".gif", ".bmp",
         ".tif", ".tiff", ".webp", ".heic", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".npz", ".npy",
         ".parquet", ".arrow", ".lance", ".h5", ".hdf5", ".pt", ".pth", ".onnx", ".tflite", ".task",
         ".pkl", ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".db", ".sqlite", ".sqlite3"}

# Paths (prefixes of tracked paths) where a category is expected. Keep this list short.
ALLOW: dict[str, tuple[str, ...]] = {
    # tests that prove credentials are scrubbed from URIs, health details, traces and judge errors
    # (the scan's own test builds its fake secrets at run time and needs no entry)
    "secret": tuple(f"packages/percept-vision/tests/{name}.py" for name in
                    ("test_monitor", "test_plans", "test_sources", "test_trace_replay", "test_vendor_judges")),
    # the template for the local term list
    "term": (".prepublish-terms.example",),
}


def allowed(category: str, path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ALLOW.get(category, ()))


Terms = list[tuple[str, re.Pattern]]


def read_terms(path: Path) -> Terms:
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.split("#", 1)[0].strip()
        if term:
            terms.append((term, re.compile(r"(?i)(?<![\w-])" + re.escape(term) + r"(?![\w-])")))
    return terms


def scan_text(label: str, text: str, terms: Terms) -> list[str]:
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for name, pattern in SECRETS:
            if pattern.search(line) and not allowed("secret", label):
                found.append(f"{label}:{n}: {name}")
        match = HOME.search(line)
        if match and not allowed("home", label):
            found.append(f"{label}:{n}: absolute home path {match.group(0)!r}")
        if PLACEHOLDER in line:
            found.append(f"{label}:{n}: repository URL placeholder {PLACEHOLDER!r}: set the public repo URL")
        for term, pattern in terms:
            if pattern.search(line) and not allowed("term", label):
                found.append(f"{label}:{n}: internal term {term!r}")
    return found


def scan_file(label: str, data: bytes, terms: Terms) -> list[str]:
    found = []
    if Path(label).suffix.lower() in MEDIA and not allowed("media", label):
        found.append(f"{label}: media or dataset file")
    if len(data) > MAX_BYTES and not allowed("size", label):
        found.append(f"{label}: {len(data):,} bytes (over {MAX_BYTES:,})")
    if b"\0" not in data[:8192]:
        found += scan_text(label, data.decode("utf-8", "replace"), terms)
    return found


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def build_wheel(py: str, tmp: Path) -> Path | None:
    sys.path.insert(0, str(ROOT / "tools"))
    from validate import build_wheel as build   # the same clean-copy build `make validate` uses

    src = tmp / "percept-vision"
    shutil.copytree(PACKAGE, src, ignore=shutil.ignore_patterns(
        "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache", "tests"))
    return build(py, src, tmp / "dist")


def build_sdist(tmp: Path) -> Path | None:
    """The source distribution the release uploads, built from the package's tracked files only
    (an untracked local file must not reach it)."""
    uv = shutil.which("uv")
    if uv is None:
        print("prepublish: building the sdist needs uv on PATH")
        return None
    src = tmp / "sdist-src"
    for name in tracked_files():
        if name.startswith("packages/percept-vision/") and (ROOT / name).is_file():
            dest = src / name[len("packages/percept-vision/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
    done = subprocess.run([uv, "build", "--sdist", "--out-dir", str(tmp / "sdist"), str(src)],
                          capture_output=True, text=True)
    found = sorted((tmp / "sdist").glob("percept_vision-*.tar.gz"))
    if done.returncode != 0 or len(found) != 1:
        print(done.stdout[-2000:], done.stderr[-2000:], file=sys.stderr)
        return None
    return found[0]


def gitleaks(directory: Path, label: str) -> list[str]:
    binary = shutil.which("gitleaks")
    if binary is None:
        return []
    done = subprocess.run([binary, "detect", "--no-git", "--source", str(directory), "--redact", "--no-banner",
                           "--exit-code", "1"], capture_output=True, text=True)
    if done.returncode == 0:
        return []
    return [f"gitleaks ({label}): {line}" for line in (done.stdout + done.stderr).splitlines()
            if line.strip() and ("Finding" in line or "File" in line or "error" in line.lower())] \
        or [f"gitleaks ({label}) exited {done.returncode}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter that builds the wheel")
    parser.add_argument("--terms", default=str(ROOT / ".prepublish-terms"), help="the internal term list")
    parser.add_argument("--no-wheel", action="store_true", help="scan the tracked tree only")
    args = parser.parse_args(argv)

    terms_path = Path(args.terms)
    if not terms_path.is_file():
        print(f"prepublish: no term list at {terms_path}; copy .prepublish-terms.example to "
              ".prepublish-terms and list the internal names to block (it is gitignored)")
        return 2
    terms = read_terms(terms_path)
    files = tracked_files()
    findings = [f"{name}: the term list must not be tracked" for name in files
                if Path(name).name == ".prepublish-terms"]
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        tree = tmp / "tree"
        for name in files:
            path = ROOT / name
            if not path.is_file():      # deleted in the working tree; the commit will drop it
                continue
            data = path.read_bytes()
            findings += scan_file(name, data, terms)
            if not any(allowed(c, name) for c in ALLOW):
                (tree / name).parent.mkdir(parents=True, exist_ok=True)
                (tree / name).write_bytes(data)
        findings += gitleaks(tree, "tracked tree")
        print(f"prepublish: scanned {len(files)} tracked files with {len(terms)} terms"
              + ("" if shutil.which("gitleaks") else "; gitleaks not installed, regex scan only"))

        if not args.no_wheel:
            wheel = build_wheel(str(Path(shutil.which(args.python) or args.python).absolute()), tmp)
            if wheel is None:
                print("prepublish: FAILED to build the wheel")
                return 2
            unpacked = tmp / "wheel"
            with zipfile.ZipFile(wheel) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                for name in names:
                    findings += scan_file(f"{wheel.name}!{name}", zf.read(name), terms)
                zf.extractall(unpacked)
            findings += gitleaks(unpacked, wheel.name)
            print(f"prepublish: scanned {len(names)} files in {wheel.name}")

            sdist = build_sdist(tmp)
            if sdist is None:
                print("prepublish: FAILED to build the sdist")
                return 2
            with tarfile.open(sdist) as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                for m in members:
                    # scanned under its repository path, so the allow-list applies as in the tree
                    repo_path = "packages/percept-vision/" + m.name.split("/", 1)[1]
                    data = tf.extractfile(m).read()
                    for finding in scan_file(repo_path, data, terms):
                        findings.append(f"{sdist.name}!{finding}")
            print(f"prepublish: scanned {len(members)} files in {sdist.name}")

    for finding in findings:
        print(f"  {finding}")
    print(f"prepublish: {len(findings)} finding(s)" if findings else "prepublish: clean")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
