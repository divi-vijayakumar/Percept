"""Fetch a documented MEVA subset: official activity annotations plus the matching videos.

MEVA (CC BY 4.0) publishes about 770 five-minute clips with evaluation-level annotations
(every instance of the 37 activities annotated) in the ``meva-data-repo`` on GitLab, and the
videos in the public ``mevadata-public-01`` S3 bucket. This takes ``--hours`` of them, spread
round-robin across sites with a fixed seed, so a subset is reproducible from its arguments::

    python -m benchmarks.fetch_meva --hours 5 --dry-run       # what would be fetched, and sizes
    python -m benchmarks.fetch_meva --hours 5                 # into eval/bench-staging/meva/

Files land in ``<out>/annotations/`` and ``<out>/videos/`` (git-ignored), with
``fetch-manifest.json`` recording every file's URL and sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

ANNOTATIONS = "https://gitlab.kitware.com/meva/meva-data-repo/-/raw/master/annotation/DIVA-phase-2/MEVA/kitware/"
VIDEOS = "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/"
CLIP = re.compile(r"^(\d{4}-\d\d-\d\d)\.(\d\d)-\d\d-\d\d\.(\d\d)-\d\d-\d\d\.(\w+)\.(G\d+)$")


def _curl(*args: str) -> str:
    """curl, not urllib: it uses the system trust store (proxies that re-sign TLS included)."""
    return subprocess.run(["curl", "-sfL", "--retry", "3", *args], capture_output=True, text=True, check=True).stdout


def select(stems: list[str], hours: float, seed: int) -> list[str]:
    """``hours`` of five-minute clips, round-robin over sites, shuffled within each with ``seed``."""
    by_site: dict[str, list[str]] = {}
    for s in sorted(stems):
        by_site.setdefault(CLIP.match(s).group(4), []).append(s)
    rng = random.Random(seed)
    for clips in by_site.values():
        rng.shuffle(clips)
    out, want = [], round(hours * 12)
    while len(out) < want and any(by_site.values()):
        for site in sorted(by_site):
            if by_site[site] and len(out) < want:
                out.append(by_site[site].pop())
    return out


def locate(stem: str) -> tuple[str, str, int] | None:
    """(annotation url, video url, video bytes); the hour folder is the end hour or the start hour."""
    date, start_h, end_h = CLIP.match(stem).group(1, 2, 3)
    for hour in dict.fromkeys((end_h, start_h)):
        video = f"{VIDEOS}{date}/{hour}/{stem}.r13.avi"
        try:
            head = _curl("-I", video)
        except subprocess.CalledProcessError:
            continue
        size = int(re.findall(r"(?im)^content-length:\s*(\d+)", head)[-1])
        return f"{ANNOTATIONS}{date}/{hour}/{stem}.activities.yml", video, size
    return None


def download(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    _curl("-o", str(part), url)
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    part.rename(dest)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.fetch_meva", description=__doc__.splitlines()[0])
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "eval" / "bench-staging" / "meva"))
    p.add_argument("--dry-run", action="store_true", help="list the selection and its size; download nothing")
    args = p.parse_args(argv)
    listing = _curl(ANNOTATIONS + "list-of-annotated-meva-clips.txt")
    stems = [s.strip() for s in listing.splitlines() if CLIP.match(s.strip())]
    chosen, out, record, total = select(stems, args.hours, args.seed), Path(args.out), [], 0
    for stem in chosen:
        found = locate(stem)
        if found is None:
            print(f"skip {stem}: video not in drops-123-r13")
            continue
        yml, video, size = found
        total += size
        print(f"{stem}  {size / 1e6:7.1f} MB")
        if not args.dry_run:
            entry = {"clip": stem, "annotations": yml, "video": video,
                     "annotations_sha256": download(yml, out / "annotations" / f"{stem}.activities.yml"),
                     "video_sha256": download(video, out / "videos" / f"{stem}.r13.avi")}
            record.append(entry)
    print(f"{len(chosen)} clips of {len(stems)}, about {len(chosen) / 12:.2f} h, {total / 1e9:.2f} GB of video")
    if not args.dry_run:
        (out / "fetch-manifest.json").write_text(json.dumps({"seed": args.seed, "hours": args.hours,
                                                              "files": record}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
