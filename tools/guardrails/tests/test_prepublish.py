"""The pre-publication scan finds each kind of leak, and stays quiet on ordinary code.

Fake secrets are assembled at run time, so this file itself holds none."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepublish_check as p  # noqa: E402


def findings(label: str, text: str, terms=()) -> list[str]:
    compiled = [(t, re.compile(r"(?i)(?<![\w-])" + re.escape(t) + r"(?![\w-])")) for t in terms]
    return p.scan_file(label, text.encode(), compiled)


def test_secrets_of_every_shape_are_found():
    samples = {
        "private key": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "AWS access key": "AKIA" + "ABCDEFGHIJKLMNOP",
        "Google API key": "AIza" + "x" * 35,
        "OpenAI or Anthropic key": "sk-" + "ant-" + "a1" * 12,
        "credentials in a URL": "rtsp://" + "admin:hunter2" + "@10.0.0.5/stream",
        "secret assignment": 'api_key = "' + "Z" * 24 + '"',
    }
    for name, text in samples.items():
        assert any(name in f for f in findings("src/app.py", text)), name


def test_home_paths_media_and_terms_are_found(tmp_path):
    assert findings("docs/x.md", "see /" + "Users/someone/notes")
    assert findings("docs/x.md", "see /" + "home/someone/notes")
    assert p.scan_file("data/clip.mp4", b"\0\0", [])
    assert p.scan_file("data/frames.npz", b"\0", [])
    terms = tmp_path / "terms"
    terms.write_text("# a comment\nProject Falcon\n\n")
    assert p.scan_file("README.md", b"Built for project falcon users.", p.read_terms(terms))
    assert not p.scan_file("README.md", b"Built for project falconry.", p.read_terms(terms))


def test_ordinary_code_and_allow_listed_fixtures_are_quiet():
    assert not findings("src/app.py", 'url = "rtsp://camera.local/stream"  # no credentials\nkey = os.environ["KEY"]')
    assert not findings("packages/percept-vision/tests/test_sources.py", "rtsp://" + "u:p" + "@cam/1")
    assert findings("packages/percept-vision/tests/test_other.py", "rtsp://" + "u:p" + "@cam/1")
