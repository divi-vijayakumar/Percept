"""Each guardrail must pass on a clean tree and fail on a planted violation."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guardrails as g  # noqa: E402


def make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    pkg = tmp_path / "packages" / "percept-vision" / "percept"
    for rel, body in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


CLEAN = {
    "__init__.py": """
        from percept.contracts import Decision
        from percept.decide import step
        from percept import monitor
        from percept.judges import gemini
        __all__ = ["Decision", "step"]
    """,
    "contracts.py": "from dataclasses import dataclass\n\n@dataclass\nclass Decision:\n    id: str\n",
    "decide.py": "from percept.contracts import Decision\nimport numpy as np\n\ndef step(): ...\n",
    "monitor.py": "import asyncio\nfrom percept.decide import step\n",
    "judges/__init__.py": "",
    "judges/base.py": "from ..contracts import Decision\n",
    "judges/gemini.py": "from .base import Decision\nimport httpx\n",
}


def test_clean_tree_passes(tmp_path):
    root = make_tree(tmp_path, CLEAN)
    assert g.check_exports(root) == []
    assert g.check_boundaries(root) == []
    assert g.check_reachability(root) == []


def test_real_repo_passes():
    root = Path(__file__).resolve().parents[3]
    assert g.check_exports(root) == []
    assert g.check_boundaries(root) == []
    assert g.check_reachability(root) == []


def test_too_many_exports_fails(tmp_path):
    names = ", ".join(f'"n{i}"' for i in range(31))
    root = make_tree(tmp_path, {**CLEAN, "__init__.py": f"__all__ = [{names}]\n"})
    assert g.check_exports(root)


def test_missing_all_fails(tmp_path):
    root = make_tree(tmp_path, {**CLEAN, "__init__.py": "x = 1\n"})
    assert g.check_exports(root)


@pytest.mark.parametrize("bad_import", [
    "import asyncio",
    "import sqlite3",
    "from concurrent.futures import ThreadPoolExecutor",
    "import cv2",
    "from google import genai",
    "from percept.monitor import x",
    "from . import monitor",
    "import yaml",
])
def test_core_io_or_shell_import_fails(tmp_path, bad_import):
    root = make_tree(tmp_path, {**CLEAN, "decide.py": f"{bad_import}\n\ndef step(): ...\n"})
    problems = g.check_boundaries(root)
    assert problems, bad_import
    assert "decide.py" in problems[0]


def test_relative_shell_import_from_nested_core_fails(tmp_path):
    root = make_tree(tmp_path, {**CLEAN, "judges/base.py": "from ..monitor import x\n"})
    assert g.check_boundaries(root)


@pytest.mark.parametrize("path", ["plans/__init__.py", "plans/later.py"])
@pytest.mark.parametrize("bad_import", ["import asyncio", "from percept.monitor import x", "from percept.judges import gemini"])
def test_plans_package_is_core(tmp_path, path, bad_import):
    files = {**CLEAN, "plans/__init__.py": "from percept.plans import later\n", "plans/later.py": "x = 1\n",
             "judges/_cascade.py": "import time\n"}
    assert g.check_boundaries(make_tree(tmp_path, files)) == []
    root = make_tree(tmp_path, {**files, path: f"{bad_import}\n"})
    problems = g.check_boundaries(root)
    assert problems and path.split("/")[1] in problems[0], bad_import


def test_plans_run_is_shell_and_core_may_not_import_it(tmp_path):
    files = {**CLEAN, "plans/__init__.py": "x = 1\n", "plans/labels.py": "x = 1\n",
             "plans/run.py": "import asyncio\nfrom percept.monitor import x\nfrom percept.plans import labels\n"}
    assert g.check_boundaries(make_tree(tmp_path, files)) == []
    for path in ("plans/__init__.py", "plans/labels.py", "decide.py"):
        root = make_tree(tmp_path, {**files, path: "from percept.plans import run\n"})
        problems = g.check_boundaries(root)
        assert problems and "run" in problems[0], path
        make_tree(tmp_path, files)


def test_plans_may_build_a_cascade_but_other_core_may_not(tmp_path):
    files = {**CLEAN, "plans/__init__.py": "from percept.judges._cascade import cascade\n",
             "judges/_cascade.py": "import time\n"}
    assert g.check_boundaries(make_tree(tmp_path, files)) == []
    root = make_tree(tmp_path, {**files, "decide.py": "from percept.judges._cascade import cascade\n"})
    assert g.check_boundaries(root)


def test_contracts_importing_percept_fails(tmp_path):
    root = make_tree(tmp_path, {**CLEAN, "contracts.py": "from percept.decide import step\n"})
    assert g.check_boundaries(root)


def test_shell_may_import_io(tmp_path):
    root = make_tree(tmp_path, {**CLEAN, "monitor.py": "import asyncio, sqlite3, threading\n"})
    assert g.check_boundaries(root) == []


def test_unreachable_module_fails(tmp_path):
    root = make_tree(tmp_path, {**CLEAN, "orphan.py": "x = 1\n"})
    problems = g.check_reachability(root)
    assert len(problems) == 1 and "orphan.py" in problems[0]


def test_size_over_threshold_exits_2(tmp_path):
    root = make_tree(tmp_path, {**CLEAN, "monitor.py": "x = 1\n" * 50})
    assert g.main(["size", "--root", str(root), "--max-lines", "10"]) == 2
    assert g.main(["size", "--root", str(root), "--max-lines", "100000"]) == 0
