#!/usr/bin/env python3
"""Percept live-watch demo — thin shim to the packaged CLI (`percept-watch`).

After `pip install percept-vision[runtime,openrouter]` you can just run `percept-watch ...`.
This file lets you run it from a repo clone without installing: `python examples/hackathon_watch.py ...`.
See `docs/integration.md` and `examples/README.md`.
"""
from percept.watch_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
