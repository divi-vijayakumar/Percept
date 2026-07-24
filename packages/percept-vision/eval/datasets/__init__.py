"""Dataset -> Percept fixture transforms (dev tooling; NOT wheel-shipped).

Each sub-package reads a public dataset's own annotation format and emits Percept
:class:`percept.eval.differential.Fixture` objects plus a pre-registered held-out split, so a
labeled corpus can be scored through the SAME loaders (``load_fixtures`` / ``load_held_out_split`` /
``load_bars``) ``percept eval`` and ``percept derive-profile`` already use.
"""
