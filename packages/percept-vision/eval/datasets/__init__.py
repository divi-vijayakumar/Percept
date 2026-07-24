"""Dataset -> Percept-fixture transforms (DEV TOOLING — NOT shipped in the wheel).

This package lives under the top-level ``eval/`` dev tree, which ``pyproject.toml`` deliberately
keeps OUT of the wheel (``[tool.setuptools.packages.find] include = ["percept*"]``). Each
sub-package reads a public research dataset's native annotation schema and emits Percept
evaluation :class:`percept.eval.differential.Fixture` objects (and their on-disk form loadable by
:func:`percept.eval.differential.load_fixtures`), plus a pre-registered held-out split and
client-supplied bars — so a :class:`~percept.domain_pack.DomainPack` can be qualified against a
real labeled corpus with ``percept eval`` / ``percept derive-profile`` instead of hand-written
ground truth.

No dataset is vendored here — only a SMALL synthetic sample per dataset, representative of the real
schema, so each transform is unit-testable with no download and no network.

Sub-packages:

- :mod:`eval.datasets.meva` — MEVA (Multiview Extended Video with Activities) -> facility_observation
  (board card DS1).
- :mod:`eval.datasets.ami` — the AMI Meeting Corpus (NXT format) -> a recorded-meeting / conversation
  pack (board card DS2).
- :mod:`eval.datasets.guiworld` — GUI-World -> a screen_document pack (board card DS3).
"""
