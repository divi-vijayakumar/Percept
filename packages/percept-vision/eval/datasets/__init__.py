"""Dataset -> Percept-fixture transforms (DEV TOOLING — NOT shipped in the wheel).

This package lives under the top-level ``eval/`` dev tree, which ``pyproject.toml`` deliberately
keeps OUT of the wheel (``[tool.setuptools.packages.find] include = ["percept*"]``). Each
sub-package here reads a public research dataset's native annotation schema and emits Percept
evaluation :class:`percept.eval.differential.Fixture` objects (and their on-disk form loadable by
:func:`percept.eval.differential.load_fixtures`) — so a client can qualify a
:class:`~percept.domain_pack.DomainPack` against a real labeled corpus with ``percept eval`` /
``percept derive-profile`` instead of hand-writing ground truth.

No dataset is vendored here — only a SMALL synthetic sample per dataset, representative of the real
schema, so the transform is unit-testable with no download and no network.
"""
