"""Dataset -> Percept-fixture transforms (dev tooling; NOT shipped in the wheel).

Each sub-package turns a public research corpus's native annotation layer into the Percept
evaluation format (``percept.eval.differential.Fixture`` / ``percept.harness.golden_eval``
``GroundTruthEpisode`` + a pre-registered held-out split + client-supplied bars), so a source
class can be benchmarked WITHOUT baking the corpus into the package.

Sub-packages:

- :mod:`eval.datasets.ami` — the AMI Meeting Corpus (NXT format) -> a recorded-meeting /
  conversation pack (board card DS2).
"""
