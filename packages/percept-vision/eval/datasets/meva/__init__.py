"""MEVA (Multiview Extended Video with Activities) -> ``facility_observation`` fixtures.

DEV TOOLING — not shipped in the wheel (see :mod:`eval.datasets`). Reads MEVA KPF activity
annotations (schema reference: ``gitlab.kitware.com/meva/meva-data-repo``) and emits Percept
evaluation fixtures for the shipped ``facility_observation@1`` domain pack.

Public surface:

* :mod:`eval.datasets.meva.kpf` — the KPF activity reader (``MevaActivity`` / ``MevaClip``).
* :mod:`eval.datasets.meva.pack_ext` — the MEVA-activity -> ``facility_observation`` template map
  and the self-contained (un-registered) MEVA-tuned pack scaffolding.
* :mod:`eval.datasets.meva.transform` — the annotation->fixture transform, the SCENE-level 50/50
  held-out split, the acceptance bars, and the pre-registration writer.

CRITICAL — the split is by SCENE (camera-location recording session), never by clip: MEVA is
multiview, so the same physical event is annotated once per camera. A clip-level split would put
one camera's view of an event in ``train`` and another camera's view of the SAME event in ``test``,
leaking the test set. See :func:`eval.datasets.meva.transform.scene_split`.
"""
