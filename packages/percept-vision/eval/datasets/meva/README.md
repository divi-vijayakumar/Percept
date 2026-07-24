# MEVA → `facility_observation` fixtures (DEV TOOLING)

Transforms [MEVA](https://gitlab.kitware.com/meva/meva-data-repo) (Multiview Extended Video with
Activities) KPF **activity** annotations into Percept evaluation
[`Fixture`](../../../percept/eval/differential.py) objects for the shipped `facility_observation@1`
domain pack, with a pre-registered, scene-disjoint 50/50 held-out split and acceptance bars.

**Not shipped in the wheel.** This lives under the top-level `eval/` dev tree, which
`pyproject.toml` keeps out of the packaged `percept*` namespace. It has a dev/test dependency on
PyYAML (KPF is YAML); `percept` core stays zero-dependency.

**No dataset is vendored.** Only a small synthetic KPF sample (`testdata/`) and the fixtures it
produces (`fixtures/`) are committed — enough to unit-test the transform end-to-end with no
download and no network.

## MEVA KPF activity schema (reference)

One clip is described by `<clip>.activities.yml` / `.geom.yml` / `.types.yml`. This transform reads
only the **activities** file:

```yaml
- { meta: "..." }
- { act: { act2: { person_opens_facility_door: 1.0 }, id2: 6, src: truth,
           timespan: [ { tsr0: [ 300, 480 ] } ],
           actors: [ { id1: 12, timespan: [ { tsr0: [ 300, 480 ] } ] } ] } }
```

- `act2` maps the activity **type** → confidence (older DIVA-style `activity: <name>` is also
  accepted).
- `timespan` → `{ tsr0: [begin_frame, end_frame] }`; the activity **begin** is the fire anchor.
- Frames → seconds at a fixed clip rate (MEVA is 30 fps).

Clip filename `<date>.<start>.<end>.<site>.<CAMERA>.activities.yml` — everything before the camera
token is the **scene** (the multiview recording session); the camera token is the **view**.

## Activity → condition mapping

| MEVA activity type | `facility_observation` template | goal shape | sense |
|---|---|---|---|
| `person_opens_facility_door` | `door_left_open` | `episode` | absence (deadline) |
| `person_loiters`, `person_enters_scene_through_structure` | `zone_occupied` | `episode` | count |
| `person_abandons_package`, `abandonment`, `person_drops_off_package` | `package_appeared` | `point` | transition |

The activity **BEGIN** time (`begin_frame / fps`) becomes both `event_start_ts` and
`decision_eligible_ts` on the `GroundTruthEpisode` — the fire timestamp. Activity types not in the
map are dropped (not every MEVA activity is a facility-observation condition). See
[`pack_ext.py`](pack_ext.py) for the authoritative map.

## The split is by SCENE, never by clip

MEVA is **multiview**: the same physical event is annotated once per camera. A clip-level split
would put one camera's view of an event in `train` and another camera's view of the **same** event
in `test`, leaking the held-out set. The transform partitions on the **scene** (recording session),
so all camera views of an event stay on one side. `test_meva_transform.py` asserts no scene id
appears on both sides.

## Regenerate

```bash
cd packages/percept-vision
# regenerate the committed synthetic sample (testdata/ -> fixtures/)
python3 -m eval.datasets.meva.transform --sample
# or run on a real MEVA checkout
python3 -m eval.datasets.meva.transform --kpf-dir /path/to/meva/kpf --out ./meva-fixtures
```

Then run a differential / derivation with Percept's own harness, e.g.:

```bash
percept-eval --baseline base.yaml --candidate cand.yaml \
             --fixtures eval/datasets/meva/fixtures --bars eval/datasets/meva/fixtures/bars.json
```

## Tests

```bash
cd packages/percept-vision && python3 -m pytest eval/datasets/meva/test_meva_transform.py -q
```

## Registry-wiring follow-up (NOT applied here)

The transform targets templates the shipped `facility_observation@1` pack **already** defines, so it
runs with no registry change. `pack_ext.py` additionally ships an **optional**, self-contained
`facility_observation_meva@1` pack whose templates are named with MEVA's own activity vocabulary. It
is deliberately **not** auto-registered (this dev module must not mutate the shared registry that
sibling pack cards also touch). To make it resolvable, apply the one-liner:

```python
from eval.datasets.meva.pack_ext import register
register()  # adds facility_observation_meva@1 to DEFAULT_DOMAIN_PACK_REGISTRY
```

Do **not** edit `percept/packs_default.py` or `percept/source_profiles_default.py` for this — the
extension point is `register_domain_pack(...)` (or the `percept.packs` entry-point group), exactly
as a third-party pack would wire itself.
