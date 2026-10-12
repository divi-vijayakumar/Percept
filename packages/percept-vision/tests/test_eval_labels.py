"""Labels for user clips and frozen splits: validation, hashing, split stability."""

from __future__ import annotations

import copy

import pytest

from percept.plans import LabelError, Labels, SplitManifest, load_labels, split
from percept.plans.labels import Occurrence, bucket, split_of


def doc(**over):
    d = {"conditions": [{"label": "person falls", "text": "a person falls", "kind": "event", "detector": "person"},
                        {"label": "door open", "text": "the door is open", "kind": "state"}],
         "clips": [{"file": "cams/a.mp4", "occurrences": [{"label": "person falls", "start": 12.0, "end": 14.5},
                                                           {"label": "door open", "start": 0, "end": 30}]},
                   {"file": "cams/b.mp4", "group": "lobby-1", "occurrences": []},
                   {"file": "cams/c.mp4", "group": "lobby-1"}]}
    d.update(over)
    return d


def test_a_valid_document_loads():
    labels = load_labels(doc())
    assert isinstance(labels, Labels)
    assert [c.label for c in labels.conditions] == ["person falls", "door open"]
    assert labels.conditions[0].detector == "person" and labels.conditions[1].kind == "state"
    a = labels.clip("a.mp4")
    assert a.group == "a"                        # defaults to the stem
    assert labels.clip("cams/b.mp4").group == "lobby-1"
    assert a.occurrences_of("person falls") == (Occurrence("person falls", 12.0, 14.5),)
    assert len(labels.labels_hash) == 64


@pytest.mark.parametrize("mutate, path", [
    (lambda d: d.update(extra=1), "extra"),
    (lambda d: d.pop("clips"), "clips"),
    (lambda d: d["conditions"][0].update(colour="red"), "conditions[0].colour"),
    (lambda d: d["conditions"][1].update(kind="moment"), "conditions[1].kind"),
    (lambda d: d["conditions"][1].update(label="person falls"), "conditions[1].label"),
    (lambda d: d["conditions"][0].update(text=""), "conditions[0].text"),
    (lambda d: d["clips"][0]["occurrences"][0].update(end=11.0), "clips[0].occurrences[0].end"),
    (lambda d: d["clips"][0]["occurrences"][0].update(start=-1), "clips[0].occurrences[0].start"),
    (lambda d: d["clips"][0]["occurrences"][0].update(start=True), "clips[0].occurrences[0].start"),
    (lambda d: d["clips"][0]["occurrences"][0].update(end=float("nan")), "clips[0].occurrences[0].end"),
    (lambda d: d["clips"][0]["occurrences"][0].update(label="cat"), "clips[0].occurrences[0].label"),
    (lambda d: d["clips"][0]["occurrences"][0].pop("end"), "clips[0].occurrences[0].end"),
    (lambda d: d["clips"][2].update(file="other/a.mp4"), "clips[2].file"),
    (lambda d: d["clips"][1].update(occurrences={}), "clips[1].occurrences"),
    (lambda d: d.update(clips="a.mp4"), "clips"),
])
def test_malformed_documents_name_the_field(mutate, path):
    d = copy.deepcopy(doc())
    mutate(d)
    with pytest.raises(LabelError) as e:
        load_labels(d)
    assert e.value.path == path
    assert isinstance(e.value, ValueError)


def test_overlapping_occurrences_of_a_label_merge():
    d = doc()
    d["clips"][0]["occurrences"] = [{"label": "person falls", "start": 5, "end": 8},
                                    {"label": "person falls", "start": 7, "end": 9},
                                    {"label": "person falls", "start": 9, "end": 10},
                                    {"label": "door open", "start": 6, "end": 7},
                                    {"label": "person falls", "start": 20, "end": 21}]
    occs = load_labels(d).clip("a.mp4").occurrences
    assert occs == (Occurrence("person falls", 5.0, 10.0), Occurrence("door open", 6.0, 7.0),
                    Occurrence("person falls", 20.0, 21.0))


def test_the_hash_ignores_key_order_list_order_and_directories():
    h = load_labels(doc()).labels_hash
    d = doc()
    d = {"clips": [dict(reversed(list(c.items()))) for c in reversed(d["clips"])],
         "conditions": [dict(reversed(list(c.items()))) for c in reversed(d["conditions"])]}
    for c in d["clips"]:
        c["file"] = "/elsewhere/checkout/" + c["file"].split("/")[-1]
    assert load_labels(d).labels_hash == h
    changed = doc()
    changed["clips"][0]["occurrences"][0]["end"] = 14.6
    assert load_labels(changed).labels_hash != h


def test_hypothesis_fuzz_only_label_errors():
    import hypothesis
    st = hypothesis.strategies
    scalar = st.one_of(st.none(), st.booleans(), st.integers(-10**30, 10**30), st.floats(allow_nan=True),
                       st.text(max_size=6), st.sampled_from(["event", "state", "person falls", "a.mp4"]))
    keys = st.sampled_from(["conditions", "clips", "label", "text", "kind", "detector", "file", "group",
                            "occurrences", "start", "end", "x"])
    tree = st.recursive(scalar, lambda kids: st.one_of(st.lists(kids, max_size=4),
                                                       st.dictionaries(keys, kids, max_size=5)), max_leaves=25)
    near = st.fixed_dictionaries({"conditions": st.one_of(tree, st.just(doc()["conditions"])),
                                  "clips": st.lists(st.fixed_dictionaries(
                                      {"file": tree, "occurrences": st.lists(st.dictionaries(keys, tree), max_size=3)}),
                                      max_size=3)})

    @hypothesis.settings(max_examples=300, deadline=None)
    @hypothesis.given(st.one_of(tree, near))
    def run(d):
        try:
            assert isinstance(load_labels(d), Labels)
        except LabelError:
            pass

    run()


# --------------------------------------------------------------------------- splits


def many(n: int, start: int = 0) -> dict:
    return {"conditions": [{"label": "x", "text": "x happens"}],
            "clips": [{"file": f"clip{i}.mp4", "group": f"rec{i // 3}",
                       "occurrences": [{"label": "x", "start": 1, "end": 2}]} for i in range(start, start + n)]}


def test_split_is_by_group_bucket():
    labels = load_labels(many(60))
    m = split(labels, {f"clip{i}.mp4": 600.0 for i in range(60)})
    assert isinstance(m, SplitManifest)
    for s, files in m.clips.items():
        for f in files:
            assert split_of(labels.clip(f).group) == s
    assert sum(len(v) for v in m.clips.values()) == 60
    assert sum(m.occurrences[s]["x"] for s in m.clips) == 60
    assert sum(m.hours.values()) == pytest.approx(10.0)
    assert all(len(m.clips[s]) > 0 for s in ("dev", "validation", "test"))
    assert 0 <= bucket("rec0") < 100


def test_a_group_never_straddles_splits():
    labels = load_labels(many(90))
    m = split(labels)
    where = {}
    for s, files in m.clips.items():
        for f in files:
            where.setdefault(labels.clip(f).group, set()).add(s)
    assert all(len(v) == 1 for v in where.values())
    assert m.hours == {"dev": None, "validation": None, "test": None}


def test_adding_clips_never_moves_existing_ones():
    before = split(load_labels(many(30)))
    after = split(load_labels(many(60)))
    for s in before.clips:
        assert set(before.clips[s]) <= set(after.clips[s])
    assert before.manifest_hash != after.manifest_hash
    assert split(load_labels(many(30))).manifest_hash == before.manifest_hash


def test_split_checks_cutoffs_and_durations():
    labels = load_labels(many(3))
    with pytest.raises(ValueError):
        split(labels, cutoffs=(70, 40))
    with pytest.raises(LabelError):
        split(labels, {"clip0.mp4": 1.0})
    assert split(labels, cutoffs=(0, 0)).clips["test"] == ("clip0.mp4", "clip1.mp4", "clip2.mp4")
    assert split(labels, cutoffs=(100, 100)).clips["dev"] == ("clip0.mp4", "clip1.mp4", "clip2.mp4")


def test_file_names_differing_only_in_case_are_one_file():
    d = doc()
    d["clips"][1]["file"] = "other/A.MP4"
    with pytest.raises(LabelError, match="duplicate file"):
        load_labels(d)


def test_durations_with_a_duplicate_file_name_are_refused():
    with pytest.raises(LabelError, match="duplicate file"):
        split(load_labels(doc()), {"cams/a.mp4": 60, "other/a.mp4": 90, "cams/b.mp4": 60, "cams/c.mp4": 60})
