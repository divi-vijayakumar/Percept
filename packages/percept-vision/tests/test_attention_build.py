"""The pure group builder is what the live Monitor uses: same groups, same labels."""

from __future__ import annotations

import percept
from percept import attention
from percept import monitor as rt
from percept.contracts import Condition
from percept.delivery import MemoryStore
from percept.judges import FakeJudge
from percept.triggers import every, on_change


class Custom:
    """An unhashable custom trigger: grouped by identity."""

    __hash__ = None  # type: ignore[assignment]

    def step(self, frame, conditions, state, now):
        return (), state


def conditions() -> list[Condition]:
    custom = Custom()
    return [
        Condition("door", "the door is open"),                                       # default state
        Condition("fall", "a person falls", kind="event"),                           # default event
        Condition("tick", "a light is on", trigger=every(1.0)),
        Condition("tock", "a fan is on", trigger=every(1.0)),                        # equal trigger shared
        Condition("combo", "a box is on the shelf", trigger=(on_change(0.3), every(1.0), every(1.0))),
        Condition("odd", "a cart is in the aisle", trigger=custom),
        Condition("odd2", "a cart is at the till", trigger=custom),
        Condition("quiet", "the alarm is on", trigger=every(7.0)),                   # paused
        Condition("calm", "a spill is on the floor", kind="event", trigger=on_change(0.4, baseline_s=3.0)),
    ]


def test_monitor_groups_and_labels_are_the_builders():
    m = percept.Monitor(FakeJudge({}), conditions(), store=MemoryStore())
    m.pause("quiet")
    groups, labels = attention.build(m._conds.values(), default_change=m._default_change,
                                     default_heartbeat=m._default_heartbeat, paused={"quiet"})
    assert m._groups == groups
    assert m._labels == labels
    by_label = {labels[k]: [c.id for c in cs] for k, _, cs in groups}
    assert by_label["every(1)"] == ["tick", "tock", "combo"]
    assert by_label["on_change(0.2)"] == ["door", "fall"]
    assert by_label["every(300)"] == ["door"]
    assert by_label["Custom"] == ["odd", "odd2"]
    assert by_label["on_change(0.4, baseline_s=3)"] == ["calm"]
    assert not any("quiet" in ids for ids in by_label.values())
    m.resume("quiet")
    groups, labels = attention.build(m._conds.values(), default_change=m._default_change,
                                     default_heartbeat=m._default_heartbeat)
    assert (m._groups, m._labels) == (groups, labels)
    assert "every(7)" in labels.values()


def test_the_look_priorities_are_the_monitors():
    assert attention.PRIORITY == {attention.EVENT: rt.EVENT, attention.STATE: rt.STATE,
                                  attention.HEARTBEAT: rt.HEARTBEAT}
    assert rt.VERIFY < rt.EVENT < rt.STATE < rt.HEARTBEAT < rt.RECHECK
