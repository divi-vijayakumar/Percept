"""Replay-and-align — the scoring core (spec §9, §10.2).

Replay a labeled clip through the system; record every fire ``(goal, ts, conf, cost, tokens)``;
align each ground-truth **rising edge** to a fire within ``[edge, edge + omega]`` -> TP / FN / FP.
*A right answer at the wrong time scores wrong* — a fire before the edge or after ``edge+omega``
does not satisfy that edge.

This module is **pure** (no models, no frames): it operates on synthetic ``Fire`` / ``Edge``
lists and is the L1-testable heart of the scorer. The model-driven replay loop lives in
``eval.run`` (L2).

Metrics emitted:

* **reaction latency** as **PAUC@omega** — area under the cumulative "fraction of edges hit by
  latency ``<= x``" curve, normalized to ``[0, 1]``; 1.0 = every edge hit instantly, 0.0 = none
  hit within omega. Higher is better (promptness).
* **precision** = TP / (TP + FP) = ``1 - false-fire-rate``.
* **recall** = TP / (TP + FN).
* **count off-by-one** = ``|#fires_aligned - #edges|`` style count error per goal (the RepCount axis).
* **cost** (total + per-fire) and **context tokens** — summed off the recorded fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fire:
    """One recorded fire from a replay. ``cost`` is in cents; ``tokens`` is context tokens."""

    goal_id: str
    ts: float
    conf: float = 1.0
    cost: float = 0.0
    tokens: int = 0


@dataclass(frozen=True)
class Edge:
    """One ground-truth rising edge: the goal's condition became true at ``t``."""

    goal_id: str
    t: float


@dataclass
class AlignmentResult:
    """The per-clip alignment of fires to edges within the omega window."""

    omega: float
    tp: int = 0  # edges hit by a fire in [edge, edge+omega]
    fn: int = 0  # edges with no qualifying fire (misses)
    fp: int = 0  # fires that aligned to no edge (false fires)
    latencies: list[float] = field(default_factory=list)  # per-TP reaction latency (fire - edge)
    matched_fire_idx: list[int] = field(default_factory=list)
    matched_edge_idx: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0  # no fires -> vacuously precise

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0  # no edges -> vacuously complete

    @property
    def false_fire(self) -> int:
        return self.fp


def align(fires: list[Fire], edges: list[Edge], omega: float) -> AlignmentResult:
    """Greedily align fires to rising edges within ``[edge, edge+omega]`` (earliest-edge first).

    Rules (the timing discipline):

    * a fire satisfies an edge only when ``edge.t <= fire.ts <= edge.t + omega`` **and** the goal
      ids match — a right answer at the wrong time scores wrong;
    * each edge is satisfied by at most one fire (the earliest qualifying one) -> the rest of the
      fires in its window, and any fire outside every window, are **false fires** (FP);
    * an edge with no qualifying fire is a **miss** (FN).

    Greedy by edge time then fire time is deterministic and optimal here because windows are
    forward-only and equal-width.
    """
    result = AlignmentResult(omega=omega)
    used_fires: set[int] = set()

    # sort indices for determinism; keep original indices for reporting.
    edge_order = sorted(range(len(edges)), key=lambda i: (edges[i].t, edges[i].goal_id))
    fire_order = sorted(range(len(fires)), key=lambda i: (fires[i].ts, fires[i].goal_id))

    for ei in edge_order:
        edge = edges[ei]
        chosen: int | None = None
        for fi in fire_order:
            if fi in used_fires:
                continue
            fire = fires[fi]
            if fire.goal_id != edge.goal_id:
                continue
            if edge.t <= fire.ts <= edge.t + omega:
                chosen = fi
                break  # fire_order is time-sorted -> first qualifying fire is the earliest
        if chosen is None:
            result.fn += 1
        else:
            used_fires.add(chosen)
            result.tp += 1
            result.latencies.append(fires[chosen].ts - edge.t)
            result.matched_fire_idx.append(chosen)
            result.matched_edge_idx.append(ei)

    result.fp = len(fires) - len(used_fires)  # every unmatched fire is a false fire
    return result


def pauc(latencies: list[float], omega: float, n_edges: int, *, steps: int = 64) -> float:
    """Promptness area-under-curve at omega, normalized to ``[0, 1]`` (higher = faster).

    The curve is ``f(x) = (# edges hit with latency <= x) / n_edges`` for ``x in [0, omega]``;
    PAUC is its mean value over the window. An edge with no qualifying fire (a miss) never enters
    the numerator, so recall is folded in: missing an edge drags PAUC down just like a slow hit.

    ``n_edges == 0`` -> 1.0 (vacuously perfect; nothing to be late for).
    """
    if n_edges <= 0:
        return 1.0
    if omega <= 0.0:
        # degenerate window: only instant (latency <= 0) hits count.
        hit = sum(1 for x in latencies if x <= 0.0)
        return hit / n_edges
    valid = [x for x in latencies if 0.0 <= x <= omega]
    total = 0.0
    for s in range(steps):
        x = omega * (s + 1) / steps  # right-edge sample of each sub-interval
        hit = sum(1 for lx in valid if lx <= x)
        total += (hit / n_edges) / steps
    return total


@dataclass
class ReplayMetrics:
    """The full scorecard for a replay (one or many clips), per spec §10.2."""

    omega: float
    tp: int
    fn: int
    fp: int
    precision: float
    recall: float
    false_fire: int
    pauc: float
    count_off_by_one: int
    total_cost: float
    cost_per_fire: float
    total_tokens: int
    n_fires: int
    n_edges: int

    def as_dict(self) -> dict:
        return {
            "omega": self.omega,
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "precision": self.precision,
            "recall": self.recall,
            "false_fire": self.false_fire,
            "pauc": self.pauc,
            "count_off_by_one": self.count_off_by_one,
            "total_cost": self.total_cost,
            "cost_per_fire": self.cost_per_fire,
            "total_tokens": self.total_tokens,
            "n_fires": self.n_fires,
            "n_edges": self.n_edges,
        }


def _count_off_by_one(fires: list[Fire], edges: list[Edge]) -> int:
    """Per-goal absolute count error summed across goals (the RepCount off-by-one axis).

    Counts distinct fires vs distinct edges per goal so a double-fire or a missed edge each shows
    up as a count error, independent of timing alignment.
    """
    goals = {f.goal_id for f in fires} | {e.goal_id for e in edges}
    total = 0
    for g in goals:
        n_fire = sum(1 for f in fires if f.goal_id == g)
        n_edge = sum(1 for e in edges if e.goal_id == g)
        total += abs(n_fire - n_edge)
    return total


def score(fires: list[Fire], edges: list[Edge], omega: float) -> ReplayMetrics:
    """Align ``fires`` to ``edges`` at ``omega`` and roll up the full scorecard."""
    a = align(fires, edges, omega)
    total_cost = sum(f.cost for f in fires)
    total_tokens = sum(f.tokens for f in fires)
    n_fires = len(fires)
    return ReplayMetrics(
        omega=omega,
        tp=a.tp,
        fn=a.fn,
        fp=a.fp,
        precision=a.precision,
        recall=a.recall,
        false_fire=a.false_fire,
        pauc=pauc(a.latencies, omega, len(edges)),
        count_off_by_one=_count_off_by_one(fires, edges),
        total_cost=total_cost,
        cost_per_fire=(total_cost / n_fires) if n_fires else 0.0,
        total_tokens=total_tokens,
        n_fires=n_fires,
        n_edges=len(edges),
    )


__all__ = [
    "Fire",
    "Edge",
    "AlignmentResult",
    "ReplayMetrics",
    "align",
    "pauc",
    "score",
]
