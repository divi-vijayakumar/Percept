"""PerceptBridge — the ONE place this application touches Percept, via public seams only.

Wiring (all pull/callback, no SDK internals):

- **command tools** — the engine's model calls ``watch``/``stop_watching``/``whats_armed``; raw
  language routes to the **conductor** (``core.arm_from_language``), so the model's request stays
  ``LANGUAGE_AUTHORED`` and can never self-grant interrupt trust. Code-armed goals go through
  :meth:`arm` → ``session.arm`` (→ ``STRUCTURED_TRUSTED``, stamped by the SDK, never caller-claimed).
- **grounding is PULL-only** — ``on_goal_change``/incidents only *invalidate* the local snapshot
  (mark dirty); the engine's ``percept_state`` tool rebuilds it on the next pull. The bridge NEVER
  pushes state into the conversation: on Gemini Live every appended turn is answered OUT LOUD and a
  user-role state blob steers later noise-turns back to the armed goals (the diagnosed
  "keeps talking about the armed goals" failure). An engine adapter MAY push only if it has
  independently demonstrated truly silent context updates (none has).
- **transcripts** — free evidence: teed to ``core.note_mention``; nobody waits on it.
- **fires** — ``on_incident`` → ``attention_intent_for(incident, goal, policy)`` (the SDK's
  deterministic requested→effective clamp) → channel selection → the :class:`~floor.Floor` (speech)
  or the renderer directly (webhook/log — they never contend for the room's audio).
- **guard at attach** — ``guard_live_language_path`` refuses to wire a live language path behind a
  scripted brain (the silently-authors-nothing demo bug), at the one call an app makes.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Set

from percept import AttentionIntent, Goal, GoalOrigin
from percept.agent import guard_live_language_path
from percept.incidents import DEFAULT_ATTENTION_POLICY, attention_intent_for

#: The RICH command verbs a realtime engine exposes to its model. Each passes RAW language down so
#: Percept's **conductor** authors the full watchspec (the levers) — not a pre-structured goal.
#: (Template-owned copies: the session-era schema constants leave the SDK with the voice diet.)
WATCH_SCHEMA = {
    "description": "start watching the camera for a condition the user described",
    "properties": {"command": {"type": "string",
                               "description": "the user's watch request, verbatim, e.g. 'tell me the second my baby cries'"}},
    "required": ["command"],
}
STOP_WATCHING_SCHEMA = {
    "description": "stop one of the armed watches",
    "properties": {"what": {"type": "string", "description": "which watch to stop (a word or phrase from it)"}},
    "required": ["what"],
}
WHATS_ARMED_SCHEMA = {
    "description": "list the currently armed watches",
    "properties": {}, "required": [],
}
PERCEPT_STATE_SCHEMA = {
    "description": "current Percept brain state — armed watches + recent alert goal-ids. "
                   "Call when asked what you are watching or doing.",
    "properties": {}, "required": [],
}

_MAX_RECENT_ALERTS = 3


class PerceptBridge:
    """Glue between one Percept ``core``/``session`` pair and one :class:`~engine_adapter.VoiceEngine`."""

    def __init__(self, core: Any, session: Any, floor: Any, renderer: Any, *,
                 policy: Any = None, telemetry: Any = None) -> None:
        self._core = core
        self._session = session
        self._floor = floor
        self._renderer = renderer
        self._policy = policy if policy is not None else DEFAULT_ATTENTION_POLICY
        self._telemetry = telemetry
        self._engine: Any = None
        # Structured goals THIS app armed (session.arm returns the SDK-stamped goal). Needed because
        # there is no public seam resolving an incident back to its Goal object (see on_incident);
        # language-authored goals fall back to goal=None → attention_intent_for's least-privileged
        # defaults, which is exactly the right trust posture for them.
        self._armed_goals: dict[str, Any] = {}
        self._recent_alert_goal_ids: list[str] = []   # goal ids ONLY — never incident messages
        self._snapshot: str | None = None             # None → dirty; rebuilt on the next pull
        self._arm_tasks: Set[asyncio.Task] = set()    # in-flight async watch-arming (watch verb)
        self._unsub_goal_change: Any = None
        self._unsub_transcript: Any = None

    # -- wiring ------------------------------------------------------------
    def attach(self, engine: Any) -> None:
        """Bind the engine: register command tools + the grounding pull tool, tee transcripts,
        subscribe goal-change invalidation. Call BEFORE ``engine.start()`` (tools are start-time
        config on a held wire)."""
        # FATAL seam: an attached engine IS a live language path (its model calls watch/stop_watching,
        # routing raw language to the conductor). Behind a scripted brain those tools silently author
        # nothing — refuse it here, at the wiring call. (mode="offline" is announced-scripted and passes.)
        guard_live_language_path(self._core)
        self._engine = engine
        engine.register_tool("watch", dict(WATCH_SCHEMA), self._watch_tool)
        engine.register_tool("stop_watching", dict(STOP_WATCHING_SCHEMA), self._stop_watching_tool)
        engine.register_tool("whats_armed", dict(WHATS_ARMED_SCHEMA), self._whats_armed_tool)
        # The grounding PULL tool: the app-registered handler wins over the engine's built-in stub
        # (PipecatVoiceEngine._ensure_state_tool is idempotent and defers to an existing registration).
        engine.register_tool("percept_state", dict(PERCEPT_STATE_SCHEMA), self._percept_state_tool)
        self._unsub_transcript = engine.on_user_transcript(self._on_transcript)
        # Snapshot INVALIDATION at the authoritative mutation point (GoalRegistry revision) — this is
        # the whole point of session.on_goal_change: core.arm_from_language mutates the registry
        # without passing through session.arm, so hooking arm/disarm would miss every voice-authored
        # watch. Invalidate only; the next percept_state pull rebuilds. NO push (see module docstring).
        self._unsub_goal_change = self._session.on_goal_change(self._invalidate_snapshot)

    def detach(self) -> None:
        for unsub in (self._unsub_goal_change, self._unsub_transcript):
            if unsub is not None:
                unsub()
        self._unsub_goal_change = self._unsub_transcript = None
        for task in list(self._arm_tasks):
            task.cancel()
        self._engine = None

    # -- structured (code-armed, trusted) goals ------------------------------
    async def arm(self, goal: Any) -> Any:
        """Arm an app-authored goal through ``session.arm`` — the SDK stamps ``STRUCTURED_TRUSTED``
        (interrupt-capable under the default policy). This is the app/config path, deliberately NOT
        exposed as an engine tool: a voice model must not reach the trusted arm path."""
        stamped = await self._session.arm(goal)
        armed = stamped if stamped is not None else goal
        self._armed_goals[getattr(armed, "id", getattr(goal, "id", ""))] = armed
        return armed

    # -- command tools: raw language → the CONDUCTOR authors the watchspec -----
    async def _watch_tool(self, args: dict) -> dict:
        """``watch(command)`` — author the watchspec from raw language and arm it, ASYNC.

        Returns instantly ("On it.") so the voice turn never blocks on the conductor, and — critically
        — routes through ``core.arm_from_language``, NOT ``session.ask``, so the specialist's frame
        answer is never streamed back into the held wire (that both poisoned the session and added
        ~8s). The goal-registry revision bump invalidates the grounding snapshot when authoring
        completes; ``whats_armed``/``percept_state`` then reflect it on the next pull."""
        command = str((args or {}).get("command") or "").strip()
        if not command:
            return {"error": "command required"}

        async def _arm() -> None:
            # The ack already went out ("On it."), so a no-op here would be a SILENT broken promise —
            # the user waits forever for a watch that never existed. Detect both failure shapes (the
            # conductor raising, and the subtler one: routing the command as scene chat and authoring
            # NOTHING) and own up through the floor. "Did anything arm?" is read through the public
            # goal-change seam — the same registry-revision signal the grounding snapshot keys off.
            changed = {"hit": False}
            unsub = self._session.on_goal_change(lambda: changed.__setitem__("hit", True))
            try:
                await self._core.arm_from_language(command)   # conductor authors + arms; no spoken reply
            except Exception:  # noqa: BLE001 - background arming must never crash the wire
                await self._authoring_failed(command)
                return
            finally:
                unsub()
            if changed["hit"]:
                print(f"[bridge] watch armed from language: {command!r}; "
                      f"goals={[g['id'] for g in self._session.list_goals()]}", file=sys.stderr)
            else:
                await self._authoring_failed(command)

        task = asyncio.ensure_future(_arm())
        self._arm_tasks.add(task)
        task.add_done_callback(self._arm_tasks.discard)
        return {"output": "On it."}

    async def _authoring_failed(self, command: str) -> None:
        """Speak the truth when a watch didn't arm — a corrective nudge through the normal floor
        path (queued, short TTL: a stale correction is worse than none)."""
        if self._floor is None:
            return
        intent = AttentionIntent(
            message=("I said \"on it\" but couldn't actually set that watch up — "
                     "try phrasing it like 'tell me when …' or 'nudge me when I drink coffee'."),
            goal_id="", incident_id="", ts=None,
            preempt="queue", ttl_s=30.0, verbatim=False,
        )
        print(f"[bridge] watch authoring FAILED (no goal armed): {command!r}", file=sys.stderr)
        await self._floor.submit(intent)

    async def _stop_watching_tool(self, args: dict) -> dict:
        what = str((args or {}).get("what") or "").strip().lower()
        if not what:
            return {"error": "what required"}
        stopped = []
        for g in self._session.list_goals():   # the bounded public read — never core.goals
            hay = f"{g['id']} {g.get('condition', '')}".lower()
            if what in hay or any(w in hay for w in what.split() if len(w) > 2):
                await self._session.disarm(g["id"])
                self._armed_goals.pop(g["id"], None)
                stopped.append(g["id"])
        return {"output": f"stopped {', '.join(stopped)}" if stopped else f"nothing matching '{what}'"}

    async def _whats_armed_tool(self, _args: dict) -> dict:
        return {"armed": [{"id": g["id"], "watching_for": g.get("condition", "")}
                          for g in self._session.list_goals()]}

    # -- grounding: pull-only snapshot -----------------------------------------
    def _invalidate_snapshot(self) -> None:
        self._snapshot = None   # dirty; the next percept_state pull rebuilds

    def _build_snapshot(self) -> str:
        watching = "; ".join(f"{g['id']}: {g.get('condition', '')}"
                             for g in self._session.list_goals()) or "nothing yet"
        # Goal ids ONLY — never incident messages. Those are DELIVERY scripts (the goal's `say` text);
        # a grounding channel that carries them invites an S2S wire to re-speak old alerts on later
        # turns. Delivery text reaches the voice through the floor/renderer and nothing else.
        recent = self._recent_alert_goal_ids[-_MAX_RECENT_ALERTS:]
        fired = ", ".join(recent) if recent else "none"
        return f"[percept state] watching for → {watching}. recent alerts (goal ids) → {fired}."

    async def _percept_state_tool(self, _args: dict) -> dict:
        if self._snapshot is None:
            self._snapshot = self._build_snapshot()
            if self._engine is not None and hasattr(self._engine, "set_state_summary"):
                # Keep the engine's built-in stub coherent too (still pull — nothing enters the
                # conversation; this only refreshes the stash its own tool would serve).
                self._engine.set_state_summary(self._snapshot)
        return {"state": self._snapshot}

    # -- transcripts: free evidence ----------------------------------------------
    async def _on_transcript(self, text: str) -> None:
        """A user-turn transcript (free from a realtime model) → low-priority evidence; nobody waits.

        For a COMPONENT engine (``speaks_for_itself=False`` — plain STT, no conversational model), a
        final transcript is also the user's utterance: run a frame-grounded turn (``session.ask``) and
        speak the reply back. An S2S engine answers its own turns — asking again would double-reply."""
        text = (text or "").strip()
        if not text:
            return
        try:
            self._core.note_mention(text)   # records into the evidence/mention path; conductor mines it later
        except Exception:  # noqa: BLE001 - evidence is best-effort, never blocks the wire
            pass
        if self._engine is not None and not getattr(self._engine, "speaks_for_itself", True):
            result = await self._session.ask(text)
            reply = getattr(result, "reply", None)
            if reply:
                from engine_adapter import SpeechRequest

                await self._engine.speak(SpeechRequest(text=reply))

    # -- fires: incident → effective intent → floor --------------------------------
    def _reconstruct_goal(self, goal_id: str, incident: Any) -> Any:
        """Rebuild a minimal :class:`Goal` from ``session.goal_view`` so a language-authored fire keeps
        its authored delivery HINTS (ttl/verbatim/preempt) through ``attention_intent_for``.

        Read-only: the view is a bounded dict (no Goal leakage); we re-stamp the durable ``origin`` and
        re-nest the authored ``delivery`` so the policy sees exactly what it would from the live goal — and
        clamps ``preempt`` by that origin all the same. Returns ``None`` for an unknown/already-gone goal
        (→ ``attention_intent_for``'s least-privileged defaults)."""
        view = self._session.goal_view(goal_id)
        if view is None:
            return None
        origin_str = view.get("origin")
        try:
            origin = GoalOrigin(origin_str) if origin_str else GoalOrigin.LANGUAGE_AUTHORED
        except ValueError:
            origin = GoalOrigin.LANGUAGE_AUTHORED
        return Goal(
            id=view["id"],
            condition=view.get("condition", ""),
            say=getattr(incident, "message", "") or "",
            state={"delivery": dict(view.get("delivery") or {})},
            origin=origin,
        )

    async def on_incident(self, incident: Any) -> None:
        """Wire as ``percept.start(..., on_incident=bridge.on_incident)``.

        Resolves the incident back to its goal so ``attention_intent_for`` honors the goal's AUTHORED
        delivery (ttl/verbatim). A structured goal THIS app armed is the real object (full state); a
        language-authored goal — which never passed through :meth:`arm` — is resolved through the SDK's
        public ``session.goal_view`` seam and reconstructed into a minimal :class:`Goal`. Either way the
        policy STILL clamps effective ``preempt`` by ``origin``: a ``LANGUAGE_AUTHORED`` interrupt request
        is clamped to ``queue`` even though its ttl/verbatim now flow through."""
        goal_id = getattr(incident, "concern_id", "") or ""
        print(f"[incident] goal={goal_id!r} message={getattr(incident, 'message', '')!r}",
              file=sys.stderr)
        self._recent_alert_goal_ids.append(goal_id or "?")
        del self._recent_alert_goal_ids[:-_MAX_RECENT_ALERTS]
        self._invalidate_snapshot()
        goal = self._armed_goals.get(goal_id) or self._reconstruct_goal(goal_id, incident)
        intent = attention_intent_for(incident, goal, self._policy)   # requested→EFFECTIVE, in ONE place
        if self._telemetry is not None:
            self._telemetry.incident(intent)
        channel = self._renderer.select_channel(intent)
        if channel == "speech":
            await self._floor.submit(intent)      # speech contends for the room → the floor decides WHEN
        else:
            await self._renderer.deliver(intent, channel)   # webhook/log never contend
