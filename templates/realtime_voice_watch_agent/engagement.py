"""Engagement policy — when is the agent LISTENING as an assistant (vs merely perceiving)?

The demo runs ``always_open``: the held S2S wire hears every turn and may always answer. That is the
right default for a desk demo and the wrong one for a room: production deployments gate engagement so
the assistant is not a hot mic with opinions. The seams are stubbed here so the shape is visible —
each stub documents what a real implementation owes.

Note the split this module protects: engagement gates the CONVERSATION, never PERCEPTION. The watch
keeps judging frames and the world-audio branch keeps feeding ``core.push_audio`` regardless of
engagement mode — an un-engaged agent still fires webhooks/logs; it just doesn't chat.
"""
from __future__ import annotations

from dataclasses import dataclass

MODES = ("always_open", "wake_word", "push_to_talk", "idle_timeout")


@dataclass(frozen=True)
class EngagementConfig:
    mode: str = "always_open"        # the demo flag — see module docstring
    wake_word: str = "percept"       # wake_word mode only
    idle_timeout_s: float = 120.0    # idle_timeout mode only


class Engagement:
    """Answers one question for the bridge/floor: may the assistant engage with this turn?"""

    def __init__(self, config: EngagementConfig | None = None) -> None:
        self.config = config or EngagementConfig()
        if self.config.mode not in MODES:
            raise ValueError(f"unknown engagement mode {self.config.mode!r} (choose from {MODES})")

    def engaged(self) -> bool:
        """True when the assistant may treat user speech as addressed to it."""
        if self.config.mode == "always_open":
            return True   # the demo flag: every turn is addressed to the agent
        if self.config.mode == "wake_word":
            return self._wake_word_engaged()
        if self.config.mode == "push_to_talk":
            return self._push_to_talk_engaged()
        return self._idle_timeout_engaged()

    # -- stubs: the production seams, shape-only --------------------------------
    def _wake_word_engaged(self) -> bool:
        """STUB — wake-word engagement. A real implementation runs a local keyword spotter on the
        VAD branch (never the gated Percept branch), opens an engagement window on a hit, and closes
        it after the reply. Requires: spotter model choice + a false-accept budget."""
        raise NotImplementedError("wake_word engagement is a stub — see docstring for what it owes")

    def _push_to_talk_engaged(self) -> bool:
        """STUB — push-to-talk. A real implementation wires a browser-UI button (or hardware GPIO)
        into an engagement latch; the mic still flows to perception, but the conversation engine only
        sees turns captured while the latch is held."""
        raise NotImplementedError("push_to_talk engagement is a stub — see docstring for what it owes")

    def _idle_timeout_engaged(self) -> bool:
        """STUB — engaged after any interaction, disengaging after ``idle_timeout_s`` of silence.
        A real implementation tracks the last engaged-turn timestamp on the floor's monotonic clock."""
        raise NotImplementedError("idle_timeout engagement is a stub — see docstring for what it owes")
