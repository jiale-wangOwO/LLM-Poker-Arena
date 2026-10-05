"""The arena: seats AI players, runs hands, records transcripts and history.

This module owns everything the pure engine deliberately does not: model
selection, retries, human input, logging and the game loop.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .ai import AIDecider, ChatTransport, HeuristicTransport, OpenAITransport
from .config import ArenaConfig
from .engine import (
    Action,
    Event,
    IllegalAction,
    Player,
    Table,
)
from .personas import DEFAULT_PERSONA, Persona, get_persona

# --------------------------------------------------------------------------
# Player specs
# --------------------------------------------------------------------------


@dataclass
class AISpec:
    """Configuration for one AI seat."""

    name: str
    persona_key: str
    model: str = ""
    """Provider id: a built-in preset key or an operator-defined provider."""

    def persona(self) -> Persona:
        return get_persona(self.persona_key)


class Arena:
    """Runs a full game: many hands, eliminations, button rotation."""

    def __init__(
        self,
        players: list[Player],
        *,
        config: ArenaConfig | None = None,
        transports: dict[int, ChatTransport] | None = None,
        human_seat: int | None = None,
        human_decider: Callable[[Table, int], Action] | None = None,
        on_event: Callable[[Table, list[Event]], None] | None = None,
        control: object | None = None,
        api_keys: dict[str, str] | None = None,
        providers: object | None = None,
        personas: object | None = None,
    ):
        self.config = config or ArenaConfig()
        self.rng = random.Random(self.config.seed)
        self.table = Table(
            players,
            self.config.small_blind,
            self.config.big_blind,
            rng=self.rng,
        )
        self.transports = transports or {}
        self.human_seat = human_seat
        self.human_decider = human_decider
        self.on_event = on_event
        self.control = control
        """Optional playback controller (pause / step / numbered actions)."""
        self.api_keys = dict(api_keys or {})
        """Model key -> API key supplied at runtime (e.g. by the web UI).

        These override the environment and are held in memory only; they are
        never written to disk or included in any snapshot.
        """
        self.providers = providers
        """Optional provider store (see :mod:`pokerarena.providers`).

        Anything with ``.get(provider_id) -> Provider | None`` works, which keeps
        the arena independent of where providers are stored.
        """
        self.persona_store = personas
        """Optional persona store, so operator-defined personas resolve too.

        Anything with ``.get(key) -> Persona`` and ``.has(key) -> bool`` works.
        """
        self.deciders: dict[int, AIDecider] = {}
        self.hand_history: list[dict] = []
        self.transcript: list[dict] = []
        self._build_deciders()

    # -- setup -------------------------------------------------------------
    def persona_for(self, key: str) -> Persona:
        """Resolve a persona key, including operator-defined ones."""
        if self.persona_store is not None:
            return self.persona_store.get(key)
        return get_persona(key)

    def _build_deciders(self) -> None:
        for player in self.table.players:
            if not player.seated:
                continue
            if not player.is_ai:
                continue
            persona = self.persona_for(player.persona or DEFAULT_PERSONA)
            transport = self.transports.get(player.seat)
            if transport is None:
                transport = self._default_transport(player, persona)
            self.deciders[player.seat] = AIDecider(
                persona,
                transport,
                model_label=player.model or "local",
                retries=self.config.llm_retries,
                rng=self.rng,
                transcript=self.transcript,
                memory_turns=self.config.memory_turns,
            )

    def _default_transport(self, player: Player, persona: Persona) -> ChatTransport:
        """Resolve a provider into a transport, or fall back to the local bot.

        With no provider, or a provider that has no key, the seat silently
        degrades to the offline brain so the game always runs.
        """
        if not player.model:
            return HeuristicTransport(persona, random.Random(self.rng.random()))

        resolved = self._resolve_provider(player.model)
        if resolved is None:
            return HeuristicTransport(persona, random.Random(self.rng.random()))
        model_name, base_url, api_key, max_tokens, timeout = resolved
        if not api_key:
            return HeuristicTransport(persona, random.Random(self.rng.random()))
        return OpenAITransport(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def _resolve_provider(self, provider_id: str) -> tuple[str, str, str, int, float] | None:
        """Return ``(model, base_url, api_key, max_tokens, timeout)`` or None.

        A provider is self-contained: its key is the one stored with it, plus any
        runtime override the UI supplied.  Nothing is read from the environment.
        """
        provider = self.providers.get(provider_id) if self.providers else None
        if provider is None:
            return None
        override = self.api_keys.get(provider_id, "").strip()
        return (
            provider.model,
            provider.base_url,
            override or provider.api_key.strip(),
            provider.max_tokens,
            provider.timeout,
        )

    # -- pacing hooks ------------------------------------------------------
    def before_decision(self, table: Table, seat: int) -> None:
        """Called by the game loop before each decision.  No-op by default.

        The web session overrides this (via ``control``) to implement pause and
        single-step playback without the arena needing to know about it.
        """
        control = self.control
        if control is not None:
            control.wait_if_paused()

    def after_action(self, table: Table, seat: int, action: Action, event_index: int) -> None:
        """Called after each applied action.  No-op by default."""
        control = self.control
        if control is not None:
            control.record_action(table, seat, action)

    # -- decisions ---------------------------------------------------------
    def decide(self, table: Table, seat: int) -> Action:
        player = table.players[seat]
        if seat == self.human_seat and self.human_decider is not None:
            action = self.human_decider(table, seat)
            action = Action(
                action.type,
                amount=action.amount,
                thought=action.thought,
                speech=action.speech,
                source="human",
            )
            return action
        decider = self.deciders.get(seat)
        if decider is None:
            # No decider configured: never stall, just check or fold.
            from .ai import safe_fallback

            return safe_fallback(table, seat, table.legal_actions(seat))
        return decider.decide(table, seat)

    # -- one hand ----------------------------------------------------------
    def play_hand(self) -> dict:
        summary_before = {
            p.seat: p.chips for p in self.table.players
        }
        table = self.table
        guard = 0
        table.start_hand()
        # A new hand is a new betting story: every seat starts watching from the
        # beginning of the log, and can see the hands that came before this one.
        for decider in self.deciders.values():
            decider._log_cursor = 0
            decider.hand_history = self.hand_history
        self._flush()
        while not table.is_hand_over:
            guard += 1
            if guard > 2000:  # pragma: no cover - defensive
                raise RuntimeError("Hand failed to terminate")
            if table.actor is None:
                break
            seat = table.actor
            self.before_decision(table, seat)
            action = self.decide(table, seat)
            applied = None
            try:
                applied = table.apply_action(seat, action)
            except IllegalAction as exc:
                # The decider should never let this happen; recover safely.
                legal = table.legal_actions(seat)
                from .ai import safe_fallback

                fallback = safe_fallback(table, seat, legal)
                applied = table.apply_action(seat, fallback)
                table.emit(
                    "illegal_action_recovered",
                    seat=seat,
                    name=table.players[seat].name,
                    error=str(exc),
                    fallback=fallback.type.value,
                )
            # Record the numbered action *before* flushing, so the events it
            # produced are tagged with its index and the replay scrubber can
            # reconstruct the table at any decision.
            if applied is not None:
                self.after_action(table, seat, applied, len(self.table.events))
            self._flush()

        self._flush()
        record = self._hand_record(summary_before)
        self.hand_history.append(record)
        # Hold the finished hand on screen before the next one is dealt.  This
        # is the last moment the showdown, board and payouts are all available,
        # and it is what lets a spectator actually see the result.
        if self.control is not None and hasattr(self.control, "between_hands"):
            self.control.between_hands(table)
        return record

    def _flush(self) -> None:
        events = self.table.drain_events()
        if events and self.on_event is not None:
            self.on_event(self.table, events)

    def _hand_record(self, before: dict[int, int]) -> dict:
        table = self.table
        deltas = {
            p.seat: p.chips - before.get(p.seat, 0) for p in table.players
        }
        winners = [p.name for p in table.players if deltas.get(p.seat, 0) > 0]
        return {
            "hand": table.hand_number,
            "hand_number": table.hand_number,
            "button": table.players[table.button].name,
            "board": [c.code for c in table.board],
            "results": table.showdown_results,
            "showdown": table.showdown_results,
            "pots": table.pots_snapshot,
            "pot": sum(pot.get("amount", 0) for pot in table.pots_snapshot),
            "winners": winners,
            "winner": winners[0] if winners else None,
            "reached_showdown": bool(table.showdown_results),
            "stacks_before": before,
            "stacks_after": {p.seat: p.chips for p in table.players},
            "deltas": deltas,
        }

    # -- the whole game ----------------------------------------------------
    def play_game(self) -> str:
        """Play until one player holds every chip (or the round cap is hit)."""
        table = self.table
        rounds = 0
        while not table.is_game_over() and rounds < self.config.max_rounds:
            rounds += 1
            # Tournament-style clock: escalating blinds and antes are what force
            # a short stack to commit.  Without them a bot can fold every hand
            # forever and the game runs to the cap with no winner.
            every = self.config.blind_increase_every
            if every and rounds > 1 and (rounds - 1) % every == 0:
                if table.raise_blinds(
                    self.config.blind_increase_factor,
                    self.config.max_blind_level,
                ):
                    # Antes scale with the level and are what actually stop the
                    # table from folding itself to a standstill.
                    table.set_ante(
                        max(1, round(table.big_blind * self.config.ante_fraction))
                    )
                    table.emit(
                        "message",
                        text=(
                            f"Blinds are up: {table.small_blind}/{table.big_blind}"
                            f", ante {table.ante} (level {table.blind_level})"
                        ),
                    )
                    self._flush()
            self.play_hand()
            eliminated = table.eliminate_broke_players()
            if eliminated:
                self._flush()
                for player in eliminated:
                    table.emit(
                        "message",
                        text=f"{player.name} is out of chips and leaves the table.",
                    )
                self._flush()
            if table.is_game_over():
                break
            table.rotate_button()

        winner = table.seated[0] if table.seated else None
        if winner is not None:
            table.emit(
                "game_end",
                seat=winner.seat,
                name=winner.name,
                chips=winner.chips,
                hands=len(self.hand_history),
            )
        self._flush()
        self.save_history()
        return winner.name if winner else "nobody"

    # -- persistence -------------------------------------------------------
    def save_history(self, directory: Path | None = None) -> Path | None:
        target_dir = Path(directory or self.config.hand_history_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = target_dir / f"arena_{stamp}.json"
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "players": [
                {
                    "name": p.name,
                    "seat": p.seat,
                    "is_ai": p.is_ai,
                    "persona": p.persona,
                    "model": p.model,
                    "chips": p.chips,
                }
                for p in self.table.players
            ],
            "hands": self.hand_history,
            "transcript": self.transcript,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def stack_report(self) -> list[str]:
        return [
            f"{p.name:<16} {p.chips:>6}  ({p.persona or 'human'})"
            for p in self.table.players
        ]


# --------------------------------------------------------------------------
# Convenience builders
# --------------------------------------------------------------------------


def build_players(
    ai_specs: Iterable[AISpec],
    *,
    starting_chips: int = 1000,
    with_human: bool = False,
    human_name: str = "You",
) -> tuple[list[Player], int | None]:
    """Create the seat list.  The human always sits at seat 0 when present."""
    players: list[Player] = []
    human_seat: int | None = None
    seat = 0
    if with_human:
        players.append(
            Player(
                name=human_name,
                seat=seat,
                chips=starting_chips,
                is_ai=False,
                persona="human",
            )
        )
        human_seat = seat
        seat += 1
    for spec in ai_specs:
        players.append(
            Player(
                name=spec.name,
                seat=seat,
                chips=starting_chips,
                is_ai=True,
                persona=spec.persona_key,
                model=spec.model,
            )
        )
        seat += 1
    return players, human_seat


def heuristic_transports(
    players: list[Player],
    seed: int | None = None,
    personas: object | None = None,
) -> dict[int, ChatTransport]:
    """Give every seated AI an offline heuristic brain.

    Empty chairs are skipped: they have no persona and never act.  ``personas``
    is an optional store so operator-defined personas resolve as well.
    """
    rng = random.Random(seed)
    out: dict[int, ChatTransport] = {}
    for player in players:
        if not (player.is_ai and player.seated and player.persona):
            continue
        try:
            persona = (
                personas.get(player.persona)
                if personas is not None
                else get_persona(player.persona)
            )
        except KeyError:
            continue
        out[player.seat] = HeuristicTransport(persona, random.Random(rng.random()))
    return out
