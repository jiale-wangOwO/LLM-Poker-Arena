"""Shared helpers for the test suite."""

from __future__ import annotations

from pokerarena.web_state import MAX_SEATS, SeatSpec, SessionConfig


def seats_for(personas, *, with_human=False, providers=None, human_name=""):
    """Build a seat list the way the UI's seat editor would.

    Seat 0 is the human when requested, then one AI per persona.
    """
    providers = list(providers or [])
    seats: list[SeatSpec] = []
    index = 0
    if with_human:
        seats.append(SeatSpec(seat=index, kind="human", name=human_name))
        index += 1
    for offset, persona in enumerate(personas):
        seats.append(
            SeatSpec(
                seat=index,
                kind="ai",
                persona=persona,
                provider=providers[offset % len(providers)] if providers else "",
            )
        )
        index += 1
    if index > MAX_SEATS:
        raise ValueError(f"too many seats: {index}")
    return seats


def session_config(personas, **kwargs):
    """A headless test config; presentation timing is explicitly opted into."""
    payload = dict(kwargs)
    payload.setdefault("hand_result_seconds", 0)
    with_human = payload.pop("with_human", False)
    providers = payload.pop("providers", None)
    payload.pop("models", None)
    seats = payload.pop("seats", None) or seats_for(
        personas, with_human=with_human, providers=providers
    )
    return SessionConfig(seats=seats, **payload)
