"""Terminal UI: a live poker table, rendered with ``rich``.

The renderer consumes engine events, so the same stream can drive the web UI.
Human input is validated against the engine's legal-action menu before it is
ever accepted, which means the CLI cannot put the game into an illegal state.
"""

from __future__ import annotations

import time
from typing import Callable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.text import Text

from .cards import SUIT_SYMBOLS, Card, ascii_suits
from .engine import (
    ACTION_TAKEN,
    BLINDS_POSTED,
    BOARD_DEALT,
    HAND_START,
    MESSAGE,
    PLAYER_ELIMINATED,
    POT_AWARDED,
    SHOWDOWN,
    STREET_START,
    Action,
    ActionType,
    Event,
    IllegalAction,
    LegalActions,
    PlayerStatus,
    Table,
)
from .personas import get_persona

CARD_COLORS = {"h": "red", "d": "red", "c": "white", "s": "white"}


def card_text(card: Card, *, hidden: bool = False) -> Text:
    if hidden:
        return Text("??", style="dim")
    text = Text()
    text.append(card.rank_char, style=f"bold {CARD_COLORS[card.suit_char]}")
    text.append(_suit_glyph(card), style=CARD_COLORS[card.suit_char])
    return text


def _suit_glyph(card: Card) -> str:
    if ascii_suits():
        return card.suit_char
    return SUIT_SYMBOLS[card.suit_char]


def cards_text(cards: list[Card], *, hidden: bool = False) -> Text:
    text = Text()
    if not cards:
        return Text("-", style="dim")
    for index, card in enumerate(cards):
        if index:
            text.append(" ")
        text.append_text(card_text(card, hidden=hidden))
    return text


STATUS_STYLE = {
    PlayerStatus.ACTIVE: "white",
    PlayerStatus.ALL_IN: "bold red",
    PlayerStatus.FOLDED: "dim",
    PlayerStatus.SITTING_OUT: "dim red",
}

STATUS_LABEL = {
    PlayerStatus.ACTIVE: "in",
    PlayerStatus.ALL_IN: "ALL-IN",
    PlayerStatus.FOLDED: "folded",
    PlayerStatus.SITTING_OUT: "out",
}


class TableView:
    """Owns all terminal state and knows how to draw the table."""

    def __init__(self, console: Console | None = None, *, show_thoughts: bool = True):
        self.console = console or Console()
        self.show_thoughts = show_thoughts
        self.log: list[Text] = []
        self.thought_feed: list[Text] = []
        self.pot_awards: list[str] = []
        self.hand_number = 0
        self.last_street = ""
        self._live: Live | None = None
        self._display_pot = 0
        """Last pot value seen in the event stream.

        The table's live ``pot`` balance is zeroed the moment a pot is awarded,
        so reading it for the panel would show "pot 0" right after every hand.
        This keeps the most recent meaningful value instead.
        """

    # -- logging -----------------------------------------------------------
    def add(self, text: str | Text, *, style: str | None = None) -> None:
        entry = text.copy() if isinstance(text, Text) else Text(text, style=style or "")
        self.log.append(entry)
        del self.log[:-200]
        if self._live is not None:
            self._live.update(self.render())

    def add_thought(self, text: Text) -> None:
        self.thought_feed.append(text)
        del self.thought_feed[:-60]

    # -- event handling ----------------------------------------------------
    def handle(self, table: Table, events: list[Event]) -> None:
        for event in events:
            self._handle_one(table, event)

    def _handle_one(self, table: Table, event: Event) -> None:
        data = event.data
        if event.kind == HAND_START:
            self.hand_number = data["hand_number"]
            self.pot_awards.clear()
            self._display_pot = 0
            self.add(
                f"\n-- Hand #{data['hand_number']}  "
                f"(button: {data['button_name']}) --",
                style="bold cyan",
            )
        elif event.kind == BLINDS_POSTED:
            small, big = data["small"], data["big"]
            self.add(
                f"  blinds: {small['name']} {small['amount']} / "
                f"{big['name']} {big['amount']}",
                style="dim",
            )
        elif event.kind == STREET_START:
            label = data["label"]
            self._display_pot = data["pot"]
            if label != self.last_street:
                self.last_street = label
                board = [Card.from_str(c) for c in data["board"]]
                line = Text(f"  {label}", style="bold yellow")
                if board:
                    line.append("  ")
                    line.append_text(cards_text(board))
                line.append(f"   pot {data['pot']}", style="dim")
                self.add(line)
        elif event.kind == BOARD_DEALT:
            cards = [Card.from_str(c) for c in data["cards"]]
            line = Text("  dealt ", style="dim")
            line.append_text(cards_text(cards))
            self.add(line)
        elif event.kind == ACTION_TAKEN:
            self._display_pot = data["pot"]
            self.add(self._action_line(data))
            if self.show_thoughts and data.get("thought"):
                self.add_thought(self._thought_line(data))
        elif event.kind == POT_AWARDED:
            self.add(self._award_line(data))
        elif event.kind == SHOWDOWN:
            self.add("  -- showdown --", style="bold magenta")
            for entry in data["players"]:
                cards = [Card.from_str(c) for c in entry["hole_cards"]]
                line = Text(f"    {entry['name']}: ", style="bold")
                line.append_text(cards_text(cards))
                line.append(f"  {entry['hand_name']}", style="green")
                self.add(line)
        elif event.kind == PLAYER_ELIMINATED:
            self.add(f"  {data['name']} busts out.", style="bold red")
        elif event.kind == MESSAGE:
            self.add(f"  {data['text']}", style="yellow")

    def _action_line(self, data: dict) -> Text:
        line = Text()
        line.append(f"  {data['name']} ", style="bold")
        verb = {
            "fold": "folds",
            "check": "checks",
            "call": "calls",
            "bet": "bets",
            "raise": "raises to",
            "all_in": "shoves ALL-IN for",
        }[data["action"]]
        style = {
            "fold": "red",
            "check": "blue",
            "call": "green",
            "bet": "yellow",
            "raise": "bold yellow",
            "all_in": "bold red",
        }[data["action"]]
        line.append(verb, style=style)
        if data.get("action") in {"call", "bet", "raise", "all_in"}:
            line.append(f" {data['amount']}", style=style)
        if data.get("source") == "fallback":
            line.append("  [auto]", style="dim red")
        if data.get("full_raise"):
            line.append("  (full raise)", style="dim")
        return line

    def _thought_line(self, data: dict) -> Text:
        line = Text()
        line.append(f"    ~ {data['name']}: ", style="dim italic")
        if data.get("speech"):
            line.append(f'"{data["speech"]}" ', style="italic cyan")
        line.append(data["thought"][:220], style="dim")
        return line

    def _award_line(self, data: dict) -> Text:
        line = Text()
        if data.get("reason") == "uncalled_bet_returned":
            line.append(
                f"  {data['name']} takes back {data['amount']} (uncalled)",
                style="dim",
            )
            return line
        pot_label = data.get("pot_label", "pot")
        line.append(f"  -> {data['name']} wins ", style="bold green")
        line.append(f"{data['amount']}", style="bold green")
        if data.get("reason") == "all_folded":
            line.append(" (everyone folded)", style="dim")
        elif data.get("split"):
            line.append(f" (split {pot_label})", style="dim")
        elif pot_label:
            line.append(f" ({pot_label})", style="dim")
        if data.get("hand"):
            line.append(f" with {data['hand']}", style="green")
        return line

    # -- rendering ---------------------------------------------------------
    def render(self, table: Table | None = None) -> Group | Panel:
        if table is None:
            return Panel(Group(*self.log[-40:]), title="Table", border_style="blue")
        return Layout()

    def build(self, table: Table, *, reveal: bool = False, title: str = "") -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="seats", size=len(table.players) + 3),
            Layout(name="body"),
            Layout(name="footer", size=12 if self.show_thoughts else 9),
        )
        layout["header"].update(
            Panel(
                Text(title or f"Hand #{self.hand_number}", justify="center"),
                style="bold white on dark_blue",
                border_style="blue",
            )
        )
        layout["seats"].update(self._seats_panel(table, reveal=reveal))
        layout["body"].update(self._board_panel(table))
        layout["footer"].update(self._footer_panel())
        return layout

    def _seats_panel(self, table: Table, *, reveal: bool) -> Panel:
        grid = RichTable.grid(padding=(0, 2))
        grid.add_column(justify="left")   # seat marker
        grid.add_column(justify="left")   # name
        grid.add_column(justify="left")   # persona
        grid.add_column(justify="right")  # chips
        grid.add_column(justify="right")  # bet
        grid.add_column(justify="left")   # cards
        grid.add_column(justify="left")   # status

        for player in table.players:
            markers = []
            if player.seat == table.button:
                markers.append(Text("D", style="bold yellow"))
            small, big = table.blind_seats() if len(table.seated) >= 2 else (None, None)
            if player.seat == small:
                markers.append(Text("SB", style="dim"))
            if player.seat == big:
                markers.append(Text("BB", style="dim"))
            if table.actor == player.seat and not table.is_hand_over:
                markers.append(Text("*", style="bold green"))

            persona_label = ""
            if player.is_ai and player.persona:
                try:
                    persona_label = get_persona(player.persona).name
                except KeyError:
                    persona_label = player.persona
            elif not player.is_ai:
                persona_label = "human"

            show_cards = (
                not player.is_ai
                or reveal
                or (table.is_hand_over and player.in_hand)
            )
            cards = cards_text(player.hole_cards, hidden=not show_cards)

            status = Text()
            if table.actor == player.seat and not table.is_hand_over:
                status.append("to act", style="bold green")
            else:
                status.append(
                    STATUS_LABEL[player.status], style=STATUS_STYLE[player.status]
                )
            if player.last_action is not None and not table.is_hand_over:
                status.append(f" | {player.last_action.describe()}", style="dim")

            grid.add_row(
                Text(" ").join(markers),
                Text(player.name, style="bold"),
                Text(persona_label, style="magenta" if player.is_ai else "cyan"),
                Text(str(player.chips), style="green"),
                Text(str(player.street_bet) if player.street_bet else "-", style="yellow"),
                cards,
                status,
            )
        return Panel(grid, title="Seats", border_style="blue")

    def _board_panel(self, table: Table) -> Panel:
        if not self.log:
            return Panel(Text(""), border_style="blue")
        # Show the running pot from the event stream, not the live balance: the
        # balance is cleared the instant a pot is awarded.
        pot = self._display_pot if table.is_hand_over else table.pot_total
        suffix = "  (awarded)" if table.is_hand_over and self._display_pot else ""
        return Panel(
            Group(*self.log[-18:]),
            title=f"Action  |  pot {pot}{suffix}",
            border_style="blue",
        )

    def _footer_panel(self) -> Panel:
        if not self.show_thoughts:
            return Panel(Text(""), border_style="dim")
        recent = self.thought_feed[-8:]
        if not recent:
            return Panel(Text("(AI reasoning appears here)", style="dim"), title="Table talk", border_style="dim")
        return Panel(Group(*recent), title="Table talk & reasoning", border_style="dim")


# --------------------------------------------------------------------------
# Human input
# --------------------------------------------------------------------------


def parse_human_action(raw: str, legal: LegalActions) -> Action:
    """Turn a typed command into an action, raising :class:`IllegalAction`.

    Accepts: ``f``/``fold``, ``c``/``call``, ``k``/``check``, ``a``/``allin``,
    ``r 200``/``raise 200``/``raise to 200``, and a bare number for a raise.
    """
    text = raw.strip().lower()
    if not text:
        raise IllegalAction("Type an action (or 'help').")
    if text in {"help", "?"}:
        raise IllegalAction(f"Legal: {legal.summary()}")
    if text in {"f", "fold"}:
        return Action(ActionType.FOLD, source="human")
    if text in {"k", "check", "x"}:
        return Action(ActionType.CHECK, source="human")
    if text in {"c", "call"}:
        return Action(ActionType.CALL, source="human")
    if text in {"a", "allin", "all-in", "all in", "shove", "jam"}:
        return Action(ActionType.ALL_IN, source="human")
    if text.startswith(("r", "raise", "bet")):
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            raise IllegalAction(
                f"Give a total amount, e.g. 'raise {legal.min_raise_to or legal.min_bet_to}'"
            )
        return Action(ActionType.RAISE, amount=int(digits), source="human")
    if text.isdigit():
        return Action(ActionType.RAISE, amount=int(text), source="human")
    raise IllegalAction(f"Unrecognised input {raw!r}. Legal: {legal.summary()}")


def make_human_decider(view: TableView, console: Console) -> Callable[[Table, int], Action]:
    """Interactive prompt loop, re-prompting until the engine accepts."""

    def decide(table: Table, seat: int) -> Action:
        player = table.players[seat]
        while True:
            legal = table.legal_actions(seat)
            console.print()
            console.rule(f"[bold]{player.name}'s turn[/bold]")
            console.print(f"  your cards  {cards_text(player.hole_cards)}")
            console.print(
                f"  pot {table.pot_total}   to call {legal.call_cost}   "
                f"your stack {player.chips}"
            )
            console.print(f"  [dim]legal: {legal.summary()}[/dim]")
            try:
                raw = console.input("[bold green]action> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]No input: folding.[/yellow]")
                return Action(ActionType.FOLD, source="human")
            try:
                return parse_human_action(raw, legal)
            except IllegalAction as exc:
                console.print(f"[red]{exc}[/red]")

    return decide


# --------------------------------------------------------------------------
# Top-level runner
# --------------------------------------------------------------------------


def run_cli(
    arena,
    *,
    console: Console | None = None,
    show_thoughts: bool = True,
    delay: float = 0.0,
    reveal_all: bool = False,
) -> str:
    """Drive an :class:`~pokerarena.arena.Arena` with the terminal table."""
    console = console or Console()
    view = TableView(console, show_thoughts=show_thoughts)

    def on_event(table: Table, events: list[Event]) -> None:
        view.handle(table, events)
        if delay:
            time.sleep(delay)

    arena.on_event = on_event
    console.print()
    console.rule("[bold cyan]LLM POKER ARENA[/bold cyan]")

    seats = RichTable.grid(padding=(0, 2))
    seats.add_column()
    seats.add_column()
    for player in arena.table.players:
        persona = ""
        if player.is_ai:
            try:
                persona = get_persona(player.persona).describe()
            except KeyError:
                persona = player.persona
        else:
            persona = "you"
        seats.add_row(f"[bold]{player.name}[/bold]", f"[magenta]{persona}[/magenta]")
    console.print(Panel(seats, title="Lineup", border_style="cyan"))

    with Live(
        view.build(arena.table, reveal=reveal_all),
        console=console,
        refresh_per_second=8,
        transient=False,
    ) as live:
        view._live = live

        def on_event_live(table: Table, events: list[Event]) -> None:
            view.handle(table, events)
            live.update(view.build(table, reveal=reveal_all))
            if delay:
                time.sleep(delay)

        arena.on_event = on_event_live
        winner = arena.play_game()
        live.update(view.build(arena.table, reveal=True))

    view._live = None
    console.rule(f"[bold green]Winner: {winner}[/bold green]")
    final = RichTable(title="Final stacks")
    final.add_column("Player")
    final.add_column("Persona")
    final.add_column("Chips", justify="right")
    for player in arena.table.players:
        if player.is_ai:
            try:
                label = get_persona(player.persona).name
            except KeyError:
                label = player.persona
        else:
            label = "human"
        final.add_row(player.name, label, str(player.chips))
    console.print(final)
    return winner


def print_llm_config(console: Console | None = None) -> None:
    """Show which seats will use a real model and which will fall back."""
    from .config import MODEL_REGISTRY

    console = console or Console()
    grid = RichTable(title="Model configuration", show_lines=False)
    grid.add_column("key")
    grid.add_column("model")
    grid.add_column("base_url")
    grid.add_column("API key", justify="center")
    for key, config in sorted(MODEL_REGISTRY.items()):
        grid.add_row(
            key,
            config.model,
            config.base_url,
            "[green]set[/green]" if config.available else "[red]missing[/red]",
        )
    console.print(grid)
    console.print(
        "[dim]Seats whose model key is missing automatically fall back to a "
        "local heuristic brain, so the game always runs.[/dim]"
    )
