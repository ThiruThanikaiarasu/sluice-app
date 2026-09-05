"""Currency conversion.

The seeded table stores each pair once. This resolves the inverse and the
cross rate on demand, and the spread is always charged against whoever is
converting -- there is no direction in which crossing a currency is free.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Quote:
    base: str
    quote: str
    mid: float
    spread_bps: int

    @property
    def effective(self) -> float:
        """Mid, less half the bid/offer spread."""
        return self.mid * (1 - self.spread_bps / 20_000)


class FXTable:
    def __init__(self, conn: sqlite3.Connection):
        self._direct: dict[tuple[str, str], Quote] = {}
        for r in conn.execute("SELECT base, quote, mid, spread_bps FROM fx_rate"):
            self._direct[(r["base"], r["quote"])] = Quote(
                r["base"], r["quote"], r["mid"], r["spread_bps"]
            )

    def _lookup(self, base: str, quote: str) -> Quote | None:
        if (base, quote) in self._direct:
            return self._direct[(base, quote)]
        inverse = self._direct.get((quote, base))
        if inverse:
            return Quote(base, quote, 1 / inverse.mid, inverse.spread_bps)
        return None

    def rate(self, base: str, quote: str) -> Quote:
        if base == quote:
            return Quote(base, quote, 1.0, 0)
        direct = self._lookup(base, quote)
        if direct:
            return direct
        # Cross via any currency quoted against both, paying both spreads.
        currencies = {c for pair in self._direct for c in pair}
        for via in currencies:
            left, right = self._lookup(base, via), self._lookup(via, quote)
            if left and right:
                return Quote(base, quote, left.mid * right.mid,
                             left.spread_bps + right.spread_bps)
        raise KeyError(f"no path from {base} to {quote}")

    def convert(self, amount: int, base: str, quote: str) -> int:
        """Minor units in base -> minor units in quote, after spread."""
        if base == quote:
            return amount
        return int(amount * self.rate(base, quote).effective)

    def spread_cost(self, amount: int, base: str, quote: str) -> int:
        """What crossing this pair costs, in minor units of `base`."""
        if base == quote:
            return 0
        return int(amount * self.rate(base, quote).spread_bps / 20_000)
