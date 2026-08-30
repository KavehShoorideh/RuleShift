"""Rule space for m,n,k-style games.

Frozen conventions (docs/scope-freeze.md):
- m = width (number of columns), n = height (number of rows).
- Cells are (r, c); r = 0 is the BOTTOM row (gravity pulls toward r = 0).
- Cell index = r * m + c; bitboards use bit i for cell index i.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

Cell = tuple[int, int]  # (r, c)

BOOL_KNOBS = ("gravity", "misere", "torus")


@dataclass(frozen=True)
class Ruleset:
    m: int
    n: int
    k: int
    gravity: bool = False
    misere: bool = False
    torus: bool = False
    forbidden: frozenset = frozenset()  # frozenset[Cell]

    def __post_init__(self) -> None:
        if self.m < 1 or self.n < 1:
            raise ValueError(f"board must be at least 1x1, got m={self.m} n={self.n}")
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        object.__setattr__(self, "forbidden", frozenset(tuple(c) for c in self.forbidden))
        for r, c in self.forbidden:
            if not (0 <= r < self.n and 0 <= c < self.m):
                raise ValueError(f"forbidden cell {(r, c)} outside {self.m}x{self.n} board")
        if len(self.forbidden) >= self.m * self.n:
            raise ValueError("all cells forbidden")

    # ------------------------------------------------------------ geometry
    @property
    def num_cells(self) -> int:
        return self.m * self.n

    def cell_index(self, r: int, c: int) -> int:
        return r * self.m + c

    def playable_cells(self) -> list[Cell]:
        return [
            (r, c)
            for r in range(self.n)
            for c in range(self.m)
            if (r, c) not in self.forbidden
        ]

    # ------------------------------------------------ identity / serialization
    @property
    def variant_id(self) -> str:
        """Stable, filesystem-safe identifier, e.g. 'm4n4k4_grav_mis'."""
        parts = [f"m{self.m}n{self.n}k{self.k}"]
        if self.gravity:
            parts.append("grav")
        if self.misere:
            parts.append("mis")
        if self.torus:
            parts.append("tor")
        if self.forbidden:
            parts.append("f" + "-".join(f"{r}.{c}" for r, c in sorted(self.forbidden)))
        return "_".join(parts)

    def to_dict(self) -> dict:
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "gravity": self.gravity,
            "misere": self.misere,
            "torus": self.torus,
            "forbidden": sorted([r, c] for r, c in self.forbidden),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ruleset":
        return cls(
            m=d["m"],
            n=d["n"],
            k=d["k"],
            gravity=d.get("gravity", False),
            misere=d.get("misere", False),
            torus=d.get("torus", False),
            forbidden=frozenset(tuple(c) for c in d.get("forbidden", [])),
        )

    def rule_vector(self) -> tuple[float, ...]:
        """Numeric conditioning vector [m, n, k, gravity, misere, torus].

        Forbidden cells enter models through the board planes (playable mask),
        not this vector. Normalization is model-side.
        """
        return (
            float(self.m),
            float(self.n),
            float(self.k),
            float(self.gravity),
            float(self.misere),
            float(self.torus),
        )

    def distance(self, other: "Ruleset") -> int:
        """Knob edit distance (docs/scope-freeze.md)."""
        d = abs(self.m - other.m) + abs(self.n - other.n) + abs(self.k - other.k)
        d += sum(getattr(self, kn) != getattr(other, kn) for kn in BOOL_KNOBS)
        d += len(self.forbidden ^ other.forbidden)
        return d


def standard_grid(
    ms: Iterable[int] = range(3, 7),
    ns: Iterable[int] = range(3, 7),
    ks: Iterable[int] = (3, 4),
    gravity: Iterable[bool] = (False,),
    misere: Iterable[bool] = (False,),
    torus: Iterable[bool] = (False,),
) -> list[Ruleset]:
    """The plan's rule grid (docs/plan.md par.3), excluding degenerate k > max(m, n)."""
    out = []
    for m, n, k, g, mi, t in product(ms, ns, ks, gravity, misere, torus):
        if k > max(m, n):
            continue
        out.append(Ruleset(m=m, n=n, k=k, gravity=g, misere=mi, torus=t))
    return out
