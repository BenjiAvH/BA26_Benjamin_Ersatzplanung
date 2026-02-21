from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


ObjectiveTuple = tuple[float, float, float, float]


@dataclass(frozen=True)
class Point:
    """
    Ein Punkt im Zielraum mit Producer-Set.

    Producer-Set:
      - Menge der Verfahren, die diesen (nach Dedupe) Zielvektor erzeugt haben.
      - Wird für Contribution (inclusive/unique) benötigt.
    """

    F: ObjectiveTuple
    producers: frozenset[str]


def _producers_key(producers: frozenset[str]) -> str:
    return ",".join(sorted(producers))


def equal_eps(a: ObjectiveTuple, b: ObjectiveTuple, eps: float) -> bool:
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def dominates(a: ObjectiveTuple, b: ObjectiveTuple, *, eps: float = 1e-9) -> bool:
    """
    Pareto-Dominanz (Minimierung) gemäß Bachelorarbeit (Kap. 5.3: Metriken & Vergleichsauswertung):

      a dominiert b  <=>  ∀k: a_k <= b_k  UND  ∃k: a_k < b_k

    Numerische Robustheit:
      - <= wird als a_k <= b_k + eps geprüft
      - <  wird als a_k <  b_k - eps geprüft
    """

    not_worse = all(x <= y + eps for x, y in zip(a, b))
    strictly_better = any(x < y - eps for x, y in zip(a, b))
    return bool(not_worse and strictly_better)


def dedupe_points(points: Iterable[Point], *, eps: float = 1e-9) -> list[Point]:
    """
    Dedupe von Punkten nach Zielvektor innerhalb `eps`.

    - Punkte, die komponentenweise innerhalb eps identisch sind, werden zusammengeführt.
    - `producers` wird als Vereinigungsmenge geführt.
    - Ausgabe ist deterministisch (Sortierung nach (F, producers)).
    """

    pts = sorted(points, key=lambda p: (p.F, _producers_key(p.producers)))
    if not pts:
        return []

    out: list[Point] = [pts[0]]
    for p in pts[1:]:
        last = out[-1]
        if equal_eps(last.F, p.F, eps):
            out[-1] = Point(F=last.F, producers=last.producers | p.producers)
            continue
        out.append(p)
    return out


def non_dominated(points: Sequence[Point], *, eps: float = 1e-9) -> list[Point]:
    """
    ND(S): alle Punkte in S, die von keinem anderen Punkt dominiert werden.

    Determinismus:
      - stabile Sortierung nach (F, producers) vor der Filterung
    """

    pts = sorted(points, key=lambda p: (p.F, _producers_key(p.producers)))
    if not pts:
        return []

    nd: list[Point] = []
    for i, a in enumerate(pts):
        dominated_flag = False
        for j, b in enumerate(pts):
            if i == j:
                continue
            if dominates(b.F, a.F, eps=eps):
                dominated_flag = True
                break
        if not dominated_flag:
            nd.append(a)
    return nd
