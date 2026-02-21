from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from etappe02_modelle.core.schema import ObjectiveVector

T = TypeVar("T")


def _to_tuple(F: ObjectiveVector | dict[str, Any] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(F, ObjectiveVector):
        return F.as_tuple()
    if isinstance(F, tuple):
        return tuple(int(x) for x in F)  # type: ignore[return-value]
    return (int(F["f_stab"]), int(F["f_ot"]), int(F["f_pref"]), int(F["f_fair"]))


def dominates(a: ObjectiveVector | dict[str, Any] | tuple[int, int, int, int], b: ObjectiveVector | dict[str, Any] | tuple[int, int, int, int]) -> bool:
    """
    Pareto-Dominanz (Minimierung):
      a dominiert b, falls a<=b komponentenweise und in mindestens einer Komponente strikt besser.
    """

    ta = _to_tuple(a)
    tb = _to_tuple(b)
    not_worse = all(x <= y for x, y in zip(ta, tb))
    strictly_better = any(x < y for x, y in zip(ta, tb))
    return bool(not_worse and strictly_better)


def non_dominated(items: Iterable[T], *, get_F: Callable[[T], Any], get_solution_id: Callable[[T], str] | None = None) -> list[T]:
    """
    Filtert nicht-dominierte Elemente.

    Determinismus: Falls `get_solution_id` angegeben ist, wird die Rückgabe stabil nach
    `(F, solution_id)` sortiert.
    """

    items_list = list(items)
    if not items_list:
        return []

    def _key(x: T) -> tuple[tuple[int, int, int, int], str]:
        sid = get_solution_id(x) if get_solution_id is not None else ""
        return (_to_tuple(get_F(x)), str(sid))

    items_list.sort(key=_key)

    nd: list[T] = []
    for i, a in enumerate(items_list):
        Fa = get_F(a)
        dominated_flag = False
        for j, b in enumerate(items_list):
            if i == j:
                continue
            if dominates(get_F(b), Fa):
                dominated_flag = True
                break
        if not dominated_flag:
            nd.append(a)
    return nd


def dedupe_by_solution_id(items: Iterable[T], *, get_solution_id: Callable[[T], str]) -> list[T]:
    """
    Dedupliziert deterministisch nach `solution_id` (behalte erstes Vorkommen).
    """

    seen: set[str] = set()
    out: list[T] = []
    for item in items:
        sid = str(get_solution_id(item))
        if sid in seen:
            continue
        seen.add(sid)
        out.append(item)
    return out


def dedupe_by_objectives_keep_min_solution_id(
    items: Iterable[T],
    *,
    get_F: Callable[[T], Any],
    get_solution_id: Callable[[T], str],
) -> list[T]:
    """
    Falls mehrere Lösungen exakt denselben Zielvektor besitzen, wird deterministisch die
    Lösung mit der kleinsten `solution_id` behalten.
    """

    best: dict[tuple[int, int, int, int], T] = {}
    best_sid: dict[tuple[int, int, int, int], str] = {}
    for item in items:
        F_tup = _to_tuple(get_F(item))
        sid = str(get_solution_id(item))
        if F_tup not in best or sid < best_sid[F_tup]:
            best[F_tup] = item
            best_sid[F_tup] = sid

    # deterministische Ausgabe: sortiert nach (F, solution_id)
    out = list(best.values())
    out.sort(key=lambda x: (_to_tuple(get_F(x)), str(get_solution_id(x))))
    return out

