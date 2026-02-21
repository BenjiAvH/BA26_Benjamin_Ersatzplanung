from __future__ import annotations

from dataclasses import dataclass
import math
import time


def split_budget_ms(total_ms: int, n: int) -> list[int]:
    """
    Deterministisches Budget-Splitting, millisekundengenau:
      base = total_ms // n
      rem  = total_ms % n
      subrun_k bekommt base + (1 wenn k < rem)
    """

    if n < 1:
        raise ValueError(f"n muss >=1 sein (n={n})")
    if total_ms < 0:
        raise ValueError(f"total_ms muss >=0 sein (total_ms={total_ms})")

    base = int(total_ms) // int(n)
    rem = int(total_ms) % int(n)
    return [int(base + (1 if k < rem else 0)) for k in range(int(n))]


def split_budget_seconds(total_seconds: float, n: int) -> list[float]:
    total_ms = int(round(float(total_seconds) * 1000.0))
    parts_ms = split_budget_ms(total_ms, n)
    return [ms / 1000.0 for ms in parts_ms]


def split_budget_ms_weighted(total_ms: int, weights: list[float]) -> list[int]:
    """
    Deterministisches, gewichtetes Budget-Splitting (ms-genau).

    Idee: allokiere `floor(total_ms * w_i/sum_w)` und verteile Rest-ms nach
    größtem fractional remainder (Tie-Breaker: kleinerer Index).

    Motivation: V1 nutzt deterministisches Budget-Splitting zwischen Bounds-Phase
    und ω-Phase (Kap. 5.1: Teilruns/Run-Budget, Vergleichbarkeit).
    """

    if total_ms < 0:
        raise ValueError(f"total_ms muss >=0 sein (total_ms={total_ms})")
    if not weights:
        raise ValueError("weights darf nicht leer sein.")

    w: list[float] = []
    for idx, val in enumerate(weights):
        try:
            fv = float(val)
        except Exception as e:
            raise ValueError(f"weights[{idx}] ist keine Zahl: {val!r}") from e
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(f"weights[{idx}] muss finite und >=0 sein (weights[{idx}]={fv}).")
        w.append(fv)

    sum_w = float(sum(w))
    if sum_w <= 0.0:
        raise ValueError("sum(weights) muss > 0 sein.")

    ideal: list[float] = [float(total_ms) * (wi / sum_w) for wi in w]
    base: list[int] = [int(math.floor(v)) for v in ideal]
    used = int(sum(base))
    rem = int(total_ms) - used
    if rem < 0:
        rem = 0

    frac = [(ideal[i] - float(base[i]), i) for i in range(len(w))]
    frac.sort(key=lambda p: (-p[0], p[1]))

    out = list(base)
    for k in range(rem):
        out[frac[k % len(out)][1]] += 1
    return out


def split_budget_seconds_weighted(total_seconds: float, weights: list[float]) -> list[float]:
    total_ms = int(round(float(total_seconds) * 1000.0))
    parts_ms = split_budget_ms_weighted(total_ms, weights)
    return [ms / 1000.0 for ms in parts_ms]


@dataclass(frozen=True)
class SubrunTimer:
    """
    Subrun-Budget als Wall-Clock (monotonic) Deadline.
    """

    start_monotonic: float
    budget_seconds: float

    @property
    def deadline(self) -> float:
        return float(self.start_monotonic) + float(self.budget_seconds)

    def time_left_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def is_expired(self) -> bool:
        return time.monotonic() >= self.deadline
