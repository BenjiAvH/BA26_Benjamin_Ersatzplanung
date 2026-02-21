from __future__ import annotations

from typing import Any

from etappe02_modelle.procedures.interfaces import Procedure
from etappe02_modelle.procedures.v1_ws_milp.procedure import V1WSMILPProcedure
from etappe02_modelle.procedures.v2_eps_milp.procedure import V2EPSMILPProcedure
from etappe02_modelle.procedures.v3_vnd_vns.procedure import V3VNDVNSProcedure
from etappe02_modelle.procedures.v4_upgh.procedure import V4UPGHProcedure


def get_procedure(verfahren: str) -> Procedure:
    v = str(verfahren).upper()
    if v == "V1":
        return V1WSMILPProcedure()
    if v == "V2":
        return V2EPSMILPProcedure()
    if v == "V3":
        return V3VNDVNSProcedure()
    if v == "V4":
        return V4UPGHProcedure()
    raise ValueError(f"Unbekanntes Verfahren: {verfahren} (erwartet: V1|V2|V3|V4)")


def normalize_verfahren(verfahren: str) -> str:
    v = str(verfahren).upper()
    if v not in {"V1", "V2", "V3", "V4"}:
        raise ValueError(f"Unbekanntes Verfahren: {verfahren} (erwartet: V1|V2|V3|V4)")
    return v


__all__ = ["Procedure", "get_procedure", "normalize_verfahren"]

