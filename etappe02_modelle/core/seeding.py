from __future__ import annotations

from etappe01_simulation.simulator.seeding import derive_seed


def seed_case(seed_global: int, dataset_id: str, case_id: str) -> int:
    return int(derive_seed(int(seed_global), str(dataset_id), str(case_id)))


def seed_group(seed_case_value: int, verfahren: str, config_hash: str) -> int:
    return int(derive_seed(int(seed_case_value), str(verfahren), str(config_hash)))


def seed_subrun(seed_group_value: int, subrun_id: str) -> int:
    return int(derive_seed(int(seed_group_value), "subrun", str(subrun_id)))

