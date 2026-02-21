from __future__ import annotations

import unittest

from etappe03_evaluation.io import CaseMeta, LoadedData, RunGroupRecord, SolutionRecord
from etappe03_evaluation.metrics import evaluate


class TestSetMode(unittest.TestCase):
    def _make_data(self) -> LoadedData:
        dataset_id = "ds1"
        case_id = "c1"

        case_meta = {
            (dataset_id, case_id): CaseMeta(
                dataset_id=dataset_id,
                case_id=case_id,
                size_class="small",
                severity="leicht",
                I=2,
                T=7,
                S=3,
                seeds_instance=11,
                seeds_scenario=22,
            )
        }

        run_groups: list[RunGroupRecord] = [
            # Replicate seed_global=1
            RunGroupRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V1",
                group_id="g_v1_1",
                seed_global=1,
                seed_group=101,
                status="ok",
                wall_seconds_used=10.0,
                time_to_first_feasible_seconds=1.0,
                solutions_found=1,
            ),
            RunGroupRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V2",
                group_id="g_v2_1",
                seed_global=1,
                seed_group=202,  # absichtlich anders (auto -> seed_global)
                status="ok",
                wall_seconds_used=12.0,
                time_to_first_feasible_seconds=2.0,
                solutions_found=1,
            ),
            # Replicate seed_global=2
            RunGroupRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V1",
                group_id="g_v1_2",
                seed_global=2,
                seed_group=303,
                status="ok",
                wall_seconds_used=20.0,
                time_to_first_feasible_seconds=3.0,
                solutions_found=1,
            ),
            RunGroupRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V2",
                group_id="g_v2_2",
                seed_global=2,
                seed_group=404,
                status="ok",
                wall_seconds_used=25.0,
                time_to_first_feasible_seconds=4.0,
                solutions_found=1,
            ),
        ]

        solutions: list[SolutionRecord] = [
            # Replicate 1
            SolutionRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V1",
                group_id="g_v1_1",
                subrun_id="s0",
                solution_id="sol_v1_1",
                F=(1.0, 1.0, 1.0, 1.0),
                feasible=True,
                elapsed_seconds=1.0,
            ),
            SolutionRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V2",
                group_id="g_v2_1",
                subrun_id="s0",
                solution_id="sol_v2_1",
                F=(2.0, 2.0, 2.0, 2.0),
                feasible=True,
                elapsed_seconds=2.0,
            ),
            # Replicate 2
            SolutionRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V1",
                group_id="g_v1_2",
                subrun_id="s0",
                solution_id="sol_v1_2",
                F=(5.0, 5.0, 5.0, 5.0),
                feasible=True,
                elapsed_seconds=3.0,
            ),
            SolutionRecord(
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren="V2",
                group_id="g_v2_2",
                subrun_id="s0",
                solution_id="sol_v2_2",
                F=(0.0, 6.0, 6.0, 6.0),
                feasible=True,
                elapsed_seconds=4.0,
            ),
        ]

        return LoadedData(
            case_meta=case_meta,
            solutions=solutions,
            run_groups=run_groups,
            subrun_ends=[],
            omegas_v1=[],
            warnings=[],
        )

    def test_union_mode_schema_and_values(self) -> None:
        data = self._make_data()
        res = evaluate(data, eps=1e-9, methods=["V1", "V2"], export_pstar_points=True, set_mode="union")

        self.assertEqual(len(res.per_instance_rows), 2)
        self.assertTrue(all("replicate_id" not in r for r in res.per_instance_rows))

        by_method = {r["method"]: r for r in res.per_instance_rows}
        self.assertEqual(by_method["V1"]["p_star_size"], 2)
        self.assertEqual(by_method["V2"]["p_star_size"], 2)
        self.assertAlmostEqual(float(by_method["V1"]["contrib_inclusive"]), 0.5)
        self.assertAlmostEqual(float(by_method["V2"]["contrib_inclusive"]), 0.5)

        # P* Punkte: (1,1,1,1) von V1 und (0,6,6,6) von V2
        self.assertEqual(len(res.pstar_rows), 2)
        producers = sorted([r["producers"] for r in res.pstar_rows])
        self.assertEqual(producers, ["V1", "V2"])

    def test_per_group_mode_separates_replicates(self) -> None:
        data = self._make_data()
        res = evaluate(
            data,
            eps=1e-9,
            methods=["V1", "V2"],
            export_pstar_points=True,
            set_mode="per_group",
            replicate_key="seed_global",
        )

        self.assertEqual(len(res.per_instance_rows), 4)
        self.assertTrue(all("replicate_id" in r for r in res.per_instance_rows))

        lookup = {(r["replicate_id"], r["method"]): r for r in res.per_instance_rows}

        # Replicate 1: V1 dominiert V2 vollständig -> Contribution V1=1, V2=0
        self.assertAlmostEqual(float(lookup[("1", "V1")]["contrib_inclusive"]), 1.0)
        self.assertAlmostEqual(float(lookup[("1", "V2")]["contrib_inclusive"]), 0.0)

        # Replicate 2: beide Punkte nicht dominiert -> Contribution 0.5 / 0.5
        self.assertAlmostEqual(float(lookup[("2", "V1")]["contrib_inclusive"]), 0.5)
        self.assertAlmostEqual(float(lookup[("2", "V2")]["contrib_inclusive"]), 0.5)

        # runtime_seconds wird gesetzt (pro Replicate)
        self.assertAlmostEqual(float(lookup[("1", "V1")]["runtime_seconds"]), 10.0)
        self.assertAlmostEqual(float(lookup[("2", "V2")]["runtime_seconds"]), 25.0)

        # P* Export ist pro Replicate vorhanden (insgesamt 1+2 Punkte = 3)
        self.assertEqual(len(res.pstar_rows), 3)
        self.assertTrue(all("replicate_id" in r for r in res.pstar_rows))

    def test_per_group_auto_prefers_seed_global_if_seed_group_inconsistent(self) -> None:
        data = self._make_data()
        res = evaluate(data, eps=1e-9, methods=["V1", "V2"], set_mode="per_group", replicate_key="auto")
        used = {r.get("replicate_key_used") for r in res.per_instance_rows}
        # seed_group ist je seed_global nicht konsistent (101 vs 202, 303 vs 404) -> auto muss seed_global nutzen
        self.assertEqual(used, {"seed_global"})


if __name__ == "__main__":
    unittest.main()
