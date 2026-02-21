# Etappe 02 — Verfahren V1–V4 (Ersatzplanung)

Etappe 02 erzeugt pro **Instanz–Szenario** (Case-JSON aus Etappe 01) zulässige **Ersatzdienstpläne** und loggt die resultierenden Lösungen als **Approximationsmenge** $\mathcal{A}_v$ je Verfahren $v\in\{V1,\dots,V4\}$. Die Modellierung und die Auswertungskonventionen folgen dem schriftlichen Teil der Arbeit (Kapitel 4–5) sowie der MKO-Literatur (Ehrgott, 2005; Miettinen, 1998).  
Die Auswertung erfolgt in Etappe 03 mengenbasiert und Pareto-orientiert.

## Verfahren (V1–V4, Kurzüberblick)
- **V1 (WS-MILP)**: gewichtete Summen-Skalarisierung $Z_{\omega}(x)$ des Zielvektors $\mathbf{F}(x)$ und Lösung via MILP/SCIP (Miettinen, 1998).
- **V2 (EPS-MILP)**: $\varepsilon$-Restriktionen (Primärziel + Schranken für die übrigen Ziele) via MILP/SCIP (Ehrgott, 2005).
- **V3 (VND/VNS)**: solverfreie, stochastische Multi-Start-Heuristik (Nachbarschaftsstrukturen, ND-Ausbeute; Wickert et al., 2019).
- **V4 (UPGH)**: Pareto-orientierte, populationsbasierte Metaheuristik mit nicht-dominiertem Archiv (Pato und Moz, 2008).

Konkrete Parametrierung/Variationen liegen in `etappe02_modelle/configs/`.

## Ein- und Ausgabe
- **Eingabe:** Case-JSON aus Etappe 01, z.B. `logs/daten/<dataset_id>/<case_id>.json` oder `fullrun_out/stage1_dataset/<case_id>.json`
- **Ausgabe:** nur anhängende Artefakte pro Case und Verfahren (Beispielpfad bei `--out logs/laeufe/<dataset_id>`):
  - `logs/laeufe/<dataset_id>/<case_id>/<verfahren>/runs.jsonl` (Laufereignisse inkl. Laufplan, Seeds, Budget, Status)
  - `logs/laeufe/<dataset_id>/<case_id>/<verfahren>/solutions.jsonl` (zulässige Lösungen inkl. Zielvektor $\mathbf{F}(x)$)

Hinweis: `logs/...` ist ein lokaler Ausgabe-Pfad und wird beim Lauf erzeugt.

## Reproduzierbarkeit (Seeds, Budget, Fortsetzen)
- `--seed` ist der **globale Lauf-Seed** (`seed_global`). Daraus werden deterministisch `seed_case`, `seed_group` und `seed_subrun` abgeleitet (siehe Laufplan in `runs.jsonl`).
- `--budget` ist das **Wall-Clock-Gesamtbudget pro (Case, Verfahren)** in Sekunden. Verfahren mit mehreren Subruns (Bounds-Phase, $\omega$/$\varepsilon$-Konfigurationen) teilen das Gesamtbudget intern auf.
- `--resume` überspringt Gruppen, für die bereits ein `group_end(status=ok)` mit derselben `group_id` geloggt wurde (Group-ID enthält u.a. Config-Hash, Seed und Budget).
- Standardmäßig werden nur Cases akzeptiert, die in Etappe 01 `ok_strict=true` erfüllen („Strict-Gate“). Für Fehlersuche kann das mit `--allow-nonstrict` deaktiviert werden.

## Einrichtung
V3/V4 sind solverfrei. V1/V2 benötigen SCIP + PySCIPOpt (siehe `environment.yml`):
```bash
conda env create -f environment.yml
conda activate ba_pipeline
python3 -m etappe02_modelle.scripts.check_scip
```

## Beispiele (CLI)
Ein einzelnes Case laufen lassen:
```bash
python3 -m etappe02_modelle.scripts.run_case \
  --case logs/daten/<dataset_id>/<case_id>.json \
  --verfahren V1 \
  --config etappe02_modelle/configs/v1_ws_milp.final.json \
  --out logs/laeufe/<dataset_id> \
  --seed 20260209 \
  --budget 90 \
  --resume
```

Einen Datensatz über `manifest.json` laufen lassen:
```bash
python3 -m etappe02_modelle.scripts.run_dataset \
  --dataset logs/daten/<dataset_id>/manifest.json \
  --plan etappe02_modelle/configs/run_dataset.final.json \
  --out logs/laeufe/<dataset_id> \
  --seed 20260209 \
  --budget 90 \
  --resume
```

Export der empirischen Referenzmenge $P^\star$ (Union aller Lösungen, ND-gefiltert):
```bash
python3 -m etappe02_modelle.scripts.export_solutions \
  --dataset logs/daten/<dataset_id>/manifest.json \
  --out logs/laeufe/<dataset_id> \
  --export P_star
```

## Artefakte (Schema-Hinweise)
- `solutions.jsonl` enthält pro Lösung u.a. `solution_id`, `F` (= `f_stab`, `f_ot`, `f_pref`, `f_fair`) und `schedule_delta` (Änderungen relativ zu $\sigma_{i,t}$).
- `runs.jsonl` enthält pro Gruppe `config_snapshot` und `config_hash` (kanonisches JSON, deterministisch), sowie die geplanten Subruns und deren Seeds/Budgets.

## Fehlersuche (etappenspezifisch)
1. V1/V2 starten nicht: `check_scip` ausführen; sicherstellen, dass `scip` und `pyscipopt` aus `environment.yml` installiert sind.
2. `--resume` überspringt unerwartet: die bestehende Gruppe hat bereits ein `group_end(status=ok)` mit gleicher `group_id`; bei Budget-/Config-Änderungen ändert sich die Group-ID.
3. Keine Lösungen in `solutions.jsonl`: Budget kann zu klein sein oder die Instanz ist (unter Strict-Gate) nicht zulässig; `runs.jsonl` auf `subrun_end`/`group_end` Status prüfen.
4. Strict-Gate blockiert: `--allow-nonstrict` nur zur Fehlersuche; für den Vergleich sollten Instanzen mit `ok_strict=true` genutzt werden.

## Literatur
- Ehrgott, M. (2005). *Multicriteria Optimization*. Springer-Verlag. DOI: 10.1007/3-540-27659-9.
- Miettinen, K. (1998). *Nonlinear Multiobjective Optimization*. Springer US. DOI: 10.1007/978-1-4615-5563-6.
- Pato, M. V. und Moz, M. (2008). *Solving a Bi-Objective Nurse Rerostering Problem by Using a Utopic Pareto Genetic Heuristic*. Journal of Heuristics, 14(4), 359–374. DOI: 10.1007/s10732-007-9040-4.
- Wickert, T. I., Smet, P. und Vanden Berghe, G. (2019). *The Nurse Rerostering Problem: Strategies for Reconstructing Disrupted Schedules*. Computers & Operations Research, 104, 319–337. DOI: 10.1016/j.cor.2018.12.014.
