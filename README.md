# BA-Pipeline: Ersatzplanung von Dienstplänen (Etappe 01–03)

Dieses Repository enthält die Python-Pipeline, mit der ich die Experimente meiner Bachelorarbeit zur **Ersatzplanung** (Repair-/Rescheduling) von Dienstplänen bei kurzfristigen Personalausfällen ausführe. Methodisch geht es um **Multi-Kriterien-Optimierung (MKO)** und einen mengenbasierten Vergleich der gefundenen Lösungen (z. B. ND/Coverage) (Wickert et al., 2019; Böðvarsdóttir und Stidsen, 2025; Ehrgott, 2005; Miettinen, 1998).  
Begriffe, Notation und Zielgrößen sind wie im schriftlichen Teil beschrieben. Etappe 01/02 decken die Implementierung ab, Etappe 03 die Auswertung.

Im Repo steht deshalb vor allem, **wie** man die Pipeline startet, **wo** die Ausgaben landen und **welche** Seeds/Configs einen Lauf festlegen. Die konkreten Zahlen/Plots entstehen beim Lauf und sind dann über die exportierten Dateien nachvollziehbar.

## Pipeline (Etappe 01–03)
- **Etappe 01 (`etappe01_simulation/`)**: Erzeugt synthetische **Instanzen–Szenarien** inkl. Referenzdienstplan $\sigma_{i,t}$, Bedarf $r_{t,s}$, Verfügbarkeit $a_{i,t,s}$, Fixierungsmenge $\mathcal{T}^{\mathrm{fix}}$ und Repair-Horizont $\mathcal{T}^{\mathrm{rep}}$. Ausgabe: Case-JSONs + `manifest.json`.
- **Etappe 02 (`etappe02_modelle/`)**: Erzeugt pro Instanz–Szenario zulässige Ersatzdienstpläne mit vier Verfahren **V1–V4** und loggt die resultierenden **Approximationsmengen** $\mathcal{A}_v$ als `runs.jsonl`/`solutions.jsonl` (Ehrgott, 2005).
- **Etappe 03 (`etappe03_evaluation/`)**: Berechnet mengenbasierte, Pareto-orientierte Metriken (ND / $P^\star$ / Coverage / Contribution) sowie sekundäre Laufzeitkennzahlen und exportiert CSV/Plots/LaTeX (Borgonjon und Maenhout, 2022).

Startpunkte:
- **`fullrun.py`**: startet Etappe 01 -> 02 -> 03 als zusammenhängenden Lauf.
- **`tools/finalrun_multiseed.py`**: Treiber für Multi-Seed-Experimente (Datensatz einmal erzeugen, Etappe 02 über eine Seed-Liste, Auswertung als `union` und `per_group`).

Optionales Notebook (Parameter wählen, Lauf starten, Exporte kopieren): `notebooks/fullrun_lab.ipynb`

## Begriffe & Notation (kurz)
Begriffe/Notation orientieren sich am schriftlichen Teil der Bachelorarbeit; im Repo verwende ich dieselben Bezeichnungen (Etappen, Verfahren V1–V4, Metriken). Wenn man etwas aus dem Code/aus den Logs nachschlagen will, hilft die Tabelle weiter unten („Mapping: BA-Begriff ↔ Repo“).

## Einrichtung
Empfohlen (Conda; vgl. `environment.yml`):
```bash
conda env create -f environment.yml
conda activate ba_pipeline
```

## Vor dem großen Run: Kurztest
Vor dem großen Run: `python3 tools/test.py`

Der Kurztest prüft grundlegende Imports und Konfigurationen, optional die Verfügbarkeit von SCIP/PySCIPOpt sowie den Aufruf von Etappe01, Etappe02, dem Treiber und Etappe03.

## Schnellstart: Fullrun (Etappe 01 -> 02 -> 03)
Ein reproduzierbarer Referenzlauf mit expliziten Parametern:
```bash
python3 fullrun.py \
  --out fullrun_out \
  --seed 101 \
  --dataset-config etappe01_simulation/configs/praxisnah_v1.json \
  --solver-config etappe02_modelle/configs/run_dataset.final.json \
  --methods V1,V2,V3,V4 \
  --budget-seconds 300 \
  --eval-group-by size,severity \
  --eval-format pdf \
  --eval-set-mode union
```
Hinweis: `fullrun.py` verwendet für Etappe 01 einen fest kodierten Seed (Legacy). Wenn der Datensatz-Seed variabel sein soll, siehe `tools/finalrun_multiseed.py` bzw. Etappe-01-CLI.

## Multi-Seed-Treiber (mehrere Seeds)
Der Treiber erzeugt den Datensatz einmal, führt Etappe 02 für mehrere `--run-seeds` aus und schreibt anschließend zwei Auswertungen (einmal `union`, einmal `per_group`).
```bash
python3 tools/finalrun_multiseed.py \
  --out-root fullrun_out \
  --dataset-config etappe01_simulation/configs/praxisnah_v1.json \
  --dataset-seed 20260219 \
  --plan etappe02_modelle/configs/run_dataset.final.json \
  --budget 600 \
  --run-seeds 101,202,303 \
  --methods V1,V2,V3,V4 \
  --package-thesis-run \
  --thesis-run-dir results/thesis_run \
  --stage2-zip zip
```
Details (Determinismus, Artefakte, Fehlersuche): `REPRODUCIBILITY.md`

## Artefakte (Ausgaben)
Generierte Ordner (entstehen beim Lauf; in einem frischen Klon nicht vorhanden):
- `fullrun_out/stage1_dataset/` (Etappe 01; Case-JSONs + `manifest.json`)
- `fullrun_out/stage2_runs/` (Etappe 02; `runs.jsonl`/`solutions.jsonl`)
- `fullrun_out/stage3_evaluation/` (Etappe 03; CSV/Plots/LaTeX + `metadata.json`)

## Mapping: BA-Begriff ↔ Repo
| Begriff (Bachelorarbeit) | Repo-/Artefaktbezug |
|---|---|
| Referenzdienstplan $\sigma_{i,t}$ | Etappe 01 Case-JSON: `params.sigma` |
| Bedarf $r_{t,s}$ | Etappe 01 Case-JSON: `params.r` |
| Verfügbarkeit $a_{i,t,s}$ | Etappe 01 Case-JSON: `params.a` (nach Störung), `params.a_base` (Basis) |
| $\mathcal{T}^{\mathrm{fix}}$, $\mathcal{T}^{\mathrm{rep}}$ | Etappe 01 Case-JSON: `params.T_fix`, `params.T_rep` |
| Zielvektor $\mathbf{F}(x)$ | Etappe 02 `solutions.jsonl`: Feld `F` (mit `f_stab`, `f_ot`, `f_pref`, `f_fair`) |
| Approximationsmenge $\mathcal{A}_v$ | Etappe 02: Menge der zulässigen `solutions.jsonl` pro Verfahren; Etappe 03: Set-Bildung `union`/`per_group` |
| Empirische Referenzmenge $P^\star$ | Etappe 02: `export_solutions --export P_star` oder Etappe 03: `--export-pstar` |
| Pareto-Metriken (ND/Coverage/Contribution) | Etappe 03 CSVs: `per_instance_metrics.csv`, `coverage_matrix*.csv`, `aggregated_metrics.csv` |

## Fehlersuche (häufige Ursachen)
1. **Python-Version zu alt**: `fullrun.py` verlangt Python ≥ 3.10 (Fehlermeldung nennt gefunden/benötigt).
2. **SCIP/PySCIPOpt fehlt** (V1/V2): `python3 -m etappe02_modelle.scripts.check_scip` ausführen und `environment.yml` nutzen.
3. **Pfad-/Config-Fehler**: `--dataset-config`/`--solver-config` zeigen auf nicht existierende JSON-Dateien.
4. **Strict-Gate in Etappe 02**: Cases mit `validation_ok_strict=false` werden standardmäßig nicht akzeptiert (Option: `--allow-nonstrict`).
5. **Multi-Seed-Auswertung `per_group`**: fehlende/inkonsistente Replicate-Keys führen zu Warnungen und ggf. Union-Fallback (vgl. Etappe 03: `--replicate-key`).

## Literatur
- Böðvarsdóttir, E. B. und Stidsen, T. (2025). *A Review of Multi-Objective Optimization Methods for Personnel Rostering Problems*. Journal of Scheduling. DOI: 10.1007/s10951-025-00845-0.
- Borgonjon, T. und Maenhout, B. (2022). *A Two-Phase Pareto Front Method for Solving the Bi-Objective Personnel Task Rescheduling Problem*. Computers & Operations Research, 138, 105624. DOI: 10.1016/j.cor.2021.105624.
- Ehrgott, M. (2005). *Multicriteria Optimization*. Springer-Verlag. DOI: 10.1007/3-540-27659-9.
- Kopanos, G. M., Capón-García, E., Espuña, A. und Puigjaner, L. (2008). *Costs for Rescheduling Actions: A Critical Issue for Reducing the Gap between Scheduling Theory and Practice*. Industrial & Engineering Chemistry Research, 47(22), 8785–8795. DOI: 10.1021/ie8005676.
- Miettinen, K. (1998). *Nonlinear Multiobjective Optimization*. Springer US. DOI: 10.1007/978-1-4615-5563-6.
- Pato, M. V. und Moz, M. (2008). *Solving a Bi-Objective Nurse Rerostering Problem by Using a Utopic Pareto Genetic Heuristic*. Journal of Heuristics, 14(4), 359–374. DOI: 10.1007/s10732-007-9040-4.
- Wickert, T. I., Smet, P. und Vanden Berghe, G. (2019). *The Nurse Rerostering Problem: Strategies for Reconstructing Disrupted Schedules*. Computers & Operations Research, 104, 319–337. DOI: 10.1016/j.cor.2018.12.014.
