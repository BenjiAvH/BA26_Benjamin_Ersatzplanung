# Reproduzierbarkeit (Einrichtung, Läufe, Artefakte)

Hier habe ich die wichtigsten Schritte gesammelt, um die Pipeline (Etappe 01–03) reproduzierbar auszuführen und die Ausgaben (CSV/Plots/LaTeX) sauber nachzuvollziehen.  
Begriffe, Notation und Zielgrößen sind wie im schriftlichen Teil (Kapitel 1–5). Dieses Dokument ist bewusst „hands-on“: Setup, Ausführung, Ausgabeordner und Exporte.

## Voraussetzungen
Empfohlen: Conda-Umgebung aus `environment.yml` (enthält u.a. SCIP/PySCIPOpt für V1/V2 sowie die Dependencies für Etappe 03).
```bash
conda env create -f environment.yml
conda activate ba_pipeline
```

## Daten (Herkunft & Platzierung)
- Die Pipeline arbeitet mit **synthetisch erzeugten** Instanzen–Szenarien (Etappe 01). Es wird keine externe Datendatei benötigt.
- Die Datengrundlage ist ausschließlich über Configs im Repo festgelegt: `etappe01_simulation/configs/` und `etappe02_modelle/configs/`.

## Seeds, Determinismus und Zeitbudget
### Etappe 01 (Datensatz-Seed)
- `python3 -m etappe01_simulation.scripts.generate_dataset --seed <int>` steuert die deterministische Datensatz-Generierung.
- `--include-timestamp` schreibt `generated_at` und ist explizit **nicht** bit-identisch (Fehlersuche/Protokollierung).

### Etappe 02 (Lauf-Seeds)
- In Etappe 02 ist `--seed` der globale Seed (`seed_global`). Sub-Seeds (`seed_case`, `seed_group`, `seed_subrun`) werden deterministisch abgeleitet und im Laufplan protokolliert (`runs.jsonl`).
- Budget: `--budget` ist das **Wall-Clock-Gesamtbudget pro (Instanz–Szenario, Verfahren)** in Sekunden.

### Etappe 03 (Deterministische Auswertung)
- Etappe 03 ist deterministisch gegeben die Eingabeartefakte. Dominanz-/Deduplizierungs-Entscheidungen verwenden `--eps` (Standard `1e-9`).

### Hinweis zum Zeitbudget (Wall-Clock)
Da die Verfahren über ein Zeitbudget (Wall-Clock) gesteuert werden, kann die exakte Menge gefundener Lösungen bei unterschiedlicher Hardware/Last variieren. Für die Vergleichbarkeit werden daher Seeds, Budget und Konfigurationen in `metadata.json`/`runs.jsonl` mitprotokolliert.

## Schnellstart: Fullrun (Etappe 01 -> 02 -> 03)
Ein reproduzierbarer Referenzlauf mit expliziten Parametern:
```bash
python3 fullrun.py \
  --out fullrun_out \
  --seed 101 \
  --dataset-config etappe01_simulation/configs/praxisnah_v1.json \
  --solver-config etappe02_modelle/configs/run_dataset.final.json \
  --methods V1,V2,V3,V4 \
  --budget-seconds 90 \
  --eval-group-by size,severity \
  --eval-format pdf \
  --eval-set-mode union
```

Hinweis: `fullrun.py` setzt den Seed für Etappe 01 fest (Legacy-Konvention). Wenn der Datensatz-Seed variabel sein soll, siehe Multi-Seed-Treiber oder Etappe-01-CLI.

## Multi-Seed-Treiber (mehrere Seeds)
Der Treiber erzeugt den Datensatz einmal, führt Etappe 02 für eine Seed-Liste aus und schreibt anschließend zwei Auswertungen (einmal `union`, einmal `per_group`).
```bash
python3 tools/finalrun_multiseed.py \
  --out-root fullrun_out \
  --dataset-config etappe01_simulation/configs/praxisnah_v1.json \
  --dataset-seed 20260219 \
  --plan etappe02_modelle/configs/run_dataset.final.json \
  --budget 90 \
  --run-seeds 101,202,303 \
  --methods V1,V2,V3,V4 \
  --package-thesis-run \
  --thesis-run-dir results/thesis_run \
  --stage2-zip zip
```

Treiber-Ausgabestruktur (wird erzeugt):
- `fullrun_out/metadata.json` (Treiber-Metadaten inkl. Seeds/Budget/Plan)
- `fullrun_out/stage1_dataset/`
- `fullrun_out/stage2_runs/`
- `fullrun_out/stage3_evaluation_union/`
- `fullrun_out/stage3_evaluation_per_group/`
- `fullrun_out/driver_logs/` (Subprozess-Protokolle + Sicherungen)

## Schrittweise Ausführung (Fehlersuche/Teilreproduktion)
Etappe 01:
```bash
python3 -m etappe01_simulation.scripts.generate_dataset \
  --config etappe01_simulation/configs/praxisnah_v1.json \
  --out logs/daten/praxisnah_v1 \
  --seed 20260208
```

Etappe 02 (Datensatz-Lauf):
```bash
python3 -m etappe02_modelle.scripts.run_dataset \
  --dataset logs/daten/praxisnah_v1/manifest.json \
  --plan etappe02_modelle/configs/run_dataset.final.json \
  --out logs/laeufe/praxisnah_v1 \
  --seed 20260209 \
  --budget 90 \
  --resume
```

Etappe 03:
```bash
python3 -m etappe03_evaluation --input logs/ --out evaluation_out/ --group-by size,severity --format pdf --set-mode union
```

## Artefakte & Exporte (wo entsteht was?)
Typische Ausgabe-Pfade (werden beim Lauf erzeugt):
- Fullrun: `fullrun_out/stage3_evaluation/` (CSV/Plots/LaTeX + `metadata.json`)
- Schrittweise Ausführung: `evaluation_out/` (analog; siehe Etappe-03-README)

Thesis-Export (optional):
- `python3 tools/thesis_artifacts.py all --src fullrun_out --dst results/thesis_run --clean --include-stage1-manifest --stage2 zip`
- `sanitize` ersetzt ausschließlich Strings in Textdateien (JSON/CSV/TeX/MD/TXT) und verändert keine Seeds/Parameter/Zahlen.

## Fehlersuche (5 typische Ursachen)
1. **Python/Conda nicht aktiv**: `conda activate ba_pipeline` (oder ein äquivalenter Python 3.12 Interpreter) vor den Läufen.
2. **SCIP/PySCIPOpt fehlt (V1/V2)**: `python3 -m etappe02_modelle.scripts.check_scip` und `environment.yml` nutzen.
3. **Strict-Gate blockiert Etappe 02**: Cases mit `ok_strict=false` werden standardmäßig nicht akzeptiert; Option zur Fehlersuche: `--allow-nonstrict`.
4. **Leere Auswertung**: wenn Etappe 02 keine zulässigen Lösungen geloggt hat, bleiben ND-/Coverage-Mengen leer; `runs.jsonl`/`solutions.jsonl` prüfen.
5. **per_group-Warnungen**: Replicate-Key fehlt/inkonsistent; `--eval-replicate-key seed_global` (Treiber) bzw. `--replicate-key` (Etappe 03) setzen.

## Literatur
Quellenangaben: siehe Literaturverzeichnis der Bachelorarbeit.
- Böðvarsdóttir, E. B. und Stidsen, T. (2025). *A Review of Multi-Objective Optimization Methods for Personnel Rostering Problems*. Journal of Scheduling. DOI: 10.1007/s10951-025-00845-0.
- Borgonjon, T. und Maenhout, B. (2022). *A Two-Phase Pareto Front Method for Solving the Bi-Objective Personnel Task Rescheduling Problem*. Computers & Operations Research, 138, 105624. DOI: 10.1016/j.cor.2021.105624.
- Ehrgott, M. (2005). *Multicriteria Optimization*. Springer-Verlag. DOI: 10.1007/3-540-27659-9.
- Kitada, M., Morizawa, K. und Nagasawa, H. (2010). *A Heuristic Method in Nurse Rerostering Following a Sudden Absence of Nurses*. S. 0–6.
- Kopanos, G. M., Capón-García, E., Espuña, A. und Puigjaner, L. (2008). *Costs for Rescheduling Actions: A Critical Issue for Reducing the Gap between Scheduling Theory and Practice*. Industrial & Engineering Chemistry Research, 47(22), 8785–8795. DOI: 10.1021/ie8005676.
- Miettinen, K. (1998). *Nonlinear Multiobjective Optimization*. Springer US. DOI: 10.1007/978-1-4615-5563-6.
- Pato, M. V. und Moz, M. (2008). *Solving a Bi-Objective Nurse Rerostering Problem by Using a Utopic Pareto Genetic Heuristic*. Journal of Heuristics, 14(4), 359–374. DOI: 10.1007/s10732-007-9040-4.
- Wickert, T. I., Smet, P. und Vanden Berghe, G. (2019). *The Nurse Rerostering Problem: Strategies for Reconstructing Disrupted Schedules*. Computers & Operations Research, 104, 319–337. DOI: 10.1016/j.cor.2018.12.014.
