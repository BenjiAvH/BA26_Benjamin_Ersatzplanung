# Etappe 03 — Evaluation (mengenbasiert, Pareto-orientiert)

Etappe 03 lädt die in Etappe 02 geloggten Lösungen, bildet pro Verfahren Approximationsmengen $\mathcal{A}_v$ und berechnet mengenbasierte, Pareto-orientierte Indikatoren (ND / $P^\star$ / Coverage / Contribution) sowie sekundäre Laufzeitkennzahlen (Laufzeit, Zeit bis zur ersten zulässigen Lösung, Zulässigkeitsquote). Die Konventionen folgen dem schriftlichen Teil der Arbeit (Kapitel 5: Metriken & Vergleichsauswertung) sowie einschlägiger Literatur zur MKO und Pareto-Auswertung (Ehrgott, 2005; Borgonjon und Maenhout, 2022).

## Wichtige Konventionen
- **Alle Ziele sind Minimierung**: $\mathbf{F}(x)=(f_{\mathrm{stab}}, f_{\mathrm{ot}}, f_{\mathrm{pref}}, f_{\mathrm{fair}})$.
- **Pareto-Metriken verwenden zulässige Lösungen**; nicht-zulässige Einträge sind ausschließlich für Fehlersuche/Zulässigkeitsquote relevant.
- **Instanz–Szenario (IS)** wird im Projekt über (`dataset_id`, `case_id`) identifiziert (siehe `etappe03_evaluation/io.py`).
- Numerische Robustheit: `--eps` (Standard `1e-9`) wird konsistent für Dominanz/Deduplizierung verwendet.

## Schnellstart
Beispiel (Eingabe: `logs/` oder `fullrun_out/`, Ausgabe: `evaluation_out/`):
```bash
python3 -m etappe03_evaluation \
  --input logs/ \
  --out evaluation_out/ \
  --methods V1,V2,V3,V4 \
  --group-by size,severity \
  --format pdf \
  --set-mode union
```

Nur CSV (ohne Plots/LaTeX):
```bash
python3 -m etappe03_evaluation --input logs/ --out evaluation_out/ --no-plots --no-latex
```

## Set-Bildung: `--set-mode union` vs. `--set-mode per_group`
- `union` (Standard): $\mathcal{A}_v$ ist die Union aller zulässigen Lösungen über alle Läufe/Seeds pro IS.
- `per_group`: Metriken werden pro Replicate berechnet (Einheit: `dataset_id`, `case_id`, `replicate_id`) und sind damit robust aggregierbar.

Replicate-Key (nur für `per_group`):
- `--replicate-key auto` (Standard): wählt einen konsistenten Schlüssel (typisch: `seed_global`).
- `--replicate-key seed_global|seed_group`: erzwingt den Schlüssel; fehlende Werte führen zu Warnungen und ggf. Union-Fallback für das betroffene IS.

## Ausgaben (Standard: `evaluation_out/`)
Wichtigste Exporte (je nach `--set-mode`):
- `evaluation_out/metadata.json` — Eingabepfad, CLI-Argumente, `eps`, Zeitstempel, Git-Commit (falls verfügbar)
- `evaluation_out/aggregated_metrics.csv` — Aggregation (Median + Q1/Q3 + IQR) nach `size_class`/`severity` (+ global)
- `evaluation_out/coverage_matrix_aggregated.csv` — aggregierte Coverage-Matrix (Median + Q1/Q3 + IQR)

Zusätzliche Kern-Exporte:
- `evaluation_out/per_instance_metrics.csv` und `evaluation_out/coverage_matrix.csv` (**nur `--set-mode union`**)
- `evaluation_out/per_group_metrics.csv` und `evaluation_out/coverage_matrix_per_group.csv` (**nur `--set-mode per_group`**)

Optionale Exporte (falls aktiviert/verfügbar):
- `evaluation_out/pstar_points.csv` oder `evaluation_out/pstar_points_per_group.csv` (mit `--export-pstar`)
- `evaluation_out/milp_subruns.csv` (wenn MILP-Subruns entsprechende Felder loggen)
- `evaluation_out/winrate_heatmap.csv`, `evaluation_out/winrate_per_is.csv` (wenn $\omega$-Informationen aus V1 vorhanden sind)
- `evaluation_out/tables/*.tex` (mit `--latex`)
- `evaluation_out/figures/*.{pdf,png}` (mit `--plots`)

## Fehlersuche (etappenspezifisch)
1. Fehlende `manifest.json`: Etappe 03 funktioniert auch ohne Case-Metadaten, aber Tags/Größenklassen können dann fehlen; empfohlen ist die Etappe-01-Ausgabe mit Manifest.
2. Warnungen zu `replicate-key`: `per_group` benötigt konsistente Keys (typisch `seed_global`); bei heterogenen Logs `--replicate-key` explizit setzen.
3. Leere Metrik-Dateien: wenn Etappe 02 keine zulässigen Lösungen geloggt hat, entstehen ND-/Coverage-Exporte mit leeren Mengen; `runs.jsonl`/`solutions.jsonl` prüfen.

## Literatur
- Borgonjon, T. und Maenhout, B. (2022). *A Two-Phase Pareto Front Method for Solving the Bi-Objective Personnel Task Rescheduling Problem*. Computers & Operations Research, 138, 105624. DOI: 10.1016/j.cor.2021.105624.
- Ehrgott, M. (2005). *Multicriteria Optimization*. Springer-Verlag. DOI: 10.1007/3-540-27659-9.
