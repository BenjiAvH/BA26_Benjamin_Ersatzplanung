# Etappe 01 — Instanz- und Szenariogenerierung (synthetisch)

Etappe 01 erzeugt einen synthetischen Datensatz aus **Instanz–Szenario**-Paaren (Case-JSONs) für die Ersatzplanung auf Basis eines **Referenzdienstplans** $\sigma_{i,t}$ (Wickert et al., 2019).  
Ziel ist eine deterministische, konfigurationsgetriebene Datengrundlage für Etappe 02 und 03, ohne externe Datenquellen.

## Ein- und Ausgabe
- **Eingabe:** JSON-Konfiguration, z.B. `etappe01_simulation/configs/praxisnah_v1.json`
- **Ausgabe:** ein Datensatz-Ordner mit
  - `manifest.json` (Index aller Cases + Tags/Seeds/Validierung),
  - `generation_log.jsonl` (nur anhängendes Log pro Case),
  - `<case_id>.json` (ein Case pro Instanz–Szenario).

Hinweis: Pfade wie `logs/daten/...` sind **lokale Ausgabeordner** und werden beim Lauf erzeugt (sie existieren in einem frischen Klon nicht).

## Schnellstart
Datensatz generieren:
```bash
python3 -m etappe01_simulation.scripts.generate_dataset \
  --config etappe01_simulation/configs/praxisnah_v1.json \
  --out logs/daten/praxisnah_v1 \
  --seed 20260208
```

Ein einzelnes Case prüfen (Basis-Checks):
```bash
python3 -m etappe01_simulation.scripts.validate_case logs/daten/praxisnah_v1/<case_id>.json
```

Strenger Modus (zusätzlich: Zulässigkeit nach der Störung):
```bash
python3 -m etappe01_simulation.scripts.validate_case --strict logs/daten/praxisnah_v1/<case_id>.json
```

## Reproduzierbarkeit (Seeds & Determinismus)
- `generate_dataset --seed` ist der **globale Seed**. Daraus werden deterministisch Instanz- und Szenario-Seeds abgeleitet (`seeds.global`, `seeds.instance`, `seeds.scenario` im Case-JSON).
- `--include-timestamp` schreibt `generated_at` in die Case-JSONs und führt damit bewusst zu **nicht bit-identischer** Ausgabe (für Fehlersuche/Protokollierung).
- Ohne `--overwrite` werden bestehende Case-Dateien nicht überschrieben; der Status erscheint in `generation_log.jsonl`.

## Case-Schema (Mapping: Begriff ↔ JSON)
Die zentralen Modellparameter liegen im Case-JSON unter `params` (Schreibweise wie im schriftlichen Teil der Bachelorarbeit, Kapitel 4):
- Referenzdienstplan $\sigma_{i,t}$: `params.sigma`
- Bedarf $r_{t,s}$: `params.r`
- Verfügbarkeit $a_{i,t,s}$: `params.a_base` (Basis), `params.a` (nach Störung; Reduktion zulässiger Zuordnungen; Kitada et al., 2010)
- Fixierungsmenge / Repair-Horizont: `params.T_fix`, `params.T_rep`
- Präferenzkosten: `params.c`
- Verbotene Schichtfolgen: `params.P`
- Wochenstruktur: `weeks.T_w`

## Fehlersuche (etappenspezifisch)
1. `scenario.status = likely_infeasible`: die Störung ist wahrscheinlich nicht reparierbar; in der Standardkonfiguration wird dies protokolliert, aber die Case-Datei kann dennoch geschrieben werden.
2. `validate_case --strict` schlägt fehl: die Zulässigkeit nach der Störung ist nicht erfüllt; für Etappe 02 ist standardmäßig `ok_strict` relevant.
3. Ausgabeordner existiert bereits: ohne `--overwrite` bleiben vorhandene Case-Dateien erhalten (`status=exists` im Log).

## Literatur
- Kitada, M., Morizawa, K. und Nagasawa, H. (2010). *A Heuristic Method in Nurse Rerostering Following a Sudden Absence of Nurses*. S. 0–6.
- Wickert, T. I., Smet, P. und Vanden Berghe, G. (2019). *The Nurse Rerostering Problem: Strategies for Reconstructing Disrupted Schedules*. Computers & Operations Research, 104, 319–337. DOI: 10.1016/j.cor.2018.12.014.
