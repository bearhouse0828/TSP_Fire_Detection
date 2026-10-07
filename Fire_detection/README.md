# Fire Detection from a Low-Cost Sensor Network

Exploratory analysis and algorithm development for detecting fire events from a
small network of outdoor air-quality / gas sensor stations. The work is
research-stage: it compares several detection approaches (rule-based
single-station, rule-based multi-station, and a time-series foundation model)
against a handful of manually labelled fire events. Nothing here is a validated
or production-ready detector.

## Data

`data/` contains one wide-format CSV per sensor type. The first line is a
metric comment (`# metric=PM1 (µg/m³)`), followed by a header row
`timestamp_utc,<station1>,<station2>,...` and one row per sample. Missing values
are empty cells. Sampling is high-frequency and irregular; most analyses first
resample to a 30 s grid.

| File | Content |
| --- | --- |
| `PM1.csv`, `PM4.csv` | Particulate matter (µg/m³) |
| `T.csv`, `RH.csv`, `P.csv` | Temperature (°C), relative humidity (%), pressure |
| `S1.csv` – `S12.csv` | SPH gas-sensor channels |
| `Location.csv` | `StationID, Customer, Status, Location="lat, lon"` |

Data covers roughly 2025-08-27 to 2025-09-18 for stations in the Daejeon area
(South Korea). Station IDs look like `aa-44-16-29`.

## Installation

```bash
pip install -r requirements.txt
```

`cartopy` is only needed for the map visualisation scripts (`visualize_*.py`).
The TSPulse pipeline additionally needs `torch` and IBM `granite-tsfm`
(see the commented lines in `requirements.txt`); it downloads the
`ibm-granite/granite-timeseries-tspulse-r1` weights from Hugging Face on first run.

## What is implemented

### Data loading and basic exploration
- `fire_detection_data_loader.py` – `FireDetectionDataLoader`: loads all sensor
  CSVs and `Location.csv`, coerces types, and provides summary statistics plus
  per-sensor, overview, correlation-matrix and station-comparison plots.
- `example_usage.py`, `test_data_loading.py`, `test_visualization.py` – small
  scripts exercising the loader (print-based checks, not a pytest suite).

### Single-station analysis (`single_site_processing.py`)
Runs for every station marked `Active` in `Location.csv` and writes to
`figures/<station>/` and `outputs/<station>/`:
- 30 s resampling, PM1/PM4 ratio and difference, time derivatives.
- PCA on the twelve SPH channels (scree, loadings, scores) and cross-station PM
  correlation heatmaps.
- Iterative robust hourly baseline for PM1/PM4 and robust z-scores.
- Rule-based event detection: thresholds on z(PM1), z(PM4) and the PM ratio with
  a persistence requirement and hysteresis for event end
  (`detect_fire_events`), then weighted event scoring and level assignment
  (`score_events`) → `<station>_pm_fire_events*.csv`.
- IsolationForest anomaly detection on the same features
  → `<station>_if_per_minute.csv`, `<station>_if_events.csv`.
- A blended event list combining the rule-based and IsolationForest results
  → `<station>_events_blended.csv`.

### Multi-station rule-based pipeline (`fire_detection_pipeline.py`)
Network-level detection over PM1+PM4 for all stations at once, writing to
`multi_output/`:
- 30 s median + EMA preprocessing, common-background estimation across stations.
- Per-station anomaly features (robust Z, slope Z, rolling MAD).
- Spatial / temporal plausibility checks: neighbour agreement, cross-correlation
  lag ordering, decay behaviour.
- Vetoes for high relative humidity and network-wide synchronous rises.
- Alert levels and scores per event; rough source localisation by least-squares
  fit to station positions → `events_located.csv`.

### Rule-based pipeline blended with TSPulse (`fire_detection_pipeline_tspulse_new.py`)
Same rule-based pipeline as above, plus a per-station reconstruction-error
anomaly score from the IBM TSPulse time-series foundation model. Rule and
TSPulse scores are blended into a second event list. Results are stored in
`tsp_output/` (the script itself writes to `./output/`).

```bash
python fire_detection_pipeline_tspulse_new.py --pm1 data/PM1.csv --pm4 data/PM4.csv --loc data/Location.csv
```

### Cross-station aggregation (`aggregate_fire_events.py`)
Clusters the per-station scored events whose time windows overlap within a
tolerance and reports a vote count per cluster
→ `outputs/aggregated/nearby_group_events_voted.csv`.

### Evaluation against labelled fires
- `calculate_metrics_aa_44_13_97.py` – precision / recall / F1 for the
  `aa-44-13-97` station against four manually recorded fire windows
  (Dae-Cheong Lake, given in KST and converted to UTC), at 15/30/60 min
  matching tolerances, for the single-station and TSPulse methods.
- `review_events.py` – heuristic review of the multi-station event list
  (strength, spatial checks, vetoes → confirm / keep / reject recommendation).

The label set is very small (a few reported fires at two locations), so these
metrics are indicative only and should not be read as a validation of any method.

### Visualisation
- `plot_tspulse_fire.py`, `plot_fire_timeline_enhanced.py`,
  `plot_all_stations_fire_events.py`, `plot_aa_44_13_97_fire_events.py` –
  per-station and all-station timelines of scores, flags and detected events,
  with the labelled fire windows marked. Output in `plots/`.
- `visualize_*.py` – cartopy maps of station status over time (static PNG and
  animated GIF/MP4) for the different event sources: `events_located.csv`,
  per-station `outputs/`, `multi_output/`, `tsp_output/`, and variants
  restricted to the labelled-fire date range. Output in `figures/`.

## Repository layout

```
Fire_detection/
├── data/                      # raw sensor CSVs + Location.csv
├── outputs/<station>/         # single-station events (rule, IF, blended)
├── outputs/aggregated/        # cross-station voted events
├── multi_output/              # multi-station rule pipeline outputs
├── tsp_output/                # TSPulse-blended pipeline outputs
├── figures/, plots/           # generated figures, maps and animations
├── fire_detection_data_loader.py
├── single_site_processing.py
├── fire_detection_pipeline.py
├── fire_detection_pipeline_tspulse_new.py
├── aggregate_fire_events.py
├── calculate_metrics_aa_44_13_97.py
├── review_events.py
├── plot_*.py, visualize_*.py
├── Fire_Detection_Slides.pptx, Results_presentation.pptx   # result slides
├── progress.md, project-status.md                          # working notes
└── requirements.txt
```

Several scripts are station-specific or were written for a particular
experiment (file names carry the station ID). They are kept as-is as a record
of the analysis. `progress.md` is a running log; `project-status.md` is an older
snapshot and predates most of the detection work.

## Dependencies

- pandas, numpy, scipy (`scipy.signal`, `scipy.optimize`)
- scikit-learn (PCA, IsolationForest, StandardScaler)
- matplotlib, seaborn
- cartopy (maps only)
- torch + granite-tsfm (TSPulse pipeline only)
