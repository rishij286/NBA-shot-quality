# NBA Shot Quality Model

Estimate the probability that an NBA shot goes in (expected FG%) from pre-shot context.

## Dataset (V1)

Place the shot-log CSV at:

```text
data/raw/shot_logs.csv
```

V1 uses the 2014–15 SportVU shot logs (partial season through early March).

**Target:** `FGM` (1 = made, 0 = missed)

**Model features:** `SHOT_DIST`, `CLOSE_DEF_DIST`, `SHOT_CLOCK`, `TOUCH_TIME`, `DRIBBLES`, `PERIOD`, `PTS_TYPE`, plus `is_home`, `game_clock_seconds`, and `is_three`.

Player/team identifiers are kept for later analysis but are not used as model inputs in V1.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

From the project root:

```bash
PYTHONPATH=src python -m nba_shot_quality.train
```

Or with an explicit path:

```bash
PYTHONPATH=src python -m nba_shot_quality.train --data-path data/raw/shot_logs.csv
```

This trains logistic regression and HistGradientBoosting, prints log loss / Brier / ROC-AUC / accuracy, and saves the selected pipeline to `models/shot_quality_v1.joblib`.

## Project layout

```text
src/nba_shot_quality/   # package
data/raw/               # local CSV (not committed)
models/                 # saved artifacts
app/                    # Streamlit stub (later)
tests/                  # lightweight tests
```
