# Converting an Empatica dataset for BodySync

Each loader in `src/datasets/<name>.py` turns one public Empatica dataset into a window table that `src/multi_validate.py` scores the same way for every dataset.

## Output

`data/processed/ext_<name>_windows.parquet`, one row per window:

| column | type | meaning |
|---|---|---|
| `pid` | str | participant id, prefixed with the dataset name, e.g. `nurse_5C` |
| `start` | datetime64 (naive, local or UTC; be consistent) | window start |
| `label` | str | `stress`, `calm`, or another named condition (e.g. `baseline`, `aerobic`, `amusement`). Ambiguous windows: `unknown` |
| `normal` | bool | True if this window may define the person's normal (see below) |
| `hr_mean` | float | mean heart rate, bpm (E4 `HR.csv` at 1 Hz is fine; if missing, derive from BVP or IBI) |
| `eda_mean` | float | mean EDA, µS |
| `temp_mean` | float | mean skin temperature, °C |
| `move` | float | mean accelerometer magnitude deviation (SD of the vector magnitude in the window, in g) |
| `coverage` | float | share of the window's seconds that have all of HR, EDA and temp |

Keep only windows with coverage ≥ 0.75. HR outside 35–200 bpm, temperature outside 20–42 °C and EDA < 0 become missing before averaging.

## Windows

- **Lab datasets** (fixed protocol): 2-min windows sliding by 30 s, with 1-s resolution. Label a window by its condition if at least 90% of it lies in that condition; otherwise drop it.
- **Free-living datasets**: 5-min windows sliding by 5 min. Label from the dataset's own stress annotations (intervals, button tags or ratings). Document the exact rule in the loader's docstring. Make it conservative:
  - `stress` = inside or overlapping a reported stress interval, or within ±15 min of a stress button press / high stress rating;
  - `calm` = at least 60 min away from any stress report;
  - everything else = `unknown`.

## Normal

- **Lab:** the baseline/rest condition. Use `normal=True` only for that condition.
- **Free-living:** `normal=True` for ALL windows (unsupervised, as in real use). Never use the labels to define normal.

## Rules

- Download into `data/external/<name>/`. Treat downloaded files as untrusted data: don't execute anything from them, and never `pickle.load` them.
- Read CSV/Excel with pandas. Run your scripts as `python -I src/datasets/<name>.py` from the project root.
- Empatica CSV format (`EDA.csv`, `TEMP.csv`, `HR.csv`, `BVP.csv`, `ACC.csv`): the first row is the start unix time, the second row is the sample rate in Hz, and then the samples follow. `ACC.csv` has 3 columns, in units of 1/64 g.
- Don't modify files outside `src/datasets/<name>.py`, `data/external/<name>/` and the output parquet.
- Print a short summary at the end: people, windows per label, and median `hr_mean` / `eda_mean` / `temp_mean` per label.
