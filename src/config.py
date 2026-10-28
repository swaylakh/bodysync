"""Shared paths and settings for the BodySync pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "final_merged_2024_to_2025.csv"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for d in (PROCESSED, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# The three signals that are present when the watch is worn (Week 1 finding)
SIGNALS = ["hr", "eda", "temp"]
PAIRS = [("hr", "eda"), ("hr", "temp"), ("eda", "temp")]

# Plausible ranges; anything outside is treated as a sensor artefact
HR_RANGE = (35, 200)        # bpm
TEMP_RANGE = (25, 40)       # deg C, wrist skin
EDA_FLOOR = 0.05            # uS; below this the sensor barely registers

# Window settings: length in minutes -> step between window starts
WINDOWS = {30: 15, 60: 30, 360: 60}
MIN_COVERAGE = 0.75         # share of minutes in a window with all 3 signals
MIN_PURITY = 0.70           # share of minutes in a window with the dominant state
MIN_EDA_ABOVE_FLOOR = 0.5   # share of a window's EDA minutes above EDA_FLOOR

STATES = ["sleep", "rest", "active"]
STATE_COLORS = {"sleep": "#3b5ba5", "rest": "#5fa55a", "active": "#e07b39", "other": "#9e9e9e"}
