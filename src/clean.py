"""Step 1: clean the raw minute data.

- one row per participant-minute on a continuous grid (so windows line up in time)
- rows where the watch was not worn are dropped
- out-of-range HR / temperature become missing
- each minute gets a state label: sleep, rest (awake + sedentary), active (awake + MPA/VPA), or other
"""
import numpy as np
import pandas as pd

from config import EDA_FLOOR, HR_RANGE, PROCESSED, RAW_CSV, TEMP_RANGE

COLS = {
    "participant_full_id": "pid",
    "timestamp_iso": "time",
    "pulse_rate_bpm": "hr",
    "eda_scl_usiemens": "eda",
    "temperature_celsius": "temp",
    "prv_rmssd_ms": "rmssd",
    "respiratory_rate_brpm": "rr",
    "sleep_detection_stage": "sleep_stage",
    "activity_intensity": "intensity",
    "activity_class": "activity",
    "wearing_detection_percentage": "wear_pct",
    "accelerometers_std_g": "acc_std",
}


def label_state(df: pd.DataFrame) -> pd.Series:
    # Empatica sleep codes: 0 = awake, 101 = asleep; 102 and 300 are ambiguous and left as "other"
    awake = df.sleep_stage == 0
    return pd.Series(
        np.select(
            [df.sleep_stage == 101,
             awake & (df.intensity == "sedentary"),
             awake & df.intensity.isin(["MPA", "VPA"])],
            ["sleep", "rest", "active"],
            "other",
        ),
        index=df.index,
    )


def clean() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV, usecols=list(COLS)).rename(columns=COLS)
    raw["time"] = pd.to_datetime(raw.time).dt.tz_localize(None)
    n_raw = len(raw)

    raw.loc[~raw.hr.between(*HR_RANGE), "hr"] = np.nan
    raw.loc[~raw.temp.between(*TEMP_RANGE), "temp"] = np.nan
    raw["eda_floor"] = raw.eda < EDA_FLOOR

    worn = raw.wear_pct >= 50
    df = raw[worn].copy()
    df["state"] = label_state(df)
    df["all3"] = df[["hr", "eda", "temp"]].notna().all(axis=1)
    df = df.sort_values(["pid", "time"]).reset_index(drop=True)

    summary = pd.DataFrame({
        "step": ["raw rows", "worn (wear >= 50%)", "worn with HR+EDA+temp"],
        "rows": [n_raw, len(df), int(df.all3.sum())],
    })
    summary.to_csv(PROCESSED / "cleaning_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(df.loc[df.all3, "state"].value_counts().to_string())

    df.to_parquet(PROCESSED / "clean.parquet", index=False)
    return df


if __name__ == "__main__":
    clean()
