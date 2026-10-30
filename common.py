"""Shared helpers: constants, data loading, lag features, feature building, split."""
import numpy as np
import pandas as pd

MEALS = ["breakfast", "lunch", "dinner"]
CAPACITY = 1000          # hostel residents
COST_WASTE = 40          # Rs lost per plate cooked but not eaten (Co)
COST_SHORT = 60          # Rs penalty per student who finds food finished (Cu)
DATA_PATH = "data/mess_data.csv"

BASE_FEATURES = [
    "dow", "month", "is_weekend", "is_holiday", "is_exam", "is_event",
    "rain_mm", "temp_c", "menu_score",
    "meal_breakfast", "meal_lunch", "meal_dinner",
]
LAG_FEATURES = ["lag1", "lag7", "roll7", "trend", "dow4", "dev1", "mood1"]
FEATURES = BASE_FEATURES + LAG_FEATURES
HISTORY_NEEDED = 29      # days of past attendance needed to build the lag features

# Used for explanations: factors a human understands, mapped to model columns.
FEATURE_GROUPS = {
    "Recent attendance (yesterday / last weeks)": LAG_FEATURES,
    "Day of week": ["dow", "is_weekend"],
    "Time of year": ["month"],
    "Holiday / break": ["is_holiday"],
    "Exam period": ["is_exam"],
    "Campus event": ["is_event"],
    "Rain": ["rain_mm"],
    "Temperature": ["temp_c"],
    "Menu popularity": ["menu_score"],
}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add yesterday / last-week / 7-day-average attendance for the same meal.

    Everything is shifted so a row only sees the PAST (no leakage).
    Assumes one row per meal per day (fill gaps in real data).
    """
    df = df.sort_values(["meal", "date"]).copy()
    g = df.groupby("meal")["attendance"]
    df["lag1"] = g.shift(1)                                    # yesterday
    df["lag7"] = g.shift(7)                                    # same weekday last week
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["trend"] = df["lag1"] - df["roll7"]
    df["dow4"] = sum(g.shift(k) for k in (7, 14, 21, 28)) / 4  # usual level for this weekday
    usual_yday = sum(g.shift(k) for k in (8, 15, 22, 29)) / 4  # usual level for YESTERDAY's weekday
    df["dev1"] = df["lag1"] - usual_yday                       # yesterday's surprise (this meal)
    df["_rel1"] = df["lag1"] / usual_yday.clip(lower=1) - 1
    df["mood1"] = df.groupby("date")["_rel1"].transform("mean")  # yesterday's surprise (all meals)
    df = df.drop(columns=["_rel1"])
    return df.sort_values(["date", "meal"]).reset_index(drop=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = add_lags(load_raw(path))
    return df.dropna(subset=LAG_FEATURES).reset_index(drop=True)


def make_features(df: pd.DataFrame, features=None) -> pd.DataFrame:
    features = list(features) if features is not None else list(FEATURES)
    d = pd.to_datetime(df["date"])
    X = pd.DataFrame(index=df.index)
    X["dow"] = d.dt.dayofweek.values
    X["month"] = d.dt.month.values
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    for c in ["is_holiday", "is_exam", "is_event", "rain_mm", "temp_c", "menu_score"]:
        X[c] = df[c].values
    for m in MEALS:
        X["meal_" + m] = (df["meal"] == m).astype(int).values
    for c in LAG_FEATURES:
        if c in features:
            if c not in df.columns:
                raise KeyError("Missing column '%s'. Use common.load_data() or add_lags()." % c)
            X[c] = df[c].values
    return X[features]


def split_by_date(df: pd.DataFrame, train=0.6, calib=0.2):
    """Chronological split: train / calibration / test (no shuffling, no leakage)."""
    dates = np.sort(df["date"].unique())
    n = len(dates)
    a, b = dates[int(n * train)], dates[int(n * (train + calib))]
    return df[df["date"] < a], df[(df["date"] >= a) & (df["date"] < b)], df[df["date"] >= b]
