"""7-day planner: recursive forecasts (each day's forecast feeds the next day's lags)."""
import numpy as np
import pandas as pd
from common import MEALS, HISTORY_NEEDED, load_raw
from decision import quantity_from_forecaster, demand_samples, expected_cost_of

KEEP = HISTORY_NEEDED + 6


def history_from_data(raw=None, keep=KEEP):
    raw = load_raw() if raw is None else raw
    return {m: raw[raw["meal"] == m].sort_values("date")["attendance"].tail(keep).astype(float).tolist()
            for m in MEALS}


def lag_table(hist):
    """Lag features for every meal from its attendance history (list, oldest -> newest).

    Mirrors common.add_lags exactly, so training and live use agree.
    """
    out, rel = {}, []
    for m in MEALS:
        h = hist[m]
        lag1 = h[-1]
        roll7 = float(np.mean(h[-7:]))
        usual_yday = float(np.mean([h[-8], h[-15], h[-22], h[-29]]))
        out[m] = dict(lag1=float(lag1), lag7=float(h[-7]), roll7=roll7, trend=float(lag1) - roll7,
                      dow4=float(np.mean([h[-7], h[-14], h[-21], h[-28]])),
                      dev1=float(lag1) - usual_yday)
        rel.append(lag1 / max(usual_yday, 1.0) - 1)
    mood = float(np.mean(rel))
    for m in MEALS:
        out[m]["mood1"] = mood
    return out


def make_row(date, meal, rain_mm, temp_c, is_holiday, is_exam, is_event, menu_score, lags):
    d = dict(date=pd.to_datetime(date).strftime("%Y-%m-%d"), meal=meal, rain_mm=float(rain_mm), temp_c=float(temp_c),
             is_holiday=int(is_holiday), is_exam=int(is_exam), is_event=int(is_event),
             menu_score=float(menu_score))
    d.update(lags)
    return pd.DataFrame([d])


def plan(fc, hist, days, cu, co, menu_score=3.3, level=0.8):
    """days: DataFrame with date, rain_mm, temp_c, is_holiday, is_exam, is_event."""
    hist = {m: list(v) for m, v in hist.items()}
    rows = []
    for _, day in days.iterrows():
        lags = lag_table(hist)
        preds = {}
        for meal in MEALS:
            row = make_row(day["date"], meal, day["rain_mm"], day["temp_c"], day["is_holiday"],
                           day["is_exam"], day["is_event"], menu_score, lags[meal])
            pred = float(fc.predict_mean(row)[0])
            qty = int(quantity_from_forecaster(fc, row, cu, co)[0])
            lo, hi = fc.interval(row, level)
            s = demand_samples(fc, row)
            naive = int(round(lags[meal]["lag1"]))
            rows.append({"Date": pd.to_datetime(day["date"]).strftime("%a %d %b"), "Meal": meal, "Expected": round(pred),
                         "Low": round(float(lo[0])), "High": round(float(hi[0])), "Cook": qty,
                         "Expected cost (Rs)": round(expected_cost_of(s, qty, cu, co)),
                         "If same as yesterday (Rs)": round(expected_cost_of(s, naive, cu, co))})
            preds[meal] = pred
        for meal in MEALS:                       # today's forecasts become tomorrow's history
            hist[meal].append(preds[meal])
    return pd.DataFrame(rows)
