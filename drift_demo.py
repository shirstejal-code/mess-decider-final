"""Drift monitoring demo (SIMULATED stress test).

Scenario: halfway through the test period attendance suddenly drops 12%
(e.g. new mess contractor). We compare
  * a static model that never changes, and
  * a monitored model: CUSUM watches the daily forecast error; on an alarm the model
    is retrained on the most recent 150 days and an online bias correction
    (EWMA of actual / forecast) keeps it on target while it re-learns.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_raw, load_data, add_lags, split_by_date, LAG_FEATURES, COST_SHORT, COST_WASTE
from decision import plate_cost, quantity_from_forecaster
from forecast import MessForecaster
from train import load_params, selected_features

DROP = 0.88
WINDOW, CALIB_DAYS = 150, 30


def main():
    params = load_params()
    cu, co = COST_SHORT, COST_WASTE
    raw = load_raw()
    tr0, ca0, te0 = split_by_date(load_data())
    calib_start, test_start = ca0["date"].min(), te0["date"].min()
    test_dates = sorted(te0["date"].unique())
    shift_date = test_dates[len(test_dates) // 2]

    raw2 = raw.copy()
    m = raw2["date"] >= shift_date
    raw2.loc[m, "attendance"] = (raw2.loc[m, "attendance"] * DROP).round().astype(int)
    df2 = add_lags(raw2).dropna(subset=LAG_FEATURES).reset_index(drop=True)
    tr = df2[df2["date"] < calib_start]
    ca = df2[(df2["date"] >= calib_start) & (df2["date"] < test_start)]
    te = df2[df2["date"] >= test_start]
    all_dates = sorted(df2["date"].unique())

    feats, _ = selected_features(tr, ca, params)
    fc0 = MessForecaster(params, features=feats).fit(tr, ca)
    daily = (ca["attendance"] - fc0.predict_mean(ca)).groupby(ca["date"]).mean()
    mu0, s0 = float(daily.mean()), float(daily.std()) or 1.0

    def retrain(day):
        i = all_dates.index(day)
        win = all_dates[max(0, i - WINDOW + 1): i + 1]
        tr_ = df2[df2["date"].isin(win[:-CALIB_DAYS])]
        ca_ = df2[df2["date"].isin(win[-CALIB_DAYS:])]
        return MessForecaster(params, features=feats).fit(tr_, ca_, cqr=False)

    def run(monitor):
        fc, scale, active = fc0, 1.0, False
        hi_s, lo_s, alarms, rows = 0.0, 0.0, [], []
        for d in test_dates:
            day = te[te["date"] == d]
            raw_pred = fc.predict_mean(day)
            pred = raw_pred * scale
            qty = np.ceil(quantity_from_forecaster(fc, day, cu, co) * scale).astype(int)
            act = day["attendance"].values
            rows.append({"date": d, "cost": float(plate_cost(qty, act, cu, co)[0].sum()),
                         "abs_err": float(np.mean(np.abs(act - pred)))})
            if not monitor:
                continue
            if active:                                   # online bias correction (EWMA of actual/forecast)
                scale = 0.65 * scale + 0.35 * float(act.sum() / max(raw_pred.sum(), 1.0))
            z = (float(np.mean(act - pred)) - mu0) / s0
            hi_s, lo_s = max(0.0, hi_s + z - 0.5), max(0.0, lo_s - z - 0.5)   # two-sided CUSUM
            if hi_s > 5 or lo_s > 5:
                alarms.append(d)
                hi_s = lo_s = 0.0
                fc = retrain(d)                          # refit on the most recent 150 days
                i = all_dates.index(d)
                last = df2[df2["date"].isin(all_dates[max(0, i - 2): i + 1])]
                scale = float(last["attendance"].sum() / max(fc.predict_mean(last).sum(), 1.0))
                active = True
        return pd.DataFrame(rows), alarms

    static, _ = run(False)
    monitored, alarms = run(True)

    post = lambda d: d[d["date"] >= shift_date]
    after = [a for a in alarms if a >= shift_date]
    delay = (pd.to_datetime(after[0]) - pd.to_datetime(shift_date)).days if after else None
    summary = pd.DataFrame([
        {"Model": "Static (never retrained)", "Cost after drop (Rs)": int(post(static)["cost"].sum()),
         "MAE after drop": round(post(static)["abs_err"].mean(), 1), "Alarms": 0},
        {"Model": "Monitored (CUSUM + retrain)", "Cost after drop (Rs)": int(post(monitored)["cost"].sum()),
         "MAE after drop": round(post(monitored)["abs_err"].mean(), 1), "Alarms": len(alarms)},
    ])
    summary["Drift started"] = shift_date
    summary["First alarm after drift (days)"] = [None, delay]
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/drift_summary.csv", index=False)
    print(summary.to_string(index=False))

    x = pd.to_datetime(static["date"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, static["abs_err"].rolling(7, min_periods=1).mean(), color="#c62828", label="Static model")
    ax.plot(x, monitored["abs_err"].rolling(7, min_periods=1).mean(), color="#2e7d32", label="Monitored model")
    ax.axvline(pd.to_datetime(shift_date), color="black", ls="--", label="Attendance drops 12% (simulated)")
    for a in alarms:
        ax.axvline(pd.to_datetime(a), color="#2e7d32", alpha=0.25)
    ax.set_ylabel("7-day average error (students)")
    ax.set_title("Drift stress test: monitoring + retraining")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/drift.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
