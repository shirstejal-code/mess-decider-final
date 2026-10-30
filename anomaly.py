"""Anomaly detection: find days when attendance behaved unlike anything the model expects.

A no-lag model is used so one odd day does not contaminate the next day's residual.
IsolationForest runs on the day's relative residuals (breakfast, lunch, dinner).
If data/anomaly_labels.csv exists (synthetic data only), precision/recall are reported.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from common import load_data, split_by_date, BASE_FEATURES, MEALS
from forecast import MessForecaster
from train import load_params


def main(contamination: float = 0.03):
    df = load_data()
    train, calib, test = split_by_date(df)
    fc = MessForecaster(load_params(), features=BASE_FEATURES).fit(train, calib, cqr=False)
    ev = pd.concat([calib, test]).copy()
    ev["pred"] = fc.predict_mean(ev)
    ev["rel"] = (ev["attendance"] - ev["pred"]) / ev["pred"].clip(lower=1)
    wide = ev.pivot(index="date", columns="meal", values="rel")[MEALS].dropna()

    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=0).fit(wide.values)
    wide["anomaly_score"] = (-iso.score_samples(wide[MEALS].values)).round(3)
    wide["flagged"] = iso.predict(wide[MEALS].values) == -1
    flagged = wide[wide["flagged"]].sort_values("anomaly_score", ascending=False).copy()

    if os.path.exists("data/anomaly_labels.csv"):
        truth = set(pd.read_csv("data/anomaly_labels.csv")["date"]) & set(wide.index)
        hit = set(flagged.index) & truth
        prec = len(hit) / max(len(flagged), 1)
        rec = len(hit) / max(len(truth), 1)
        flagged["was_real_shock"] = [d in truth for d in flagged.index]
        print("Hidden shock days in evaluation window: %d | flagged: %d | caught: %d"
              % (len(truth), len(flagged), len(hit)))
        print("Precision %.0f%% | Recall %.0f%%" % (prec * 100, rec * 100))
        pd.DataFrame([{"Shock days": len(truth), "Days flagged": len(flagged), "Caught": len(hit),
                       "Precision (%)": round(prec * 100), "Recall (%)": round(rec * 100)}]
                     ).to_csv("results/anomaly_summary.csv", index=False)
    for m in MEALS:
        flagged[m] = (flagged[m] * 100).round(1)
    os.makedirs("results", exist_ok=True)
    flagged.rename(columns={m: m + " vs forecast (%)" for m in MEALS}).reset_index().to_csv(
        "results/anomalies.csv", index=False)
    print(flagged.head(10).to_string())


if __name__ == "__main__":
    main()
