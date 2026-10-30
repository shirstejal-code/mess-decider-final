"""Train the forecaster (mean + calibrated uncertainty) and save it."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from common import load_data, split_by_date, BASE_FEATURES, FEATURES
from decision import prediction_interval
from forecast import MessForecaster, DEFAULT_PARAMS, MODEL_PATH

PARAMS_PATH = "results/best_params.json"


def load_params():
    p = dict(DEFAULT_PARAMS)
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            p.update(json.load(f))
    return p


def selected_features(train, calib, params=None):
    """Choose calendar-only vs calendar + attendance-lag features.

    Rolling-origin (time-series) cross-validation on train+calibration data, never the
    test set. The simpler calendar-only set wins unless lags cut the error by >2%.
    """
    params = params or DEFAULT_PARAMS
    full = pd.concat([train, calib]).sort_values(["date", "meal"])
    dates = np.sort(full["date"].unique())
    cuts = [int(len(dates) * f) for f in (0.4, 0.55, 0.7, 0.85)] + [len(dates)]
    options = {"Calendar + weather + menu": BASE_FEATURES,
               "Calendar + weather + menu + attendance lags": FEATURES}
    scores = {}
    for name, feats in options.items():
        errs = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            tr_ = full[full["date"] < dates[a]]
            va_ = full[(full["date"] >= dates[a]) & (full["date"] < (dates[b] if b < len(dates) else "9999"))]
            m = MessForecaster(params, features=feats).fit(tr_, va_, cqr=False)
            errs.append(mean_absolute_error(va_["attendance"], m.predict_mean(va_)))
        scores[name] = float(np.mean(errs))
    simple, rich = list(options)
    best = rich if scores[rich] < 0.98 * scores[simple] else simple
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"Feature set": n, "Time-series CV MAE": round(v, 2), "Chosen": n == best}
                  for n, v in scores.items()]).to_csv("results/feature_set_choice.csv", index=False)
    return list(options[best]), best


def main():
    df = load_data()
    train, calib, test = split_by_date(df)
    params = load_params()
    feats, fname = selected_features(train, calib, params)
    fc = MessForecaster(params, features=feats).fit(train, calib)

    pred = fc.predict_mean(test)
    actual = test["attendance"].values
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs(actual - pred) / np.clip(actual, 1, None)) * 100

    lo, hi = fc.interval(test, 0.8)
    cov = np.mean((actual >= lo) & (actual <= hi)) * 100
    lo_r, hi_r = prediction_interval(pred, fc.residuals, 0.8)
    cov_r = np.mean((actual >= lo_r) & (actual <= hi_r)) * 100

    fc.save(MODEL_PATH)
    print("Feature set chosen on validation data: %s" % fname)
    print("Uncertainty method chosen on validation data: %s  (%s)"
          % (fc.method, {k: round(v, 2) for k, v in getattr(fc, "method_scores", {}).items()}))
    print("Test MAE: %.1f students | MAPE: %.2f%%" % (mae, mape))
    print("80%% interval  calibrated (%s): coverage %.1f%%, avg width %.0f"
          % (fc.method, cov, np.mean(hi - lo)))
    print("80%% interval  fixed-width (v1):  coverage %.1f%%, avg width %.0f"
          % (cov_r, np.mean(hi_r - lo_r)))


if __name__ == "__main__":
    main()
