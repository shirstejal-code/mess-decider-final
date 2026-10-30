"""Hyper-parameter tuning with time-series cross-validation (no shuffling)."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from common import load_data, split_by_date, make_features
from forecast import DEFAULT_PARAMS

PARAMS_PATH = "results/best_params.json"


def main(n_iter: int = 20):
    from train import selected_features
    df = load_data()
    train, calib, test = split_by_date(df)
    feats, fname = selected_features(train, calib)
    print("Tuning on feature set:", fname)
    X, y = make_features(train, feats), train["attendance"].values

    dist = {"max_iter": [150, 250, 400, 600],
            "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 0.1, 1.0]}
    search = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=0), dist, n_iter=n_iter,
                                cv=TimeSeriesSplit(n_splits=4), scoring="neg_mean_absolute_error",
                                random_state=0, n_jobs=-1)
    search.fit(X, y)

    best = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
            for k, v in search.best_params_.items()}
    Xte, yte = make_features(test, feats), test["attendance"].values
    default = HistGradientBoostingRegressor(random_state=0, **DEFAULT_PARAMS).fit(X, y)
    tuned = HistGradientBoostingRegressor(random_state=0, **best).fit(X, y)
    mae_d = mean_absolute_error(yte, default.predict(Xte))
    mae_t = mean_absolute_error(yte, tuned.predict(Xte))

    os.makedirs("results", exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    cv = pd.DataFrame(search.cv_results_)
    cv["cv_MAE"] = -cv["mean_test_score"]
    cv.sort_values("cv_MAE")[["params", "cv_MAE"]].head(8).to_csv("results/tuning_top8.csv", index=False)
    pd.DataFrame([{"Setting": "Default parameters", "Test MAE (students)": round(mae_d, 2)},
                  {"Setting": "Tuned (time-series CV)", "Test MAE (students)": round(mae_t, 2)}]
                 ).to_csv("results/tuning.csv", index=False)
    print("Best params:", best)
    print("Test MAE  default: %.2f | tuned: %.2f" % (mae_d, mae_t))


if __name__ == "__main__":
    main()
