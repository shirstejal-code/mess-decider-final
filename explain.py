"""Global explainability: grouped permutation importance on the test period."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from common import load_data, split_by_date, make_features, FEATURE_GROUPS
from forecast import load_forecaster


def main(repeats: int = 10):
    df = load_data()
    train, calib, test = split_by_date(df)
    fc = load_forecaster()
    X = make_features(test, fc.features)
    y = test["attendance"].values
    base_mae = mean_absolute_error(y, fc.mean_model.predict(X))

    groups = dict(FEATURE_GROUPS)
    groups["Meal type"] = ["meal_breakfast", "meal_lunch", "meal_dinner"]
    rng = np.random.default_rng(0)
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].values[idx]            # shuffle the group jointly
            inc.append(mean_absolute_error(y, fc.mean_model.predict(Xp)) - base_mae)
        rows.append({"Factor": name, "MAE increase when shuffled (students)": round(float(np.mean(inc)), 2)})
    imp = pd.DataFrame(rows).sort_values("MAE increase when shuffled (students)", ascending=False)

    os.makedirs("results", exist_ok=True)
    imp.to_csv("results/feature_importance.csv", index=False)
    print(imp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = imp.sort_values("MAE increase when shuffled (students)")
    ax.barh(s["Factor"], s["MAE increase when shuffled (students)"], color="#1565c0")
    ax.set_xlabel("Increase in error when this factor is scrambled (students)")
    ax.set_title("What drives mess attendance? (permutation importance)")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
