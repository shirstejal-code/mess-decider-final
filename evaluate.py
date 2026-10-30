"""Simulate cooking policies on the held-out test period."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_data, split_by_date, COST_WASTE, COST_SHORT
from decision import optimal_quantity, plate_cost, quantity_from_forecaster
from forecast import load_forecaster


def main():
    df = load_data()
    train, calib, test = split_by_date(df)
    fc = load_forecaster()
    cu, co = COST_SHORT, COST_WASTE
    actual = test["attendance"].values
    pred = fc.predict_mean(test)
    hist_mean = train.groupby("meal")["attendance"].mean()

    policies = {
        "Historical avg + 10% buffer": np.ceil(test["meal"].map(hist_mean).values * 1.10),
        "Same as yesterday": test["lag1"].values,
        "Point forecast only": np.ceil(pred),
        "Forecast + fixed buffer (v1)": optimal_quantity(pred, fc.residuals, cu, co),
        "Calibrated quantile forecast (ours)": quantity_from_forecaster(fc, test, cu, co),
    }
    rows = []
    for name, cooked in policies.items():
        cost, waste, short = plate_cost(cooked, actual, cu, co)
        rows.append({"Policy": name, "Wasted plates": int(waste.sum()),
                     "Shortage (students)": int(short.sum()), "Total cost (Rs)": int(cost.sum())})
    res = pd.DataFrame(rows)
    base = res.loc[res["Policy"] == "Same as yesterday", "Total cost (Rs)"].iloc[0]
    res["Saving vs yesterday (%)"] = ((base - res["Total cost (Rs)"]) / base * 100).round(1)

    os.makedirs("results", exist_ok=True)
    res.to_csv("results/summary.csv", index=False)
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = ["#999999", "#999999", "#6a9fd8", "#6a9fd8", "#2e7d32"]
    ax.barh(res["Policy"], res["Total cost (Rs)"], color=colors)
    ax.set_xlabel("Total cost on test period (Rs)")
    ax.set_title("Cost of each cooking policy (lower is better)")
    plt.tight_layout()
    plt.savefig("results/cost_comparison.png", dpi=150)
    plt.close()

    d = test[test["meal"] == "dinner"].tail(30)
    q = quantity_from_forecaster(fc, d, cu, co)
    lo, hi = fc.interval(d, 0.8)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x, lo, hi, color="#bbdefb", label="80% interval")
    ax.plot(x, d["attendance"].values, "o-", color="black", label="Actual")
    ax.plot(x, fc.predict_mean(d), "--", color="#1565c0", label="Forecast")
    ax.plot(x, q, "-", color="#2e7d32", label="Plates to cook")
    ax.set_xticks(x[::5])
    ax.set_xticklabels(d["date"].values[::5], rotation=30)
    ax.set_ylabel("Students / plates")
    ax.set_title("Dinner: forecast, uncertainty band and decision")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/forecast_vs_actual.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
