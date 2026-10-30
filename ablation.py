"""Ablation study: which ingredient of the system actually helps? (all on the test period)"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import menu_nlp
from common import load_data, split_by_date, BASE_FEATURES, FEATURES, COST_WASTE, COST_SHORT
from decision import optimal_quantity, prediction_interval, plate_cost, quantity_from_forecaster
from forecast import MessForecaster, load_forecaster
from train import load_params

CU, CO = COST_SHORT, COST_WASTE


def evaluate(name, fc, d, method):
    """method: 'fixed' (v1 buffer), 'relative' or 'cqr' (calibrated quantiles), None = model's choice."""
    pred = fc.predict_mean(d)
    actual = d["attendance"].values
    if method == "fixed":
        qty = optimal_quantity(pred, fc.residuals, CU, CO)
        lo, hi = prediction_interval(pred, fc.residuals, 0.8)
    else:
        qty = quantity_from_forecaster(fc, d, CU, CO, method)
        lo, hi = fc.interval(d, 0.8, method)
    cost, waste, short = plate_cost(qty, actual, CU, CO)
    return {"Setup": name,
            "MAE (students)": round(mean_absolute_error(actual, pred), 1),
            "80% coverage (%)": round(np.mean((actual >= lo) & (actual <= hi)) * 100, 1),
            "80% avg width": round(float(np.mean(hi - lo)), 0),
            "Wasted plates": int(waste.sum()), "Shortage (students)": int(short.sum()),
            "Total cost (Rs)": int(cost.sum())}


def main():
    df = load_data()
    train, calib, test = split_by_date(df)
    params = load_params()
    fc_base = MessForecaster(params, features=BASE_FEATURES).fit(train, calib)
    fc_lag = MessForecaster(params, features=FEATURES).fit(train, calib)

    rows = [
        evaluate("A. Calendar features + fixed buffer (v1)", fc_base, test, "fixed"),
        evaluate("B. Calendar features + relative conformal", fc_base, test, "relative"),
        evaluate("C. Calendar features + conformal quantile regression", fc_base, test, "cqr"),
        evaluate("D. + attendance lags + fixed buffer", fc_lag, test, "fixed"),
        evaluate("E. + attendance lags + relative conformal", fc_lag, test, "relative"),
        evaluate("F. + attendance lags + conformal quantile regression", fc_lag, test, "cqr"),
    ]
    final = load_forecaster()
    rows.append(evaluate("FINAL model (features + method chosen on validation data)", final, test, None))
    pipe = menu_nlp.load()
    if pipe is not None and "menu_text" in test.columns:
        t2 = test.copy()
        t2["menu_score"] = np.clip(pipe.predict(t2["menu_text"].fillna("")), 1, 5)
        rows.append(evaluate("FINAL model, menu popularity estimated by NLP from dish names", final, t2, None))

    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
