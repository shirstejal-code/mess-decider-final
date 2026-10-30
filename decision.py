"""Decision layer: turn a forecast + its uncertainty into a cooking quantity.

Newsvendor logic: cook the quantile of demand at the critical ratio
    CR = Cu / (Cu + Co)
where Cu = cost of a shortage, Co = cost of a wasted plate.
"""
import numpy as np


def critical_ratio(cu, co):
    return cu / (cu + co)


# ---- v1: fixed-width buffer from past errors (kept as a baseline) ----------
def optimal_quantity(pred, residuals, cu, co):
    cr = critical_ratio(cu, co)
    return np.maximum(0, np.ceil(np.asarray(pred) + np.quantile(residuals, cr))).astype(int)


def prediction_interval(pred, residuals, level=0.8):
    lo = np.quantile(residuals, (1 - level) / 2)
    hi = np.quantile(residuals, 1 - (1 - level) / 2)
    return np.maximum(0, np.asarray(pred) + lo), np.asarray(pred) + hi


# ---- v2: adaptive quantile from the conformal quantile forecaster ----------
def quantity_from_forecaster(fc, df, cu, co, method=None):
    return np.ceil(fc.predict_quantile(df, critical_ratio(cu, co), method)).astype(int)


def plate_cost(cooked, actual, cu, co):
    cooked, actual = np.asarray(cooked), np.asarray(actual)
    waste = np.maximum(cooked - actual, 0)
    short = np.maximum(actual - cooked, 0)
    return co * waste + cu * short, waste, short


def demand_samples(fc, row, n=37):
    """Equally likely demand scenarios for ONE row, from the quantile function."""
    taus, Q = fc.quantile_function(row)
    grid = np.linspace(taus[0], taus[-1], n)
    return np.interp(grid, taus, Q[0])


def expected_cost_of(samples, qty, cu, co):
    return float(plate_cost(np.full(len(samples), qty), samples, cu, co)[0].mean())


def expected_cost_curve_q(fc, row, cu, co, span=150):
    s = demand_samples(fc, row)
    mid = int(np.median(s))
    qs = np.arange(max(0, mid - span), mid + span)
    return qs, np.array([expected_cost_of(s, q, cu, co) for q in qs])
