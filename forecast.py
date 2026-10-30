"""Forecaster = mean model + calibrated uncertainty + explanations.

Two ways to turn the forecast into a distribution (both conformally calibrated
on a held-out calibration period, so they are honest about unseen days):

* "relative" : residuals are measured as a fraction of the forecast, so the
               safety buffer scales with the expected crowd (cheap, stable).
* "cqr"      : conformalized quantile regression - one gradient-boosting
               quantile model per tau (5%..95%) + a conformal offset per tau.

The better one is picked automatically by pinball loss on the second half of
the calibration set (never on the test set).
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from common import make_features, FEATURES, FEATURE_GROUPS

TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEFAULT_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                      min_samples_leaf=20, l2_regularization=0.0)
MODEL_PATH = "models/forecaster.joblib"
DECISION_TAU = 0.6        # = Cu/(Cu+Co) for the default costs (60 and 40)


def _level(tau, n):
    return min(1.0, tau * (1 + 1.0 / n))             # finite-sample conformal correction


def _pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


class MessForecaster:
    def __init__(self, params=None, features=None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.features = list(features) if features is not None else list(FEATURES)

    # ------------------------------------------------------------------ fit
    def fit(self, train, calib, cqr=True):
        Xtr = make_features(train, self.features)
        ytr = train["attendance"].values
        Xca = make_features(calib, self.features)
        yca = calib["attendance"].values

        self.mean_model = HistGradientBoostingRegressor(random_state=0, **self.params).fit(Xtr, ytr)
        mu_ca = self.mean_model.predict(Xca)
        self.residuals = yca - mu_ca                                   # fixed-width baseline
        self.rel_resid = (yca - mu_ca) / np.clip(mu_ca, 1, None)       # relative residuals

        base = Xtr.copy()
        base["meal"] = train["meal"].values
        self.baseline = base.groupby("meal").median()                  # a 'typical day' per meal

        self.q_models, self.offsets = {}, {}
        if cqr:
            for t in TAUS:
                self.q_models[t] = HistGradientBoostingRegressor(
                    loss="quantile", quantile=t, random_state=0, **self.params).fit(Xtr, ytr)
            self.offsets = self._cqr_offsets(Xca, yca)
        self.method = self._choose_method(Xca, yca, mu_ca)
        return self

    def _cqr_offsets(self, X, y):
        n = len(y)
        return {t: float(np.quantile(y - m.predict(X), _level(t, n))) for t, m in self.q_models.items()}

    def _choose_method(self, Xca, yca, mu_ca):
        """Pick the uncertainty method on the 2nd half of the calibration set.

        Score = pinball loss at the decision quantile (= newsvendor cost up to a constant).
        The simpler 'relative' method wins unless CQR is clearly better (>5%).
        """
        if not self.q_models or len(yca) < 40:
            return "relative"
        h = len(yca) // 2
        tau = min(TAUS, key=lambda t: abs(t - DECISION_TAU))
        qrel = np.quantile(self.rel_resid[:h], _level(tau, h))
        off = float(np.quantile(yca[:h] - self.q_models[tau].predict(Xca[:h]), _level(tau, h)))
        s_rel = _pinball(yca[h:], mu_ca[h:] * (1 + qrel), tau)
        s_cqr = _pinball(yca[h:], self.q_models[tau].predict(Xca[h:]) + off, tau)
        self.method_scores = {"relative": s_rel, "cqr": s_cqr}
        return "cqr" if s_cqr < 0.95 * s_rel else "relative"

    # -------------------------------------------------------------- predict
    def predict_mean(self, df):
        return self.mean_model.predict(make_features(df, self.features))

    def quantile_function(self, df, method=None):
        """Return (taus, Q) where Q[i, j] is the tau_j-quantile of demand for row i."""
        method = method or self.method
        X = make_features(df, self.features)
        taus = np.array(TAUS)
        if method == "cqr" and self.q_models:
            Q = np.column_stack([self.q_models[t].predict(X) + self.offsets[t] for t in taus])
        else:
            mu = self.mean_model.predict(X)
            n = len(self.rel_resid)
            qrel = np.array([np.quantile(self.rel_resid, _level(t, n)) for t in taus])
            Q = mu[:, None] * (1 + qrel[None, :])
        Q = np.maximum.accumulate(np.maximum(Q, 0), axis=1)            # no crossing quantiles
        return taus, Q

    def predict_quantile(self, df, tau, method=None):
        taus, Q = self.quantile_function(df, method)
        tau = float(np.clip(tau, taus[0], taus[-1]))
        j = min(int(np.searchsorted(taus, tau, side="right")) - 1, len(taus) - 2)
        w = (tau - taus[j]) / (taus[j + 1] - taus[j])
        return Q[:, j] * (1 - w) + Q[:, j + 1] * w

    def interval(self, df, level=0.8, method=None):
        a = (1 - level) / 2
        return self.predict_quantile(df, a, method), self.predict_quantile(df, 1 - a, method)

    # -------------------------------------------------------------- explain
    def explain(self, row):
        """What-if attribution for ONE row: prediction minus prediction if a factor were 'typical'.

        Returns (prediction, typical_day_prediction, DataFrame[factor, effect]).
        """
        X = make_features(row, self.features)
        meal = row["meal"].iloc[0]
        pred = float(self.mean_model.predict(X)[0])
        Xt = X.copy()
        effects = []
        for name, cols in FEATURE_GROUPS.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            X2 = X.copy()
            for c in cols:
                X2[c] = self.baseline.loc[meal, c]
                Xt[c] = self.baseline.loc[meal, c]
            effects.append((name, pred - float(self.mean_model.predict(X2)[0])))
        typical = float(self.mean_model.predict(Xt)[0])
        out = pd.DataFrame(effects, columns=["factor", "effect"]).sort_values("effect")
        return pred, typical, out

    # ----------------------------------------------------------------- io
    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)


def load_forecaster(path=MODEL_PATH):
    return joblib.load(path)
