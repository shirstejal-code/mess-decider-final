"""Menu NLP: predict how popular a menu is (1-5) from its dish names.

TF-IDF (word + bigram) + Ridge regression. The dashboard uses it so a mess
manager can just type the menu instead of guessing a popularity score.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from common import load_raw, split_by_date

MENU_PATH = "models/menu_nlp.joblib"


def build():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), Ridge(alpha=0.3))


def predict_score(pipe, text: str) -> float:
    return float(np.clip(pipe.predict([text])[0], 1, 5))


def load():
    try:
        return joblib.load(MENU_PATH)
    except Exception:
        return None


def main():
    df = load_raw()
    if "menu_text" not in df.columns or df["menu_text"].isna().all():
        print("No menu_text column found - skipping menu NLP.")
        return
    df = df.dropna(subset=["menu_text"])
    train, calib, test = split_by_date(df)
    pipe = build().fit(train["menu_text"], train["menu_score"])
    pred = np.clip(pipe.predict(test["menu_text"]), 1, 5)
    mae = mean_absolute_error(test["menu_score"], pred)
    base = mean_absolute_error(test["menu_score"], np.full(len(test), train["menu_score"].mean()))
    print("Menu NLP test MAE: %.3f (always-predict-average baseline: %.3f)" % (mae, base))

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(pipe, MENU_PATH)

    dishes = sorted({d.strip() for t in df["menu_text"] for d in str(t).split(",") if d.strip()})
    table = pd.DataFrame({"dish": dishes,
                          "ml_popularity": [round(predict_score(pipe, x), 2) for x in dishes]})
    table.sort_values("ml_popularity", ascending=False).to_csv("results/dish_scores.csv", index=False)
    pd.DataFrame([{"Model": "TF-IDF + Ridge", "MAE (score points)": round(mae, 3)},
                  {"Model": "Always predict average", "MAE (score points)": round(base, 3)}]
                 ).to_csv("results/menu_nlp.csv", index=False)


if __name__ == "__main__":
    main()
