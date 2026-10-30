"""Compare the text readers on FAMILIAR and on BRAND-NEW wording, then test them END TO END.

1. Reader benchmark: how well does each reader read notices and complaints?
     "Seen wording"   = same sentence patterns as training (fresh dates / dishes)
     "Unseen wording" = sentence patterns the trained classifier never saw (stress test)
2. End to end: a reader's flags (holiday / exam / event) are fed into the attendance forecaster.
   If the reader misreads a notice, the plate decision gets worse, so we measure real cost.

Honest limits: all text is simulated from templates. The keyword rules were written by the same
person who wrote the templates, so their scores are optimistic. Replace the text with real notices
and complaints before trusting any number.

Optional readers are included when available:
   RUN_LAYA=1            also run the local Laya model (needs `pip install laya`, Python 3.10+)
   SYSTEMONE_URL=...     also run a Laya server or JEV over HTTP (+ SYSTEMONE_KEY)
"""
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import system1
import text_data
from common import COST_SHORT, COST_WASTE, load_data, split_by_date
from decision import plate_cost, quantity_from_forecaster
from forecast import load_forecaster

SAMPLE = int(os.environ.get("TEXT_EVAL_SAMPLE", "0") or 0)   # e.g. 60 = quick run with 60 texts per test set
FLAGS = system1.FLAGS
COLS = ["is_holiday", "is_exam", "is_event"]
SETS = [("Seen wording", "seen_test"), ("Unseen wording", "test")]


def _ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        hi = edges[i + 1] if i < bins - 1 else 1.0001
        m = (p >= edges[i]) & (p < hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def evaluate_reader(name, reader, notices, feedback, test_set):
    t0, calls = time.time(), 0
    P = []
    for t in notices["text"]:
        P.append(reader.predict(t, "notice"))
        calls += 1
    P = pd.DataFrame(P)[FLAGS].values
    Y = notices[COLS].values
    pb = (P >= 0.5).astype(int)
    F = [reader.predict(t, "feedback") for t in feedback["text"]]
    calls += len(F)
    safety_p = np.array([f["safety_risk"] for f in F])
    return {"Reader": name, "Test set": test_set,
            "Notice flags correct (%)": round(float((pb == Y).mean() * 100), 1),
            "Notice fully correct (%)": round(float((pb == Y).all(axis=1).mean() * 100), 1),
            "Complaint category correct (%)": round(float(np.mean([f["category"][0] == c for f, c in zip(F, feedback["category"])]) * 100), 1),
            "Urgency correct (%)": round(float(np.mean([f["urgency"][0] == u for f, u in zip(F, feedback["urgency"])]) * 100), 1),
            "Safety-risk F1 (%)": round(float(f1_score(feedback["safety_risk"], (safety_p >= 0.5).astype(int), zero_division=0) * 100), 1),
            "Calibration error (lower is better)": round(_ece(np.concatenate([P.ravel(), safety_p]),
                                                             np.concatenate([Y.ravel(), feedback["safety_risk"].values])), 3),
            "ms per call": round((time.time() - t0) / calls * 1000, 2)}


def majority_row(train_f, notices, feedback, test_set):
    cat, urg = train_f["category"].mode()[0], train_f["urgency"].mode()[0]
    Y = notices[COLS].values
    return {"Reader": "Always guess the most common answer", "Test set": test_set,
            "Notice flags correct (%)": round(float((Y == 0).mean() * 100), 1),
            "Notice fully correct (%)": round(float((Y == 0).all(axis=1).mean() * 100), 1),
            "Complaint category correct (%)": round(float((feedback["category"] == cat).mean() * 100), 1),
            "Urgency correct (%)": round(float((feedback["urgency"] == urg).mean() * 100), 1),
            "Safety-risk F1 (%)": 0.0, "Calibration error (lower is better)": None, "ms per call": 0.0}


def end_to_end(readers):
    fc = load_forecaster()
    cu, co = COST_SHORT, COST_WASTE
    _, _, test = split_by_date(load_data())
    days = test.groupby("date")[COLS].max().reset_index()
    if SAMPLE:                                            # quick run: fewer days (every k-th day)
        days = days.iloc[::max(1, len(days) // SAMPLE)].reset_index(drop=True)
        test = test[test["date"].isin(days["date"])]
    actual = test["attendance"].values
    truth = days.set_index("date")[COLS]

    def cost_with(flags_by_date):
        t2 = test.copy()
        for c in COLS:
            t2[c] = t2["date"].map(flags_by_date[c]).astype(int)
        q = quantity_from_forecaster(fc, t2, cu, co)
        cost, waste, short = plate_cost(q, actual, cu, co)
        return int(waste.sum()), int(short.sum()), int(cost.sum())

    def row(scenario, phrasing, ok, flags):
        w, s, c = cost_with(flags)
        return {"Scenario": scenario, "Wording": phrasing, "Days read correctly (%)": round(ok, 1),
                "Wasted plates": w, "Shortage (students)": s, "Total cost (Rs)": c}

    rows = [row("Perfect flags (upper bound)", "-", 100.0, truth)]
    for label, unseen in (("seen", False), ("unseen", True)):
        rng = random.Random(11)
        texts = [text_data.notice_for_day(rng, bool(r.is_holiday), bool(r.is_exam), bool(r.is_event), unseen=unseen)
                 for r in days.itertuples()]
        for name, reader in readers.items():
            pred = pd.DataFrame([reader.predict(t, "notice") for t in texts], index=days["date"])
            flags = pd.DataFrame({"is_holiday": (pred["holiday"] >= 0.5).astype(int),
                                  "is_exam": (pred["exam"] >= 0.5).astype(int),
                                  "is_event": (pred["event"] >= 0.5).astype(int)})
            ok = float((flags.values == truth.values).all(axis=1).mean() * 100)
            rows.append(row("Flags read by: " + name, label, ok, flags))
    zero = pd.DataFrame(0, index=days["date"], columns=COLS)
    rows.append(row("Ignore all notices (no flags)", "-", float((truth.values == 0).all(axis=1).mean() * 100), zero))
    out = pd.DataFrame(rows)
    best = out.loc[0, "Total cost (Rs)"]
    out["Extra cost vs perfect flags (%)"] = ((out["Total cost (Rs)"] - best) / best * 100).round(1)
    return out


def main():
    if not (os.path.exists("data/notices.csv") and os.path.exists("data/feedback.csv")):
        text_data.main()
    notices, feedback = pd.read_csv("data/notices.csv"), pd.read_csv("data/feedback.csv")
    if "seen_test" not in set(notices["split"]):          # data from an older version
        text_data.main()
        notices, feedback = pd.read_csv("data/notices.csv"), pd.read_csv("data/feedback.csv")
    trained = system1.train_and_save(notices, feedback)
    readers = system1.available_readers(trained, laya_local=os.environ.get("RUN_LAYA") == "1")
    if system1.LayaLocalReader.name not in readers:
        print("(Laya local model skipped: set RUN_LAYA=1 after `pip install laya` on Python 3.10+)")
    if system1.RemoteReader.name not in readers:
        print("(HTTP reader skipped: set SYSTEMONE_URL and SYSTEMONE_KEY to include it)")

    train_f = feedback[feedback.split == "train"]
    rows = []
    for label, split in SETS:
        n, f = notices[notices.split == split], feedback[feedback.split == split]
        if SAMPLE:
            n, f = n.sample(min(SAMPLE, len(n)), random_state=0), f.sample(min(SAMPLE, len(f)), random_state=0)
        rows.append(majority_row(train_f, n, f, label))
        for name in list(readers):
            try:
                rows.append(evaluate_reader(name, readers[name], n, f, label))
            except Exception as e:
                print("Reader '%s' failed and was skipped: %s" % (name, e))
                readers.pop(name, None)
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/text_eval.csv", index=False)
    pd.set_option("display.width", 260)
    print("\n-- Reading notices and complaints --")
    print(res.to_string(index=False))

    e2e = end_to_end(readers)
    e2e.to_csv("results/text_end_to_end.csv", index=False)
    print("\n-- End to end: notice reader -> forecaster -> plates (test period) --")
    print(e2e.to_string(index=False))

    lab = [("%s (%s wording)" % (s.replace("Flags read by: ", "").split(" (")[0], w)) if w in ("seen", "unseen") else s
           for s, w in zip(e2e["Scenario"], e2e["Wording"])]
    colors = ["#2e7d32" if "Perfect" in s else "#c62828" if "Ignore" in s else "#6a9fd8" if "unseen" not in l else "#f0a030"
              for s, l in zip(e2e["Scenario"], lab)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(lab[::-1], e2e["Total cost (Rs)"].values[::-1], color=colors[::-1])
    ax.set_xlabel("Total cost on test days (Rs), lower is better")
    ax.set_title("Does reading hostel notices help the plate decision?")
    plt.tight_layout()
    plt.savefig("results/text_end_to_end.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
