"""Run the whole project in one go:   python run_all.py

Options:
  --skip-tune   skip the (slower) hyper-parameter search
  --keep-data   do not regenerate data/mess_data.csv (use your own data)
"""
import importlib
import sys
import time

STEPS = [
    ("Generate data", "generate_data"),
    ("Menu NLP model", "menu_nlp"),
    ("Tune with time-series cross-validation", "tune"),
    ("Train forecaster (mean + conformal quantiles)", "train"),
    ("Evaluate cooking policies", "evaluate"),
    ("Ablation study", "ablation"),
    ("Explainability", "explain"),
    ("Anomaly detection", "anomaly"),
    ("Drift monitoring stress test", "drift_demo"),
    ("Text decision layer: notice + complaint readers, end-to-end test", "text_eval"),
]

if __name__ == "__main__":
    skip = set()
    if "--skip-tune" in sys.argv:
        skip.add("tune")
    if "--keep-data" in sys.argv:
        skip.add("generate_data")
    for i, (title, mod) in enumerate(STEPS, 1):
        if mod in skip:
            continue
        print("\n=== [%d/%d] %s ===" % (i, len(STEPS), title))
        t = time.time()
        importlib.import_module(mod).main()
        print("(%.1fs)" % (time.time() - t))
    print("\nAll done. Now run:  streamlit run app.py")
