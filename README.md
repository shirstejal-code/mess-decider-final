# MessDecider Final: Hostel Mess Food-Waste Decider (v3)

Forecast attendance -> calibrate the uncertainty -> decide how many plates to cook.
New in v3: a **System 1 decision layer** that reads hostel notices and student complaints
(inspired by decision models such as Convai's Laya and TypeSafe's JEV).

## Run (Mac, VS Code terminal, inside the mess_planner folder)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python run_all.py            # everything: data, models, results (about 1-2 minutes)
    streamlit run app.py

Options for run_all.py:  --skip-tune (faster)   --keep-data (use your own data/mess_data.csv)

## What is inside
| File | What it does |
|---|---|
| generate_data.py | simulated mess data (weather, holidays, exams, menus, hidden "mood", rare shock days) |
| common.py | features, attendance-lag features, time-based split |
| menu_nlp.py | TF-IDF + Ridge predicts menu popularity from dish names |
| tune.py | hyper-parameter search with time-series cross-validation |
| forecast.py | gradient boosting forecaster + calibrated uncertainty + what-if explanations |
| decision.py | newsvendor decision: cook the quantile Cu/(Cu+Co) of demand |
| train.py | picks feature set + uncertainty method on VALIDATION data, trains, saves |
| evaluate.py / ablation.py | cooking-policy simulation and an honest A/B of every ingredient |
| explain.py | permutation importance (what drives attendance) |
| anomaly.py | IsolationForest finds unusual days |
| drift_demo.py | CUSUM drift detection + retraining stress test |
| planner.py | 7-day recursive planner |
| **text_data.py** | simulated notices + complaints with exact labels (hard negations and look-alikes) |
| **system1.py** | the readers: keyword rules, our trained classifier, optional Laya (local) and Laya/JEV (HTTP) |
| **text_eval.py** | benchmark of the readers + END-TO-END test (reader -> forecaster -> plates) |
| app.py | dashboard (6 tabs) |

## The System 1 layer in one paragraph
A System 1 decision model does not write text. You give it a text and typed questions
(`noul` = yes/no probability, `choice` = pick one label). It answers all questions in one quick pass.
Here it reads a notice ("Is this a holiday? An exam period? An event?") and fills the forecaster's
checkboxes, and it triages complaints (category, urgency, food-safety risk). The forecaster still makes the numbers.

## Using Laya or JEV (optional, not needed to run the project)
* **Laya on your machine:** needs Python 3.10+, PyTorch and a ~1 GB download:
  `pip install -r requirements-laya.txt` then `RUN_LAYA=1 python text_eval.py`.
  The code follows Laya's documented `Router().predict(text, questions)`, but was not run end to end by the author.
* **Laya server or JEV over HTTP:** both use `POST /v1/systemone`. Set `SYSTEMONE_URL` and `SYSTEMONE_KEY`
  (environment variables, or `.streamlit/secrets.toml`; never commit a key). The HTTP client was tested only
  against a mock server that follows the documented protocol. For JEV, take the URL and key from TypeSafe's docs.

## Honest limits
* All data is simulated. Notices and complaints are generated from templates I wrote.
* The keyword rules were written by the same person who wrote the templates, so their complaint scores are optimistic.
* Our trained classifier is near-perfect on wording it has seen but fails on new wording, and it can be
  sensitive to sentence order. That gap is the reason pretrained decision models exist; Laya/JEV were NOT benchmarked here.
* The end-to-end cost percentages depend on simulation choices (e.g. 25% of non-events get a look-alike notice).

## Real data
Mess data columns: date, meal, rain_mm, temp_c, is_holiday, is_exam, is_event, menu_text (optional), menu_score, attendance
(one row per meal per day, no missing days, ideally 90+ days). Save as data/mess_data.csv, then `python run_all.py --keep-data`.
For text: replace data/notices.csv (text, is_holiday, is_exam, is_event, split) and data/feedback.csv
(text, category, urgency, safety_risk, split); split values: train / seen_test / test.

## Deploy (Streamlit Community Cloud)
Push the folder to GitHub (models/ is git-ignored and rebuilt on first launch), then
share.streamlit.io -> Create app -> repo, branch main, main file app.py.
The Laya parts are optional and are not installed on the cloud; the app works without them.
