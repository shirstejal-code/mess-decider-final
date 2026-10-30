"""Streamlit dashboard for the Hostel Mess Food-Waste Decider."""
import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from common import MEALS, LAG_FEATURES, COST_WASTE, COST_SHORT, DATA_PATH
from decision import quantity_from_forecaster, expected_cost_curve_q, demand_samples, expected_cost_of
from forecast import load_forecaster
import menu_nlp
import planner
import system1

st.set_page_config(page_title="Mess Food-Waste Decider", page_icon="🍛", layout="wide")


@st.cache_resource(show_spinner=False)
def bootstrap():
    """Build whatever is missing (fresh cloud deploy) and load the model."""
    import generate_data
    import train
    if not os.path.exists(DATA_PATH):
        generate_data.main()
    if not os.path.exists(menu_nlp.MENU_PATH):
        menu_nlp.main()
    try:
        fc = load_forecaster()
    except Exception:                      # missing or saved with a different library version
        train.main()
        fc = load_forecaster()
    return fc, menu_nlp.load(), planner.history_from_data(), system1.load_or_train()


with st.spinner("First launch: preparing data and training the model (about a minute)..."):
    fc, menu_model, hist, text_reader = bootstrap()

@st.cache_resource(show_spinner=False)
def get_readers(_trained):
    """Build the text readers once per server process, so a local Laya model is not reloaded on every click."""
    return system1.available_readers(_trained)


uses_lags = any(c in fc.features for c in LAG_FEATURES)
DEFAULT_MENU = {"breakfast": "idli, medu vada", "lunch": "veg biryani, curd rice, cabbage poriyal",
                "dinner": "chicken gravy, parotta, veg pulao"}


def show_csv(path, title, note=None):
    if os.path.exists(path):
        st.subheader(title)
        st.dataframe(pd.read_csv(path), hide_index=True)
        if note:
            st.caption(note)


def show_img(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption)


st.title("🍛 Hostel Mess Food-Waste Decider")
st.caption("Forecast attendance → calibrate the uncertainty → decide how many plates to cook.")

with st.sidebar:
    st.header("Costs (Rs)")
    co = st.number_input("Cost of a wasted plate", 1, 500, COST_WASTE)
    cu = st.number_input("Penalty per student left hungry", 1, 500, COST_SHORT)
    level = st.slider("Confidence interval", 0.5, 0.9, 0.8, 0.05)
    st.caption("Uncertainty method (auto-selected on validation data): **%s**" % fc.method)
    st.caption("Model inputs: " + ("calendar, weather, menu + recent attendance"
                                   if uses_lags else "calendar, weather, menu"))

def _apply_flags(flags):
    """Button callback: copy a notice reader's answers into the 'Today's decision' checkboxes."""
    st.session_state["hol1"] = bool(flags["holiday"])
    st.session_state["exam1"] = bool(flags["exam"])
    st.session_state["event1"] = bool(flags["event"])
    st.toast("Applied to the 'Today's decision' tab")


t1, t2, t3, t4, t5, t6 = st.tabs(["🍽️ Today's decision", "📅 Weekly plan", "🔍 Why this number?",
                                  "📊 Model results", "🛰️ Monitoring & NLP", "📰 Notices & feedback"])

# ------------------------------------------------------------------ tab 1
with t1:
    c1, c2 = st.columns(2)
    date = c1.date_input("Date", dt.date.today() + dt.timedelta(days=1), key="date1")
    meal = c2.selectbox("Meal", MEALS, index=2, key="meal1")
    c3, c4 = st.columns(2)
    rain = c3.slider("Expected rain (mm)", 0.0, 80.0, 0.0, key="rain1")
    temp = c4.slider("Temperature (°C)", 18.0, 42.0, 30.0, key="temp1")

    menu_mode = st.radio("Menu popularity", ["Type the menu (ML estimates popularity)", "Set manually"],
                         horizontal=True, key="menumode") if menu_model is not None else "Set manually"
    if menu_mode.startswith("Type"):
        text = st.text_input("Menu items (comma-separated)", DEFAULT_MENU[meal], key="menutext_" + meal)
        score = menu_nlp.predict_score(menu_model, text)
        st.caption("ML-estimated menu popularity: **%.2f / 5**" % score)
    else:
        score = float(st.slider("Menu popularity (1 = disliked, 5 = favourite)", 1.0, 5.0, 3.0, 0.5, key="menuscore"))

    c5, c6, c7 = st.columns(3)
    holiday = c5.checkbox("Holiday / break", key="hol1")
    exam = c6.checkbox("Exam period", key="exam1")
    event = c7.checkbox("Campus event", key="event1")

    hist_use = {m: list(v) for m, v in hist.items()}
    if uses_lags:
        y_in = st.number_input("Yesterday's attendance for this meal", 0, 2000,
                               int(hist_use[meal][-1]), key="yday_" + meal)
        hist_use[meal][-1] = float(y_in)
    lags = planner.lag_table(hist_use)[meal]
    row = planner.make_row(date, meal, rain, temp, holiday, exam, event, score, lags)

    pred = float(fc.predict_mean(row)[0])
    qty = int(quantity_from_forecaster(fc, row, cu, co)[0])
    lo, hi = fc.interval(row, level)
    samples = demand_samples(fc, row)
    naive = int(round(lags["lag1"]))

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected students", f"{pred:.0f}")
    m2.metric(f"{int(level * 100)}% range", f"{float(lo[0]):.0f} – {float(hi[0]):.0f}")
    m3.metric("🍽️ Cook this many", qty, delta=f"{qty - round(pred):+d} buffer vs forecast")
    m4.metric("Expected cost", f"Rs {expected_cost_of(samples, qty, cu, co):,.0f}",
              delta=f"{expected_cost_of(samples, qty, cu, co) - expected_cost_of(samples, naive, cu, co):+,.0f} vs 'same as yesterday'",
              delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.subheader("Expected cost vs plates cooked")
        qs, costs = expected_cost_curve_q(fc, row, cu, co)
        st.line_chart(pd.DataFrame({"Expected cost (Rs)": costs}, index=qs))
        st.caption("The lowest point of the curve is the recommended quantity.")
    with right:
        st.subheader("What if students' hunger matters more?")
        pens = [10, 20, 30, 40, 60, 80, 120, 160, 240, 320]
        sens = pd.DataFrame({"Plates to cook": [int(quantity_from_forecaster(fc, row, p, co)[0]) for p in pens]},
                            index=pens)
        sens.index.name = "Penalty per hungry student (Rs)"
        st.line_chart(sens)
        st.caption("The higher the penalty for running out, the bigger the safety buffer.")

# ------------------------------------------------------------------ tab 2
with t2:
    st.subheader("7-day cooking plan")
    st.caption("Edit the forecast for each day (weather, holidays...). Each day's forecast feeds the next.")
    start = st.date_input("Plan starts on", dt.date.today() + dt.timedelta(days=1), key="wkstart")
    default = pd.DataFrame({"date": [start + dt.timedelta(days=i) for i in range(7)],
                            "rain_mm": 0.0, "temp_c": 30.0,
                            "is_holiday": False, "is_exam": False, "is_event": False})
    edited = st.data_editor(default, hide_index=True, key="wk_%s" % start)
    wk_menu = st.slider("Average menu popularity this week (1-5)", 1.0, 5.0, 3.3, 0.1, key="wkmenu")
    plan_df = planner.plan(fc, hist, edited, cu, co, menu_score=wk_menu, level=level)
    st.dataframe(plan_df, hide_index=True)
    a, b, c = st.columns(3)
    a.metric("Total plates to cook", f"{int(plan_df['Cook'].sum()):,}")
    ours, naive_cost = plan_df["Expected cost (Rs)"].sum(), plan_df["If same as yesterday (Rs)"].sum()
    b.metric("Expected cost (this plan)", f"Rs {ours:,.0f}")
    c.metric("Expected cost if 'same as yesterday'", f"Rs {naive_cost:,.0f}",
             delta=f"{ours - naive_cost:+,.0f}", delta_color="inverse")
    st.bar_chart(plan_df.pivot(index="Date", columns="Meal", values="Cook")[MEALS], sort=False)

# ------------------------------------------------------------------ tab 3
with t3:
    st.subheader("Why does the model say %.0f students?" % pred)
    p_, typical, eff = fc.explain(row)
    st.write("A *typical* %s would draw about **%.0f** students. Today's conditions move that to **%.0f**."
             % (meal, typical, p_))
    st.bar_chart(eff.set_index("factor")["effect"])
    st.caption("Each bar = how many students this factor adds (+) or removes (-) compared with a typical day "
               "(what-if: set that factor to its typical value and re-predict).")
    st.divider()
    st.subheader("What drives attendance overall?")
    show_img("results/feature_importance.png")
    show_csv("results/feature_importance.csv", "Permutation importance (test period)",
             "Higher = the model relies on that factor more.")

# ------------------------------------------------------------------ tab 4
with t4:
    show_csv("results/summary.csv", "Cooking policies compared on unseen test days",
             "Cost = Rs 40 x wasted plates + Rs 60 x students left hungry. The data is simulated.")
    show_img("results/cost_comparison.png")
    show_img("results/forecast_vs_actual.png")
    show_csv("results/ablation.csv", "Ablation: what actually helps?",
             "Honest comparison of feature sets and uncertainty methods on the same test days.")
    show_csv("results/feature_set_choice.csv", "Feature-set selection (time-series cross-validation)")
    show_csv("results/tuning.csv", "Hyper-parameter tuning (time-series cross-validation)")

# ------------------------------------------------------------------ tab 5
with t5:
    st.subheader("Drift monitoring (simulated stress test)")
    st.caption("Attendance drops 12% halfway through the test period. CUSUM detects it, the model retrains and "
               "an online bias correction keeps forecasts on target.")
    show_csv("results/drift_summary.csv", "Static vs monitored model")
    show_img("results/drift.png")
    st.subheader("Anomaly detection")
    show_csv("results/anomaly_summary.csv", "How well unusual days were caught",
             "Possible only because the simulator recorded which days it disturbed.")
    if os.path.exists("results/anomalies.csv"):
        st.dataframe(pd.read_csv("results/anomalies.csv").head(12), hide_index=True)
    st.subheader("Menu NLP")
    show_csv("results/menu_nlp.csv", "Menu popularity model (TF-IDF + Ridge)")
    if os.path.exists("results/dish_scores.csv"):
        ds = pd.read_csv("results/dish_scores.csv")
        st.dataframe(ds, hide_index=True)

# ------------------------------------------------------------------ tab 6
EXAMPLE_NOTICE = ("Mid-semester examinations begin on Monday. Hostel students must stay on campus. "
                  "Holiday homework for DBMS is due on Friday.")
EXAMPLE_COMPLAINT = "The parotta smelled spoiled and two of my friends felt sick after eating."
with t6:
    st.subheader("📰 Read a hostel notice")
    st.caption("A 'System 1' reader answers typed yes/no questions with a probability instead of writing text. "
               "Your forecaster still makes the numbers; the reader only fills in the checkboxes.")
    readers = get_readers(text_reader)
    rname = st.selectbox("Reader", list(readers), index=1, key="reader_sel")

    def read_text(text, task):
        cache = st.session_state.setdefault("_s1_cache", {})
        key = (rname, task, text)
        if key not in cache:
            cache[key] = readers[rname].predict(text, task)
        return cache[key]

    notice = st.text_area("Paste a notice", EXAMPLE_NOTICE, height=90, key="notice_txt")
    try:
        ans = read_text(notice, "notice")
    except Exception as e:
        ans = None
        st.error("This reader could not answer: %s" % e)
    if ans:
        labels = {"holiday": "Holiday / break", "exam": "Exam period", "event": "Campus event"}
        for col, k in zip(st.columns(3), labels):
            p = ans[k]
            col.metric(labels[k], "%s · %.0f%% sure" % ("Yes" if p >= 0.5 else "No", 100 * max(p, 1 - p)))
        st.button("➡️ Apply these to the 'Today's decision' tab", on_click=_apply_flags,
                  args=({k: ans[k] >= 0.5 for k in labels},), key="apply_btn")

    st.divider()
    st.subheader("🍽️ Triage a student complaint")
    complaint = st.text_area("Student feedback", EXAMPLE_COMPLAINT, height=80, key="fb_txt")
    try:
        fb = read_text(complaint, "feedback")
    except Exception as e:
        fb = None
        st.error("This reader could not answer: %s" % e)
    if fb:
        a, b, c = st.columns(3)
        a.metric("Category", "%s · %.0f%%" % (fb["category"][0], 100 * fb["category"][1]))
        b.metric("Urgency", "%s · %.0f%%" % (fb["urgency"][0], 100 * fb["urgency"][1]))
        c.metric("Food-safety risk", "%.0f%%" % (100 * fb["safety_risk"]))
        if fb["safety_risk"] >= 0.5 or fb["urgency"][0] == "high":
            st.error("🚨 Possible food-safety issue: tell the warden today.")

    st.divider()
    show_csv("results/text_eval.csv", "How well do the readers read? (simulated text)",
             "Seen wording = patterns the classifier trained on. Unseen wording = brand-new patterns. "
             "The keyword rules were written alongside the test text, so their scores are optimistic.")
    show_csv("results/text_end_to_end.csv", "Does reading notices help the plate decision?",
             "The reader's flags feed the forecaster. 25% of non-events get a look-alike notice "
             "(a simulation choice), so exact percentages depend on it.")
    show_img("results/text_end_to_end.png")
    with st.expander("How to plug in Laya or JEV"):
        st.markdown(
            "**Laya on your own machine** (needs Python 3.10+): `pip install laya`, then pick "
            "*Laya (local model)* above. The first call downloads about 1 GB of weights.\n\n"
            "**Laya or JEV over HTTP:** run `pip install \"laya[serve]\"` and `laya-serve`, or use TypeSafe's "
            "hosted JEV API. Then set `SYSTEMONE_URL` (and `SYSTEMONE_KEY`) as environment variables, or in "
            "`.streamlit/secrets.toml` (never commit the key). *Laya server / JEV (HTTP)* then appears above.\n\n"
            "Both speak the same `/v1/systemone` protocol, so one reader handles either.")
