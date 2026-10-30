"""Generate a realistic synthetic mess-attendance dataset (v2).

What is simulated (so you can describe it honestly in your report):
  * weekday / weekend / holiday / exam / event / rain / temperature effects
  * menu text (real-sounding dishes) -> hidden dish popularity -> menu_score
  * a persistent unobserved 'campus mood' shock (AR(1)), so yesterday matters
  * a few rare unlogged shock days (outage, food scare...) saved to
    data/anomaly_labels.csv ONLY to evaluate the anomaly detector

Replace data/mess_data.csv with your own logs (same columns) to use real data:
date, meal, rain_mm, temp_c, is_holiday, is_exam, is_event, menu_text, menu_score, attendance
(menu_text is optional; menu_score is a 1-5 rating).
"""
import os
import numpy as np
import pandas as pd
from common import CAPACITY, MEALS

DISHES = {
    "breakfast": {"idli": 4.0, "dosa": 4.6, "pongal": 3.2, "poori masala": 4.2, "upma": 1.8,
                  "medu vada": 3.8, "bread omelette": 4.4, "chapati kurma": 3.0,
                  "kesari": 3.9, "sevai": 2.4},
    "lunch": {"sambar rice": 3.0, "curd rice": 3.4, "veg biryani": 4.0, "chicken biryani": 4.9,
              "rajma rice": 3.6, "lemon rice": 2.8, "paneer butter masala": 4.5, "fish fry": 4.3,
              "egg fried rice": 3.9, "bisibelebath": 2.6, "mixed veg curry": 2.0,
              "cabbage poriyal": 1.6},
    "dinner": {"parotta": 4.7, "chapati dal": 3.0, "veg fried rice": 3.8, "noodles": 4.1,
               "egg curry": 4.2, "dal tadka": 3.1, "veg pulao": 3.3, "chicken gravy": 4.8,
               "ragi dosa": 2.5, "pasta": 3.7, "bottle gourd curry": 1.5, "kurma": 3.5},
}
N_DISHES = {"breakfast": 2, "lunch": 3, "dinner": 3}


def main(seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-07-01", "2026-09-30")
    base = {"breakfast": 0.62, "lunch": 0.70, "dinner": 0.86}
    anomaly_idx = set(rng.choice(np.arange(30, len(dates)), size=12, replace=False).tolist())
    shock = 0.0
    rows, labels = [], []
    for i, d in enumerate(dates):
        m, dow = d.month, d.dayofweek
        temp = 28 + 4 * np.sin((m - 3) / 12 * 2 * np.pi) + rng.normal(0, 1.5)
        p_rain = 0.45 if m in (6, 7, 8, 9, 10, 11) else 0.08
        rain = rng.exponential(12) if rng.random() < p_rain else 0.0
        in_break = ((m == 12 and d.day >= 20) or (m == 1 and d.day <= 8)
                    or (m == 5 and d.day >= 10) or (m == 6 and d.day <= 20))
        holiday = int(in_break or rng.random() < 0.03)
        exam = int((m == 10 and d.day <= 14) or (m == 3 and d.day <= 14)
                   or (m == 12 and 1 <= d.day <= 14))
        event = int(rng.random() < 0.04)
        shock = 0.6 * shock + rng.normal(0, 0.03)          # persistent hidden mood
        is_anom = i in anomaly_idx
        if is_anom:
            labels.append(d.strftime("%Y-%m-%d"))
        anom_factor = rng.uniform(0.65, 0.80) if is_anom else 1.0
        for meal in MEALS:
            names = [str(x) for x in rng.choice(list(DISHES[meal]), size=N_DISHES[meal], replace=False)]
            pop = float(np.mean([DISHES[meal][n] for n in names]))
            menu = int(np.clip(round(pop + rng.normal(0, 0.35)), 1, 5))
            f = base[meal]
            if dow >= 5:
                f += {"breakfast": -0.15, "lunch": 0.05, "dinner": -0.06}[meal]
            if holiday:
                f -= 0.35
            if exam:
                f += {"breakfast": -0.08, "lunch": 0.0, "dinner": 0.05}[meal]
            if event and meal != "breakfast":
                f += 0.08
            f += 0.0006 * min(rain, 60)
            f += 0.04 * (menu - 3)
            f += -0.004 * max(temp - 32, 0)
            f += shock
            f *= anom_factor
            f *= rng.normal(1.0, 0.04)
            att = int(np.clip(round(f * CAPACITY), 0, CAPACITY))
            rows.append(dict(date=d.strftime("%Y-%m-%d"), meal=meal, rain_mm=round(rain, 1),
                             temp_c=round(temp, 1), is_holiday=holiday, is_exam=exam,
                             is_event=event, menu_text=", ".join(names), menu_score=menu,
                             attendance=att))
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(rows).to_csv("data/mess_data.csv", index=False)
    pd.DataFrame({"date": labels}).to_csv("data/anomaly_labels.csv", index=False)
    print("Saved data/mess_data.csv with %d rows (+ %d hidden shock days in anomaly_labels.csv)"
          % (len(rows), len(labels)))


if __name__ == "__main__":
    main()
