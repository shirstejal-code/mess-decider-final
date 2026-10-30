"""Synthetic labelled text for the 'System 1' decision layer.

Two small datasets, written from templates (so every label is exact):
  data/notices.csv   hostel notices      -> is_holiday / is_exam / is_event
  data/feedback.csv  student complaints  -> category / urgency / safety_risk

Many templates are HARD on purpose: negations ("NOT a holiday", "postponed",
"cancelled") and look-alike words ("holiday homework"). Some templates are
reserved for the TEST set, so test sentences use phrasings the classifier never
saw in training. This text is simulated: replace it with real notices and
complaints (same columns) before trusting any number.
"""
import os
import random
import pandas as pd

WHEN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "12 Nov", "3 Dec", "18 Jan", "7 Feb", "21 Mar", "9 Apr"]
FESTIVAL = ["Diwali", "Pongal", "Christmas", "Eid", "Republic Day", "Holi", "Onam"]
SUBJECT = ["Maths", "Physics", "DBMS", "Operating Systems", "Machine Learning", "Networks"]
FEST = ["Vibrance", "Riviera", "TechnoFest", "Aarohan", "Gravitas"]
DISH = ["sambar rice", "parotta", "chicken gravy", "veg biryani", "idli", "chapati",
        "curd rice", "egg curry", "noodles", "dal"]


def _fill(template, rng):
    return template.format(when=rng.choice(WHEN), festival=rng.choice(FESTIVAL),
                           subject=rng.choice(SUBJECT), fest=rng.choice(FEST), dish=rng.choice(DISH))


# ------------------------------------------------------------------ notices
def _n(text, kind, topic=None, test=False):
    """kind: pos (flag on) / neg (look-alike, flag off) / neu (unrelated)."""
    flags = {"holiday": (1, 0, 0), "exam": (0, 1, 0), "event": (0, 0, 1)}[topic] if kind == "pos" else (0, 0, 0)
    return {"t": text, "kind": kind, "topic": topic, "flags": flags, "test": test}


NOTICES = [
    # holiday ---------------------------------------------------------------
    _n("{when} is a holiday on account of {festival}. Classes will not be held.", "pos", "holiday"),
    _n("Classes are suspended on {when}; hostel students may go home for the long weekend.", "pos", "holiday"),
    _n("The {festival} vacation begins from {when}. The hostel stays open for students who remain.", "pos", "holiday"),
    _n("No classes this {when} as the institute observes {festival}.", "pos", "holiday"),
    _n("Official holiday declared for {when}. Mess will serve brunch and dinner only.", "pos", "holiday", True),
    _n("Due to the {festival} break, most students will leave campus after {when}.", "pos", "holiday", True),
    _n("Mid-term break from {when}: academic block closed, hostels open.", "pos", "holiday", True),
    _n("Reminder: {when} is NOT a holiday. All classes will run as per timetable.", "neg", "holiday"),
    _n("The proposed holiday on {when} has been cancelled and regular classes will be held.", "neg", "holiday"),
    _n("Holiday homework for {subject} is due on {when}.", "neg", "holiday"),
    _n("Please fill the holiday list feedback form by {when}.", "neg", "holiday"),
    _n("There is no holiday on {when}; attendance is compulsory.", "neg", "holiday", True),
    _n("Vacation travel concession forms must be collected from the office before {when}.", "neg", "holiday", True),
    # exam ------------------------------------------------------------------
    _n("Mid-semester examinations begin on {when}. Hostel students must stay on campus.", "pos", "exam"),
    _n("CAT-1 for all second-year students will be held from {when}.", "pos", "exam"),
    _n("End semester exams start {when}; library and mess timings are extended.", "pos", "exam"),
    _n("Lab internal assessment for {subject} on {when}.", "pos", "exam"),
    _n("The {subject} test is scheduled for {when}, so expect late-night study groups.", "pos", "exam", True),
    _n("FAT for third-year B.Tech will commence on {when}.", "pos", "exam", True),
    _n("The {subject} exam on {when} has been postponed until further notice.", "neg", "exam"),
    _n("Results of the mid-semester examination are now available on the portal.", "neg", "exam"),
    _n("Exam fee payment last date is {when}.", "neg", "exam"),
    _n("Re-exam applications for {subject} are closed.", "neg", "exam"),
    _n("The {subject} test scheduled for {when} stands cancelled.", "neg", "exam", True),
    _n("Queries about the previous semester examinations should go to the controller's office.", "neg", "exam", True),
    # event -------------------------------------------------------------------
    _n("Annual tech fest {fest} will be held on {when} in the main auditorium.", "pos", "event"),
    _n("Cultural night on {when}; participants from other colleges will dine in the mess.", "pos", "event"),
    _n("Inter-college sports meet from {when}. Visiting teams will use the hostel mess.", "pos", "event"),
    _n("Alumni meet on {when} with a special dinner arranged for guests.", "pos", "event"),
    _n("{fest} begins {when}: expect a heavy crowd on campus and late-night food stalls.", "pos", "event", True),
    _n("The college hosts a hackathon on {when}; over 200 outside participants are expected.", "pos", "event", True),
    _n("The {fest} fest has been cancelled due to heavy rain.", "neg", "event"),
    _n("Feedback form for last month's cultural night is open until {when}.", "neg", "event"),
    _n("Club recruitment event postponed to next semester.", "neg", "event"),
    _n("Photos from the {fest} fest are now uploaded to the gallery.", "neg", "event", True),
    _n("The alumni meet scheduled for {when} is called off.", "neg", "event", True),
    # unrelated -----------------------------------------------------------------
    _n("Water supply will be interrupted on {when} between 10 am and 12 noon.", "neu"),
    _n("Wi-Fi maintenance in the hostel blocks tonight.", "neu"),
    _n("Laundry pickup has been moved to {when}.", "neu"),
    _n("Hostel fee is due on {when}; late fee applies after that.", "neu"),
    _n("Lost: a black water bottle near the library, please contact the warden.", "neu", None, True),
    _n("The gym will remain closed for renovation till {when}.", "neu", None, True),
]


def make_notices(n=1600, seed=7):
    rng = random.Random(seed)
    pos = [t for t in NOTICES if t["kind"] == "pos"]
    neg = [t for t in NOTICES if t["kind"] == "neg"]
    neu = [t for t in NOTICES if t["kind"] == "neu"]
    rows = []
    for _ in range(n):
        parts = []
        for _ in range(1 if rng.random() < 0.6 else 2):
            r = rng.random()
            parts.append(rng.choice(pos if r < 0.4 else neg if r < 0.7 else neu))
        text = " ".join(_fill(p["t"], rng) for p in parts)
        flags = [max(p["flags"][i] for p in parts) for i in range(3)]
        rows.append(dict(text=text, is_holiday=flags[0], is_exam=flags[1], is_event=flags[2],
                         split="test" if any(p["test"] for p in parts) else "train"))
    return _hold_out_seen(pd.DataFrame(rows), rng)


def _hold_out_seen(df, rng):
    """split = train (learn from) | seen_test (same templates, fresh fillers) | test (NEW templates)."""
    idx = list(df.index[df["split"] == "train"])
    df.loc[rng.sample(idx, int(0.15 * len(idx))), "split"] = "seen_test"
    return df


def notice_for_day(rng, holiday, exam, event, unseen=True, distractor_rate=0.25):
    """One synthetic notice for a calendar day.

    unseen=True  -> phrasings reserved for the test set (the classifier never saw this wording)
    unseen=False -> phrasings the classifier trained on
    distractor_rate: chance that a topic which is NOT happening still gets a look-alike notice
    (e.g. 'holiday homework due', 'exam postponed'). This number is a choice of the simulation."""
    pieces = []
    for flag, topic in ((holiday, "holiday"), (exam, "exam"), (event, "event")):
        if flag:
            pieces.append(rng.choice([t for t in NOTICES if t["kind"] == "pos" and t["topic"] == topic
                                      and t["test"] == unseen]))
        elif rng.random() < distractor_rate:
            pieces.append(rng.choice([t for t in NOTICES if t["kind"] == "neg" and t["topic"] == topic
                                      and t["test"] == unseen]))
    if not pieces or rng.random() < 0.3:
        pieces.append(rng.choice([t for t in NOTICES if t["kind"] == "neu" and t["test"] == unseen]))
    rng.shuffle(pieces)
    return " ".join(_fill(p["t"], rng) for p in pieces)


# ------------------------------------------------------------------ feedback
def _f(text, cat, urg, safe, test=False):
    return {"t": text, "cat": cat, "urg": urg, "safe": safe, "test": test}


FEEDBACK = [
    _f("The {dish} today was too salty and tasteless.", "taste", "low", 0),
    _f("{dish} was very oily and the taste was bad.", "taste", "low", 0),
    _f("The {dish} was bland and cold when it was served.", "taste", "low", 0),
    _f("Please improve the taste of {dish}; it is dull every time.", "taste", "low", 0, True),
    _f("Too much chilli in the {dish} this evening.", "taste", "low", 0, True),
    _f("The plates were not washed properly and still had oil on them.", "hygiene", "medium", 0),
    _f("The dining hall floor and tables are dirty.", "hygiene", "medium", 0),
    _f("The drinking water dispenser in the mess has not been cleaned for weeks.", "hygiene", "medium", 0),
    _f("Flies all over the serving counter at lunch.", "hygiene", "medium", 0, True),
    _f("Spoons and glasses smell bad even after washing.", "hygiene", "medium", 0, True),
    _f("Found a cockroach in my {dish} tonight.", "hygiene", "high", 1),
    _f("The {dish} smelled spoiled and two of my friends felt sick after eating.", "hygiene", "high", 1),
    _f("There was a hair and a small insect in the {dish}.", "hygiene", "high", 1),
    _f("The chicken looked undercooked and half our table had stomach pain later.", "hygiene", "high", 1, True),
    _f("Vomiting after dinner yesterday; I think the {dish} was stale.", "hygiene", "high", 1, True),
    _f("Food ran out by 8:15 pm and many of us were left hungry.", "quantity", "medium", 0),
    _f("The portion of {dish} is far too small for the price.", "quantity", "medium", 0),
    _f("They stopped serving {dish} before half the students had eaten.", "quantity", "medium", 0),
    _f("Only two chapatis per person is not enough.", "quantity", "medium", 0, True),
    _f("The counter was empty at 8 pm, nothing left to eat.", "quantity", "medium", 0, True),
    _f("The mess staff were rude when I asked for more {dish}.", "service", "medium", 0),
    _f("The queue at the counter took 30 minutes today.", "service", "low", 0),
    _f("Billing at the mess counter is confusing and slow.", "service", "low", 0),
    _f("Staff refused to serve me after 9 pm even though timing runs till 9:30.", "service", "medium", 0, True),
    _f("The token system keeps breaking and we wait forever.", "service", "low", 0, True),
    _f("We got {dish} four times this week, please change the menu.", "variety", "low", 0),
    _f("The menu has no variety at all, the same thing every day.", "variety", "low", 0),
    _f("Can we have {dish} on weekends instead of the usual?", "variety", "low", 0),
    _f("Please add more vegetarian options to the dinner menu.", "variety", "low", 0, True),
    _f("Same breakfast again and again, this is boring.", "variety", "low", 0, True),
]


def make_feedback(n=1200, seed=9):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        t = rng.choice(FEEDBACK)
        rows.append(dict(text=_fill(t["t"], rng), category=t["cat"], urgency=t["urg"],
                         safety_risk=t["safe"], split="test" if t["test"] else "train"))
    return _hold_out_seen(pd.DataFrame(rows), rng)


def main():
    os.makedirs("data", exist_ok=True)
    nt, fb = make_notices(), make_feedback()
    nt.to_csv("data/notices.csv", index=False)
    fb.to_csv("data/feedback.csv", index=False)
    print("Saved data/notices.csv (%d rows) and data/feedback.csv (%d rows); splits: %s | %s"
          % (len(nt), len(fb), nt.split.value_counts().to_dict(), fb.split.value_counts().to_dict()))


if __name__ == "__main__":
    main()
