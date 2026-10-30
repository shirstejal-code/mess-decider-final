"""'System 1' readers: fast, typed decisions from a short piece of text.

A System 1 decision model does not write sentences. You give it a text ("state") and
typed questions; it answers every question in one quick pass with a probability:
    noul    -> probability that a yes/no statement is true
    choice  -> one label out of options you define (+ a confidence)

Readers (all expose  predict(text, task) -> {question_id: value}):
    KeywordReader      hand-written rules (naive baseline, no learning)
    TrainedReader      OUR model: TF-IDF + logistic regression, trained on data/notices.csv + feedback.csv
    LayaLocalReader    Convai's open-source Laya model, run on your own machine (optional, Python 3.10+)
    RemoteReader       any server that speaks the /v1/systemone protocol: a self-hosted `laya-serve`
                       or TypeSafe's hosted JEV API (optional, needs a URL and an API key)

value = float (noul probability)  or  (label, confidence)  for choice questions.
"""
import json
import os
import re
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TEXT_MODEL_PATH = "models/text_reader.joblib"

NOTICE_QUESTIONS = {
    "holiday": {"type": "noul", "instructions":
                "Does this notice say students have a holiday, break or no classes "
                "(not cancelled or denied, and not merely about holiday homework or forms)?"},
    "exam": {"type": "noul", "instructions":
             "Does this notice say an examination or test period is happening or starting "
             "(not postponed, cancelled, or only about results or fees)?"},
    "event": {"type": "noul", "instructions":
              "Does this notice announce a campus event, fest or gathering that will actually "
              "take place (not cancelled, called off or already over)?"},
}
FEEDBACK_QUESTIONS = {
    "category": {"type": "choice", "instructions": "What is the main complaint about?",
                 "criteria": {"taste": "flavour, salt, spice, oil, food served cold",
                              "hygiene": "dirty plates or tables, flies, insects, spoiled or unsafe food",
                              "quantity": "portions too small or food running out",
                              "service": "staff behaviour, queues, billing, tokens",
                              "variety": "repetitive menu, wants different or more dishes"}},
    "urgency": {"type": "choice", "instructions": "How urgently should the mess manager act?",
                "criteria": {"low": "minor annoyance, can wait for the next menu review",
                             "medium": "a real problem to fix this week",
                             "high": "someone may be sick or hurt, act today"}},
    "safety_risk": {"type": "noul", "instructions":
                    "Does this complaint describe a food-safety risk such as sickness, spoiled or "
                    "undercooked food, or insects in the food?"},
}
QUESTIONS = {"notice": NOTICE_QUESTIONS, "feedback": FEEDBACK_QUESTIONS}
FLAGS = ["holiday", "exam", "event"]


def parse_answers(answers, questions):
    """Turn a Laya / JEV style `answers` object into {qid: float | (label, confidence)}."""
    out = {}
    for qid, q in questions.items():
        a = answers.get(qid) if isinstance(answers, dict) else None
        if not isinstance(a, dict):
            raise RuntimeError("the answer for '%s' is missing from the response" % qid)
        if q["type"] == "noul":
            out[qid] = float(a["noul"])
        else:
            out[qid] = (str(a["choice"]), float(a.get("confidence", 0.0)))
    return out


# ------------------------------------------------------------------ 1. keyword rules
_NOTICE_RULES = {
    "holiday": re.compile(r"\b(holiday|vacation|break|no classes|suspended)\b", re.I),
    "exam": re.compile(r"\b(exams?|examinations?|cat-?\d|fat|tests?|assessments?|mid[- ]?(sem|term)\w*|end[- ]?sem\w*)\b", re.I),
    "event": re.compile(r"\b(fest|event|cultural|sports meet|alumni|hackathon|symposium|competition)\b", re.I),
}
_SAFETY = re.compile(r"\b(sick|vomit\w*|insects?|cockroach\w*|spoiled|stale|undercooked|worms?|stomach|poison\w*|hair)\b", re.I)
_CATEGORY_RULES = [
    ("hygiene", re.compile(r"\b(dirty|clean\w*|wash\w*|flies|fly|smell\w*|hygien\w*)\b", re.I)),
    ("quantity", re.compile(r"\b(ran out|portion|not enough|hungry|empty|stopped serving|only two|small)\b", re.I)),
    ("service", re.compile(r"\b(staff|queue|counter|billing|token|rude|wait\w*)\b", re.I)),
    ("variety", re.compile(r"\b(variety|same|menu|again and again|boring|options|instead)\b", re.I)),
]


class KeywordReader:
    name = "Keyword rules (naive baseline)"

    def predict(self, text, task):
        if task == "notice":
            return {k: 1.0 if rx.search(text) else 0.0 for k, rx in _NOTICE_RULES.items()}
        safety = bool(_SAFETY.search(text))
        cat, conf = "taste", 0.4
        if safety:
            cat, conf = "hygiene", 1.0
        else:
            for label, rx in _CATEGORY_RULES:
                if rx.search(text):
                    cat, conf = label, 1.0
                    break
        if safety:
            urgency = "high"
        elif cat in ("hygiene", "quantity") or re.search(r"\brude\b", text, re.I):
            urgency = "medium"
        else:
            urgency = "low"
        return {"category": (cat, conf), "urgency": (urgency, 1.0), "safety_risk": 1.0 if safety else 0.0}


# ------------------------------------------------------------------ 2. our trained model
def _pipe():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True),
                         LogisticRegression(C=20, max_iter=3000))


class TrainedReader:
    name = "Our trained classifier (TF-IDF + logistic regression)"

    def __init__(self):
        self.models = {}

    def fit(self, notices, feedback):
        for f in FLAGS:
            self.models["notice:" + f] = _pipe().fit(notices["text"], notices["is_" + f])
        for q, col in (("category", "category"), ("urgency", "urgency"), ("safety_risk", "safety_risk")):
            self.models["feedback:" + q] = _pipe().fit(feedback["text"], feedback[col])
        return self

    def predict(self, text, task):
        if task == "notice":
            return {f: float(self.models["notice:" + f].predict_proba([text])[0][1]) for f in FLAGS}
        out = {}
        for q in ("category", "urgency"):
            m = self.models["feedback:" + q]
            p = m.predict_proba([text])[0]
            i = int(np.argmax(p))
            out[q] = (str(m.classes_[i]), float(p[i]))
        out["safety_risk"] = float(self.models["feedback:safety_risk"].predict_proba([text])[0][1])
        return out


def train_and_save(notices=None, feedback=None, path=TEXT_MODEL_PATH):
    import text_data
    if notices is None or feedback is None:
        if not (os.path.exists("data/notices.csv") and os.path.exists("data/feedback.csv")):
            text_data.main()
        notices, feedback = pd.read_csv("data/notices.csv"), pd.read_csv("data/feedback.csv")
    reader = TrainedReader().fit(notices[notices["split"] == "train"], feedback[feedback["split"] == "train"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(reader, path)
    return reader


def load_or_train(path=TEXT_MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception:                       # missing, or saved with another library version
        return train_and_save(path=path)


# ------------------------------------------------------------------ 3. Laya, on your machine
class LayaLocalReader:
    """Convai's open-source Laya. Needs `pip install laya` (Python 3.10+, PyTorch) and downloads the
    weights from the Hugging Face hub on first use (around 1 GB). Call pattern follows Laya's README:
        Router().predict(state, questions)["answers"][qid]["noul" | "choice"]
    Written from the package documentation; not run end to end by the author of this project."""
    name = "Laya (local model)"

    def __init__(self, model=None):
        self.model = model or os.environ.get("LAYA_MODEL", "typed-decisions")
        self._router = None

    @staticmethod
    def available():
        try:
            import importlib.util
            return importlib.util.find_spec("laya") is not None
        except Exception:
            return False

    def predict(self, text, task):
        if self._router is None:
            from laya import Router
            self._router = Router()
        questions = QUESTIONS[task]
        res = self._router.predict(text, questions, model=self.model)
        return parse_answers(res["answers"], questions)


# ------------------------------------------------------------------ 4. any /v1/systemone server
class RemoteReader:
    """Talks to anything that implements POST /v1/systemone: a self-hosted `laya-serve`, or TypeSafe's
    hosted JEV API (Laya's docs say its wire protocol is the same; take the base URL and key from
    TypeSafe's documentation)."""
    name = "Laya server / JEV (HTTP)"

    def __init__(self, base_url, api_key=None, model=None, timeout=30):
        if not str(base_url).lower().startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    def predict(self, text, task):
        questions = QUESTIONS[task]
        body = {"state": {"body": text}, "questions": questions}
        if self.model:
            body["model"] = self.model
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base_url + "/v1/systemone", data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError("server error %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:200]))
        except Exception as e:
            raise RuntimeError("could not reach %s: %s" % (self.base_url, e))
        if not isinstance(parsed, dict):
            raise RuntimeError("unexpected response from server")
        return parse_answers(parsed.get("answers"), questions)


# ------------------------------------------------------------------ helpers
def get_setting(name):
    """Read a setting from an environment variable or (in Streamlit) from secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st
        v = st.secrets.get(name)
        return str(v) if v else None
    except Exception:
        return None


def available_readers(trained, laya_local=None):
    """Ordered {name: reader} of everything usable right now."""
    readers = {KeywordReader.name: KeywordReader(), trained.name: trained}
    if laya_local is None:
        laya_local = LayaLocalReader.available()
    if laya_local:
        readers[LayaLocalReader.name] = LayaLocalReader()
    url = get_setting("SYSTEMONE_URL")
    if url:
        try:
            readers[RemoteReader.name] = RemoteReader(url, get_setting("SYSTEMONE_KEY"), get_setting("SYSTEMONE_MODEL"))
        except ValueError:
            pass
    return readers
